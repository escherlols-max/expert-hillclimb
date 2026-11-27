# 09. Sidecar experts

A sidecar is a new expert that rides along with an existing one. The original (parent) expert stays exactly as it is.
On every token that the router sends to the parent, the sidecar also runs, and its output is added to the MoE output:

    moe_out += w_parent * tanh(a) * f_sc(x)

- `w_parent` is the parent's final gate weight for that token. For GLM-5.x it already includes the top-8 normalisation and
  the routed scaling factor 2.5, so it is the same `w` that the shard stores.
- `f_sc` is a full expert of the shard's family (`glm5`: silu SwiGLU; `inkling2`: the softcapped form). Its inner width
  may differ from the parent's.
- `a` is one scalar. `tanh(a)` sets how strongly the sidecar speaks. With `a = 0` the model is exactly the stock model.

The router does not change, so a sidecar never moves tokens between experts. That makes it safe to change the sidecar's
weights a lot: there is no weight-delta limit, only a limit on how much the branch adds to the output.

## Score

The first-order loss change of a sidecar is `dL = sum_t w_t tanh(a) <g_t, f_sc(x_t)>`. The tools report:

    score        = -dL / sum_t w_t
    score_norm   = -dL / ( ||w g||_F * ||f0||_F )         # f0 = the parent's own output on the same rows
    branch_drift = ||tanh(a) f_sc||_F^2 / ||f0||_F^2       # must be <= 0.05 on the held-out rows

`score_norm` is the leaderboard number, the same scale-free form as for in-place edits (docs/03). Both splits are reported.
The held-out split decides.

    python tools/tune_sidecar.py  --shard S.npz --out sidecar.npz --steps 400 --lam 3
    python tools/score_sidecar.py --shard S.npz --overlay sidecar.npz

A sidecar overlay is an npz with `W_gate`, `W_up`, `W_down` (float32) and the scalar `a`.

## Ways to build one

- **Tuner:** `tune_sidecar.py` starts from a copy of the parent with a small `a` (`--a0`, default 0.05), or from any
  weights you pass with `--init`. It runs Adam on the train rows with the normalised linear term plus `lam * branch_drift`.
  It has no weight anchor.
- **Weight-space edits:** every `weight_space.py` command takes `--sidecar`. The edited expert is then written as a sidecar
  instead of an in-place change. Without `--a`, the tool picks the best strength in closed form on the train rows: the gain
  is linear in `tanh(a)` and the branch drift grows with `tanh(a)^2`, so the best `tanh(a)` sits at 90% of the drift cap,
  with the sign that makes the gain positive.
- **Your own fit:** any method is fine (ridge regression on `W_down`, a smaller inner width, distillation from another model)
  as long as the output fits the overlay format.

## When to prefer a sidecar

- When the best in-place edit hits the weight-delta limit before the drift limit.
- When you want to add a new skill without touching what the parent already does well.
- When several submissions should stack: sidecars on the same parent add up, and each one keeps its own `a`.

Sidecars are scored on the same shards as in-place edits. The merge into the full model wraps the parent's MoE block, so both
the routed experts and the shared expert stay stock, and only the sidecar term is added.
