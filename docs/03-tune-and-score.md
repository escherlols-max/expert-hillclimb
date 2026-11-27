# 03. Tune and score (reference pipeline)

    pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
    python tools/tune_expert.py --shard expert_L9_E41.npz --out overlay_L9_E41.npz --steps 400 --lam 30 --seed 0
    python tools/score_expert.py --shard expert_L9_E41.npz --overlay overlay_L9_E41.npz

`tune_expert.py` minimises, on the TRAIN rows only:

    sum_t w_t <g_t, f(x_t)>  +  lam * mean_t ||f(x_t) - f0(x_t)||^2  +  anchor * ||W - W0||^2

with Adam on CPU. Expect minutes, not hours, on 4 vCPUs for 20k tokens (to be measured on the first real shards).

`score_expert.py` prints the held-out score, the drift and the weight-delta norm, plus VALID or INVALID, and writes `report.json`.

## Ideas that tend to help (the hill-climbing part)
- **Tune the trust region:** sweep `lam` and the step count. Stop early on held-out drift.
- **Fit only part of the expert:** e.g. `W_down` only, or a low-rank delta, which keeps drift small for the same gain.
- **Reweight tokens:** rows from looped passes 2-3 may matter differently.
- **Use better target data:** build your own shard from a better teacher, see `docs/07-own-shards.md`. A frontier model can write the target responses for the expert's pillar, which counts as "predicting the responses".
- **Combine with second-order information:** the export can also provide per-token loss curvature in later rounds.

Report everything you tried in `report.json` (`notes` field). Failed ideas are useful to everyone.

## Practice baseline (synthetic shard, seed 0)

    python tools/make_practice_shard.py --out practice.npz
    python tools/tune_expert.py --shard practice.npz --out ov.npz --steps 300
    python tools/score_expert.py --shard practice.npz --overlay ov.npz
    # score=0.381 drift=0.0067 rel_delta=0.0465 VALID

With `--lam 1` the score jumps to 5.1, but drift goes to 1.94: INVALID. The drift cap is the whole
game. Find more gain per unit of drift.
