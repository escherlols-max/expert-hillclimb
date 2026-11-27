# 01. Concepts: why one expert can be tuned on its own

## The expert
Each MoE layer has 128 experts. For every token, the router picks 8 of them and mixes their outputs with
gate weights `w`. An expert is a function `f(x)` from a 2304-dim hidden vector to a 2304-dim update:

    a(x) = cap4(W_gate x) * sigmoid(W_gate x) * cap25(W_up x)     # cap_c(z) = c * tanh(z / c)
    f(x) = W_down a(x)                                            # 2304 -> 1152 -> 2304, about 8M params

The MoE layer adds `w * f(x)` to the hidden state for each of the token's 8 chosen experts.

## What the GPU export gives you (one shard per expert)
We run the full host model once over target texts (good answers for that pillar) and, for every token
routed to the chosen expert, record:

| field | shape | meaning |
|---|---|---|
| `x` | [N, 2304] | the expert's input for that token |
| `w` | [N] | the router's gate weight for this expert on that token |
| `g` | [N, 2304] | dLoss/d(MoE layer output) at that token. Since the expert adds `w * f(x)`, the expert's own gradient is `w * g`. Stored divided by `g_scale` (fp16 range); the tools multiply it back |
| weights | 3 matrices | the expert's current weights, plus `gate_cap` / `up_cap` |

`N` is up to about 20k tokens, split into `train` and `heldout` parts. Looped-core experts include
all 3 passes as separate rows (column `pass`).

## The objective you are improving
To first order, changing the expert from `f0` to `f` changes the loss by:

    dL  ~=  sum_t  w_t * < g_t , f(x_t) - f0(x_t) >

Lower is better. A pure first-order step can overshoot, because the gradients were measured at the old
weights and the rest of the model has not adapted to the new expert. So every submission must also stay
close to the original behaviour:

    drift  =  mean_t || f(x_t) - f0(x_t) ||^2 / mean_t || f0(x_t) ||^2

## Score (computed on the HELD-OUT rows, never the train rows)

The leaderboard ranks by the scale-free score:

    score_norm  =  - sum_t w_t <g_t, f(x_t) - f0(x_t)>  /  ( ||w g||_F * ||f0||_F )
    valid only if  drift_heldout <= 0.05  and  the weight delta's relative norm <= 0.10

`score_norm` is a weighted cosine between your output change and the ideal direction, times the relative size of the
change, so it does not depend on how large a shard's gradients happen to be. Under the drift cap it can never exceed
sqrt(0.05), about 0.22. The raw first-order gain `-dL / sum w` is also reported.

**Watch the train vs held-out gap.** On a real GLM-5.3 shard with about 1k training tokens, the reference tuner reaches
score_norm 0.15 on train but only 0.002 on held-out rows: most of what it learns does not carry over to new texts.
Closing that gap is the main thing to climb.

The score is a cheap proxy. The real test is merging your expert into the full model and measuring
held-out exam accuracy on GPU. We do that for the leaderboard leaders, and we demote proxy winners
that don't hold up.

## Why rounds
Gradients go stale after experts change. We run in rounds: merge winners, re-export fresh shards, and
climb again. Each round's shards are versioned (`round-01`, `round-02`, ...). Always say which round
you tuned against.
