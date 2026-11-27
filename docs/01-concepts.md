# 01. Concepts: why one expert can be tuned on its own

## The expert
Each MoE layer has 128 experts. For every token, the router picks 8 of them and mixes their outputs with
gate weights `w`. An expert is a function `f(x)` from a 2304-dim hidden vector to a 2304-dim update:

    f(x) = W_down( silu(W_gate x) * (W_up x) )        # 2304 -> 1152 -> 2304, about 8M params

## What the GPU export gives you (one shard per expert)
We run the full host model once over target texts (good answers for that pillar) and, for every token
routed to the chosen expert, record:

| field | shape | meaning |
|---|---|---|
| `x` | [N, 2304] | the expert's input for that token |
| `w` | [N] | the router's gate weight for this expert on that token |
| `g` | [N, 2304] | dLoss/d(expert output): the direction the expert's output should move to make the target text more likely |
| weights | 3 matrices | the expert's current weights |

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

    score  =  - dL_heldout / sum_t w_t        (higher is better)
    valid only if  drift_heldout <= 0.05  and  the weight delta's relative norm <= 0.10

The score is a cheap proxy. The real test is merging your expert into the full model and measuring
held-out exam accuracy on GPU. We do that for the leaderboard leaders, and we demote proxy winners
that don't hold up.

## Why rounds
Gradients go stale after experts change. We run in rounds: merge winners, re-export fresh shards, and
climb again. Each round's shards are versioned (`round-01`, `round-02`, ...). Always say which round
you tuned against.
