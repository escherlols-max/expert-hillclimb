# 04. Translated experts ("Plato's key")

## Idea
Different models learn hidden spaces with similar geometry. Hazy Research ("Plato's key", 2026-11-25:
https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key) showed that a single rotation plus a scale,
fitted in closed form (orthogonal Procrustes) on a few thousand shared facts, maps one model's hidden
states into another's well enough to transfer facts. Median transfer was 41%, against 55% when a model
swaps its own states (single-token factual recall, base models up to 27B).

We use the same trick to bring an expert from a **different architecture** (a donor MoE such as a Qwen,
DeepSeek or OLMoE expert) into the host:

    h  -> R_in  (host -> donor space, rotation + scale, fitted)
       -> donor expert  E_d
       -> R_out (donor -> host space)
       -> added as a bundled branch:  parent_gate_weight * tanh(a) * R_out( E_d( R_in h ) )

- `a` starts at 0, so the host is unchanged at step 0.
- The branch fires when its **parent** host expert is routed. The parent is a host expert in the same pillar.
- Tokenizers don't matter: experts act on hidden states, not tokens.

## What you get
For a translated-expert task, the shard adds three things:
- **paired hidden states:** host `H` [M, 2304] and donor `D` [M, d_donor] at matched layers, on the same texts (aligned by word), to fit `R_in` / `R_out`;
- **the donor expert's weights;**
- **the usual host-side fields** (`x`, `w`, `g`) for the parent expert.

## Steps
1. Fit the maps: `python tools/fit_procrustes.py --pairs pairs.npz --out maps.npz`. This is closed form (SVD) plus a scale; report fit quality on held-out pairs.
2. Optionally refine the maps or the donor expert with the same first-order objective as native experts, keeping both close to their fitted or original values.
3. Score exactly as for native experts (held-out first-order gain, drift cap). The drift is measured against the host with the branch at `a = 0`, so it is the branch's own contribution.

## Caveats
- The source result is about single facts, not skills. Treat transfer as an experiment.
- Layers don't line up one to one between models. Matching layers is part of the work: try relative depth first, and report what you tried.
- Licences: only donors whose licence allows modification and redistribution of derived weights.
