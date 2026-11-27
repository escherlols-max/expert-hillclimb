# 02. Getting a shard

Shards are published per pillar and round as GitHub release assets on this repo. Each is about
100-200 MB of `float16` `.npz`.

    pillars/<pillar>/round-NN/shards.json     # index: expert id, layer, pass mix, n_tokens, sha256, download URL

Each shard is a file named `expert_L{layer}_E{expert}.npz` with these arrays:

- `x_train`, `w_train`, `g_train`
- `x_heldout`, `w_heldout`, `g_heldout`
- `W_gate`, `W_up`, `W_down`
- `pass_train`, `pass_heldout`

Each shard comes with a `meta.json` holding the sha256, round, pillar, token counts, the source text set, and the host model revision.

**Always check the sha256** against `shards.json` before tuning.

Until real shards are released, generate a practice shard from the tiny synthetic MoE:

    python tools/make_practice_shard.py --out practice.npz
