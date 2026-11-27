# 08. Weight-space edits (GLM-5.x experts)

Gradient tuning (`tune_expert.py`) is one way to climb. The other is to edit the expert's weight matrices directly
and let the scorer judge the result. `tools/weight_space.py` holds the common edits. Each one runs on a plain CPU in
seconds to minutes and writes an overlay that `score_expert.py` accepts.

## Expert families

A shard says which expert function it uses in the field `family`:

| family | model | expert function |
|---|---|---|
| `inkling2` (or no field) | Inkling 2 Mini | `cap4(W_gate x) * sigmoid(W_gate x) * cap25(W_up x)`, then `W_down` |
| `glm5` | GLM-5.2 / GLM-5.3 | `silu(W_gate x) * (W_up x)`, then `W_down` (hidden 6144, inter 2048, no caps) |

All tools read the field, so you do not pass it yourself. A GLM-5.3 expert has about 37.7M parameters
(3 x 6144 x 2048). Its gate weight `w` already includes the top-8 normalisation and the routed scaling factor 2.5,
so the shard's `w` values sum to 2.5 over a token's 8 experts.

Practice shard for the GLM family:

    python tools/make_practice_shard.py --family glm5 --out practice_glm5.npz

## The edits

    python tools/weight_space.py lowrank  --shard S.npz --overlay tuned.npz --rank 8 --out ov.npz
    python tools/weight_space.py svd      --shard S.npz --rank 512 --only W_down --out ov.npz
    python tools/weight_space.py rescale  --shard S.npz --top 16 --factor 1.05 --fit --out ov.npz
    python tools/weight_space.py merge    --shard S.npz --inputs a.npz b.npz --coefs 0.7 0.3 --out ov.npz
    python tools/weight_space.py taskvec  --shard S.npz --base glm52_L40_E17.npz --sweep=-0.25,0.25,0.5 --fit --out ov.npz

- **lowrank:** keeps only the best rank-r part of a tuned delta (`W1 - W0`). A low-rank delta often keeps most of the
  gain and cuts the drift.
- **svd:** truncates the expert itself to rank r. Use it to test how much of the expert the pillar really needs.
- **rescale:** multiplies the largest singular values by a factor. Values above 1 make the dominant directions stronger.
- **merge:** a linear mix of N experts with the same shapes, for example two tuned overlays, or an expert and a
  translated donor.
- **taskvec:** task-vector extrapolation `W = W53 + alpha * (W53 - W52)`. GLM-5.2 and GLM-5.3 have the same shapes,
  so the same (layer, expert) in 5.2 gives the direction of the 5.2 to 5.3 update. `alpha > 0` pushes further along
  that update. `alpha < 0` moves back toward 5.2. Use `--sweep` to try several values. The tool keeps the best train
  score among the valid ones.

`--fit` shrinks any edit toward the base weights until it is valid on the TRAIN rows (drift and weight-delta norm
under 90% of the limits). The tools never read held-out rows. Score the overlay with `score_expert.py` as usual.

## Getting a GLM-5.2 (or any GLM-5.x) expert

`tools/extract_expert.py` reads one expert straight from a Hugging Face safetensors repo. It fetches only the byte
ranges of the three matrices and their FP8 block scales, which is about 40 MB instead of a 5 GB shard. It then
dequantises them to float32.

    python tools/extract_expert.py --repo zai-org/GLM-5.2 --layer 40 --expert 17 --out glm52_L40_E17.npz

Use `--full_shard` if a mirror refuses range requests. Use `--local_dir` for a checkpoint that is already on disk.
Layers 0 to 2 are dense and have no routed experts. Layer 78 is the MTP head and is not part of the exported model.

## Good habits

- Compare every edit against the plain gradient tuner at the same drift. An edit that beats it is worth a note in
  your report.
- Report the exact command line, including `--rank`, `--alpha` and the shrink factor that `--fit` printed.
- An edit from GLM-5.2 is a different model's weights. Check the licence of the source repo before you submit. GLM-5.2
  is MIT.
