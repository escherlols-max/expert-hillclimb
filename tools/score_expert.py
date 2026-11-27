"""Score an overlay against a shard's HELD-OUT rows.

    python tools/score_expert.py --shard expert.npz --overlay overlay.npz [--report report.json]
"""
import argparse
import hashlib
import json

import numpy as np

from expert_math import MAX_DRIFT, MAX_REL_DELTA, evaluate, load_shard, weights


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--report", default="report.json")
    a = ap.parse_args()

    shard = load_shard(a.shard)
    ov = np.load(a.overlay)
    W_new = weights({k: ov[k] for k in ov.files})
    for new, old in zip(W_new, weights(shard)):
        if new.shape != old.shape:
            raise SystemExit(f"shape mismatch {new.shape} vs {old.shape}")

    score, drift, rel, snorm = evaluate(shard, W_new, "heldout", with_norm=True)
    tscore, tdrift, _, tnorm = evaluate(shard, W_new, "train", with_norm=True)
    valid = drift <= MAX_DRIFT and rel <= MAX_REL_DELTA
    rep = {
        "shard_sha256": sha256(a.shard),
        "overlay_sha256": sha256(a.overlay),
        "score_norm_heldout": snorm,
        "score_heldout": score,
        "drift_heldout": drift,
        "rel_weight_delta": rel,
        "score_norm_train": tnorm,
        "score_train": tscore,
        "drift_train": tdrift,
        "valid": valid,
        "limits": {"max_drift": MAX_DRIFT, "max_rel_weight_delta": MAX_REL_DELTA},
        "notes": "",
    }
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"score_norm={snorm:.5f} drift={drift:.4f} rel_delta={rel:.4f} (raw score={score:.6g}; "
          f"train score_norm={tnorm:.5f}) {'VALID' if valid else 'INVALID'}")


if __name__ == "__main__":
    main()
