"""Score a SIDECAR overlay against a shard (held-out rows decide; train rows are reported too).

    python tools/score_sidecar.py --shard expert.npz --overlay sidecar.npz [--report report.json]

The overlay holds the sidecar's W_gate, W_up, W_down and the scalar a. The sidecar fires on the parent's tokens and adds
w * tanh(a) * f_sc(x); the parent stays untouched. VALID when branch_drift <= 0.05 on held-out rows (no weight-delta limit).
See docs/09-sidecars.md.
"""
import argparse
import hashlib
import json

import numpy as np

from expert_math import MAX_BRANCH_DRIFT, evaluate_sidecar, load_shard, load_sidecar


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
    W, av = load_sidecar(a.overlay)
    d = shard["x_heldout"].shape[1]
    if W[0].shape[1] != d or W[2].shape[0] != d or W[0].shape != W[1].shape or W[2].shape[1] != W[0].shape[0]:
        raise SystemExit(f"sidecar shapes {[w.shape for w in W]} do not fit hidden {d}")
    s, bd, sn = evaluate_sidecar(shard, W, av, "heldout")
    ts, tbd, tsn = evaluate_sidecar(shard, W, av, "train")
    valid = bd <= MAX_BRANCH_DRIFT
    rep = {
        "kind": "sidecar", "shard_sha256": sha256(a.shard), "overlay_sha256": sha256(a.overlay), "a": av, "tanh_a": float(np.tanh(av)),
        "sidecar_inter": int(W[0].shape[0]), "parent_inter": int(shard["W_gate"].shape[0]),
        "score_norm_heldout": sn, "score_heldout": s, "branch_drift_heldout": bd,
        "score_norm_train": tsn, "score_train": ts, "branch_drift_train": tbd,
        "valid": valid, "limits": {"max_branch_drift": MAX_BRANCH_DRIFT}, "notes": "",
    }
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"sidecar score_norm={sn:.5f} branch_drift={bd:.4f} tanh(a)={np.tanh(av):.4f} (raw score={s:.6g}; "
          f"train score_norm={tsn:.5f}) {'VALID' if valid else 'INVALID'}")


if __name__ == "__main__":
    main()
