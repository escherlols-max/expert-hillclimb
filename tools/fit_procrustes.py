"""Fit Plato's-key maps between host and donor hidden spaces (orthogonal Procrustes + scale).

    python tools/fit_procrustes.py --pairs pairs.npz --out maps.npz [--rank 0]
    python tools/fit_procrustes.py --selftest

pairs.npz holds H [M, d_host] and D [M, d_donor] (same tokens / words, matched layers).
Both are centred; R_in maps host -> donor, R_out maps donor -> host. If the widths differ,
the map is a semi-orthogonal matrix (thin SVD). Reports cosine fit on a 10% held-out split.
"""
import argparse

import numpy as np


def procrustes(A, B):
    """Find semi-orthogonal Q and scale s minimising ||s A Q - B||, A [M, da], B [M, db]."""
    U, S, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    Q = U @ Vt  # [da, db]
    s = S.sum() / max(float((A * A).sum()), 1e-12)
    return Q, s


def cos_fit(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float(np.median(num / den))


def fit(H, D, seed=0):
    r = np.random.default_rng(seed)
    idx = r.permutation(len(H))
    cut = int(0.9 * len(H))
    tr, ho = idx[:cut], idx[cut:]
    mh, md = H[tr].mean(0), D[tr].mean(0)
    Hc, Dc = H - mh, D - md
    Q_in, s_in = procrustes(Hc[tr], Dc[tr])
    Q_out, s_out = procrustes(Dc[tr], Hc[tr])
    rep = {
        "cos_in_heldout": cos_fit(s_in * Hc[ho] @ Q_in, Dc[ho]),
        "cos_out_heldout": cos_fit(s_out * Dc[ho] @ Q_out, Hc[ho]),
        "n_train": int(cut), "n_heldout": int(len(ho)),
    }
    maps = {"Q_in": Q_in, "s_in": np.float32(s_in), "mu_host": mh, "mu_donor": md,
            "Q_out": Q_out, "s_out": np.float32(s_out)}
    return maps, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs")
    ap.add_argument("--out", default="maps.npz")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = np.random.default_rng(0)
        H = r.normal(0, 1, (4000, 64))
        R, _ = np.linalg.qr(r.normal(0, 1, (64, 64)))
        D = 2.0 * H @ R + 0.1 * r.normal(0, 1, (4000, 64))
        maps, rep = fit(H, D)
        print(rep)
        assert rep["cos_in_heldout"] > 0.95, rep
        print("selftest ok")
        return
    z = np.load(a.pairs)
    maps, rep = fit(z["H"].astype(np.float64), z["D"].astype(np.float64))
    np.savez(a.out, **{k: np.asarray(v, np.float32) for k, v in maps.items()})
    print(rep)


if __name__ == "__main__":
    main()
