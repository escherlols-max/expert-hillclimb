"""CPU weight-space edits for one expert, scored with the same maths as score_expert.py. numpy only.

Every command reads base weights W0 from a shard (or any npz with W_gate / W_up / W_down), builds new weights, optionally
shrinks the edit until it fits the validity limits on the TRAIN rows (--fit), prints train metrics, and writes an overlay
npz that score_expert.py accepts. Held-out rows are never used here: score the result with score_expert.py.

    python tools/weight_space.py lowrank  --shard S.npz --overlay tuned.npz --rank 8 --out ov.npz
    python tools/weight_space.py svd      --shard S.npz --rank 512 --out ov.npz                # truncate each matrix to rank r
    python tools/weight_space.py rescale  --shard S.npz --top 16 --factor 1.05 --out ov.npz   # scale the top singular values
    python tools/weight_space.py merge    --shard S.npz --inputs a.npz b.npz --coefs 0.5 0.5 --out ov.npz
    python tools/weight_space.py taskvec  --shard S53.npz --base glm52_L40_E17.npz --alpha 0.3 --out ov.npz
    python tools/weight_space.py taskvec  --shard S53.npz --base glm52_L40_E17.npz --sweep=-0.5,-0.25,0.25,0.5,1 --fit --out ov.npz
    python tools/weight_space.py selftest

--sidecar (any command): the edited weights become a SIDECAR expert instead of an in-place change. The parent stays
untouched and the overlay gets the scalar a (default: the closed-form best strength on the train rows under the
branch-drift cap, or --a). Score it with score_sidecar.py (docs/09-sidecars.md).

Task vector: W = W53 + alpha * (W53 - W52), where W52 is the same (layer, expert) of GLM-5.2 (same shapes; extract it with
extract_expert.py). alpha > 0 extrapolates along the 5.2 -> 5.3 update, alpha < 0 moves back toward 5.2.
"""
import argparse, json, sys

import numpy as np

from expert_math import MAX_BRANCH_DRIFT, MAX_DRIFT, MAX_REL_DELTA, best_sidecar_strength, evaluate, evaluate_sidecar, load_shard, weights

NAMES = ("W_gate", "W_up", "W_down")


def load_expert(path):
    z = np.load(path)
    missing = [k for k in NAMES if k not in z.files]
    if missing:
        raise SystemExit(f"{path} lacks {missing}")
    return tuple(z[k].astype(np.float32) for k in NAMES)


def save_overlay(path, W, info=None):
    np.savez(path, **{k: np.ascontiguousarray(w, np.float32) for k, w in zip(NAMES, W)})
    if info is not None:
        print(json.dumps(info))


# ------------------------------------------------------------------ edits (each maps weights to weights)

def svd_truncate(W, rank):
    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    r = min(rank, len(S))
    return ((U[:, :r] * S[:r]) @ Vt[:r]).astype(np.float32)


def lowrank_delta(W0, W1, rank):
    """W0 + best rank-r approximation of (W1 - W0), per matrix."""
    return tuple((a + svd_truncate(b - a, rank)).astype(np.float32) for a, b in zip(W0, W1))


def svd_rescale(W, top, factor):
    """Multiply the `top` largest singular values by `factor` (factor > 1 sharpens the dominant directions)."""
    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    S = S.copy()
    S[:top] *= factor
    return ((U * S) @ Vt).astype(np.float32)


def linear_merge(experts, coefs):
    if len(experts) != len(coefs):
        raise ValueError("one coefficient per expert")
    return tuple(sum(c * e[i] for c, e in zip(coefs, experts)).astype(np.float32) for i in range(3))


def task_vector(W53, W52, alpha):
    for a, b in zip(W53, W52):
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {a.shape} vs {b.shape}: not the same expert family / layer")
    return tuple((a + alpha * (a - b)).astype(np.float32) for a, b in zip(W53, W52))


def fit_to_limits(shard, W0, W1, margin=0.9, iters=30):
    """Largest t in [0, 1] such that W0 + t (W1 - W0) keeps train drift and the weight delta under margin * limits."""
    def ok(t):
        Wt = tuple(a + t * (b - a) for a, b in zip(W0, W1))
        _, drift, rel = evaluate(shard, Wt, "train")
        return drift <= margin * MAX_DRIFT and rel <= margin * MAX_REL_DELTA
    if ok(1.0):
        return 1.0, W1
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo, tuple(a + lo * (b - a) for a, b in zip(W0, W1))


def report(shard, W, extra=None):
    s, d, r = evaluate(shard, W, "train")
    out = {"score_train": s, "drift_train": d, "rel_delta": r, "valid_on_train": d <= MAX_DRIFT and r <= MAX_REL_DELTA}
    out.update(extra or {})
    return out


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lowrank", "svd", "rescale", "merge", "taskvec"):
        p = sub.add_parser(name)
        p.add_argument("--shard", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--fit", action="store_true", help="shrink the edit toward W0 until it is valid on the train rows")
        p.add_argument("--sidecar", action="store_true", help="write the edited expert as a sidecar overlay (with scalar a)")
        p.add_argument("--a", type=float, default=None, help="--sidecar: strength a (default: best on train rows)")
        if name == "lowrank":
            p.add_argument("--overlay", required=True, help="tuned weights whose delta is compressed")
            p.add_argument("--rank", type=int, required=True)
        if name == "svd":
            p.add_argument("--rank", type=int, required=True)
            p.add_argument("--only", default="", help="comma list of W_gate,W_up,W_down (default all)")
        if name == "rescale":
            p.add_argument("--top", type=int, required=True)
            p.add_argument("--factor", type=float, required=True)
            p.add_argument("--only", default="")
        if name == "merge":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--coefs", nargs="+", type=float, required=True)
        if name == "taskvec":
            p.add_argument("--base", required=True, help="the same expert from the older model (e.g. GLM-5.2), npz with W_*")
            p.add_argument("--alpha", type=float, default=0.25)
            p.add_argument("--sweep", default="", help="comma list of alphas: keep the best train score among valid ones")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    shard = load_shard(a.shard)
    W0 = weights(shard)
    only = set(a.only.split(",")) if getattr(a, "only", "") else set(NAMES)
    if a.cmd == "lowrank":
        W1 = lowrank_delta(W0, load_expert(a.overlay), a.rank)
    elif a.cmd == "svd":
        W1 = tuple(svd_truncate(w, a.rank) if n in only else w for n, w in zip(NAMES, W0))
    elif a.cmd == "rescale":
        W1 = tuple(svd_rescale(w, a.top, a.factor) if n in only else w for n, w in zip(NAMES, W0))
    elif a.cmd == "merge":
        W1 = linear_merge([load_expert(p) for p in a.inputs], a.coefs)
    else:
        W52 = load_expert(a.base)
        alphas = [float(v) for v in a.sweep.split(",") if v.strip()] or [a.alpha]
        best = None
        for al in alphas:
            Wa = task_vector(W0, W52, al)
            t = 1.0
            if a.fit:
                t, Wa = fit_to_limits(shard, W0, Wa)
            r = report(shard, Wa, {"alpha": al, "shrink": t})
            print(json.dumps(r))
            if r["valid_on_train"] and (best is None or r["score_train"] > best[0]["score_train"]):
                best = (r, Wa)
        if best is None:
            raise SystemExit("no alpha gave a valid edit on the train rows; try smaller |alpha| or --fit")
        if a.sidecar:
            return save_sidecar(a.out, shard, best[1], a.a, dict(best[0], chosen=True))
        save_overlay(a.out, best[1], dict(best[0], chosen=True))
        return
    if a.sidecar:
        return save_sidecar(a.out, shard, W1, a.a, {"cmd": a.cmd})
    t = 1.0
    if a.fit:
        t, W1 = fit_to_limits(shard, W0, W1)
    save_overlay(a.out, W1, report(shard, W1, {"cmd": a.cmd, "shrink": t}))


def save_sidecar(path, shard, W, a_val=None, extra=None):
    """Write W as a sidecar overlay; a defaults to the closed-form best strength on the TRAIN rows."""
    if a_val is None:
        a_val = best_sidecar_strength(shard, W, "train")
    s, bd, sn = evaluate_sidecar(shard, W, a_val, "train")
    np.savez(path, **{k: np.ascontiguousarray(w, np.float32) for k, w in zip(NAMES, W)}, a=np.float32(a_val))
    info = {"kind": "sidecar", "a": a_val, "tanh_a": float(np.tanh(a_val)), "score_norm_train": sn, "score_train": s,
            "branch_drift_train": bd, "valid_on_train": bd <= MAX_BRANCH_DRIFT}
    info.update(extra or {})
    print(json.dumps(info))


def selftest():
    r = np.random.default_rng(0)
    A = r.normal(size=(40, 30)).astype(np.float32)
    assert np.allclose(svd_truncate(A, 30), A, atol=1e-4)
    B = (r.normal(size=(40, 3)) @ r.normal(size=(3, 30))).astype(np.float32)
    assert np.allclose(svd_truncate(B, 3), B, atol=1e-4)
    W0 = (A, A.T.copy(), A)
    W1 = tuple(w + 0.1 * B if w.shape == B.shape else w + 0.1 * B.T for w in W0)
    assert all(np.allclose(x, y, atol=1e-4) for x, y in zip(lowrank_delta(W0, W1, 3), W1))
    assert np.allclose(svd_rescale(A, 5, 1.0), A, atol=1e-4)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    assert np.allclose(np.linalg.svd(svd_rescale(A, 2, 2.0), compute_uv=False)[:2], np.sort(S[:2] * 2)[::-1], rtol=1e-4)
    assert all(np.allclose(x, 0.25 * a + 0.75 * b) for x, a, b in zip(linear_merge([W0, W1], [0.25, 0.75]), W0, W1))
    assert all(np.allclose(x, a + 0.5 * (a - b)) for x, a, b in zip(task_vector(W1, W0, 0.5), W1, W0))
    # closed-form sidecar strength: on a toy shard the chosen a sits on 90% of the branch-drift cap with a positive gain
    E0 = (r.normal(size=(16, 30)).astype(np.float32) * 0.2, r.normal(size=(16, 30)).astype(np.float32) * 0.2,
          r.normal(size=(30, 16)).astype(np.float32) * 0.2)
    E1 = tuple(e + 0.1 * r.normal(size=e.shape).astype(np.float32) for e in E0)
    sh = {"W_gate": E0[0], "W_up": E0[1], "W_down": E0[2], "gate_cap": 4.0, "up_cap": 25.0, "family": "glm5",
          "x_train": r.normal(size=(300, 30)).astype(np.float32), "w_train": np.full(300, 0.3, np.float32),
          "g_train": r.normal(size=(300, 30)).astype(np.float32)}
    av = best_sidecar_strength(sh, E1, "train")
    s_, bd_, _ = evaluate_sidecar(sh, E1, av, "train")
    assert s_ > 0 and abs(bd_ - 0.9 * MAX_BRANCH_DRIFT) < 1e-4, (s_, bd_)
    print("selftest ok")


if __name__ == "__main__":
    main()
