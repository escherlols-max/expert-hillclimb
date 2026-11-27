"""Reference CPU tuner: first-order objective + trust region, Adam, TRAIN rows only.

    python tools/tune_expert.py --shard expert.npz --out overlay.npz --steps 400 --lam 30 --seed 0

Minimises  mean_w <g, f(x)>  +  lam * mean||f(x)-f0(x)||^2 / mean||f0(x)||^2  +  anchor * ||W-W0||^2/||W0||^2
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def fwd(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--only", default="", help="comma list of W_gate,W_up,W_down to train (default all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    z = np.load(a.shard)
    t = lambda k: torch.from_numpy(z[k].astype(np.float32))
    x, w, g = t("x_train"), t("w_train"), t("g_train")
    names = ("W_gate", "W_up", "W_down")
    W0 = [t(k) for k in names]
    train = set(a.only.split(",")) if a.only else set(names)
    W = [p.clone().requires_grad_(n in train) for p, n in zip(W0, names)]
    opt = torch.optim.Adam([p for p in W if p.requires_grad], lr=a.lr)
    n0 = sum(float((p ** 2).sum()) for p in W0)
    with torch.no_grad():
        f0_all = fwd(x, *W0)
    f0_scale = float((f0_all ** 2).sum(1).mean())
    wsum = float(w.mean())

    for step in range(a.steps):
        idx = torch.randint(0, len(x), (min(a.batch, len(x)),))
        xb, wb, gb, f0 = x[idx], w[idx], g[idx], f0_all[idx]
        f = fwd(xb, *W)
        lin = (wb * (gb * (f - f0)).sum(1)).mean() / wsum
        drift = ((f - f0) ** 2).sum(1).mean() / f0_scale
        anc = sum(((p - q) ** 2).sum() for p, q in zip(W, W0)) / n0
        loss = lin + a.lam * drift + a.anchor * anc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} lin={lin.item():.5g} drift={drift.item():.4f} anchor={anc.item():.2e}")

    np.savez(a.out, **{n: p.detach().numpy().astype(np.float32) for n, p in zip(names, W)})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
