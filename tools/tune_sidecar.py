"""Reference CPU tuner for a SIDECAR expert: Adam on TRAIN rows only.

    python tools/tune_sidecar.py --shard expert.npz --out sidecar.npz --steps 400 --lam 3 --seed 0
    python tools/tune_sidecar.py --shard expert.npz --init my_weights.npz --out sidecar.npz     # start from any weights

Sidecar output on the parent's tokens: w * tanh(a) * f_sc(x). Init: a copy of the parent with a = --a0 (small), or --init
(W_gate, W_up, W_down, optional a; any inner width). Minimises

    mean_t w_t tanh(a) <g_t, f_sc(x_t)> / (RMS||w g|| * RMS||f0||)  +  lam * mean||tanh(a) f_sc||^2 / mean||f0||^2

(the same scale-free linear term as tune_expert.py, with the branch drift as the trust region). No weight anchor: the
sidecar may move as far from its init as it likes. Score with score_sidecar.py.
"""
import argparse

import numpy as np
import torch

from tune_expert import GATE_CAP, UP_CAP, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="npz with W_gate, W_up, W_down (and optionally a); default: copy of the parent")
    ap.add_argument("--a0", type=float, default=0.05, help="initial a when --init has none")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_a", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    z = np.load(a.shard)
    t = lambda k, src=z: torch.from_numpy(src[k].astype(np.float32))
    x, w, g = t("x_train"), t("w_train"), t("g_train")
    if "g_scale" in z.files:
        g = g * float(z["g_scale"])
    gc = float(z["gate_cap"]) if "gate_cap" in z.files else GATE_CAP
    uc = float(z["up_cap"]) if "up_cap" in z.files else UP_CAP
    fam = str(z["family"]) if "family" in z.files else "inkling2"
    names = ("W_gate", "W_up", "W_down")
    W0 = [t(k) for k in names]
    a0 = a.a0
    if a.init:
        zi = np.load(a.init)
        Wi = [t(k, zi) for k in names]
        if "a" in zi.files:
            a0 = float(zi["a"])
    else:
        Wi = [p.clone() for p in W0]
    W = [p.clone().requires_grad_(True) for p in Wi]
    av = torch.tensor(float(a0), requires_grad=True)
    opt = torch.optim.Adam([{"params": W, "lr": a.lr}, {"params": [av], "lr": a.lr_a}])
    with torch.no_grad():
        f0_all = fwd(x, *W0, gc, uc, fam)
    f0_scale = float((f0_all ** 2).sum(1).mean())
    lin_scale = float(((w[:, None] * g) ** 2).sum(1).mean().sqrt()) * f0_scale ** 0.5 + 1e-30
    print("family", fam, "init a", a0)
    for step in range(a.steps):
        idx = torch.randint(0, len(x), (min(a.batch, len(x)),))
        xb, wb, gb = x[idx], w[idx], g[idx]
        fs = torch.tanh(av) * fwd(xb, *W, gc, uc, fam)
        lin = (wb * (gb * fs).sum(1)).mean() / lin_scale
        drift = (fs ** 2).sum(1).mean() / f0_scale
        loss = lin + a.lam * drift
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} lin={lin.item():.5g} branch_drift={drift.item():.4f} a={av.item():.4f}")
    np.savez(a.out, **{n: p.detach().numpy().astype(np.float32) for n, p in zip(names, W)}, a=np.float32(av.item()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
