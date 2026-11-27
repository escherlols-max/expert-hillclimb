"""Reference CPU tuner: first-order objective + trust region, Adam, TRAIN rows only.

    python tools/tune_expert.py --shard expert.npz --out overlay.npz --steps 400 --lam 3 --seed 0

Minimises  mean_w <g, f(x)> / (RMS||w g|| * RMS||f0||)  +  lam * mean||f(x)-f0(x)||^2 / mean||f0(x)||^2  +  anchor * ||W-W0||^2/||W0||^2
"""
import argparse

import numpy as np
import torch


GATE_CAP, UP_CAP = 4.0, 25.0


def fwd(x, Wg, Wu, Wd, gc=GATE_CAP, uc=UP_CAP, family="inkling2"):
    """inkling2: cap(gate) * sigmoid(gate) * cap(up), then down.  glm5: silu(gate) * up, then down (no caps)."""
    gate, up = x @ Wg.T, x @ Wu.T
    if family == "glm5":
        return (torch.nn.functional.silu(gate) * up) @ Wd.T
    return (torch.tanh(gate / gc) * gc * torch.sigmoid(gate) * (torch.tanh(up / uc) * uc)) @ Wd.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lam", type=float, default=3.0)
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
    if "g_scale" in z.files:
        g = g * float(z["g_scale"])
    gc = float(z["gate_cap"]) if "gate_cap" in z.files else GATE_CAP
    uc = float(z["up_cap"]) if "up_cap" in z.files else UP_CAP
    fam = str(z["family"]) if "family" in z.files else "inkling2"
    if fam not in ("inkling2", "glm5"):
        raise SystemExit(f"unknown expert family {fam!r}")
    print("family", fam)
    names = ("W_gate", "W_up", "W_down")
    W0 = [t(k) for k in names]
    train = set(a.only.split(",")) if a.only else set(names)
    W = [p.clone().requires_grad_(n in train) for p, n in zip(W0, names)]
    opt = torch.optim.Adam([p for p in W if p.requires_grad], lr=a.lr)
    n0 = sum(float((p ** 2).sum()) for p in W0)
    with torch.no_grad():
        f0_all = fwd(x, *W0, gc, uc, fam)
    f0_scale = float((f0_all ** 2).sum(1).mean())
    # scale-free linear term (matches score_norm in expert_math.evaluate): divide by RMS ||w g|| * RMS ||f0||,
    # so --lam means the same thing on every shard whatever its gradient scale
    lin_scale = float(((w[:, None] * g) ** 2).sum(1).mean().sqrt()) * f0_scale ** 0.5 + 1e-30

    for step in range(a.steps):
        idx = torch.randint(0, len(x), (min(a.batch, len(x)),))
        xb, wb, gb, f0 = x[idx], w[idx], g[idx], f0_all[idx]
        f = fwd(xb, *W, gc, uc, fam)
        lin = (wb * (gb * (f - f0)).sum(1)).mean() / lin_scale
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
