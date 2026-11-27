"""Make a small synthetic shard so you can practise the whole loop before real shards exist.

The hidden "ideal" expert is the base expert plus a small random change. g is the gradient of
0.5*||f(x) - f_ideal(x)||^2, so a good tuner moves toward f_ideal and gets a positive score.

    python tools/make_practice_shard.py --out practice.npz [--d 256 --h 128 --n 8000 --seed 0]
"""
import argparse

import numpy as np

from expert_math import expert_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="practice.npz")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = np.random.default_rng(a.seed)
    s = 1.0 / np.sqrt(a.d)
    W = {
        "W_gate": r.normal(0, s, (a.h, a.d)),
        "W_up": r.normal(0, s, (a.h, a.d)),
        "W_down": r.normal(0, 1.0 / np.sqrt(a.h), (a.d, a.h)),
    }
    W_ideal = {k: v + 0.03 * r.normal(0, v.std(), v.shape) for k, v in W.items()}
    x = r.normal(0, 1, (a.n, a.d))
    w = r.uniform(0.05, 0.5, a.n)
    f0 = expert_forward(x, W["W_gate"], W["W_up"], W["W_down"])
    fi = expert_forward(x, W_ideal["W_gate"], W_ideal["W_up"], W_ideal["W_down"])
    g = (f0 - fi) + 0.05 * r.normal(0, (f0 - fi).std(), f0.shape)  # noisy gradient
    cut = int(0.8 * a.n)
    out = {k: v.astype(np.float16) for k, v in W.items()}
    out.update({
        "x_train": x[:cut], "w_train": w[:cut], "g_train": g[:cut],
        "x_heldout": x[cut:], "w_heldout": w[cut:], "g_heldout": g[cut:],
        "pass_train": np.zeros(cut, np.int8), "pass_heldout": np.zeros(a.n - cut, np.int8),
    })
    out = {k: (v.astype(np.float16) if v.dtype.kind == "f" else v) for k, v in out.items()}
    np.savez(a.out, **out)
    print("wrote", a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
