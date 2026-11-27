"""Shared numpy maths for scoring. Deterministic, CPU only."""
import numpy as np


def silu(z):
    return z / (1.0 + np.exp(-z))


def expert_forward(x, W_gate, W_up, W_down):
    """SwiGLU expert. x [N, d]; W_gate/W_up [h, d]; W_down [d, h]."""
    return (silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T


def load_shard(path):
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


def weights(d, prefix=""):
    return tuple(d[prefix + k].astype(np.float32) for k in ("W_gate", "W_up", "W_down"))


def evaluate(shard, W_new, split="heldout", batch=4096):
    """Return (score, drift, rel_delta) for new weights against the shard's base weights."""
    W0 = weights(shard)
    x, w, g = shard["x_" + split], shard["w_" + split], shard["g_" + split]
    dL = 0.0
    num = 0.0
    den = 0.0
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        f0 = expert_forward(xb, *W0)
        f1 = expert_forward(xb, *W_new)
        diff = f1 - f0
        dL += float(np.sum(w[i:i + batch] * np.sum(g[i:i + batch] * diff, axis=1)))
        num += float(np.sum(diff * diff))
        den += float(np.sum(f0 * f0))
    score = -dL / float(np.sum(w))
    drift = num / max(den, 1e-12)
    d2 = sum(float(np.sum((a - b) ** 2)) for a, b in zip(W_new, W0))
    n2 = sum(float(np.sum(b ** 2)) for b in W0)
    rel_delta = (d2 / max(n2, 1e-12)) ** 0.5
    return score, drift, rel_delta


MAX_DRIFT = 0.05
MAX_REL_DELTA = 0.10
