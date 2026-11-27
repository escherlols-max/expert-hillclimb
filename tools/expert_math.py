"""Shared numpy maths for scoring. Deterministic, CPU only.

Two expert families, chosen by the shard field `family` (absent means "inkling2"):
  inkling2  Inkling 2 Mini: cap(gate) * sigmoid(gate) * cap(up), then down   (cap_c(z) = c * tanh(z / c))
  glm5      GLM-5.x:        silu(gate) * up, then down                        (plain SwiGLU, no caps)
"""
import numpy as np


GATE_CAP, UP_CAP = 4.0, 25.0  # Inkling 2 Mini mlp_gate_softcap / mlp_up_softcap
FAMILIES = ("inkling2", "glm5")


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def expert_forward(x, W_gate, W_up, W_down, gate_cap=GATE_CAP, up_cap=UP_CAP, family="inkling2"):
    """One expert. x [N, d]; W_gate/W_up [h, d]; W_down [d, h]."""
    gate = x @ W_gate.T
    up = x @ W_up.T
    if family == "glm5":
        act = gate * sigmoid(gate) * up
    elif family == "inkling2":
        act = np.tanh(gate / gate_cap) * gate_cap * sigmoid(gate) * (np.tanh(up / up_cap) * up_cap)
    else:
        raise ValueError(f"unknown expert family {family!r}")
    return act @ W_down.T


def shard_family(z):
    """Family of an opened npz (or dict): the `family` string array, else inkling2."""
    keys = z.files if hasattr(z, "files") else z.keys()
    return str(np.asarray(z["family"])) if "family" in keys else "inkling2"


def load_shard(path):
    """Loads a shard as float32. g arrays are stored divided by g_scale (fp16 underflow); this restores them."""
    z = np.load(path)
    fam = shard_family(z)
    if fam not in FAMILIES:
        raise ValueError(f"unknown expert family {fam!r}")
    d = {k: z[k].astype(np.float32) for k in z.files if z[k].dtype.kind in "fiu"}
    gs = float(d.get("g_scale", 1.0))
    for k in ("g_train", "g_heldout"):
        if k in d:
            d[k] = d[k] * gs
    d["family"] = fam
    d["gate_cap"] = float(d.get("gate_cap", GATE_CAP))
    d["up_cap"] = float(d.get("up_cap", UP_CAP))
    return d


def weights(d, prefix=""):
    return tuple(d[prefix + k].astype(np.float32) for k in ("W_gate", "W_up", "W_down"))


def forward_shard(shard, x, W):
    """Expert output for a loaded shard's family and caps."""
    return expert_forward(x, *W, shard["gate_cap"], shard["up_cap"], shard.get("family", "inkling2"))


def evaluate(shard, W_new, split="heldout", batch=4096):
    """Return (score, drift, rel_delta) for new weights against the shard's base weights."""
    W0 = weights(shard)
    x, w, g = shard["x_" + split], shard["w_" + split], shard["g_" + split]
    dL = 0.0
    num = 0.0
    den = 0.0
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        f0 = forward_shard(shard, xb, W0)
        f1 = forward_shard(shard, xb, W_new)
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
