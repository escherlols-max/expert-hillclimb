"""Pull ONE routed expert (W_gate, W_up, W_down, dequantised to float32) of a GLM-5.x model straight from a Hugging Face
safetensors repo, without downloading whole shards: the tool reads the shard header and then only the byte ranges of the
three weights and their FP8 block scales (about 40 MB for a GLM-5.3 expert instead of a 5 GB shard).

    python tools/extract_expert.py --repo zai-org/GLM-5.2 --layer 40 --expert 17 --out glm52_L40_E17.npz
    python tools/extract_expert.py --local_dir path/to/checkpoint --layer 2 --expert 1 --out e.npz     # local copy
    python tools/extract_expert.py --selftest

The output npz has W_gate [I, H], W_up [I, H], W_down [H, I] (float32), `family` = "glm5", and a JSON `meta` string
(repo, revision, layer, expert, source dtypes). It works for any checkpoint that names experts
`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` (FP8 e4m3 + `weight_scale_inv`, or BF16 / F16 / F32).
Use `--full_shard` to download whole shard files instead of byte ranges (for mirrors that refuse range requests).
"""
import argparse, json, os, struct

import numpy as np


def _e4m3fn_table():
    """All 256 float8 e4m3fn values (1 sign, 4 exponent bits with bias 7, 3 mantissa bits, no inf, 0x7f / 0xff = NaN)."""
    v = np.arange(256, dtype=np.int64)
    s = np.where(v & 0x80, -1.0, 1.0)
    e = (v >> 3) & 0xF
    m = (v & 0x7).astype(np.float64)
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7.0))
    out = s * val
    out[(v & 0x7F) == 0x7F] = np.nan
    return out.astype(np.float32)


E4M3 = _e4m3fn_table()


def decode(raw, dtype, shape):
    if dtype == "F8_E4M3":
        return E4M3[np.frombuffer(raw, np.uint8)].reshape(shape)
    if dtype == "BF16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, np.float32).reshape(shape).copy()
    raise ValueError(f"unsupported dtype {dtype}")


def dequant(w, scale_inv, block):
    """w [R, C] * scale_inv [ceil(R/br), ceil(C/bc)] broadcast over 128x128 (or `block`) tiles; ragged edge tiles included."""
    br, bc = block
    R, C = w.shape
    if scale_inv.shape != (-(-R // br), -(-C // bc)):
        raise ValueError(f"scale grid {scale_inv.shape} does not match {w.shape} / {block}")
    s = np.repeat(np.repeat(scale_inv, br, axis=0)[:R], bc, axis=1)[:, :C]
    return (w * s).astype(np.float32)


class Source:
    """Byte-range reader over HF hub files (or a local directory)."""

    def __init__(self, repo=None, revision=None, local_dir=None, full_shard=False, cache_dir=None):
        self.repo, self.rev, self.local, self.full, self.cache_dir = repo, revision, local_dir, full_shard, cache_dir
        self.headers = {}

    def small_file(self, name):
        if self.local:
            return os.path.join(self.local, name)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, name, revision=self.rev, cache_dir=self.cache_dir)

    def read(self, fname, start, end):
        """Bytes [start, end) of file fname."""
        if self.local or self.full:
            path = os.path.join(self.local, fname) if self.local else self.small_file(fname)
            with open(path, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        h = build_hf_headers()
        h["Range"] = f"bytes={start}-{end - 1}"
        for _ in range(3):
            r = requests.get(hf_hub_url(self.repo, fname, revision=self.rev), headers=h, timeout=120, allow_redirects=True)
            if r.status_code == 206 and len(r.content) == end - start:
                return r.content
        raise IOError(f"range request failed for {fname} [{start}, {end}): HTTP {r.status_code}; retry with --full_shard")

    def header(self, fname):
        if fname not in self.headers:
            n = struct.unpack("<Q", self.read(fname, 0, 8))[0]
            hdr = json.loads(self.read(fname, 8, 8 + n))
            hdr.pop("__metadata__", None)
            self.headers[fname] = (8 + n, hdr)
        return self.headers[fname]

    def tensor(self, weight_map, name):
        fname = weight_map[name]
        base, hdr = self.header(fname)
        info = hdr[name]
        a, b = info["data_offsets"]
        return decode(self.read(fname, base + a, base + b), info["dtype"], info["shape"]), info["dtype"]


def extract(src, layer, expert):
    cfg = json.load(open(src.small_file("config.json"), encoding="utf-8"))
    ip = src.small_file("model.safetensors.index.json")
    wm = json.load(open(ip, encoding="utf-8"))["weight_map"]
    block = tuple((cfg.get("quantization_config") or {}).get("weight_block_size", (128, 128)))
    out, dts = {}, {}
    for which, key in (("gate", "W_gate"), ("up", "W_up"), ("down", "W_down")):
        name = f"model.layers.{layer}.mlp.experts.{expert}.{which}_proj.weight"
        if name not in wm:
            raise SystemExit(f"{name} is not in the checkpoint index (dense layer, MTP layer or wrong expert id?)")
        w, dt = src.tensor(wm, name)
        sname = name[:-6] + "weight_scale_inv"
        if sname in wm:
            s, _ = src.tensor(wm, sname)
            w = dequant(w, s, block)
        out[key], dts[key] = w.astype(np.float32), dt
    meta = {"repo": src.repo or "local", "revision": src.rev, "layer": layer, "expert": expert, "source_dtypes": dts,
            "block": list(block), "family": "glm5", "hidden": int(out["W_gate"].shape[1]), "inter": int(out["W_gate"].shape[0])}
    return out, meta


def selftest():
    import tempfile
    r = np.random.default_rng(0)
    # e4m3 table spot checks: 1.0 = 0x38, 448 = 0x7e, smallest subnormal 2^-9 = 0x01, -2 = 0xc0
    assert E4M3[0x38] == 1.0 and E4M3[0x7E] == 448.0 and E4M3[0x01] == 2.0 ** -9 and E4M3[0xC0] == -2.0 and np.isnan(E4M3[0x7F])
    # write a tiny fake checkpoint (FP8 codes + scales, ragged tiles) and read one expert back
    d = tempfile.mkdtemp()
    H, I, blk = 20, 12, (8, 8)
    tensors, true = {}, {}
    for which, shp in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
        codes = r.integers(0, 0x7E, size=shp).astype(np.uint8)
        sc = r.uniform(0.001, 0.01, size=(-(-shp[0] // 8), -(-shp[1] // 8))).astype(np.float32)
        n = f"model.layers.3.mlp.experts.5.{which}_proj.weight"
        tensors[n] = ("F8_E4M3", codes)
        tensors[n[:-6] + "weight_scale_inv"] = ("F32", sc)
        true[which] = E4M3[codes] * np.repeat(np.repeat(sc, 8, 0)[: shp[0]], 8, 1)[:, : shp[1]]
    hdr, blob, off = {}, b"", 0
    for n, (dt, a) in tensors.items():
        bts = a.tobytes()
        hdr[n] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(bts)]}
        blob += bts
        off += len(bts)
    hj = json.dumps(hdr).encode()
    with open(os.path.join(d, "model-1.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + blob)
    json.dump({"weight_map": {n: "model-1.safetensors" for n in tensors}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"quantization_config": {"weight_block_size": list(blk)}}, open(os.path.join(d, "config.json"), "w"))
    out, meta = extract(Source(local_dir=d), 3, 5)
    for which, key in (("gate", "W_gate"), ("up", "W_up"), ("down", "W_down")):
        assert np.array_equal(out[key], true[which].astype(np.float32)), key
    assert meta["hidden"] == H and meta["inter"] == I
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="zai-org/GLM-5.3")
    ap.add_argument("--revision")
    ap.add_argument("--local_dir")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--expert", type=int)
    ap.add_argument("--out")
    ap.add_argument("--full_shard", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.layer is None or a.expert is None or not a.out:
        raise SystemExit("--layer, --expert and --out are required")
    src = Source(None if a.local_dir else a.repo, a.revision, a.local_dir, a.full_shard)
    out, meta = extract(src, a.layer, a.expert)
    np.savez(a.out, **out, family=np.array("glm5"), meta=np.array(json.dumps(meta)))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
