from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np

def load_yaml_simple(p: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise SystemExit("Missing dependency: pyyaml. Install: pip install pyyaml") from e
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True, help="e.g., ts_coarse / ts_fine / tl_stem ...")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_yaml_simple(Path(args.config))

    # configs/aihub_71874_paths.yaml 에서 index_dir 읽기
    try:
        idx_dir = Path(cfg["output"]["index_dir"])
    except Exception as e:
        raise SystemExit("Config missing key: output.index_dir") from e

    flat_path = idx_dir / f"{args.index}_flat.npy"
    sidecar = Path(str(flat_path) + ".json")

    if not flat_path.exists():
        raise SystemExit(f"Missing: {flat_path}")

    emb = np.load(flat_path, mmap_mode="r")
    n, d = emb.shape
    print(f"[INDEX] {args.index}")
    print(f"[FILE ] {flat_path}")
    print(f"[SHAPE] n={n:,} d={d} dtype={emb.dtype}")

    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            print(f"[SIDECAR] {sidecar} -> keys={sorted(meta.keys())}")
            for k in ["normalized", "normalize", "is_normalized"]:
                if k in meta:
                    print(f"  - {k} = {meta[k]}")
        except Exception as e:
            print(f"[SIDECAR] exists but failed to read: {e}")

    rng = np.random.default_rng(args.seed)
    m = min(args.sample, n)
    idx = rng.choice(n, size=m, replace=False)

    x = np.asarray(emb[idx], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)

    mean = float(norms.mean())
    std = float(norms.std())
    mn = float(norms.min())
    mx = float(norms.max())
    frac_1pct = float(((norms > 0.99) & (norms < 1.01)).mean())
    frac_5pct = float(((norms > 0.95) & (norms < 1.05)).mean())

    print("\n[NORM CHECK] (sample)")
    print(f"  sample = {m:,}")
    print(f"  mean   = {mean:.6f}")
    print(f"  std    = {std:.6f}")
    print(f"  min    = {mn:.6f}")
    print(f"  max    = {mx:.6f}")
    print(f"  within 1%  (0.99~1.01) : {frac_1pct*100:.2f}%")
    print(f"  within 5%  (0.95~1.05) : {frac_5pct*100:.2f}%")

    if frac_5pct > 0.99:
        print("\n[OK] Stored embeddings look L2-normalized (dot ~= cosine).")
    else:
        print("\n[WARN] Stored embeddings do NOT look normalized.")
        print("  - If you use dot as cosine, ranking may be wrong.")
        print("  - Fix: rebuild embeddings with normalize_embeddings=True, OR compute true cosine at search-time.")

if __name__ == "__main__":
    main()
