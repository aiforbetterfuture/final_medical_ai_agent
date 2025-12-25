"""Compare fused keys between two demo_cli --json_out outputs.

Usage:
  python tools/compare_fused_keys.py experiments/retrieval_tuning/out1.json experiments/retrieval_tuning/out2.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def _get_fused(d: dict):
    if isinstance(d.get("fused_ranked"), list):
        return d["fused_ranked"]
    # backward-compat
    out = d.get("output")
    if isinstance(out, dict) and isinstance(out.get("fused_ranked"), list):
        return out["fused_ranked"]
    return None

def main(a_path: str, b_path: str) -> int:
    a = _load(Path(a_path))
    b = _load(Path(b_path))
    fa = _get_fused(a); fb = _get_fused(b)
    if fa is None or fb is None:
        print("[ERR] fused_ranked not found.")
        print(" a.keys=", sorted(a.keys()))
        print(" b.keys=", sorted(b.keys()))
        if isinstance(a.get("output"), dict): print(" a.output.keys=", sorted(a["output"].keys()))
        if isinstance(b.get("output"), dict): print(" b.output.keys=", sorted(b["output"].keys()))
        return 2
    ka = [x.get("key") for x in fa if isinstance(x, dict)]
    kb = [x.get("key") for x in fb if isinstance(x, dict)]
    print("same_fused_keys", ka == kb)
    if ka != kb:
        print("a_first10", ka[:10])
        print("b_first10", kb[:10])
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/compare_fused_keys.py <out1.json> <out2.json>")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
