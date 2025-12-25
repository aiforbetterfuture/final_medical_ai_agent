from __future__ import annotations

"""
retrieval.aihub_flat.demo_cli

CLI for smoke-testing TS/TL retrieval + fusion.

Drop-in patch:
- Adds --json_out to write full retrieval output to a JSON file (utf-8).
- Makes stdout/stderr more robust on Windows consoles (cp949) to avoid UnicodeEncodeError.
"""

import argparse
import json
import sys
from pathlib import Path

from .config import AIHubIndexConfig
from .fused_retriever import AIHubFusedRetriever


def _force_utf8_stdout() -> None:
    # Windows PowerShell default encoding can be cp949; printing Korean/Unicode may error.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            try:
                reconfig(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()

    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", type=str, default=None, help="indexes 폴더 (기본: data/aihub_71874/indexes)")
    p.add_argument("--runtime_yaml", type=str, default="auto", help='runtime yaml 경로 또는 "auto"')
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--domain_id", type=int, default=None)

    p.add_argument("--show_cfg", action="store_true")
    p.add_argument("--debug_keys", action="store_true")
    p.add_argument("--json_out", type=str, default="", help="Write full output JSON to this path (utf-8).")

    args = p.parse_args(argv)

    cfg = AIHubIndexConfig.default()
    if args.index_dir:
        cfg = AIHubIndexConfig(index_dir=Path(args.index_dir))

    r = AIHubFusedRetriever.build(cfg, runtime_yaml=args.runtime_yaml)
    out = r.retrieve_fused(args.query, domain_id=args.domain_id)

    # Optional JSON output (determinism checks / debugging)
    if args.json_out:
        try:
            payload = {
                "query": args.query,
                "domain_id": args.domain_id,
                "runtime_yaml_arg": args.runtime_yaml,
                "runtime_yaml_resolved": str(getattr(r, "runtime_yaml_path", "") or ""),
                "runtime": getattr(r, "runtime", {}) or {},
                "cfg": {
                    "index_dir": str(cfg.index_dir),
                    "topk_coarse_docs": int(cfg.topk_coarse_docs),
                    "topk_ts_final": int(cfg.topk_ts_final),
                    "topk_tl_final": int(cfg.topk_tl_final),
                    "rrf_k": int(cfg.rrf_k),
                    "weight_ts": float(getattr(cfg, "weight_ts", 1.0)),
                    "weight_tl": float(getattr(cfg, "weight_tl", 1.0)),
                },
                "output": out,
            }
            fused_ranked = out.get("fused_ranked")
            # Convenience: top-level fused_ranked/fused_keys for quick diff checks
            payload["fused_ranked"] = fused_ranked if isinstance(fused_ranked, list) else []
            payload["fused_keys"] = [x.get("key") for x in payload["fused_ranked"] if isinstance(x, dict)]
            payload["schema_version"] = 1
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] wrote json_out: {args.json_out}")
        except Exception as e:
            print(f"[WARN] failed to write json_out: {e}")

    if args.show_cfg:
        ry = getattr(r, "runtime_yaml_path", None)
        print("\n==== RUNTIME YAML (auto/resolved) ====")
        print("runtime_yaml :", str(ry) if ry else "(none)")
        if getattr(r, "runtime", None):
            keys = [
                "ts_index", "tl_index", "backend", "kdoc", "kts_pool", "rrf_k",
                "fusion_mode", "out_k", "tl_quota", "ts_quota", "quota_strategy",
                "no_self_hit_tl",
            ]
            for k in keys:
                if k in r.runtime:
                    print(f"{k:<13}: {r.runtime.get(k)}")
        else:
            print("(runtime yaml not found or empty)")
        print("=====================================\n")

        print("==== FINAL CFG (selected fields) ====")
        print("index_dir       :", cfg.index_dir)
        print("topk_coarse_docs:", cfg.topk_coarse_docs)
        print("topk_ts_final   :", cfg.topk_ts_final)
        print("topk_tl_final   :", cfg.topk_tl_final)
        print("rrf_k           :", cfg.rrf_k)
        print("weight_ts       :", getattr(cfg, "weight_ts", None))
        print("weight_tl       :", getattr(cfg, "weight_tl", None))
        print("====================================\n")

    if args.debug_keys:
        ts0 = out["ts_context"][0] if out.get("ts_context") else None
        tl0 = out["tl_hints"][0] if out.get("tl_hints") else None
        f0 = out["fused_ranked"][0] if out.get("fused_ranked") else None

        if isinstance(ts0, dict):
            print("[DEBUG] First TS item keys:", sorted(list(ts0.keys())))
        if isinstance(tl0, dict):
            print("[DEBUG] First TL item keys:", sorted(list(tl0.keys())))
        if isinstance(f0, dict):
            print("[DEBUG] First FUSED item keys:", sorted(list(f0.keys())))
        print()

    print("\n=== TS CONTEXT (evidence pool) ===")
    for i, x in enumerate(out.get("ts_context", []), 1):
        if not isinstance(x, dict):
            continue
        print(f"[TS#{i}] score={x.get('score', 0.0):.4f} lid={x.get('lid')} doc_id={x.get('doc_id')} chunk_id={x.get('chunk_id')}")
        op = x.get("origin_path")
        if op:
            print("  origin_path:", op)
        txt = (x.get("text") or "")[:240].replace("\n", " ")
        print(" ", txt, "...\n")

    print("\n=== TL HINTS (hint pool) ===")
    for i, x in enumerate(out.get("tl_hints", []), 1):
        if not isinstance(x, dict):
            continue
        print(f"[TL#{i}] score={x.get('score', 0.0):.4f} lid={x.get('lid')} qa_id={x.get('qa_id')} domain_id={x.get('domain_id')} q_type={x.get('q_type')}")
        op = x.get("origin_path")
        if op:
            print("  origin_path:", op)
        q = (x.get("question") or "")
        if q:
            print(" ", q[:260].replace("\n", " "), "...\n")
        else:
            print("  (no question text in meta)\n")

    c = out.get("fused_counts") or {"total": len(out.get("fused_ranked", [])), "tl": None, "ts": None}
    print("\n=== FUSED (what the system will actually use) ===")
    print(f"fusion_mode={out.get('fusion_mode')}  fused_total={c.get('total')}  TL={c.get('tl')}  TS={c.get('ts')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
