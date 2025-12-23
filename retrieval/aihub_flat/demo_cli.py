from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from retrieval.aihub_flat.config import AIHubIndexConfig
from retrieval.aihub_flat.fused_retriever import AIHubFusedRetriever


def _force_utf8_stdout() -> None:
    """
    Windows PowerShell 기본 cp949에서 UnicodeEncodeError 방지.
    Python 3.7+ 에서만 reconfigure 지원.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return s


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    _force_utf8_stdout()

    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--domain_id", type=int, default=None)
    p.add_argument("--runtime_yaml", type=str, default=None, help="configs/aihub_retrieval_runtime.yaml (기본: auto)")
    p.add_argument("--index_dir", type=str, default=None, help="indexes 폴더 override")
    p.add_argument("--show_cfg", action="store_true")
    p.add_argument("--debug_keys", action="store_true")
    p.add_argument("--max_ts_text", type=int, default=240)
    p.add_argument("--max_tl_q", type=int, default=260)

    # ✅ 재현성 체크용 JSON 저장 옵션
    p.add_argument("--json_out", type=str, default=None, help="결과를 JSON으로 저장 (e.g., out1.json)")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    # (optional) seed 고정: numpy만이라도 고정
    try:
        np.random.seed(int(args.seed))
    except Exception:
        pass

    cfg = AIHubIndexConfig.default()

    # index_dir override
    if args.index_dir:
        object.__setattr__(cfg, "index_dir", Path(args.index_dir))

    # build (runtime_yaml 포함: quota/out_k/tlq/tsq는 retriever.runtime 에 저장되어야 함)
    r = AIHubFusedRetriever.build(cfg, runtime_yaml=args.runtime_yaml)

    out = r.retrieve_fused(args.query, domain_id=args.domain_id)

    # runtime/cfg 요약 (있는 경우)
    runtime_yaml_path = getattr(r, "runtime_yaml", None)
    runtime_dict = getattr(r, "runtime", {}) or {}
    cfg_dict = {
        "index_dir": str(getattr(r.cfg, "index_dir", "")),
        "topk_coarse_docs": int(getattr(r.cfg, "topk_coarse_docs", 0)),
        "topk_ts_final": int(getattr(r.cfg, "topk_ts_final", 0)),
        "topk_tl_final": int(getattr(r.cfg, "topk_tl_final", 0)),
        "rrf_k": int(getattr(r.cfg, "rrf_k", 0)),
        "weight_ts": float(getattr(r.cfg, "weight_ts", 0.0)),
        "weight_tl": float(getattr(r.cfg, "weight_tl", 0.0)),
    }

    if args.show_cfg:
        print("\n==== RUNTIME YAML (auto/resolved) ====")
        print("runtime_yaml :", str(runtime_yaml_path or args.runtime_yaml or "(auto)"))
        if runtime_dict:
            for k in [
                "ts_index",
                "tl_index",
                "backend",
                "kdoc",
                "kts_pool",
                "rrf_k",
                "fusion_mode",
                "out_k",
                "tl_quota",
                "ts_quota",
                "quota_strategy",
                "no_self_hit_tl",
            ]:
                if k in runtime_dict:
                    print(f"{k:14s}: {runtime_dict.get(k)}")
        print("=====================================")

        print("\n==== FINAL CFG (selected fields) ====")
        for k, v in cfg_dict.items():
            print(f"{k:14s}: {v}")
        print("====================================\n")

    # debug: key list
    if args.debug_keys:
        ts0 = out.get("ts_context", [])[:1]
        tl0 = out.get("tl_hints", [])[:1]
        if ts0:
            print("[DEBUG] First TS item keys:", sorted(list(ts0[0].keys())))
        if tl0:
            print("[DEBUG] First TL item keys:", sorted(list(tl0[0].keys())))
        print()

    # pretty print
    print("\n=== TS CONTEXT (evidence pool) ===")
    for i, x in enumerate(out.get("ts_context", []), 1):
        score = float(x.get("score", 0.0))
        lid = x.get("lid")
        doc_id = x.get("doc_id")
        chunk_id = x.get("chunk_id")
        origin_path = x.get("origin_path")
        print(f"[TS#{i}] score={score:.4f} lid={lid} doc_id={doc_id} chunk_id={chunk_id}")
        if origin_path:
            print("  origin_path:", _safe_text(origin_path))
        txt = (_safe_text(x.get("text"))[: int(args.max_ts_text)]).replace("\n", " ")
        print(" ", txt, "...\n")

    print("\n=== TL HINTS (hint pool) ===")
    for i, x in enumerate(out.get("tl_hints", []), 1):
        score = float(x.get("score", 0.0))
        lid = x.get("lid")
        qa_id = x.get("qa_id")
        domain_id = x.get("domain_id")
        q_type = x.get("q_type")
        origin_path = x.get("origin_path")
        print(f"[TL#{i}] score={score:.4f} lid={lid} qa_id={qa_id} domain_id={domain_id} q_type={q_type}")
        if origin_path:
            print("  origin_path:", _safe_text(origin_path))
        q = _safe_text(x.get("question"))
        if q:
            print(" ", q[: int(args.max_tl_q)].replace("\n", " "), "...\n")
        else:
            print("  (no question text in meta)\n")

    fused = out.get("fused_ranked", []) or []
    # fused 구성 카운트
    n_ts = sum(1 for it in fused if str(it.get("key", "")).startswith("TS::") or str(it.get("source", "")).startswith("ts_"))
    n_tl = len(fused) - n_ts
    fm = runtime_dict.get("fusion_mode", "rrf")
    print("\n=== FUSED (what the system will actually use) ===")
    print(f"fusion_mode={fm}  fused_total={len(fused)}  TL={n_tl}  TS={n_ts}")

    # ✅ json_out 저장
    if args.json_out:
        payload: Dict[str, Any] = {
            "query": args.query,
            "domain_id": args.domain_id,
            "runtime_yaml": str(runtime_yaml_path or args.runtime_yaml or ""),
            "runtime": runtime_dict,
            "cfg": cfg_dict,
            "result": out,  # fused_ranked / ts_context / tl_hints / use_opts 등 포함
        }
        _write_json(Path(args.json_out), payload)
        print("\n[OK] saved json_out:", args.json_out)


if __name__ == "__main__":
    main()
