from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .config import AIHubIndexConfig
from .fused_retriever import AIHubFusedRetriever

DEFAULT_RUNTIME_YAML = Path("configs/aihub_retrieval_runtime.yaml")


def _read_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _unwrap_meta(maybe_row: Any) -> Dict[str, Any]:
    """
    AIHub 메타 row는 종종 {id,lid,meta:{...}} 구조입니다.
    - x.get("meta")가 row 전체일 수도 있고
    - 이미 meta flat dict일 수도 있음
    따라서:
      1) dict 아니면 {}
      2) dict이면, 안쪽에 "meta"(dict)가 있으면 그걸 반환
      3) 아니면 그대로 반환
    """
    if not isinstance(maybe_row, dict):
        return {}
    inner = maybe_row.get("meta")
    if isinstance(inner, dict):
        return inner
    return maybe_row


def _debug_print_first(label: str, item: Dict[str, Any]) -> None:
    print(f"\n[DEBUG] First {label} item keys:", sorted(item.keys()))
    mr = item.get("meta")
    if isinstance(mr, dict):
        print(f"[DEBUG] First {label}.meta keys:", sorted(mr.keys()))
        inner = mr.get("meta")
        if isinstance(inner, dict):
            print(f"[DEBUG] First {label}.meta.meta keys:", sorted(inner.keys()))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--runtime_yaml", default=os.getenv("AIHUB_RETRIEVAL_RUNTIME_YAML", str(DEFAULT_RUNTIME_YAML)))
    p.add_argument("--index_dir", default=None, help="indexes 폴더 override")
    p.add_argument("--show_cfg", action="store_true")
    p.add_argument("--debug_keys", action="store_true", help="첫 TS/TL 결과 dict 키 구조 출력")
    args = p.parse_args()

    ry = Path(args.runtime_yaml)
    if not ry.exists():
        raise SystemExit(f"Runtime YAML not found: {ry}")

    runtime_cfg = _read_yaml(ry)

    # 1) cfg 생성
    cfg = AIHubIndexConfig.default()
    if args.index_dir:
        object.__setattr__(cfg, "index_dir", Path(args.index_dir))

    # 2) runtime yaml -> cfg로 'cfg가 담을 수 있는 것'만 반영
    # (a) TS
    if "kdoc" in runtime_cfg and hasattr(cfg, "topk_coarse_docs"):
        object.__setattr__(cfg, "topk_coarse_docs", int(runtime_cfg["kdoc"]))
    if "kts_pool" in runtime_cfg and hasattr(cfg, "topk_ts_final"):
        object.__setattr__(cfg, "topk_ts_final", int(runtime_cfg["kts_pool"]))
    if "rrf_k" in runtime_cfg and hasattr(cfg, "rrf_k"):
        object.__setattr__(cfg, "rrf_k", int(runtime_cfg["rrf_k"]))

    # (b) TL 후보 풀(매우 중요)
    # quota(tl_quota=20)를 쓰려면 최소 20~30 후보가 있어야 함
    tl_quota = int(runtime_cfg.get("tl_quota", 0) or 0)
    kdoc = int(runtime_cfg.get("kdoc", getattr(cfg, "topk_coarse_docs", 30)) or 30)
    out_k = int(runtime_cfg.get("out_k", 0) or 0)

    tl_pool = max(
        int(getattr(cfg, "topk_tl_final", 5)),
        tl_quota,
        out_k,
        kdoc,  # 보통 kdoc(30) 정도로 맞추는 게 안전
    )
    if hasattr(cfg, "topk_tl_final"):
        object.__setattr__(cfg, "topk_tl_final", int(tl_pool))

    # (c) weight (옵션)
    if "weight_tl" in runtime_cfg and hasattr(cfg, "weight_tl"):
        object.__setattr__(cfg, "weight_tl", float(runtime_cfg["weight_tl"]))
    if "weight_ts" in runtime_cfg and hasattr(cfg, "weight_ts"):
        object.__setattr__(cfg, "weight_ts", float(runtime_cfg["weight_ts"]))

    # 3) retriever build
    r = AIHubFusedRetriever.build(cfg)

    if args.show_cfg:
        print("\n==== RUNTIME YAML (what you intended) ====")
        keys = [
            "ts_index","tl_index","backend","kdoc","kts_pool","rrf_k",
            "fusion_mode","out_k","tl_quota","ts_quota","quota_strategy","no_self_hit_tl"
        ]
        for k in keys:
            if k in runtime_cfg:
                print(f"{k:16s}: {runtime_cfg[k]}")
        print("runtime_yaml     :", str(ry))
        print("=========================================\n")

        print("==== FINAL CFG (what cfg can carry) ====")
        for k in ["index_dir","topk_coarse_docs","topk_ts_final","topk_tl_final","rrf_k","weight_ts","weight_tl"]:
            if hasattr(cfg, k):
                print(f"{k:16s}: {getattr(cfg, k)}")
        print("========================================\n")

    out = r.retrieve_fused(args.query)

    ts_list = out.get("ts_context", []) or []
    tl_list = out.get("tl_hints", []) or []

    if args.debug_keys:
        if ts_list:
            _debug_print_first("TS", ts_list[0])
        if tl_list:
            _debug_print_first("TL", tl_list[0])

    print("\n=== TS CONTEXT (evidence) ===")
    for i, x in enumerate(ts_list, 1):
        if not isinstance(x, dict):
            continue
        score = float(x.get("score", 0.0))
        meta_row = x.get("meta") or {}
        meta = _unwrap_meta(meta_row)  # ✅ 핵심

        # row-level id/lid도 같이 출력하면 중복 여부도 바로 보임
        lid = None
        if isinstance(meta_row, dict):
            lid = meta_row.get("lid")

        doc_id = x.get("doc_id") or meta.get("doc_id")
        chunk_id = x.get("chunk_id") or meta.get("chunk_id")
        origin = x.get("origin_path") or meta.get("origin_path")
        title = x.get("title") or meta.get("title")

        print(f"[TS#{i}] score={score:.4f} lid={lid} doc_id={doc_id} chunk_id={chunk_id}")
        if title:
            print("  title:", str(title)[:140])
        if origin:
            print("  origin_path:", str(origin)[:180])

        txt = (x.get("text") or meta.get("text") or "")[:240].replace("\n", " ")
        print(" ", txt, "...\n")

    print("\n=== TL HINTS (qa patterns) ===")
    for i, x in enumerate(tl_list, 1):
        if not isinstance(x, dict):
            continue
        score = float(x.get("score", 0.0))
        meta_row = x.get("meta") or {}
        meta = _unwrap_meta(meta_row)  # ✅ 핵심

        lid = None
        if isinstance(meta_row, dict):
            lid = meta_row.get("lid")

        qa_id = x.get("qa_id") or meta.get("qa_id")
        domain_id = x.get("domain") or x.get("domain_id") or meta.get("domain_id")
        q_type = x.get("q_type") or meta.get("q_type")
        origin = x.get("origin_path") or meta.get("origin_path")

        print(f"[TL#{i}] score={score:.4f} lid={lid} qa_id={qa_id} domain_id={domain_id} q_type={q_type}")
        if origin:
            print("  origin_path:", str(origin)[:180])

        q = (x.get("question") or meta.get("question") or "")[:200].replace("\n", " ")
        if q:
            print(" ", q, "...\n")
        else:
            print("  (no question text in meta)\n")


if __name__ == "__main__":
    main()
