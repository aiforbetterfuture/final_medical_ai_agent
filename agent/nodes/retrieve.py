"""agent.nodes.retrieve

SSOT 원칙:
- indexes, quota, out_k 등의 '실제 동작 값'은 configs/aihub_retrieval_runtime.yaml 한 곳에서만 결정
- agent에서는 runtime.yaml을 읽어 retriever를 빌드하고, retrieve_fused() 결과를 state에 기록

설계 의도:
- LLM에 넣는 '근거(evidence)'는 TS만 사용 (RAG 근거)
- TL은 '패턴 힌트'로만 사용 (근거로 인용 금지)
- quota 모드일 때는 fused_ranked에서 TS/TL을 분리해 "실제로 선택된" 항목만 사용
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root_from_this_file() -> Path:
    # .../final_medical_ai_agent/agent/nodes/retrieve.py
    return Path(__file__).resolve().parents[2]


def _resolve_runtime_yaml(runtime_yaml: Optional[str]) -> Path:
    root = _repo_root_from_this_file()
    if runtime_yaml:
        p = Path(runtime_yaml)
        return p if p.is_absolute() else (root / p)
    return root / "configs" / "aihub_retrieval_runtime.yaml"


@lru_cache(maxsize=2)
def _get_aihub_retriever(runtime_yaml_abs: str):
    # ⚠️ 지연 import: agent 로딩 시 heavy 모델 초기화 방지
    from retrieval.aihub_flat.runtime import build_retriever_from_runtime_yaml

    return build_retriever_from_runtime_yaml(Path(runtime_yaml_abs))


def _is_ts(item: Dict[str, Any]) -> bool:
    src = str(item.get("source", ""))
    return src.startswith("ts_") or src in {"ts_coarse", "ts_fine", "ts"}


def _is_tl(item: Dict[str, Any]) -> bool:
    src = str(item.get("source", ""))
    return src.startswith("tl_") or src in {"tl_stem", "tl_stem_opts", "tl"}


@dataclass
class RetrieveResult:
    retrieved_docs: List[Dict[str, Any]]  # TS evidence actually used
    ts_pool: List[Dict[str, Any]]         # TS pool (debug)
    tl_pool: List[Dict[str, Any]]         # TL pool (debug)
    fused_ranked: List[Dict[str, Any]]    # final fused ranking (TS+TL)
    fusion_mode: str


def _retrieve_aihub_quota(query: str, runtime_yaml: Path, *, domain_id: Optional[int] = None) -> RetrieveResult:
    r = _get_aihub_retriever(str(runtime_yaml))
    out = r.retrieve_fused(query, domain_id=domain_id)

    fused_ranked = list(out.get("fused_ranked", []))
    ts_pool = list(out.get("ts_context", []))
    tl_pool = list(out.get("tl_hints", []))

    # ✅ LLM 근거는 "선택된 TS"만 사용: quota/rrf 결과와 일치
    selected_ts = [x for x in fused_ranked if _is_ts(x)]

    # fallback (혹시 fused_ranked에 TS가 없으면 풀에서 상위 사용)
    if not selected_ts:
        selected_ts = ts_pool[:]

    fusion_mode = "rrf"
    if hasattr(r, "runtime") and isinstance(getattr(r, "runtime"), dict):
        fusion_mode = str(r.runtime.get("fusion_mode", fusion_mode))

    return RetrieveResult(
        retrieved_docs=selected_ts,
        ts_pool=ts_pool,
        tl_pool=tl_pool,
        fused_ranked=fused_ranked,
        fusion_mode=fusion_mode,
    )


async def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: retrieve"""

    feature_flags = state.get("feature_flags", {}) or {}
    retrieval_mode = str(feature_flags.get("retrieval_mode", "aihub_quota"))

    query = state.get("query_for_retrieval") or state.get("user_text") or ""
    if not query.strip():
        state["retrieval_attempted"] = True
        return state

    # domain_id가 있으면 TL 품질(분과 필터)에 도움
    domain_id = state.get("domain_id")

    if retrieval_mode in {"aihub_quota", "aihub_flat", "aihub_runtime"}:
        runtime_yaml = _resolve_runtime_yaml(
            (state.get("agent_config", {}) or {}).get("aihub_runtime_yaml")
            or state.get("aihub_runtime_yaml")
            or feature_flags.get("aihub_runtime_yaml")
        )

        res = _retrieve_aihub_quota(query, runtime_yaml, domain_id=domain_id)

        # ✅ evidence: TS만
        state["retrieved_docs"] = res.retrieved_docs

        # ✅ 힌트/디버그: TL, fused
        state["tl_hints"] = [x for x in res.fused_ranked if _is_tl(x)] or res.tl_pool
        state["fused_ranked"] = res.fused_ranked
        state["fusion_mode"] = res.fusion_mode
        state["ts_context_pool"] = res.ts_pool
        state["tl_hints_pool"] = res.tl_pool
        state["runtime_yaml_used"] = str(runtime_yaml)

    else:
        # 기존 하이브리드 경로 유지 (ablation용)
        from retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        docs = await retriever.retrieve(query, k=feature_flags.get("default_k", 8))
        state["retrieved_docs"] = docs
        state.pop("tl_hints", None)
        state.pop("fused_ranked", None)
        state["fusion_mode"] = "hybrid"

    # self-refine/quality-check 디버깅용
    state.setdefault("retrieved_docs_history", []).append(state.get("retrieved_docs", []))

    state["retrieval_attempted"] = True
    return state
