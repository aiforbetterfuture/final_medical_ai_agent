"""
노드 7: 품질 검사 및 재검색 결정 (Strategy Pattern 기반)

전략에 따라 다른 재검색 로직 사용:
- CorrectiveRAGStrategy: 안전장치 포함 재검색 판단
- BasicRAGStrategy: 항상 종료 (재검색 없음)
"""

# agent/nodes/quality_check.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import END


# ====== 타이트 고정 파라미터(재현성 목적: 상수로 둠) ======
MIN_TS_EVIDENCE_FOR_PASS = 2        # TS 근거 최소 개수
MIN_TS_TEXT_CHARS = 240             # TS 텍스트(합) 최소 길이(너무 빈약하면 재검색)
MAX_QC_RETRIEVE_LOOPS = 1           # QC에서 재검색 루프는 1번만 허용 (무한루프 방지)

# 의료 답변에서 "강한 처방/중단/용량" 류 문구가 나오면 근거가 더 필요함
HIGH_STAKES_KEYWORDS = (
    "용량", "증량", "감량", "중단", "복용을 멈", "복용 중지", "처방", "투여", "인슐린",
    "즉시", "응급", "119", "자살", "흉통", "호흡곤란", "의식", "쇼크", "출혈",
)


def _get_text(d: Dict[str, Any]) -> str:
    t = d.get("text") or ""
    return t if isinstance(t, str) else str(t)


def _is_ts_item(d: Dict[str, Any]) -> bool:
    src = str(d.get("source") or "")
    # ts_coarse/ts_fine/ts_context 등 다양한 표기를 다 허용
    return src.startswith("ts") or src.startswith("TS") or src.startswith("ts_") or src.startswith("TS_")


def _extract_pools(state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    state에서 TS/TL/fused를 최대한 호환적으로 꺼낸다.
    """
    ts_pool = state.get("ts_context") or state.get("retrieved_docs") or []
    tl_pool = state.get("tl_hints") or state.get("retrieved_hints") or []
    fused = state.get("fused_ranked") or []

    # 혹시 retrieval 결과를 통째로 묶어둔 경우까지 커버
    retrieval = state.get("retrieval") or {}
    if isinstance(retrieval, dict):
        ts_pool = ts_pool or retrieval.get("ts_context") or []
        tl_pool = tl_pool or retrieval.get("tl_hints") or []
        fused = fused or retrieval.get("fused_ranked") or retrieval.get("fused") or []

    # 타입 방어
    ts_pool = ts_pool if isinstance(ts_pool, list) else []
    tl_pool = tl_pool if isinstance(tl_pool, list) else []
    fused = fused if isinstance(fused, list) else []
    return ts_pool, tl_pool, fused


def _pick_answer_text(state: Dict[str, Any]) -> str:
    """
    generate/self_refine가 무엇이라는 키로 답변을 저장하든 최대한 찾아본다.
    """
    for k in ("final_answer", "answer", "response", "draft_answer", "model_answer"):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _high_stakes(answer: str) -> bool:
    a = answer.replace(" ", "")
    return any(kw.replace(" ", "") in a for kw in HIGH_STAKES_KEYWORDS)


def quality_check_node(state: Dict[str, Any]) -> Any:
    """
    graph.py에서 conditional router로 직접 호출됨.
    반드시 "retrieve" 또는 END 를 반환해야 함.
    (타이트 고정: 규칙 기반, 재현성 우선)
    """
    ts_pool, tl_pool, fused = _extract_pools(state)
    answer = _pick_answer_text(state)

    # ---- TS 근거량 평가 ----
    ts_count = len(ts_pool)
    ts_chars = sum(len(_get_text(x)) for x in ts_pool[:max(4, MIN_TS_EVIDENCE_FOR_PASS)])

    # fused 안의 TS 개수(참고 지표)
    fused_ts = [x for x in fused if isinstance(x, dict) and _is_ts_item(x)]
    fused_ts_count = len(fused_ts)

    # ---- QC 루프 카운터(상태에 고정 기록) ----
    qc_loops = int(state.get("qc_retrieve_loops") or 0)

    # ---- 판정 로직(타이트) ----
    reasons: List[str] = []

    if not answer:
        reasons.append("empty_answer")

    # 1) TS 근거가 최소 기준 미달이면 재검색(단, 1번만)
    if ts_count < MIN_TS_EVIDENCE_FOR_PASS:
        reasons.append(f"ts_count<{MIN_TS_EVIDENCE_FOR_PASS}({ts_count})")

    # 2) TS 텍스트가 너무 빈약하면 재검색
    if ts_chars < MIN_TS_TEXT_CHARS:
        reasons.append(f"ts_chars<{MIN_TS_TEXT_CHARS}({ts_chars})")

    # 3) high-stakes 문구가 있으면 TS 근거를 더 엄격히 요구
    if answer and _high_stakes(answer) and fused_ts_count < 1:
        reasons.append("high_stakes_without_ts_in_fused")

    need_retrieve = len(reasons) > 0

    # ---- 상태에 기록(디버깅/재현성) ----
    state["qc_ts_count"] = ts_count
    state["qc_ts_chars"] = ts_chars
    state["qc_fused_ts_count"] = fused_ts_count
    state["qc_reasons"] = reasons
    state["qc_pass"] = (not need_retrieve)

    # ---- 라우팅 ----
    if need_retrieve and qc_loops < MAX_QC_RETRIEVE_LOOPS:
        state["qc_retrieve_loops"] = qc_loops + 1
        # iteration_count를 올려두면(있다면) 추후 라우팅/로그에서 “2회차”가 명확해짐
        try:
            state["iteration_count"] = int(state.get("iteration_count") or 0) + 1
        except Exception:
            pass
        return "retrieve"

    return END
