"""agent.nodes.assemble_context

목표:
- Context Engineering: (profile + history + memory) + (retrieval evidence) + (pattern hints)
- evidence는 TS만 포함하고, TL은 별도 섹션으로 분리해 "근거로 인용하지 않도록" 프롬프트에 강하게 고정
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.prompts import build_system_prompt


def _format_tl_hints_for_prompt(tl_hints: List[Dict[str, Any]], *, k: int = 5) -> str:
    """TL은 '패턴' 용도. 근거로 인용하지 않게 프롬프트에서 명시적으로 분리."""
    if not tl_hints:
        return ""

    lines = []
    for x in tl_hints[:k]:
        q = (x.get("question") or "").replace("\n", " ").strip()
        if not q:
            continue
        domain_id = x.get("domain_id")
        q_type = x.get("q_type")
        src = x.get("source")
        lines.append(f"- [domain_id={domain_id} q_type={q_type} src={src}] {q[:260]}")

    if not lines:
        return ""

    return (
        "\n\n" +
        "[TL HINTS — patterns only, NOT evidence]\n"
        "- 아래 TL은 '유사 문항 패턴' 힌트입니다. \n"
        "- TL 내용을 근거처럼 인용/출처표기/복사하지 마세요. \n"
        "- TL은 답변 구조(항목, 체크리스트)나 추가 확인질문 설계에만 사용하세요.\n"
        + "\n".join(lines)
    )


async def assemble_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    feature_flags = state.get("feature_flags", {}) or {}

    retrieved_docs = state.get("retrieved_docs", []) or []  # ✅ TS evidence only
    # evidence 로그: LLM에 넣은 TS id 기록
    state["prompt_evidence_ts_ids"] = [
        {"doc_id": x.get("doc_id"), "chunk_id": x.get("chunk_id")}
        for x in retrieved_docs
    ]

    system_prompt = build_system_prompt(
        mode=state.get("mode", "agent"),
        session_context=state.get("session_context", ""),
        longterm_context=state.get("longterm_context", ""),
        profile_context=state.get("profile_context", ""),
        retrieved_docs=retrieved_docs,
        feature_flags=feature_flags,
    )

    # ✅ quota/rrf 결과에서 TL 힌트를 별도 섹션으로 넣기 (옵션)
    tl_used = False
    if feature_flags.get("include_tl_hints", True):
        tl_hints = state.get("tl_hints", []) or []
        k = int(feature_flags.get("tl_hints_k", 5))
        system_prompt += _format_tl_hints_for_prompt(tl_hints, k=k)
        tl_used = bool(tl_hints)

    state["system_prompt"] = system_prompt
    state["tl_used_as_hint"] = tl_used

    # user prompt는 기존 흐름 유지
    state["user_prompt"] = state.get("user_text", "")
    state["context_prompt"] = system_prompt
    return state
