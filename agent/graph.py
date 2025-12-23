"""LangGraph workflow (agent) with aihub fused retrieval defaults.

핵심:
- retrieval_mode 기본값을 aihub_quota로 고정
- runtime yaml 기본 경로: configs/aihub_retrieval_runtime.yaml
- evidence는 TS만, TL은 힌트 전용
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.classify_intent import classify_intent_node
from agent.nodes.extract_slots import extract_slots_node
from agent.nodes.retrieve import retrieve_node
from agent.nodes.assemble_context import assemble_context_node
from agent.nodes.generate_answer import generate_answer_node
from agent.nodes.quality_check import quality_check_node


def build_agent_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent_node)
    g.add_node("extract_slots", extract_slots_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("assemble_context", assemble_context_node)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("quality_check", quality_check_node)

    g.set_entry_point("classify_intent")

    g.add_edge("classify_intent", "extract_slots")
    g.add_edge("extract_slots", "retrieve")
    g.add_edge("retrieve", "assemble_context")
    g.add_edge("assemble_context", "generate_answer")
    g.add_edge("generate_answer", "quality_check")

    # quality_check_node는 "retrieve" 또는 END를 반환하도록 설계되어 있음
    g.add_conditional_edges(
        "quality_check",
        lambda s: quality_check_node(s),
        {"retrieve": "retrieve", END: END},
    )

    # 재검색 후에도 context를 다시 조립하도록
    g.add_edge("retrieve", "assemble_context")

    return g.compile()


_APP = None


def get_agent_graph():
    global _APP
    if _APP is None:
        _APP = build_agent_graph()
    return _APP


def run_agent(
    user_text: str,
    *,
    mode: str = "agent",
    conversation_history: Optional[list] = None,
    session_state: Optional[dict] = None,
    feature_overrides: Optional[dict] = None,
    agent_config: Optional[dict] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    return_state: bool = False,
):
    feature_flags: Dict[str, Any] = {}
    if agent_config and isinstance(agent_config, dict):
        feature_flags.update(agent_config.get("feature_flags", {}) or {})
    if feature_overrides and isinstance(feature_overrides, dict):
        feature_flags.update(feature_overrides)

    # 고정 기본값 (SSOT)
    feature_flags.setdefault("retrieval_mode", "aihub_quota")
    feature_flags.setdefault("aihub_runtime_yaml", "configs/aihub_retrieval_runtime.yaml")
    feature_flags.setdefault("include_tl_hints", True)
    feature_flags.setdefault("tl_hints_k", 5)
    feature_flags.setdefault("quality_check_enabled", True)
    feature_flags.setdefault("llm_based_quality_check", True)
    feature_flags.setdefault("quality_threshold", 0.75)
    feature_flags.setdefault("max_refine_iterations", 2)
    feature_flags.setdefault("include_history", True)
    feature_flags.setdefault("include_profile", True)
    feature_flags.setdefault("include_longterm", False)
    feature_flags.setdefault("include_evidence", True)
    feature_flags.setdefault("include_personalization", True)
    feature_flags.setdefault("max_ts_in_prompt", 5)
    feature_flags.setdefault("max_tl_in_prompt", 5)

    initial_state: Dict[str, Any] = {
        "user_text": user_text,
        "mode": mode,
        "conversation_history": conversation_history or [],
        "session_id": session_id,
        "user_id": user_id,
        "feature_flags": feature_flags,
        "agent_config": agent_config or {},
        "retrieval_attempted": False,
        "retrieved_docs": [],
        "tl_hints": [],
        "fused_ranked": [],
        "fusion_mode": None,
        "runtime_yaml_used": None,
        "system_prompt": "",
        "user_prompt": "",
        "context_prompt": "",
        "answer": "",
        "quality_score": 0.0,
        "needs_retrieval": False,
        "iteration_count": 0,
        "retrieved_docs_history": [],
        "quality_score_history": [],
        "query_rewrite_history": [],
        "refine_iteration_logs": [],
        "query_for_retrieval": user_text,
    }

    if session_state:
        initial_state.update(session_state)

    app = get_agent_graph()
    final_state = app.invoke(initial_state)
    return final_state if return_state else final_state.get("answer", "")
