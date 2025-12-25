"""
LangGraph 워크플로우 정의
- 7개 노드로 구성된 최소 그래프
- 순환 구조 (Self-Refine 루프)
"""

import asyncio
import inspect
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.extract_slots import extract_slots_node
from agent.nodes.store_memory import store_memory_node
from agent.nodes.assemble_context import assemble_context_node
from agent.nodes.retrieve import retrieve_node

def _retrieve_node_sync(state: AgentState):
    """LangGraph sync node wrapper.

    In some environments retrieve_node is implemented as an async-only Runnable (has ainvoke but no invoke).
    LangGraph app.invoke() requires a synchronous callable, so we adapt here.
    """
    rn = retrieve_node
    # Plain function
    if inspect.iscoroutinefunction(rn):
        return asyncio.run(rn(state))
    if callable(rn) and not hasattr(rn, "ainvoke") and not hasattr(rn, "invoke"):
        return rn(state)

    # LangChain Runnable-like
    if hasattr(rn, "invoke"):
        return rn.invoke(state)

    if hasattr(rn, "ainvoke"):
        return asyncio.run(rn.ainvoke(state))

    # Fallback: try calling
    return rn(state)


def _assemble_context_node_sync(state: AgentState):
    """Ensure assemble_context works when the graph is invoked synchronously."""
    node = assemble_context_node
    if inspect.iscoroutinefunction(node):
        return asyncio.run(node(state))
    if hasattr(node, "invoke"):
        return node.invoke(state)
    if hasattr(node, "ainvoke"):
        return asyncio.run(node.ainvoke(state))
    return node(state)

from agent.nodes.generate_answer import generate_answer_node
from agent.nodes.refine import refine_node
from agent.nodes.quality_check import quality_check_node
from agent.nodes.check_similarity import check_similarity_node, store_response_node
from agent.nodes.classify_intent import classify_intent_node
from core.config import get_agent_config

# 그래프 캐시 (성능 최적화)
_agent_graph_cache = None


def build_agent_graph():
    """
    Agent 그래프 빌드

    Returns:
        컴파일된 LangGraph
    """
    # StateGraph 생성
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("check_similarity", check_similarity_node)
    workflow.add_node("classify_intent", classify_intent_node)  # Active Retrieval
    workflow.add_node("extract_slots", extract_slots_node)
    workflow.add_node("store_memory", store_memory_node)
    workflow.add_node("assemble_context", _assemble_context_node_sync)
    workflow.add_node("retrieve", _retrieve_node_sync)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("quality_check", quality_check_node)
    workflow.add_node("store_response", store_response_node)

    # 엣지 추가 - 캐시 확인이 첫 번째
    workflow.set_entry_point("check_similarity")

    # 조건부 엣지 - 캐시 히트 시 바로 종료
    workflow.add_conditional_edges(
        "check_similarity",
        lambda x: "store_response" if x.get('skip_pipeline', False) else "classify_intent",
        {
            "store_response": "store_response",  # 캐시 히트 - 바로 종료
            "classify_intent": "classify_intent"  # 캐시 미스 - 의도 분류
        }
    )

    # Active Retrieval 조건부 엣지
    def _active_retrieval_router(state: AgentState) -> str:
        """
        Active Retrieval 라우팅 로직

        - needs_retrieval=False: 검색 스킵, 바로 컨텍스트 조립 후 생성
        - needs_retrieval=True: 정상 플로우 (슬롯 추출 → 검색)
        """
        # Feature flag 체크
        feature_flags = state.get('feature_flags', {})
        active_retrieval_enabled = feature_flags.get('active_retrieval_enabled', False)

        if not active_retrieval_enabled:
            # 비활성화 시 기존 플로우
            return "extract_slots"

        # 검색 필요성 판단
        needs_retrieval = state.get('needs_retrieval', True)

        if needs_retrieval:
            return "extract_slots"  # 정상 플로우
        else:
            return "assemble_context"  # 검색 스킵

    workflow.add_conditional_edges(
        "classify_intent",
        _active_retrieval_router,
        {
            "extract_slots": "extract_slots",      # 검색 필요 - 정상 플로우
            "assemble_context": "assemble_context"  # 검색 불필요 - 스킵
        }
    )

    workflow.add_edge("extract_slots", "store_memory")
    workflow.add_edge("store_memory", "assemble_context")

    # assemble_context 후 조건부 라우팅
    def _retrieval_router(state: AgentState) -> str:
        """
        검색 필요성에 따른 라우팅

        - needs_retrieval=False이고 이미 assemble_context를 거쳤으면 바로 생성
        - 재검색 루프에서 이미 검색을 마친 경우 (iteration_count > 0 and retrieved_docs 있음) 바로 생성
        - 첫 번째 검색 후 문서가 있으면 바로 생성 (무한 루프 방지)
        - 그 외에는 retrieve 실행
        """
        needs_retrieval = state.get('needs_retrieval', True)
        iteration_count = state.get('iteration_count', 0)
        retrieved_docs = state.get('retrieved_docs', [])

        # Check if retrieval has been attempted (flag set by retrieve node)
        retrieval_attempted = state.get('retrieval_attempted', False)

        # 첫 번째 검색을 완료했으면 (문서 유무와 관계없이) 바로 답변 생성
        # 무한 루프 방지: 검색을 시도했으나 문서를 못 찾은 경우에도 진행
        if iteration_count == 0 and retrieval_attempted:
            return "generate_answer"

        # 재검색 루프에서 이미 검색을 마친 경우: 바로 답변 생성
        if iteration_count > 0:
            return "generate_answer"

        # 검색 스킵 조건: Active Retrieval이 검색 불필요 판단
        if not needs_retrieval and state.get('classification_skipped') is False:
            return "generate_answer"
        else:
            return "retrieve"

    workflow.add_conditional_edges(
        "assemble_context",
        _retrieval_router,
        {
            "retrieve": "retrieve",                # 검색 필요
            "generate_answer": "generate_answer"   # 검색 스킵
        }
    )

    # ===== 핵심 수정: Self-Refine 루프에서 재검색 시 assemble_context를 다시 거치도록 =====
    # retrieve → assemble_context (재조립) → generate_answer
    workflow.add_edge("retrieve", "assemble_context")
    # assemble_context는 이미 _retrieval_router에서 라우팅되므로, 검색 후에는 항상 generate_answer로

    workflow.add_edge("generate_answer", "refine")

    # 조건부 엣지 (품질 검사) - 재검색 시 retrieve로 돌아가고, retrieve는 다시 assemble_context로
    workflow.add_conditional_edges(
        "refine",
        quality_check_node,
        {
            "retrieve": "retrieve",  # 재검색 → assemble_context (재조립) → generate_answer
            END: "store_response"  # 응답 캐싱 후 종료
        }
    )

    # 응답 저장 후 종료
    workflow.add_edge("store_response", END)

    # 그래프 컴파일
    app = workflow.compile()

    return app


def get_agent_graph():
    """
    Agent 그래프 가져오기 (캐싱)
    
    Returns:
        컴파일된 LangGraph (재사용)
    """
    global _agent_graph_cache
    if _agent_graph_cache is None:
        _agent_graph_cache = build_agent_graph()
    return _agent_graph_cache


def run_agent(
    user_text: str,
    mode: str = 'ai_agent',
    conversation_history: str = None,
    session_state: dict = None,
    feature_overrides: dict = None,
    return_state: bool = False,
    session_id: str = "session-default",
    user_id: str = "user-anonymous",
) -> str:
    """
    Agent 실행
    
    Args:
        user_text: 사용자 입력
        mode: 'llm' 또는 'ai_agent'
        conversation_history: 대화 이력 (멀티턴 대화용)
    
    Returns:
        생성된 답변
    """
    # 기능 플래그 로드 및 병합 (on/off 실험 지원)
    agent_config = get_agent_config()
    feature_flags = (agent_config.get('features') or {}).copy()
    if feature_overrides:
        feature_flags.update(feature_overrides)

    # 기본값 주입(실험/ablation용 토글)
    feature_flags.setdefault('self_refine_enabled', True)
    feature_flags.setdefault('max_refine_iterations', 2)
    feature_flags.setdefault('quality_threshold', 0.5)
    feature_flags.setdefault('use_context_manager', True)

    # Context Engineering 기반 Self-Refine 강화 설정
    feature_flags.setdefault('llm_based_quality_check', True)  # LLM 기반 품질 평가 (vs 휴리스틱)
    feature_flags.setdefault('dynamic_query_rewrite', True)  # 동적 질의 재작성 (vs 정적)
    feature_flags.setdefault('quality_check_enabled', True)  # Quality Check 노드 활성화
    feature_flags.setdefault('duplicate_detection', True)  # 동일 문서 재검색 방지
    feature_flags.setdefault('progress_monitoring', True)  # 품질 점수 진행도 모니터링
    feature_flags.setdefault('include_history', True)
    feature_flags.setdefault('include_profile', True)
    feature_flags.setdefault('include_longterm', False)
    feature_flags.setdefault('include_evidence', True)
    feature_flags.setdefault('include_personalization', True)
    feature_flags.setdefault('profile_update_enabled', True)
    feature_flags.setdefault('temporal_weight_enabled', True)
    feature_flags.setdefault('retrieval_mode', 'hybrid')  # hybrid/bm25/faiss
    feature_flags.setdefault('budget_aware_retrieval', True)
    feature_flags.setdefault('avg_doc_tokens', 200)
    feature_flags.setdefault('response_cache_enabled', True)  # 캐시 활성화
    feature_flags.setdefault('cache_similarity_threshold', 0.85)  # 85% 유사도 임계값
    feature_flags.setdefault('style_variation_level', 0.3)  # 30% 스타일 변경

    # Active Retrieval 설정 (Ablation study용)
    feature_flags.setdefault('active_retrieval_enabled', False)  # 기본값: 비활성화 (안전)
    feature_flags.setdefault('default_k', 8)  # 기본 k 값
    feature_flags.setdefault('simple_query_k', 3)  # 간단한 쿼리
    feature_flags.setdefault('moderate_query_k', 8)  # 보통 쿼리
    feature_flags.setdefault('complex_query_k', 15)  # 복잡한 쿼리

    # Context Compression 설정 (Ablation study용)
    feature_flags.setdefault('context_compression_enabled', False)  # 기본값: 비활성화 (안전)
    feature_flags.setdefault('compression_strategy', 'extractive')  # extractive/abstractive/hybrid
    feature_flags.setdefault('compression_target_ratio', 0.5)  # 50% 압축 목표

    # Hierarchical Memory 설정 (Ablation study용)
    feature_flags.setdefault('hierarchical_memory_enabled', False)  # 기본값: 비활성화 (안전)
    feature_flags.setdefault('working_memory_capacity', 5)  # Working Memory 용량 (턴 수)
    feature_flags.setdefault('compression_threshold', 5)  # 압축 시작 턴 수

    # 초기 상태 (세션 상태가 있으면 병합)
    initial_state = {
        'user_text': user_text,
        'mode': mode,
        'conversation_history': conversation_history,
        'session_id': session_id,
        'user_id': user_id,
        'context_prompt': '',
        'token_plan': {},
        'session_context': '',
        'longterm_context': '',
        'profile_context': '',
        'slot_out': {},
        'profile_summary': '',
        'retrieved_docs': [],
        'query_vector': [],
        'retrieval_attempted': False,  # Initialize retrieval attempted flag
        'system_prompt': '',
        'user_prompt': '',
        'answer': '',
        'quality_score': 0.0,
        'needs_retrieval': False,
        'iteration_count': 0,
        'feature_flags': feature_flags,
        'active_route': 'default',
        'query_for_retrieval': user_text,
        'agent_config': agent_config,
        'cache_hit': False,
        'cached_response': None,
        'cache_similarity_score': 0.0,
        'skip_pipeline': False,
        'cache_stats': {},
        'dynamic_k': None,
        'query_complexity': None,
        'classification_skipped': None,
        'classification_time_ms': None,
        'classification_error': None,
        'intent_classifier': None,
        'compression_stats': None,
        'context_compressor': None,
        'hierarchical_memory': None,
        'hierarchical_memory_stats': None,
        'hierarchical_contexts': None,

        # Self-Refine 강화 필드 초기화
        'quality_feedback': None,
        'retrieved_docs_history': [],
        'quality_score_history': [],
        'query_rewrite_history': [],
        'refine_iteration_logs': [],
    }

    if session_state:
        initial_state.update(session_state)
    
    # 그래프 실행 (캐싱된 그래프 재사용)
    app = get_agent_graph()
    final_state = app.invoke(initial_state)
    
    if return_state:
        return final_state
    return final_state.get('answer', '')
