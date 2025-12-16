"""
노드 7: 품질 검사 및 재검색 결정 (Strategy Pattern 기반)

전략에 따라 다른 재검색 로직 사용:
- CorrectiveRAGStrategy: 안전장치 포함 재검색 판단
- BasicRAGStrategy: 항상 종료 (재검색 없음)
"""

from agent.state import AgentState
from langgraph.graph import END
from core.utils import is_llm_mode
from agent.refine_strategies import RefineStrategyFactory


def quality_check_node(state: AgentState) -> str:
    """
    품질 검사 노드 (Strategy 기반)

    전략별 재검색 로직:
    - CRAG: 품질 점수, 안전장치 기반 조건부 재검색
    - Basic RAG: 항상 종료 (재검색 없음)

    Returns:
        "retrieve": 재검색 수행
        END: 종료
    """
    print("[Node] quality_check (Strategy-based)")

    feature_flags = state.get('feature_flags', {})
    self_refine_enabled = feature_flags.get('self_refine_enabled', True)
    quality_check_enabled = feature_flags.get('quality_check_enabled', True)

    # LLM 모드 또는 품질 체크 비활성화: 항상 종료
    if is_llm_mode(state) or not self_refine_enabled or not quality_check_enabled:
        print("[Quality Check] 품질 체크 비활성 또는 LLM 모드: 종료")
        return END

    # Strategy 생성 (refine_node에서 사용한 것과 동일한 전략)
    try:
        strategy = RefineStrategyFactory.create(feature_flags)
    except ValueError as e:
        print(f"[ERROR] {e}, 기본값(corrective_rag) 사용")
        feature_flags['refine_strategy'] = 'corrective_rag'
        strategy = RefineStrategyFactory.create(feature_flags)

    # 전략별 재검색 판단
    should_retrieve = strategy.should_retrieve(state)

    if should_retrieve:
        iteration_count = state.get('iteration_count', 0)
        quality_score = state.get('quality_score', 0.0)
        print(f"[Quality Check] 재검색 수행 (전략: {strategy.get_strategy_name()}, "
              f"점수: {quality_score:.2f}, iteration: {iteration_count + 1})")
        return "retrieve"
    else:
        quality_score = state.get('quality_score', 0.0)
        print(f"[Quality Check] 종료 (전략: {strategy.get_strategy_name()}, "
              f"점수: {quality_score:.2f})")
        return END
