"""
노드 6: Self-Refine (Strategy Pattern 기반)

기존 CRAG 로직을 유지하면서, Ablation 연구를 위해 Basic RAG로 전환 가능
- Strategy Pattern: CorrectiveRAGStrategy, BasicRAGStrategy
- feature_flags['refine_strategy']로 전환
- 기본값: 'corrective_rag' (CRAG)
"""

from agent.state import AgentState
from core.utils import is_llm_mode
from agent.refine_strategies import RefineStrategyFactory


def refine_node(state: AgentState) -> AgentState:
    """
    Self-Refine 노드 (Strategy 패턴 기반)

    feature_flags에 따라 다른 전략 사용:
    - 'corrective_rag': CRAG (기본값) - LLM 평가, 동적 재작성, 조건부 재검색
    - 'basic_rag': Basic RAG (Baseline) - 품질 평가 없음, 재검색 없음

    이를 통해 CRAG의 효과를 정량적으로 비교 가능
    """
    print("[Node] refine (Strategy-based)")

    feature_flags = state.get('feature_flags', {})
    self_refine_enabled = feature_flags.get('self_refine_enabled', True)

    # LLM 모드 또는 셀프 리파인 비활성화: 품질 검증 건너뛰기
    if is_llm_mode(state) or not self_refine_enabled:
        return {
            **state,
            'quality_score': 1.0,
            'needs_retrieval': False,
            'refine_strategy': 'disabled'
        }

    # Strategy 생성 (팩토리 패턴)
    try:
        strategy = RefineStrategyFactory.create(feature_flags)
        print(f"[Refine] 전략 선택: {strategy.get_strategy_name()}")
    except ValueError as e:
        print(f"[ERROR] {e}, 기본값(corrective_rag) 사용")
        feature_flags['refine_strategy'] = 'corrective_rag'
        strategy = RefineStrategyFactory.create(feature_flags)

    # 전략 실행
    result = strategy.refine(state)

    # 메트릭 수집 (성능 비교용)
    metrics = strategy.get_metrics(state)
    result['refine_metrics'] = metrics

    # 상태 업데이트
    return {
        **state,
        **result
    }
