"""
Basic RAG Strategy (Baseline)

CRAG와 비교를 위한 기본 RAG 구현
- 품질 평가 없음 (또는 최소한의 평가)
- 재검색 없음
- 단순 통과 (Pass-through)
"""

from typing import Dict, Any
from agent.state import AgentState
from agent.refine_strategies.base_strategy import BaseRefineStrategy


class BasicRAGStrategy(BaseRefineStrategy):
    """
    Basic RAG 전략 (Baseline)

    특징:
    - 품질 평가 없음 (강제 통과)
    - 재검색 없음 (항상 종료)
    - 최소한의 로직 (단순 RAG)

    목적:
    - CRAG와의 성능 비교를 위한 Baseline
    - Ablation Study의 대조군
    """

    def get_strategy_name(self) -> str:
        return 'basic_rag'

    def refine(self, state: AgentState) -> Dict[str, Any]:
        """
        Basic RAG: 품질 평가 없이 통과

        단순히 다음 정보만 기록:
        - quality_score: 1.0 (강제 만점)
        - needs_retrieval: False (재검색 없음)
        """
        print(f"[{self.get_strategy_name().upper()}] Basic RAG - 품질 평가 스킵, 통과")

        answer = state.get('answer', '')
        retrieved_docs = state.get('retrieved_docs', [])
        iteration_count = state.get('iteration_count', 0)

        # 강제 통과: 품질 점수 1.0
        quality_score = 1.0
        needs_retrieval = False

        # 최소한의 품질 피드백 (분석용)
        quality_feedback = {
            'overall_score': quality_score,
            'grounding_score': 1.0 if len(retrieved_docs) > 0 else 0.0,
            'completeness_score': 1.0,
            'accuracy_score': 1.0,
            'missing_info': [],
            'improvement_suggestions': [],
            'needs_retrieval': False,
            'reason': 'Basic RAG (no evaluation)'
        }

        # 이력 추적 (비교 분석용)
        quality_score_history = state.get('quality_score_history') or []
        quality_score_history.append(quality_score)

        query_rewrite_history = state.get('query_rewrite_history') or []
        query_rewrite_history.append(state.get('user_text', ''))  # 원본 쿼리 유지

        refine_iteration_logs = state.get('refine_iteration_logs') or []
        refine_iteration_logs.append({
            'iteration': iteration_count + 1,
            'strategy': self.get_strategy_name(),
            'quality_score': quality_score,
            'quality_feedback': quality_feedback,
            'needs_retrieval': needs_retrieval,
            'rewritten_query': state.get('user_text', ''),
            'num_docs': len(retrieved_docs)
        })

        print(f"[{self.get_strategy_name().upper()}] 품질 점수: {quality_score:.2f} (강제 통과)")

        return {
            'quality_score': quality_score,
            'quality_feedback': quality_feedback,
            'needs_retrieval': needs_retrieval,
            'query_for_retrieval': state.get('user_text', ''),  # 원본 유지
            'quality_score_history': quality_score_history,
            'query_rewrite_history': query_rewrite_history,
            'refine_iteration_logs': refine_iteration_logs,
            'refine_strategy': self.get_strategy_name(),
        }

    def should_retrieve(self, state: AgentState) -> bool:
        """
        Basic RAG: 항상 재검색 불필요 (종료)

        Returns:
            False (항상 종료)
        """
        print(f"[{self.get_strategy_name().upper()}] 재검색 없음 (Basic RAG)")
        return False

    def get_metrics(self, state: AgentState) -> Dict[str, Any]:
        """Basic RAG 메트릭"""
        base_metrics = super().get_metrics(state)

        basic_rag_metrics = {
            'llm_evaluations': 0,  # LLM 평가 없음
            'query_rewrites': 0,  # 쿼리 재작성 없음
            'quality_improvements': 0.0,  # 품질 개선 없음
            'duplicate_detections': 0,  # 중복 검사 없음
            'early_terminations': 0,  # 조기 종료 없음
        }

        return {**base_metrics, **basic_rag_metrics}
