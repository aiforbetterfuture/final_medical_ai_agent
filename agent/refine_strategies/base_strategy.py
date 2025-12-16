"""
Base Strategy for Refine Process

추상 인터페이스: 모든 Refine 전략이 구현해야 할 메서드 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from agent.state import AgentState


class BaseRefineStrategy(ABC):
    """
    Refine 전략 추상 클래스

    CRAG, Basic RAG 등 다양한 전략을 플러그인처럼 교체 가능하도록 설계
    """

    def __init__(self, feature_flags: Dict[str, Any]):
        """
        Args:
            feature_flags: 실험 설정 플래그
        """
        self.feature_flags = feature_flags

    @abstractmethod
    def refine(self, state: AgentState) -> Dict[str, Any]:
        """
        품질 평가 및 재검색 필요성 판단

        Args:
            state: 현재 Agent 상태

        Returns:
            업데이트할 상태 딕셔너리
            - quality_score: 품질 점수
            - needs_retrieval: 재검색 필요 여부
            - quality_feedback: 품질 피드백 (선택)
            - 기타 전략 특화 필드
        """
        pass

    @abstractmethod
    def should_retrieve(self, state: AgentState) -> bool:
        """
        재검색 필요성 판단

        Args:
            state: 현재 Agent 상태

        Returns:
            True: 재검색 필요
            False: 재검색 불필요 (종료)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        전략 이름 반환 (로깅/분석용)

        Returns:
            전략 이름 (예: 'corrective_rag', 'basic_rag')
        """
        pass

    def get_metrics(self, state: AgentState) -> Dict[str, Any]:
        """
        성능 메트릭 수집 (공통 인터페이스)

        Args:
            state: 현재 Agent 상태

        Returns:
            메트릭 딕셔너리
        """
        return {
            'strategy': self.get_strategy_name(),
            'quality_score': state.get('quality_score', 0.0),
            'iteration_count': state.get('iteration_count', 0),
            'num_docs': len(state.get('retrieved_docs', [])),
            'answer_length': len(state.get('answer', '')),
        }
