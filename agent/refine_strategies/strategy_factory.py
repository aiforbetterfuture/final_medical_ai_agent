"""
Strategy Factory

feature_flags에 따라 적절한 Refine 전략을 생성하는 팩토리 클래스
"""

from typing import Dict, Any
from agent.refine_strategies.base_strategy import BaseRefineStrategy
from agent.refine_strategies.corrective_rag_strategy import CorrectiveRAGStrategy
from agent.refine_strategies.basic_rag_strategy import BasicRAGStrategy


class RefineStrategyFactory:
    """
    Refine 전략 팩토리

    feature_flags['refine_strategy']에 따라 전략 인스턴스 생성
    """

    # 전략 레지스트리
    _strategies = {
        'corrective_rag': CorrectiveRAGStrategy,
        'crag': CorrectiveRAGStrategy,  # 별칭
        'basic_rag': BasicRAGStrategy,
        'baseline': BasicRAGStrategy,  # 별칭
    }

    @classmethod
    def create(cls, feature_flags: Dict[str, Any]) -> BaseRefineStrategy:
        """
        전략 인스턴스 생성

        Args:
            feature_flags: 실험 설정

        Returns:
            BaseRefineStrategy 인스턴스

        Raises:
            ValueError: 알 수 없는 전략 이름
        """
        strategy_name = feature_flags.get('refine_strategy', 'corrective_rag').lower()

        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Unknown refine strategy: '{strategy_name}'. "
                f"Available: {list(cls._strategies.keys())}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(feature_flags=feature_flags)

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        새로운 전략 등록 (확장용)

        Args:
            name: 전략 이름
            strategy_class: BaseRefineStrategy 서브클래스
        """
        if not issubclass(strategy_class, BaseRefineStrategy):
            raise TypeError(f"{strategy_class} must inherit from BaseRefineStrategy")

        cls._strategies[name.lower()] = strategy_class
        print(f"[Factory] Registered strategy: '{name}' -> {strategy_class.__name__}")

    @classmethod
    def list_strategies(cls) -> list:
        """
        등록된 전략 목록 반환

        Returns:
            전략 이름 리스트
        """
        return list(cls._strategies.keys())
