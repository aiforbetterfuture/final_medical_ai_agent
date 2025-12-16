"""
유틸리티 함수
- 공통 헬퍼 함수
- 코드 중복 제거
"""

from agent.state import AgentState


def is_llm_mode(state: AgentState) -> bool:
    """
    LLM 모드 여부 확인
    
    Args:
        state: AgentState
    
    Returns:
        LLM 모드이면 True
    """
    return state.get('mode') == 'llm'


