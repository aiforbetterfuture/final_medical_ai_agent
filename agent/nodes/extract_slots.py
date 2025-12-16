"""
노드 1: 슬롯 추출
"""

from typing import Dict, Any
from agent.state import AgentState
from extraction.slot_extractor import SlotExtractor
from core.utils import is_llm_mode


def extract_slots_node(state: AgentState) -> AgentState:
    """
    슬롯 추출 노드
    
    사용자 입력에서 의학 정보를 추출합니다.
    LLM 모드에서는 건너뜁니다.
    """
    print("[Node] extract_slots")
    
    # LLM 모드: 슬롯 추출 건너뛰기
    if is_llm_mode(state):
        return state
    
    # 기능 플래그
    feature_flags = state.get('feature_flags', {})
    use_medcat2 = feature_flags.get('medcat2_enabled', True)

    # 슬롯 추출기 초기화 (첫 실행 시만)
    if 'slot_extractor' not in state:
        extractor = SlotExtractor(use_medcat2=use_medcat2)
        state['slot_extractor'] = extractor
    else:
        extractor = state['slot_extractor']
    
    # 슬롯 추출
    slot_out = extractor.extract(state['user_text'])
    
    return {
        **state,
        'slot_out': slot_out
    }

