"""
노드 2: 메모리 저장
"""

from agent.state import AgentState
from memory.profile_store import ProfileStore
from memory.hierarchical_memory import HierarchicalMemorySystem
from core.utils import is_llm_mode


def store_memory_node(state: AgentState) -> AgentState:
    """
    메모리 저장 노드
    
    추출된 슬롯을 프로필 저장소에 저장합니다.
    LLM 모드에서는 건너뜁니다.
    """
    print("[Node] store_memory")
    
    # LLM 모드: 메모리 저장 건너뛰기
    if is_llm_mode(state):
        return {
            **state,
            'profile_summary': ''
        }
    
    feature_flags = state.get('feature_flags', {})
    profile_update_enabled = feature_flags.get('profile_update_enabled', True)
    temporal_weight_enabled = feature_flags.get('temporal_weight_enabled', True)
    memory_mode = feature_flags.get('memory_mode', 'structured')

    if memory_mode == 'none' or not profile_update_enabled:
        # 메모리 저장 없이 진행 (ablation 모드)
        return {
            **state,
            'profile_summary': ''
        }
    
    # 프로필 저장소 초기화 (첫 실행 시만)
    if 'profile_store' not in state:
        profile_store = ProfileStore()
        state['profile_store'] = profile_store
    else:
        profile_store = state['profile_store']
    
    # 슬롯 업데이트
    profile_store.update_slots(state['slot_out'])
    if temporal_weight_enabled:
        profile_store.apply_temporal_weights()
    
    # 프로필 요약 생성
    profile_summary = profile_store.get_profile_summary()

    # Hierarchical Memory 통합 (선택적)
    hierarchical_memory_enabled = feature_flags.get('hierarchical_memory_enabled', False)
    hierarchical_memory_stats = {}

    if hierarchical_memory_enabled:
        print("[Hierarchical Memory] Updating memory tiers...")

        try:
            # Hierarchical Memory 초기화 (첫 실행 시만)
            if 'hierarchical_memory' not in state:
                # LLM client와 MedCAT adapter 가져오기
                llm_client = state.get('llm_client')
                medcat_adapter = state.get('medcat_adapter')

                working_capacity = feature_flags.get('working_memory_capacity', 5)
                compression_threshold = feature_flags.get('compression_threshold', 5)

                hierarchical_memory = HierarchicalMemorySystem(
                    user_id=state.get('user_id', 'default_patient'),
                    llm_client=llm_client,
                    medcat_adapter=medcat_adapter,
                    feature_flags=feature_flags,
                    working_capacity=working_capacity,
                    compression_threshold=compression_threshold
                )
                state['hierarchical_memory'] = hierarchical_memory
            else:
                hierarchical_memory = state['hierarchical_memory']

            # 현재 턴 추가
            user_query = state.get('user_text', '')
            agent_response = state.get('answer', '')
            extracted_slots = state.get('slot_out', {})

            hierarchical_memory.add_turn(
                user_query=user_query,
                agent_response=agent_response,
                extracted_slots=extracted_slots
            )

            # 통계 수집
            hierarchical_memory_stats = {
                'total_turns': hierarchical_memory.total_turns,
                'working_memory_size': len(hierarchical_memory.working_memory),
                'compressed_memory_count': len(hierarchical_memory.compressing_memory),
                'chronic_conditions_count': len(hierarchical_memory.semantic_memory.chronic_conditions),
                'chronic_medications_count': len(hierarchical_memory.semantic_memory.chronic_medications),
                'allergies_count': len(hierarchical_memory.semantic_memory.allergies)
            }

            print(f"[Hierarchical Memory] Turn {hierarchical_memory.total_turns} added")
            print(f"  Working Memory: {len(hierarchical_memory.working_memory)} turns")
            print(f"  Compressed Memory: {len(hierarchical_memory.compressing_memory)} summaries")
            print(f"  Semantic Memory: {len(hierarchical_memory.semantic_memory.chronic_conditions)} conditions, "
                  f"{len(hierarchical_memory.semantic_memory.chronic_medications)} medications")

        except Exception as e:
            print(f"[ERROR] Hierarchical Memory update failed: {e}")
            import traceback
            traceback.print_exc()
            hierarchical_memory_stats = {'error': str(e)}

    result_state = {
        **state,
        'profile_summary': profile_summary,
        'profile_store': profile_store,
    }

    # Hierarchical Memory 상태 추가
    if hierarchical_memory_enabled and hierarchical_memory_stats:
        result_state['hierarchical_memory_stats'] = hierarchical_memory_stats

    return result_state

