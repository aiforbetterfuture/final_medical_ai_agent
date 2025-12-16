"""
노드 3: 컨텍스트 조립
"""

from agent.state import AgentState
from core.prompts import build_system_prompt, format_user_prompt
from context.token_manager import TokenManager
from context.context_manager import ContextManager
from context.context_compressor import ContextCompressor

# 컨텍스트/토큰 매니저는 모듈 단위로 1회 초기화
_token_manager = TokenManager(max_total_tokens=4000)
_context_manager = ContextManager(token_manager=_token_manager)
# Context Compressor는 필요시 초기화 (feature flag 기반)


def assemble_context_node(state: AgentState) -> AgentState:
    """
    컨텍스트 조립 노드
    
    - LLM 모드: 간단한 프롬프트
    - AI Agent 모드: 프로필/검색 근거 포함
    - TokenManager/ContextManager로 토큰 예산 내 컨텍스트 구성
    """
    print("[Node] assemble_context")
    
    mode = state.get('mode', 'ai_agent')
    conversation_history = state.get('conversation_history')
    profile_summary = state.get('profile_summary', '')
    feature_flags = state.get('feature_flags', {})
    use_context_manager = feature_flags.get('use_context_manager', True)
    include_history = feature_flags.get('include_history', True)
    include_profile = feature_flags.get('include_profile', True)
    include_longterm = feature_flags.get('include_longterm', False)
    include_evidence = feature_flags.get('include_evidence', True)
    include_personalization = feature_flags.get('include_personalization', True)

    # Initialize variables that may be used later in both modes
    hierarchical_contexts = {}
    compression_stats = {}

    # LLM 모드: 간단한 프롬프트만 사용
    if mode == 'llm':
        system_prompt = build_system_prompt(mode='llm')
        user_prompt = format_user_prompt(
            state['user_text'],
            conversation_history=conversation_history if include_history else None
        )
    else:
        # AI Agent 모드: 검색된 문서 포맷팅
        retrieved_docs = state.get('retrieved_docs', [])

        # Hierarchical Memory 컨텍스트 가져오기 (선택적)
        hierarchical_memory_enabled = feature_flags.get('hierarchical_memory_enabled', False)

        if hierarchical_memory_enabled and 'hierarchical_memory' in state:
            print("[Hierarchical Memory] Retrieving context from 3 tiers...")

            try:
                hierarchical_memory = state['hierarchical_memory']

                # 토큰 예산 확인
                token_plan = state.get('token_plan', {})
                history_budget = token_plan.get('for_history', 300)

                # 3-tier 컨텍스트 검색
                hierarchical_contexts = hierarchical_memory.retrieve_context(
                    query=state['user_text'],
                    max_tokens=history_budget
                )

                # 검색된 컨텍스트 통계
                working_ctx = hierarchical_contexts.get('working_context', '')
                compressed_ctx = hierarchical_contexts.get('compressed_context', '')
                semantic_ctx = hierarchical_contexts.get('semantic_context', '')

                print(f"[Hierarchical Memory] Retrieved contexts:")
                print(f"  Working Memory: {len(working_ctx)} chars")
                print(f"  Compressed Memory: {len(compressed_ctx)} chars")
                print(f"  Semantic Memory: {len(semantic_ctx)} chars")

            except Exception as e:
                print(f"[ERROR] Hierarchical Memory retrieval failed: {e}")
                import traceback
                traceback.print_exc()
                hierarchical_contexts = {}

        # Context Compression 적용 (선택적)
        compression_enabled = feature_flags.get('context_compression_enabled', False)

        if compression_enabled and retrieved_docs:
            print("[Context Compression] Applying compression...")

            # Compressor 초기화 (캐싱)
            if 'context_compressor' not in state:
                # LLM client 가져오기 (abstractive 압축용)
                llm_client = state.get('llm_client')

                compressor = ContextCompressor(
                    token_manager=_token_manager,
                    llm_client=llm_client,
                    feature_flags=feature_flags
                )
                state['context_compressor'] = compressor
            else:
                compressor = state['context_compressor']

            # 토큰 예산 확인
            token_plan = state.get('token_plan', {})
            docs_budget = token_plan.get('for_docs', 900)

            # 압축 실행
            compressed_docs, compression_stats = compressor.compress_docs(
                docs=retrieved_docs,
                query=state['user_text'],
                budget=docs_budget
            )

            # 압축된 문서 사용
            if compression_stats.get('compression_applied'):
                retrieved_docs = compressed_docs
                print(f"[Context Compression] Compressed: {compression_stats.get('original_tokens', 0)} → {compression_stats.get('compressed_tokens', 0)} tokens")
            else:
                print(f"[Context Compression] Skipped: {compression_stats.get('reason', 'unknown')}")

        # 문서 포맷팅
        docs_text = ""
        if retrieved_docs:
            docs_text = "\n\n".join([
                f"[문서 {i+1}]\n{doc.get('text', '')}"
                for i, doc in enumerate(retrieved_docs[:5])
            ])
        
        # 시스템 프롬프트 생성
        system_prompt = build_system_prompt(
            mode='ai_agent',
            include_personalization=include_personalization and include_profile and bool(profile_summary),
            profile_summary=profile_summary if include_profile else '',
            include_evidence=include_evidence and len(retrieved_docs) > 0,
            retrieved_docs=docs_text
        )
        
        # 사용자 프롬프트 생성 (대화 이력 포함)
        # Hierarchical Memory가 활성화된 경우 티어 컨텍스트 사용
        if hierarchical_memory_enabled and hierarchical_contexts:
            # 3-tier 컨텍스트를 하나의 문자열로 결합
            tier_contexts = []

            working_ctx = hierarchical_contexts.get('working_context', '')
            if working_ctx:
                tier_contexts.append(f"[Recent Dialogue]\n{working_ctx}")

            compressed_ctx = hierarchical_contexts.get('compressed_context', '')
            if compressed_ctx:
                tier_contexts.append(f"[Previous Context Summary]\n{compressed_ctx}")

            semantic_ctx = hierarchical_contexts.get('semantic_context', '')
            if semantic_ctx:
                tier_contexts.append(f"[Long-term Medical Information]\n{semantic_ctx}")

            hierarchical_history = "\n\n".join(tier_contexts) if tier_contexts else ""

            user_prompt = format_user_prompt(
                state['user_text'],
                conversation_history=hierarchical_history if include_history else None
            )
        else:
            user_prompt = format_user_prompt(
                state['user_text'],
                conversation_history=conversation_history if include_history else None
            )
    
    # 토큰 예산 내에서 컨텍스트 조합 (옵션)
    if use_context_manager:
        context_result = _context_manager.build_context(
            user_id=state.get('user_id', 'anonymous'),
            session_id=state.get('session_id', 'session-default'),
            current_query=state['user_text'],
            conversation_history=conversation_history if include_history else "",
            profile_summary=profile_summary if include_profile else "",
            longterm_summary=state.get('longterm_context', '') if include_longterm else "",
            max_tokens=4000,
        )
        context_prompt = context_result.get('context_prompt', '')
        token_plan = context_result.get('token_plan', {})
        session_context = context_result.get('session_context', '')
        profile_context = context_result.get('profile_context', '')
        longterm_context = context_result.get('longterm_context', '')
    else:
        context_prompt = ""
        token_plan = {}
        session_context = conversation_history if include_history else ""
        profile_context = profile_summary if include_profile else ""
        longterm_context = state.get('longterm_context', '') if include_longterm else ""
    
    # 반환 상태 구성
    result_state = {
        **state,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'context_prompt': context_prompt,
        'token_plan': token_plan,
        'session_context': session_context,
        'profile_context': profile_context,
        'longterm_context': longterm_context,
    }

    # Context Compression 메트릭 추가
    if compression_stats:
        result_state['compression_stats'] = compression_stats

    # Hierarchical Memory 컨텍스트 추가
    if hierarchical_contexts:
        result_state['hierarchical_contexts'] = hierarchical_contexts

    return result_state

