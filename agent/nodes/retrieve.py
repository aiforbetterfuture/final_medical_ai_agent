"""
노드 4: 하이브리드 검색
"""

from agent.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever
from core.llm_client import get_llm_client
from core.config import get_retrieval_config, get_embedding_config
from core.utils import is_llm_mode
from core.config import get_agent_config
from context.token_manager import TokenManager


def _select_route(slot_out: dict, feature_flags: dict) -> str:
    """
    간단한 라우팅 규칙:
    - 약물 언급 시 medication
    - 증상/질환 언급 시 symptom
    - 그 외 default
    """
    if not feature_flags.get('dynamic_rag_routing', True):
        return 'default'

    if slot_out.get('medications'):
        return 'medication'
    if slot_out.get('symptoms') or slot_out.get('conditions'):
        return 'symptom'
    return 'default'


def _rewrite_query(user_text: str, slot_out: dict, profile_summary: str, feature_flags: dict) -> str:
    """슬롯/프로필을 활용한 질의 재작성"""
    if not feature_flags.get('query_rewrite_enabled', True):
        return user_text

    parts = [user_text]
    demo = slot_out.get('demographics', {})
    additions = []
    if demo.get('age'):
        additions.append(f"age: {demo.get('age')}")
    if demo.get('gender'):
        additions.append(f"gender: {demo.get('gender')}")
    if slot_out.get('conditions'):
        additions.append("conditions: " + ", ".join(c.get('name', '') for c in slot_out.get('conditions', []) if c.get('name')))
    if slot_out.get('medications'):
        additions.append("medications: " + ", ".join(m.get('name', '') for m in slot_out.get('medications', []) if m.get('name')))

    if additions:
        parts.append(" | ".join(additions))
    if profile_summary:
        parts.append(f"profile: {profile_summary.replace(chr(10), ' ')}")

    return "\n".join(parts)


def retrieve_node(state: AgentState) -> AgentState:
    """
    검색 노드
    
    하이브리드 검색을 수행합니다.
    LLM 모드에서는 건너뜁니다.
    """
    print("[Node] retrieve")
    
    # LLM 모드: 검색 건너뛰기
    if is_llm_mode(state):
        return {
            **state,
            'retrieved_docs': [],
            'retrieval_attempted': True
        }

    feature_flags = state.get('feature_flags', {})
    
    # 재검색 시 iteration_count 증가 (이미 답변이 생성된 경우에만)
    if state.get('answer', ''):
        state['iteration_count'] = state.get('iteration_count', 0) + 1
    else:
        # 첫 검색인 경우 0으로 초기화
        state['iteration_count'] = 0
    
    # 설정 로드
    retrieval_config = get_retrieval_config()
    embedding_config = get_embedding_config()
    agent_config = state.get('agent_config') or get_agent_config()
    routing_table = (agent_config.get('routing') or {})
    
    # LLM 클라이언트 초기화 (임베딩용)
    if 'llm_client' not in state:
        embedding_model = embedding_config.get('model', 'text-embedding-3-large')
        llm_client = get_llm_client(
            provider=embedding_config.get('provider', 'openai'),
            embedding_model=embedding_model
        )
        state['llm_client'] = llm_client
    else:
        llm_client = state['llm_client']
    
    # 질의 재작성 (개인화 정보 포함)
    profile_summary = state.get('profile_summary', '')
    slot_out = state.get('slot_out', {})
    rewritten_query = _rewrite_query(state['user_text'], slot_out, profile_summary, feature_flags)
    state['query_for_retrieval'] = rewritten_query
    
    # 쿼리 벡터 생성 (3072차원)
    try:
        embedding_model = embedding_config.get('model', 'text-embedding-3-large')
        query_vector = llm_client.embed(rewritten_query, embedding_model=embedding_model)
        state['query_vector'] = query_vector
        print(f"[INFO] 쿼리 벡터 생성 완료: {len(query_vector)}차원 (모델: {embedding_model})")
    except Exception as e:
        print(f"[WARNING] 임베딩 생성 실패: {e}")
        query_vector = None
    
    # 라우팅 규칙 적용
    route = _select_route(slot_out, feature_flags)
    state['active_route'] = route

    retriever_key = f"hybrid_retriever::{route}"
    retriever_cache = state.get('retriever_cache', {})

    if retriever_key in retriever_cache:
        hybrid_retriever = retriever_cache[retriever_key]
    else:
        route_cfg = routing_table.get(route) or routing_table.get('default', {})
        retriever_config = {
            'bm25_corpus_path': route_cfg.get('bm25_corpus_path') or retrieval_config.get('bm25_corpus_path'),
            'faiss_index_path': route_cfg.get('faiss_index_path') or retrieval_config.get('faiss_index_path'),
            'faiss_meta_path': route_cfg.get('faiss_meta_path') or retrieval_config.get('faiss_meta_path'),
            'rrf_k': retrieval_config.get('multi', {}).get('rrf_k', 60)
        }
        hybrid_retriever = HybridRetriever(retriever_config)
        retriever_cache[retriever_key] = hybrid_retriever
        state['retriever_cache'] = retriever_cache
    
    # 토큰 예산 확인 (없으면 기본값 사용)
    token_plan = state.get('token_plan', {})
    docs_budget = token_plan.get('for_docs', 900)

    # Active Retrieval: dynamic_k 우선 사용
    dynamic_k = state.get('dynamic_k')

    if dynamic_k is not None and feature_flags.get('active_retrieval_enabled', False):
        # Active Retrieval이 활성화되고 dynamic_k가 설정된 경우
        print(f"[Active Retrieval] Using dynamic_k={dynamic_k}")

        # 예산 제약 적용 (안전장치)
        avg_doc_tokens = feature_flags.get('avg_doc_tokens', 200)
        max_k_by_budget = max(1, docs_budget // max(1, avg_doc_tokens))
        final_k = min(dynamic_k, max_k_by_budget)

        if final_k < dynamic_k:
            print(f"[Active Retrieval] dynamic_k={dynamic_k} reduced to {final_k} due to budget constraint")
    else:
        # 기존 로직 (Fallback)
        base_k = feature_flags.get(
            'top_k_override',
            retrieval_config.get('multi', {}).get('retrievers', [{}])[0].get('k', 8)
        )

        # 예산 기반 k 계산 (평균 문서 길이 근사치)
        avg_doc_tokens = feature_flags.get('avg_doc_tokens', 200)
        max_k_by_budget = max(1, docs_budget // max(1, avg_doc_tokens))
        final_k = min(base_k, max_k_by_budget) if feature_flags.get('budget_aware_retrieval', True) else base_k

    # 검색 모드에 따라 query/query_vector 선택
    retrieval_mode = feature_flags.get('retrieval_mode', 'hybrid')  # hybrid/bm25/faiss
    query_arg = rewritten_query if retrieval_mode != 'faiss' else ""
    query_vec_arg = query_vector if retrieval_mode != 'bm25' else None
    
    # 검색 실행
    candidate_docs = hybrid_retriever.search(
        query=query_arg,
        query_vector=query_vec_arg,
        k=final_k
    )

    # 예산 내 문서만 선택 (토큰 수가 예산을 넘지 않도록 필터, 옵션)
    if feature_flags.get('budget_aware_retrieval', True):
        token_manager = state.get('token_manager') or TokenManager(max_total_tokens=4000)
        state['token_manager'] = token_manager

        selected_docs = []
        used_tokens = 0
        for doc in candidate_docs:
            doc_text = doc.get('text', '')
            doc_tokens = token_manager.count_tokens(doc_text)
            if used_tokens + doc_tokens <= docs_budget:
                selected_docs.append(doc)
                used_tokens += doc_tokens
            else:
                break
    else:
        selected_docs = candidate_docs
    
    return {
        **state,
        'retrieved_docs': selected_docs,
        'retrieval_attempted': True  # Flag to indicate retrieval has been attempted
    }

