"""
AgentState 정의 (TypedDict)
"""

from typing import TypedDict, Annotated, Dict, Any, List, Optional
from operator import add


class AgentState(TypedDict):
    """
    LangGraph 에이전트 상태 정의
    
    모든 노드는 이 상태를 받아서 업데이트된 상태를 반환합니다.
    """
    # 입력
    user_text: str
    mode: str  # 'llm' 또는 'ai_agent'
    conversation_history: Optional[str]  # 대화 이력 (멀티턴 대화용)
    session_id: str  # 세션 식별자
    user_id: str  # 사용자 식별자
    context_prompt: str  # ContextManager가 조립한 컨텍스트
    token_plan: Dict[str, Any]  # 토큰 예산 정보
    session_context: str  # 토큰 예산 내 포함된 최근 대화
    longterm_context: str  # 장기 컨텍스트 요약 (옵션)
    profile_context: str  # 프로필 요약의 컨텍스트 표현 (옵션)
    
    # 슬롯 추출 결과
    slot_out: Dict[str, Any]
    
    # 메모리
    profile_summary: str
    profile_store: Any  # ProfileStore 인스턴스 (세션 유지용)
    
    # 검색 관련
    retrieved_docs: Annotated[List[Dict[str, Any]], add]
    query_vector: List[float]  # 임베딩 벡터
    retrieval_attempted: bool  # 검색 시도 여부 (무한 루프 방지용)

    # 생성 관련
    system_prompt: str
    user_prompt: str
    answer: str
    
    # Self-Refine 관련
    quality_score: float
    needs_retrieval: bool
    iteration_count: int  # 재검색 횟수

    # 기능 플래그 / 라우팅
    feature_flags: Dict[str, Any]
    active_route: str
    query_for_retrieval: str
    agent_config: Dict[str, Any]

    # 캐시 관련
    cache_hit: bool
    cached_response: Optional[str]
    cache_similarity_score: float
    skip_pipeline: bool
    cache_stats: Dict[str, Any]

    # Active Retrieval 관련 (선택적 - 비활성화 시 None/False)
    dynamic_k: Optional[int]  # 동적으로 결정된 k 값 (None = 기본값 사용)
    query_complexity: Optional[str]  # simple/moderate/complex/default
    classification_skipped: Optional[bool]  # 분류 스킵 여부
    classification_time_ms: Optional[float]  # 분류 소요 시간
    classification_error: Optional[str]  # 분류 에러 메시지
    intent_classifier: Optional[Any]  # IntentClassifier 인스턴스 (캐싱용)

    # Context Compression 관련 (선택적 - 비활성화 시 None/False)
    compression_stats: Optional[Dict[str, Any]]  # 압축 통계 (compression_applied, ratio 등)
    context_compressor: Optional[Any]  # ContextCompressor 인스턴스 (캐싱용)

    # Hierarchical Memory 관련 (선택적 - 비활성화 시 None/False)
    hierarchical_memory: Optional[Any]  # HierarchicalMemorySystem 인스턴스
    hierarchical_memory_stats: Optional[Dict[str, Any]]  # 메모리 통계 (턴 수, 티어 크기 등)
    hierarchical_contexts: Optional[Dict[str, str]]  # 검색된 티어 컨텍스트 (working/compressed/semantic)

    # Self-Refine 강화 (Context Engineering 기반)
    quality_feedback: Optional[Dict[str, Any]]  # LLM 기반 품질 평가 결과 (grounding, completeness, missing_info 등)
    retrieved_docs_history: Optional[List[List[str]]]  # 각 iteration의 문서 해시 이력 (중복 검색 방지)
    quality_score_history: Optional[List[float]]  # iteration별 품질 점수 이력 (진행도 모니터링)
    query_rewrite_history: Optional[List[str]]  # iteration별 질의 재작성 이력
    refine_iteration_logs: Optional[List[Dict[str, Any]]]  # 상세 iteration 로그 (분석용)

