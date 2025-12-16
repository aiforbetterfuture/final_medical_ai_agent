# Ablation 실행 가이드 (CRAG + LangGraph, Context Engineering)

## 1. 이번에 추가된 주요 토글 (feature_flags)
- 루프/품질
  - `self_refine_enabled`: True/False (Self-Refine 루프 on/off)
  - `max_refine_iterations`: 기본 2
  - `quality_threshold`: 기본 0.5
- 컨텍스트/프롬프트
  - `use_context_manager`: True/False
  - `include_history`, `include_profile`, `include_longterm`: True/False
  - `include_evidence`, `include_personalization`: True/False
- 검색
  - `retrieval_mode`: 'hybrid' | 'bm25' | 'faiss'
  - `budget_aware_retrieval`: True/False
  - `avg_doc_tokens`: 기본 200 (예산 기반 k 계산 시 사용)
  - `top_k_override`: 기본 None → 설정 시 k 강제
- 메모리
  - `profile_update_enabled`: True/False
  - `temporal_weight_enabled`: True/False
  - `memory_mode`: 'structured' | 'none'

## 2. 실행 예시 (Streamlit / 코드)
```python
from agent.graph import run_agent

feature_overrides = {
    "self_refine_enabled": False,
    "budget_aware_retrieval": True,
    "retrieval_mode": "bm25",
    "use_context_manager": True,
    "include_history": True,
    "include_profile": True,
    "include_longterm": False,
    "quality_threshold": 0.6,
    "max_refine_iterations": 1,
}

answer = run_agent(
    user_text="65세 남성, 당뇨병 관리 조언",
    mode="ai_agent",
    conversation_history="",
    feature_overrides=feature_overrides,
    return_state=False,
)
```

## 3. 실험 자동화 가이드 (제안)
1) 공통 러너 스크립트 작성 (`scripts/run_ablation.py` 예시)
   - 입력: 실험 설정 리스트 (name, feature_overrides, dataset_path)
   - 처리: 각 설정별 `run_agent` 실행 → 토큰/latency/품질(RAGAS 등) 수집
   - 출력: CSV/MD 요약
2) 추천 실험 축
   - 루프: `self_refine_enabled` on/off, `max_refine_iterations` 0/1/2
   - 검색: `retrieval_mode` hybrid/bm25/faiss, `budget_aware_retrieval` on/off, `avg_doc_tokens` 150/200/300
   - 컨텍스트: `use_context_manager` on/off, `include_history/profile/longterm` on/off
   - 프롬프트: `include_personalization/evidence` on/off
   - 메모리: `profile_update_enabled` on/off
3) 지표
   - 정량: 총 토큰, 검색 문서 토큰, 응답 토큰, 평균 latency, 오류율
   - 품질: RAGAS answer_relevance/faithfulness (가능 시), 간단 BLEU/유사도 지표
4) 결과 정리
   - CSV: run_name, flags..., tokens_total, tokens_docs, latency, quality_score
   - MD: 상위/하위 설정 3개, 개선율 요약

## 4. 기대 효과 요약 (정량/정성)
- 토큰/비용: 컨텍스트 예산 + 예산 기반 검색 → 프롬프트 길이 20~30% 절감, 비용·지연 감소
- 품질/안전: Self-Refine on, evidence/personalization on → faithfulness·개인화 체감 향상
- 통제성: 노드/상태/feature_flags로 구성 요소별 on/off가 쉬워, ablation/AB 테스트가 재현성 있게 수행됨
- 연구 기여: CRAG+LangGraph 이중 순환 구조에서 컨텍스트/검색/루프/메모리 요소별 기여도를 분리해 실험 가능

