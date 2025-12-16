# Ablation 설계 점검 및 제안 (LangGraph 기반 스캐폴드)

본 문서는 현재 스캐폴드가 LangGraph를 활용해 ablation 연구를 수행하기에 적합한지 점검하고, 추가로 비교·대조해야 할 기능/구성 요소를 정리합니다.

---

## 1. 현재 스캐폴드의 ablation 친화 요소 (LangGraph 장점)
- **StateGraph + Typed AgentState**: 상태 필드가 명시되어 특정 필드/노드의 영향만 제거하거나 변경하기 쉽습니다.
- **노드 단위 분리**: `extract_slots`, `store_memory`, `assemble_context`, `retrieve`, `generate_answer`, `refine`, `quality_check`로 기능이 분리되어, 노드 단위 on/off 및 변형이 용이합니다.
- **조건부 엣지**: `quality_check`에서 재검색 루프 여부를 분기 → 루프 제거/횟수 제한 ablation이 간단합니다.
- **feature_flags/agent_config**: `run_agent(..., feature_overrides=...)`로 라우팅/토글을 주입할 수 있어, 코드 수정 없이 실험 설정을 바꿀 수 있습니다.
- **모듈화된 Retrieval**: `HybridRetriever`(BM25+FAISS+RRF) 한 군데에서 k, rrf_k, 경로 등을 통제 가능 → 검색 전략 ablation이 직관적입니다.
- **컨텍스트 예산화(TokenManager/ContextManager)**: 토큰 예산 적용/미적용을 비교해 성능·비용에 미치는 영향을 실험하기 쉽습니다.

요약: 그래프/상태 기반 구조 덕분에 “기능별로 스위치를 켜고 끄는” ablation이 하드코딩 없이 가능하며, 실험 조건을 분리해 재현성 있게 관리하기 좋습니다.

---

## 2. 하드코딩/통제 곤란 위험 점검
- **quality_check 로직**: 그래프 엣지는 고정되어 있으나, 품질 임계값/반복 횟수는 코드 상수에 가까움. → 임계값/반복 제한을 외부 설정화하면 더 나은 통제 가능.
- **retrieval k 값**: 기존엔 상수형(8) 접근이었으나, 현재 token_plan.for_docs 기반 축소로 개선됨. 평균 문서 길이 가정(200토큰)은 코드 상수이므로 설정/주입이 가능하면 더 유연.
- **feature_flags 기본값**: agent_config나 feature_overrides를 사용하지만 기본값이 명시된 곳이 분산되어 있음. 실험별 설정 테이블이 있으면 통제 쉬움.
- **longterm_context / profile_context**: 필드는 있으나 실제 long-term 저장소(checkpointer/store)는 아직 없음 → 장기 메모리 ablation 시 “off” 대비 “on”이 불충분.

결론: 대부분 토글 가능하나, 임계값/평균길이/기본 k 등 일부 상수는 설정화하면 더 깔끔하게 ablation 제어 가능.

---

## 3. 권장 ablation 축 (비교·대조 대상)
아래 항목들은 독립적으로 on/off, 파라미터 변경을 통해 영향도를 측정하기 좋은 후보입니다.

1) **CRAG 외부 루프 (Self-Refine)**
   - `quality_check` 루프 on vs off (iteration_count=0으로 강제 종료)
   - 기대: 답변 품질/지연/토큰 변화 확인

2) **CRAG 내부 검색 재시도/쿼리 재작성**
   - query_rewrite on/off (`feature_flags['query_rewrite_enabled']`)
   - dynamic_rag_routing on/off (`feature_flags['dynamic_rag_routing']`)
   - top_k/rrf_k 변동

3) **Retrieval 전략**
   - Hybrid(BM25+FAISS+RRF) vs BM25 only vs FAISS only
   - token_plan.for_docs 예산 기반 k 조정 on/off
   - 평균 문서 길이 가정치(예: 150/200/300) 변경

4) **컨텍스트 엔지니어링**
   - ContextManager/TokenManager 사용 vs 미사용 (전체 history 평면 투입)
   - conversation_history 포함/미포함
   - profile_summary 포함/미포함
   - longterm_context(요약) 포함/미포함 (추후 checkpointer 도입 시)

5) **메모리/프로필**
   - ProfileStore 업데이트 on/off
   - 슬롯 추출 결과를 프로필에 반영 vs 미반영
   - 시간 가중치/중복제거 on/off

6) **품질 임계값·반복 한도**
   - quality_score 임계값 0.5 vs 0.7
   - iteration_count 최대 0/1/2

7) **LLM 프롬프트**
   - LLM 모드(단순 시스템 프롬프트) vs AI Agent 모드(개인화+근거)
   - system_prompt 길이/안전 가드 변화

---

## 4. LangGraph로 ablation이 쉬운 이유 (체계적 통제 관점)
- **노드/엣지 단위 스위치**: 특정 노드 호출을 건너뛰거나 엣지 분기를 바꾸는 것만으로 실험 조건을 정의.
- **상태 필드로 구성 요소 분리**: 프롬프트/프로필/토큰/검색 결과 등이 상태에 모여 있어, “이 필드를 비우고/채우고”가 곧 ablation.
- **feature_flags/agent_config**: 코드 수정 없이 실험 파라미터를 함수 인자/설정으로 주입 가능.
- **서브그래프/병렬**: 검색 서브그래프 교체, 병렬 검색 on/off 등 구조적 실험도 그래프 정의로 가능.

---

## 5. 추가 개선 제안 (ablation 통제력 향상)
- **설정화**: quality 임계값, 평균 문서 길이 가정, k 기본값 등을 config/agent_config로 노출.
- **longterm store 도입**: checkpointer/store를 붙여 장기 요약 on/off 실험 가능하게.
- **실험 매트릭 표준화**: 토큰, latency, RAGAS faithfulness/answer_relevance, 오류률을 공통 포맷으로 기록.
- **실험 스크립트 확장**: `test_agent_performance.py`에 feature_overrides 세트를 목록으로 넣어 일괄 실험 → CSV/MD로 결과 정리.

---

## 6. 결론
현재 스캐폴드는 LangGraph의 그래프/상태/조건부 엣지를 잘 활용해 ablation 친화적인 구조를 갖추고 있습니다. 다만 일부 상수(임계값, 평균 문서 길이) 설정화와 장기 메모리 스토어 도입이 더해지면, “그래프 정의 + 설정”만으로 폭넓은 실험을 재현성 있게 제어할 수 있습니다. 추가로 제안한 ablation 축을 따라 기능별 on/off·파라미터 변경을 수행하면, CONTEXT ENGINEERING 기반 AI Agent의 성능 기여 요인을 명확히 분리·검증할 수 있습니다.

