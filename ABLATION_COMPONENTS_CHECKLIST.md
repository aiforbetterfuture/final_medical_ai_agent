# Ablation 연구 컴포넌트 체크리스트

**작성일**: 2025-12-16  
**목적**: 새 스캐폴드에서 ablation 연구를 수행하기 위해 필요한 모든 컴포넌트 확인

---

## ✅ 1. 기본 RAG 구현

### 1.1 전략 패턴 (Strategy Pattern)
- [x] `agent/refine_strategies/base_strategy.py` - 추상 인터페이스
- [x] `agent/refine_strategies/basic_rag_strategy.py` - Basic RAG 구현
- [x] `agent/refine_strategies/corrective_rag_strategy.py` - Corrective RAG 구현
- [x] `agent/refine_strategies/strategy_factory.py` - 전략 팩토리
- [x] `agent/refine_strategies/__init__.py` - 패키지 초기화

### 1.2 노드 통합
- [x] `agent/nodes/refine.py` - Strategy 패턴 기반 refine 노드
- [x] `agent/nodes/quality_check.py` - Strategy 패턴 기반 quality_check 노드
- [x] `agent/graph.py` - LangGraph 정의 및 run_agent 함수
- [x] `agent/state.py` - AgentState 정의

---

## ✅ 2. 검색 모듈 (Retrieval)

- [x] `retrieval/hybrid_retriever.py` - 하이브리드 검색 (BM25 + FAISS)
- [x] `retrieval/faiss_index.py` - FAISS 벡터 인덱스
- [x] `retrieval/rrf_fusion.py` - RRF 융합 알고리즘
- [x] `retrieval/singleton_cache.py` - 싱글톤 캐시
- [x] `retrieval/__init__.py` - 패키지 초기화

---

## ✅ 3. 핵심 모듈 (Core)

- [x] `core/config.py` - 설정 관리 (YAML 로드, 환경 변수)
- [x] `core/llm_client.py` - LLM 클라이언트
- [x] `core/prompts.py` - 프롬프트 템플릿
- [x] `core/utils.py` - 유틸리티 함수
- [x] `core/__init__.py` - 패키지 초기화

---

## ✅ 4. Ablation 설정

- [x] `config/ablation_config.py` - 8개 사전 정의 프로파일
  - `baseline` - 최소 기능
  - `self_refine_heuristic` - 휴리스틱 품질 평가
  - `self_refine_llm_quality` - LLM 품질 평가
  - `self_refine_dynamic_query` - 동적 쿼리 재작성
  - `self_refine_full_safety` - 전체 안전장치
  - `full_context_engineering` - 최종 시스템
  - `quality_check_only` - Quality Check만
  - `self_refine_no_safety` - 안전장치 없음

- [x] `config/agent_config.yaml` - 에이전트 설정
- [x] `config/corpus_config.yaml` - 코퍼스 설정
- [x] `config/model_config.yaml` - 모델 설정

---

## ✅ 5. 메트릭 수집

- [x] `agent/metrics/ablation_metrics.py` - Ablation 메트릭 수집 시스템
  - QueryMetrics 데이터클래스
  - AblationMetrics 클래스
  - compare_experiments 함수

---

## ✅ 6. 실험 스크립트

- [x] `experiments/run_ablation_single.py` - 단일 ablation 실험 실행
- [x] `experiments/run_ablation_comparison.py` - 다중 프로파일 비교
- [x] `experiments/analyze_ablation_results.py` - 결과 분석
- [x] `experiments/test_basic_rag.py` - Basic RAG 테스트 (새로 생성)

---

## ✅ 7. 평가 모듈

- [x] `experiments/evaluation/ragas_metrics.py` - RAGAS 메트릭 계산
  - Faithfulness (근거 충실도)
  - Answer Relevancy (답변 관련성)
- [x] `experiments/evaluation/__init__.py` - 패키지 초기화

---

## ✅ 8. 문서

- [x] `ABLATION_STUDY_GUIDE.md` - 종합 가이드
- [x] `ABLATION_QUICK_START.md` - 빠른 시작 가이드
- [x] `ABLATION_RUN_GUIDE.md` - 실행 가이드
- [x] `ABLATION_LANGGRAPH_DESIGN.md` - LangGraph 설계
- [x] `ABLATION_THESIS_INTEGRATION_GUIDE.md` - 논문 통합 가이드
- [x] `CHAPTER_4_ABLATION_STUDY_KO.md` - 4장 (한국어)
- [x] `CHAPTER_5_CONCLUSION_ABLATION_KO.md` - 5장 (한국어)
- [x] `CRAG_VS_BASIC_RAG_GUIDE.md` - CRAG vs Basic RAG 가이드
- [x] `BASIC_VS_CRAG_EXPERIMENT_GUIDE.md` - 실험 가이드

---

## ✅ 9. Feature Flags (Ablation 변수)

### 9.1 핵심 Ablation Axes
- [x] `self_refine_enabled` - Self-Refine 루프 활성화
- [x] `quality_check_enabled` - 품질 검사 활성화
- [x] `llm_based_quality_check` - LLM 기반 품질 평가
- [x] `dynamic_query_rewrite` - 동적 질의 재작성
- [x] `duplicate_detection` - 중복 검색 방지
- [x] `progress_monitoring` - 진행도 모니터링
- [x] `refine_strategy` - 전략 선택 ('basic_rag', 'corrective_rag')

### 9.2 검색 관련
- [x] `retrieval_mode` - 검색 모드 ('hybrid', 'bm25', 'faiss')
- [x] `active_retrieval_enabled` - Active Retrieval 활성화
- [x] `default_k`, `simple_query_k`, `moderate_query_k`, `complex_query_k` - 동적 k 값

### 9.3 컨텍스트 관리
- [x] `use_context_manager` - Context Manager 활성화
- [x] `include_history` - 대화 이력 포함
- [x] `include_profile` - 환자 프로필 포함
- [x] `context_compression_enabled` - Context Compression 활성화
- [x] `hierarchical_memory_enabled` - Hierarchical Memory 활성화

### 9.4 캐싱
- [x] `response_cache_enabled` - 응답 캐싱 활성화
- [x] `cache_similarity_threshold` - 캐시 유사도 임계값

### 9.5 파라미터
- [x] `max_refine_iterations` - 최대 재검색 횟수
- [x] `quality_threshold` - 품질 임계값
- [x] `temperature` - LLM temperature
- [x] `top_k` (BM25/FAISS) - 검색 문서 수
- [x] `chunk_size`, `chunk_overlap` - 청킹 파라미터

---

## ✅ 10. 의존성

- [x] `requirements.txt` - 패키지 의존성
  - `langgraph` - LangGraph 프레임워크
  - `langchain-openai` - LangChain OpenAI 통합
  - `openai` - OpenAI API
  - `faiss-cpu` 또는 `faiss-gpu` - FAISS 벡터 검색
  - `rank-bm25` - BM25 검색
  - `ragas` - RAGAS 평가 프레임워크
  - `datasets` - 데이터셋 처리

---

## ✅ 11. 테스트 및 검증

### 11.1 기본 테스트
- [x] `experiments/test_basic_rag.py` - Basic RAG 테스트 스크립트

### 11.2 실행 방법
```python
# Basic RAG 테스트
python experiments/test_basic_rag.py

# Ablation 단일 실험
python experiments/run_ablation_single.py

# Ablation 비교 실험
python experiments/run_ablation_comparison.py

# 결과 분석
python experiments/analyze_ablation_results.py
```

---

## ✅ 12. 환경 설정

- [x] `.env` 또는 `env_template.txt` - 환경 변수 템플릿
  - `OPENAI_API_KEY` - OpenAI API 키
  - `MEDCAT_API_KEY` - MedCAT API 키 (선택)
  - `MEDCAT_LICENSE_CODE` - MedCAT 라이선스 코드 (선택)

---

## 📋 Ablation 연구 실행 체크리스트

### 사전 준비
1. [ ] 환경 변수 설정 (`.env` 파일)
2. [ ] 의존성 설치 (`pip install -r requirements.txt`)
3. [ ] FAISS 인덱스 생성 (필요 시)
4. [ ] 코퍼스 데이터 준비 (필요 시)

### 실험 실행
1. [ ] Basic RAG 테스트 실행 (`experiments/test_basic_rag.py`)
2. [ ] Ablation 프로파일 확인 (`config/ablation_config.py`)
3. [ ] 단일 실험 실행 (`experiments/run_ablation_single.py`)
4. [ ] 비교 실험 실행 (`experiments/run_ablation_comparison.py`)

### 결과 분석
1. [ ] 결과 파일 확인 (`experiments/results/`)
2. [ ] 메트릭 분석 (`experiments/analyze_ablation_results.py`)
3. [ ] RAGAS 메트릭 계산 (`experiments/evaluation/ragas_metrics.py`)

---

## 🎯 핵심 Ablation 실험 설계

### 최소 실험 (논문용)
1. **Exp-A**: Baseline LLM (`mode='llm'`)
2. **Exp-B**: Basic RAG (`refine_strategy='basic_rag'`)
3. **Exp-C**: RAG + Self-Refine (`self_refine_enabled=True`)
4. **Exp-D**: Full System (`profile='full_context_engineering'`)

### 평가 메트릭
- Faithfulness (근거 충실도)
- Answer Relevancy (답변 관련성)
- Perplexity (불확실성)
- Judge Total Score (LLM 평가)
- Iteration Count (반복 횟수)
- Cost & Time (비용 & 시간)

---

## 📝 참고 문서

- `ABLATION_STUDY_GUIDE.md` - 종합 가이드
- `ABLATION_QUICK_START.md` - 빠른 시작
- `MODULAR_RAG_STRATEGY_AND_ARCHITECTURE_ANALYSIS.md` - 모듈식 RAG 전략
- `ZERO_TO_ONE_REDESIGN_STRATEGY.md` - 재설계 전략

---

## ✅ 완료 상태

**모든 필수 컴포넌트가 설치되었습니다!**

- ✅ 기본 RAG 구현 완료
- ✅ Corrective RAG 구현 완료
- ✅ Ablation 설정 완료
- ✅ 메트릭 수집 시스템 완료
- ✅ 실험 스크립트 완료
- ✅ 평가 모듈 완료
- ✅ 문서 완료

**다음 단계**: `experiments/test_basic_rag.py` 실행하여 Basic RAG 동작 확인

