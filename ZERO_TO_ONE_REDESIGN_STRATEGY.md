# 의학 지식 기반 AI 에이전트 제로베이스 재설계 전략

**작성일**: 2025-12-15  
**목적**: Context Engineering 및 컴퓨터공학 관점에서 Ablation 연구에 최적화된 시스템 설계 전략 제시

---

## 목차

1. [설계 철학 및 원칙](#1-설계-철학-및-원칙)
2. [데이터 레이어: 청킹 & 임베딩 전략](#2-데이터-레이어-청킹--임베딩-전략)
3. [아키텍처 레이어: 모듈 설계 순서](#3-아키텍처-레이어-모듈-설계-순서)
4. [Ablation 연구 설계 전략](#4-ablation-연구-설계-전략)
5. [구현 로드맵](#5-구현-로드맵)
6. [측정 및 평가 프레임워크](#6-측정-및-평가-프레임워크)

---

## 1. 설계 철학 및 원칙

### 1.1 핵심 설계 원칙

#### **Principle 1: Modularity First (모듈성 우선)**
```
각 컴포넌트는 독립적으로 교체/비활성화 가능해야 함
→ Ablation 연구의 전제 조건
```

#### **Principle 2: Data-Centric Design (데이터 중심 설계)**
```
알고리즘보다 데이터 품질이 성능을 좌우함
→ 청킹/임베딩 전략이 최우선 설계 대상
```

#### **Principle 3: Measurability by Design (측정 가능성 내재화)**
```
모든 컴포넌트는 입출력 메트릭을 자동 수집해야 함
→ 실험 재현성 보장
```

#### **Principle 4: Progressive Enhancement (점진적 개선)**
```
Baseline → Basic RAG → Advanced Features 순으로 구축
→ 각 단계의 기여도를 명확히 측정 가능
```

### 1.2 Ablation 연구를 위한 아키텍처 요구사항

| 요구사항 | 구현 방법 | 우선순위 |
|---------|----------|---------|
| **독립 변수 격리** | Feature flags + Config injection | ⭐⭐⭐⭐⭐ |
| **재현 가능성** | Deterministic seed + Version control | ⭐⭐⭐⭐⭐ |
| **메트릭 자동 수집** | Instrumentation layer | ⭐⭐⭐⭐⭐ |
| **빠른 실험 반복** | Cached components + Parallel execution | ⭐⭐⭐⭐ |
| **결과 비교 용이성** | Standardized output format | ⭐⭐⭐⭐ |

---

## 2. 데이터 레이어: 청킹 & 임베딩 전략

### 2.1 현재 시스템 분석

**현재 설정** (from `config/corpus_config.yaml`):
```yaml
chunking:
  strategy: sliding_window
  chunk_size: 900 tokens
  chunk_overlap: 200 tokens

embedding:
  model: text-embedding-3-large
  normalize: true
  metric: inner_product
```

**문제점**:
- ❌ 900 토큰은 의학 지식 검증에 너무 큼 (상충 정보 혼재 위험)
- ❌ 단일 청킹 전략으로 모든 문서 타입 처리 (약물 금기 vs 가이드라인 차별화 없음)
- ❌ 청크 메타데이터 부족 (섹션, 출처, 신뢰도 정보 없음)

### 2.2 제안: 계층적 멀티-전략 청킹

#### **Phase 1: 문서 타입 분류 기반 청킹**

```python
# 문서 타입별 청킹 전략 매핑
CHUNKING_STRATEGIES = {
    'drug_contraindication': {
        'chunk_size': 180,      # 짧게: 금기사항은 혼합 방지
        'chunk_overlap': 40,
        'preserve_sentences': True,
        'section_aware': True
    },
    'clinical_guideline': {
        'chunk_size': 280,      # 중간: 권고사항 단위
        'chunk_overlap': 70,
        'preserve_paragraphs': True
    },
    'case_report': {
        'chunk_size': 320,      # 긴: 맥락 유지 필요
        'chunk_overlap': 80,
        'preserve_context': True
    },
    'general_knowledge': {
        'chunk_size': 400,      # 가장 긴: 설명형 콘텐츠
        'chunk_overlap': 100
    }
}
```

**구현 우선순위**:
1. **P0 (필수)**: `drug_contraindication`, `clinical_guideline` 분리
2. **P1 (권장)**: `case_report` 추가
3. **P2 (선택)**: `general_knowledge` 세분화

#### **Phase 2: 메타데이터 강화**

```python
# 각 청크에 포함할 메타데이터
CHUNK_METADATA = {
    'chunk_id': str,           # 고유 식별자
    'doc_id': str,             # 원본 문서 ID
    'doc_type': str,           # 문서 타입 (위 4가지)
    'section': str,            # 섹션명 (e.g., "Contraindications", "Dosage")
    'span_start': int,         # 원본 문서 내 시작 위치
    'span_end': int,           # 원본 문서 내 종료 위치
    'source': str,             # 출처 (e.g., "FDA Label", "UpToDate")
    'published_date': str,     # 발행일 (신뢰도 계산용)
    'confidence_score': float, # 출처 신뢰도 (0-1)
    'entities': List[str],     # 추출된 의학 엔티티 (MedCAT2)
    'keywords': List[str]      # BM25용 키워드
}
```

**Ablation 변수**:
- `metadata_richness`: `minimal` (chunk_id만) vs `full` (전체)
- 메타데이터가 retrieval/reranking에 미치는 영향 측정

#### **Phase 3: 듀얼 인덱스 전략**

```
┌─────────────────────────────────────────┐
│         Query Router                     │
│  (Intent: symptom/drug/general)          │
└─────────────┬───────────────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌──────────┐      ┌──────────┐
│ Fine-    │      │ Coarse-  │
│ Grained  │      │ Grained  │
│ Index    │      │ Index    │
│          │      │          │
│ 180-280  │      │ 320-400  │
│ tokens   │      │ tokens   │
│          │      │          │
│ k=12-16  │      │ k=5-8    │
└──────────┘      └──────────┘
      │                │
      └────────┬───────┘
               ▼
         RRF Fusion
         (k=60)
```

**Ablation 변수**:
- `index_strategy`: `single` vs `dual` vs `triple`
- `routing_enabled`: True/False
- `k_values`: (fine, coarse) 조합 실험

### 2.3 임베딩 전략

#### **Option A: 단일 임베딩 모델 (현재 방식)**

```yaml
embedding:
  model: text-embedding-3-large  # 3072 dim
  normalize: true
  batch_size: 128
```

**장점**: 단순, 일관성
**단점**: 쿼리와 문서의 특성 차이 반영 불가

#### **Option B: 쿼리 증강 임베딩 (권장)**

```python
# 쿼리를 2가지 형태로 임베딩
def embed_query_dual(query: str, profile: dict, slots: dict):
    # 1. Full context query (기존 방식)
    full_query = f"{query}\n환자 정보: {profile}\n증상: {slots}"
    emb_full = embed(full_query)
    
    # 2. Keyword-only query (BM25 보완용)
    keywords = extract_medical_keywords(query, slots)
    emb_keywords = embed(" ".join(keywords))
    
    # 가중 결합
    return 0.7 * emb_full + 0.3 * emb_keywords
```

**Ablation 변수**:
- `query_augmentation`: `none` / `profile` / `dual_embedding`
- `weight_ratio`: (0.5, 0.5) / (0.7, 0.3) / (0.9, 0.1)

#### **Option C: 도메인 특화 파인튜닝 (장기 전략)**

```python
# 의학 도메인 triplet 데이터로 파인튜닝
# (query, positive_doc, negative_doc) 쌍 생성
# 현재 시스템의 train_qa 데이터 활용 가능
```

**우선순위**: P2 (연구 확장 시)

### 2.4 청킹/임베딩 Ablation 실험 매트릭스

| 실험 ID | 청크 크기 | 중첩 | 메타데이터 | 임베딩 전략 | 예상 영향 |
|--------|---------|------|-----------|-----------|----------|
| **D1** | 900 (현재) | 200 | minimal | single | Baseline |
| **D2** | 280 | 70 | minimal | single | 청크 크기 효과 |
| **D3** | 280 | 70 | full | single | 메타데이터 효과 |
| **D4** | 280 | 70 | full | dual | 쿼리 증강 효과 |
| **D5** | 타입별 가변 | 가변 | full | dual | 최종 시스템 |

**측정 메트릭**:
- Recall@k (k=5, 10, 20)
- Precision@k
- MRR (Mean Reciprocal Rank)
- Hallucination Rate (LLM judge)
- Retrieval Latency

---

## 3. 아키텍처 레이어: 모듈 설계 순서

### 3.1 Bottom-Up 설계 접근

```
Layer 5: Application (UI/API)
         ↑
Layer 4: Agent Orchestration (LangGraph)
         ↑
Layer 3: RAG Components (Retrieval, Generation, Refinement)
         ↑
Layer 2: Core Services (LLM Client, Memory, Cache)
         ↑
Layer 1: Data Infrastructure (Chunking, Indexing, Storage)
         ↑
Layer 0: Configuration & Instrumentation
```

**설계 순서**: Layer 0 → Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5

### 3.2 Layer 0: Configuration & Instrumentation (최우선)

#### **3.2.1 Feature Flag 시스템**

```python
# config/feature_flags.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class FeatureFlags:
    """모든 ablation 변수를 관리하는 중앙 설정"""
    
    # === Data Layer ===
    chunk_size: int = 280
    chunk_overlap: int = 70
    chunking_strategy: str = 'type_aware'  # 'uniform' / 'type_aware'
    metadata_richness: str = 'full'  # 'minimal' / 'full'
    
    # === Embedding Layer ===
    embedding_model: str = 'text-embedding-3-large'
    query_augmentation: str = 'dual'  # 'none' / 'profile' / 'dual'
    normalize_embeddings: bool = True
    
    # === Retrieval Layer ===
    retrieval_mode: str = 'hybrid'  # 'bm25' / 'faiss' / 'hybrid'
    index_strategy: str = 'dual'  # 'single' / 'dual'
    routing_enabled: bool = True
    k_fine: int = 12
    k_coarse: int = 5
    rrf_k: int = 60
    
    # === Reranking Layer ===
    reranking_enabled: bool = False
    reranker_model: Optional[str] = None
    rerank_top_n: int = 30
    rerank_keep_n: int = 5
    
    # === Generation Layer ===
    llm_model: str = 'gpt-4o-mini'
    temperature: float = 0.2
    max_tokens: int = 800
    
    # === Self-Refine Layer ===
    self_refine_enabled: bool = True
    max_refine_iterations: int = 2
    quality_threshold: float = 0.5
    llm_based_quality: bool = True
    dynamic_query_rewrite: bool = True
    
    # === Context Engineering ===
    include_profile: bool = True
    include_history: bool = True
    context_manager_enabled: bool = True
    token_budget: int = 3000
    
    # === Memory & Cache ===
    response_cache_enabled: bool = True
    cache_similarity_threshold: float = 0.85
    hierarchical_memory: bool = False
    
    # === Safety & Monitoring ===
    duplicate_detection: bool = True
    progress_monitoring: bool = True
    timeout_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """실험 결과 저장용"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_profile(cls, profile_name: str) -> 'FeatureFlags':
        """사전 정의 프로파일 로드"""
        return ABLATION_PROFILES[profile_name]

# 사전 정의 프로파일
ABLATION_PROFILES = {
    'baseline': FeatureFlags(
        self_refine_enabled=False,
        retrieval_mode='faiss',
        routing_enabled=False,
        include_profile=False,
        include_history=False,
    ),
    'basic_rag': FeatureFlags(
        self_refine_enabled=False,
        retrieval_mode='hybrid',
    ),
    'full_system': FeatureFlags(
        # 모든 기능 활성화 (기본값)
    ),
}
```

#### **3.2.2 Instrumentation Layer**

```python
# core/instrumentation.py
from contextlib import contextmanager
from typing import Dict, Any
import time
import json

class MetricsCollector:
    """모든 컴포넌트의 메트릭을 자동 수집"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.metrics = []
    
    @contextmanager
    def measure(self, component: str, operation: str, **metadata):
        """컴포넌트 실행 시간 및 메트릭 측정"""
        start = time.time()
        result = {'component': component, 'operation': operation}
        result.update(metadata)
        
        try:
            yield result
        finally:
            result['elapsed_ms'] = (time.time() - start) * 1000
            self.metrics.append(result)
    
    def save(self, output_path: str):
        """실험 결과 저장"""
        with open(output_path, 'w') as f:
            json.dump({
                'experiment_id': self.experiment_id,
                'metrics': self.metrics
            }, f, indent=2)

# 사용 예시
collector = MetricsCollector('exp_001')

with collector.measure('retrieval', 'hybrid_search', k=10):
    docs = retriever.search(query, k=10)
    result['num_docs'] = len(docs)
    result['avg_score'] = sum(d['score'] for d in docs) / len(docs)
```

**Ablation 기여도**:
- ✅ 모든 실험의 재현성 보장
- ✅ 컴포넌트별 성능 병목 자동 식별
- ✅ 실험 간 비교 용이

### 3.3 Layer 1: Data Infrastructure

#### **구현 순서**:

```
1. Chunking Pipeline (2-3일)
   ├─ 문서 타입 분류기
   ├─ 타입별 청킹 전략 구현
   └─ 메타데이터 추출기

2. Embedding Pipeline (1-2일)
   ├─ 배치 임베딩 생성
   ├─ 정규화 및 저장
   └─ 쿼리 증강 로직

3. Index Builder (2-3일)
   ├─ FAISS 인덱스 생성 (fine/coarse)
   ├─ BM25 인덱스 생성
   └─ 메타데이터 매핑

4. Validation Suite (1일)
   ├─ 청크 품질 검증
   ├─ 임베딩 일관성 검사
   └─ 인덱스 무결성 테스트
```

**핵심 코드 구조**:

```python
# data_pipeline/chunker.py
class TypeAwareChunker:
    """문서 타입별 청킹 전략 적용"""
    
    def __init__(self, strategies: Dict[str, ChunkStrategy]):
        self.strategies = strategies
    
    def chunk_document(self, doc: Document) -> List[Chunk]:
        doc_type = self.classify_document(doc)
        strategy = self.strategies[doc_type]
        
        chunks = strategy.chunk(doc.text)
        
        # 메타데이터 추가
        for chunk in chunks:
            chunk.metadata.update({
                'doc_type': doc_type,
                'doc_id': doc.id,
                'source': doc.source,
                'confidence': doc.confidence
            })
        
        return chunks

# data_pipeline/indexer.py
class DualIndexBuilder:
    """Fine-grained / Coarse-grained 듀얼 인덱스 생성"""
    
    def build(self, chunks: List[Chunk]):
        # 청크 크기로 분류
        fine_chunks = [c for c in chunks if c.token_count < 300]
        coarse_chunks = [c for c in chunks if c.token_count >= 300]
        
        # 각각 인덱스 생성
        self.fine_index = self._build_faiss_index(fine_chunks)
        self.coarse_index = self._build_faiss_index(coarse_chunks)
        
        # BM25는 전체 청크 사용
        self.bm25_index = self._build_bm25_index(chunks)
```

### 3.4 Layer 2: Core Services

#### **구현 순서**:

```
1. LLM Client (1일) - 이미 구현됨
   ├─ OpenAI/Gemini 통합
   └─ Rate limiting & retry

2. Memory Manager (2일)
   ├─ Profile storage
   ├─ History management
   └─ Temporal weighting

3. Cache Layer (1일)
   ├─ Response cache
   ├─ Retrieval cache
   └─ Similarity-based lookup
```

**Ablation 변수**:
- `memory_enabled`: True/False
- `cache_enabled`: True/False
- `cache_threshold`: 0.7 / 0.8 / 0.85 / 0.9

### 3.5 Layer 3: RAG Components

#### **구현 순서 (Ablation 우선순위 기준)**:

```
Priority 1: Baseline Components (3-4일)
├─ 1. Simple Retriever (BM25 only)
├─ 2. Simple Generator (LLM direct call)
└─ 3. Basic Evaluator (heuristic quality)

Priority 2: Hybrid Retrieval (2-3일)
├─ 4. FAISS Retriever
├─ 5. RRF Fusion
└─ 6. Query Router

Priority 3: Self-Refine Loop (3-4일)
├─ 7. Quality Checker (LLM-based)
├─ 8. Query Rewriter
└─ 9. Iteration Controller

Priority 4: Advanced Features (2-3일)
├─ 10. Reranker
├─ 11. Context Compressor
└─ 12. Active Retrieval
```

**핵심: 각 컴포넌트는 독립적으로 on/off 가능**

```python
# retrieval/retriever_factory.py
class RetrieverFactory:
    """Feature flags에 따라 적절한 retriever 생성"""
    
    @staticmethod
    def create(flags: FeatureFlags) -> BaseRetriever:
        if flags.retrieval_mode == 'bm25':
            return BM25Retriever(...)
        elif flags.retrieval_mode == 'faiss':
            return FAISSRetriever(...)
        elif flags.retrieval_mode == 'hybrid':
            return HybridRetriever(
                bm25=BM25Retriever(...),
                faiss=FAISSRetriever(...),
                fusion=RRFFusion(k=flags.rrf_k)
            )
```

### 3.6 Layer 4: Agent Orchestration (LangGraph)

#### **노드 설계 원칙**:

```python
# agent/nodes/base_node.py
from abc import ABC, abstractmethod

class BaseNode(ABC):
    """모든 노드의 기본 클래스"""
    
    def __init__(self, flags: FeatureFlags, collector: MetricsCollector):
        self.flags = flags
        self.collector = collector
    
    def __call__(self, state: AgentState) -> AgentState:
        """메트릭 수집이 자동화된 실행"""
        with self.collector.measure(
            component=self.__class__.__name__,
            operation='execute',
            state_keys=list(state.keys())
        ) as metrics:
            result_state = self.execute(state)
            metrics['output_keys'] = list(result_state.keys())
        
        return result_state
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """실제 로직 구현"""
        pass
```

#### **노드 구현 순서**:

```
Phase 1: Minimal Pipeline (2일)
1. ExtractSlotsNode
2. RetrieveNode (simple)
3. GenerateAnswerNode (simple)

Phase 2: Context Engineering (2일)
4. StoreMemoryNode
5. AssembleContextNode

Phase 3: Self-Refine (3일)
6. QualityCheckNode
7. RefineNode (query rewrite + re-retrieve)

Phase 4: Advanced (2일)
8. ClassifyIntentNode (routing)
9. CheckSimilarityNode (cache)
```

**그래프 구성**:

```python
# agent/graph_builder.py
class GraphBuilder:
    """Feature flags에 따라 동적으로 그래프 구성"""
    
    def build(self, flags: FeatureFlags) -> StateGraph:
        graph = StateGraph(AgentState)
        
        # 필수 노드
        graph.add_node("extract_slots", ExtractSlotsNode(flags))
        graph.add_node("retrieve", RetrieveNode(flags))
        graph.add_node("generate", GenerateAnswerNode(flags))
        
        # 조건부 노드
        if flags.include_profile or flags.include_history:
            graph.add_node("store_memory", StoreMemoryNode(flags))
            graph.add_node("assemble_context", AssembleContextNode(flags))
        
        if flags.self_refine_enabled:
            graph.add_node("quality_check", QualityCheckNode(flags))
            graph.add_node("refine", RefineNode(flags))
        
        # 엣지 연결 (flags에 따라 조건부)
        graph.set_entry_point("extract_slots")
        # ... (생략)
        
        return graph.compile()
```

---

## 4. Ablation 연구 설계 전략

### 4.1 실험 설계 원칙

#### **Principle 1: Single Variable Testing**
```
한 번에 하나의 변수만 변경
→ 인과관계 명확화
```

#### **Principle 2: Cumulative Building**
```
Baseline → +Feature A → +Feature B → ...
→ 각 기능의 marginal contribution 측정
```

#### **Principle 3: Cross-Validation**
```
Train/Dev/Test split 엄격히 준수
→ Overfitting 방지
```

### 4.2 3-Tier Ablation 전략

#### **Tier 1: Data Layer Ablation (최우선)**

**가설**: "청킹/임베딩 전략이 전체 성능의 40-60%를 결정한다"

| 실험 | 변수 | 설정 | 예상 결과 |
|-----|------|------|----------|
| **D1** | Baseline | chunk=900, single_index | Recall@5: 0.65 |
| **D2** | Fine chunking | chunk=280, single_index | Recall@5: 0.72 (+7%p) |
| **D3** | + Metadata | chunk=280, metadata=full | Recall@5: 0.75 (+3%p) |
| **D4** | + Dual index | chunk=280, dual_index | Recall@5: 0.78 (+3%p) |
| **D5** | + Query aug | chunk=280, dual_emb | Recall@5: 0.82 (+4%p) |

**실행 방법**:
```bash
python experiments/run_data_ablation.py \
  --experiments D1,D2,D3,D4,D5 \
  --dataset val_qa \
  --metrics recall,precision,mrr
```

#### **Tier 2: Retrieval Layer Ablation**

**가설**: "Hybrid retrieval이 단일 방법 대비 10-15% 성능 향상"

| 실험 | 검색 방법 | RRF | Reranking | 예상 MRR |
|-----|----------|-----|-----------|---------|
| **R1** | BM25 only | - | - | 0.55 |
| **R2** | FAISS only | - | - | 0.62 |
| **R3** | Hybrid (BM25+FAISS) | RRF k=60 | - | 0.70 |
| **R4** | Hybrid + Rerank | RRF k=60 | Top 30→5 | 0.75 |

#### **Tier 3: Generation Layer Ablation**

**가설**: "Self-Refine이 답변 품질을 20-30% 향상"

| 실험 | Self-Refine | Quality Check | Query Rewrite | Judge Score |
|-----|-------------|---------------|---------------|------------|
| **G1** | OFF | - | - | 6.5/10 |
| **G2** | ON (heuristic) | Heuristic | Static | 7.2/10 |
| **G3** | ON (LLM) | LLM-based | Static | 7.8/10 |
| **G4** | ON (LLM) | LLM-based | Dynamic | 8.5/10 |

### 4.3 Full Factorial Experiment (선택적)

**목적**: 컴포넌트 간 상호작용 효과 측정

```python
# experiments/factorial_design.py
FACTORS = {
    'chunk_size': [280, 500, 900],
    'retrieval_mode': ['bm25', 'faiss', 'hybrid'],
    'self_refine': [False, True],
    'llm_model': ['gpt-4o-mini', 'gpt-4o']
}

# 3 x 3 x 2 x 2 = 36 combinations
# 실제로는 invalid 조합 제거 후 ~25개
```

**주의**: 비용 및 시간 고려 (1 combination = 10분 → 25개 = 4시간)

### 4.4 실험 실행 프레임워크

```python
# experiments/runner.py
class AblationRunner:
    """체계적인 ablation 실험 실행"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.results = []
    
    def run_experiment(self, exp_id: str, flags: FeatureFlags):
        """단일 실험 실행"""
        print(f"\n{'='*60}")
        print(f"실험 {exp_id} 시작")
        print(f"{'='*60}")
        
        collector = MetricsCollector(exp_id)
        agent = build_agent(flags, collector)
        
        # 테스트 데이터셋 로드
        test_cases = load_test_cases(self.config['dataset'])
        
        results = []
        for i, case in enumerate(test_cases):
            print(f"[{i+1}/{len(test_cases)}] {case['query'][:50]}...")
            
            result = agent.run(
                user_text=case['query'],
                ground_truth=case.get('answer')
            )
            
            # 메트릭 계산
            metrics = self.evaluate(result, case)
            results.append(metrics)
        
        # 집계 통계
        summary = self.aggregate(results)
        
        # 저장
        self.save_results(exp_id, flags, results, summary)
        
        return summary
    
    def run_batch(self, experiment_ids: List[str]):
        """여러 실험 순차 실행"""
        for exp_id in experiment_ids:
            flags = self.config['experiments'][exp_id]
            self.run_experiment(exp_id, flags)
        
        # 비교 리포트 생성
        self.generate_comparison_report()
```

**사용 예시**:
```bash
# experiments/ablation_config.yaml
experiments:
  D1:
    chunk_size: 900
    retrieval_mode: faiss
    self_refine_enabled: false
  
  D2:
    chunk_size: 280
    retrieval_mode: faiss
    self_refine_enabled: false
  
  # ... (생략)

dataset: val_qa
metrics:
  - recall@5
  - precision@5
  - mrr
  - judge_score
  - latency

# 실행
python experiments/runner.py \
  --config experiments/ablation_config.yaml \
  --experiments D1,D2,D3,D4,D5
```

---

## 5. 구현 로드맵

### 5.1 Phase 0: 기반 구축 (1주)

```
Week 1: Infrastructure Setup
├─ Day 1-2: Feature flag 시스템 구현
├─ Day 3-4: Instrumentation layer 구현
├─ Day 5: 테스트 데이터셋 준비
└─ Day 6-7: Baseline 시스템 구현 (최소 기능)
```

**Deliverables**:
- [ ] `config/feature_flags.py` 완성
- [ ] `core/instrumentation.py` 완성
- [ ] Baseline agent 동작 (LLM direct call)
- [ ] 테스트 스크립트 작성

### 5.2 Phase 1: Data Layer (1-2주)

```
Week 2-3: Chunking & Embedding
├─ Day 1-3: 문서 타입 분류 및 청킹 파이프라인
├─ Day 4-5: 메타데이터 추출 및 강화
├─ Day 6-8: 임베딩 생성 (배치 처리)
├─ Day 9-10: 듀얼 인덱스 구축
└─ Day 11-12: Data ablation 실험 (D1-D5)
```

**Deliverables**:
- [ ] `data_pipeline/chunker.py` 완성
- [ ] `data_pipeline/indexer.py` 완성
- [ ] Fine/Coarse FAISS 인덱스 생성
- [ ] Data ablation 결과 리포트

**Critical Milestone**: Recall@5 > 0.75 달성

### 5.3 Phase 2: Retrieval Layer (1주)

```
Week 4: Hybrid Retrieval
├─ Day 1-2: BM25 retriever 구현
├─ Day 3-4: FAISS retriever + 듀얼 인덱스 통합
├─ Day 5: RRF fusion 구현
├─ Day 6: Query router 구현
└─ Day 7: Retrieval ablation 실험 (R1-R4)
```

**Deliverables**:
- [ ] `retrieval/hybrid_retriever.py` 완성
- [ ] Retrieval ablation 결과 리포트

**Critical Milestone**: MRR > 0.70 달성

### 5.4 Phase 3: Generation & Self-Refine (1-2주)

```
Week 5-6: Self-Refine Loop
├─ Day 1-2: Basic generator 구현
├─ Day 3-4: Quality checker (heuristic + LLM)
├─ Day 5-6: Query rewriter
├─ Day 7-8: Refine loop 통합
├─ Day 9-10: Generation ablation 실험 (G1-G4)
└─ Day 11-12: 전체 시스템 통합 테스트
```

**Deliverables**:
- [ ] `agent/nodes/refine.py` 완성
- [ ] Generation ablation 결과 리포트
- [ ] End-to-end 시스템 동작 검증

**Critical Milestone**: Judge Score > 8.0 달성

### 5.5 Phase 4: Context Engineering (1주)

```
Week 7: Memory & Context
├─ Day 1-2: Profile storage 구현
├─ Day 3-4: History management
├─ Day 5: Context assembly
├─ Day 6: Token budget manager
└─ Day 7: Context ablation 실험
```

**Deliverables**:
- [ ] `memory/profile_store.py` 완성
- [ ] `context/context_manager.py` 완성
- [ ] Multi-turn 테스트 통과

### 5.6 Phase 5: Advanced Features (1주, 선택적)

```
Week 8: Optional Enhancements
├─ Day 1-2: Reranker 통합
├─ Day 3-4: Context compressor
├─ Day 5: Active retrieval
└─ Day 6-7: Advanced ablation 실험
```

### 5.7 Phase 6: Evaluation & Analysis (1주)

```
Week 9: Comprehensive Evaluation
├─ Day 1-2: Full factorial experiment (선택)
├─ Day 3-4: 결과 분석 및 시각화
├─ Day 5-6: 논문 작성용 표/그래프 생성
└─ Day 7: 최종 리포트 작성
```

**Deliverables**:
- [ ] 전체 ablation 결과 테이블
- [ ] 성능 비교 그래프
- [ ] 논문 초안 (Method + Results 섹션)

---

## 6. 측정 및 평가 프레임워크

### 6.1 메트릭 계층 구조

```
┌─────────────────────────────────────────┐
│     Business Metrics (최종 목표)          │
│  - User Satisfaction                     │
│  - Clinical Accuracy                     │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│     System Metrics (전체 성능)            │
│  - End-to-End Quality Score              │
│  - Hallucination Rate                    │
│  - Response Time                         │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│   Component Metrics (모듈별 성능)         │
│  Retrieval: Recall, Precision, MRR       │
│  Generation: Faithfulness, Relevance     │
│  Refine: Iteration Count, Improvement    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│   Infrastructure Metrics (효율성)         │
│  - Latency (per component)               │
│  - Token Usage                           │
│  - Cache Hit Rate                        │
└─────────────────────────────────────────┘
```

### 6.2 핵심 메트릭 정의

#### **Retrieval Metrics**

```python
# evaluation/retrieval_metrics.py
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """상위 k개 중 관련 문서 비율"""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """상위 k개 중 정확도"""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / k

def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Mean Reciprocal Rank"""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: List[Tuple[str, float]], relevant: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain"""
    dcg = sum(
        (1 if doc_id in relevant else 0) / np.log2(i + 2)
        for i, (doc_id, score) in enumerate(retrieved[:k])
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0
```

#### **Generation Metrics**

```python
# evaluation/generation_metrics.py
def faithfulness(answer: str, contexts: List[str], llm_judge) -> float:
    """답변이 검색된 문서에 근거하는 정도 (0-1)"""
    prompt = f"""
    Answer: {answer}
    Contexts: {contexts}
    
    Rate how well the answer is supported by the contexts (0-1).
    Only return a number.
    """
    score = llm_judge.evaluate(prompt)
    return float(score)

def answer_relevance(answer: str, query: str, llm_judge) -> float:
    """답변이 질문에 관련된 정도 (0-1)"""
    prompt = f"""
    Query: {query}
    Answer: {answer}
    
    Rate how relevant the answer is to the query (0-1).
    Only return a number.
    """
    score = llm_judge.evaluate(prompt)
    return float(score)

def hallucination_rate(answers: List[str], contexts: List[List[str]], llm_judge) -> float:
    """전체 답변 중 hallucination 비율"""
    hallucinations = 0
    for answer, context in zip(answers, contexts):
        if faithfulness(answer, context, llm_judge) < 0.5:
            hallucinations += 1
    return hallucinations / len(answers)
```

#### **Self-Refine Metrics**

```python
# evaluation/refine_metrics.py
def quality_improvement(initial_score: float, final_score: float) -> float:
    """품질 개선 정도"""
    return final_score - initial_score

def iteration_efficiency(iterations: int, improvement: float) -> float:
    """반복당 개선 효율"""
    return improvement / iterations if iterations > 0 else 0.0

def convergence_rate(quality_history: List[float]) -> float:
    """품질 수렴 속도 (기울기)"""
    if len(quality_history) < 2:
        return 0.0
    return (quality_history[-1] - quality_history[0]) / len(quality_history)
```

### 6.3 자동 평가 파이프라인

```python
# evaluation/auto_evaluator.py
class AutoEvaluator:
    """모든 메트릭을 자동으로 계산"""
    
    def __init__(self, llm_judge):
        self.llm_judge = llm_judge
    
    def evaluate_full(self, result: Dict, ground_truth: Dict) -> Dict[str, float]:
        """전체 메트릭 계산"""
        metrics = {}
        
        # Retrieval metrics
        if 'retrieved_docs' in result and 'relevant_docs' in ground_truth:
            metrics['recall@5'] = recall_at_k(
                result['retrieved_docs'], 
                ground_truth['relevant_docs'], 
                k=5
            )
            metrics['precision@5'] = precision_at_k(
                result['retrieved_docs'], 
                ground_truth['relevant_docs'], 
                k=5
            )
            metrics['mrr'] = mrr(
                result['retrieved_docs'], 
                ground_truth['relevant_docs']
            )
        
        # Generation metrics
        if 'answer' in result:
            metrics['faithfulness'] = faithfulness(
                result['answer'], 
                result['retrieved_docs'], 
                self.llm_judge
            )
            metrics['relevance'] = answer_relevance(
                result['answer'], 
                result['query'], 
                self.llm_judge
            )
        
        # Self-refine metrics
        if 'refine_history' in result:
            history = result['refine_history']
            metrics['iterations'] = len(history)
            metrics['quality_improvement'] = quality_improvement(
                history[0]['quality'], 
                history[-1]['quality']
            )
            metrics['convergence_rate'] = convergence_rate(
                [h['quality'] for h in history]
            )
        
        # Efficiency metrics
        metrics['latency_ms'] = result.get('elapsed_ms', 0)
        metrics['tokens_used'] = result.get('total_tokens', 0)
        metrics['cost_usd'] = result.get('estimated_cost', 0)
        
        return metrics
```

### 6.4 결과 시각화

```python
# evaluation/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AblationVisualizer:
    """Ablation 결과 시각화"""
    
    def plot_comparison(self, results: Dict[str, Dict]):
        """여러 실험 결과 비교"""
        df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Recall 비교
        df['recall@5'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Recall@5 Comparison')
        axes[0, 0].set_ylabel('Recall@5')
        
        # 2. Quality Score 비교
        df['judge_score'].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('Judge Score Comparison')
        axes[0, 1].set_ylabel('Score (0-10)')
        
        # 3. Latency 비교
        df['latency_ms'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('Latency Comparison')
        axes[1, 0].set_ylabel('Milliseconds')
        
        # 4. Cost 비교
        df['cost_usd'].plot(kind='bar', ax=axes[1, 1], color='gold')
        axes[1, 1].set_title('Cost Comparison')
        axes[1, 1].set_ylabel('USD')
        
        plt.tight_layout()
        plt.savefig('ablation_comparison.png', dpi=300)
    
    def plot_refine_trajectory(self, refine_history: List[Dict]):
        """Self-Refine 품질 개선 궤적"""
        iterations = [h['iteration'] for h in refine_history]
        quality = [h['quality'] for h in refine_history]
        
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, quality, marker='o', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Quality Score')
        plt.title('Self-Refine Quality Improvement')
        plt.grid(True, alpha=0.3)
        plt.savefig('refine_trajectory.png', dpi=300)
```

---

## 7. 실전 체크리스트

### 7.1 시작 전 준비

- [ ] **데이터 준비**
  - [ ] 원본 의학 지식 데이터 수집 (최소 1,000 문서)
  - [ ] Train/Dev/Test split (70/15/15)
  - [ ] Ground truth 레이블 생성 (relevant docs, ideal answers)

- [ ] **환경 설정**
  - [ ] Python 3.9+ 환경 구축
  - [ ] API 키 설정 (OpenAI, Gemini)
  - [ ] Git repository 초기화
  - [ ] 의존성 설치 (`requirements.txt`)

- [ ] **베이스라인 구현**
  - [ ] LLM direct call (검색 없음)
  - [ ] 성능 측정 (Judge Score < 6.0 예상)

### 7.2 Phase별 체크포인트

#### **Phase 0 완료 기준**
- [ ] Feature flag 시스템 동작
- [ ] Instrumentation 자동 수집 확인
- [ ] Baseline agent 실행 성공

#### **Phase 1 완료 기준**
- [ ] 청크 생성 완료 (타입별 전략 적용)
- [ ] 임베딩 생성 완료 (FAISS 인덱스 구축)
- [ ] Recall@5 > 0.75 달성

#### **Phase 2 완료 기준**
- [ ] Hybrid retrieval 동작
- [ ] MRR > 0.70 달성
- [ ] Retrieval ablation 실험 완료

#### **Phase 3 완료 기준**
- [ ] Self-Refine loop 동작
- [ ] Judge Score > 8.0 달성
- [ ] Generation ablation 실험 완료

### 7.3 최종 검증

- [ ] **재현성 검증**
  - [ ] 동일 seed로 3회 실행 → 동일 결과
  - [ ] 다른 환경에서 실행 → 동일 결과

- [ ] **Ablation 완전성**
  - [ ] 모든 주요 컴포넌트에 대한 ablation 완료
  - [ ] 각 기능의 marginal contribution 측정 완료

- [ ] **문서화**
  - [ ] 코드 주석 완성
  - [ ] README 작성
  - [ ] 실험 결과 리포트 작성

---

## 8. 예상 결과 및 기대 효과

### 8.1 정량적 목표

| 메트릭 | Baseline | After Phase 1 | After Phase 3 | 목표 |
|-------|---------|---------------|---------------|------|
| **Recall@5** | 0.50 | 0.75 (+50%) | 0.80 (+60%) | > 0.75 |
| **MRR** | 0.45 | 0.65 (+44%) | 0.72 (+60%) | > 0.70 |
| **Judge Score** | 5.5/10 | 7.0/10 (+27%) | 8.5/10 (+55%) | > 8.0 |
| **Hallucination Rate** | 35% | 20% (-43%) | 10% (-71%) | < 15% |
| **Latency (p95)** | 2.5s | 3.5s (+40%) | 4.5s (+80%) | < 5.0s |

### 8.2 Ablation 연구 기여도

**예상 발견**:
1. **데이터 레이어가 40-50% 성능 결정** (가장 큰 영향)
2. **Self-Refine이 20-30% 품질 향상** (두 번째 영향)
3. **Hybrid retrieval이 10-15% 개선** (세 번째 영향)
4. **Context engineering이 5-10% 개선** (부가 효과)

**논문 기여**:
- ✅ 의학 도메인 RAG의 체계적 ablation 연구
- ✅ 청킹 전략이 성능에 미치는 영향 정량화
- ✅ Self-Refine의 효과 실증
- ✅ 재현 가능한 오픈소스 프레임워크 제공

---

## 9. 참고 자료

### 9.1 관련 논문

1. **RAG 기초**
   - Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   
2. **Chunking 전략**
   - Gao et al. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels"
   
3. **Self-Refine**
   - Madaan et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback"
   
4. **의학 도메인 RAG**
   - Xiong et al. (2024). "Benchmarking Retrieval-Augmented Generation for Medicine"

### 9.2 코드 참고

- **LangGraph**: https://github.com/langchain-ai/langgraph
- **FAISS**: https://github.com/facebookresearch/faiss
- **Ragas**: https://github.com/explodinggradients/ragas (평가 메트릭)

---

## 10. 결론

### 10.1 핵심 전략 요약

```
1. 데이터부터 시작 (Bottom-Up)
   → 청킹/임베딩 최적화가 최우선

2. 모듈성 확보 (Modularity)
   → 모든 컴포넌트 독립적으로 on/off

3. 측정 자동화 (Instrumentation)
   → 실험 재현성 보장

4. 점진적 구축 (Progressive)
   → Baseline → Basic → Advanced 순서

5. 체계적 Ablation (Systematic)
   → 단일 변수 테스트 → 누적 효과 측정
```

### 10.2 성공 기준

**최소 목표** (논문 게재 가능):
- ✅ Recall@5 > 0.75
- ✅ Judge Score > 8.0
- ✅ 5개 이상 주요 컴포넌트 ablation 완료
- ✅ 재현 가능한 코드 및 데이터 공개

**이상적 목표** (우수 논문):
- ✅ Recall@5 > 0.80
- ✅ Judge Score > 8.5
- ✅ Hallucination Rate < 10%
- ✅ 10개 이상 컴포넌트 ablation 완료
- ✅ Full factorial experiment 완료

### 10.3 다음 단계

이 문서를 기반으로:
1. **즉시 시작**: Phase 0 (Feature flag 시스템)
2. **1주 내**: Phase 1 (데이터 레이어) 착수
3. **1개월 내**: Phase 3까지 완료 (Self-Refine)
4. **2개월 내**: 전체 시스템 완성 및 논문 작성

---

**문서 버전**: 1.0  
**최종 수정**: 2025-12-15  
**작성자**: Medical AI Agent Research Team

**라이선스**: MIT License

---

**END OF DOCUMENT**

