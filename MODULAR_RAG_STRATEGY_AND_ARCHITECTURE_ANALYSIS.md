# Modular RAG 전략 및 아키텍처 종합 분석

**작성일**: 2025년 12월 15일  
**목적**: Basic RAG, Modular RAG, Corrective RAG 모듈식 전환 및 LangGraph 아키텍처 평가

---

## 📋 목차

1. [Modular RAG 개요 및 전략](#1-modular-rag-개요-및-전략)
2. [선행 작업 완전 체크리스트](#2-선행-작업-완전-체크리스트)
3. [RAG 변형별 구현 요구사항](#3-rag-변형별-구현-요구사항)
4. [LangGraph vs 대안 아키텍처 심층 분석](#4-langgraph-vs-대안-아키텍처-심층-분석)
5. [최종 권장사항 및 로드맵](#5-최종-권장사항-및-로드맵)

---

## 1. Modular RAG 개요 및 전략

### 1.1 RAG 진화 단계

```
Generation 1: Basic RAG (2020-2021)
  └─ Query → Retrieve → Generate
     단순 파이프라인, 고정된 흐름

Generation 2: Advanced RAG (2022-2023)
  └─ Pre-retrieval + Retrieval + Post-retrieval
     쿼리 재작성, 하이브리드 검색, 리랭킹

Generation 3: Modular RAG (2023-2024) ⭐
  └─ Pluggable Modules + Orchestration
     모듈 조합 자유, 동적 라우팅, 적응형 처리

Generation 4: Agentic RAG (2024-) 🚀
  └─ Self-Reflection + Tool Use + Planning
     자율 에이전트, 도구 활용, 멀티 스텝 추론
```

### 1.2 Modular RAG의 핵심 개념

**정의**: RAG 시스템을 독립적인 모듈로 분해하여 조합 가능하게 만드는 설계 패러다임

**핵심 원칙**:
1. **Module Independence**: 각 모듈은 독립적으로 교체 가능
2. **Interface Standardization**: 표준화된 입출력 인터페이스
3. **Dynamic Composition**: 런타임에 모듈 조합 변경 가능
4. **Pluggability**: 새로운 모듈 추가가 용이

**모듈 분류**:

```
┌─────────────────────────────────────────────────┐
│           Modular RAG Architecture              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │  Pre-Retrieval Modules                  │   │
│  ├─────────────────────────────────────────┤   │
│  │  - Query Rewriter                       │   │
│  │  - Query Decomposer                     │   │
│  │  - Query Router                         │   │
│  │  - HyDE (Hypothetical Document)        │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐   │
│  │  Retrieval Modules                      │   │
│  ├─────────────────────────────────────────┤   │
│  │  - Dense Retriever (FAISS)              │   │
│  │  - Sparse Retriever (BM25)              │   │
│  │  - Hybrid Retriever (RRF)               │   │
│  │  - Graph Retriever (Neo4j)              │   │
│  │  - SQL Retriever (Structured Data)      │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐   │
│  │  Post-Retrieval Modules                 │   │
│  ├─────────────────────────────────────────┤   │
│  │  - Reranker (Cross-encoder)             │   │
│  │  - Compressor (LLMLingua)               │   │
│  │  - Filter (Relevance Threshold)         │   │
│  │  - Deduplicator                         │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐   │
│  │  Generation Modules                     │   │
│  ├─────────────────────────────────────────┤   │
│  │  - Generator (LLM)                      │   │
│  │  - Summarizer                           │   │
│  │  - Fact Checker                         │   │
│  │  - Citation Adder                       │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐   │
│  │  Evaluation & Correction Modules        │   │
│  ├─────────────────────────────────────────┤   │
│  │  - Quality Evaluator                    │   │
│  │  - Hallucination Detector               │   │
│  │  - Self-Refine Loop                     │   │
│  │  - Feedback Collector                   │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 1.3 Basic RAG vs Modular RAG vs Corrective RAG 비교

| 특성 | Basic RAG | Modular RAG | Corrective RAG (CRAG) |
|-----|-----------|-------------|----------------------|
| **구조** | 고정 파이프라인 | 모듈 조합 | 평가 + 교정 루프 |
| **복잡도** | 낮음 ⭐ | 중간 ⭐⭐ | 높음 ⭐⭐⭐ |
| **유연성** | 없음 | 높음 | 중간 |
| **성능** | 기본 | 향상 (+10-20%) | 최고 (+20-40%) |
| **구현 난이도** | 쉬움 | 중간 | 어려움 |
| **유지보수** | 어려움 | 쉬움 | 중간 |
| **Ablation 용이성** | 낮음 | 매우 높음 ⭐⭐⭐ | 중간 |

**Basic RAG**:
```python
def basic_rag(query):
    docs = retrieve(query, k=5)
    answer = generate(query, docs)
    return answer
```

**Modular RAG**:
```python
def modular_rag(query, modules):
    # 동적 모듈 조합
    query = modules['pre_retrieval'](query)
    docs = modules['retrieval'](query)
    docs = modules['post_retrieval'](docs)
    answer = modules['generation'](query, docs)
    answer = modules['evaluation'](answer)
    return answer
```

**Corrective RAG (CRAG)**:
```python
def corrective_rag(query, max_iterations=3):
    for i in range(max_iterations):
        docs = retrieve(query, k=10)
        
        # 문서 품질 평가
        relevance_scores = evaluate_relevance(query, docs)
        
        if all(score > threshold for score in relevance_scores):
            # 모든 문서가 관련성 높음
            answer = generate(query, docs)
            break
        elif any(score > threshold for score in relevance_scores):
            # 일부 문서만 관련성 높음
            filtered_docs = filter_by_score(docs, relevance_scores)
            answer = generate(query, filtered_docs)
            break
        else:
            # 모든 문서 관련성 낮음 → 웹 검색 또는 쿼리 재작성
            query = rewrite_query(query, docs)
            if i == max_iterations - 1:
                answer = generate_with_web_search(query)
    
    return answer
```

---

## 2. 선행 작업 완전 체크리스트

### 2.1 Phase 0: 아키텍처 설계 (1-2일)

#### ✅ 모듈 인터페이스 정의

```python
# core/module_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class RAGContext:
    """모든 모듈 간 공유되는 컨텍스트"""
    query: str
    original_query: str
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    metadata: Dict[str, Any]
    iteration: int

class RAGModule(ABC):
    """모든 RAG 모듈의 기본 인터페이스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, context: RAGContext) -> RAGContext:
        """
        모듈 실행 메서드
        
        Args:
            context: 현재 RAG 컨텍스트
        
        Returns:
            업데이트된 RAG 컨텍스트
        """
        pass
    
    def validate_input(self, context: RAGContext) -> bool:
        """입력 검증"""
        return True
    
    def validate_output(self, context: RAGContext) -> bool:
        """출력 검증"""
        return True
```

#### ✅ 모듈 레지스트리 구축

```python
# core/module_registry.py
from typing import Dict, Type, List
from core.module_interface import RAGModule

class ModuleRegistry:
    """모듈 등록 및 관리"""
    
    def __init__(self):
        self._modules: Dict[str, Type[RAGModule]] = {}
        self._instances: Dict[str, RAGModule] = {}
    
    def register(self, name: str, module_class: Type[RAGModule]):
        """모듈 클래스 등록"""
        self._modules[name] = module_class
        print(f"[Registry] 모듈 등록: {name}")
    
    def create(self, name: str, config: Dict) -> RAGModule:
        """모듈 인스턴스 생성"""
        if name not in self._modules:
            raise ValueError(f"Unknown module: {name}")
        
        instance = self._modules[name](config)
        self._instances[name] = instance
        return instance
    
    def get(self, name: str) -> RAGModule:
        """등록된 모듈 인스턴스 가져오기"""
        if name not in self._instances:
            raise ValueError(f"Module not instantiated: {name}")
        return self._instances[name]
    
    def list_modules(self) -> List[str]:
        """등록된 모듈 목록"""
        return list(self._modules.keys())

# 글로벌 레지스트리
registry = ModuleRegistry()
```

#### ✅ 파이프라인 오케스트레이터 설계

```python
# core/pipeline.py
from typing import List, Dict, Any, Callable
from core.module_interface import RAGModule, RAGContext
from core.module_registry import registry

class RAGPipeline:
    """모듈식 RAG 파이프라인"""
    
    def __init__(self, name: str):
        self.name = name
        self.modules: List[RAGModule] = []
        self.conditional_branches: Dict[str, Callable] = {}
    
    def add_module(self, module_name: str, config: Dict = None):
        """파이프라인에 모듈 추가"""
        config = config or {}
        module = registry.create(module_name, config)
        self.modules.append(module)
        return self
    
    def add_conditional(self, condition_fn: Callable, 
                       true_modules: List[str], 
                       false_modules: List[str]):
        """조건부 분기 추가"""
        self.conditional_branches[condition_fn.__name__] = {
            'condition': condition_fn,
            'true': true_modules,
            'false': false_modules
        }
        return self
    
    def execute(self, query: str, **kwargs) -> RAGContext:
        """파이프라인 실행"""
        context = RAGContext(
            query=query,
            original_query=query,
            retrieved_docs=[],
            generated_answer='',
            metadata=kwargs,
            iteration=0
        )
        
        for module in self.modules:
            print(f"[Pipeline] 실행: {module.name}")
            
            # 입력 검증
            if not module.validate_input(context):
                raise ValueError(f"Invalid input for {module.name}")
            
            # 모듈 실행
            context = module.execute(context)
            
            # 출력 검증
            if not module.validate_output(context):
                raise ValueError(f"Invalid output from {module.name}")
        
        return context
```

### 2.2 Phase 1: 데이터 레이어 준비 (이미 설계됨, 재확인 필요)

#### ✅ 체크리스트

- [ ] **청킹 전략 구현 완료**
  - [ ] TypeAwareChunker 구현 (180-400 tokens)
  - [ ] 문서 타입 분류기 (drug/guideline/case/general)
  - [ ] 메타데이터 추출 (source, confidence, section)

- [ ] **임베딩 생성 완료**
  - [ ] text-embedding-3-large 사용
  - [ ] 배치 임베딩 (batch_size=128)
  - [ ] L2 정규화 적용

- [ ] **듀얼 인덱스 구축 완료**
  - [ ] Fine-grained index (< 300 tokens)
  - [ ] Coarse-grained index (≥ 300 tokens)
  - [ ] BM25 인덱스 (전체 청크)

**선행 작업 확인**:
```bash
# 1. 청크 품질 검증
python scripts/validate_chunks.py \
  --input data/corpus_v2/train_source/chunks.jsonl \
  --check_size --check_metadata

# 2. 인덱스 무결성 검사
python scripts/validate_index.py \
  --fine_index data/index_v2/train_source/fine.index.faiss \
  --coarse_index data/index_v2/train_source/coarse.index.faiss

# 3. 검색 성능 베이스라인
python scripts/measure_retrieval_baseline.py
```

### 2.3 Phase 2: Modular RAG 핵심 모듈 구현 (1-2주)

#### ✅ Pre-Retrieval 모듈

**1. Query Rewriter Module**

```python
# modules/pre_retrieval/query_rewriter.py
from core.module_interface import RAGModule, RAGContext
from core.llm_client import LLMClient

class QueryRewriterModule(RAGModule):
    """쿼리 재작성 모듈"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMClient()
        self.strategy = config.get('strategy', 'expansion')  # expansion/simplification/medical_focus
    
    def execute(self, context: RAGContext) -> RAGContext:
        if self.strategy == 'expansion':
            rewritten = self._expand_query(context.query)
        elif self.strategy == 'simplification':
            rewritten = self._simplify_query(context.query)
        elif self.strategy == 'medical_focus':
            rewritten = self._add_medical_context(context.query, context.metadata)
        
        context.query = rewritten
        context.metadata['rewritten_queries'] = context.metadata.get('rewritten_queries', [])
        context.metadata['rewritten_queries'].append(rewritten)
        
        return context
    
    def _expand_query(self, query: str) -> str:
        """쿼리 확장 (동의어, 관련 용어 추가)"""
        prompt = f"""
        Expand this medical query by adding synonyms and related terms.
        Keep it concise and focused.
        
        Original query: {query}
        
        Expanded query:
        """
        return self.llm.generate(prompt, temperature=0.3, max_tokens=100).strip()
    
    def _simplify_query(self, query: str) -> str:
        """복잡한 쿼리 단순화"""
        prompt = f"""
        Simplify this medical query to its core question.
        
        Original query: {query}
        
        Simplified query:
        """
        return self.llm.generate(prompt, temperature=0.2, max_tokens=50).strip()
    
    def _add_medical_context(self, query: str, metadata: Dict) -> str:
        """환자 프로필 반영"""
        profile = metadata.get('patient_profile', {})
        if not profile:
            return query
        
        context_parts = []
        if profile.get('age'):
            context_parts.append(f"Patient age: {profile['age']}")
        if profile.get('conditions'):
            context_parts.append(f"Conditions: {', '.join(profile['conditions'])}")
        if profile.get('medications'):
            context_parts.append(f"Current medications: {', '.join(profile['medications'])}")
        
        if context_parts:
            context_str = ". ".join(context_parts)
            return f"{query}. Context: {context_str}"
        
        return query
```

**2. Query Decomposer Module**

```python
# modules/pre_retrieval/query_decomposer.py
from core.module_interface import RAGModule, RAGContext
from core.llm_client import LLMClient
from typing import List

class QueryDecomposerModule(RAGModule):
    """복잡한 쿼리를 하위 질문으로 분해"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMClient()
        self.max_subqueries = config.get('max_subqueries', 3)
    
    def execute(self, context: RAGContext) -> RAGContext:
        # 복잡도 평가
        if not self._is_complex(context.query):
            return context
        
        # 하위 질문 생성
        subqueries = self._decompose(context.query)
        
        context.metadata['subqueries'] = subqueries
        context.metadata['is_decomposed'] = True
        
        return context
    
    def _is_complex(self, query: str) -> bool:
        """쿼리 복잡도 판단"""
        # 간단한 휴리스틱: 단어 수, 'and' 개수 등
        words = query.split()
        has_conjunction = any(word.lower() in ['and', 'or', 'also'] for word in words)
        return len(words) > 15 or has_conjunction
    
    def _decompose(self, query: str) -> List[str]:
        """쿼리 분해"""
        prompt = f"""
        Break down this complex medical question into {self.max_subqueries} simpler sub-questions.
        Each sub-question should be independently answerable.
        
        Complex question: {query}
        
        Sub-questions (one per line):
        """
        
        response = self.llm.generate(prompt, temperature=0.3, max_tokens=200)
        subqueries = [line.strip() for line in response.split('\n') if line.strip()]
        return subqueries[:self.max_subqueries]
```

**3. HyDE (Hypothetical Document Embeddings) Module**

```python
# modules/pre_retrieval/hyde.py
from core.module_interface import RAGModule, RAGContext
from core.llm_client import LLMClient

class HyDEModule(RAGModule):
    """가상 문서 생성 후 임베딩으로 검색"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMClient()
    
    def execute(self, context: RAGContext) -> RAGContext:
        # 가상 답변 생성
        hypothetical_doc = self._generate_hypothetical_answer(context.query)
        
        # 원본 쿼리 대신 가상 문서로 검색
        context.metadata['hypothetical_doc'] = hypothetical_doc
        context.metadata['use_hyde'] = True
        
        return context
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """가상의 이상적인 답변 생성"""
        prompt = f"""
        Generate a hypothetical, ideal answer to this medical question.
        Write as if you're citing from a medical textbook.
        
        Question: {query}
        
        Hypothetical answer:
        """
        return self.llm.generate(prompt, temperature=0.5, max_tokens=300).strip()
```

#### ✅ Retrieval 모듈 (이미 구현됨, 모듈화 필요)

```python
# modules/retrieval/hybrid_retrieval.py
from core.module_interface import RAGModule, RAGContext
from retrieval.dual_retriever import DualIndexRetriever

class HybridRetrievalModule(RAGModule):
    """하이브리드 검색 모듈"""
    
    def __init__(self, config):
        super().__init__(config)
        self.retriever = DualIndexRetriever(
            index_dir=config['index_dir'],
            embedding_model=config.get('embedding_model', 'text-embedding-3-large')
        )
        self.k_fine = config.get('k_fine', 12)
        self.k_coarse = config.get('k_coarse', 5)
    
    def execute(self, context: RAGContext) -> RAGContext:
        # HyDE 사용 여부 확인
        if context.metadata.get('use_hyde'):
            query = context.metadata['hypothetical_doc']
        else:
            query = context.query
        
        # 검색 실행
        docs = self.retriever.search(
            query=query,
            k_fine=self.k_fine,
            k_coarse=self.k_coarse,
            route='both'
        )
        
        context.retrieved_docs = docs
        context.metadata['num_retrieved'] = len(docs)
        
        return context
```

#### ✅ Post-Retrieval 모듈

**1. Reranker Module**

```python
# modules/post_retrieval/reranker.py
from core.module_interface import RAGModule, RAGContext
from sentence_transformers import CrossEncoder

class RerankerModule(RAGModule):
    """교차 인코더 기반 리랭킹"""
    
    def __init__(self, config):
        super().__init__(config)
        model_name = config.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.reranker = CrossEncoder(model_name)
        self.top_k = config.get('top_k', 5)
    
    def execute(self, context: RAGContext) -> RAGContext:
        docs = context.retrieved_docs
        
        if len(docs) <= self.top_k:
            return context
        
        # 쿼리-문서 쌍 생성
        pairs = [(context.query, doc['text']) for doc in docs]
        
        # 리랭킹 점수 계산
        scores = self.reranker.predict(pairs)
        
        # 점수로 정렬
        for doc, score in zip(docs, scores):
            doc['rerank_score'] = float(score)
        
        docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # 상위 k개만 유지
        context.retrieved_docs = docs[:self.top_k]
        context.metadata['reranked'] = True
        
        return context
```

**2. Relevance Filter Module**

```python
# modules/post_retrieval/relevance_filter.py
from core.module_interface import RAGModule, RAGContext
from core.llm_client import LLMClient

class RelevanceFilterModule(RAGModule):
    """관련성 낮은 문서 필터링 (Corrective RAG 핵심)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMClient()
        self.threshold = config.get('threshold', 0.5)
        self.use_llm = config.get('use_llm', True)
    
    def execute(self, context: RAGContext) -> RAGContext:
        docs = context.retrieved_docs
        
        if self.use_llm:
            relevance_scores = self._evaluate_with_llm(context.query, docs)
        else:
            relevance_scores = self._evaluate_heuristic(context.query, docs)
        
        # 점수 추가
        for doc, score in zip(docs, relevance_scores):
            doc['relevance_score'] = score
        
        # 필터링
        filtered_docs = [doc for doc in docs if doc['relevance_score'] >= self.threshold]
        
        context.retrieved_docs = filtered_docs
        context.metadata['filtered_count'] = len(docs) - len(filtered_docs)
        context.metadata['all_irrelevant'] = len(filtered_docs) == 0
        
        return context
    
    def _evaluate_with_llm(self, query: str, docs: List[Dict]) -> List[float]:
        """LLM 기반 관련성 평가"""
        scores = []
        
        for doc in docs:
            prompt = f"""
            Rate the relevance of this document to the query on a scale of 0-1.
            Only return a number.
            
            Query: {query}
            Document: {doc['text'][:500]}
            
            Relevance score:
            """
            
            try:
                score_str = self.llm.generate(prompt, temperature=0.0, max_tokens=10).strip()
                score = float(score_str)
                scores.append(max(0.0, min(1.0, score)))
            except:
                scores.append(0.5)  # 기본값
        
        return scores
    
    def _evaluate_heuristic(self, query: str, docs: List[Dict]) -> List[float]:
        """휴리스틱 기반 관련성 평가"""
        # 간단한 키워드 매칭
        query_keywords = set(query.lower().split())
        scores = []
        
        for doc in docs:
            doc_keywords = set(doc['text'].lower().split())
            overlap = len(query_keywords & doc_keywords)
            score = overlap / len(query_keywords) if query_keywords else 0.0
            scores.append(min(1.0, score))
        
        return scores
```

#### ✅ Generation 모듈

```python
# modules/generation/generator.py
from core.module_interface import RAGModule, RAGContext
from core.llm_client import LLMClient

class GeneratorModule(RAGModule):
    """LLM 기반 답변 생성"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMClient()
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 800)
    
    def execute(self, context: RAGContext) -> RAGContext:
        # 컨텍스트 조립
        context_text = self._assemble_context(context.retrieved_docs)
        
        # 프롬프트 생성
        prompt = self._build_prompt(context.query, context_text, context.metadata)
        
        # 답변 생성
        answer = self.llm.generate(
            prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        context.generated_answer = answer
        
        return context
    
    def _assemble_context(self, docs: List[Dict]) -> str:
        """검색된 문서를 컨텍스트로 조립"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, metadata: Dict) -> str:
        """프롬프트 구성"""
        prompt = f"""You are a medical AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Provide accurate, evidence-based information
- Cite document numbers when referencing information
- If information is insufficient, state clearly
- Use professional medical terminology

Answer:"""
        
        return prompt
```

#### ✅ Evaluation & Correction 모듈

```python
# modules/evaluation/quality_evaluator.py
from core.module_interface import RAGModule, RAGContext
from core.llm_client import LLMClient

class QualityEvaluatorModule(RAGModule):
    """답변 품질 평가 (Self-Refine용)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMClient()
        self.threshold = config.get('threshold', 0.5)
    
    def execute(self, context: RAGContext) -> RAGContext:
        # 품질 점수 계산
        quality_score = self._evaluate_quality(
            context.query,
            context.generated_answer,
            context.retrieved_docs
        )
        
        context.metadata['quality_score'] = quality_score
        context.metadata['needs_refinement'] = quality_score < self.threshold
        
        return context
    
    def _evaluate_quality(self, query: str, answer: str, docs: List[Dict]) -> float:
        """LLM 기반 품질 평가"""
        prompt = f"""
        Evaluate the quality of this answer on a scale of 0-1.
        Consider: relevance, accuracy, completeness, clarity.
        
        Question: {query}
        Answer: {answer}
        
        Available context: {len(docs)} documents
        
        Quality score (0-1):
        """
        
        try:
            score_str = self.llm.generate(prompt, temperature=0.0, max_tokens=10).strip()
            return float(score_str)
        except:
            return 0.5
```

### 2.4 Phase 3: RAG 변형 파이프라인 구성 (3-5일)

#### ✅ Basic RAG Pipeline

```python
# pipelines/basic_rag.py
from core.pipeline import RAGPipeline
from core.module_registry import registry

def build_basic_rag_pipeline():
    """기본 RAG 파이프라인"""
    pipeline = RAGPipeline('basic_rag')
    
    # 단순 파이프라인: Retrieve → Generate
    pipeline.add_module('hybrid_retrieval', {
        'index_dir': 'data/index_v2/train_source',
        'k_fine': 8,
        'k_coarse': 3
    })
    
    pipeline.add_module('generator', {
        'model': 'gpt-4o-mini',
        'temperature': 0.2
    })
    
    return pipeline
```

#### ✅ Modular RAG Pipeline

```python
# pipelines/modular_rag.py
from core.pipeline import RAGPipeline

def build_modular_rag_pipeline(config):
    """모듈형 RAG 파이프라인 (동적 구성)"""
    pipeline = RAGPipeline('modular_rag')
    
    # Pre-retrieval
    if config.get('use_query_rewriter'):
        pipeline.add_module('query_rewriter', {
            'strategy': config.get('rewrite_strategy', 'medical_focus')
        })
    
    if config.get('use_query_decomposer'):
        pipeline.add_module('query_decomposer', {
            'max_subqueries': 3
        })
    
    if config.get('use_hyde'):
        pipeline.add_module('hyde', {})
    
    # Retrieval
    pipeline.add_module('hybrid_retrieval', {
        'index_dir': config['index_dir'],
        'k_fine': config.get('k_fine', 12),
        'k_coarse': config.get('k_coarse', 5)
    })
    
    # Post-retrieval
    if config.get('use_reranker'):
        pipeline.add_module('reranker', {
            'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'top_k': config.get('rerank_top_k', 5)
        })
    
    if config.get('use_relevance_filter'):
        pipeline.add_module('relevance_filter', {
            'threshold': config.get('relevance_threshold', 0.5),
            'use_llm': config.get('llm_filter', True)
        })
    
    # Generation
    pipeline.add_module('generator', {
        'model': config.get('llm_model', 'gpt-4o-mini'),
        'temperature': config.get('temperature', 0.2)
    })
    
    # Evaluation
    if config.get('use_quality_evaluator'):
        pipeline.add_module('quality_evaluator', {
            'threshold': config.get('quality_threshold', 0.5)
        })
    
    return pipeline
```

#### ✅ Corrective RAG (CRAG) Pipeline

```python
# pipelines/corrective_rag.py
from core.pipeline import RAGPipeline

def build_corrective_rag_pipeline():
    """교정 RAG 파이프라인 (Self-Refine 루프 포함)"""
    pipeline = RAGPipeline('corrective_rag')
    
    # 1. Initial Retrieval
    pipeline.add_module('hybrid_retrieval', {
        'index_dir': 'data/index_v2/train_source',
        'k_fine': 15,  # 더 많이 검색
        'k_coarse': 8
    })
    
    # 2. Relevance Filtering (핵심!)
    pipeline.add_module('relevance_filter', {
        'threshold': 0.6,  # 높은 임계값
        'use_llm': True
    })
    
    # 3. Conditional: 모든 문서 관련성 낮으면 웹 검색 또는 쿼리 재작성
    def check_relevance(context):
        return not context.metadata.get('all_irrelevant', False)
    
    pipeline.add_conditional(
        condition_fn=check_relevance,
        true_modules=['generator'],  # 관련 문서 있음 → 생성
        false_modules=['query_rewriter', 'hybrid_retrieval', 'generator']  # 재검색
    )
    
    # 4. Quality Evaluation
    pipeline.add_module('quality_evaluator', {
        'threshold': 0.6
    })
    
    # 5. Self-Refine Loop (최대 2회)
    # (LangGraph에서 구현하는 것이 더 적합)
    
    return pipeline
```

### 2.5 Phase 4: Ablation 실험 설계 (2-3일)

#### ✅ RAG 변형 비교 실험

```python
# experiments/rag_variants_ablation.py
"""RAG 변형 비교 실험"""

EXPERIMENTS = {
    'E1_basic_rag': {
        'pipeline': 'basic_rag',
        'config': {
            'k_fine': 8,
            'k_coarse': 3
        }
    },
    
    'E2_modular_rag_minimal': {
        'pipeline': 'modular_rag',
        'config': {
            'use_query_rewriter': True,
            'rewrite_strategy': 'medical_focus',
            'use_reranker': False,
            'use_relevance_filter': False
        }
    },
    
    'E3_modular_rag_full': {
        'pipeline': 'modular_rag',
        'config': {
            'use_query_rewriter': True,
            'use_query_decomposer': False,
            'use_hyde': False,
            'use_reranker': True,
            'use_relevance_filter': True,
            'use_quality_evaluator': True
        }
    },
    
    'E4_corrective_rag': {
        'pipeline': 'corrective_rag',
        'config': {
            'max_refine_iterations': 2,
            'relevance_threshold': 0.6,
            'quality_threshold': 0.6
        }
    },
    
    'E5_modular_with_hyde': {
        'pipeline': 'modular_rag',
        'config': {
            'use_hyde': True,
            'use_reranker': True,
            'rerank_top_k': 5
        }
    },
}

def run_ablation_experiment(exp_id: str, test_cases: List[Dict]):
    """단일 실험 실행"""
    exp_config = EXPERIMENTS[exp_id]
    
    # 파이프라인 구축
    if exp_config['pipeline'] == 'basic_rag':
        pipeline = build_basic_rag_pipeline()
    elif exp_config['pipeline'] == 'modular_rag':
        pipeline = build_modular_rag_pipeline(exp_config['config'])
    elif exp_config['pipeline'] == 'corrective_rag':
        pipeline = build_corrective_rag_pipeline()
    
    results = []
    
    for case in test_cases:
        # 실행
        context = pipeline.execute(
            query=case['query'],
            patient_profile=case.get('profile', {})
        )
        
        # 메트릭 수집
        metrics = {
            'query': case['query'],
            'answer': context.generated_answer,
            'num_docs': len(context.retrieved_docs),
            'quality_score': context.metadata.get('quality_score', 0.0),
            'iteration_count': context.iteration,
            # ... 추가 메트릭
        }
        
        results.append(metrics)
    
    return results
```

---

## 3. RAG 변형별 구현 요구사항

### 3.1 Basic RAG 요구사항

**필수 컴포넌트**:
- [x] Retriever (BM25 또는 FAISS)
- [x] Generator (LLM)

**선택 컴포넌트**:
- [ ] Query preprocessing (선택)
- [ ] Context assembly (선택)

**구현 난이도**: ⭐ (쉬움)

**예상 성능**:
- Recall@5: 0.60-0.70
- Judge Score: 6.5-7.5/10

### 3.2 Modular RAG 요구사항

**필수 컴포넌트**:
- [x] Module Interface (RAGModule 추상 클래스)
- [x] Module Registry (모듈 등록 시스템)
- [x] Pipeline Orchestrator (모듈 조합 관리)
- [x] 최소 3개 모듈 (Pre-retrieval, Retrieval, Generation)

**권장 컴포넌트**:
- [ ] Query Rewriter (쿼리 개선)
- [ ] Reranker (검색 결과 재정렬)
- [ ] Quality Evaluator (품질 평가)

**구현 난이도**: ⭐⭐ (중간)

**예상 성능**:
- Recall@5: 0.70-0.80 (+10-15%p vs Basic)
- Judge Score: 7.5-8.5/10

**선행 작업**:
1. ✅ 모듈 인터페이스 정의 (Phase 0)
2. ✅ 레지스트리 구축 (Phase 0)
3. ✅ 파이프라인 오케스트레이터 (Phase 0)
4. ✅ 최소 5-7개 모듈 구현 (Phase 2)

### 3.3 Corrective RAG (CRAG) 요구사항

**필수 컴포넌트**:
- [x] Relevance Evaluator (문서 관련성 평가) ⭐ 핵심!
- [x] Conditional Branching (조건부 분기)
- [x] Query Rewriter (재작성)
- [x] Web Search Fallback (선택적)
- [x] Self-Refine Loop (반복 개선)

**구현 난이도**: ⭐⭐⭐ (어려움)

**예상 성능**:
- Recall@5: 0.75-0.85 (+15-25%p vs Basic)
- Judge Score: 8.0-9.0/10
- Hallucination Rate: -30-50% vs Basic

**선행 작업**:
1. ✅ Relevance Filter Module 구현 (LLM 기반) ⭐ 최우선!
2. ✅ Conditional Pipeline 지원 (Phase 0)
3. ✅ Query Rewriter Module (Phase 2)
4. ✅ Quality Evaluator Module (Phase 2)
5. ✅ Self-Refine Loop (LangGraph에서 구현)

**CRAG 특화 요구사항**:

```python
# CRAG 핵심: 3단계 관련성 평가
def evaluate_document_relevance(query, docs):
    """
    문서 관련성을 3단계로 분류:
    - Correct: 높은 관련성 (score > 0.7)
    - Ambiguous: 중간 관련성 (0.4 < score <= 0.7)
    - Incorrect: 낮은 관련성 (score <= 0.4)
    """
    scores = []
    for doc in docs:
        score = llm_evaluate_relevance(query, doc)
        if score > 0.7:
            label = 'correct'
        elif score > 0.4:
            label = 'ambiguous'
        else:
            label = 'incorrect'
        scores.append({'doc': doc, 'score': score, 'label': label})
    
    return scores

def corrective_action(relevance_results):
    """
    관련성 평가 결과에 따른 교정 액션:
    - All Correct → Generate directly
    - Some Correct → Filter + Generate
    - All Incorrect → Web search OR Query rewrite
    """
    correct_docs = [r for r in relevance_results if r['label'] == 'correct']
    
    if len(correct_docs) == len(relevance_results):
        return 'generate', relevance_results
    elif len(correct_docs) > 0:
        return 'filter_and_generate', correct_docs
    else:
        return 'web_search_or_rewrite', []
```

---

## 4. LangGraph vs 대안 아키텍처 심층 분석

### 4.1 LangGraph 상세 분석

#### **핵심 특징**

1. **State Graph 기반**
```python
# LangGraph의 핵심: StateGraph
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)

# 조건부 엣지 (Self-Refine 루프)
def should_refine(state):
    return state['quality_score'] < 0.5

graph.add_conditional_edges(
    "evaluate",
    should_refine,
    {
        True: "retrieve",  # 재검색
        False: END         # 종료
    }
)
```

2. **순환 그래프 지원**
```
┌─────────────────────────────────────────┐
│         LangGraph Cycle Support         │
├─────────────────────────────────────────┤
│                                         │
│  Retrieve → Generate → Evaluate         │
│      ↑                      ↓           │
│      └──────── (if low) ────┘           │
│                                         │
│  최대 N회 반복 가능 (max_iterations)    │
│  무한 루프 방지 내장                     │
└─────────────────────────────────────────┘
```

3. **중앙집중식 상태 관리**
```python
@dataclass
class AgentState:
    """모든 노드가 공유하는 상태"""
    query: str
    retrieved_docs: List[Dict]
    answer: str
    quality_score: float
    iteration_count: int
    # ... 모든 중간 결과 저장
```

#### **장점 (LangGraph를 유지해야 하는 이유)**

| 장점 | 설명 | 중요도 |
|-----|------|--------|
| **순환 로직 네이티브 지원** | Self-Refine, CRAG 같은 반복 패턴을 자연스럽게 표현 | ⭐⭐⭐⭐⭐ |
| **상태 관리 자동화** | 모든 노드 간 상태 자동 전달, 수동 관리 불필요 | ⭐⭐⭐⭐⭐ |
| **시각화 용이** | 그래프 구조를 Mermaid 등으로 자동 시각화 가능 | ⭐⭐⭐⭐ |
| **디버깅 편의성** | 각 노드의 입출력 추적 용이 | ⭐⭐⭐⭐ |
| **LangChain 생태계** | LangChain의 모든 컴포넌트 재사용 가능 | ⭐⭐⭐⭐ |
| **Checkpointing** | 중간 상태 저장 및 복구 지원 | ⭐⭐⭐ |
| **Human-in-the-loop** | 사람 개입 지점 쉽게 추가 가능 | ⭐⭐⭐ |

#### **단점**

| 단점 | 설명 | 영향도 |
|-----|------|--------|
| **학습 곡선** | StateGraph 개념 이해 필요 | ⭐⭐ |
| **오버헤드** | 단순 파이프라인에는 과도한 추상화 | ⭐⭐ |
| **성능** | 상태 복사 오버헤드 (대부분 무시 가능) | ⭐ |

#### **LangGraph가 특히 우수한 시나리오**

1. **Self-Refine Loop**
```python
# LangGraph: 자연스러운 표현
graph.add_conditional_edges(
    "evaluate",
    lambda state: state['quality_score'] < 0.5,
    {True: "rewrite_query", False: END}
)

# 대안 (수동 구현): 복잡하고 오류 가능성 높음
def manual_self_refine(query, max_iter=3):
    for i in range(max_iter):
        docs = retrieve(query)
        answer = generate(query, docs)
        quality = evaluate(answer)
        if quality >= 0.5:
            break
        query = rewrite(query, answer)
    return answer
```

2. **Corrective RAG의 조건부 분기**
```python
# LangGraph: 명확한 분기 표현
def route_by_relevance(state):
    if state['all_docs_irrelevant']:
        return "web_search"
    elif state['some_docs_relevant']:
        return "filter_and_generate"
    else:
        return "generate"

graph.add_conditional_edges("evaluate_relevance", route_by_relevance)
```

3. **멀티턴 대화 상태 관리**
```python
# LangGraph: 자동 상태 유지
@dataclass
class ConversationState:
    history: List[Dict]  # 자동으로 누적
    current_query: str
    context: Dict
    # ... 모든 턴의 정보 유지
```

### 4.2 대안 아키텍처 비교

#### **Option 1: LlamaIndex**

**특징**:
- 데이터 중심 프레임워크
- 다양한 인덱스 타입 지원 (Vector, Tree, Keyword)
- Query Engine 추상화

**장점**:
- 인덱싱 및 검색에 특화
- 다양한 데이터 소스 커넥터
- 간단한 API

**단점**:
- 순환 로직 지원 약함
- 복잡한 워크플로우 표현 어려움
- 상태 관리 수동

**비교**:
```python
# LlamaIndex: 단순 RAG에 적합
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is X?")

# LangGraph: 복잡한 워크플로우에 적합
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_node)
graph.add_conditional_edges("evaluate", should_refine)
```

**결론**: LlamaIndex는 Basic RAG에 적합, Modular/Corrective RAG에는 LangGraph가 우수

#### **Option 2: Haystack**

**특징**:
- 파이프라인 기반 프레임워크
- 노드와 파이프라인 개념
- 프로덕션 배포에 강점

**장점**:
- 명확한 파이프라인 구조
- REST API 자동 생성
- 프로덕션 도구 풍부

**단점**:
- 순환 파이프라인 지원 제한적
- 상태 관리 복잡
- LangChain 생태계 미지원

**비교**:
```python
# Haystack: 선형 파이프라인
pipeline = Pipeline()
pipeline.add_node("retriever", retriever, inputs=["Query"])
pipeline.add_node("reader", reader, inputs=["retriever"])

# LangGraph: 순환 가능
graph.add_conditional_edges("reader", should_refine, {
    True: "retriever",  # 순환!
    False: END
})
```

**결론**: Haystack은 프로덕션 배포에 강점, 연구/실험에는 LangGraph가 유연

#### **Option 3: Custom Implementation (순수 Python)**

**장점**:
- 완전한 제어
- 의존성 최소화
- 최적화 가능

**단점**:
- 개발 시간 증가
- 버그 가능성
- 유지보수 부담

**비교**:
```python
# Custom: 모든 것을 직접 구현
class CustomRAG:
    def __init__(self):
        self.state = {}
    
    def run(self, query):
        for i in range(self.max_iter):
            docs = self.retrieve(query)
            answer = self.generate(query, docs)
            quality = self.evaluate(answer)
            if quality >= self.threshold:
                break
            query = self.rewrite(query)
        return answer
    # ... 수백 줄의 상태 관리 코드

# LangGraph: 선언적 정의
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)
graph.add_conditional_edges("evaluate", should_refine)
# 끝!
```

**결론**: 특수한 요구사항이 없다면 LangGraph가 효율적

### 4.3 아키텍처 선택 매트릭스

| 요구사항 | LangGraph | LlamaIndex | Haystack | Custom |
|---------|-----------|------------|----------|--------|
| **Basic RAG** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Modular RAG** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Corrective RAG** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Self-Refine** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **순환 로직** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **상태 관리** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **학습 곡선** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **프로덕션** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Ablation 용이성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4.4 최종 판단: LangGraph 유지 권장

**결론**: **LangGraph를 유지하는 것을 강력히 권장합니다.**

**이유**:

1. **현재 프로젝트 요구사항과 완벽히 일치**
   - ✅ Corrective RAG 구현 필요 → LangGraph의 조건부 분기 필수
   - ✅ Self-Refine Loop → LangGraph의 순환 그래프 최적
   - ✅ Ablation 연구 → LangGraph의 모듈성이 이상적

2. **기존 코드베이스와의 호환성**
   - 현재 시스템이 이미 LangGraph 기반
   - 마이그레이션 비용 > 유지 비용

3. **연구 유연성**
   - 다양한 RAG 변형 실험에 최적
   - 노드 추가/제거가 매우 쉬움

4. **커뮤니티 및 생태계**
   - LangChain 생태계 활용 가능
   - 활발한 개발 및 업데이트

**아키텍처 변경이 필요한 경우**:
- ❌ 프로덕션 배포가 최우선 → Haystack 고려
- ❌ 단순 RAG만 필요 → LlamaIndex 고려
- ❌ 극한의 성능 최적화 필요 → Custom 고려

**현재 프로젝트에는 해당 없음!**

---

## 5. 최종 권장사항 및 로드맵

### 5.1 종합 권장사항

#### **1. 아키텍처: LangGraph 유지 ✅**

**이유**:
- Corrective RAG의 조건부 분기 구현에 최적
- Self-Refine Loop를 자연스럽게 표현
- 현재 코드베이스와 호환성 100%
- Ablation 연구에 이상적

**기대 효과**:
- 개발 시간 절약: 50-70% (vs 새 아키텍처)
- 안정성 유지: 기존 코드 재사용
- 유연성 확보: 실험 변형 용이

#### **2. Modular RAG 접근 채택 ✅**

**이유**:
- Basic, Modular, Corrective RAG 모두 지원
- 모듈 단위 Ablation 가능
- 점진적 개선 가능

**구현 전략**:
```
Week 1: 모듈 인터페이스 + 레지스트리
Week 2: 핵심 모듈 5-7개 구현
Week 3: 파이프라인 3종 구축
Week 4: Ablation 실험 및 분석
```

#### **3. 선행 작업 우선순위**

**P0 (필수, 1주)**:
- [ ] 모듈 인터페이스 정의 (RAGModule, RAGContext)
- [ ] 모듈 레지스트리 구축
- [ ] 파이프라인 오케스트레이터

**P1 (핵심, 2주)**:
- [ ] Query Rewriter Module
- [ ] Relevance Filter Module (CRAG 핵심!)
- [ ] Reranker Module
- [ ] Quality Evaluator Module

**P2 (고급, 1주)**:
- [ ] Query Decomposer Module
- [ ] HyDE Module
- [ ] Context Compressor Module

### 5.2 4주 구현 로드맵

#### **Week 1: Foundation (기반 구축)**

**Day 1-2: 아키텍처 설계**
- [ ] 모듈 인터페이스 정의
- [ ] 레지스트리 구현
- [ ] 파이프라인 오케스트레이터 기본 구현

**Day 3-4: LangGraph 통합**
- [ ] LangGraph와 모듈 시스템 통합
- [ ] StateGraph에서 모듈 호출 방식 설계
- [ ] 순환 로직 테스트

**Day 5-7: 첫 번째 파이프라인**
- [ ] Basic RAG Pipeline 구현
- [ ] 테스트 및 검증
- [ ] 베이스라인 성능 측정

**Deliverable**: 
- 동작하는 Basic RAG Pipeline
- 베이스라인 메트릭 (Recall@5, Judge Score)

#### **Week 2: Core Modules (핵심 모듈)**

**Day 8-10: Pre/Post-Retrieval**
- [ ] Query Rewriter Module
- [ ] Reranker Module
- [ ] Relevance Filter Module (CRAG용)

**Day 11-12: Generation & Evaluation**
- [ ] Generator Module (기존 코드 모듈화)
- [ ] Quality Evaluator Module

**Day 13-14: Modular RAG Pipeline**
- [ ] Modular RAG Pipeline 구축
- [ ] 모듈 조합 테스트
- [ ] 성능 측정

**Deliverable**:
- 5-7개 핵심 모듈
- Modular RAG Pipeline
- 성능 개선 확인 (+10-15%p)

#### **Week 3: Corrective RAG (교정 RAG)**

**Day 15-17: CRAG 구현**
- [ ] Relevance Evaluator 고도화
- [ ] Conditional Branching 구현
- [ ] Self-Refine Loop 통합

**Day 18-19: Web Search Fallback**
- [ ] 웹 검색 모듈 (선택적)
- [ ] Query Rewrite 전략 개선

**Day 20-21: CRAG Pipeline 완성**
- [ ] Corrective RAG Pipeline 구축
- [ ] 전체 시스템 테스트
- [ ] 성능 측정

**Deliverable**:
- Corrective RAG Pipeline
- 성능 개선 확인 (+20-30%p vs Basic)
- Hallucination Rate 감소 (-30-50%)

#### **Week 4: Ablation & Analysis (실험 및 분석)**

**Day 22-24: Ablation 실험**
- [ ] E1-E5 실험 실행
- [ ] 메트릭 자동 수집
- [ ] 결과 데이터 정리

**Day 25-26: 분석 및 시각화**
- [ ] 성능 비교 테이블
- [ ] 그래프 생성
- [ ] 통계 분석

**Day 27-28: 문서화**
- [ ] 실험 결과 리포트
- [ ] 논문용 표/그래프
- [ ] 코드 문서화

**Deliverable**:
- 전체 Ablation 결과
- 논문 초안 (Method + Results)
- 재현 가능한 코드

### 5.3 예상 성능 개선

```
┌─────────────────────────────────────────────────────────┐
│           Performance Improvement Roadmap               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Baseline (Current)                                     │
│  ├─ Recall@5: 0.65                                     │
│  ├─ Judge Score: 7.2/10                                │
│  └─ Hallucination: 35%                                 │
│                                                         │
│  ↓ Week 1: Basic RAG (확인)                            │
│                                                         │
│  Week 2: Modular RAG                                   │
│  ├─ Recall@5: 0.75 (+15%p) ⭐                         │
│  ├─ Judge Score: 7.8/10 (+0.6)                        │
│  └─ Hallucination: 25% (-29%)                         │
│                                                         │
│  ↓ +Query Rewriter, Reranker                          │
│                                                         │
│  Week 3: Corrective RAG                                │
│  ├─ Recall@5: 0.82 (+26%p) ⭐⭐                       │
│  ├─ Judge Score: 8.5/10 (+1.3) ⭐                     │
│  └─ Hallucination: 12% (-66%) ⭐⭐                    │
│                                                         │
│  ↓ +Relevance Filter, Self-Refine                     │
│                                                         │
│  Week 4: Optimized System                              │
│  ├─ Recall@5: 0.85 (+31%p) ⭐⭐⭐                     │
│  ├─ Judge Score: 8.8/10 (+1.6) ⭐⭐                   │
│  └─ Hallucination: 8% (-77%) ⭐⭐⭐                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.4 비용 및 리소스 추정

**개발 시간**: 4주 (하루 6-8시간 작업 기준)

**API 비용** (1,000 쿼리 실험 기준):
- Embedding (재생성 불필요): $0
- LLM 호출 (실험):
  - Basic RAG: ~$5-10
  - Modular RAG: ~$10-20 (Reranker, Evaluator 추가)
  - Corrective RAG: ~$20-30 (반복 호출)
- 총 예상: $50-100

**컴퓨팅 리소스**:
- CPU: 일반 노트북 충분
- RAM: 16GB 권장 (FAISS 인덱스 로드)
- GPU: 불필요 (Reranker는 CPU로 충분)

### 5.5 최종 체크리스트

#### **시작 전 확인**
- [ ] 기존 데이터 레이어 완성 (청킹, 듀얼 인덱스)
- [ ] LangGraph 기본 이해
- [ ] Python 3.9+ 환경
- [ ] API 키 설정 (OpenAI)

#### **Week 1 완료 기준**
- [ ] 모듈 인터페이스 동작
- [ ] Basic RAG Pipeline 실행 성공
- [ ] 베이스라인 메트릭 측정 완료

#### **Week 2 완료 기준**
- [ ] 5개 이상 모듈 구현
- [ ] Modular RAG Pipeline 동작
- [ ] 성능 개선 확인 (+10%p)

#### **Week 3 완료 기준**
- [ ] Corrective RAG Pipeline 동작
- [ ] Self-Refine Loop 정상 작동
- [ ] 성능 개선 확인 (+20%p)

#### **Week 4 완료 기준**
- [ ] 5개 이상 Ablation 실험 완료
- [ ] 결과 분석 및 시각화
- [ ] 논문 초안 작성

---

## 6. 결론

### 6.1 핵심 메시지

```
1. LangGraph 유지 ✅
   → Corrective RAG 구현에 최적
   → 순환 로직 네이티브 지원
   → 현재 코드베이스와 100% 호환

2. Modular RAG 접근 채택 ✅
   → Basic, Modular, Corrective 모두 지원
   → 모듈 단위 Ablation 가능
   → 점진적 개선 전략

3. 선행 작업 명확 ✅
   → Phase 0: 아키텍처 (1-2일)
   → Phase 1: 데이터 (이미 완료)
   → Phase 2: 모듈 (1-2주)
   → Phase 3: 파이프라인 (3-5일)
   → Phase 4: Ablation (2-3일)

4. 예상 성능 ✅
   → Recall@5: +26-31%p
   → Judge Score: +1.3-1.6점
   → Hallucination: -66-77%
```

### 6.2 다음 단계

**즉시 (오늘)**:
1. 이 문서 정독 (1-2시간)
2. 모듈 인터페이스 설계 시작
3. 기존 데이터 레이어 검증

**Week 1**:
1. 모듈 시스템 구축
2. Basic RAG Pipeline 구현
3. 베이스라인 측정

**Week 2-4**:
1. 핵심 모듈 구현
2. 3가지 파이프라인 구축
3. Ablation 실험 및 분석

---

**문서 버전**: 1.0  
**최종 수정**: 2025년 12월 15일  
**작성자**: Medical AI Agent Research Team  
**저장 위치**: `C:\Users\KHIDI\Downloads\final_medical_ai_agent\`

**관련 문서**:
- `재설계_전략_핵심요약_KO.md`
- `ZERO_TO_ONE_REDESIGN_STRATEGY.md`
- `IMPLEMENTATION_EXAMPLES.md`
- `REDESIGN_QUICK_START.md`

---

**END OF DOCUMENT**

