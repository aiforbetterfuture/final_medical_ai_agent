# 재설계 전략 구현 예시 코드

**연관 문서**: `ZERO_TO_ONE_REDESIGN_STRATEGY.md`  
**목적**: 각 레이어의 구체적인 구현 코드 예시 제공

---

## 목차

1. [Layer 0: Configuration & Instrumentation](#layer-0-configuration--instrumentation)
2. [Layer 1: Data Infrastructure](#layer-1-data-infrastructure)
3. [Layer 2: Retrieval Components](#layer-2-retrieval-components)
4. [Layer 3: Self-Refine Loop](#layer-3-self-refine-loop)
5. [Layer 4: Experiment Runner](#layer-4-experiment-runner)

---

## Layer 0: Configuration & Instrumentation

### 1.1 Feature Flags 시스템

```python
# config/feature_flags.py
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Literal
import yaml
from pathlib import Path

@dataclass
class FeatureFlags:
    """
    모든 ablation 변수를 관리하는 중앙 설정
    
    각 필드는 독립적으로 변경 가능하며, 실험 재현을 위해
    모든 설정이 자동으로 로깅됨
    """
    
    # ============ Data Layer ============
    chunk_size: int = 280
    chunk_overlap: int = 70
    chunking_strategy: Literal['uniform', 'type_aware'] = 'type_aware'
    metadata_richness: Literal['minimal', 'full'] = 'full'
    
    # ============ Embedding Layer ============
    embedding_model: str = 'text-embedding-3-large'
    embedding_dimension: int = 3072
    query_augmentation: Literal['none', 'profile', 'dual'] = 'dual'
    normalize_embeddings: bool = True
    
    # ============ Retrieval Layer ============
    retrieval_mode: Literal['bm25', 'faiss', 'hybrid'] = 'hybrid'
    index_strategy: Literal['single', 'dual'] = 'dual'
    routing_enabled: bool = True
    
    # Fine-grained index
    k_fine: int = 12
    fine_chunk_threshold: int = 300
    
    # Coarse-grained index
    k_coarse: int = 5
    
    # RRF fusion
    rrf_k: int = 60
    bm25_weight: float = 0.5
    faiss_weight: float = 0.5
    
    # ============ Reranking Layer ============
    reranking_enabled: bool = False
    reranker_model: Optional[str] = None
    rerank_top_n: int = 30
    rerank_keep_n: int = 5
    
    # ============ Generation Layer ============
    llm_model: str = 'gpt-4o-mini'
    temperature: float = 0.2
    max_tokens: int = 800
    top_p: float = 1.0
    
    # ============ Self-Refine Layer ============
    self_refine_enabled: bool = True
    max_refine_iterations: int = 2
    quality_threshold: float = 0.5
    llm_based_quality: bool = True
    dynamic_query_rewrite: bool = True
    
    # ============ Context Engineering ============
    include_profile: bool = True
    include_history: bool = True
    max_history_turns: int = 3
    context_manager_enabled: bool = True
    token_budget: int = 3000
    
    # ============ Memory & Cache ============
    response_cache_enabled: bool = True
    cache_similarity_threshold: float = 0.85
    cache_ttl_seconds: int = 3600
    hierarchical_memory: bool = False
    
    # ============ Safety & Monitoring ============
    duplicate_detection: bool = True
    progress_monitoring: bool = True
    timeout_seconds: int = 60
    max_retries: int = 3
    
    # ============ Experiment Metadata ============
    experiment_id: str = 'default'
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """실험 결과 저장용 딕셔너리 변환"""
        return asdict(self)
    
    def save(self, path: str):
        """YAML 파일로 저장"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureFlags':
        """YAML 파일에서 로드"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    @classmethod
    def from_profile(cls, profile_name: str) -> 'FeatureFlags':
        """사전 정의 프로파일 로드"""
        if profile_name not in ABLATION_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        return ABLATION_PROFILES[profile_name]()


# ============ 사전 정의 프로파일 ============

def baseline_profile() -> FeatureFlags:
    """최소 기능 베이스라인"""
    return FeatureFlags(
        chunk_size=900,  # 현재 설정 유지
        chunking_strategy='uniform',
        metadata_richness='minimal',
        query_augmentation='none',
        retrieval_mode='faiss',
        index_strategy='single',
        routing_enabled=False,
        self_refine_enabled=False,
        include_profile=False,
        include_history=False,
        context_manager_enabled=False,
        response_cache_enabled=False,
        experiment_id='baseline'
    )


def data_optimized_profile() -> FeatureFlags:
    """데이터 레이어만 최적화"""
    return FeatureFlags(
        chunk_size=280,
        chunking_strategy='type_aware',
        metadata_richness='full',
        query_augmentation='dual',
        retrieval_mode='faiss',
        index_strategy='dual',
        routing_enabled=True,
        self_refine_enabled=False,  # 아직 비활성화
        experiment_id='data_optimized'
    )


def hybrid_retrieval_profile() -> FeatureFlags:
    """하이브리드 검색 활성화"""
    base = data_optimized_profile()
    base.retrieval_mode = 'hybrid'
    base.rrf_k = 60
    base.experiment_id = 'hybrid_retrieval'
    return base


def self_refine_profile() -> FeatureFlags:
    """Self-Refine 추가"""
    base = hybrid_retrieval_profile()
    base.self_refine_enabled = True
    base.max_refine_iterations = 2
    base.quality_threshold = 0.5
    base.llm_based_quality = True
    base.dynamic_query_rewrite = True
    base.experiment_id = 'self_refine'
    return base


def full_system_profile() -> FeatureFlags:
    """모든 기능 활성화"""
    base = self_refine_profile()
    base.include_profile = True
    base.include_history = True
    base.context_manager_enabled = True
    base.response_cache_enabled = True
    base.reranking_enabled = True
    base.reranker_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    base.experiment_id = 'full_system'
    return base


ABLATION_PROFILES = {
    'baseline': baseline_profile,
    'data_optimized': data_optimized_profile,
    'hybrid_retrieval': hybrid_retrieval_profile,
    'self_refine': self_refine_profile,
    'full_system': full_system_profile,
}


# ============ 유틸리티 함수 ============

def list_profiles() -> Dict[str, str]:
    """사용 가능한 프로파일 목록"""
    return {
        'baseline': '최소 기능 베이스라인',
        'data_optimized': '데이터 레이어 최적화',
        'hybrid_retrieval': '하이브리드 검색',
        'self_refine': 'Self-Refine 추가',
        'full_system': '전체 시스템',
    }


def print_profiles():
    """프로파일 목록 출력"""
    print("사용 가능한 Ablation 프로파일:\n")
    for name, desc in list_profiles().items():
        print(f"  - {name:20s} : {desc}")
    print("\n사용법:")
    print("  flags = FeatureFlags.from_profile('baseline')")


if __name__ == '__main__':
    # 테스트
    print_profiles()
    
    # 프로파일 생성 및 저장
    baseline = FeatureFlags.from_profile('baseline')
    baseline.save('config/profiles/baseline.yaml')
    print(f"\nBaseline 설정 저장: config/profiles/baseline.yaml")
    
    # 로드 테스트
    loaded = FeatureFlags.load('config/profiles/baseline.yaml')
    assert loaded.experiment_id == 'baseline'
    print("✅ 로드 테스트 성공")
```

### 1.2 Instrumentation Layer

```python
# core/instrumentation.py
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path
from datetime import datetime
import threading

class MetricsCollector:
    """
    모든 컴포넌트의 메트릭을 자동 수집
    
    Thread-safe하며, 실시간으로 메트릭을 수집하고
    실험 종료 시 JSON 파일로 저장
    """
    
    def __init__(self, experiment_id: str, output_dir: str = 'runs'):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        self.start_time = time.time()
        
        print(f"[MetricsCollector] 초기화: {self.experiment_id}")
        print(f"[MetricsCollector] 출력 경로: {self.output_dir}")
    
    @contextmanager
    def measure(self, component: str, operation: str, **metadata):
        """
        컴포넌트 실행 시간 및 메트릭 측정
        
        사용 예시:
            with collector.measure('retrieval', 'hybrid_search', k=10):
                docs = retriever.search(query, k=10)
                result['num_docs'] = len(docs)
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'metadata': metadata
        }
        
        start = time.time()
        exception = None
        
        try:
            yield result
        except Exception as e:
            exception = e
            result['error'] = str(e)
            raise
        finally:
            result['elapsed_ms'] = (time.time() - start) * 1000
            result['success'] = exception is None
            
            with self.lock:
                self.metrics.append(result)
    
    def record(self, component: str, operation: str, **metrics):
        """
        단순 메트릭 기록 (시간 측정 없음)
        
        사용 예시:
            collector.record('generation', 'answer_quality',
                            quality_score=0.85, tokens=150)
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            **metrics
        }
        
        with self.lock:
            self.metrics.append(record)
    
    def get_summary(self) -> Dict[str, Any]:
        """집계 통계 계산"""
        total_time = time.time() - self.start_time
        
        # 컴포넌트별 통계
        component_stats = {}
        for metric in self.metrics:
            comp = metric['component']
            if comp not in component_stats:
                component_stats[comp] = {
                    'count': 0,
                    'total_time_ms': 0,
                    'errors': 0
                }
            
            component_stats[comp]['count'] += 1
            if 'elapsed_ms' in metric:
                component_stats[comp]['total_time_ms'] += metric['elapsed_ms']
            if not metric.get('success', True):
                component_stats[comp]['errors'] += 1
        
        # 평균 계산
        for comp, stats in component_stats.items():
            if stats['count'] > 0:
                stats['avg_time_ms'] = stats['total_time_ms'] / stats['count']
        
        return {
            'experiment_id': self.experiment_id,
            'total_time_sec': total_time,
            'total_metrics': len(self.metrics),
            'component_stats': component_stats
        }
    
    def save(self):
        """메트릭을 파일로 저장"""
        # 전체 메트릭
        metrics_file = self.output_dir / 'metrics.jsonl'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            for metric in self.metrics:
                f.write(json.dumps(metric, ensure_ascii=False) + '\n')
        
        # 요약 통계
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        
        print(f"\n[MetricsCollector] 메트릭 저장 완료:")
        print(f"  - {metrics_file}")
        print(f"  - {summary_file}")
    
    def print_summary(self):
        """요약 통계 출력"""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"실험 요약: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"총 실행 시간: {summary['total_time_sec']:.2f}초")
        print(f"총 메트릭 수: {summary['total_metrics']}")
        print(f"\n컴포넌트별 통계:")
        print(f"{'컴포넌트':<20} {'호출 횟수':>10} {'평균 시간(ms)':>15} {'오류':>8}")
        print(f"{'-'*60}")
        
        for comp, stats in summary['component_stats'].items():
            avg_time = stats.get('avg_time_ms', 0)
            print(f"{comp:<20} {stats['count']:>10} {avg_time:>15.2f} {stats['errors']:>8}")


# ============ 글로벌 싱글톤 ============

_global_collector: Optional[MetricsCollector] = None

def init_collector(experiment_id: str, output_dir: str = 'runs'):
    """글로벌 collector 초기화"""
    global _global_collector
    _global_collector = MetricsCollector(experiment_id, output_dir)
    return _global_collector

def get_collector() -> MetricsCollector:
    """글로벌 collector 가져오기"""
    if _global_collector is None:
        raise RuntimeError("Collector가 초기화되지 않았습니다. init_collector()를 먼저 호출하세요.")
    return _global_collector


# ============ 데코레이터 ============

def measure_time(component: str, operation: str):
    """함수 실행 시간 자동 측정 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_collector()
            with collector.measure(component, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============ 사용 예시 ============

if __name__ == '__main__':
    # 초기화
    collector = init_collector('test_experiment')
    
    # 예시 1: Context manager 사용
    with collector.measure('retrieval', 'bm25_search', k=10) as result:
        time.sleep(0.1)  # 실제 작업 시뮬레이션
        result['num_docs'] = 8
        result['avg_score'] = 0.75
    
    # 예시 2: 단순 기록
    collector.record('generation', 'quality_check',
                    quality_score=0.85,
                    tokens=150)
    
    # 예시 3: 데코레이터 사용
    @measure_time('processing', 'transform')
    def process_data(data):
        time.sleep(0.05)
        return data.upper()
    
    process_data("test")
    
    # 결과 출력 및 저장
    collector.print_summary()
    collector.save()
```

---

## Layer 1: Data Infrastructure

### 2.1 Type-Aware Chunker

```python
# data_pipeline/chunker.py
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
import re
from abc import ABC, abstractmethod

@dataclass
class Chunk:
    """청크 데이터 클래스"""
    text: str
    chunk_id: str
    doc_id: str
    doc_type: str
    span_start: int
    span_end: int
    metadata: Dict[str, Any]
    
    @property
    def token_count(self) -> int:
        """대략적인 토큰 수 (단어 수 * 1.3)"""
        return int(len(self.text.split()) * 1.3)


class ChunkStrategy(ABC):
    """청킹 전략 추상 클래스"""
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        pass


class SlidingWindowStrategy(ChunkStrategy):
    """슬라이딩 윈도우 청킹"""
    
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        words = text.split()
        chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                doc_id=doc_id,
                doc_type='unknown',
                span_start=start,
                span_end=end,
                metadata={}
            ))
            
            chunk_idx += 1
            start += (self.chunk_size - self.overlap)
        
        return chunks


class SentenceAwareStrategy(ChunkStrategy):
    """문장 경계를 존중하는 청킹"""
    
    def __init__(self, target_size: int, max_size: int):
        self.target_size = target_size
        self.max_size = max_size
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for sent in sentences:
            sent_size = len(sent.split())
            
            # 현재 청크에 추가 가능한지 확인
            if current_size + sent_size <= self.target_size:
                current_chunk.append(sent)
                current_size += sent_size
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, doc_id, chunk_idx
                    ))
                    chunk_idx += 1
                
                # 새 청크 시작
                current_chunk = [sent]
                current_size = sent_size
        
        # 마지막 청크
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, doc_id, chunk_idx
            ))
        
        return chunks
    
    def _create_chunk(self, sentences: List[str], doc_id: str, idx: int) -> Chunk:
        text = ' '.join(sentences)
        return Chunk(
            text=text,
            chunk_id=f"{doc_id}_chunk_{idx}",
            doc_id=doc_id,
            doc_type='unknown',
            span_start=0,  # TODO: 실제 위치 계산
            span_end=0,
            metadata={'num_sentences': len(sentences)}
        )


class TypeAwareChunker:
    """
    문서 타입별로 다른 청킹 전략 적용
    
    - drug_contraindication: 짧게 (180 tokens)
    - clinical_guideline: 중간 (280 tokens)
    - case_report: 길게 (320 tokens)
    - general_knowledge: 가장 길게 (400 tokens)
    """
    
    def __init__(self):
        self.strategies = {
            'drug_contraindication': SentenceAwareStrategy(
                target_size=180, max_size=220
            ),
            'clinical_guideline': SentenceAwareStrategy(
                target_size=280, max_size=320
            ),
            'case_report': SentenceAwareStrategy(
                target_size=320, max_size=380
            ),
            'general_knowledge': SlidingWindowStrategy(
                chunk_size=400, overlap=100
            ),
        }
    
    def classify_document(self, doc: Dict[str, Any]) -> str:
        """
        문서 타입 분류
        
        실제로는 더 정교한 분류기 사용 (키워드, 메타데이터 등)
        """
        text = doc.get('text', '').lower()
        
        # 간단한 키워드 기반 분류
        if any(kw in text for kw in ['contraindication', 'warning', 'adverse']):
            return 'drug_contraindication'
        elif any(kw in text for kw in ['guideline', 'recommendation', 'should']):
            return 'clinical_guideline'
        elif any(kw in text for kw in ['case', 'patient', 'presented']):
            return 'case_report'
        else:
            return 'general_knowledge'
    
    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        """문서를 청크로 분할"""
        doc_type = self.classify_document(doc)
        strategy = self.strategies[doc_type]
        
        chunks = strategy.chunk(doc['text'], doc['id'])
        
        # 메타데이터 추가
        for chunk in chunks:
            chunk.doc_type = doc_type
            chunk.metadata.update({
                'source': doc.get('source', 'unknown'),
                'published_date': doc.get('published_date'),
                'confidence': doc.get('confidence', 1.0),
            })
        
        return chunks
    
    def chunk_corpus(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """전체 코퍼스 청킹"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        print(f"[TypeAwareChunker] 총 {len(documents)}개 문서 → {len(all_chunks)}개 청크 생성")
        
        # 타입별 통계
        type_counts = {}
        for chunk in all_chunks:
            type_counts[chunk.doc_type] = type_counts.get(chunk.doc_type, 0) + 1
        
        print("[TypeAwareChunker] 타입별 청크 수:")
        for doc_type, count in type_counts.items():
            print(f"  - {doc_type}: {count}")
        
        return all_chunks


# ============ 사용 예시 ============

if __name__ == '__main__':
    # 테스트 문서
    documents = [
        {
            'id': 'doc_001',
            'text': 'Metformin is contraindicated in patients with severe renal impairment. '
                   'It should not be used in patients with eGFR < 30 mL/min/1.73m². '
                   'Warning: Risk of lactic acidosis.',
            'source': 'FDA Label',
            'published_date': '2023-01-15'
        },
        {
            'id': 'doc_002',
            'text': 'Clinical guidelines recommend lifestyle modification as first-line therapy '
                   'for type 2 diabetes. This includes diet, exercise, and weight management. '
                   'Metformin should be initiated if lifestyle changes are insufficient.',
            'source': 'ADA Guidelines',
            'published_date': '2024-03-20'
        }
    ]
    
    # 청킹
    chunker = TypeAwareChunker()
    chunks = chunker.chunk_corpus(documents)
    
    # 결과 출력
    for chunk in chunks:
        print(f"\n{chunk.chunk_id} ({chunk.doc_type}):")
        print(f"  Text: {chunk.text[:80]}...")
        print(f"  Tokens: ~{chunk.token_count}")
        print(f"  Metadata: {chunk.metadata}")
```

### 2.2 Dual Index Builder

```python
# data_pipeline/indexer.py
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from pathlib import Path
import json
from tqdm import tqdm

from data_pipeline.chunker import Chunk
from core.llm_client import LLMClient

class DualIndexBuilder:
    """
    Fine-grained / Coarse-grained 듀얼 인덱스 생성
    
    - Fine-grained: < 300 tokens (정밀 검색용)
    - Coarse-grained: >= 300 tokens (맥락 검색용)
    """
    
    def __init__(self, embedding_model: str = 'text-embedding-3-large'):
        self.embedding_model = embedding_model
        self.llm_client = LLMClient()
        
        self.fine_chunks: List[Chunk] = []
        self.coarse_chunks: List[Chunk] = []
        
        self.fine_index: Optional[faiss.Index] = None
        self.coarse_index: Optional[faiss.Index] = None
    
    def build(self, chunks: List[Chunk], output_dir: str):
        """듀얼 인덱스 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[DualIndexBuilder] 듀얼 인덱스 생성 시작")
        print(f"  - 총 청크 수: {len(chunks)}")
        
        # 1. 청크 분류
        self._classify_chunks(chunks)
        
        # 2. 임베딩 생성
        fine_embeddings = self._embed_chunks(self.fine_chunks, "fine-grained")
        coarse_embeddings = self._embed_chunks(self.coarse_chunks, "coarse-grained")
        
        # 3. FAISS 인덱스 생성
        self.fine_index = self._build_faiss_index(fine_embeddings, "fine")
        self.coarse_index = self._build_faiss_index(coarse_embeddings, "coarse")
        
        # 4. 저장
        self._save(output_path)
        
        print(f"[DualIndexBuilder] 완료!")
        print(f"  - Fine-grained: {len(self.fine_chunks)} chunks")
        print(f"  - Coarse-grained: {len(self.coarse_chunks)} chunks")
    
    def _classify_chunks(self, chunks: List[Chunk]):
        """청크를 크기로 분류"""
        for chunk in chunks:
            if chunk.token_count < 300:
                self.fine_chunks.append(chunk)
            else:
                self.coarse_chunks.append(chunk)
        
        print(f"[DualIndexBuilder] 청크 분류 완료:")
        print(f"  - Fine-grained: {len(self.fine_chunks)}")
        print(f"  - Coarse-grained: {len(self.coarse_chunks)}")
    
    def _embed_chunks(self, chunks: List[Chunk], index_type: str) -> np.ndarray:
        """청크 임베딩 생성"""
        print(f"[DualIndexBuilder] {index_type} 임베딩 생성 중...")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = []
        
        batch_size = 128
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            batch_embeddings = [
                self.llm_client.embed(text, embedding_model=self.embedding_model)
                for text in batch
            ]
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # L2 정규화
        faiss.normalize_L2(embeddings_array)
        
        print(f"  - 임베딩 shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def _build_faiss_index(self, embeddings: np.ndarray, index_type: str) -> faiss.Index:
        """FAISS 인덱스 생성"""
        print(f"[DualIndexBuilder] {index_type} FAISS 인덱스 생성 중...")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        index.add(embeddings)
        
        print(f"  - 인덱스 크기: {index.ntotal}")
        
        return index
    
    def _save(self, output_path: Path):
        """인덱스 및 메타데이터 저장"""
        print(f"[DualIndexBuilder] 저장 중...")
        
        # FAISS 인덱스 저장
        faiss.write_index(self.fine_index, str(output_path / 'fine.index.faiss'))
        faiss.write_index(self.coarse_index, str(output_path / 'coarse.index.faiss'))
        
        # 메타데이터 저장
        self._save_metadata(self.fine_chunks, output_path / 'fine.meta.jsonl')
        self._save_metadata(self.coarse_chunks, output_path / 'coarse.meta.jsonl')
        
        print(f"  - 저장 완료: {output_path}")
    
    def _save_metadata(self, chunks: List[Chunk], path: Path):
        """청크 메타데이터 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                meta = {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'doc_type': chunk.doc_type,
                    'text': chunk.text,
                    'span_start': chunk.span_start,
                    'span_end': chunk.span_end,
                    'token_count': chunk.token_count,
                    'metadata': chunk.metadata
                }
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')


# ============ 사용 예시 ============

if __name__ == '__main__':
    from data_pipeline.chunker import TypeAwareChunker
    
    # 1. 문서 로드 (예시)
    documents = [...]  # 실제 문서 데이터
    
    # 2. 청킹
    chunker = TypeAwareChunker()
    chunks = chunker.chunk_corpus(documents)
    
    # 3. 듀얼 인덱스 생성
    builder = DualIndexBuilder(embedding_model='text-embedding-3-large')
    builder.build(chunks, output_dir='data/index/dual')
```

---

## Layer 2: Retrieval Components

### 3.1 Dual Index Retriever

```python
# retrieval/dual_retriever.py
from typing import List, Dict, Any, Tuple, Literal
import numpy as np
import faiss
import json
from pathlib import Path

from core.llm_client import LLMClient

class DualIndexRetriever:
    """
    Fine/Coarse 듀얼 인덱스 검색기
    
    쿼리 타입에 따라 적절한 인덱스 선택:
    - symptom/drug 쿼리 → fine-grained 우선
    - general 쿼리 → coarse-grained 우선
    """
    
    def __init__(self, index_dir: str, embedding_model: str = 'text-embedding-3-large'):
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.llm_client = LLMClient()
        
        # 인덱스 로드
        self.fine_index = faiss.read_index(str(self.index_dir / 'fine.index.faiss'))
        self.coarse_index = faiss.read_index(str(self.index_dir / 'coarse.index.faiss'))
        
        # 메타데이터 로드
        self.fine_meta = self._load_metadata(self.index_dir / 'fine.meta.jsonl')
        self.coarse_meta = self._load_metadata(self.index_dir / 'coarse.meta.jsonl')
        
        print(f"[DualIndexRetriever] 초기화 완료")
        print(f"  - Fine index: {self.fine_index.ntotal} vectors")
        print(f"  - Coarse index: {self.coarse_index.ntotal} vectors")
    
    def _load_metadata(self, path: Path) -> List[Dict[str, Any]]:
        """메타데이터 로드"""
        metadata = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                metadata.append(json.loads(line))
        return metadata
    
    def search(
        self,
        query: str,
        k_fine: int = 12,
        k_coarse: int = 5,
        route: Literal['fine', 'coarse', 'both'] = 'both'
    ) -> List[Dict[str, Any]]:
        """
        듀얼 인덱스 검색
        
        Args:
            query: 검색 쿼리
            k_fine: Fine-grained 인덱스에서 가져올 개수
            k_coarse: Coarse-grained 인덱스에서 가져올 개수
            route: 검색 전략 ('fine', 'coarse', 'both')
        
        Returns:
            검색된 문서 리스트 (score 포함)
        """
        # 쿼리 임베딩
        query_vector = self.llm_client.embed(query, embedding_model=self.embedding_model)
        query_vector = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        results = []
        
        # Fine-grained 검색
        if route in ['fine', 'both']:
            fine_results = self._search_index(
                self.fine_index,
                self.fine_meta,
                query_vector,
                k=k_fine,
                index_type='fine'
            )
            results.extend(fine_results)
        
        # Coarse-grained 검색
        if route in ['coarse', 'both']:
            coarse_results = self._search_index(
                self.coarse_index,
                self.coarse_meta,
                query_vector,
                k=k_coarse,
                index_type='coarse'
            )
            results.extend(coarse_results)
        
        # 점수로 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def _search_index(
        self,
        index: faiss.Index,
        metadata: List[Dict],
        query_vector: np.ndarray,
        k: int,
        index_type: str
    ) -> List[Dict[str, Any]]:
        """단일 인덱스 검색"""
        scores, indices = index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metadata):
                doc = metadata[idx].copy()
                doc['score'] = float(score)
                doc['index_type'] = index_type
                results.append(doc)
        
        return results


# ============ 사용 예시 ============

if __name__ == '__main__':
    retriever = DualIndexRetriever(index_dir='data/index/dual')
    
    # 테스트 쿼리
    query = "What are the contraindications of metformin?"
    
    # 검색
    results = retriever.search(query, k_fine=12, k_coarse=5, route='both')
    
    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results[:5], 1):
        print(f"\n[{i}] Score: {doc['score']:.4f} ({doc['index_type']})")
        print(f"    {doc['text'][:100]}...")
```

---

이 문서는 계속됩니다. 다음 섹션에서는 Self-Refine Loop와 Experiment Runner 구현을 다룹니다.

**다음 섹션 미리보기**:
- Layer 3: Self-Refine Loop 구현
- Layer 4: Experiment Runner 및 Ablation 자동화
- 전체 시스템 통합 예시

---

**문서 버전**: 1.0  
**최종 수정**: 2025-12-15

