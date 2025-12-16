# 재설계 전략 Quick Start 가이드

**목적**: 오늘 당장 시작할 수 있는 실전 가이드  
**소요 시간**: 1-2주로 MVP 구축 가능

---

## 🎯 핵심 질문과 답변

### Q1: 지금 당장 무엇부터 시작해야 하나요?

**A: 데이터 레이어부터 시작하세요 (Bottom-Up 접근)**

```
Day 1-2: 현재 데이터 분석
  ├─ 청크 크기 분포 확인
  ├─ 문서 타입 분류 가능성 검토
  └─ Retrieval 성능 베이스라인 측정

Day 3-5: 청킹 전략 개선
  ├─ 280 토큰으로 재청킹
  ├─ 메타데이터 추가
  └─ 듀얼 인덱스 구축

Day 6-7: 성능 측정
  ├─ Recall@5, MRR 측정
  └─ 개선 효과 확인 (목표: +10-20%p)
```

### Q2: 현재 시스템을 어떻게 개선할까요?

**A: 3단계 점진적 개선 전략**

```
Phase 1 (1주): 데이터 최적화
  → 청킹 + 임베딩 개선
  → 예상 효과: Recall +15%p

Phase 2 (1주): Retrieval 강화
  → Hybrid + Dual Index
  → 예상 효과: MRR +10%p

Phase 3 (1주): Self-Refine 추가
  → Quality Check + Rewrite
  → 예상 효과: Judge Score +1.5점
```

### Q3: Ablation 연구는 언제 시작하나요?

**A: 각 Phase 완료 후 즉시 측정**

```
Phase 1 완료 → Data Ablation (D1-D5)
Phase 2 완료 → Retrieval Ablation (R1-R4)
Phase 3 완료 → Generation Ablation (G1-G4)
```

---

## 📋 1주차 실행 계획

### Day 1: 현재 상태 분석

#### 1.1 데이터 분석 스크립트 실행

```python
# scripts/analyze_current_data.py
"""현재 청킹 상태 분석"""

import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_chunks(meta_path: str):
    """청크 메타데이터 분석"""
    chunks = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # 토큰 수 분포
    token_counts = []
    for chunk in chunks:
        text = chunk.get('text', '')
        token_count = len(text.split()) * 1.3  # 대략적 토큰 수
        token_counts.append(token_count)
    
    # 통계
    print(f"총 청크 수: {len(chunks)}")
    print(f"평균 토큰 수: {sum(token_counts) / len(token_counts):.1f}")
    print(f"중앙값: {sorted(token_counts)[len(token_counts)//2]:.1f}")
    print(f"최소/최대: {min(token_counts):.1f} / {max(token_counts):.1f}")
    
    # 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, edgecolor='black')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.title('Current Chunk Size Distribution')
    plt.axvline(x=900, color='r', linestyle='--', label='Current target (900)')
    plt.axvline(x=280, color='g', linestyle='--', label='Proposed target (280)')
    plt.legend()
    plt.savefig('chunk_distribution.png', dpi=300)
    print("\n✅ 분포 차트 저장: chunk_distribution.png")
    
    return token_counts

if __name__ == '__main__':
    # 현재 메타데이터 파일 경로
    meta_path = 'data/index/train_qa/train_questions.meta.jsonl'
    
    if Path(meta_path).exists():
        analyze_chunks(meta_path)
    else:
        print(f"⚠️  파일이 없습니다: {meta_path}")
        print("먼저 인덱스를 생성하세요.")
```

**실행**:
```bash
python scripts/analyze_current_data.py
```

#### 1.2 베이스라인 성능 측정

```python
# scripts/measure_baseline.py
"""현재 시스템 성능 측정"""

from agent.graph import run_agent
from evaluation.retrieval_metrics import recall_at_k, mrr
import json

# 테스트 쿼리 (val set에서 샘플링)
TEST_CASES = [
    {
        'query': 'What are the side effects of metformin?',
        'relevant_docs': ['doc_123', 'doc_456', 'doc_789']  # Ground truth
    },
    # ... 최소 20-30개
]

def measure_baseline():
    """베이스라인 성능 측정"""
    results = []
    
    for case in TEST_CASES:
        # 현재 시스템으로 검색
        state = run_agent(
            user_text=case['query'],
            mode='ai_agent',
            return_state=True
        )
        
        # 검색된 문서 ID 추출
        retrieved_ids = [doc['doc_id'] for doc in state['retrieved_docs']]
        
        # 메트릭 계산
        metrics = {
            'query': case['query'],
            'recall@5': recall_at_k(retrieved_ids, case['relevant_docs'], k=5),
            'recall@10': recall_at_k(retrieved_ids, case['relevant_docs'], k=10),
            'mrr': mrr(retrieved_ids, case['relevant_docs']),
            'quality_score': state.get('quality_score', 0.0)
        }
        
        results.append(metrics)
        print(f"✓ {case['query'][:50]}... → R@5={metrics['recall@5']:.2f}")
    
    # 집계
    avg_metrics = {
        'recall@5': sum(r['recall@5'] for r in results) / len(results),
        'recall@10': sum(r['recall@10'] for r in results) / len(results),
        'mrr': sum(r['mrr'] for r in results) / len(results),
        'quality_score': sum(r['quality_score'] for r in results) / len(results),
    }
    
    print(f"\n{'='*60}")
    print("베이스라인 성능 (현재 시스템)")
    print(f"{'='*60}")
    for metric, value in avg_metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # 저장
    with open('baseline_results.json', 'w') as f:
        json.dump({
            'avg_metrics': avg_metrics,
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n✅ 결과 저장: baseline_results.json")
    
    return avg_metrics

if __name__ == '__main__':
    measure_baseline()
```

**실행**:
```bash
python scripts/measure_baseline.py
```

**예상 결과**:
```
베이스라인 성능 (현재 시스템)
============================================================
recall@5            : 0.6500
recall@10           : 0.7800
mrr                 : 0.5200
quality_score       : 7.2000
```

### Day 2-3: 청킹 전략 개선

#### 2.1 새로운 청킹 파이프라인 구축

```bash
# 1. 구현 파일 복사 (IMPLEMENTATION_EXAMPLES.md에서)
mkdir -p data_pipeline
# data_pipeline/chunker.py 작성

# 2. 문서 재청킹
python scripts/rechunk_corpus.py \
  --input data/corpus/train_source \
  --output data/corpus_v2/train_source \
  --strategy type_aware \
  --target_size 280
```

```python
# scripts/rechunk_corpus.py
"""코퍼스 재청킹 스크립트"""

import argparse
from pathlib import Path
import json
from data_pipeline.chunker import TypeAwareChunker

def rechunk_corpus(input_dir: str, output_dir: str, strategy: str, target_size: int):
    """코퍼스 재청킹"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 원본 문서 로드
    documents = []
    for jsonl_file in input_path.glob('**/*.jsonl'):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
    
    print(f"[Rechunking] 로드된 문서 수: {len(documents)}")
    
    # 청킹
    if strategy == 'type_aware':
        chunker = TypeAwareChunker()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    chunks = chunker.chunk_corpus(documents)
    
    # 저장
    output_file = output_path / 'chunks.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'doc_type': chunk.doc_type,
                'text': chunk.text,
                'span_start': chunk.span_start,
                'span_end': chunk.span_end,
                'token_count': chunk.token_count,
                'metadata': chunk.metadata
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
    
    print(f"✅ 청크 저장 완료: {output_file}")
    print(f"   총 {len(chunks)}개 청크 생성")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--strategy', default='type_aware')
    parser.add_argument('--target_size', type=int, default=280)
    
    args = parser.parse_args()
    rechunk_corpus(args.input, args.output, args.strategy, args.target_size)
```

#### 2.2 듀얼 인덱스 구축

```bash
python scripts/build_dual_index.py \
  --chunks data/corpus_v2/train_source/chunks.jsonl \
  --output data/index_v2/train_source \
  --embedding_model text-embedding-3-large
```

```python
# scripts/build_dual_index.py
"""듀얼 인덱스 구축 스크립트"""

import argparse
import json
from pathlib import Path
from data_pipeline.indexer import DualIndexBuilder
from data_pipeline.chunker import Chunk

def build_dual_index(chunks_path: str, output_dir: str, embedding_model: str):
    """듀얼 인덱스 구축"""
    # 청크 로드
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunk = Chunk(
                text=data['text'],
                chunk_id=data['chunk_id'],
                doc_id=data['doc_id'],
                doc_type=data['doc_type'],
                span_start=data['span_start'],
                span_end=data['span_end'],
                metadata=data['metadata']
            )
            chunks.append(chunk)
    
    print(f"[Build Index] 로드된 청크 수: {len(chunks)}")
    
    # 인덱스 생성
    builder = DualIndexBuilder(embedding_model=embedding_model)
    builder.build(chunks, output_dir=output_dir)
    
    print(f"✅ 듀얼 인덱스 생성 완료: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--embedding_model', default='text-embedding-3-large')
    
    args = parser.parse_args()
    build_dual_index(args.chunks, args.output, args.embedding_model)
```

### Day 4-5: 성능 재측정

```python
# scripts/compare_performance.py
"""개선 전후 성능 비교"""

import json
from scripts.measure_baseline import measure_baseline, TEST_CASES
from retrieval.dual_retriever import DualIndexRetriever

def measure_improved():
    """개선된 시스템 성능 측정"""
    # 새로운 retriever 사용
    retriever = DualIndexRetriever(
        index_dir='data/index_v2/train_source',
        embedding_model='text-embedding-3-large'
    )
    
    results = []
    
    for case in TEST_CASES:
        # 검색
        docs = retriever.search(
            query=case['query'],
            k_fine=12,
            k_coarse=5,
            route='both'
        )
        
        retrieved_ids = [doc['doc_id'] for doc in docs]
        
        # 메트릭 계산
        from evaluation.retrieval_metrics import recall_at_k, mrr
        metrics = {
            'query': case['query'],
            'recall@5': recall_at_k(retrieved_ids, case['relevant_docs'], k=5),
            'recall@10': recall_at_k(retrieved_ids, case['relevant_docs'], k=10),
            'mrr': mrr(retrieved_ids, case['relevant_docs']),
        }
        
        results.append(metrics)
    
    # 집계
    avg_metrics = {
        'recall@5': sum(r['recall@5'] for r in results) / len(results),
        'recall@10': sum(r['recall@10'] for r in results) / len(results),
        'mrr': sum(r['mrr'] for r in results) / len(results),
    }
    
    return avg_metrics

def compare():
    """비교 리포트 생성"""
    # 베이스라인 로드
    with open('baseline_results.json', 'r') as f:
        baseline = json.load(f)['avg_metrics']
    
    # 개선 버전 측정
    improved = measure_improved()
    
    # 비교 출력
    print(f"\n{'='*70}")
    print("성능 비교: Baseline vs Improved")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Baseline':>12} {'Improved':>12} {'Δ':>12} {'Δ%':>10}")
    print(f"{'-'*70}")
    
    for metric in ['recall@5', 'recall@10', 'mrr']:
        base_val = baseline[metric]
        impr_val = improved[metric]
        delta = impr_val - base_val
        delta_pct = (delta / base_val) * 100 if base_val > 0 else 0
        
        print(f"{metric:<20} {base_val:>12.4f} {impr_val:>12.4f} "
              f"{delta:>+12.4f} {delta_pct:>+9.1f}%")
    
    # 저장
    with open('comparison_results.json', 'w') as f:
        json.dump({
            'baseline': baseline,
            'improved': improved,
            'delta': {k: improved[k] - baseline[k] for k in improved.keys()}
        }, f, indent=2)
    
    print(f"\n✅ 비교 결과 저장: comparison_results.json")

if __name__ == '__main__':
    compare()
```

**실행**:
```bash
python scripts/compare_performance.py
```

**예상 결과**:
```
성능 비교: Baseline vs Improved
======================================================================
Metric               Baseline     Improved           Δ        Δ%
----------------------------------------------------------------------
recall@5               0.6500       0.7800      +0.1300    +20.0%
recall@10              0.7800       0.8500      +0.0700     +9.0%
mrr                    0.5200       0.6500      +0.1300    +25.0%
```

### Day 6-7: Data Ablation 실험

```python
# experiments/data_ablation.py
"""데이터 레이어 ablation 실험"""

from config.feature_flags import FeatureFlags
from scripts.measure_baseline import TEST_CASES
from evaluation.retrieval_metrics import recall_at_k, mrr
import json

# 실험 설정
EXPERIMENTS = {
    'D1_baseline': {
        'chunk_size': 900,
        'chunking_strategy': 'uniform',
        'metadata_richness': 'minimal',
        'index_strategy': 'single'
    },
    'D2_fine_chunking': {
        'chunk_size': 280,
        'chunking_strategy': 'uniform',
        'metadata_richness': 'minimal',
        'index_strategy': 'single'
    },
    'D3_metadata': {
        'chunk_size': 280,
        'chunking_strategy': 'type_aware',
        'metadata_richness': 'full',
        'index_strategy': 'single'
    },
    'D4_dual_index': {
        'chunk_size': 280,
        'chunking_strategy': 'type_aware',
        'metadata_richness': 'full',
        'index_strategy': 'dual'
    },
}

def run_experiment(exp_id: str, config: dict):
    """단일 실험 실행"""
    print(f"\n{'='*60}")
    print(f"실험: {exp_id}")
    print(f"{'='*60}")
    
    # TODO: 각 설정에 맞는 retriever 로드
    # retriever = load_retriever(config)
    
    # 성능 측정
    # results = measure_performance(retriever, TEST_CASES)
    
    # 임시: 시뮬레이션
    results = {
        'recall@5': 0.65 + (0.15 * (list(EXPERIMENTS.keys()).index(exp_id) / len(EXPERIMENTS))),
        'mrr': 0.52 + (0.13 * (list(EXPERIMENTS.keys()).index(exp_id) / len(EXPERIMENTS)))
    }
    
    print(f"  Recall@5: {results['recall@5']:.4f}")
    print(f"  MRR:      {results['mrr']:.4f}")
    
    return results

def run_all_experiments():
    """전체 ablation 실험 실행"""
    all_results = {}
    
    for exp_id, config in EXPERIMENTS.items():
        results = run_experiment(exp_id, config)
        all_results[exp_id] = results
    
    # 비교 테이블
    print(f"\n{'='*70}")
    print("Data Ablation 결과")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Recall@5':>12} {'MRR':>12} {'Δ R@5':>12}")
    print(f"{'-'*70}")
    
    baseline_recall = all_results['D1_baseline']['recall@5']
    
    for exp_id, results in all_results.items():
        delta = results['recall@5'] - baseline_recall
        print(f"{exp_id:<25} {results['recall@5']:>12.4f} {results['mrr']:>12.4f} "
              f"{delta:>+12.4f}")
    
    # 저장
    with open('data_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 결과 저장: data_ablation_results.json")

if __name__ == '__main__':
    run_all_experiments()
```

**실행**:
```bash
python experiments/data_ablation.py
```

---

## 📊 2주차 실행 계획

### Day 8-10: Self-Refine 구현

```python
# agent/nodes/refine_v2.py
"""개선된 Self-Refine 노드"""

from agent.state import AgentState
from core.llm_client import LLMClient

def refine_node(state: AgentState) -> AgentState:
    """
    Self-Refine 노드
    
    1. 현재 답변 품질 평가
    2. 품질이 낮으면 쿼리 재작성
    3. 재검색 및 재생성
    """
    llm_client = LLMClient()
    
    # 품질 평가
    quality_score = evaluate_quality(state['answer'], state['retrieved_docs'], llm_client)
    state['quality_score'] = quality_score
    
    # 임계값 확인
    threshold = state.get('quality_threshold', 0.5)
    max_iterations = state.get('max_refine_iterations', 2)
    current_iteration = state.get('iteration_count', 0)
    
    if quality_score < threshold and current_iteration < max_iterations:
        # 재작성 필요
        state['needs_retrieval'] = True
        state['iteration_count'] = current_iteration + 1
        
        # 쿼리 재작성
        rewritten_query = rewrite_query(
            original_query=state['user_text'],
            current_answer=state['answer'],
            quality_score=quality_score,
            llm_client=llm_client
        )
        state['query_for_retrieval'] = rewritten_query
        
        print(f"[Refine] 품질 낮음 ({quality_score:.2f}), 재검색 필요")
        print(f"[Refine] 재작성 쿼리: {rewritten_query}")
    else:
        state['needs_retrieval'] = False
        print(f"[Refine] 품질 충분 ({quality_score:.2f}), 완료")
    
    return state

def evaluate_quality(answer: str, docs: list, llm_client: LLMClient) -> float:
    """LLM 기반 품질 평가"""
    prompt = f"""
    Rate the quality of this answer on a scale of 0-1.
    Consider: relevance, accuracy, completeness, clarity.
    
    Answer: {answer}
    
    Context documents: {[d['text'][:200] for d in docs[:3]]}
    
    Return only a number between 0 and 1.
    """
    
    response = llm_client.generate(prompt, temperature=0.0, max_tokens=10)
    
    try:
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5  # 기본값

def rewrite_query(original_query: str, current_answer: str, quality_score: float, llm_client: LLMClient) -> str:
    """쿼리 재작성"""
    prompt = f"""
    The current answer has low quality ({quality_score:.2f}).
    Rewrite the search query to find better information.
    
    Original query: {original_query}
    Current answer: {current_answer}
    
    Rewritten query:
    """
    
    rewritten = llm_client.generate(prompt, temperature=0.3, max_tokens=100)
    return rewritten.strip()
```

### Day 11-12: Generation Ablation 실험

```python
# experiments/generation_ablation.py
"""Generation 레이어 ablation 실험"""

EXPERIMENTS = {
    'G1_no_refine': {
        'self_refine_enabled': False
    },
    'G2_heuristic': {
        'self_refine_enabled': True,
        'llm_based_quality': False
    },
    'G3_llm_quality': {
        'self_refine_enabled': True,
        'llm_based_quality': True,
        'dynamic_query_rewrite': False
    },
    'G4_full': {
        'self_refine_enabled': True,
        'llm_based_quality': True,
        'dynamic_query_rewrite': True
    },
}

# 실행 로직은 data_ablation.py와 유사
```

### Day 13-14: 전체 시스템 통합 및 논문 작성

```python
# experiments/final_comparison.py
"""최종 시스템 비교"""

SYSTEMS = {
    'Baseline (LLM Only)': {...},
    'Basic RAG': {...},
    'RAG + Self-Refine': {...},
    'Full System': {...},
}

# 전체 메트릭 측정 및 비교 테이블 생성
```

---

## 🎯 체크리스트

### Week 1: Data Layer

- [ ] **Day 1**: 현재 데이터 분석 완료
  - [ ] 청크 크기 분포 확인
  - [ ] 베이스라인 성능 측정 (Recall@5, MRR)

- [ ] **Day 2-3**: 청킹 개선
  - [ ] TypeAwareChunker 구현
  - [ ] 코퍼스 재청킹 (280 tokens)
  - [ ] 메타데이터 추가

- [ ] **Day 4-5**: 인덱스 구축
  - [ ] 듀얼 인덱스 생성
  - [ ] 성능 재측정
  - [ ] 개선 효과 확인 (목표: Recall@5 +15%p)

- [ ] **Day 6-7**: Data Ablation
  - [ ] D1-D4 실험 실행
  - [ ] 결과 분석 및 시각화

### Week 2: Self-Refine & Integration

- [ ] **Day 8-10**: Self-Refine 구현
  - [ ] Quality evaluator 구현
  - [ ] Query rewriter 구현
  - [ ] Refine loop 통합

- [ ] **Day 11-12**: Generation Ablation
  - [ ] G1-G4 실험 실행
  - [ ] Judge Score 측정

- [ ] **Day 13-14**: 최종 통합
  - [ ] 전체 시스템 테스트
  - [ ] 논문용 표/그래프 생성
  - [ ] 코드 정리 및 문서화

---

## 📈 예상 결과

### 정량적 목표

| 메트릭 | Baseline | Week 1 | Week 2 | 목표 |
|-------|---------|--------|--------|------|
| **Recall@5** | 0.65 | 0.78 (+20%) | 0.82 (+26%) | > 0.75 |
| **MRR** | 0.52 | 0.65 (+25%) | 0.72 (+38%) | > 0.70 |
| **Judge Score** | 7.2 | 7.5 (+4%) | 8.5 (+18%) | > 8.0 |

### Ablation 발견 (예상)

```
Data Layer 기여도:     40-50% (가장 큰 영향)
Retrieval 기여도:      10-15%
Self-Refine 기여도:    20-30%
Context Eng 기여도:    5-10%
```

---

## 🚨 주의사항

### 1. API 비용 관리

```python
# 비용 추정
EMBEDDING_COST = 0.00013 / 1K tokens  # text-embedding-3-large
LLM_COST = 0.15 / 1M tokens  # gpt-4o-mini input

# 예상 비용 (1,000 문서 기준)
# - 재임베딩: ~$2-5
# - Ablation 실험 (100 쿼리): ~$1-3
# 총 예상: $10-20
```

### 2. 시간 관리

```
재청킹:       1-2시간
재임베딩:     2-4시간 (1,000 문서 기준)
인덱스 구축:  10-30분
실험 실행:    1-2시간 (per ablation)
```

### 3. 데이터 백업

```bash
# 원본 데이터 백업
cp -r data/corpus data/corpus_backup
cp -r data/index data/index_backup

# 실험 결과 버전 관리
git add experiments/results/
git commit -m "Add ablation results"
```

---

## 💡 트러블슈팅

### Q: 재임베딩이 너무 오래 걸려요

**A: 배치 크기 증가 및 병렬 처리**

```python
# data_pipeline/indexer.py
batch_size = 256  # 128 → 256
# 또는 multiprocessing 사용
```

### Q: 메모리 부족 오류

**A: 청크 단위로 처리**

```python
# 전체를 한 번에 로드하지 말고 스트리밍
for chunk_batch in load_chunks_in_batches(batch_size=1000):
    embeddings = embed_batch(chunk_batch)
    save_embeddings(embeddings)
```

### Q: Ablation 실험이 너무 많아요

**A: 우선순위 실험만 실행**

```python
# 필수 실험만 (논문용)
PRIORITY_EXPERIMENTS = [
    'D1_baseline',
    'D4_dual_index',  # 최종 데이터 설정
    'G1_no_refine',
    'G4_full'  # 최종 시스템
]
```

---

## 📚 다음 단계

### 2주 후 (MVP 완성 시)

1. **논문 작성 시작**
   - Method 섹션: 아키텍처 설명
   - Results 섹션: Ablation 결과 테이블

2. **추가 실험 (선택)**
   - Reranker 추가
   - Context compression
   - Multi-turn 테스트

3. **코드 정리**
   - README 작성
   - 주석 추가
   - 테스트 코드 작성

---

## 🎓 학습 자료

### 추천 논문 (읽기 순서)

1. **RAG 기초** (1-2시간)
   - Lewis et al. (2020). "Retrieval-Augmented Generation"
   
2. **Self-Refine** (1시간)
   - Madaan et al. (2023). "Self-Refine"
   
3. **의학 도메인** (2-3시간)
   - Xiong et al. (2024). "Benchmarking RAG for Medicine"

### 추천 코드 참고

- **LangGraph 튜토리얼**: https://langchain-ai.github.io/langgraph/
- **FAISS 가이드**: https://github.com/facebookresearch/faiss/wiki
- **Ragas (평가)**: https://docs.ragas.io/

---

## ✅ 최종 점검

시작 전 확인:

- [ ] Python 3.9+ 설치
- [ ] OpenAI API 키 설정
- [ ] 최소 10GB 디스크 공간
- [ ] 최소 8GB RAM
- [ ] Git 저장소 초기화

준비 완료 시:

```bash
# 환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Day 1 시작!
python scripts/analyze_current_data.py
```

---

**문서 버전**: 1.0  
**최종 수정**: 2025-12-15  
**예상 완료 시간**: 2주 (하루 4-6시간 작업 기준)

**행운을 빕니다! 🚀**

