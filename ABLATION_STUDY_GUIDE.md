# Ablation Study 종합 가이드

**Medical AI Agent - 체계적 Ablation 연구 설계 및 실행 가이드**

작성일: 2025-12-14
버전: 1.0

---

## 목차

1. [개요](#1-개요)
2. [설계된 Ablation 요소 전체 목록](#2-설계된-ablation-요소-전체-목록)
3. [사전 정의된 Ablation 프로파일](#3-사전-정의된-ablation-프로파일)
4. [개별 테스트 실행 방법](#4-개별-테스트-실행-방법)
5. [효율적인 Ablation 연구 전략](#5-효율적인-ablation-연구-전략)
6. [실험 설계 추천안](#6-실험-설계-추천안)
7. [메트릭 수집 및 분석](#7-메트릭-수집-및-분석)

---

## 1. 개요

### 1.1 Ablation Study란?

Ablation study는 시스템의 **각 구성요소를 제거하거나 변형**하여 그 구성요소가 전체 성능에 미치는 영향을 정량적으로 측정하는 연구 방법입니다.

### 1.2 본 시스템의 Ablation 설계 철학

본 Medical AI Agent 시스템은 다음과 같은 설계 원칙으로 ablation study를 지원합니다:

- ✅ **코드 수정 없이** YAML/Python config만으로 모든 기능 on/off 가능
- ✅ **30+ 독립적인 feature flags**로 세밀한 제어 가능
- ✅ **8개 사전 정의 프로파일**로 빠른 실험 가능
- ✅ **자동 메트릭 수집**으로 재현 가능한 실험 보장
- ✅ **LangGraph 기반 모듈형 설계**로 노드 단위 분석 가능

---

## 2. 설계된 Ablation 요소 전체 목록

### 2.1 핵심 Ablation Axes (독립 변수)

| # | 카테고리 | 구성요소 | 설정 위치 | 옵션 | 영향도 |
|---|---------|---------|----------|------|-------|
| 1 | **LLM 모델** | 모델 선택 | `config/model_config.yaml` | `gpt-4o-mini` / `gpt-4o` / `gemini-2.0-flash` | ⭐⭐⭐⭐⭐ |
| 2 | **검색 전략** | Retrieval Mode | `feature_flags['retrieval_mode']` | `hybrid` / `bm25` / `faiss` | ⭐⭐⭐⭐⭐ |
| 3 | **Self-Refine** | 루프 활성화 | `feature_flags['self_refine_enabled']` | `True` / `False` | ⭐⭐⭐⭐⭐ |
| 4 | **품질 평가** | 평가 방법 | `feature_flags['llm_based_quality_check']` | LLM / Heuristic | ⭐⭐⭐⭐ |
| 5 | **쿼리 재작성** | 동적 재작성 | `feature_flags['dynamic_query_rewrite']` | `True` / `False` | ⭐⭐⭐⭐ |
| 6 | **Active Retrieval** | 동적 k 조정 | `feature_flags['active_retrieval_enabled']` | `True` / `False` | ⭐⭐⭐ |
| 7 | **Context Manager** | 토큰 예산 관리 | `feature_flags['use_context_manager']` | `True` / `False` | ⭐⭐⭐ |
| 8 | **Response Cache** | 응답 캐싱 | `feature_flags['response_cache_enabled']` | `True` / `False` | ⭐⭐⭐ |
| 9 | **대화 이력** | History 포함 | `feature_flags['include_history']` | `True` / `False` | ⭐⭐⭐ |
| 10 | **환자 프로필** | Profile 포함 | `feature_flags['include_profile']` | `True` / `False` | ⭐⭐⭐ |
| 11 | **Embedding** | 임베딩 모델 | `config/corpus_config.yaml` | `text-embedding-3-large` / `-small` | ⭐⭐ |
| 12 | **Chunking** | 청크 크기 | `config/corpus_config.yaml` | 500-1500 tokens | ⭐⭐ |
| 13 | **Context Compression** | 압축 전략 | `feature_flags['context_compression_enabled']` | `extractive` / `abstractive` / `hybrid` | ⭐⭐ |
| 14 | **Hierarchical Memory** | 계층적 메모리 | `feature_flags['hierarchical_memory_enabled']` | `True` / `False` | ⭐⭐ |
| 15 | **Dynamic Routing** | 의도 기반 라우팅 | `feature_flags['dynamic_rag_routing']` | `True` / `False` | ⭐ |

### 2.2 파라미터 수준 Ablation

| 파라미터 | 설정 위치 | 테스트 범위 | 추천 값 |
|---------|----------|-----------|--------|
| `max_refine_iterations` | `feature_flags` | 0, 1, 2, 3 | 2 |
| `quality_threshold` | `feature_flags` | 0.3, 0.5, 0.6, 0.8 | 0.5-0.6 |
| `temperature` | `config/model_config.yaml` | 0.0, 0.2, 0.5, 0.7, 1.0 | 0.2-0.7 |
| `top_k` (BM25) | `config/corpus_config.yaml` | 3, 5, 8, 10, 15 | 8 |
| `top_k` (FAISS) | `config/corpus_config.yaml` | 3, 5, 8, 10, 15 | 8 |
| `rrf_k` (Fusion) | `config/corpus_config.yaml` | 20, 40, 60, 80 | 60 |
| `chunk_size` | `config/corpus_config.yaml` | 500, 700, 900, 1200 | 900 |
| `chunk_overlap` | `config/corpus_config.yaml` | 0, 100, 200, 300 | 200 |
| `cache_similarity_threshold` | `feature_flags` | 0.7, 0.8, 0.85, 0.9 | 0.85 |

---

## 3. 사전 정의된 Ablation 프로파일

시스템에는 8개의 사전 정의된 프로파일이 제공됩니다 ([config/ablation_config.py](config/ablation_config.py#L15)):

### 3.1 프로파일 전체 목록

| 프로파일 이름 | 설명 | 주요 특징 | 연구 목적 |
|-------------|------|----------|---------|
| `baseline` | 베이스라인 | Self-Refine OFF, 모든 안전장치 OFF | 최소 기능 성능 측정 |
| `self_refine_heuristic` | 휴리스틱 품질 평가 | Self-Refine ON, LLM 평가 OFF | 휴리스틱의 효과 측정 |
| `self_refine_llm_quality` | LLM 품질 평가 | Self-Refine ON, LLM 평가 ON, 정적 쿼리 | LLM 평가의 가치 측정 |
| `self_refine_dynamic_query` | 동적 쿼리 재작성 | Self-Refine ON, 동적 재작성 ON | 쿼리 재작성 효과 측정 |
| `self_refine_full_safety` | 전체 안전장치 | 중복 검출 + 진행도 모니터링 | 안전장치 필요성 검증 |
| `full_context_engineering` | 최종 시스템 | 모든 기능 ON, 높은 품질 기준 | 최대 성능 측정 |
| `quality_check_only` | 품질 검사만 | Self-Refine OFF, Quality Check ON | Quality Check 단독 효과 |
| `self_refine_no_safety` | 안전장치 없음 | Self-Refine ON, 안전장치 OFF | 안전장치 필요성 검증 |

### 3.2 프로파일 사용 예시

```python
from config.ablation_config import get_ablation_profile, list_ablation_profiles

# 1. 사용 가능한 프로파일 목록 확인
profiles = list_ablation_profiles()
for name, desc in profiles.items():
    print(f"{name}: {desc}")

# 2. 특정 프로파일 로드
features = get_ablation_profile("full_context_engineering")

# 3. Agent 실행
from agent.graph import run_agent

result = run_agent(
    user_text="당뇨병 환자에게 메트포르민의 부작용은?",
    mode="ai_agent",
    feature_overrides=features,  # 프로파일 적용
    return_state=True
)

# 4. 결과 분석
print(f"품질 점수: {result['quality_score']}")
print(f"반복 횟수: {result['iteration_count']}")
print(f"검색 문서 수: {len(result['retrieved_docs'])}")
```

---

## 4. 개별 테스트 실행 방법

### 4.1 방법 1: Python 스크립트로 실행

**파일 생성**: `experiments/run_ablation_single.py`

```python
"""
단일 Ablation 테스트 실행 스크립트
"""
import json
from pathlib import Path
from agent.graph import run_agent
from datetime import datetime

# ============ 설정 ============
ABLATION_NAME = "self_refine_off"  # 실험 이름
TEST_QUERIES = [
    "당뇨병 환자에게 메트포르민의 부작용은?",
    "고혈압 환자의 식이요법은?",
    "아스피린을 복용하는 환자가 피해야 할 음식은?",
]

# Feature flags 설정
FEATURE_CONFIG = {
    'self_refine_enabled': False,  # 테스트 변수
    'retrieval_mode': 'hybrid',
    'active_retrieval_enabled': False,
}

# ============ 실행 ============
results = []

for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n[{i}/{len(TEST_QUERIES)}] 실행 중: {query}")

    result = run_agent(
        user_text=query,
        mode="ai_agent",
        feature_overrides=FEATURE_CONFIG,
        return_state=True
    )

    # 메트릭 추출
    metrics = {
        'query': query,
        'answer': result['answer'],
        'quality_score': result.get('quality_score', 0.0),
        'iteration_count': result.get('iteration_count', 0),
        'num_docs': len(result.get('retrieved_docs', [])),
        'cache_hit': result.get('cache_hit', False),
    }

    results.append(metrics)
    print(f"  → 품질: {metrics['quality_score']:.2f}, 문서: {metrics['num_docs']}")

# ============ 결과 저장 ============
output_dir = Path(f"runs/ablation_{ABLATION_NAME}")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'ablation_name': ABLATION_NAME,
        'feature_config': FEATURE_CONFIG,
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'avg_quality': sum(r['quality_score'] for r in results) / len(results),
            'avg_iterations': sum(r['iteration_count'] for r in results) / len(results),
            'avg_docs': sum(r['num_docs'] for r in results) / len(results),
            'cache_hit_rate': sum(r['cache_hit'] for r in results) / len(results),
        }
    }, f, ensure_ascii=False, indent=2)

print(f"\n✅ 결과 저장됨: {output_file}")
```

**실행**:
```bash
python experiments/run_ablation_single.py
```

### 4.2 방법 2: Batch 파일로 실행 (Windows)

**파일 생성**: `run_ablation_test.bat`

```batch
@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ========================================
echo Ablation Test 실행
echo ========================================

REM 가상환경 활성화
call .venv\Scripts\activate.bat

REM 테스트 이름 설정
set TEST_NAME=%1
if "%TEST_NAME%"=="" set TEST_NAME=default

echo.
echo [실행] Ablation Test: %TEST_NAME%
echo.

REM Python 스크립트 실행
python experiments/run_ablation_single.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [오류] 실행 실패
    pause
    exit /b 1
)

echo.
echo [완료] 결과가 runs/ 디렉토리에 저장되었습니다.
echo.
pause
```

### 4.3 방법 3: 다중 프로파일 비교 실험

**파일 생성**: `experiments/run_ablation_comparison.py`

```python
"""
다중 Ablation 프로파일 비교 실험
"""
import json
from pathlib import Path
from agent.graph import run_agent
from config.ablation_config import ABLATION_PROFILES, get_ablation_profile
from datetime import datetime
import time

# ============ 설정 ============
# 비교할 프로파일 목록
PROFILES_TO_TEST = [
    "baseline",
    "self_refine_heuristic",
    "self_refine_llm_quality",
    "full_context_engineering",
]

# 테스트 쿼리 (적은 수로 빠른 비교)
TEST_QUERIES = [
    "당뇨병 환자에게 메트포르민의 부작용은?",
    "고혈압 환자의 식이요법은?",
    "아스피린 복용 시 피해야 할 음식은?",
    "임신 중 복용 가능한 진통제는?",
    "간 질환 환자에게 금기인 약물은?",
]

# ============ 실행 ============
all_results = {}

for profile_name in PROFILES_TO_TEST:
    print(f"\n{'='*60}")
    print(f"프로파일: {profile_name}")
    print(f"{'='*60}")

    features = get_ablation_profile(profile_name)
    profile_results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"  [{i}/{len(TEST_QUERIES)}] {query[:30]}...")

        start_time = time.time()
        result = run_agent(
            user_text=query,
            mode="ai_agent",
            feature_overrides=features,
            return_state=True
        )
        elapsed = time.time() - start_time

        metrics = {
            'query': query,
            'quality_score': result.get('quality_score', 0.0),
            'iteration_count': result.get('iteration_count', 0),
            'num_docs': len(result.get('retrieved_docs', [])),
            'elapsed_sec': elapsed,
        }

        profile_results.append(metrics)
        print(f"    → Q={metrics['quality_score']:.2f}, Iter={metrics['iteration_count']}, Time={elapsed:.1f}s")

    all_results[profile_name] = {
        'feature_config': features,
        'results': profile_results,
        'summary': {
            'avg_quality': sum(r['quality_score'] for r in profile_results) / len(profile_results),
            'avg_iterations': sum(r['iteration_count'] for r in profile_results) / len(profile_results),
            'avg_docs': sum(r['num_docs'] for r in profile_results) / len(profile_results),
            'avg_time': sum(r['elapsed_sec'] for r in profile_results) / len(profile_results),
        }
    }

# ============ 비교 테이블 출력 ============
print(f"\n{'='*80}")
print("비교 결과 요약")
print(f"{'='*80}")
print(f"{'프로파일':<30} {'품질':>8} {'반복':>6} {'문서':>6} {'시간(s)':>8}")
print(f"{'-'*80}")

for profile_name, data in all_results.items():
    s = data['summary']
    print(f"{profile_name:<30} {s['avg_quality']:>8.3f} {s['avg_iterations']:>6.1f} {s['avg_docs']:>6.1f} {s['avg_time']:>8.1f}")

# ============ 결과 저장 ============
output_dir = Path("runs/ablation_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'profiles_tested': PROFILES_TO_TEST,
        'num_queries': len(TEST_QUERIES),
        'results': all_results,
    }, f, ensure_ascii=False, indent=2)

print(f"\n✅ 결과 저장: {output_file}")
```

**실행**:
```bash
python experiments/run_ablation_comparison.py
```

### 4.4 방법 4: 단일 변수 Sweep 실험

특정 변수의 여러 값을 테스트하는 실험입니다.

**예시**: `max_refine_iterations` 0, 1, 2, 3 비교

```python
"""
Single Variable Sweep: max_refine_iterations
"""
from agent.graph import run_agent

TEST_VALUES = [0, 1, 2, 3]
TEST_QUERY = "당뇨병 환자에게 메트포르민의 부작용은?"

results = {}

for max_iter in TEST_VALUES:
    print(f"\n[테스트] max_refine_iterations = {max_iter}")

    result = run_agent(
        user_text=TEST_QUERY,
        mode="ai_agent",
        feature_overrides={
            'self_refine_enabled': True if max_iter > 0 else False,
            'max_refine_iterations': max_iter,
        },
        return_state=True
    )

    results[max_iter] = {
        'quality': result['quality_score'],
        'actual_iters': result['iteration_count'],
    }

    print(f"  품질: {results[max_iter]['quality']:.2f}")
    print(f"  실제 반복: {results[max_iter]['actual_iters']}")

# 결과 분석
print("\n=== 결과 요약 ===")
for max_iter, data in results.items():
    print(f"Max={max_iter}: 품질={data['quality']:.2f}, 실제반복={data['actual_iters']}")
```

---

## 5. 효율적인 Ablation 연구 전략

### 5.1 단계별 Ablation 전략

#### Phase 1: 핵심 구성요소 (최우선)

**목표**: 시스템 성능에 가장 큰 영향을 미치는 구성요소 식별

| 실험 | 비교 대상 | 예상 소요 | 중요도 |
|-----|---------|----------|-------|
| **E1: Self-Refine 효과** | `baseline` vs `full_context_engineering` | 1시간 | ⭐⭐⭐⭐⭐ |
| **E2: Retrieval 전략** | `hybrid` vs `bm25` vs `faiss` | 1시간 | ⭐⭐⭐⭐⭐ |
| **E3: LLM 모델 비교** | `gpt-4o-mini` vs `gpt-4o` | 2시간 | ⭐⭐⭐⭐⭐ |
| **E4: 품질 평가 방법** | `llm_based` vs `heuristic` | 30분 | ⭐⭐⭐⭐ |

**실행 스크립트**:
```bash
# E1: Self-Refine 효과
python experiments/run_ablation_comparison.py --profiles baseline full_context_engineering

# E2: Retrieval 전략
python experiments/run_ablation_single.py --retrieval_mode hybrid
python experiments/run_ablation_single.py --retrieval_mode bm25
python experiments/run_ablation_single.py --retrieval_mode faiss
```

#### Phase 2: Self-Refine 최적화

**목표**: Self-Refine의 최적 구성 찾기

| 실험 | 변수 | 테스트 값 | 소요 |
|-----|------|----------|------|
| **E5: Iteration 횟수** | `max_refine_iterations` | 0, 1, 2, 3 | 1시간 |
| **E6: 품질 임계값** | `quality_threshold` | 0.3, 0.5, 0.6, 0.8 | 1시간 |
| **E7: 동적 쿼리 재작성** | `dynamic_query_rewrite` | True vs False | 30분 |
| **E8: 안전장치** | `duplicate_detection`, `progress_monitoring` | ON vs OFF | 30분 |

#### Phase 3: Context Engineering

**목표**: 컨텍스트 구성 최적화

| 실험 | 비교 대상 | 소요 |
|-----|---------|------|
| **E9: 대화 이력** | `include_history` True vs False | 30분 |
| **E10: 환자 프로필** | `include_profile` True vs False | 30분 |
| **E11: Context Manager** | `use_context_manager` True vs False | 30분 |

#### Phase 4: 고급 기능

**목표**: 추가 최적화 기능의 가치 평가

| 실험 | 기능 | 소요 |
|-----|------|------|
| **E12: Active Retrieval** | `active_retrieval_enabled` | 1시간 |
| **E13: Response Cache** | `response_cache_enabled` | 1시간 |
| **E14: Context Compression** | `compression_strategy` | 1시간 |
| **E15: Hierarchical Memory** | `hierarchical_memory_enabled` | 1시간 |

### 5.2 실험 우선순위 Matrix

```
영향도 vs 복잡도 Matrix:

High Impact │ E1(Self-Refine) │ E2(Retrieval) │
            │ E3(LLM Model)   │ E4(Quality)   │
            ├─────────────────┼───────────────┤
Medium      │ E5(Iterations)  │ E12(Active)   │
Impact      │ E9(History)     │ E13(Cache)    │
            ├─────────────────┼───────────────┤
Low Impact  │ E14(Compress)   │ E15(Memory)   │
            │                 │               │
            └─────────────────┴───────────────┘
              Low Complexity    High Complexity
```

**추천 순서**: E1 → E2 → E3 → E4 → E5 → E9 → E10 → ...

### 5.3 시간별 실험 계획

#### 🕐 1시간 Quick Test
```python
profiles = ["baseline", "full_context_engineering"]
queries = TEST_QUERIES[:5]  # 5개만
```

#### 🕒 3시간 Core Test
```python
profiles = [
    "baseline",
    "self_refine_heuristic",
    "self_refine_llm_quality",
    "full_context_engineering"
]
queries = TEST_QUERIES[:10]  # 10개
```

#### 🕔 1일 Full Test
```python
# 전체 80 환자 x 5턴 실험
bash 5_run_multiturn_test.bat
```

---

## 6. 실험 설계 추천안

### 6.1 최소 실험 설계 (논문용)

**목표**: 논문에 포함할 최소한의 ablation 결과

| 실험 ID | 이름 | 설정 | 목적 |
|--------|------|------|------|
| **Exp-A** | Baseline LLM | `mode='llm'` | 검색 없는 베이스라인 |
| **Exp-B** | Basic RAG | `mode='ai_agent'`, `self_refine_enabled=False` | 기본 RAG 성능 |
| **Exp-C** | RAG + Self-Refine | `mode='ai_agent'`, `self_refine_enabled=True` | Self-Refine 효과 |
| **Exp-D** | Full System | `full_context_engineering` 프로파일 | 최종 시스템 성능 |

**평가 메트릭**:
- Faithfulness (근거 충실도)
- Answer Relevance (답변 관련성)
- Perplexity (불확실성)
- Judge Total Score (LLM 평가 점수)

**예상 결과**:
```
Exp-A (Baseline) < Exp-B (Basic RAG) < Exp-C (Self-Refine) < Exp-D (Full)
```

### 6.2 중간 실험 설계

**추가 실험**:

| 실험 ID | 이름 | 차이점 | 목적 |
|--------|------|--------|------|
| **Exp-E** | BM25 Only | `retrieval_mode='bm25'` | 키워드 검색 성능 |
| **Exp-F** | FAISS Only | `retrieval_mode='faiss'` | 시맨틱 검색 성능 |
| **Exp-G** | Hybrid | `retrieval_mode='hybrid'` | 하이브리드 검색 효과 |
| **Exp-H** | Heuristic Quality | `llm_based_quality_check=False` | 휴리스틱 평가 효율성 |

### 6.3 전체 실험 설계 (연구용)

**30+ 실험 조합**:

```python
# experiments/run_full_ablation_study.py
FULL_ABLATION_MATRIX = {
    'llm_model': ['gpt-4o-mini', 'gpt-4o'],
    'retrieval_mode': ['bm25', 'faiss', 'hybrid'],
    'self_refine_enabled': [False, True],
    'llm_based_quality_check': [False, True],
    'dynamic_query_rewrite': [False, True],
    'active_retrieval_enabled': [False, True],
}

# 2 x 3 x 2 x 2 x 2 x 2 = 96 조합
# 실제로는 invalid 조합 제거 후 ~50개
```

---

## 7. 메트릭 수집 및 분석

### 7.1 자동 수집되는 메트릭

실험 실행 시 다음 메트릭이 자동으로 수집됩니다:

#### 성능 메트릭
- `quality_score`: 품질 점수 (0.0-1.0)
- `faithfulness`: 근거 충실도
- `answer_relevance`: 답변 관련성
- `perplexity`: 불확실성 점수
- `judge_total_score`: LLM 평가 점수

#### 효율성 메트릭
- `iteration_count`: 실제 반복 횟수
- `retrieval_time_ms`: 검색 소요 시간
- `generation_time_ms`: 생성 소요 시간
- `total_tokens`: 총 토큰 사용량
- `estimated_cost_usd`: 예상 비용

#### 동작 메트릭
- `num_docs_retrieved`: 검색된 문서 수
- `dynamic_k`: Active Retrieval의 동적 k 값
- `query_complexity`: 쿼리 복잡도 (simple/moderate/complex)
- `cache_hit`: 캐시 히트 여부
- `compression_ratio`: 압축 비율

### 7.2 메트릭 접근 방법

```python
result = run_agent(
    user_text="쿼리",
    mode="ai_agent",
    feature_overrides={...},
    return_state=True  # ⭐ 중요: 전체 상태 반환
)

# 메트릭 추출
print(f"품질: {result['quality_score']}")
print(f"반복: {result['iteration_count']}")
print(f"문서: {len(result['retrieved_docs'])}")
print(f"토큰: {result.get('total_tokens', 0)}")

# Refine 로그 확인
for log in result.get('refine_iteration_logs', []):
    print(f"Iter {log['iteration']}: Q={log['quality_score']:.2f}")
```

### 7.3 결과 분석 스크립트

**파일 생성**: `experiments/analyze_ablation_results.py`

```python
"""
Ablation 결과 분석 및 시각화
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============ 결과 로드 ============
results_file = Path("runs/ablation_comparison/comparison_20251214_120000.json")
with open(results_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ============ DataFrame 변환 ============
rows = []
for profile_name, profile_data in data['results'].items():
    summary = profile_data['summary']
    rows.append({
        'Profile': profile_name,
        'Avg Quality': summary['avg_quality'],
        'Avg Iterations': summary['avg_iterations'],
        'Avg Docs': summary['avg_docs'],
        'Avg Time (s)': summary['avg_time'],
    })

df = pd.DataFrame(rows)

# ============ 통계 출력 ============
print("=== Ablation Study Results ===")
print(df.to_string(index=False))

# ============ 시각화 ============
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 품질 비교
axes[0, 0].bar(df['Profile'], df['Avg Quality'], color='skyblue')
axes[0, 0].set_title('Average Quality Score')
axes[0, 0].set_ylabel('Quality')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. 반복 횟수
axes[0, 1].bar(df['Profile'], df['Avg Iterations'], color='lightcoral')
axes[0, 1].set_title('Average Iterations')
axes[0, 1].set_ylabel('Iterations')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. 문서 수
axes[1, 0].bar(df['Profile'], df['Avg Docs'], color='lightgreen')
axes[1, 0].set_title('Average Documents Retrieved')
axes[1, 0].set_ylabel('Docs')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. 실행 시간
axes[1, 1].bar(df['Profile'], df['Avg Time (s)'], color='gold')
axes[1, 1].set_title('Average Execution Time')
axes[1, 1].set_ylabel('Time (s)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(results_file.parent / 'analysis_charts.png', dpi=300)
print(f"\n✅ 차트 저장: {results_file.parent / 'analysis_charts.png'}")
```

---

## 8. 체크리스트 및 베스트 프랙티스

### 8.1 실험 시작 전 체크리스트

- [ ] API 키 설정 확인 (`OPENAI_API_KEY`)
- [ ] 가상환경 활성화 (`.venv`)
- [ ] 테스트 쿼리 준비 (최소 5개 이상)
- [ ] 실험 이름 및 목적 명확히 정의
- [ ] 예상 소요 시간 및 비용 계산
- [ ] 결과 저장 경로 확인 (`runs/` 디렉토리)

### 8.2 실험 중 모니터링

- [ ] 로그 출력 확인 (오류 없는지)
- [ ] 메트릭 수집 확인 (quality_score 등)
- [ ] 시간 초과 없는지 확인 (timeout)
- [ ] API 호출 제한 없는지 확인 (rate limit)

### 8.3 실험 후 분석

- [ ] 결과 파일 저장 확인 (JSON)
- [ ] 통계 요약 계산 (평균, 표준편차)
- [ ] 시각화 차트 생성
- [ ] 논문/보고서에 포함할 표 작성
- [ ] 코드 및 설정 버전 기록 (Git commit)

### 8.4 재현성 보장

- [ ] Git commit hash 기록
- [ ] Python 패키지 버전 기록 (`pip freeze`)
- [ ] 사용한 설정 파일 저장 (YAML/JSON)
- [ ] Random seed 설정 (`global_seed: 42`)
- [ ] 데이터셋 버전 기록 (corpus hash)

---

## 9. 빠른 참조 (Quick Reference)

### 9.1 핵심 명령어

```bash
# 1. 프로파일 목록 확인
python -c "from config.ablation_config import print_ablation_profiles; print_ablation_profiles()"

# 2. 단일 테스트 실행
python experiments/run_ablation_single.py

# 3. 다중 프로파일 비교
python experiments/run_ablation_comparison.py

# 4. 전체 멀티턴 실험
5_run_multiturn_test.bat

# 5. 결과 분석
python experiments/analyze_ablation_results.py
```

### 9.2 주요 파일 위치

| 파일 | 경로 | 용도 |
|-----|------|------|
| Ablation 프로파일 | `config/ablation_config.py` | 사전 정의 프로파일 |
| Feature Flags | `agent/graph.py` (line 196-243) | 기능 토글 설정 |
| 실험 설정 | `experiments/config.yaml` | 멀티턴 실험 설정 |
| 모델 설정 | `config/model_config.yaml` | LLM 모델 선택 |
| 검색 설정 | `config/corpus_config.yaml` | 검색 파라미터 |

### 9.3 자주 사용하는 Feature Flags

```python
# 최소 기능 (베이스라인)
{'self_refine_enabled': False}

# Self-Refine 활성화
{'self_refine_enabled': True, 'max_refine_iterations': 2}

# 하이브리드 검색
{'retrieval_mode': 'hybrid'}

# Active Retrieval
{'active_retrieval_enabled': True, 'dynamic_k': True}

# 전체 활성화
get_ablation_profile("full_context_engineering")
```

---

## 10. 문제 해결 (Troubleshooting)

### 10.1 일반적인 오류

**문제**: `ValueError: 존재하지 않는 ablation 프로파일`
- **해결**: 프로파일 이름 확인 (`list_ablation_profiles()` 실행)

**문제**: API 호출 오류 (`OpenAI API Error`)
- **해결**: API 키 확인, 요청 제한(rate limit) 확인

**문제**: 메모리 부족 (`MemoryError`)
- **해결**: 배치 크기 줄이기, 쿼리 수 줄이기

### 10.2 성능 최적화

- **실험 속도 향상**: `temperature=0.2`로 낮춰서 빠른 응답
- **비용 절감**: `gpt-4o-mini` 사용, 쿼리 수 줄이기
- **병렬 실행**: 여러 프로파일을 동시에 실행 (주의: rate limit)

---

## 11. 결론 및 권장 사항

### 11.1 핵심 요약

본 Medical AI Agent 시스템은 **30+ 독립 변수**를 가진 체계적인 ablation study 플랫폼입니다:

✅ **8개 사전 정의 프로파일**로 빠른 실험 가능
✅ **코드 수정 없이** YAML/Python config만으로 제어
✅ **자동 메트릭 수집**으로 재현 가능한 실험
✅ **LangGraph 모듈형 설계**로 노드 단위 분석 가능

### 11.2 추천 실험 순서

1. **Quick Test** (1시간): E1(Self-Refine), E2(Retrieval)
2. **Core Test** (3시간): E3(LLM Model), E4(Quality)
3. **Full Test** (1일): 80 환자 x 5턴 멀티턴 실험

### 11.3 논문 작성 시 권장 사항

- **Table 1**: 4개 주요 프로파일 비교 (Exp-A ~ Exp-D)
- **Figure 1**: 품질 점수 비교 차트
- **Figure 2**: Iteration별 품질 향상 그래프
- **Table 2**: Ablation study 전체 결과 (10+ 실험)

---

**문서 버전**: 1.0
**최종 수정**: 2025-12-14
**작성자**: Medical AI Agent Research Team
**연락처**: GitHub Issues

---

## 부록 A: 전체 Feature Flags 목록

```python
ALL_FEATURE_FLAGS = {
    # Self-Refine 관련
    'self_refine_enabled': True,
    'max_refine_iterations': 2,
    'quality_threshold': 0.5,
    'llm_based_quality_check': True,
    'dynamic_query_rewrite': True,
    'quality_check_enabled': True,
    'duplicate_detection': True,
    'progress_monitoring': True,

    # 검색 관련
    'retrieval_mode': 'hybrid',  # hybrid/bm25/faiss
    'active_retrieval_enabled': False,
    'default_k': 8,
    'simple_query_k': 3,
    'moderate_query_k': 8,
    'complex_query_k': 15,
    'dynamic_rag_routing': False,

    # Context 관련
    'use_context_manager': True,
    'include_history': True,
    'include_profile': True,
    'include_longterm': False,
    'include_evidence': True,
    'include_personalization': True,
    'budget_aware_retrieval': True,
    'avg_doc_tokens': 200,

    # 메모리 관련
    'profile_update_enabled': True,
    'temporal_weight_enabled': True,
    'response_cache_enabled': True,
    'cache_similarity_threshold': 0.85,
    'style_variation_level': 0.3,

    # 고급 기능
    'context_compression_enabled': False,
    'compression_strategy': 'extractive',
    'compression_target_ratio': 0.5,
    'hierarchical_memory_enabled': False,
    'working_memory_capacity': 5,
    'compression_threshold': 5,
}
```

## 부록 B: 결과 파일 구조

```
runs/
├── ablation_baseline/
│   └── results_20251214_120000.json
├── ablation_comparison/
│   ├── comparison_20251214_130000.json
│   └── analysis_charts.png
└── 2025-12-13_primary_v1/
    ├── events.jsonl          # 턴별 메트릭
    ├── node_trace.jsonl      # 노드 실행 로그
    └── summary.json          # 집계 통계
```

---

**END OF DOCUMENT**