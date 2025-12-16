# Ablation Study 빠른 시작 가이드

**5분 안에 Ablation 실험 시작하기**

---

## 🎯 핵심 요약

현재 시스템은 **30+ 독립 변수**를 가진 체계적인 ablation study 플랫폼입니다:

### 주요 Ablation 요소

| 카테고리 | 구성요소 | 옵션 | 중요도 |
|---------|---------|------|-------|
| **Self-Refine** | 루프 활성화 | ON/OFF | ⭐⭐⭐⭐⭐ |
| **검색 전략** | Retrieval Mode | `hybrid` / `bm25` / `faiss` | ⭐⭐⭐⭐⭐ |
| **LLM 모델** | 모델 선택 | `gpt-4o-mini` / `gpt-4o` | ⭐⭐⭐⭐⭐ |
| **품질 평가** | 평가 방법 | LLM / Heuristic | ⭐⭐⭐⭐ |
| **쿼리 재작성** | 동적 재작성 | ON/OFF | ⭐⭐⭐⭐ |
| **Active Retrieval** | 동적 k 조정 | ON/OFF | ⭐⭐⭐ |
| **Context Manager** | 토큰 관리 | ON/OFF | ⭐⭐⭐ |
| **대화 이력** | History 포함 | ON/OFF | ⭐⭐⭐ |
| **환자 프로필** | Profile 포함 | ON/OFF | ⭐⭐⭐ |

### 8개 사전 정의 프로파일

1. **`baseline`** - 최소 기능 (Self-Refine OFF)
2. **`self_refine_heuristic`** - 휴리스틱 품질 평가
3. **`self_refine_llm_quality`** - LLM 품질 평가
4. **`self_refine_dynamic_query`** - 동적 쿼리 재작성
5. **`self_refine_full_safety`** - 전체 안전장치
6. **`full_context_engineering`** - 최종 시스템 (모든 기능 ON)
7. **`quality_check_only`** - Quality Check만
8. **`self_refine_no_safety`** - 안전장치 없음

---

## 🚀 3가지 실행 방법

### 방법 1: Batch 파일로 실행 (가장 쉬움)

#### A. 단일 테스트

```batch
run_ablation_single.bat
```

**설정 변경**: `experiments\run_ablation_single.py` 파일의 `FEATURE_CONFIG` 수정

```python
FEATURE_CONFIG = {
    'self_refine_enabled': False,  # ⭐ 이 부분 수정
    'retrieval_mode': 'hybrid',
}
```

#### B. 다중 프로파일 비교

```batch
run_ablation_comparison.bat
```

**설정 변경**: `experiments\run_ablation_comparison.py` 파일의 `PROFILES_TO_TEST` 수정

```python
PROFILES_TO_TEST = [
    "baseline",
    "full_context_engineering",  # ⭐ 비교할 프로파일 선택
]
```

#### C. 결과 분석

```batch
run_analyze_results.bat
```

자동으로 가장 최근 결과 파일을 분석합니다.

---

### 방법 2: Python 직접 실행

```bash
# 가상환경 활성화
.venv\Scripts\activate

# 단일 테스트
python experiments/run_ablation_single.py

# 다중 비교
python experiments/run_ablation_comparison.py

# 결과 분석
python experiments/analyze_ablation_results.py
```

---

### 방법 3: 코드에서 직접 사용

```python
from agent.graph import run_agent
from config.ablation_config import get_ablation_profile

# 1. 프로파일 사용
features = get_ablation_profile("full_context_engineering")

# 2. Agent 실행
result = run_agent(
    user_text="당뇨병 환자에게 메트포르민의 부작용은?",
    mode="ai_agent",
    feature_overrides=features,
    return_state=True
)

# 3. 결과 확인
print(f"품질: {result['quality_score']}")
print(f"반복: {result['iteration_count']}")
```

---

## 📊 실험 시나리오별 가이드

### 시나리오 1: Self-Refine 효과 측정 (가장 중요)

**목표**: Self-Refine이 성능 향상에 기여하는가?

**실행**:
```batch
run_ablation_comparison.bat
```

**설정** (`experiments\run_ablation_comparison.py`):
```python
PROFILES_TO_TEST = [
    "baseline",                  # Self-Refine OFF
    "full_context_engineering",  # Self-Refine ON
]
```

**예상 결과**:
- `full_context_engineering`이 `baseline`보다 품질 점수 **10-20% 향상**
- 대신 실행 시간과 비용은 증가 (반복 때문)

---

### 시나리오 2: 검색 전략 비교

**목표**: BM25 vs FAISS vs Hybrid 중 어느 것이 가장 효과적인가?

**실행**:
```python
# experiments/run_ablation_single.py 3번 실행

# 1차: BM25
FEATURE_CONFIG = {'retrieval_mode': 'bm25'}

# 2차: FAISS
FEATURE_CONFIG = {'retrieval_mode': 'faiss'}

# 3차: Hybrid
FEATURE_CONFIG = {'retrieval_mode': 'hybrid'}
```

**예상 결과**:
- `hybrid`가 가장 높은 품질 (BM25 + FAISS 장점 결합)
- `bm25`가 가장 빠름
- `faiss`가 시맨틱 유사도에서 강점

---

### 시나리오 3: LLM 품질 평가 vs 휴리스틱

**목표**: LLM 기반 품질 평가가 휴리스틱보다 나은가?

**실행**:
```batch
run_ablation_comparison.bat
```

**설정**:
```python
PROFILES_TO_TEST = [
    "self_refine_heuristic",      # 휴리스틱 평가
    "self_refine_llm_quality",    # LLM 평가
]
```

**예상 결과**:
- `LLM 평가`가 더 정확하지만 비용/시간 증가
- `휴리스틱`이 빠르고 저렴하지만 정확도 낮음

---

### 시나리오 4: 반복 횟수 최적화

**목표**: Self-Refine을 몇 번 반복하는 것이 최적인가?

**실행**:
```python
# experiments/run_ablation_single.py 수정

TEST_ITERATIONS = [0, 1, 2, 3]

for max_iter in TEST_ITERATIONS:
    FEATURE_CONFIG = {
        'self_refine_enabled': max_iter > 0,
        'max_refine_iterations': max_iter,
    }
    # 실행...
```

**예상 결과**:
- 대부분의 경우 2번 반복이 최적
- 3번은 marginal gain만 있고 비용 크게 증가

---

## 📁 결과 파일 위치

```
runs/
├── ablation_self_refine_off/
│   ├── results_20251214_120000.json  # 전체 결과
│   └── results_20251214_120000.csv   # Excel용 요약
│
├── ablation_comparison/
│   ├── comparison_20251214_130000.json  # 비교 결과
│   ├── summary_20251214_130000.csv      # 요약 테이블
│   └── charts_comparison_*.png          # 차트 (matplotlib 필요)
│
└── 2025-12-13_primary_v1/  # 멀티턴 실험
    ├── events.jsonl
    ├── node_trace.jsonl
    └── summary.json
```

---

## 🎓 논문용 최소 실험 설계

### 필수 4개 실험

| ID | 이름 | 설정 | 목적 |
|----|------|------|------|
| **Exp-A** | Baseline LLM | `mode='llm'` | 검색 없는 베이스라인 |
| **Exp-B** | Basic RAG | `self_refine_enabled=False` | 기본 RAG 성능 |
| **Exp-C** | RAG + Self-Refine | `self_refine_enabled=True` | Self-Refine 효과 |
| **Exp-D** | Full System | `full_context_engineering` | 최종 성능 |

### 실행 방법

1. **Exp-A (LLM)**: 멀티턴 실험에서 이미 수집됨
   ```batch
   5_run_multiturn_test.bat
   ```
   → `runs/.../events.jsonl`에서 `mode=llm` 필터

2. **Exp-B, C, D**: 비교 실험
   ```python
   # experiments/run_ablation_comparison.py
   PROFILES_TO_TEST = [
       "baseline",                  # Exp-B
       "self_refine_llm_quality",   # Exp-C
       "full_context_engineering",  # Exp-D
   ]
   ```

### 평가 메트릭

자동 수집:
- ✅ **Faithfulness** (근거 충실도)
- ✅ **Answer Relevance** (답변 관련성)
- ✅ **Perplexity** (불확실성)
- ✅ **Judge Total Score** (LLM 평가)
- ✅ **Iteration Count** (반복 횟수)
- ✅ **Cost & Time** (비용 & 시간)

---

## ⚡ 빠른 테스트 (5분)

### 1단계: 프로파일 확인

```bash
python -c "from config.ablation_config import print_ablation_profiles; print_ablation_profiles()"
```

### 2단계: 간단한 비교

```python
# experiments/run_ablation_comparison.py 수정
PROFILES_TO_TEST = ["baseline", "full_context_engineering"]
TEST_QUERIES = TEST_QUERIES[:3]  # 3개만
```

```batch
run_ablation_comparison.bat
```

### 3단계: 결과 확인

```batch
run_analyze_results.bat
```

**예상 출력**:
```
프로파일                          품질   반복   문서   시간(s)
baseline                        0.650    0.0    8.0      3.2
full_context_engineering        0.820    1.8    9.2      8.5
```

---

## 🔧 커스터마이징 가이드

### Feature Flags 직접 제어

```python
custom_features = {
    # Self-Refine 관련
    'self_refine_enabled': True,
    'max_refine_iterations': 2,
    'quality_threshold': 0.5,

    # 검색 전략
    'retrieval_mode': 'hybrid',  # hybrid/bm25/faiss
    'active_retrieval_enabled': False,

    # Context 관련
    'include_history': True,
    'include_profile': True,
    'use_context_manager': True,

    # 고급 기능
    'response_cache_enabled': False,
    'context_compression_enabled': False,
}

result = run_agent(
    user_text="쿼리",
    feature_overrides=custom_features,
    return_state=True
)
```

### 새 프로파일 추가

`config/ablation_config.py` 수정:

```python
ABLATION_PROFILES = {
    # ... 기존 프로파일들 ...

    "my_custom_profile": {
        "description": "내 커스텀 설정",
        "features": {
            "self_refine_enabled": True,
            "max_refine_iterations": 3,
            # ... 원하는 설정 ...
        }
    },
}
```

---

## 🐛 문제 해결

### 문제 1: 실행 안 됨

```
[오류] 모듈을 찾을 수 없습니다
```

**해결**:
```bash
# 가상환경 활성화 확인
.venv\Scripts\activate

# 패키지 설치 확인
pip install -r requirements.txt
```

---

### 문제 2: API 오류

```
OpenAI API Error
```

**해결**:
1. `.env` 파일에서 `OPENAI_API_KEY` 확인
2. API 키 유효성 확인
3. Rate limit 확인 (너무 빠르게 호출 시)

---

### 문제 3: 메모리 부족

**해결**:
- 쿼리 수 줄이기: `TEST_QUERIES[:5]`
- 프로파일 수 줄이기
- 배치 크기 줄이기

---

## 📚 더 자세한 정보

- **전체 가이드**: [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md)
- **설정 파일**: [config/ablation_config.py](config/ablation_config.py)
- **실험 스크립트**: `experiments/run_ablation_*.py`

---

## ✅ 체크리스트

실험 시작 전:
- [ ] API 키 설정 확인
- [ ] 가상환경 활성화
- [ ] 테스트 쿼리 준비
- [ ] 실험 이름 정의
- [ ] 예상 시간/비용 계산

실험 후:
- [ ] 결과 파일 확인
- [ ] 통계 요약 확인
- [ ] 차트 생성
- [ ] Git commit (재현성)

---

**마지막 업데이트**: 2025-12-14
**작성자**: Medical AI Agent Team

---

## 🎯 다음 단계

1. ✅ **Quick Test** (5분): `run_ablation_comparison.bat` 실행
2. ✅ **결과 확인**: `run_analyze_results.bat`
3. ✅ **Full Test** (1시간): 전체 프로파일 비교
4. ✅ **논문 작성**: 결과를 표와 그래프로 정리

**시작하기**: `run_ablation_comparison.bat` 더블클릭! 🚀