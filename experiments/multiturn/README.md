# 멀티턴 대화 테스트 시스템

재현성과 객관성을 확보한 멀티턴 대화 평가 프레임워크

## 개요

이 시스템은 Basic RAG vs Agentic RAG 비교 실험을 위한 **프로토콜 기반 멀티턴 대화 생성 및 평가** 도구입니다.

### 핵심 특징

1. **재현성 확보**
   - 고정된 턴 타입 프로토콜 (P6)
   - 슬롯 기반 템플릿 은행
   - 결정론적 환자 응답 생성 (temperature=0, seed 고정)

2. **객관성 확보**
   - 서브스코어 기반 평가 루브릭
   - 코호트 분리 + 게이팅으로 공정한 비교
   - LLM-as-a-Judge 지원

3. **차별화된 평가**
   - Agentic RAG의 강점(컨텍스트/메모리/근거) 정량화
   - 턴 타입별 세부 분석

## 디렉토리 구조

```
experiments/multiturn/
├── README.md                      # 이 파일
├── patient_scenarios.py           # 환자 시나리오 생성기
├── question_bank.py               # 질문은행 (템플릿 + 게이팅)
├── multiturn_simulator.py         # 대화 시뮬레이터
├── evaluation_rubric.py           # 평가 루브릭
└── run_multiturn_experiment.py    # 메인 실험 러너
```

## 설치

```bash
# 프로젝트 루트에서
pip install -r requirements.txt

# 멀티턴 실험용 디렉토리 생성
mkdir -p data/multiturn
mkdir -p results/multiturn
```

## 사용법

### 1. 환자 시나리오 생성

```bash
cd experiments/multiturn
python patient_scenarios.py
```

**출력**: `data/multiturn/patient_scenarios.json` (13개 시나리오)

### 2. 질문은행 확인

```bash
python question_bank.py
```

**출력**: 턴 타입별 템플릿 개수 및 샘플

### 3. 대화 시뮬레이션 테스트

```bash
python multiturn_simulator.py
```

**출력**: `data/multiturn/sample_dialogue.json` (샘플 대화 1개)

### 4. 전체 실험 실행

```bash
python run_multiturn_experiment.py \
  --output-dir results/multiturn \
  --protocol P6 \
  --seed 42 \
  --num-scenarios 13 \
  --mode cooperative
```

**주요 옵션**:
- `--output-dir`: 결과 저장 디렉토리
- `--protocol`: 프로토콜 (현재 P6만 지원)
- `--seed`: 랜덤 시드 (재현성)
- `--num-scenarios`: 실험할 시나리오 개수
- `--mode`: 환자 응답 모드 (`cooperative`, `minimal`, `noisy`)
- `--skip-generation`: 대화 생성 스킵 (기존 대화 재사용)

### 5. 결과 확인

```bash
# 결과 파일
ls results/multiturn/

# basic_rag_results.json
# agentic_rag_results.json
# dialogues/D-P-F001-*.json
```

## 프로토콜 (P6)

### 턴 구조

| Turn | 타입 | 목적 | 평가 포인트 |
|------|------|------|-------------|
| T1 | L1: 단순 사실 조회 | 기본 슬롯 추출 | 정확성, 환각 억제 |
| T2 | L2: 수치 비교/집계 | 계산 능력 + 컨텍스트 누적 | 계산 정확성, 추론 논리 |
| T3 | L3: 문맥 의존 | 지시대명사 해소 | **컨텍스트 파악** |
| T4 | L4: 복합 계획/권고 | 가이드라인 기반 추론 | **근거 인용, 개인화** |
| T5 | Correction | 정정/모순 투입 | **메모리 업데이트** |
| T6 | Consistency | 전체 회상 + 일관성 | **장기 일관성** |

### 코호트 분류

- **Full Cohort**: 진단, 약물, 검사(2회 이상) 모두 있음
- **No-Meds Cohort**: 복용약 없음 (비약물 치료 환자)
- **No-Trend Cohort**: 검사 1회만 (신규 진단 환자)

## 평가 루브릭

### 서브스코어 (각 0~2점)

1. **Accuracy (정확성)**: GT 값/날짜/수치 오류
2. **Context (문맥 파악)**: 지시대명사 해소 정확도
3. **Reasoning (추론 논리)**: 계산/비교 논리 타당성
4. **Hallucination (환각 억제)**: GT/근거 없는 주장 수
5. **Evidence (근거 인용)**: 근거 문서 인용 정확도 (Agentic만)

### 가중치

**Basic RAG**:
- Accuracy: 35%, Context: 25%, Reasoning: 25%, Hallucination: 15%

**Agentic RAG**:
- Accuracy: 30%, Context: 20%, Reasoning: 20%, Hallucination: 10%, Evidence: 20%

## 실제 RAG 시스템 연결

현재 코드는 **더미 구현**입니다. 실제 실험을 위해서는 다음을 수정하세요:

### `run_multiturn_experiment.py`

```python
def _call_basic_rag(self, turn) -> str:
    from retrieval.hybrid_retriever import HybridRetriever
    from core.llm_client import LLMClient
    
    retriever = HybridRetriever(...)
    llm = LLMClient(...)
    
    docs = retriever.retrieve(turn.question)
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""다음 문서를 참고하여 질문에 답하세요.

문서:
{context}

질문: {turn.question}

답변:"""
    
    answer = llm.generate(prompt, temperature=0.0)
    return answer
```

```python
def _call_agentic_rag(self, turn) -> tuple[str, List[str]]:
    from agent.graph import create_graph
    from agent.state import AgentState
    
    graph = create_graph()
    
    initial_state = AgentState(
        query=turn.question,
        conversation_history=[...],  # 이전 턴들
        patient_context=turn.canonical_state_snapshot
    )
    
    result = graph.invoke(initial_state)
    
    answer = result["final_answer"]
    context = [doc.page_content for doc in result.get("retrieved_docs", [])]
    
    return answer, context
```

## 결과 분석

### 자동 분석

실험 종료 시 자동으로 다음 통계가 출력됩니다:

- 전체 평균 점수 (총점, 가중 점수)
- 턴 타입별 평균 점수
- 서브스코어별 평균
- Basic vs Agentic 비교

### 추가 분석 (Python)

```python
import json

# 결과 로드
with open("results/multiturn/basic_rag_results.json") as f:
    basic_results = json.load(f)

with open("results/multiturn/agentic_rag_results.json") as f:
    agentic_results = json.load(f)

# T3 (컨텍스트) 점수만 추출
basic_t3_scores = [
    t["evaluation"]["subscores"][1]["score"]  # Context
    for d in basic_results
    for t in d["turns"]
    if t["turn_type"] == "T3"
]

agentic_t3_scores = [
    t["evaluation"]["subscores"][1]["score"]
    for d in agentic_results
    for t in d["turns"]
    if t["turn_type"] == "T3"
]

# t-test
from scipy import stats
t_stat, p_value = stats.ttest_ind(agentic_t3_scores, basic_t3_scores)
print(f"T3 Context Score: t={t_stat:.3f}, p={p_value:.4f}")
```

## 논문 작성 가이드

### 방법론 섹션

```
본 연구는 멀티턴 대화에서 Basic RAG와 Agentic RAG의 성능을 비교하기 위해
프로토콜 기반 평가 프레임워크를 설계하였다.

**환자 시나리오**: 13개의 구조화된 환자 시나리오를 3개 코호트(Full, No-Meds, 
No-Trend)로 분류하여 생성하였다. 모든 시나리오는 동일한 JSON 스키마를 따르며,
결측값은 "없음"으로 명시하여 템플릿 오류를 방지하였다.

**대화 프로토콜**: 6턴 프로토콜(P6)을 사용하여 각 턴은 고정된 목적(사실 조회,
수치 비교, 문맥 의존, 복합 추론, 정정, 일관성 검증)을 가진다. 질문은 슬롯 기반
템플릿 은행에서 prerequisites 충족 여부에 따라 자동 선택되며, 환자 응답은
temperature=0, seed=42로 결정론적으로 생성하여 재현성을 확보하였다.

**평가 루브릭**: 5개 서브스코어(정확성, 문맥, 추론, 환각, 근거)를 0~2점 척도로
평가하고 가중합으로 총점을 산출하였다. Agentic RAG는 근거 인용 항목이 추가되어
검색 품질을 별도 평가하였다.
```

### 결과 섹션

```
Agentic RAG는 Basic RAG 대비 평균 가중 점수에서 X.XX점 높은 성능을 보였다
(p < 0.05). 특히 T3(문맥 의존) 턴에서 컨텍스트 파악 점수가 XX% 향상되었으며
(p < 0.01), T4(복합 추론) 턴에서 근거 인용 점수가 XX% 높았다 (p < 0.01).
정정 턴(T5) 후 일관성 유지(T6)에서도 Agentic RAG가 우수한 성능을 나타냈다.
```

## 확장 가능성

### 1. 새로운 턴 타입 추가

`question_bank.py`에서 `TurnType` enum과 템플릿 추가:

```python
class TurnType(Enum):
    # 기존 턴들...
    T7_SAFETY_CHECK = "T7"  # 새로운 턴

# QuestionBank._initialize_templates()에 추가
self.add_template(QuestionTemplate(
    template_id="T7_safety_check",
    turn_type=TurnType.T7_SAFETY_CHECK,
    goal="위험 신호 확인",
    # ...
))
```

### 2. 새로운 코호트 추가

`patient_scenarios.py`에서 생성 함수 추가:

```python
def generate_pregnancy_cohort_scenarios(self, count: int = 3):
    # 임신 환자 코호트
    pass
```

### 3. LLM-as-a-Judge 활성화

`evaluation_rubric.py`의 `LLMJudge` 클래스에서 실제 LLM 호출:

```python
from core.llm_client import LLMClient

class LLMJudge:
    def __init__(self):
        self.llm = LLMClient(model="gpt-4", temperature=0.0)
    
    def evaluate_turn(self, ...):
        response = self.llm.generate(prompt)
        judge_result = json.loads(response)
        # ...
```

## 문제 해결

### 템플릿 변수 누락 오류

```
Warning: Missing template variable 'test_name' in T2_lab_comparison_full
```

**해결**: `multiturn_simulator.py`의 `_format_question()` 함수에서 해당 변수 추가

### 코호트 불일치

```
ValueError: No suitable template for T2 in cohort No-Trend
```

**해결**: `question_bank.py`에서 해당 코호트용 fallback 템플릿 확인

### 평가 점수 이상

모든 점수가 동일하게 나오는 경우:

**원인**: 더미 구현 사용 중  
**해결**: 실제 RAG 시스템 연결 필요

## 참고 문헌

- 전략 문서: `251216_multiturn_upgrade_v0.md`
- ChatGPT/Gemini 대화 로그: (사용자 제공)
- 기존 실험: `experiments/run_ablation_comparison.py`

## 라이선스

MIT License

## 문의

프로젝트 관련 문의: (저자 이메일)

