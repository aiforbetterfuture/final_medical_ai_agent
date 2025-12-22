# 멀티턴 대화 테스트 구현 완료 보고서

**작성일**: 2024-12-16  
**상태**: ✅ 구현 완료 및 테스트 통과

---

## 1. 개요

ChatGPT 및 Gemini와의 대화를 기반으로, **재현성과 객관성을 확보한 멀티턴 대화 평가 프레임워크**를 설계하고 구현하였습니다.

### 핵심 목표

1. **재현성 확보**: 고정된 프로토콜 + 결정론적 생성 (temperature=0, seed 고정)
2. **객관성 확보**: 서브스코어 기반 평가 루브릭 + 코호트 분리
3. **차별화된 평가**: Agentic RAG의 강점(컨텍스트/메모리/근거) 정량화

---

## 2. 구현 내용

### 2.1 파일 구조

```
experiments/multiturn/
├── __init__.py                      # 모듈 초기화
├── patient_scenarios.py             # 환자 시나리오 생성기 (13개 시나리오)
├── question_bank.py                 # 질문은행 (16개 템플릿 + 게이팅)
├── multiturn_simulator.py           # 대화 시뮬레이터 (P6 프로토콜)
├── evaluation_rubric.py             # 평가 루브릭 (5개 서브스코어)
├── run_multiturn_experiment.py      # 메인 실험 러너
├── quick_test.py                    # 빠른 테스트 (✅ 통과)
└── README.md                        # 사용 가이드

data/multiturn/                      # 생성된 데이터 저장
results/multiturn/                   # 실험 결과 저장

251216_multiturn_upgrade_v0.md       # 전략 문서 (상세 설계)
```

### 2.2 주요 컴포넌트

#### A. 환자 시나리오 생성기 (`patient_scenarios.py`)

- **13개 구조화된 환자 시나리오** (3개 코호트)
  - Full Cohort (7개): 진단, 약물, 검사(2회 이상) 모두 있음
  - No-Meds Cohort (3개): 복용약 없음 (비약물 치료)
  - No-Trend Cohort (3개): 검사 1회만 (신규 진단)

- **슬롯 기반 JSON 스키마**
  ```python
  {
    "patient_id": "P-F001",
    "diagnosis": ["Type 2 Diabetes", "Hypertension"],
    "medications": [{"name": "Metformin", "dosage": "500mg", "frequency": "BID"}],
    "allergy": "없음",  # 결측값도 명시
    "lab_results": [{"date": "2024-01-15", "test": "HbA1c", "value": 7.8, "unit": "%"}],
    "cohort": "Full",
    "scenario_level": "L2"
  }
  ```

#### B. 질문은행 (`question_bank.py`)

- **16개 템플릿** (턴 타입별 2~3개)
  - T1 (사실 조회): 2개
  - T2 (수치 비교): 3개
  - T3 (문맥 의존): 3개
  - T4 (복합 추론): 3개
  - T5 (정정): 3개
  - T6 (일관성): 2개

- **게이팅 로직**
  ```python
  template = bank.select_template_with_gating(
      turn_type=TurnType.T2_COMPARISON,
      patient_data=current_state,
      cohort="Full"
  )
  # prerequisites 충족 여부 확인 → fallback 처리
  ```

- **Null-Safe 설계**: 결측 슬롯 시 자동으로 대체 템플릿 선택

#### C. 멀티턴 시뮬레이터 (`multiturn_simulator.py`)

- **P6 프로토콜** (6턴 고정)
  1. T1: 단순 사실 조회
  2. T2: 수치 비교/집계
  3. T3: 문맥 의존 (지시대명사 해소)
  4. T4: 복합 계획/권고
  5. T5: 정정/모순 투입
  6. T6: 일관성/회상

- **환자 응답 생성기** (4블록 구조)
  1. Direct Answer (필수)
  2. Disclosed Slots (규칙 기반)
  3. Unknown Handling (모름/없음 명시)
  4. Follow-up Question (선택적)

- **결정론적 생성**: `temperature=0`, `seed=42`

#### D. 평가 루브릭 (`evaluation_rubric.py`)

- **5개 서브스코어** (각 0~2점)
  1. **Accuracy** (정확성): GT 값/날짜/수치 오류
  2. **Context** (문맥 파악): 지시대명사 해소 정확도
  3. **Reasoning** (추론 논리): 계산/비교 논리 타당성
  4. **Hallucination** (환각 억제): GT/근거 없는 주장 수
  5. **Evidence** (근거 인용): 근거 문서 인용 정확도 (Agentic만)

- **가중치**
  - Basic RAG: Accuracy 35%, Context 25%, Reasoning 25%, Hallucination 15%
  - Agentic RAG: Accuracy 30%, Context 20%, Reasoning 20%, Hallucination 10%, Evidence 20%

- **LLM-as-a-Judge 지원** (선택적)

#### E. 실험 러너 (`run_multiturn_experiment.py`)

- **전체 파이프라인 자동화**
  1. 환자 시나리오 생성
  2. 대화 생성 (P6 프로토콜)
  3. Basic RAG 실행 및 평가
  4. Agentic RAG 실행 및 평가
  5. 결과 분석 및 통계

- **명령줄 인터페이스**
  ```bash
  python run_multiturn_experiment.py \
    --output-dir results/multiturn \
    --protocol P6 \
    --seed 42 \
    --num-scenarios 13 \
    --mode cooperative
  ```

---

## 3. 테스트 결과

### 3.1 빠른 테스트 (`quick_test.py`)

```bash
python experiments/multiturn/quick_test.py
```

**결과**: ✅ **모든 테스트 통과**

```
============================================================
[SUCCESS] 모든 테스트 통과!
============================================================

1. 환자 시나리오 생성: 13개 (Full 7, No-Meds 3, No-Trend 3)
2. 질문은행 로드: 16개 템플릿
3. 대화 시뮬레이션: 6턴 생성 완료
4. 평가 루브릭: 서브스코어 계산 정상
5. 코호트별 게이팅: Full/No-Trend 자동 분기 확인
```

### 3.2 검증된 기능

- ✅ 환자 시나리오 JSON 직렬화/역직렬화
- ✅ 코호트별 템플릿 자동 선택 (게이팅)
- ✅ prerequisites 충족 여부 확인 + fallback
- ✅ 환자 응답 4블록 구조 생성
- ✅ 정정 턴(T5) 후 상태 업데이트
- ✅ Gold Answer 자동 생성 (T1, T2, T3, T6)
- ✅ 서브스코어 기반 평가
- ✅ 턴 타입별/코호트별 통계 분석

---

## 4. 핵심 설계 결정 (ChatGPT/Gemini 대화 반영)

### 4.1 "빈 슬롯" 문제 해결

**문제**: 이전 멀티턴 테스트에서 `{pmh}`가 없는 환자에게 "제가 {pmh} 있고..."라는 질문을 하면 빈칸이 생겨 응답 유효성 저하

**해결책 (3가지 적용)**:
1. **슬롯 값 강제**: 모든 환자가 동일 스키마, 결측값은 `"없음"` 명시
2. **코호트 분리**: Full/No-Meds/No-Trend로 나눠 각 코호트별 질문 프로토콜 고정
3. **게이팅 로직**: prerequisites 불충족 시 자동으로 fallback 템플릿 선택

### 4.2 "외부 지식 의존" 최소화

**문제**: Gemini 원안의 T3 "약물 부작용/기전" 질문은 지식베이스 커버리지 차이가 되어 순수한 컨텍스트 능력 비교 불가

**해결책**:
- T3를 **"지시대명사 해소 + 이전 턴 재사용"** 문제로 변경
- 예: "방금 비교한 그 수치(HbA1c)의 이전값/최근값/변화량을 다시 요약해 주세요"
- 외부 지식 없이 **대화 메모리/컨텍스트 주입 품질**만 측정

### 4.3 "환자 응답 프로토콜" 고정

**문제**: 질문 프로토콜만 고정하면 환자 응답의 표현 다양성 때문에 편차 발생

**해결책**:
- **2-레이어 발화**: Canonical State (정답 JSON) + Surface Utterance (표면 문장)
- **4블록 구조 강제**: Direct Answer + Disclosed Slots + Unknown Handling + Follow-up
- `temperature=0`, `seed=42`로 결정론적 생성

### 4.4 "평가 루브릭 세분화"

**문제**: Gemini 원안의 단일 1~5점 척도는 Judge 변동이 큼

**해결책**:
- **5개 서브스코어** (각 0~2점) + 가중합
- 항목별로 채점 근거 명시 → 통계 분석 용이 (항목별 t-test 가능)
- Agentic RAG는 **Evidence (근거 인용)** 항목 추가

---

## 5. 실제 RAG 시스템 연결 가이드

현재 구현은 **더미 답변**을 사용합니다. 실제 실험을 위해 다음을 수정하세요:

### 5.1 Basic RAG 연결

`run_multiturn_experiment.py`의 `_call_basic_rag()` 함수:

```python
def _call_basic_rag(self, turn) -> str:
    from retrieval.hybrid_retriever import HybridRetriever
    from core.llm_client import LLMClient
    
    retriever = HybridRetriever(...)
    llm = LLMClient(model="gpt-4o-mini", temperature=0.0)
    
    docs = retriever.retrieve(turn.question, top_k=5)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""다음 문서를 참고하여 질문에 답하세요.

문서:
{context}

질문: {turn.question}

답변:"""
    
    answer = llm.generate(prompt)
    return answer
```

### 5.2 Agentic RAG 연결

`run_multiturn_experiment.py`의 `_call_agentic_rag()` 함수:

```python
def _call_agentic_rag(self, turn) -> tuple[str, List[str]]:
    from agent.graph import create_graph
    from agent.state import AgentState
    
    graph = create_graph()
    
    # 이전 턴들을 conversation_history로 전달
    history = [
        {"role": "user", "content": prev_turn.question}
        for prev_turn in self.current_dialogue_turns
    ]
    
    initial_state = AgentState(
        query=turn.question,
        conversation_history=history,
        patient_context=turn.canonical_state_snapshot
    )
    
    result = graph.invoke(initial_state)
    
    answer = result["final_answer"]
    context = [doc.page_content for doc in result.get("retrieved_docs", [])]
    
    return answer, context
```

---

## 6. 예상 실험 결과

### 6.1 가설

1. **H1**: Agentic RAG는 Basic RAG 대비 **T3 (Context)** 점수가 유의미하게 높을 것
   - 이유: 메모리/컨텍스트 주입 메커니즘

2. **H2**: Agentic RAG는 **T4 (Evidence)** 점수가 유의미하게 높을 것
   - 이유: 근거 인용 능력

3. **H3**: Agentic RAG는 **T5→T6 (Consistency)** 점수가 유의미하게 높을 것
   - 이유: 상태 업데이트 및 장기 일관성

### 6.2 통계 분석 예시

```python
import json
from scipy import stats

# 결과 로드
with open("results/multiturn/basic_rag_results.json") as f:
    basic_results = json.load(f)
with open("results/multiturn/agentic_rag_results.json") as f:
    agentic_results = json.load(f)

# T3 Context 점수 추출
basic_t3_context = [
    t["evaluation"]["subscores"][1]["score"]  # Context
    for d in basic_results
    for t in d["turns"]
    if t["turn_type"] == "T3"
]

agentic_t3_context = [
    t["evaluation"]["subscores"][1]["score"]
    for d in agentic_results
    for t in d["turns"]
    if t["turn_type"] == "T3"
]

# t-test
t_stat, p_value = stats.ttest_ind(agentic_t3_context, basic_t3_context)
print(f"T3 Context: Agentic {np.mean(agentic_t3_context):.2f} vs Basic {np.mean(basic_t3_context):.2f}")
print(f"  t={t_stat:.3f}, p={p_value:.4f}")
```

---

## 7. 논문 작성 가이드

### 7.1 방법론 섹션 (예시)

> 본 연구는 멀티턴 대화에서 Basic RAG와 Agentic RAG의 성능을 비교하기 위해 프로토콜 기반 평가 프레임워크를 설계하였다. 13개의 구조화된 환자 시나리오를 3개 코호트(Full, No-Meds, No-Trend)로 분류하여 생성하였으며, 모든 시나리오는 동일한 JSON 스키마를 따른다. 결측값은 "없음"으로 명시하여 템플릿 오류를 방지하였다.
>
> 대화는 6턴 프로토콜(P6)을 사용하여 각 턴은 고정된 목적(사실 조회, 수치 비교, 문맥 의존, 복합 추론, 정정, 일관성 검증)을 가진다. 질문은 슬롯 기반 템플릿 은행에서 prerequisites 충족 여부에 따라 자동 선택되며, 환자 응답은 temperature=0, seed=42로 결정론적으로 생성하여 재현성을 확보하였다.
>
> 평가는 5개 서브스코어(정확성, 문맥, 추론, 환각, 근거)를 0~2점 척도로 측정하고 가중합으로 총점을 산출하였다. Agentic RAG는 근거 인용 항목이 추가되어 검색 품질을 별도 평가하였다.

### 7.2 결과 섹션 (템플릿)

> Agentic RAG는 Basic RAG 대비 평균 가중 점수에서 X.XX점 높은 성능을 보였다 (p < 0.05). 특히 T3(문맥 의존) 턴에서 컨텍스트 파악 점수가 XX% 향상되었으며 (p < 0.01), T4(복합 추론) 턴에서 근거 인용 점수가 XX% 높았다 (p < 0.01). 정정 턴(T5) 후 일관성 유지(T6)에서도 Agentic RAG가 우수한 성능을 나타냈다 (p < 0.05).

---

## 8. 확장 가능성

### 8.1 새로운 턴 타입 추가

- T7: Safety Check (위험 신호 확인)
- T8: Explanation (진단/치료 근거 설명)

### 8.2 새로운 코호트 추가

- Pregnancy Cohort (임신 환자)
- Pediatric Cohort (소아 환자)
- Polypharmacy Cohort (다제 복용 환자)

### 8.3 다국어 지원

- 영어 템플릿 추가
- 번역 품질 평가 항목 추가

---

## 9. 제한사항 및 향후 과제

### 9.1 현재 제한사항

1. **더미 RAG 구현**: 실제 RAG 시스템 연결 필요
2. **Gold Answer 제한**: T4 (가이드라인 기반 판단)은 정답 고정 어려움
3. **평가 자동화**: 현재는 규칙 기반, LLM-as-a-Judge는 선택적

### 9.2 향후 과제

1. **실제 RAG 시스템 통합**
   - `retrieval.hybrid_retriever` 연결
   - `agent.graph` 연결

2. **대규모 실험**
   - 환자 시나리오 50개 이상
   - 3회 반복 실험 (seed 변경)

3. **평가 고도화**
   - LLM-as-a-Judge 활성화
   - 인간 평가자와의 일치도 검증

4. **논문 작성**
   - 방법론/결과/고찰 섹션 완성
   - 통계 분석 및 시각화

---

## 10. 결론

ChatGPT 및 Gemini와의 대화를 기반으로, **재현성과 객관성을 극대화한 멀티턴 대화 평가 프레임워크**를 성공적으로 구현하였습니다.

### 주요 성과

1. ✅ **프로토콜 기반 설계**: P6 (6턴) 고정 + 16개 템플릿
2. ✅ **코호트 분리 + 게이팅**: 빈 슬롯 문제 해결
3. ✅ **환자 응답 프로토콜**: 4블록 구조 + 결정론적 생성
4. ✅ **서브스코어 평가**: 5개 항목 + 가중합
5. ✅ **전체 파이프라인**: 생성 → 시뮬레이션 → 평가 → 분석
6. ✅ **테스트 통과**: `quick_test.py` 모든 검증 완료

### 다음 단계

1. **실제 RAG 시스템 연결** (`run_multiturn_experiment.py` 수정)
2. **전체 실험 실행** (13개 시나리오 × P6 × Basic/Agentic)
3. **결과 분석 및 논문 작성**

---

**문의**: 프로젝트 관련 질문은 README.md 참조  
**라이선스**: MIT License

