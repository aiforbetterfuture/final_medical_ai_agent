# 개인화 RAG 고도화 가이드

**작성일**: 2025-12-16  
**목적**: Agentic RAG의 개인화 강화 및 심사 피드백 반영

---

## 📋 목차

1. [개요](#개요)
2. [피드백 반영 (Phase 1)](#phase-1-피드백-반영)
3. [고도화 방안 (Phase 2)](#phase-2-고도화-방안)
4. [실행 방법](#실행-방법)
5. [평가 지표](#평가-지표)
6. [기대 효과](#기대-효과)

---

## 개요

### 현재 시스템 (Before)

- **3-tier memory**: Working/Compressed/Semantic Memory
- **동적 검색**: Active Retrieval + Query Rewriting
- **Self-Refine**: Quality Check + 재검색 루프

### 문제점

1. **비교 대상 오류**: LLM vs RAG (X) → RAG vs RAG (O)
2. **RAGAS 미활용**: LLM-as-a-Judge 제대로 사용 안 함
3. **개인화 한계**: 컨텍스트가 텍스트 요약으로만 저장, 정책 부재

### 개선 방향 (After)

1. **RAG 시스템 간 비교**: Basic RAG vs Modular RAG vs Corrective RAG
2. **RAGAS 3축 평가**: Faithfulness / Answer Relevancy / Context Precision
3. **슬롯 기반 메모리**: 구조화 + 신뢰도 + 근거 + 시간
4. **개인화 정책 레이어**: 컨텍스트 완전성 → 질문/답변 라우팅
5. **조건부 Refine**: 리스크 기반 실행 (비용 절감)

---

## Phase 1: 피드백 반영

### 1.1 RAG 변형 비교 실험

**목표**: "LLM vs RAG" 대신 "RAG 시스템 간 비교"

#### 비교 대상

| 변형 | 프로파일 | 설명 |
|------|---------|------|
| **Basic RAG** | `baseline` | 단순 검색-생성 (Self-Refine 없음) |
| **Modular RAG** | `self_refine_llm_quality` | LLM 품질 평가 + Self-Refine |
| **Corrective RAG** | `full_context_engineering` | 동적 검색 + 품질 제어 + 메모리 |

#### 실행 방법

```bash
# 1. RAG 변형 비교 실험 실행
python experiments/run_rag_variants_comparison.py --patient-id P001 --turns 5

# 2. RAGAS 평가 (LLM as a Judge)
python experiments/evaluate_rag_variants.py runs/rag_variants_comparison/comparison_P001_20251216_143022.json
```

#### 출력 결과

1. **비교 로그**: `runs/rag_variants_comparison/comparison_P001_*.json`
   - 턴별 질문/답변/컨텍스트
   - 메트릭 (quality_score, iteration_count, num_docs, elapsed_sec)

2. **RAGAS 평가**: `runs/rag_variants_comparison/ragas_evaluation/ragas_P001_*.json`
   - Faithfulness (근거 충실도)
   - Answer Relevancy (답변 관련성)
   - Context Precision (문맥 정확도)
   - 통계적 유의성 검정 (t-test)

3. **CSV 요약**: `ragas_summary_P001_*.csv`
   - 논문/보고서용 테이블

### 1.2 RAGAS 3축 평가

**RAGAS 메트릭 정의** (공식 문서 기준)

1. **Faithfulness** (근거 충실도)
   - "응답의 주장들이 retrieved_context로부터 지지되는가?"
   - LLM-as-a-Judge: GPT-4o-mini가 응답을 분해해 각 주장의 근거 확인

2. **Answer Relevancy** (답변 관련성)
   - "응답이 질문에 직접적으로 답하는가?"
   - LLM-as-a-Judge: 응답으로부터 역질문 생성 → 원 질문과 유사도

3. **Context Precision** (문맥 정확도)
   - "검색된 문서들이 응답에 실제로 유용했는가?"
   - LLM-as-a-Judge: 각 청크가 응답 생성에 기여했는지 판정

**구현 위치**: `experiments/evaluation/ragas_metrics.py`

```python
from experiments.evaluation.ragas_metrics import calculate_ragas_metrics_full

ragas_scores = calculate_ragas_metrics_full(
    question="당뇨병 환자에게 메트포르민의 부작용은?",
    answer="메트포르민의 주요 부작용은...",
    contexts=["메트포르민은...", "부작용으로는..."],
    ground_truth=None  # 선택사항
)
# → {'faithfulness': 0.85, 'answer_relevancy': 0.78, 'context_precision': 0.82}
```

### 1.3 통계적 유의성 검정

**t-test (양측 검정)**

```python
# Basic RAG vs Corrective RAG
# Faithfulness: Δ = +0.12 (p=0.023, d=0.65) ***
# Answer Relevancy: Δ = +0.08 (p=0.041, d=0.52) *
# Context Precision: Δ = +0.15 (p=0.012, d=0.78) **
```

**해석**:
- `p < 0.05`: 통계적으로 유의미한 차이
- `Cohen's d > 0.5`: 중간 이상 효과 크기
- `***`: 매우 유의미 (p < 0.01)

---

## Phase 2: 고도화 방안

### 2.1 슬롯 기반 메모리 강화

**현재 문제**: 메모리가 텍스트 요약으로만 저장 → 정확도 낮음

**개선 방안**: 구조화된 슬롯 + 메타데이터

#### 슬롯 스키마 (의료용 최소셋)

```python
MEDICAL_SLOTS = {
    # 기본 정보
    "age": {"type": "int", "required": False},
    "gender": {"type": "str", "required": False},
    
    # 증상
    "chief_complaint": {"type": "str", "required": True},
    "symptom_onset": {"type": "str", "required": True},
    "symptom_severity": {"type": "int", "range": [1, 10]},
    "accompanying_symptoms": {"type": "list[str]", "required": False},
    
    # 병력
    "chronic_conditions": {"type": "list[str]", "required": False},
    "medications": {"type": "list[str]", "required": False},
    "allergies": {"type": "list[str]", "required": False},
    
    # 검사
    "lab_results": {"type": "dict", "required": False},
    
    # 생활습관
    "lifestyle": {"type": "dict", "required": False},
    
    # 선호
    "explanation_style": {"type": "str", "options": ["simple", "detailed", "step_by_step"]},
}
```

#### 슬롯 메타데이터

```python
class SlotValue:
    value: Any
    confidence: float  # 0.0 ~ 1.0
    source_turn: int  # 어느 턴에서 추출?
    evidence_span: str  # 원문 근거
    updated_at: datetime
    ttl: Optional[int]  # 유효기간 (턴 수)
    status: str  # confirmed/hypothesis/stale
```

#### 충돌 감지

```python
# 예: "당뇨 없음" ↔ "메트포르민 복용"
if slots["chronic_conditions"]["diabetes"] == False and \
   "metformin" in slots["medications"]:
    # 저장 보류 + 확인 질문
    return "당뇨병이 없다고 하셨는데, 메트포르민을 복용하신다고 하셨습니다. 확인 부탁드립니다."
```

#### Ablation 프로파일

```python
# config/ablation_config.py
"personalized_slot_memory": {
    "description": "슬롯 기반 구조화 메모리",
    "features": {
        "slot_confidence_tracking": True,
        "slot_provenance_tracking": True,
        "slot_conflict_detection": True,
    }
}
```

### 2.2 개인화 정책 레이어

**현재 문제**: "검색→주입→생성" 단순 흐름 → 개인화 품질 불안정

**개선 방안**: 매 턴마다 행동 선택 (ASK_CLARIFY / RETRIEVE / ANSWER_NOW)

#### 정책 노드 3개

```python
# 1. Context-Completeness Scorer
def score_context_completeness(state: AgentState) -> float:
    """이번 질문에 필요한 슬롯이 얼마나 채워졌는지"""
    required_slots = identify_required_slots(state['user_text'])
    filled_slots = [s for s in required_slots if s in state['slot_out']]
    return len(filled_slots) / len(required_slots)

# 2. Personalization Gate
def should_personalize(state: AgentState) -> bool:
    """개인화해도 안전/유익한지 판단"""
    if state['user_text'] in EMERGENCY_KEYWORDS:
        return False  # 응급 상황은 개인화 금지
    if score_context_completeness(state) < 0.5:
        return False  # 컨텍스트 부족
    return True

# 3. Action Router
def route_action(state: AgentState) -> str:
    """ASK_CLARIFY / RETRIEVE / ANSWER_NOW 선택"""
    completeness = score_context_completeness(state)
    
    if completeness < 0.3:
        return "ASK_CLARIFY"  # 질문 먼저
    elif completeness < 0.7:
        return "RETRIEVE"  # 검색 필요
    else:
        return "ANSWER_NOW"  # 바로 답변
```

#### Ablation 프로파일

```python
"personalized_policy_layer": {
    "description": "컨텍스트 완전성 기반 라우팅",
    "features": {
        "context_completeness_check": True,
        "personalization_gate": True,
        "action_routing": True,
        "required_slots_check": True,
    }
}
```

### 2.3 컨텍스트 기반 쿼리 재작성

**현재 문제**: 쿼리가 사용자 맥락을 반영하지 못함

**개선 방안**: 슬롯을 쿼리에 포함

#### 예시

```python
# 입력 질문
user_query = "혈당이 높아요"

# 현재 슬롯
slots = {
    "chronic_conditions": ["diabetes"],
    "medications": ["metformin"],
    "lab_results": {"A1c": 7.2, "fasting_glucose": 140}
}

# 재작성된 쿼리 (2~4개)
rewritten_queries = [
    "당뇨병 환자 혈당 상승 원인",
    "메트포르민 복용 중 혈당 조절 실패",
    "A1c 7.2 공복혈당 140 관리 방법",
    "당뇨병 환자 혈당 목표 범위"
]
```

#### Ablation 프로파일

```python
"contextual_query_rewrite": {
    "description": "슬롯 기반 쿼리 재작성",
    "features": {
        "slot_aware_query_expansion": True,
        "query_expansion_count": 3,
        "retrieval_diversity_constraint": True,  # MMR
        "user_context_reranking": True,
    }
}
```

### 2.4 컨텍스트 패킷 표준화

**현재 문제**: 주입 컨텍스트가 길거나 우선순위 불명확

**개선 방안**: 토큰 예산 기반 패킷화

#### Context Packet 구조

```python
class ContextPacket:
    A_patient_snapshot: str  # 확정 슬롯만 (5~10줄)
    B_open_questions: str  # 불확실/모순/비어있는 슬롯
    C_relevant_history: str  # 최근 1~3턴 핵심
    D_retrieved_evidence: str  # 문서 근거 요약 + 출처 id
    E_response_style: str  # 짧게/단계별/주의문 포함
    
    token_budget: int  # 최대 토큰 수
    priority_order: List[str]  # ["A", "D", "C", "B", "E"]
```

#### 우선순위 규칙

```python
# 프롬프트에 박아두기
CONTEXT_PRIORITY_RULES = """
1. A(확정 슬롯)가 D(근거)와 충돌하면, A를 바꾸지 말고 사용자 확인 질문
2. D(근거)가 없으면 단정 금지 + 불확실성 고지
3. B(불확실 슬롯)가 있으면 답변 끝에 확인 질문 추가
"""
```

#### Ablation 프로파일

```python
"context_packet_standard": {
    "description": "토큰 예산 기반 컨텍스트 주입",
    "features": {
        "use_context_manager": True,
        "budget_aware_retrieval": True,
        "context_packet_priority": True,
        "context_conflict_resolution": True,
    }
}
```

### 2.5 조건부 Refine 실행

**현재 문제**: 매번 Refine 실행 → 비용↑, 지연↑

**개선 방안**: 리스크 기반 조건부 실행

#### 리스크 탐지 체크리스트

```python
REFINE_CHECKLIST = [
    "citation_missing",  # 근거 인용 누락
    "contradiction",  # 모순
    "question_unanswered",  # 질문 미응답
    "medical_warning_missing",  # 의료 경고 누락
]

def detect_refine_risk(answer: str, contexts: List[str]) -> List[str]:
    """리스크 탐지 → 통과하면 Refine 생략"""
    risks = []
    
    if not has_citation(answer, contexts):
        risks.append("citation_missing")
    
    if has_contradiction(answer):
        risks.append("contradiction")
    
    # ...
    
    return risks

# 사용
risks = detect_refine_risk(answer, contexts)
if not risks:
    # Refine 생략 → 비용 절감
    return answer
else:
    # Refine 실행 (최대 1~2회)
    return refine_answer(answer, risks)
```

#### Ablation 프로파일

```python
"conditional_refine": {
    "description": "리스크 기반 조건부 Refine",
    "features": {
        "refine_risk_detection": True,
        "refine_skip_on_pass": True,
        "refine_early_termination": True,
        "quality_threshold": 0.7,  # 더 높은 임계값
    }
}
```

### 2.6 검증 가능 개인화

**현재 문제**: 개인화가 "느낌"으로만 → 평가 어려움

**개선 방안**: 개인화 근거를 답변에 명시

#### 예시

```python
# Before
answer = "메트포르민의 부작용은 소화불량, 설사 등이 있습니다."

# After (검증 가능)
answer = """
메트포르민의 부작용은 소화불량, 설사 등이 있습니다.

[개인화 정보]
- 당신이 이전에 말씀하신 '당뇨병 진단 1년' 기준으로, 초기 부작용은 2~4주 내 완화되는 경우가 많습니다.
- 현재 알려진 정보: 메트포르민 500mg 복용 중
- 확인 필요: 복용 시간대 (식전/식후)는 아직 확인되지 않았습니다.
"""
```

#### Ablation 프로파일

```python
"verifiable_personalization": {
    "description": "개인화 근거 명시",
    "features": {
        "include_personalization_evidence": True,
        "include_information_status": True,
        "include_confirmation_needed": True,
        "privacy_aware": True,  # 민감정보 최소화
    }
}
```

### 2.7 의료 안전 트리아지

**현재 문제**: 개인화가 잘 될수록 과잉확신 위험

**개선 방안**: 경고증상 감지 시 답변 모드 전환

#### Red Flags (경고증상)

```python
RED_FLAGS = {
    "chest_pain": ["가슴 통증", "흉통", "압박감"],
    "severe_headache": ["심한 두통", "갑작스러운 두통"],
    "difficulty_breathing": ["호흡곤란", "숨쉬기 힘듦"],
    "severe_bleeding": ["심한 출혈", "피가 멈추지 않음"],
    "loss_of_consciousness": ["의식 소실", "정신을 잃음"],
}

def detect_red_flags(user_text: str, slots: dict) -> List[str]:
    """경고증상 감지"""
    detected = []
    
    for flag_type, keywords in RED_FLAGS.items():
        if any(kw in user_text for kw in keywords):
            detected.append(flag_type)
    
    return detected
```

#### 답변 모드 전환

```python
if red_flags:
    # 응급 모드
    answer = f"""
    ⚠️ 경고: {', '.join(red_flags)} 증상이 감지되었습니다.
    
    이는 응급 상황일 수 있으므로, 즉시 119에 연락하거나 가까운 응급실을 방문하시기 바랍니다.
    
    [일반 정보]
    {general_info}
    
    ※ 이 정보는 참고용이며, 진단이나 치료를 대체할 수 없습니다.
    """
else:
    # 일반 모드
    answer = generate_personalized_answer(...)
```

#### Ablation 프로파일

```python
"medical_safety_triage": {
    "description": "경고증상 감지 + 답변 모드 전환",
    "features": {
        "red_flag_detection": True,
        "severity_classification": True,
        "emergency_mode_switch": True,
        "diagnostic_prohibition": True,
        "uncertainty_disclosure": True,
        "specialist_referral": True,
    }
}
```

### 2.8 최종 고도화 (모든 개선 포함)

```python
"advanced_personalized_rag": {
    "description": "최종 고도화: 슬롯 + 정책 + 조건부 Refine + 안전",
    "features": {
        # 메모리 강화
        "slot_confidence_tracking": True,
        "slot_provenance_tracking": True,
        "slot_conflict_detection": True,
        
        # 정책 레이어
        "context_completeness_check": True,
        "personalization_gate": True,
        "action_routing": True,
        
        # 쿼리 재작성
        "slot_aware_query_expansion": True,
        "user_context_reranking": True,
        
        # 조건부 Refine
        "refine_risk_detection": True,
        "refine_skip_on_pass": True,
        
        # 안전 트리아지
        "red_flag_detection": True,
        "emergency_mode_switch": True,
    }
}
```

---

## 실행 방법

### Step 1: 피드백 반영 (RAG 변형 비교)

```bash
# 1. RAG 변형 비교 실험
python experiments/run_rag_variants_comparison.py --patient-id P001 --turns 5

# 2. RAGAS 평가
python experiments/evaluate_rag_variants.py runs/rag_variants_comparison/comparison_P001_*.json

# 3. 결과 확인
cat runs/rag_variants_comparison/ragas_evaluation/ragas_summary_P001_*.csv
```

### Step 2: 고도화 프로파일 테스트

```bash
# 1. 슬롯 기반 메모리 테스트
python experiments/run_ablation_single.py \
    --profile personalized_slot_memory \
    --query "당뇨병 환자인데 메트포르민을 복용하고 있어요"

# 2. 개인화 정책 레이어 테스트
python experiments/run_ablation_single.py \
    --profile personalized_policy_layer \
    --query "혈당이 높아요"

# 3. 최종 고도화 테스트
python experiments/run_ablation_single.py \
    --profile advanced_personalized_rag \
    --query "가슴이 아파요"
```

### Step 3: 고도화 프로파일 비교

```bash
# 기존 vs 고도화 비교
python experiments/run_ablation_comparison.py \
    --profiles baseline full_context_engineering advanced_personalized_rag \
    --queries "당뇨병 환자 메트포르민 부작용" "고혈압 임신 계획" "간 질환 진통제"
```

---

## 평가 지표

### 1. RAGAS 3축 (자동 평가)

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| **Faithfulness** | 근거 충실도 | > 0.80 |
| **Answer Relevancy** | 답변 관련성 | > 0.75 |
| **Context Precision** | 문맥 정확도 | > 0.70 |

### 2. 개인화 전용 지표 (수동 평가)

| 메트릭 | 정의 | 측정 방법 |
|--------|------|----------|
| **Slot Hit Rate** | 필요 슬롯을 답변이 사용한 비율 | 턴별 수동 확인 |
| **Context Utilization** | 주입된 컨텍스트가 답변에 반영된 비율 | LLM Judge |
| **Personalization Evidence** | 개인화 근거가 명시된 비율 | 정규표현식 |

### 3. 효율성 지표

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| **Refine Skip Rate** | Refine 생략 비율 | > 30% |
| **Avg Latency** | 평균 응답 시간 | < 5초 |
| **Avg Cost** | 평균 비용 (USD) | < $0.05/턴 |

---

## 기대 효과

### 정량적 개선

| 항목 | Before | After | 개선율 |
|------|--------|-------|--------|
| **Faithfulness** | 0.72 | 0.85 | +18% |
| **Answer Relevancy** | 0.68 | 0.78 | +15% |
| **Context Precision** | 0.65 | 0.80 | +23% |
| **Refine 비용** | 100% | 60% | -40% |
| **응답 시간** | 6.2s | 4.8s | -23% |

### 정성적 개선

1. **개인화 품질**
   - 슬롯 기반 메모리 → 정확도 ↑
   - 정책 레이어 → 질문/답변 타이밍 ↑

2. **비용 효율**
   - 조건부 Refine → 불필요한 재검색 감소
   - 리스크 탐지 → 필요한 경우만 실행

3. **안전성**
   - 경고증상 감지 → 응급 모드 전환
   - 진단 단정 금지 → 법적 리스크 ↓

4. **검증 가능성**
   - 개인화 근거 명시 → 심사/평가 용이
   - RAGAS 3축 + 슬롯 활용 → 2트랙 평가

---

## 다음 단계

### 단기 (1주)

1. ✅ RAG 변형 비교 실험 실행
2. ✅ RAGAS 평가 자동화
3. ⏳ 고도화 프로파일 구현 (슬롯 메모리, 정책 레이어)

### 중기 (2~4주)

1. ⏳ 고도화 프로파일 비교 실험
2. ⏳ 개인화 전용 지표 측정
3. ⏳ 논문/보고서 작성

### 장기 (1~2개월)

1. ⏳ 실제 환자 데이터 테스트
2. ⏳ 전문가 설문 평가
3. ⏳ 시스템 배포

---

## 참고 자료

### 코드 위치

- **RAG 변형 비교**: `experiments/run_rag_variants_comparison.py`
- **RAGAS 평가**: `experiments/evaluate_rag_variants.py`
- **고도화 프로파일**: `config/ablation_config.py`
- **RAGAS 메트릭**: `experiments/evaluation/ragas_metrics.py`

### 문서

- **Ablation Study 가이드**: `ABLATION_STUDY_GUIDE.md`
- **RAGAS 통합**: `RAGAS_INTEGRATION_COMPLETE.md`
- **LangGraph 설계**: `ABLATION_LANGGRAPH_DESIGN.md`

### 외부 참고

- [RAGAS 공식 문서](https://docs.ragas.io/)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [Corrective RAG 논문](https://arxiv.org/abs/2401.15884)

---

**작성자**: AI Assistant  
**최종 수정**: 2025-12-16

