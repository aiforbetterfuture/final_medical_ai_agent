# 251225 3-Metrics 평가 시스템 통합 가이드

**작성일**: 2025년 12월 25일
**목적**: faithfulness, answer_relevance, perplexity 3개 핵심 메트릭을 중심으로 한 LLM-as-a-Judge 평가 시스템의 안정화 및 스키마 고정

---

## 📋 목차

1. [개요](#개요)
2. [핵심 3-메트릭 정의](#핵심-3-메트릭-정의)
3. [파일 구조 및 역할](#파일-구조-및-역할)
4. [주요 개선 사항](#주요-개선-사항)
5. [사용 방법](#사용-방법)
6. [자주 터지는 지점 및 해결책](#자주-터지는-지점-및-해결책)
7. [테스트 및 검증](#테스트-및-검증)
8. [문제 해결 가이드](#문제-해결-가이드)

---

## 개요

이 프로젝트는 의료 AI 에이전트의 답변 품질을 평가하기 위해 **3개의 핵심 메트릭**을 사용합니다:

1. **faithfulness** (신뢰성): 답변이 검색된 TS 근거로 뒷받침되는가?
2. **answer_relevance** (관련성): 질문에 실제로 답하는가?
3. **perplexity** (복잡도): 답변의 언어적 자연스러움 (낮을수록 좋음)

이 문서는 평가 시스템의 **스키마 안정성**을 확보하고, ChatGPT가 지적한 "자주 터지는 지점"들을 체계적으로 해결한 내용을 정리합니다.

---

## 핵심 3-메트릭 정의

### 1. Faithfulness (신뢰성)
- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **의미**: 답변이 검색된 TS(Two-Stage) 근거로 뒷받침되는가?
- **평가 기준**:
  - ✅ TS 근거에 명시된 정보만 사용
  - ❌ 환각(hallucination), 추측 → 크게 감점
  - ❌ TL 힌트를 사실처럼 단정 → 감점
- **임계값**: 일반적으로 0.75 이상이면 PASS

### 2. Answer Relevance (관련성)
- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **의미**: 사용자 질문에 실제로 답하는가?
- **평가 기준**:
  - ✅ 질문의 핵심 의도를 파악하고 직접 답변
  - ❌ 질문과 무관한 일반론만 나열
  - ❌ 질문의 일부만 답하고 나머지는 무시
- **임계값**: 0.75 이상

### 3. Perplexity (복잡도)
- **범위**: float (낮을수록 좋음, 일반적으로 10~100)
- **의미**: 답변의 언어적 자연스러움 / 예측 가능성
- **계산 방법**:
  - `transformers` + `torch` 있으면: 로컬에서 causal LM으로 계산 (기본: `distilgpt2`)
  - 없으면: `-1.0`으로 기록하고 이유 남김
- **사용 목적**: 문법/어색함 감지, 과도한 반복 방지

### 추가 메트릭 (선택)
- **context_use** (맥락 활용): 환자 정보(나이/성별/병력/복약 등)를 적절히 활용하는가? (0.0~1.0)

---

## 파일 구조 및 역할

### 1. `configs/eval_rubric.yaml`
**역할**: 평가 설정의 **단일 진실원(SSOT, Single Source of Truth)**

```yaml
llm_judge:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 800
  timeout_s: 60
  threshold: 0.75

  system_prompt: |
    당신은 의료 QA 품질 심사관입니다.
    근거(TS evidence)만을 신뢰하며, TL 힌트는 참고용으로만 봅니다.
    반드시 유효한 JSON만 출력하세요.
    - faithfulness: TS 근거로 뒷받침되는가?
    - answer_relevance: 질문에 실제로 답하는가?
    - context_use: 환자 맥락을 활용하는가?

  scoring_criteria:
    - key: faithfulness
      desc: "TS 근거와 일치/지지 여부"
    - key: answer_relevance
      desc: "질문-답변 적합성"
    - key: context_use
      desc: "환자/대화 맥락 활용"

perplexity:
  enabled: true
  model: distilgpt2
  env_var: HF_PERPLEXITY_MODEL
```

**핵심 포인트**:
- ✅ 모든 설정이 명시적으로 정의됨
- ✅ YAML 구조가 명확 (한 줄 압축 금지)
- ✅ 메트릭 키가 코드와 정확히 일치

### 2. `configs/question_templates.yaml`
**역할**: 평가셋 케이스 정의 및 한영 양언어 지원

```yaml
bilingual:
  enabled: true
  glossary:
    - canonical: HbA1c
      ko: [HbA1c, 당화혈색소, 헤모글로빈 A1c]
      en: [HbA1c, hemoglobin A1c]
    - canonical: diabetes mellitus
      ko: [당뇨, 당뇨병, 당뇨병성]
      en: [diabetes, diabetes mellitus]

cases:
  - case_id: DM_A1C_TARGET_basic
    domain_id: 3
    q_type: 2
    difficulty: easy
    turns:
      - turn_id: 1
        utterance: 당뇨 환자 HbA1c 목표 범위가 어떻게 되나요?
        utterance_en: What is the target HbA1c range for diabetes patients?
```

**핵심 포인트**:
- ✅ 양언어(한/영) glossary로 슬롯 매칭 강화
- ✅ Case 구조가 일관됨 (turn_id, utterance, expected_slot_updates 등)
- ✅ 평가 시 언어 차이로 인한 오류 방지

### 3. `tools/llm_as_judge.py`
**역할**: LLM 기반 평가 실행 + Perplexity 계산

**주요 개선 사항**:
```python
@dataclass
class LLMJudgeConfig:
    """eval_rubric.yaml에서 설정 로드 가능"""

    @classmethod
    def from_rubric(cls, rubric_path: str) -> "LLMJudgeConfig":
        """rubric YAML 자동 로드"""
        # configs/eval_rubric.yaml의 llm_judge 섹션 읽기
        ...

def judge_one(
    *,
    question: str,
    answer: str,
    evidence: str,
    rubric_path: Optional[str] = None,  # 🆕 추가됨
    cfg: Optional[LLMJudgeConfig] = None,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "scores": {"faithfulness": float, "answer_relevance": float, "context_use": float},
          "perplexity": float,
          "perplexity_ok": bool,
          "verdict": "pass|fail|skip",
          ...
        }
    """
```

**핵심 포인트**:
- ✅ `rubric_path` 파라미터로 YAML 자동 로드
- ✅ 하위 호환 alias 매핑 (factuality → faithfulness, relevance → answer_relevance 등)
- ✅ Non-JSON 출력 방어 (```json ... ``` 펜스 블록 파싱)
- ✅ Perplexity 의존성 부재 시에도 계속 진행 (perplexity=-1.0 기록)

---

## 주요 개선 사항

### 1. 스키마 드리프트 방지 (Schema Drift Prevention)

**문제**: LLM이 가끔 다른 키로 응답 (예: `factuality` 대신 `faithfulness`)

**해결**:
```python
alias_map = {
    "factuality": "faithfulness",      # 레거시 키
    "relevance": "answer_relevance",   # 레거시 키
    "completeness": "answer_relevance", # 일부 모델
    "faithfulness": "faithfulness",    # 표준 키
    "answer_relevance": "answer_relevance",
    "context_use": "context_use",
}

# scores dict + top-level fields 양쪽에서 추출
for k, v in scores.items():
    dst = alias_map.get(k)
    if dst:
        norm_scores[dst] = _clamp01(v)
```

### 2. Non-JSON 출력 방어

**문제**: LLM이 마크다운, 설명문, 코드블록 등과 함께 JSON 반환

**해결**:
```python
def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    1) 직접 json.loads
    2) ```json ... ``` 펜스 블록 찾기
    3) 첫 {...} 블록 찾기
    """
    # 정규표현식으로 JSON 추출
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    ...
```

### 3. Perplexity 의존성 우아한 처리

**문제**: `transformers`/`torch` 미설치 시 perplexity 계산 실패 → 전체 평가 중단

**해결**:
```python
def compute_perplexity(text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        ppl, src = _try_compute_perplexity_hf(text, model_name)
        return {"perplexity": ppl, "perplexity_ok": True, ...}
    except Exception as e:
        return {
            "perplexity": -1.0,
            "perplexity_source": f"unavailable:{type(e).__name__}",
            "perplexity_ok": False,
        }
```

### 4. YAML 한 줄 압축 금지

**문제 (Before)**:
```yaml
llm_judge: enabled: true model: gpt-4o-mini temperature: 0.0 ...
```
→ 파싱은 되지만 diff/리뷰/유지보수 악몽

**해결 (After)**:
```yaml
llm_judge:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 800
  timeout_s: 60
```

### 5. Rubric 기반 설정 로드

**Before**: 코드에 하드코딩
```python
cfg = LLMJudgeConfig(model="gpt-4o-mini", temperature=0.0)
```

**After**: YAML에서 자동 로드
```python
cfg = LLMJudgeConfig.from_rubric("configs/eval_rubric.yaml")
# 또는
result = judge_one(
    question=q,
    answer=a,
    evidence=ev,
    rubric_path="configs/eval_rubric.yaml"  # 자동 로드
)
```

---

## 사용 방법

### 1. 기본 사용 (Rubric 자동 로드)

```python
from tools.llm_as_judge import judge_one

result = judge_one(
    question="당뇨 환자 HbA1c 목표 범위가 어떻게 되나요?",
    answer="일반적으로 7% 미만을 목표로 합니다.",
    evidence="[TS evidence] HbA1c 목표는 7% 미만... (근거 텍스트)",
    rubric_path="configs/eval_rubric.yaml"
)

print(result)
# {
#   "scores": {
#     "faithfulness": 0.85,
#     "answer_relevance": 0.90,
#     "context_use": 0.60
#   },
#   "perplexity": 15.3,
#   "perplexity_ok": True,
#   "perplexity_source": "hf:distilgpt2@cpu",
#   "verdict": "pass",
#   "rationale": "TS 근거와 잘 일치함",
#   "raw_text": "{...}"
# }
```

### 2. Grade Run 파이프라인 통합

```bash
python tools/grade_run.py \
  --pipeline \
  --evalset "experiments/retrieval_tuning/eval_tl.jsonl" \
  --rubric "configs/eval_rubric.yaml" \
  --run "experiments/eval_runs/run.jsonl" \
  --out "experiments/eval_runs/grades.jsonl"
```

내부에서 `llm_as_judge.judge_one(..., rubric_path=args.rubric)`로 호출

### 3. Perplexity 설정

#### Option 1: 환경 변수
```bash
export HF_PERPLEXITY_MODEL=gpt2  # 또는 distilgpt2, gpt2-medium 등
python tools/grade_run.py ...
```

#### Option 2: YAML 설정
```yaml
perplexity:
  enabled: true
  model: gpt2  # distilgpt2보다 크지만 정확
```

#### Option 3: 의존성 설치
```bash
pip install transformers torch

# GPU 사용 (선택)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 자주 터지는 지점 및 해결책

### 1. 스키마 드리프트 (Schema Drift)

**증상**:
```
KeyError: 'faithfulness'  # LLM이 'factuality'로 반환
```

**원인**: LLM이 프롬프트를 무시하고 다른 키 이름 사용

**해결책**:
- ✅ `alias_map`으로 여러 변형 수용
- ✅ `system_prompt`에서 "정확한 스키마" 강조
- ✅ Temperature=0.0으로 일관성 향상

### 2. Non-JSON 출력

**증상**:
```
json.JSONDecodeError: Expecting value: line 1 column 1
```

**원인**: LLM이 설명문이나 마크다운과 함께 JSON 반환
```
Here's the evaluation:
```json
{"scores": ...}
```
```

**해결책**:
- ✅ `_extract_first_json_object()`: 정규표현식으로 JSON 추출
- ✅ System prompt에 "ONLY valid JSON. No markdown." 명시

### 3. Perplexity 의존성 미설치

**증상**:
```
ModuleNotFoundError: No module named 'transformers'
```

**원인**: `transformers`/`torch` 미설치

**해결책**:
- ✅ 예외 처리로 `perplexity=-1.0` 기록 후 계속 진행
- ✅ `perplexity_ok: false` 플래그로 유효성 표시
- ✅ 사용자가 나중에 설치 가능 (선택적)

### 4. YAML 파싱 오류

**증상**:
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**원인**: 한 줄에 여러 키 압축
```yaml
llm_judge: enabled: true model: gpt-4o-mini  # ❌ 잘못됨
```

**해결책**:
```yaml
llm_judge:
  enabled: true
  model: gpt-4o-mini  # ✅ 올바름
```

### 5. 한영 매칭 실패

**증상**: "HbA1c"와 "당화혈색소"를 다른 것으로 인식

**원인**: 양언어 alias 부재

**해결책**:
```yaml
bilingual:
  glossary:
    - canonical: HbA1c
      ko: [HbA1c, 당화혈색소, 헤모글로빈 A1c]
      en: [HbA1c, hemoglobin A1c]
```

---

## 테스트 및 검증

### 1. Import 테스트

```bash
python -c "from tools.llm_as_judge import judge_one, LLMJudgeConfig; print('OK')"
```

### 2. Rubric 로드 테스트

```python
from tools.llm_as_judge import LLMJudgeConfig

cfg = LLMJudgeConfig.from_rubric("configs/eval_rubric.yaml")
print(f"Model: {cfg.model}")
print(f"Threshold: {cfg.threshold}")
print(f"Enabled: {cfg.enabled}")
```

**예상 출력**:
```
Model: gpt-4o-mini
Threshold: 0.75
Enabled: True
```

### 3. Perplexity 계산 테스트

```python
from tools.llm_as_judge import compute_perplexity

result = compute_perplexity("당뇨 환자는 혈당 관리가 중요합니다.")
print(result)
```

**예상 출력** (transformers 설치 시):
```python
{
  'perplexity': 23.45,
  'perplexity_source': 'hf:distilgpt2@cpu',
  'perplexity_ok': True
}
```

**예상 출력** (transformers 미설치 시):
```python
{
  'perplexity': -1.0,
  'perplexity_source': 'unavailable:ModuleNotFoundError',
  'perplexity_ok': False
}
```

### 4. End-to-End 평가 테스트

```python
from tools.llm_as_judge import judge_one

result = judge_one(
    question="메트포르민 부작용이 뭔가요?",
    answer="메트포르민의 주요 부작용은 위장 장애입니다.",
    evidence="[TS] 메트포르민은 위장관 부작용(설사, 복통)이 흔합니다.",
    rubric_path="configs/eval_rubric.yaml"
)

assert result['verdict'] in ['pass', 'fail', 'skip']
assert 0.0 <= result['scores']['faithfulness'] <= 1.0
assert 'perplexity' in result
print("✅ 모든 테스트 통과")
```

---

## 문제 해결 가이드

### Q1: "LLM이 계속 factuality 키를 반환해요"

**A**: 정상입니다. `alias_map`이 자동으로 `faithfulness`로 매핑합니다.

확인:
```python
result = judge_one(...)
print(result['scores'])  # {'faithfulness': 0.8, ...}  ← 정상 변환됨
```

### Q2: "Perplexity가 항상 -1.0이에요"

**A**: `transformers`/`torch` 미설치입니다. 선택사항이므로 계속 진행 가능합니다.

설치 (선택):
```bash
pip install transformers torch
```

### Q3: "Rubric 로드 실패 경고가 떠요"

**A**: YAML 경로를 확인하세요.

```python
import os
rubric_path = "configs/eval_rubric.yaml"
print(f"Exists: {os.path.exists(rubric_path)}")  # True여야 함
```

### Q4: "한국어 질문에 영어로 답해도 점수가 낮아요"

**A**: `system_prompt`에 명시되어 있습니다:
```
※ 한국어/영어가 섞여도 언어 자체로 감점하지 말고, 의미/정확성으로만 평가하세요.
```

LLM이 프롬프트를 무시하는 경우 temperature를 더 낮추거나 (0.0), few-shot 예제 추가를 고려하세요.

### Q5: "Grade run이 실패해요"

**A**: 다음을 확인하세요:

1. **필수 파일 존재**:
   ```bash
   ls configs/eval_rubric.yaml
   ls configs/question_templates.yaml
   ```

2. **Agent import 문제** (이전 이슈):
   ```bash
   python -c "from agent.entrypoint import run_agent; import agent.graph; print('OK')"
   ```

3. **LLM API 키**:
   ```bash
   echo $OPENAI_API_KEY  # 설정되어 있어야 함
   ```

---

## 다음 단계

### 1. 통합 테스트 스크립트 작성

```bash
# tests/test_llm_judge.sh
#!/bin/bash
set -e

echo "1. Import 테스트..."
python -c "from tools.llm_as_judge import judge_one; print('✅ Import OK')"

echo "2. Rubric 로드 테스트..."
python -c "
from tools.llm_as_judge import LLMJudgeConfig
cfg = LLMJudgeConfig.from_rubric('configs/eval_rubric.yaml')
print(f'✅ Loaded: {cfg.model}')
"

echo "3. Perplexity 테스트..."
python -c "
from tools.llm_as_judge import compute_perplexity
result = compute_perplexity('테스트 문장')
print(f'✅ Perplexity: {result}')
"

echo "✅ 모든 테스트 통과!"
```

### 2. CI/CD 통합

```yaml
# .github/workflows/test.yml
name: LLM Judge Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: bash tests/test_llm_judge.sh
```

### 3. 메트릭 모니터링 대시보드

```python
# tools/analyze_grades.py
import json
import pandas as pd

with open("experiments/eval_runs/grades.jsonl") as f:
    grades = [json.loads(line) for line in f]

df = pd.DataFrame([
    {
        'question_id': g['question_id'],
        'faithfulness': g['scores']['faithfulness'],
        'answer_relevance': g['scores']['answer_relevance'],
        'perplexity': g['perplexity'],
        'verdict': g['verdict']
    }
    for g in grades
])

print(df.describe())
print(f"\nPass rate: {(df['verdict']=='pass').mean():.1%}")
```

---

## 결론

이 문서는 **faithfulness, answer_relevance, perplexity** 3개 핵심 메트릭을 중심으로 한 평가 시스템의 안정화 방법을 정리했습니다.

### 핵심 성과
- ✅ 스키마 드리프트 방지 (alias 매핑)
- ✅ Non-JSON 출력 방어 (정규표현식 추출)
- ✅ Perplexity 의존성 우아한 처리
- ✅ YAML 기반 설정 자동 로드
- ✅ 한영 양언어 지원 강화

### 유지보수 원칙
1. **SSOT**: `configs/eval_rubric.yaml`이 모든 설정의 단일 진실원
2. **하위 호환**: 레거시 키도 alias로 수용
3. **방어적 프로그래밍**: 외부 입력(LLM 출력, YAML) 항상 검증
4. **명시적 에러**: 실패 시 이유를 명확히 기록

---

**문의 및 피드백**: 이슈가 발생하면 다음을 첨부하여 보고해주세요:
- `configs/eval_rubric.yaml` 내용
- `llm_as_judge` 호출 코드
- 전체 에러 메시지 및 스택 트레이스
- `perplexity_ok` 값 (True/False)
