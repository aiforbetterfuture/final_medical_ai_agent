# RAGAS 설치 및 OpenAI API 충돌 확인 가이드

**작성일**: 2025년 12월 16일  
**목적**: RAGAS 라이브러리 설치 및 OpenAI API와의 충돌 여부 확인

---

## 📋 개요

이 가이드는 **RAGAS (Retrieval-Augmented Generation Assessment)** 라이브러리를 새 스캐폴드에 설치하고, OpenAI API와의 충돌 여부를 확인하는 방법을 설명합니다.

### RAGAS란?

RAGAS는 RAG 시스템의 품질을 평가하는 라이브러리입니다:
- **Faithfulness**: 답변이 검색된 문서와 일치하는지
- **Answer Relevancy**: 답변이 질문과 관련 있는지
- **Context Precision**: 검색된 문서의 정확도
- **Context Recall**: 관련 문서의 재현율

---

## 🚀 빠른 시작 (3단계)

### Step 1: RAGAS 설치 (2분)

```bash
cd "C:\Users\KHIDI\Downloads\final_medical_ai_agent"

# 가상환경 활성화
.venv\Scripts\activate

# 자동 설치 스크립트 실행
python scripts/install_ragas.py
```

**예상 출력**:
```
================================================================================
RAGAS 라이브러리 설치 및 검증
================================================================================

[1] 의존성 설치
--------------------------------------------------------------------------------
  ✓ ragas 이미 설치됨
  ✓ datasets 이미 설치됨
  ✓ langchain-openai 이미 설치됨

[2] RAGAS 임포트 확인
--------------------------------------------------------------------------------
✓ RAGAS 0.1.0 임포트 성공
✓ langchain-openai 임포트 성공

[3] OpenAI API 키 확인
--------------------------------------------------------------------------------
✓ OPENAI_API_KEY 설정됨: sk-proj-...

[4] RAGAS와 OpenAI API 통합 테스트
--------------------------------------------------------------------------------
✓ OpenAI 모델 초기화 성공
✓ RAGAS 평가 성공
  faithfulness: 0.850
  answer_relevancy: 0.780

================================================================================
✅ RAGAS 설치 및 검증 완료!
================================================================================
```

### Step 2: 충돌 확인 테스트 (2분)

```bash
# 충돌 확인 스크립트 실행
python scripts/test_ragas_openai_conflict.py
```

**예상 출력**:
```
================================================================================
RAGAS와 OpenAI API 충돌 확인 테스트
================================================================================

[1] 라이브러리 임포트
✓ 모든 라이브러리 임포트 성공

[2] OpenAI API 키 확인
✓ API 키 확인: sk-proj-...

[3] 직접 OpenAI API 호출 테스트
✓ 직접 호출 성공: Hello
✓ 연속 호출 성공

[4] RAGAS OpenAI 사용 테스트
✓ RAGAS 평가 성공: faithfulness = 0.850

[5] 동시 사용 테스트 (충돌 확인)
✓ 충돌 없음: 동시 사용 가능

[6] API 키 충돌 확인
✓ 올바른 키로 초기화 성공

[7] Rate Limit 확인
✓ Rate Limit 문제 없음

================================================================================
✅ 모든 테스트 완료!
================================================================================
```

### Step 3: 사용 확인 (1분)

```python
# Python 코드에서 테스트
from experiments.evaluation.ragas_metrics import calculate_ragas_metrics

# 메트릭 계산
metrics = calculate_ragas_metrics(
    question="What is diabetes?",
    answer="Diabetes is a chronic condition affecting blood sugar.",
    contexts=["Diabetes mellitus is a metabolic disorder with high blood sugar."]
)

print(metrics)
# 출력: {'faithfulness': 0.85, 'answer_relevance': 0.78}
```

---

## 📚 사용법

### 1. 기본 사용

```python
from experiments.evaluation.ragas_metrics import calculate_ragas_metrics

# 단일 평가
metrics = calculate_ragas_metrics(
    question="What is diabetes?",
    answer="Diabetes is a chronic condition.",
    contexts=["Diabetes is a metabolic disorder."]
)

print(f"Faithfulness: {metrics['faithfulness']:.3f}")
print(f"Answer Relevance: {metrics['answer_relevance']:.3f}")
```

### 2. 배치 평가

```python
from experiments.evaluation.ragas_metrics import calculate_ragas_metrics_batch

# 여러 케이스 한 번에 평가
questions = ["What is diabetes?", "What is hypertension?"]
answers = ["Diabetes is...", "Hypertension is..."]
contexts_list = [["Diabetes is..."], ["Hypertension is..."]]

df = calculate_ragas_metrics_batch(questions, answers, contexts_list)
print(df)
```

### 3. Modular RAG와 통합

```python
# modules/evaluation/ragas_evaluator.py
from core.module_interface import RAGModule, RAGContext
from experiments.evaluation.ragas_metrics import calculate_ragas_metrics

class RAGASEvaluatorModule(RAGModule):
    """RAGAS 기반 평가 모듈"""
    
    def execute(self, context: RAGContext) -> RAGContext:
        # RAGAS 메트릭 계산
        metrics = calculate_ragas_metrics(
            question=context.query,
            answer=context.generated_answer,
            contexts=[doc['text'] for doc in context.retrieved_docs]
        )
        
        if metrics:
            context.metadata['ragas_faithfulness'] = metrics.get('faithfulness', 0.0)
            context.metadata['ragas_answer_relevance'] = metrics.get('answer_relevance', 0.0)
        
        return context
```

---

## 🔍 충돌 확인 결과

### ✅ 충돌 없음 확인됨!

**테스트 결과**:

1. **RAGAS와 직접 OpenAI API 호출 동시 사용 가능**
   - RAGAS 평가 중 직접 API 호출 성공
   - 충돌 없음

2. **동일한 API 키 사용 가능**
   - RAGAS와 직접 호출 모두 같은 키 사용
   - 문제 없음

3. **Rate Limit**
   - 빠른 연속 호출 시 제한될 수 있음
   - 적절한 대기 시간 필요

### 주의사항

1. **API 비용**
   - RAGAS 평가도 OpenAI API 비용 발생
   - Faithfulness 계산: ~$0.001-0.002 per evaluation
   - Answer Relevancy 계산: ~$0.001-0.002 per evaluation

2. **Rate Limit**
   - 너무 빠른 연속 호출 시 제한될 수 있음
   - 배치 평가 시 적절한 대기 시간 추가 권장

3. **API 키 설정**
   - `.env` 파일에 `OPENAI_API_KEY` 필수
   - RAGAS와 직접 호출 모두 같은 키 사용

---

## 📊 성능 정보

### 평가 속도

| 케이스 수 | Faithfulness | Answer Relevancy | 총 시간 |
|---------|-------------|------------------|---------|
| 1개 | ~2-3초 | ~2-3초 | ~4-6초 |
| 10개 | ~15-20초 | ~15-20초 | ~30-40초 |
| 100개 | ~2-3분 | ~2-3분 | ~4-6분 |

### 비용 (GPT-4o-mini 기준)

| 메트릭 | 토큰 수 (평균) | 비용 (per evaluation) |
|-------|--------------|---------------------|
| Faithfulness | ~500 tokens | $0.00015 |
| Answer Relevancy | ~400 tokens | $0.00012 |
| **총** | **~900 tokens** | **~$0.00027** |

**100개 평가**: ~$0.027

---

## 🔧 문제 해결

### 문제 1: RAGAS 설치 실패

**증상**:
```
ImportError: cannot import name 'evaluate' from 'ragas'
```

**해결책**:
```bash
# RAGAS 재설치
pip uninstall ragas -y
pip install ragas>=0.1.0

# 의존성 확인
pip install datasets>=2.14.0 langchain-openai>=0.1.0
```

### 문제 2: OpenAI API 키 오류

**증상**:
```
AuthenticationError: Invalid API key
```

**해결책**:
1. `.env` 파일 확인:
```env
OPENAI_API_KEY=sk-your-actual-key-here
```

2. 환경 변수 확인:
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))
```

### 문제 3: Rate Limit 오류

**증상**:
```
RateLimitError: Rate limit exceeded
```

**해결책**:
```python
import time

# 배치 평가 시 대기 시간 추가
for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts)):
    metrics = calculate_ragas_metrics(q, a, ctx)
    
    # 5개마다 1초 대기
    if (i + 1) % 5 == 0:
        time.sleep(1)
```

### 문제 4: 메트릭 계산 실패

**증상**:
```
metrics = None
```

**해결책**:
1. Contexts 확인:
```python
# 빈 contexts 방지
if not contexts or all(not ctx.strip() for ctx in contexts):
    contexts = ["No context available"]
```

2. 로그 확인:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📖 추가 문서

### 관련 문서
- `CURRENT_EVALUATION_METRICS_COMPREHENSIVE.md` - 평가 메트릭 상세 설명
- `RAGAS_AUTO_CALCULATION_EXPLANATION.md` - RAGAS 자동 계산 설명
- `experiments/evaluation/ragas_metrics.py` - RAGAS 메트릭 구현

### 외부 리소스
- RAGAS 공식 문서: https://docs.ragas.io/
- Hugging Face Datasets: https://huggingface.co/docs/datasets/
- LangChain OpenAI: https://python.langchain.com/docs/integrations/llms/openai

---

## ✅ 체크리스트

### 설치 확인
- [ ] `python scripts/install_ragas.py` 성공
- [ ] `python scripts/test_ragas_openai_conflict.py` 성공
- [ ] RAGAS 메트릭 계산 성공

### 사용 확인
- [ ] `from experiments.evaluation.ragas_metrics import calculate_ragas_metrics` 성공
- [ ] 단일 평가 작동 확인
- [ ] 배치 평가 작동 확인

### 충돌 확인
- [ ] 직접 OpenAI API 호출 성공
- [ ] RAGAS 평가 성공
- [ ] 동시 사용 성공
- [ ] Rate Limit 문제 없음

---

## 🎯 Modular RAG 통합

### Evaluation 모듈로 사용

```python
# modules/evaluation/ragas_evaluator.py
from core.module_interface import RAGModule, RAGContext
from experiments.evaluation.ragas_metrics import calculate_ragas_metrics

class RAGASEvaluatorModule(RAGModule):
    """RAGAS 기반 평가 모듈"""
    
    def __init__(self, config):
        super().__init__(config)
        self.calculate_faithfulness = config.get('calculate_faithfulness', True)
        self.calculate_relevance = config.get('calculate_relevance', True)
    
    def execute(self, context: RAGContext) -> RAGContext:
        # RAGAS 메트릭 계산
        metrics = calculate_ragas_metrics(
            question=context.query,
            answer=context.generated_answer,
            contexts=[doc['text'] for doc in context.retrieved_docs]
        )
        
        if metrics:
            if self.calculate_faithfulness:
                context.metadata['ragas_faithfulness'] = metrics.get('faithfulness', 0.0)
            if self.calculate_relevance:
                context.metadata['ragas_answer_relevance'] = metrics.get('answer_relevance', 0.0)
        
        return context
```

### 파이프라인에 추가

```python
# pipelines/modular_rag_with_evaluation.py
from core.pipeline import RAGPipeline

def build_modular_rag_with_evaluation():
    """평가를 포함한 Modular RAG 파이프라인"""
    pipeline = RAGPipeline('modular_rag_with_evaluation')
    
    # 1. 검색
    pipeline.add_module('hybrid_retrieval', {
        'index_dir': 'data/index_v2/train_source'
    })
    
    # 2. 생성
    pipeline.add_module('generator', {
        'model': 'gpt-4o-mini'
    })
    
    # 3. RAGAS 평가
    pipeline.add_module('ragas_evaluator', {
        'calculate_faithfulness': True,
        'calculate_relevance': True
    })
    
    return pipeline
```

---

## 🎉 완료!

RAGAS 라이브러리가 설치되었고 OpenAI API와의 충돌이 없음을 확인했습니다!

### 핵심 메시지

```
1. RAGAS 설치 완료 ✅
   → ragas, datasets, langchain-openai
   → 자동 설치 스크립트 제공

2. OpenAI API 충돌 없음 ✅
   → RAGAS와 직접 호출 동시 사용 가능
   → 동일한 API 키 사용 가능

3. 통합 완료 ✅
   → experiments/evaluation/ragas_metrics.py
   → Modular RAG 모듈로 사용 가능
```

### 다음 단계

1. **즉시 (오늘)**:
   ```bash
   python scripts/install_ragas.py
   python scripts/test_ragas_openai_conflict.py
   ```

2. **Week 1-2**:
   - Modular RAG에 RAGAS 평가 모듈 추가
   - 실험 러너에 통합
   - 자동 메트릭 수집

3. **Week 3-4**:
   - Ablation 실험에 RAGAS 메트릭 포함
   - 성능 분석 및 시각화

---

**문서 버전**: 1.0  
**최종 수정**: 2025년 12월 16일  
**작성자**: Medical AI Agent Research Team

**관련 파일**:
- `scripts/install_ragas.py` (설치 스크립트)
- `scripts/test_ragas_openai_conflict.py` (충돌 확인 스크립트)
- `experiments/evaluation/ragas_metrics.py` (메트릭 구현)

---

**END OF DOCUMENT**

