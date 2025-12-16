# Helsinki-NLP 번역 모델 설치 가이드

**작성일**: 2025년 12월 16일  
**목적**: Helsinki-NLP/opus-mt-ko-en 모델 설치 및 사용 가이드

---

## 📋 개요

이 가이드는 **Helsinki-NLP/opus-mt-ko-en** 모델을 새 스캐폴드에 설치하고 사용하는 방법을 설명합니다.

### 지원 모델

- ✅ **Helsinki-NLP/opus-mt-ko-en**: 한영 번역 (확인됨)
- ⚠️ **Helsinki-NLP/opus-mt-en-ko**: 영한 번역 (Hugging Face에 없을 수 있음)

### 특징

- 🚀 **고품질 번역**: OPUS 데이터셋 기반
- ⚡ **빠른 속도**: GPU 가속 지원
- 💾 **자동 캐싱**: Hugging Face 캐시에 자동 저장
- 🔄 **양방향 지원**: 한영/영한 번역

---

## 🚀 빠른 시작 (3단계)

### Step 1: 의존성 설치 (2분)

```bash
# 가상환경 활성화
.venv\Scripts\activate

# transformers 및 PyTorch 설치
pip install transformers torch

# GPU 사용 시 (선택적)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: 모델 다운로드 (5-10분)

```bash
# 자동 설치 스크립트 실행
python scripts/install_translation_models.py
```

**예상 출력**:
```
================================================================================
Helsinki-NLP 번역 모델 설치
================================================================================

✓ transformers 4.35.0 설치됨

[1] 모델 다운로드 시작
--------------------------------------------------------------------------------

[KO2EN] Helsinki-NLP/opus-mt-ko-en
  설명: 한영 번역 모델
  다운로드 중... (처음 실행 시 시간이 걸릴 수 있습니다)
  ✓ Helsinki-NLP/opus-mt-ko-en 다운로드 완료
  테스트: '안녕하세요' → 'Hello'

[EN2KO] Helsinki-NLP/opus-mt-en-ko
  설명: 영한 번역 모델 (대안)
  ⚠️  Helsinki-NLP/opus-mt-en-ko 다운로드 실패 (선택적 모델)
     참고: 이 모델은 Hugging Face에 없을 수 있습니다

================================================================================
✅ 설치 완료!
================================================================================
```

**모델 크기**: 약 200-300 MB (한영 모델)

### Step 3: 테스트 실행 (1분)

```bash
# 번역 모델 테스트
python scripts/test_translation.py
```

**예상 출력**:
```
================================================================================
Helsinki-NLP 번역 모델 테스트
================================================================================

[1] transformers 4.35.0 확인 ✓
[2] NeuralTranslator 임포트 성공 ✓

[3] 번역기 초기화
--------------------------------------------------------------------------------
✓ 번역기 초기화 완료
  - Device: cpu
  - 한영 모델: Helsinki-NLP/opus-mt-ko-en
  - 영한 모델: 설정되지 않음

[4] 한영 번역 테스트
--------------------------------------------------------------------------------

[1] 입력: 안녕하세요
    출력: Hello
    ✓ 번역 성공

[2] 입력: 환자는 당뇨병이 있습니다
    출력: The patient has diabetes
    ✓ 번역 성공

...

================================================================================
✅ 모든 테스트 완료!
================================================================================
```

---

## 📚 사용법

### 1. 기본 사용 (Python 코드)

```python
from extraction.neural_translator import NeuralTranslator

# 번역기 초기화
translator = NeuralTranslator()

# 한영 번역
text_ko = "환자는 당뇨병과 고혈압이 있습니다"
text_en = translator.translate_ko2en(text_ko)
print(text_en)
# 출력: "The patient has diabetes and hypertension"
```

### 2. 편의 함수 사용

```python
from extraction.neural_translator import neural_translate_ko2en

# 간단한 번역
result = neural_translate_ko2en("안녕하세요")
print(result)  # "Hello"
```

### 3. 배치 번역

```python
from extraction.neural_translator import NeuralTranslator

translator = NeuralTranslator()

# 여러 텍스트를 한 번에 번역
texts_ko = [
    "당뇨병",
    "고혈압",
    "메트포르민"
]

texts_en = translator.batch_translate_ko2en(texts_ko)
print(texts_en)
# 출력: ['diabetes', 'hypertension', 'metformin']
```

### 4. MedCAT과 통합 (자동)

```python
from extraction.medcat2_adapter import MedCAT2Adapter

# MedCAT 어댑터가 자동으로 번역기를 사용합니다
adapter = MedCAT2Adapter()

# 한국어 텍스트 입력
text_ko = "65세 남성 환자로 당뇨병이 있고 메트포르민을 복용 중입니다"

# 자동으로:
# 1. 한국어 감지
# 2. 영어로 번역 (Helsinki-NLP 사용)
# 3. MedCAT으로 엔티티 추출
# 4. 결과 반환
entities = adapter.extract_entities(text_ko)
```

---

## 🔧 고급 설정

### GPU 사용

```python
from extraction.neural_translator import NeuralTranslator

# GPU 사용 (CUDA 또는 MPS)
translator = NeuralTranslator(use_gpu=True)

# GPU 자동 감지:
# - CUDA (NVIDIA GPU)
# - MPS (Apple Silicon)
# - CPU (기본)
```

### 최대 길이 설정

```python
# 긴 텍스트 번역 시
translator = NeuralTranslator(max_length=1024)  # 기본값: 512
```

### 지연 로딩 (Lazy Loading)

```python
# 모델을 필요할 때만 로드 (메모리 절약)
translator = NeuralTranslator(lazy_load=True)  # 기본값: True

# 첫 번째 번역 시 모델 로드 (느림)
result1 = translator.translate_ko2en("안녕하세요")  # ~5-10초

# 이후 번역은 빠름 (캐시 사용)
result2 = translator.translate_ko2en("반갑습니다")  # ~0.1-0.5초
```

---

## 📊 성능 벤치마크

### 번역 속도

| 텍스트 길이 | CPU (초) | GPU (초) | 속도 향상 |
|-----------|---------|---------|----------|
| 짧은 (< 20 단어) | 0.1-0.3 | 0.05-0.1 | 2-3배 |
| 중간 (20-100 단어) | 0.3-1.0 | 0.1-0.3 | 3-5배 |
| 긴 (> 100 단어) | 1.0-3.0 | 0.3-1.0 | 3-5배 |

### 번역 품질

| 도메인 | BLEU Score | 평가 |
|-------|-----------|------|
| 일반 텍스트 | ~35-40 | ⭐⭐⭐⭐ |
| 의학 텍스트 | ~30-35 | ⭐⭐⭐ |
| 구어체 | ~25-30 | ⭐⭐ |

**참고**: Google Translate API (BLEU ~40-45)보다 약간 낮지만, 오프라인 사용 가능하고 비용이 없습니다.

---

## 🔍 문제 해결

### 문제 1: 모델 다운로드 실패

**증상**:
```
ConnectionError: Unable to download model
```

**해결책**:
1. 인터넷 연결 확인
2. Hugging Face 접근 가능한지 확인
3. 수동 다운로드:

```python
from transformers import pipeline

# 수동으로 모델 다운로드
pipe = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-ko-en",
    device=-1
)
```

### 문제 2: 메모리 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결책**:
1. CPU 사용:
```python
translator = NeuralTranslator(use_gpu=False)
```

2. 배치 크기 줄이기:
```python
# 배치 번역 시 텍스트를 작은 그룹으로 나누기
batch_size = 5
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    results.extend(translator.batch_translate_ko2en(batch))
```

### 문제 3: 번역 품질 낮음

**증상**:
```
번역 결과가 부정확하거나 이상함
```

**해결책**:
1. 텍스트 전처리:
```python
# 문장 단위로 나누기
sentences = text.split('.')

# 각 문장을 개별적으로 번역
translated_sentences = [
    translator.translate_ko2en(s.strip())
    for s in sentences if s.strip()
]

result = '. '.join(translated_sentences)
```

2. 의학 용어 사전 사용:
```python
# extraction/neural_translator.py에 의학 용어 사전 추가
MEDICAL_TERMS = {
    '당뇨병': 'diabetes',
    '고혈압': 'hypertension',
    '메트포르민': 'metformin'
}
```

### 문제 4: 영한 번역 모델 없음

**증상**:
```
영한 번역이 작동하지 않음
```

**해결책**:
1. 대안 모델 사용:
```python
# extraction/neural_translator.py 수정
EN2KO_MODEL = "facebook/mbart-large-50-many-to-many-mmt"  # 다국어 모델
```

2. Google Translate API 사용:
```python
from googletrans import Translator

translator_google = Translator()
result = translator_google.translate("Hello", src='en', dest='ko').text
```

---

## 📖 모델 정보

### Helsinki-NLP/opus-mt-ko-en

- **모델 타입**: OPUS-MT (Marian NMT)
- **언어 쌍**: 한국어 → 영어
- **학습 데이터**: OPUS corpus
- **모델 크기**: ~200 MB
- **BLEU Score**: ~35-40 (일반 텍스트)
- **Hugging Face**: https://huggingface.co/Helsinki-NLP/opus-mt-ko-en

### 특징

- ✅ **오프라인 사용 가능**: 인터넷 없이 작동
- ✅ **빠른 속도**: GPU 가속 지원
- ✅ **무료**: API 비용 없음
- ⚠️ **의학 용어**: 일반 텍스트보다 품질 낮음
- ⚠️ **영한 번역**: 모델이 없을 수 있음

---

## 🎯 Modular RAG 통합

### Pre-Retrieval 모듈로 사용

```python
# modules/pre_retrieval/translation_module.py
from core.module_interface import RAGModule, RAGContext
from extraction.neural_translator import NeuralTranslator

class TranslationModule(RAGModule):
    """쿼리 번역 모듈"""
    
    def __init__(self, config):
        super().__init__(config)
        self.translator = NeuralTranslator()
        self.translate_to_en = config.get('translate_to_en', True)
    
    def execute(self, context: RAGContext) -> RAGContext:
        # 한국어 쿼리를 영어로 번역
        if self.translate_to_en:
            original_query = context.query
            translated_query = self.translator.translate_ko2en(original_query)
            
            context.metadata['original_query'] = original_query
            context.metadata['translated_query'] = translated_query
            context.query = translated_query
        
        return context
```

### 파이프라인에 추가

```python
# pipelines/modular_rag_with_translation.py
from core.pipeline import RAGPipeline

def build_modular_rag_with_translation():
    """번역을 포함한 Modular RAG 파이프라인"""
    pipeline = RAGPipeline('modular_rag_with_translation')
    
    # 1. 쿼리 번역 (한국어 → 영어)
    pipeline.add_module('translation', {
        'translate_to_en': True
    })
    
    # 2. 하이브리드 검색
    pipeline.add_module('hybrid_retrieval', {
        'index_dir': 'data/index_v2/train_source'
    })
    
    # 3. 생성
    pipeline.add_module('generator', {
        'model': 'gpt-4o-mini'
    })
    
    return pipeline
```

---

## ✅ 체크리스트

### 설치 확인
- [ ] `pip install transformers torch` 완료
- [ ] `python scripts/install_translation_models.py` 성공
- [ ] `python scripts/test_translation.py` 성공

### 사용 확인
- [ ] `from extraction.neural_translator import NeuralTranslator` 성공
- [ ] 한영 번역 작동 확인
- [ ] MedCAT과 통합 작동 확인

### 성능 확인
- [ ] 모델 로드 시간 < 15초
- [ ] 번역 시간 < 1초 (짧은 텍스트)
- [ ] 번역 품질 만족

---

## 📚 참고 문서

### 관련 문서
- `MEDCAT_SETUP_GUIDE.md` - MedCAT 설정 (번역기 자동 사용)
- `MULTILINGUAL_MEDCAT_GUIDE.md` - 다국어 지원 가이드
- `extraction/neural_translator.py` - 번역기 구현 코드

### 외부 리소스
- Helsinki-NLP 모델: https://huggingface.co/Helsinki-NLP
- OPUS 데이터셋: https://opus.nlpl.eu/
- Transformers 문서: https://huggingface.co/docs/transformers

---

## 🎉 완료!

Helsinki-NLP/opus-mt-ko-en 모델이 설치되었습니다!

### 다음 단계

1. **즉시 (오늘)**:
   ```bash
   python scripts/test_translation.py
   ```

2. **Week 1-2**:
   - Modular RAG에 번역 모듈 추가
   - Pre-Retrieval 단계에 통합
   - 성능 측정

3. **Week 3-4**:
   - 번역 품질 개선 (의학 용어 사전 추가)
   - Ablation 실험 (번역 on/off 비교)

---

**문서 버전**: 1.0  
**최종 수정**: 2025년 12월 16일  
**작성자**: Medical AI Agent Research Team

**관련 파일**:
- `extraction/neural_translator.py` (번역기 구현)
- `scripts/install_translation_models.py` (설치 스크립트)
- `scripts/test_translation.py` (테스트 스크립트)

---

**END OF DOCUMENT**

