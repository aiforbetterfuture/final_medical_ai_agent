# MedCAT 설정 및 실행 가이드

**작성일**: 2025년 12월 16일  
**목적**: 새 스캐폴드에서 MedCAT을 빠르게 설정하고 실행하는 방법

---

## 📋 개요

이 가이드는 기존 스캐폴드에서 복사된 MedCAT 파일들을 새 스캐폴드에서 실행하는 방법을 설명합니다.

### 복사된 파일 목록

```
C:\Users\KHIDI\Downloads\final_medical_ai_agent\
├── extraction\
│   ├── medcat2_adapter.py          ⭐ 핵심 어댑터
│   ├── multilingual_medcat.py      (다국어 지원)
│   ├── neural_translator.py        (번역기)
│   ├── slot_extractor.py           (슬롯 추출)
│   └── __init__.py
│
├── medcat2\
│   └── mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5\  ⭐ 모델팩
│       ├── cdb.dat
│       ├── vocab.dat
│       ├── model_card.json
│       ├── meta_Status\
│       └── spacy_model\
│
├── medcat2_install\
│   ├── MEDCAT2_QUICK_START.md      ⭐ 빠른 시작
│   ├── MEDCAT2_INTEGRATION_GUIDE.md
│   ├── medcat2_usage_example.py
│   └── ... (27개 문서)
│
├── test_medcat_integration.py      ⭐ 테스트 스크립트
├── test_multilingual.py
├── MEDCAT2_INSTALLATION_REPORT.md  ⭐ 설치 보고서
└── MULTILINGUAL_MEDCAT_GUIDE.md
```

---

## 🚀 빠른 시작 (5분)

### Step 1: 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# MedCAT 모델 경로 (절대 경로 사용)
MEDCAT2_MODEL_PATH=C:\Users\KHIDI\Downloads\final_medical_ai_agent\medcat2\mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5

# OpenAI API 키 (이미 설정되어 있을 수 있음)
OPENAI_API_KEY=your_openai_api_key_here

# Google API 키 (번역용, 선택적)
GOOGLE_API_KEY=your_google_api_key_here
```

**중요**: 경로는 실제 파일 위치에 맞게 수정하세요!

### Step 2: 의존성 설치

```bash
# 가상환경 활성화
.venv\Scripts\activate

# MedCAT 설치 (아직 설치 안 했다면)
pip install medcat>=2.0

# 추가 의존성
pip install spacy
pip install langdetect
pip install googletrans==4.0.0rc1
```

### Step 3: 테스트 실행

```bash
# MedCAT 통합 테스트
python test_medcat_integration.py
```

**예상 출력**:
```
================================================================================
MedCAT 통합 테스트
================================================================================

[1] 환경 변수 확인
--------------------------------------------------------------------------------
✓ MEDCAT2_MODEL_PATH: C:\Users\KHIDI\Downloads\final_medical_ai_agent\medcat2\...
✓ 모델 파일 존재 확인

[2] MedCAT 모델 로드
--------------------------------------------------------------------------------
✓ 모델 로드 성공
  - CDB 크기: 12345 concepts
  - Vocab 크기: 67890 tokens

[3] 영어 텍스트 엔티티 추출 테스트
--------------------------------------------------------------------------------
✓ 엔티티 추출 성공
  - Conditions: 4개
  - Symptoms: 2개
  - Medications: 1개

[4] 한국어 텍스트 처리 테스트
--------------------------------------------------------------------------------
✓ 한국어 → 영어 번역 성공
✓ 엔티티 추출 성공
  - Symptoms: 1개

================================================================================
✅ 모든 테스트 통과!
================================================================================
```

---

## 📚 상세 사용법

### 1. Python 코드에서 MedCAT 사용

```python
# extraction/medcat2_adapter.py 사용 예시
from extraction.medcat2_adapter import MedCAT2Adapter

# 어댑터 초기화
adapter = MedCAT2Adapter()

# 영어 텍스트 엔티티 추출
text_en = "55 year old male with hypertension and diabetes, taking metformin"
entities = adapter.extract_entities(text_en)

print(f"추출된 엔티티: {len(entities)}개")
for entity in entities:
    print(f"  - {entity['text']}: {entity['cui']} ({entity['category']})")

# 한국어 텍스트 처리 (자동 번역)
text_ko = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중"
entities_ko = adapter.extract_entities(text_ko)

print(f"한국어 텍스트에서 추출된 엔티티: {len(entities_ko)}개")
```

### 2. 슬롯 추출기와 통합

```python
# extraction/slot_extractor.py 사용
from extraction.slot_extractor import SlotExtractor

extractor = SlotExtractor()

# 환자 텍스트에서 슬롯 추출
user_text = "65세 남성 환자로 당뇨병이 있고 메트포르민을 복용 중입니다"
slots = extractor.extract(user_text)

print("추출된 슬롯:")
print(f"  - 나이: {slots.get('age')}")
print(f"  - 성별: {slots.get('gender')}")
print(f"  - 질환: {slots.get('conditions')}")
print(f"  - 약물: {slots.get('medications')}")
```

### 3. 다국어 지원

```python
# extraction/multilingual_medcat.py 사용
from extraction.multilingual_medcat import MultilingualMedCAT

ml_medcat = MultilingualMedCAT()

# 자동 언어 감지 및 처리
texts = [
    "Patient has diabetes and hypertension",  # 영어
    "환자는 당뇨병과 고혈압이 있습니다",      # 한국어
    "患者有糖尿病和高血压",                   # 중국어
]

for text in texts:
    entities = ml_medcat.extract_entities(text)
    print(f"텍스트: {text}")
    print(f"추출: {len(entities)}개 엔티티")
```

---

## 🔧 문제 해결

### 문제 1: 모델 로드 실패

**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```

**해결책**:
1. `.env` 파일의 `MEDCAT2_MODEL_PATH` 경로 확인
2. 경로에 한글이 포함되어 있지 않은지 확인
3. 절대 경로 사용 권장

```env
# ❌ 잘못된 예
MEDCAT2_MODEL_PATH=./medcat2/mc_modelpack_...

# ✅ 올바른 예
MEDCAT2_MODEL_PATH=C:\Users\KHIDI\Downloads\final_medical_ai_agent\medcat2\mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5
```

### 문제 2: 엔티티 추출 안 됨

**증상**:
```python
entities = adapter.extract_entities(text)
# entities = []  # 빈 리스트
```

**해결책**:
1. 텍스트가 영어인지 확인 (한국어는 자동 번역됨)
2. 의학 용어가 포함되어 있는지 확인
3. confidence threshold 조정

```python
# confidence threshold 낮추기
adapter = MedCAT2Adapter(confidence_threshold=0.3)  # 기본값: 0.5
```

### 문제 3: 한국어 번역 품질 낮음

**증상**:
```
"고혈압" → "고현압" (오타 발생)
```

**해결책**:
1. `extraction/neural_translator.py`의 번역 사전 업데이트
2. Google Translate API 사용 (더 정확)

```python
# .env 파일에 Google API 키 추가
GOOGLE_API_KEY=your_google_api_key

# multilingual_medcat.py에서 자동으로 Google Translate 사용
```

### 문제 4: 의존성 충돌

**증상**:
```
ImportError: cannot import name 'CAT' from 'medcat'
```

**해결책**:
```bash
# MedCAT 재설치
pip uninstall medcat -y
pip install medcat>=2.0

# spacy 모델 다운로드
python -m spacy download en_core_web_md
```

---

## 📊 성능 최적화

### 1. 모델 캐싱

```python
# 싱글톤 패턴으로 모델 재사용
from extraction.medcat2_adapter import MedCAT2Adapter

# 첫 호출: 모델 로드 (느림, ~10초)
adapter1 = MedCAT2Adapter()

# 이후 호출: 캐시된 모델 사용 (빠름, ~0.1초)
adapter2 = MedCAT2Adapter()  # 동일한 인스턴스 반환
```

### 2. 배치 처리

```python
# 여러 텍스트를 한 번에 처리
texts = [
    "Patient 1: diabetes",
    "Patient 2: hypertension",
    "Patient 3: asthma"
]

# 배치 처리 (더 빠름)
all_entities = []
for text in texts:
    entities = adapter.extract_entities(text)
    all_entities.append(entities)
```

### 3. 멀티스레딩 (주의!)

```python
# ⚠️ MedCAT은 thread-safe하지 않음!
# 멀티스레딩 사용 시 각 스레드마다 별도 인스턴스 생성 필요

from concurrent.futures import ThreadPoolExecutor

def process_text(text):
    # 각 스레드에서 새 인스턴스 생성
    adapter = MedCAT2Adapter()
    return adapter.extract_entities(text)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_text, texts))
```

---

## 🔍 고급 기능

### 1. Confidence Threshold 조정

```python
# 낮은 confidence 엔티티도 포함
adapter = MedCAT2Adapter(confidence_threshold=0.3)

# 높은 confidence만 포함 (더 정확)
adapter = MedCAT2Adapter(confidence_threshold=0.7)
```

### 2. 특정 카테고리만 추출

```python
# 조건(질환)만 추출
entities = adapter.extract_entities(text)
conditions = [e for e in entities if e['category'] == 'condition']

# 약물만 추출
medications = [e for e in entities if e['category'] == 'medication']
```

### 3. CUI 코드로 UMLS 정보 조회

```python
# 추출된 CUI로 UMLS 정보 조회
for entity in entities:
    cui = entity['cui']
    print(f"CUI: {cui}")
    print(f"  - Name: {entity['text']}")
    print(f"  - Semantic Type: {entity.get('semantic_type', 'N/A')}")
    print(f"  - Confidence: {entity['confidence']:.2f}")
```

---

## 📖 추가 문서

### 필수 문서
- **MEDCAT2_QUICK_START.md**: 빠른 시작 가이드
- **MEDCAT2_INSTALLATION_REPORT.md**: 설치 및 테스트 보고서
- **MULTILINGUAL_MEDCAT_GUIDE.md**: 다국어 지원 가이드

### 고급 문서
- **MEDCAT2_INTEGRATION_GUIDE.md**: 시스템 통합 가이드
- **MEDCAT2_KOREAN_EXTRACTION_METHODOLOGY.md**: 한국어 추출 방법론
- **MEDCAT2_VS_LLM_EXTRACTION_COMPARISON.md**: MedCAT vs LLM 비교

### 학습 문서
- **MEDCAT2_SUPERVISED_TRAINING_STRATEGY.md**: 지도 학습 전략
- **MEDCAT2_UNSUPERVISED_TRAINING_STRATEGY.md**: 비지도 학습 전략
- **MEDCAT2_UMLS_RRF_GUIDE.md**: UMLS RRF 가이드

---

## ✅ 체크리스트

### 설치 확인
- [ ] `.env` 파일에 `MEDCAT2_MODEL_PATH` 설정
- [ ] 모델팩 파일 존재 확인
- [ ] `pip install medcat>=2.0` 완료
- [ ] `python test_medcat_integration.py` 성공

### 통합 확인
- [ ] `from extraction.medcat2_adapter import MedCAT2Adapter` 성공
- [ ] 영어 텍스트 엔티티 추출 성공
- [ ] 한국어 텍스트 처리 성공
- [ ] 슬롯 추출기와 통합 성공

### 성능 확인
- [ ] 모델 로드 시간 < 15초
- [ ] 엔티티 추출 시간 < 1초 (per text)
- [ ] Confidence > 0.5인 엔티티 추출됨

---

## 🎯 다음 단계

### 1. Modular RAG와 통합

```python
# modules/pre_retrieval/medcat_entity_extractor.py
from core.module_interface import RAGModule, RAGContext
from extraction.medcat2_adapter import MedCAT2Adapter

class MedCATEntityExtractorModule(RAGModule):
    """MedCAT 기반 엔티티 추출 모듈"""
    
    def __init__(self, config):
        super().__init__(config)
        self.adapter = MedCAT2Adapter()
    
    def execute(self, context: RAGContext) -> RAGContext:
        # 쿼리에서 의학 엔티티 추출
        entities = self.adapter.extract_entities(context.query)
        
        # 컨텍스트에 추가
        context.metadata['medical_entities'] = entities
        context.metadata['num_entities'] = len(entities)
        
        # 엔티티로 쿼리 증강
        entity_terms = [e['text'] for e in entities]
        if entity_terms:
            context.query += f" [Entities: {', '.join(entity_terms)}]"
        
        return context
```

### 2. Query Rewriter와 결합

```python
# 엔티티 정보를 활용한 쿼리 재작성
class EntityAwareQueryRewriter(RAGModule):
    def execute(self, context: RAGContext) -> RAGContext:
        entities = context.metadata.get('medical_entities', [])
        
        # 엔티티의 CUI 코드로 동의어 확장
        synonyms = []
        for entity in entities:
            cui = entity['cui']
            # UMLS에서 동의어 조회
            synonyms.extend(self.get_synonyms(cui))
        
        # 쿼리에 동의어 추가
        if synonyms:
            context.query += f" OR {' OR '.join(synonyms)}"
        
        return context
```

### 3. Ablation 실험에 포함

```python
# experiments/medcat_ablation.py
EXPERIMENTS = {
    'E1_without_medcat': {
        'use_medcat': False
    },
    'E2_with_medcat': {
        'use_medcat': True,
        'confidence_threshold': 0.5
    },
    'E3_medcat_low_threshold': {
        'use_medcat': True,
        'confidence_threshold': 0.3
    }
}
```

---

## 📞 지원 및 문의

### 문제 발생 시
1. `test_medcat_integration.py` 실행 결과 확인
2. `MEDCAT2_INSTALLATION_REPORT.md` 참고
3. GitHub Issues 또는 담당자에게 문의

### 추가 리소스
- MedCAT 공식 문서: https://github.com/CogStack/MedCAT
- UMLS 브라우저: https://uts.nlm.nih.gov/uts/
- SNOMED CT: https://www.snomed.org/

---

**문서 버전**: 1.0  
**최종 수정**: 2025년 12월 16일  
**작성자**: Medical AI Agent Research Team

**관련 파일**:
- `extraction/medcat2_adapter.py` (핵심 어댑터)
- `test_medcat_integration.py` (테스트 스크립트)
- `.env` (환경 변수 설정)

---

**END OF DOCUMENT**

