# 다국어 MedCAT2 파이프라인 가이드

**작성일**: 2025-12-08  
**버전**: 1.0

---

## 📋 개요

이 문서는 한국어 의료 텍스트에서 MedCAT2를 사용하여 엔티티를 추출하는 다국어 파이프라인을 설명합니다.

### 핵심 구조

```
[사용자 입력 ko] 
   └─(langdetect: 'ko')→ [ko → en 번역]
           └─→ MedCAT2.get_entities(text_en)
                    └─→ (concept_id, cui, semantic_type 등)
                              └─(설명만 en → ko 번역)
```

---

## 🏗️ 아키텍처

### 파일 구조

```
extraction/
├── neural_translator.py     # Helsinki-NLP opus-mt 신경망 번역기
├── multilingual_medcat.py   # 다국어 MedCAT2 래퍼
├── medcat2_adapter.py       # MedCAT2 어댑터 (다국어 지원 확장)
└── slot_extractor.py        # 슬롯 추출기 (다국어 통합)

korean_translator.py         # 사전 기반 번역기 (의료 용어)
```

### 번역 방법

1. **사전 기반 번역** (의료 용어 정확도 높음)
   - 300+ 의료 용어 한영 매핑
   - 질환, 증상, 약물, 검사/수치 포함

2. **Helsinki-NLP 신경망 번역** (일반 문장)
   - `Helsinki-NLP/opus-mt-ko-en`: 한영 번역
   - `Helsinki-NLP/opus-mt-en-ko`: 영한 번역

3. **하이브리드 방식** (권장)
   - 사전 번역으로 의료 용어 먼저 처리
   - 남은 한국어는 신경망 번역으로 처리

---

## 📦 설치

### 필수 패키지

```bash
pip install transformers>=4.30.0 torch>=2.0.0 langdetect>=1.0.9 sentencepiece>=0.1.99
```

### 환경 변수

```powershell
# Windows PowerShell
$env:MEDCAT2_MODEL_PATH = "C:\path\to\modelpack.zip"

# 영구 설정
setx MEDCAT2_MODEL_PATH "C:\path\to\modelpack.zip"
```

---

## 🚀 사용법

### 기본 사용법 (다국어 추출)

```python
from extraction.multilingual_medcat import MultilingualMedCAT

# 다국어 추출기 초기화
medcat = MultilingualMedCAT(
    use_neural_translation=True,   # Helsinki-NLP 신경망 번역
    use_dict_translation=True      # 사전 기반 번역
)

# 한국어 텍스트에서 엔티티 추출
text = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중입니다"
result = medcat.extract_entities(text)

# 결과 확인
print(result["conditions"])    # 질환 목록
print(result["symptoms"])      # 증상 목록
print(result["medications"])   # 약물 목록
print(result["metadata"])      # 번역 메타데이터
```

### 슬롯 추출기 사용 (권장)

```python
from extraction.slot_extractor import SlotExtractor

# 슬롯 추출기 초기화 (다국어 자동 지원)
extractor = SlotExtractor(
    use_medcat2=True,
    use_multilingual=True,
    use_neural_translation=False,  # 사전 기반만 사용 (빠름)
    use_dict_translation=True
)

# 한국어 텍스트에서 슬롯 추출
text = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중이며 혈압 140/90 mmHg, A1c 7.5%"
slots = extractor.extract(text)

# 결과 확인
print(slots["demographics"])   # {'age': 55, 'gender': 'male'}
print(slots["conditions"])     # 질환 목록
print(slots["symptoms"])       # 증상 목록
print(slots["medications"])    # 약물 목록
print(slots["vitals"])         # [{'name': 'SBP', 'value': 140.0, ...}, ...]
print(slots["labs"])           # [{'name': 'A1c', 'value': 7.5, ...}]
```

### 편의 함수

```python
from extraction.medcat2_adapter import medcat2_extract_korean

# 한국어 전용 추출 (간단 버전)
result = medcat2_extract_korean("고혈압과 당뇨가 있습니다")
```

### ChatGPT 제안 형식 (상세)

```python
from extraction.multilingual_medcat import extract_medcat_entities_multilingual

# 상세 엔티티 정보 추출
result = extract_medcat_entities_multilingual("55세 남성, 고혈압 환자")

# 각 엔티티 구조:
# {
#     "cui": "C0011849",
#     "pretty_name_en": "Diabetes mellitus",
#     "pretty_name_ko": "당뇨병",
#     "semantic_type": "Disease or Syndrome",
#     "source_text": "55세 남성, 당뇨가 있습니다",
#     "translated_text": "55 year old male, I have diabetes",
#     "confidence": 0.95,
#     "span_start_en": 27,
#     "span_end_en": 35,
#     "type_ids": [...],
#     "icd10": ["E14.9"]
# }
```

---

## 📊 테스트 결과

### 언어 감지

| 입력 | 감지 결과 |
|------|-----------|
| "55세 남성, 고혈압과 당뇨가 있고..." | ko |
| "55 year old male with hypertension..." | en |
| "I have 고혈압 and 당뇨" | mixed |

### 사전 기반 번역

| 한국어 | 영어 번역 |
|--------|-----------|
| 고혈압 | hypertension |
| 당뇨 | diabetes |
| 메트포르민 | metformin |
| 가슴 답답 | chest tightness |
| 어지러움 | dizziness |

### 엔티티 추출 (한국어 → 영어 번역 후)

| 카테고리 | 추출된 엔티티 | CUI |
|----------|---------------|-----|
| Conditions | Hypertension | 160357008 |
| Conditions | Diabetes mellitus | 73211009 |
| Symptoms | Tight chest | 23924001 |
| Symptoms | Dizziness | 404640003 |
| Medications | Metformin | 372567009 |

---

## 🔧 LangGraph 통합

### 노드에 추가하기

```python
from extraction.multilingual_medcat import preprocess_text_for_medcat

def slot_extraction_node(state):
    """슬롯 추출 노드 (다국어 지원)"""
    user_query = state["query"]
    
    # 1. 전처리 (언어 감지 + 번역)
    translated_text, metadata = preprocess_text_for_medcat(user_query)
    
    # 2. MedCAT2 엔티티 추출
    from extraction.slot_extractor import SlotExtractor
    extractor = SlotExtractor(use_multilingual=True)
    slots = extractor.extract(user_query)
    
    # 3. 상태 업데이트
    state["extracted_slots"] = slots
    state["translation_metadata"] = metadata
    
    return state
```

---

## 📝 의료 용어 사전 확장

### 용어 추가 방법

`korean_translator.py`의 `MEDICAL_TERM_DICT`에 추가:

```python
MEDICAL_TERM_DICT = {
    # 기존 용어들...
    
    # 새 용어 추가
    "심근경색": "myocardial infarction",
    "협심증": "angina",
    "부정맥": "arrhythmia",
}
```

### 포함된 용어 카테고리

- **질환**: 100+ 용어 (당뇨, 고혈압, 암, 심장병 등)
- **증상**: 80+ 용어 (흉통, 호흡곤란, 두통 등)
- **약물**: 50+ 용어 (메트포르민, 스타틴, 항생제 등)
- **검사/수치**: 40+ 용어 (혈압, 혈당, 콜레스테롤 등)
- **인구통계**: 10+ 용어 (나이, 성별, 임신 등)

---

## ⚠️ 알려진 제한사항

1. **번역 품질**: 사전에 없는 신조어나 구어체는 번역되지 않을 수 있음
2. **노이즈 엔티티**: "per year", "Male structure" 같은 노이즈가 추출될 수 있음
3. **신경망 번역 속도**: Helsinki-NLP 모델 첫 로딩에 시간 소요 (약 30초)
4. **한글 이름 매핑**: 영어 엔티티 → 한글 역번역이 완벽하지 않을 수 있음

---

## 🔗 관련 파일

- `extraction/neural_translator.py`: Helsinki-NLP 번역기
- `extraction/multilingual_medcat.py`: 다국어 MedCAT2 래퍼
- `extraction/medcat2_adapter.py`: MedCAT2 어댑터
- `extraction/slot_extractor.py`: 슬롯 추출기
- `korean_translator.py`: 사전 기반 번역기
- `test_multilingual.py`: 테스트 스크립트

---

## 📚 참고 자료

- [MedCAT2 Documentation](https://medcat2.readthedocs.io/)
- [Helsinki-NLP opus-mt](https://huggingface.co/Helsinki-NLP)
- [langdetect](https://pypi.org/project/langdetect/)

---

## ✅ 체크리스트

- [x] 사전 기반 번역 구현
- [x] 신경망 번역 구현 (Helsinki-NLP)
- [x] 다국어 MedCAT2 래퍼 구현
- [x] 슬롯 추출기 통합
- [x] 테스트 및 검증
- [ ] 번역 사전 추가 확장
- [ ] 노이즈 필터링 개선
- [ ] LLM 기반 번역 통합 (선택적)

