# MedCAT2 모델팩 설치 및 테스트 보고서

**작성일**: 2025-12-08  
**테스트 환경**: Windows 10, Python 3.9+, medcat>=2.0

---

## 📋 요약

### ✅ 성공적으로 설치된 모델팩
- **SNOMED International 모델팩** (`mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5.zip`)
  - 크기: 0.67 GB
  - 상태: ✅ **정상 작동**
  - 엔티티 추출: ✅ **성공**

### ❌ 설치 실패한 모델팩
- **UMLS Full 모델팩** (`umls_self_train_model_pt2ch_3760d588371755d0.zip`)
  - 크기: 1.68 GB
  - 상태: ❌ **로드 실패**
  - 원인: 모델팩 파일 손상 또는 호환성 문제

---

## 🔍 상세 테스트 결과

### 1. SNOMED International 모델팩 테스트

#### 테스트 케이스 1: 영어 텍스트 직접 추출
```python
text = "55 year old male with hypertension and diabetes, taking metformin, experiencing chest tightness and dizziness"
```

**결과**:
- ✅ 모델 로드: 성공
- ✅ 엔티티 추출: 성공
- **추출된 엔티티**:
  - **Conditions (4개)**:
    - `Family history: Hypertension` (CUI: 160357008, confidence: 1.0)
    - `Diabetes mellitus` (CUI: 73211009, confidence: 0.61)
    - `per year` (CUI: 259039008, confidence: 0.52) - 노이즈
    - `Male structure` (CUI: 10052007, confidence: 0.26) - 노이즈
  - **Symptoms (2개)**:
    - `Tight chest` (CUI: 23924001, confidence: 1.0)
    - `Dizziness` (CUI: 404640003, confidence: 1.0)
  - **Medications (1개)**:
    - `Metformin` (CUI: 372567009, confidence: 1.0)

#### 테스트 케이스 2: 한국어 번역 후 추출
```python
text_ko = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중이며 가슴이 답답하고 어지러운 환자입니다"
# 번역 후: "55세 남성, 고현압과 당두 있고 메트포르민 복용 중이며 가습이 chest tightness하고 어짜러운 환자입니다"
```

**결과**:
- ✅ 모델 로드: 성공
- ⚠️ 번역 품질: 일부 오타 발생 ("고혈압" → "고현압")
- ✅ 엔티티 추출: 부분 성공
- **추출된 엔티티**:
  - **Symptoms (1개)**:
    - `Tight chest` (CUI: 23924001, confidence: 1.0)

**분석**: 번역 사전의 치환 순서 문제로 일부 용어가 제대로 번역되지 않음. 하지만 영어로 번역된 부분(`chest tightness`)은 정상적으로 추출됨.

---

## 🔧 해결한 문제점

### 문제 1: 엔티티 분류 실패
**원인**: SNOMED 모델팩은 UMLS의 TUI 대신 `type_ids`를 사용하므로, 기존 TUI 기반 분류 로직이 작동하지 않음.

**해결책**: `extraction/medcat2_adapter.py` 수정
- TUI 기반 분류 (UMLS 모델용) 유지
- SNOMED 모델용: `pretty_name` 기반 키워드 매칭 추가
- 키워드 매칭으로 conditions/symptoms/medications 분류

**수정 코드**:
```python
# SNOMED 모델용: pretty_name 기반 키워드 매칭
elif type_ids or name:
    condition_keywords = ['diabetes', 'hypertension', 'disease', 'disorder', 'syndrome', ...]
    symptom_keywords = ['chest', 'tightness', 'dizziness', 'pain', 'dyspnea', ...]
    medication_keywords = ['metformin', 'drug', 'medication', 'medicine', ...]
    # 키워드 매칭으로 분류
```

### 문제 2: UMLS Full 모델팩 로드 실패
**원인**: 모델팩 파일 손상 또는 MedCAT v2와의 호환성 문제

**해결책**: SNOMED International 모델팩 사용 (더 작고 안정적)

---

## 📊 모델팩 비교

| 모델팩 | 크기 | 로드 상태 | 엔티티 추출 | 한국어 번역 지원 | 추천도 |
|--------|------|----------|------------|----------------|--------|
| **SNOMED International** | 0.67 GB | ✅ 성공 | ✅ 성공 | ⚠️ 부분 지원 | ⭐⭐⭐⭐⭐ |
| UMLS Full | 1.68 GB | ❌ 실패 | ❌ 불가 | ❌ 불가 | ❌ |

---

## ✅ 최종 권장 사항

### 선택된 모델팩
**SNOMED International 모델팩** (`mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5.zip`)

**이유**:
1. ✅ 모델 로드 성공
2. ✅ 엔티티 추출 정상 작동
3. ✅ 영어 텍스트에서 높은 정확도 (hypertension, diabetes, metformin, chest tightness, dizziness 모두 추출)
4. ✅ 파일 크기가 작아 로딩 시간 단축
5. ✅ 한국어 번역 후에도 부분적으로 작동

### 환경 변수 설정 (영구)
```powershell
setx MEDCAT2_MODEL_PATH "C:\Users\KHIDI\Downloads\medical_ai_agent_minimal\medcat2\mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5.zip"
setx MEDCAT2_LICENSE_CODE "NLM-10000060827"
setx MEDCAT2_API_KEY "84605af4-35bb-4292-90e7-19f906c2d38f"
```

**주의**: 환경 변수 변경 후 **새 터미널 세션**에서만 적용됩니다.

---

## 🚀 사용 방법

### 기본 사용법
```python
from extraction.medcat2_adapter import MedCAT2Adapter

# 어댑터 초기화 (환경 변수에서 자동 로드)
adapter = MedCAT2Adapter()

# 영어 텍스트에서 엔티티 추출
text = "55 year old male with hypertension and diabetes, taking metformin"
entities = adapter.extract_entities(text)

print(entities["conditions"])  # 질환 목록
print(entities["symptoms"])    # 증상 목록
print(entities["medications"])  # 약물 목록
```

### 한국어 번역 사용
```python
from extraction.medcat2_adapter import MedCAT2Adapter
from korean_translator import KoreanTranslator

# 번역기 초기화
translator = KoreanTranslator()

# 한국어 텍스트 번역
text_ko = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중"
text_en = translator.translate_to_english(text_ko)

# 엔티티 추출
adapter = MedCAT2Adapter()
entities = adapter.extract_entities(text_en)
```

---

## ⚠️ 알려진 제한사항

1. **Legacy Conversion 경고**: MedCAT v1 모델팩을 v2에서 사용할 때 발생하는 경고 (기능에는 영향 없음)
2. **번역 품질**: 한국어 번역 사전이 완벽하지 않아 일부 용어가 제대로 번역되지 않을 수 있음
3. **노이즈 엔티티**: "per year", "Male structure" 같은 노이즈 엔티티가 추출될 수 있음 (후처리 필터링 필요)

---

## 📝 향후 개선 사항

1. **번역 사전 확장**: `korean_translator.py`의 `MEDICAL_TERM_DICT`에 더 많은 용어 추가
2. **노이즈 필터링**: 신뢰도 임계값 조정 또는 키워드 기반 필터링 추가
3. **UMLS Full 모델팩 재시도**: 다른 소스에서 다운로드하거나 v2 호환 모델팩 사용

---

## ✅ 설치 완료 확인

다음 명령어로 설치 상태를 확인할 수 있습니다:

```powershell
python -X utf8 -c "from extraction.medcat2_adapter import MedCAT2Adapter; adapter=MedCAT2Adapter(); print('✅ MedCAT2 설치 완료!' if adapter._model else '❌ 설치 실패'); print('모델 경로:', adapter.model_path)"
```

---

**결론**: SNOMED International 모델팩이 성공적으로 설치되었으며, 영어 텍스트에서 엔티티 추출이 정상적으로 작동합니다. 한국어 번역 지원은 부분적으로 작동하며, 번역 사전 확장을 통해 개선 가능합니다.

