# MEDCAT2 튜토리얼 방식 통합 가이드

## [클립보드] 개요

이 가이드는 **MEDCAT2 튜토리얼 방식**으로 CDB/VCB를 직접 생성하고 학습하여 의학적 질의에서 엔티티를 추출하는 방법을 설명합니다.

기존의 기성 모델 팩 다운로드 방식과 달리, 이 방식은:
- UMLS/SNOMED 용어집에서 직접 CDB/VCB 생성
- 도메인 코퍼스로 Unsupervised Training
- (선택) 라벨 데이터로 Supervised Training
- 한국어 질의 처리 지원

## [이모지] 라이센스 정보

- **UMLS API KEY**: `84605af4-35bb-4292-90e7-19f906c2d38f`
- **UMLS LICENSE CODE**: `NLM-10000060827`

## [이모지] 사전 준비

### 1. 패키지 설치

```bash
pip install medcat>=2.0 spacy pandas
python -m spacy download en_core_web_md
```

### 2. UMLS 데이터 준비

두 가지 방법 중 선택:

#### 방법 A: UMLS CSV 파일 사용 (권장)

UMLS에서 다운로드한 데이터를 CSV 형식으로 변환:

```csv
cui,name,type,synonyms,description
C0011849,Diabetes Mellitus,T047,"diabetes,DM,당뇨병,당뇨","A metabolic disorder..."
C0020538,Hypertension,T047,"high blood pressure,고혈압","Persistently high arterial..."
```

#### 방법 B: UMLS RRF 파일 사용

UMLS RRF 파일 디렉토리를 준비합니다.

## [실행] 단계별 구현

### Step 1: CDB & Vocab 생성

```bash
# 샘플 CSV 생성 (테스트용)
python scripts/medcat2_build_cdb_vocab.py --create-sample

# 실제 UMLS CSV로 CDB/VCB 생성
python scripts/medcat2_build_cdb_vocab.py \
    --umls-csv data/umls/umls_terms.csv \
    --output-dir models/medcat2 \
    --semantic-types T047 T184 T121 T200
```

**출력 파일:**
- `models/medcat2/cdb.dat` - Concept Database
- `models/medcat2/vocab.dat` - Vocabulary
- `models/medcat2/config.json` - Configuration

### Step 2: Unsupervised Training

```bash
python scripts/medcat2_train_unsupervised.py \
    --cdb-path models/medcat2/cdb.dat \
    --corpus-dir data/corpus \
    --output-dir models/medcat2 \
    --pack-name medcat2_umls_symptom_disease \
    --n-workers 4 \
    --max-docs 100000
```

**입력:**
- CDB/Vocab/Config 파일
- 코퍼스 디렉토리 (`data/corpus/` 내의 .txt, .jsonl, .json 파일)

**출력:**
- `models/medcat2/medcat2_umls_symptom_disease.zip` - 학습된 모델 팩

### Step 3: Supervised Training (선택)

MedCATtrainer에서 export한 JSON 파일이 있는 경우:

```bash
python scripts/medcat2_train_supervised.py \
    --model-pack models/medcat2/medcat2_umls_symptom_disease.zip \
    --train-json data/medcattrainer_export/project_train.json \
    --output-dir models/medcat2 \
    --pack-name medcat2_umls_symptom_disease_supervised \
    --n-epochs 2
```

**출력:**
- `models/medcat2/medcat2_umls_symptom_disease_supervised.zip` - 지도 학습된 모델 팩

## [이모지][이모지] 한국어 질의 처리

### 자동 번역 기능

MEDCAT2 어댑터는 한국어 질의를 자동으로 감지하고 번역하여 처리합니다:

```python
from nlp.medcat2_adapter import MedCAT2Adapter

# 어댑터 초기화 (한국어 번역 자동 활성화)
adapter = MedCAT2Adapter(
    model_path="models/medcat2/medcat2_umls_symptom_disease.zip",
    enable_korean_translation=True  # 기본값: True
)

# 한국어 질의에서 엔티티 추출
korean_text = "당뇨병 환자가 혈압 140/90을 보이고 있습니다."
entities = adapter.extract_entities(korean_text)

# 결과:
# {
#     "conditions": [{"name": "당뇨병", "cui": "C0011849", ...}],
#     "symptoms": [],
#     "labs": [],
#     "vitals": [{"name": "SBP", "value": 140, ...}, ...]
# }
```

### 번역 프로세스

1. **한국어 감지**: 텍스트에 한글이 30% 이상 포함되면 한국어로 간주
2. **의료 용어 번역**: 사전 기반으로 의료 용어를 영어로 번역
3. **MEDCAT2 추출**: 번역된 영어 텍스트로 엔티티 추출
4. **한국어 매핑**: 추출된 엔티티를 다시 한국어로 매핑

### 의료 용어 사전 확장

`nlp/korean_translator.py`의 `MEDICAL_TERM_DICT`에 용어를 추가하여 번역 정확도를 높일 수 있습니다:

```python
MEDICAL_TERM_DICT = {
    "당뇨병": "diabetes mellitus",
    "고혈압": "hypertension",
    # ... 추가 용어
}
```

## [수정] 스캐폴드 통합

### SlotsExtractor에 통합

`agent/nodes/slots_extract.py`에서 이미 MEDCAT2를 지원합니다:

```python
from agent.nodes.slots_extract import SlotsExtractor

# MEDCAT2 활성화
extractor = SlotsExtractor(
    cfg_paths={},
    use_medcat2=True  # MEDCAT2 사용
)

# 슬롯 추출 (한국어 질의 자동 처리)
result = extractor.extract("당뇨병 환자가 흉통을 호소합니다.")
print(result['raw_slots'])
```

### 환경 변수 설정

`.env` 파일에 다음을 추가:

```env
# MEDCAT2 설정
MEDCAT2_MODEL_PATH=models/medcat2/medcat2_umls_symptom_disease.zip
MEDCAT2_LICENSE_CODE=NLM-10000060827
MEDCAT2_API_KEY=84605af4-35bb-4292-90e7-19f906c2d38f
```

## [차트] 엔티티 추출 결과 형식

```python
{
    "conditions": [
        {
            "name": "당뇨병",
            "cui": "C0011849",
            "text": "diabetes mellitus",
            "confidence": 0.95,
            "start": 0,
            "end": 15,
            "icd10": ["E11.9"],
            "snomed": ["73211009"]
        }
    ],
    "symptoms": [
        {
            "name": "흉통",
            "cui": "C0008031",
            "negated": False,
            "confidence": 0.88
        }
    ],
    "labs": [
        {
            "name": "A1c",
            "value": 8.1,
            "unit": "%"
        }
    ],
    "vitals": [
        {
            "name": "SBP",
            "value": 140,
            "unit": "mmHg"
        }
    ]
}
```

## [이모지] 문제 해결

### CDB 생성 실패

**문제**: UMLS CSV 파일을 찾을 수 없음

**해결책**:
1. `--create-sample` 옵션으로 샘플 CSV 생성
2. 실제 UMLS 데이터를 CSV 형식으로 변환
3. 파일 경로 확인

### Unsupervised Training 메모리 부족

**문제**: 대용량 코퍼스 처리 시 메모리 부족

**해결책**:
1. `--max-docs` 옵션으로 문서 수 제한
2. `--batch-size` 조정
3. 코퍼스를 작은 파일로 분할

### 한국어 번역 실패

**문제**: 한국어 엔티티가 제대로 추출되지 않음

**해결책**:
1. `nlp/korean_translator.py`의 사전에 용어 추가
2. `enable_korean_translation=False`로 비활성화 후 영어로 직접 처리
3. LLM 기반 번역 활성화 (향후 구현)

## [참고] 참고 자료

- [MEDCAT2 공식 튜토리얼](https://github.com/CogStack/cogstack-nlp/tree/main/medcat-v2-tutorials)
- [UMLS API 문서](https://documentation.uts.nlm.nih.gov/rest/home.html)
- [MedCAT2 문서](https://medcat2.readthedocs.io/)

## [완료] 체크리스트

- [ ] MEDCAT2 패키지 설치 완료
- [ ] UMLS 데이터 준비 (CSV 또는 RRF)
- [ ] CDB/VCB 생성 완료
- [ ] Unsupervised Training 완료
- [ ] (선택) Supervised Training 완료
- [ ] 모델 팩 생성 확인
- [ ] 한국어 번역 테스트
- [ ] SlotsExtractor 통합 테스트
- [ ] 전체 파이프라인 테스트

## [목표] 다음 단계

1. **의료 용어 사전 확장**: `nlp/korean_translator.py`에 더 많은 용어 추가
2. **LLM 기반 번역**: OpenAI/Claude API를 활용한 고급 번역 구현
3. **도메인 특화 학습**: 의료 코퍼스로 추가 학습
4. **성능 평가**: 엔티티 추출 정확도 측정 및 개선

