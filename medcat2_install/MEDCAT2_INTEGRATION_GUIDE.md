# MEDCAT2 통합 가이드

## [클립보드] 개요

MEDCAT2는 의료 텍스트에서 의학적 개념(Entity)을 추출하고 UMLS/SNOMED-CT와 같은 생의학 온톨로지에 연결하는 도구입니다. 이 가이드는 프로젝트에 MEDCAT2를 통합하는 방법을 설명합니다.

## [이모지] 라이센스 정보

- **LICENSE CODE**: `NLM-10000060827`
- **API KEY**: `84605af4-35bb-4292-90e7-19f906c2d38f`

## [이모지] 설치 방법

### 1. MEDCAT2 패키지 설치

```bash
pip install "medcat>=2.0"
```

또는 `requirements.txt`에 추가:

```
medcat>=2.0
```

### 2. 모델 팩 다운로드

MEDCAT2 모델 팩은 GitHub 저장소의 모델 섹션에서 다운로드할 수 있습니다:
- [MedCAT Models](https://github.com/CogStack/MedCAT/tree/master/models)

다운로드한 모델 팩(zip 파일)을 프로젝트 디렉토리에 저장하세요. 예:
```
resources/
  └── medcat2_models/
      └── medcat_model_pack.zip
```

### 3. 환경 변수 설정

`.env` 파일에 다음 변수들을 추가하세요:

```env
# MEDCAT2 설정
MEDCAT2_MODEL_PATH=resources/medcat2_models/medcat_model_pack.zip
MEDCAT2_LICENSE_CODE=NLM-10000060827
MEDCAT2_API_KEY=84605af4-35bb-4292-90e7-19f906c2d38f
```

## [실행] 기본 사용법

### 간단한 예제

```python
from medcat.cat import CAT

# 모델 팩 로드
cat = CAT.load_model_pack('resources/medcat2_models/medcat_model_pack.zip')

# 텍스트에서 엔티티 추출
text = "My simple document with kidney failure"
entities = cat.get_entities(text)
print(entities)
```

### 프로젝트 통합 사용법

프로젝트에서는 `nlp/medcat2_adapter.py`를 통해 MEDCAT2를 사용할 수 있습니다:

```python
from nlp.medcat2_adapter import MedCAT2Adapter

# 어댑터 초기화
adapter = MedCAT2Adapter()

# 텍스트에서 엔티티 추출
text = "당뇨병 환자가 혈압 140/90을 보이고 있습니다."
entities = adapter.extract_entities(text)

# 결과는 다음 형식으로 반환됩니다:
# {
#     "conditions": [{"name": "당뇨병", "cui": "C0011849", ...}],
#     "symptoms": [...],
#     "labs": [...],
#     "vitals": [...]
# }
```

## [수정] 프로젝트 통합

### 1. SlotsExtractor에 통합

`agent/nodes/slots_extract.py`의 `SlotsExtractor` 클래스에서 MEDCAT2를 사용하려면:

```python
from nlp.medcat2_adapter import MedCAT2Adapter

class SlotsExtractor:
    def __init__(self, cfg_paths: Dict[str, str], use_medcat2: bool = True):
        # ... 기존 코드 ...
        
        # MEDCAT2 어댑터 초기화
        if use_medcat2:
            try:
                self.medcat2 = MedCAT2Adapter()
            except Exception as e:
                print(f"MEDCAT2 초기화 실패: {e}")
                self.medcat2 = None
        else:
            self.medcat2 = None
    
    def extract(self, text: str) -> Dict[str, Any]:
        # MEDCAT2를 사용하여 엔티티 추출
        if self.medcat2:
            medcat_entities = self.medcat2.extract_entities(text)
            # MEDCAT2 결과를 기존 슬롯 형식으로 변환
            # ...
        
        # 기존 정규표현식 기반 추출도 병행
        # ...
```

### 2. 비지도 학습 (Unsupervised Training)

MEDCAT2는 문서에 대한 비지도 학습을 지원합니다:

```python
from medcat.cat import CAT

# 모델 로드
cat = CAT.load_model_pack('path/to/model_pack.zip')

# 데이터 이터레이터 정의 (문서 리스트)
data_iterator = [
    "Patient has diabetes and hypertension.",
    "Kidney failure with elevated creatinine.",
    # ... 더 많은 문서들
]

# 학습 수행
cat.train(data_iterator)

# 학습된 모델 저장
cat.create_model_pack('path/to/saved_model_pack.zip')
```

## [차트] 엔티티 추출 결과 형식

MEDCAT2는 다음과 같은 형식으로 엔티티를 반환합니다:

```python
{
    "entities": [
        {
            "text": "kidney failure",      # 원본 텍스트
            "start": 25,                    # 시작 위치
            "end": 39,                      # 끝 위치
            "cui": "C0022660",              # UMLS CUI
            "type": "Disease or Syndrome",  # 의미 유형
            "icd10": ["N18.9"],            # ICD-10 코드
            "snomed": ["42399005"],        # SNOMED-CT 코드
            "confidence": 0.95              # 신뢰도
        },
        # ... 더 많은 엔티티들
    ]
}
```

프로젝트의 `MedCAT2Adapter`는 이를 프로젝트 내부 슬롯 형식으로 변환합니다:

```python
{
    "conditions": [
        {"name": "신부전", "cui": "C0022660", "confidence": 0.95}
    ],
    "symptoms": [...],
    "labs": [...],
    "vitals": [...]
}
```

## [검색] 고급 사용법

### 1. 특정 엔티티 타입만 추출

```python
# 특정 의미 유형만 필터링
entities = cat.get_entities(text, filters={"type": ["Disease or Syndrome"]})
```

### 2. 신뢰도 임계값 설정

```python
# 신뢰도가 0.8 이상인 엔티티만 추출
entities = cat.get_entities(text, min_confidence=0.8)
```

### 3. 커스텀 어노테이션

```python
# 특정 개념에 대한 커스텀 어노테이션 추가
cat.add_annotation(
    text="diabetes",
    cui="C0011849",
    start=0,
    end=8
)
```

## [이모지] 문제 해결

### 모델 팩 로드 실패

**문제**: `FileNotFoundError` 또는 모델 로드 오류

**해결책**:
1. 모델 팩 파일 경로가 올바른지 확인
2. 파일이 손상되지 않았는지 확인
3. 환경 변수 `MEDCAT2_MODEL_PATH`가 올바르게 설정되었는지 확인

### 라이센스 오류

**문제**: 라이센스 관련 오류 메시지

**해결책**:
1. `.env` 파일에 `MEDCAT2_LICENSE_CODE`와 `MEDCAT2_API_KEY`가 올바르게 설정되었는지 확인
2. 라이센스 코드와 API 키가 유효한지 확인

### 메모리 부족

**문제**: 대용량 문서 처리 시 메모리 부족

**해결책**:
1. 문서를 작은 청크로 나누어 처리
2. 배치 처리 대신 스트리밍 방식 사용

## [참고] 참고 자료

- [MEDCAT2 공식 문서](https://medcat2.readthedocs.io/en/latest/main.html)
- [MedCAT GitHub 저장소](https://github.com/CogStack/MedCAT)
- [MedCAT Tutorials](https://github.com/CogStack/MedCAT/tree/master/tutorials)
- [MedCAT 데모](https://medcatv2.rosalind.kcl.ac.uk)

## [새로고침] 기존 MedCAT 1.x와의 차이점

MEDCAT2는 MedCAT 1.x의 개선된 버전입니다. 주요 차이점:

1. **모델 팩 형식**: 단일 zip 파일로 모델 관리
2. **API 개선**: 더 간단하고 직관적인 API
3. **성능 향상**: 더 빠른 추론 속도
4. **라이센스 관리**: 라이센스 코드 및 API 키 지원

기존 `nlp/medcat_adapter.py`는 MedCAT 1.x용이므로, MEDCAT2를 사용하려면 `nlp/medcat2_adapter.py`를 사용하세요.

## [완료] 체크리스트

통합 완료를 확인하기 위한 체크리스트:

- [ ] MEDCAT2 패키지 설치 완료
- [ ] 모델 팩 다운로드 및 경로 설정
- [ ] `.env` 파일에 라이센스 정보 추가
- [ ] `MedCAT2Adapter` 클래스 테스트
- [ ] `SlotsExtractor`에 MEDCAT2 통합
- [ ] 엔티티 추출 결과 검증
- [ ] 성능 테스트 및 최적화



















