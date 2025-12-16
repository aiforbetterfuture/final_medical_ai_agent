# MEDCAT2 빠른 시작 가이드

## [실행] 5분 안에 시작하기

### 1단계: 패키지 설치

```bash
pip install medcat>=2.0
```

또는 프로젝트 전체 의존성 설치:

```bash
pip install -r requirements.txt
```

### 2단계: 모델 팩 다운로드

1. [MedCAT Models 페이지](https://github.com/CogStack/MedCAT/tree/master/models)에서 모델 팩 다운로드
2. 프로젝트 디렉토리에 저장 (예: `resources/medcat2_models/medcat_model_pack.zip`)

### 3단계: 환경 변수 설정

`.env` 파일에 다음 내용 추가:

```env
MEDCAT2_MODEL_PATH=resources/medcat2_models/medcat_model_pack.zip
MEDCAT2_LICENSE_CODE=NLM-10000060827
MEDCAT2_API_KEY=84605af4-35bb-4292-90e7-19f906c2d38f
```

### 4단계: 테스트 실행

```python
from nlp.medcat2_adapter import MedCAT2Adapter

# 어댑터 초기화
adapter = MedCAT2Adapter()

# 텍스트에서 엔티티 추출
text = "당뇨병 환자가 혈압 140/90을 보이고 있습니다."
entities = adapter.extract_entities(text)

print(entities)
```

또는 예제 스크립트 실행:

```bash
python examples/medcat2_usage_example.py
```

## [메모] 기본 사용법

### 간단한 엔티티 추출

```python
from nlp.medcat2_adapter import MedCAT2Adapter

adapter = MedCAT2Adapter()
entities = adapter.extract_entities("당뇨병 환자가 흉통을 호소합니다.")

# 결과:
# {
#     "conditions": [{"name": "당뇨병", "cui": "C0011849", ...}],
#     "symptoms": [{"name": "흉통", "cui": "C0008031", ...}],
#     "labs": [],
#     "vitals": []
# }
```

### SlotsExtractor와 함께 사용

```python
from agent.nodes.slots_extract import SlotsExtractor

# MEDCAT2 활성화
extractor = SlotsExtractor(cfg_paths={}, use_medcat2=True)

# 슬롯 추출
result = extractor.extract("당뇨병 환자가 혈압 140/90을 보이고 있습니다.")
print(result['raw_slots'])
```

## [수정] 문제 해결

### 모델 팩을 찾을 수 없음

**오류**: `FileNotFoundError: MEDCAT2 모델 팩 파일을 찾을 수 없습니다`

**해결**:
1. `.env` 파일의 `MEDCAT2_MODEL_PATH`가 올바른지 확인
2. 파일 경로가 상대 경로인 경우 프로젝트 루트 기준인지 확인
3. 파일이 실제로 존재하는지 확인

### 패키지 설치 오류

**오류**: `ImportError: medcat 패키지가 설치되지 않았습니다`

**해결**:
```bash
pip install medcat>=2.0
```

### 라이센스 오류

**오류**: 라이센스 관련 오류 메시지

**해결**:
1. `.env` 파일에 `MEDCAT2_LICENSE_CODE`와 `MEDCAT2_API_KEY`가 올바르게 설정되었는지 확인
2. 제공받은 라이센스 코드와 API 키가 유효한지 확인

## [참고] 더 알아보기

- [상세 통합 가이드](MEDCAT2_INTEGRATION_GUIDE.md)
- [MEDCAT2 공식 문서](https://medcat2.readthedocs.io/en/latest/main.html)
- [사용 예제 코드](../examples/medcat2_usage_example.py)



















