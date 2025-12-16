# MEDCAT2 지도 학습 통합 전략

## [클립보드] 튜토리얼 코드 분석

### 핵심 프로세스

```python
# 1. 비지도 학습 모델 로드
cat = CAT.load_model_pack("models/unsup_trained_model.zip")

# 2. 새로운 개념 추가 (CDBMaker 사용)
from medcat.model_creation.cdb_maker import CDBMaker
cdb_maker = CDBMaker(cat.config, cat.cdb)
df = pd.DataFrame({"name": 'abscess', "cui": ['44132006', '128477000']})
cdb_maker.prepare_csvs([df])

# 3. 학습 전 테스트 (구분 불가)
abscess_text_morph = "Histopathology reveals a well-encapsulated abscess..."
abscess_text_disorder = "An abscess is a disorder..."
cat.get_entities(abscess_text_morph)['entities']  # {}
cat.get_entities(abscess_text_disorder)['entities']  # {}

# 4. MedCATtrainer export JSON 로드 및 지도 학습
import json
with open("in_data/MCT_export_abscess.json") as f:
    mct_export = json.load(f)
cat.trainer.train_supervised_raw(mct_export, use_filters=True)

# 5. 학습 후 테스트 (구분 가능)
cat.get_entities(abscess_text_morph)['entities']  # {'cui': '44132006', ...}
cat.get_entities(abscess_text_disorder)['entities']  # {'cui': '128477000', ...}

# 6. 모델 저장
cat.save_model_pack("models", pack_name="sup_trained_model", add_hash_to_pack_name=False)
```

### 주요 특징

1. **모호한 개념 구분**: 같은 이름(abscess)을 가진 두 개념을 문맥으로 구분
2. **CDBMaker 활용**: 새로운 개념을 CDB에 추가
3. **MedCATtrainer JSON**: 사람이 라벨링한 데이터 사용
4. **use_filters 파라미터**: 튜토리얼에서는 `use_filters=True`만 사용

## [목표] 스캐폴드 통합 전략

### 전략 1: 지도 학습 스크립트 개선

**현재 문제점**:
- 기존 스크립트는 `n_epochs`, `use_project_filters` 파라미터 사용
- 튜토리얼은 `use_filters=True`만 사용

**개선 방안**:
- 튜토리얼 방식 우선 지원 (`use_filters=True`)
- 기존 파라미터도 호환성 유지
- 학습 전후 성능 비교 기능 추가

### 전략 2: 샘플 데이터 생성

**요구사항**:
- MedCATtrainer export JSON 형식의 샘플 데이터
- 모호한 개념(abscess 등)을 구분할 수 있는 예제

**해결 방안**:
- 튜토리얼 예제 기반 샘플 JSON 생성
- 의료 엔티티 추출 테스트 케이스 포함

### 전략 3: 성능 평가

**요구사항**:
- 학습 전후 Precision, Recall 계산
- 엔티티 추출 향상도 측정

**해결 방안**:
- 테스트 케이스 세트 준비
- 학습 전후 엔티티 추출 결과 비교
- Precision, Recall, F1 Score 계산

## [차트] 구현 계획

### Phase 1: 스크립트 개선

1. 튜토리얼 방식 API 사용 (`use_filters=True`)
2. 학습 전후 상태 비교
3. 새로운 개념 추가 기능 (CDBMaker)

### Phase 2: 샘플 데이터 및 테스트

1. MedCATtrainer export JSON 샘플 생성
2. 테스트 케이스 준비
3. 성능 평가 스크립트 작성

### Phase 3: 통합 및 자동화

1. 학습 파이프라인 자동화
2. 성능 리포트 자동 생성
3. 스캐폴드 통합

## [수정] 구현 세부 사항

### 1. MedCATtrainer Export JSON 형식

```json
{
    "projects": [
        {
            "id": "project_1",
            "name": "Abscess Disambiguation"
        }
    ],
    "documents": [
        {
            "id": "doc_1",
            "text": "Histopathology reveals a well-encapsulated abscess...",
            "project_id": "project_1"
        }
    ],
    "annotations": [
        {
            "id": "ann_1",
            "document_id": "doc_1",
            "start": 43,
            "end": 50,
            "cui": "44132006",
            "value": "abscess",
            "validated": true
        }
    ]
}
```

### 2. 학습 전후 성능 비교

```python
def evaluate_model(cat, test_cases):
    """모델 성능 평가"""
    results = {
        "before": [],
        "after": []
    }
    
    for text, expected_cui in test_cases:
        entities = cat.get_entities(text)['entities']
        predicted_cui = entities[0]['cui'] if entities else None
        
        results["before"].append({
            "text": text,
            "expected": expected_cui,
            "predicted": predicted_cui,
            "correct": predicted_cui == expected_cui
        })
    
    # Precision, Recall 계산
    precision = sum(1 for r in results["before"] if r["correct"]) / len(results["before"])
    recall = sum(1 for r in results["before"] if r["correct"]) / len(results["before"])
    
    return precision, recall
```

## [실행] 통합 시나리오

### 시나리오 1: 기본 지도 학습

```bash
# 비지도 학습 모델 로드 후 지도 학습
python scripts/medcat2_train_supervised.py \
    --model-pack models/medcat2/medcat2_unsupervised_trained \
    --train-json data/medcattrainer_export/sample_train.json \
    --output-dir models/medcat2 \
    --pack-name medcat2_supervised_trained
```

### 시나리오 2: 학습 전후 성능 비교

```bash
# 학습 전후 성능 평가
python scripts/medcat2_evaluate_training.py \
    --model-before models/medcat2/medcat2_unsupervised_trained \
    --model-after models/medcat2/medcat2_supervised_trained \
    --test-cases data/test_cases.json
```

## [메모] 다음 단계

1. **스크립트 개선**: 튜토리얼 방식으로 업데이트
2. **샘플 데이터**: MedCATtrainer export JSON 생성
3. **성능 평가**: Precision, Recall 계산 스크립트 작성
4. **테스트 실행**: 실제 데이터로 검증

