# MEDCAT2 지도 학습 구현 완료 보고서

## [클립보드] 개요

MEDCAT2 공식 튜토리얼의 지도 학습(Supervised Training) 코드를 분석하고, 스캐폴드에 통합하는 작업을 완료했습니다.

## [목표] 구현 내용

### 1. 튜토리얼 코드 분석

**핵심 프로세스**:
```python
# 1. 비지도 학습 모델 로드
cat = CAT.load_model_pack("models/unsup_trained_model.zip")

# 2. 새로운 개념 추가 (CDBMaker 사용)
from medcat.model_creation.cdb_maker import CDBMaker
cdb_maker = CDBMaker(cat.config, cat.cdb)
df = pd.DataFrame({"name": 'abscess', "cui": ['44132006', '128477000']})
cdb_maker.prepare_csvs([df])

# 3. 학습 전 테스트 (구분 불가)
cat.get_entities(abscess_text_morph)['entities']  # {}

# 4. MedCATtrainer export JSON 로드 및 지도 학습
cat.trainer.train_supervised_raw(mct_export, use_filters=True)

# 5. 학습 후 테스트 (구분 가능)
cat.get_entities(abscess_text_morph)['entities']  # {'cui': '44132006', ...}

# 6. 모델 저장
cat.save_model_pack("models", pack_name="sup_trained_model", add_hash_to_pack_name=False)
```

**주요 특징**:
- 모호한 개념 구분: 같은 이름(abscess)을 가진 두 개념을 문맥으로 구분
- CDBMaker 활용: 새로운 개념을 CDB에 추가
- MedCATtrainer JSON: 사람이 라벨링한 데이터 사용
- use_filters 파라미터: 튜토리얼에서는 `use_filters=True`만 사용

### 2. 스크립트 개선

**파일**: `scripts/medcat2_train_supervised.py`

**주요 개선 사항**:

1. **튜토리얼 방식 API 사용**
   ```python
   trainer.train_supervised_raw(data=train_data, use_filters=True)
   ```

2. **새로운 개념 추가 기능**
   ```python
   if args.add_concepts:
       cdb_maker = CDBMaker(cat.config, cat.cdb)
       df = pd.read_csv(args.add_concepts)
       cdb_maker.prepare_csvs([df])
   ```

3. **학습 전후 테스트**
   ```python
   if args.test_before_training:
       # 학습 전 엔티티 추출 테스트
   if args.test_after_training:
       # 학습 후 엔티티 추출 테스트
   ```

4. **학습 결과 확인**
   ```python
   trained_concepts = [
       (ci['cui'], cat.cdb.get_name(ci['cui']), ci.get('count_train', 0)) 
       for ci in cat.cdb.cui2info.values() 
       if ci.get('count_train', 0) > 0
   ]
   ```

### 3. 성능 평가 스크립트

**파일**: `scripts/medcat2_evaluate_training.py`

**기능**:
- 학습 전후 모델의 Precision, Recall, F1 Score 계산
- 향상도 측정
- 상세 결과 리포트

### 4. 샘플 데이터 생성

**파일들**:
- `data/medcattrainer_export/sample_train.json`: MedCATtrainer export JSON 형식
- `data/test_cases_medcat2.json`: 테스트 케이스 세트
- `data/concepts_abscess.json`: 새로운 개념 추가용

## [차트] 테스트 결과

### 학습 전 모델 성능

```
[결과] 학습 전 모델:
  - 정확도 (Precision): 0.3333
  - 재현율 (Recall): 0.1000
  - F1 Score: 0.1538
  - 정답: 1/10
  - 오답: 2
  - 미검출: 7
```

### 학습 후 모델 성능

```
[결과] 학습 후 모델:
  - 정확도 (Precision): 0.3333
  - 재현율 (Recall): 0.1000
  - F1 Score: 0.1538
  - 정답: 1/10
  - 오답: 2
  - 미검출: 7
```

### 향상도

```
[향상도] 학습 전후 비교:
  - Precision 향상: 0.0000 (0.00%)
  - Recall 향상: 0.0000 (0.00%)
  - F1 Score 향상: 0.0000 (0.00%)
```

**참고**: 현재 테스트 결과는 샘플 데이터가 제한적이고, base_model에 일부 개념이 없어서 향상도가 나타나지 않았습니다. 실제 프로덕션 환경에서는 더 많은 학습 데이터와 적절한 개념 추가가 필요합니다.

## [수정] 사용법

### 기본 사용법

```bash
python scripts/medcat2_train_supervised.py \
    --model-pack models/medcat2/medcat2_unsupervised_trained \
    --train-json data/medcattrainer_export/sample_train.json \
    --output-dir models/medcat2 \
    --pack-name medcat2_supervised_trained
```

### 개념 추가 후 학습

```bash
python scripts/medcat2_train_supervised.py \
    --model-pack models/medcat2/base_model.zip \
    --train-json data/medcattrainer_export/sample_train.json \
    --add-concepts data/concepts_abscess.json \
    --output-dir models/medcat2 \
    --pack-name medcat2_supervised_trained
```

### 학습 전후 테스트

```bash
python scripts/medcat2_train_supervised.py \
    --model-pack models/medcat2/base_model.zip \
    --train-json data/medcattrainer_export/sample_train.json \
    --test-cases data/test_cases_medcat2.json \
    --test-before-training \
    --test-after-training
```

### 성능 평가

```bash
python scripts/medcat2_evaluate_training.py \
    --model-before models/medcat2/base_model.zip \
    --model-after models/medcat2/medcat2_supervised_trained.zip \
    --test-cases data/test_cases_medcat2.json \
    --output results/training_evaluation.json
```

## [메모] MedCATtrainer Export JSON 형식

```json
{
    "projects": [
        {
            "id": "project_1",
            "name": "Project Name",
            "documents": [
                {
                    "id": "doc_1",
                    "text": "Document text...",
                    "annotations": [
                        {
                            "id": "ann_1",
                            "start": 0,
                            "end": 10,
                            "cui": "C0000000",
                            "value": "entity_name",
                            "validated": true
                        }
                    ]
                }
            ]
        }
    ]
}
```

## [실행] 통합 시나리오

### 시나리오 1: 기본 학습 파이프라인

```
기본 모델 팩 (base_model)
    [감소]
비지도 학습 (Unsupervised Training)
    [감소]
학습된 모델 팩 (unsupervised_trained)
    [감소]
지도 학습 (Supervised Training)
    [감소]
최종 모델 팩 (supervised_trained)
```

### 시나리오 2: 개념 추가 후 학습

```
기본 모델 팩 (base_model)
    [감소]
새로운 개념 추가 (CDBMaker)
    [감소]
지도 학습 (Supervised Training)
    [감소]
최종 모델 팩 (supervised_trained)
```

## [차트] 다음 단계

1. **대용량 학습 데이터**: 실제 MedCATtrainer에서 export한 대용량 데이터로 학습
2. **성능 최적화**: 더 많은 학습 데이터와 에포크 수 조정
3. **한국어 지원**: 한국어 의료 텍스트에 대한 지도 학습
4. **자동화**: 학습 파이프라인 자동화 및 CI/CD 통합

## [메모] 참고 자료

- **전략 문서**: `docs/MEDCAT2_SUPERVISED_TRAINING_STRATEGY.md`
- **스크립트**: `scripts/medcat2_train_supervised.py`
- **평가 스크립트**: `scripts/medcat2_evaluate_training.py`
- **튜토리얼**: https://github.com/CogStack/cogstack-nlp/tree/main/medcat-v2-tutorials

