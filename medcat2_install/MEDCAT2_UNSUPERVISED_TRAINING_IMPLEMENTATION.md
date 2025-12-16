# MEDCAT2 비지도 학습 구현 완료 보고서

## [클립보드] 개요

MEDCAT2 공식 튜토리얼의 비지도 학습(Unsupervised Training) 코드를 분석하고, 스캐폴드에 통합하는 작업을 완료했습니다.

## [목표] 구현 내용

### 1. 튜토리얼 코드 분석

**핵심 프로세스**:
```python
# 1. 모델 팩 로드
cat = CAT.load_model_pack("models/base_model.zip")

# 2. 학습 텍스트 준비
unsup_train_texts = [
    "Diabetes mellitus is a metabolic disorder...",
    "A renowned painter... dermatomyositis..."
]

# 3. 비지도 학습 실행
cat.trainer.train_unsupervised(unsup_train_texts)

# 4. 학습 결과 확인
trained_concepts = [
    (ci['cui'], cat.cdb.get_name(ci['cui']), ci['count_train']) 
    for ci in cat.cdb.cui2info.values() 
    if ci['count_train']
]

# 5. 모델 저장
cat.save_model_pack("models", pack_name="unsup_trained_model", add_hash_to_pack_name=False)
```

**주요 특징**:
- 모델 팩 기반 로드 (CDB/Vocab/Config 개별 로드 불필요)
- 간단한 API: 위치 인자로 텍스트 리스트 전달
- 학습 추적: `count_train` 필드로 학습된 개념/이름 추적
- 문맥 기반 학습: 모호한 이름(DM)을 문맥으로 구분

### 2. 스크립트 개선

**파일**: `scripts/medcat2_train_unsupervised.py`

**주요 개선 사항**:

1. **모델 팩 기반 로드 지원** (공식 튜토리얼 방식)
   ```python
   if args.model_pack:
       cat = CAT.load_model_pack(str(model_pack_path))
   ```

2. **공식 튜토리얼 방식 학습**
   ```python
   trainer.train_unsupervised(texts_for_training)  # 위치 인자
   ```

3. **학습 결과 확인 및 리포트**
   ```python
   trained_concepts = [
       (ci['cui'], cat.cdb.get_name(ci['cui']), ci.get('count_train', 0)) 
       for ci in cat.cdb.cui2info.values() 
       if ci.get('count_train', 0) > 0
   ]
   ```

4. **한국어 코퍼스 번역 지원**
   ```python
   if args.enable_korean_translation:
       translator = KoreanTranslator()
       translated_texts = [translator.translate_to_english(text) for text in original_texts]
   ```

5. **배치 단위 학습** (대용량 데이터 처리)
   ```python
   for i in range(0, len(texts_for_training), args.batch_size):
       batch = texts_for_training[i:i+args.batch_size]
       trainer.train_unsupervised(batch)
   ```

### 3. 사용법

#### 기본 사용법 (모델 팩 기반)
```bash
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained
```

#### 한국어 코퍼스 학습
```bash
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --enable-korean-translation \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained_korean
```

#### 대용량 코퍼스 학습 (배치 처리)
```bash
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained \
    --batch-size 1000 \
    --max-docs 100000
```

### 4. 테스트 결과

**실행 명령**:
```bash
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained \
    --max-docs 10 \
    --show-training-results
```

**결과**:
```
[INFO] 모델 팩 로드 (공식 튜토리얼 방식): models\medcat2\base_model
[SUCCESS] 모델 팩 로드 완료
[INFO] 코퍼스 디렉토리 로드: data/corpus/train_source
[INFO] 총 15021개 문서 발견
[INFO] 문서 수 제한: 15021개 -> 10개
[INFO] 실제 학습 문서 수: 10

[INFO] 학습 전 상태:
  - 학습된 개념: 0개
  - 학습된 이름: 0개

[INFO] Unsupervised Training 시작...
  - 학습 문서 수: 10
  - 병렬 프로세스: 4
[SUCCESS] Unsupervised Training 완료!

[INFO] 학습 후 상태:
  - 학습된 개념: 1개
    * Asthma (CUI: C0004096, 학습 횟수: 5)
  - 학습된 이름: 1개
    * asthma (학습 횟수: 5)

[INFO] 학습 효과:
  - 새로 학습된 개념: 1개
  - 새로 학습된 이름: 1개

[SUCCESS] 모델 팩 저장 완료: models\medcat2\medcat2_unsupervised_trained.zip
```

## [수정] 기술적 세부 사항

### API 시그니처 확인

**공식 튜토리얼 방식**:
```python
cat.trainer.train_unsupervised(unsup_train_texts)  # 위치 인자
```

**주의사항**:
- 키워드 인자(`texts=...`)가 아닌 위치 인자로 전달
- `n_workers` 등의 추가 파라미터는 지원되지 않을 수 있음 (버전에 따라 다름)

### 학습 결과 추적

**CDB에서 학습된 개념 확인**:
```python
trained_concepts = [
    (ci['cui'], cat.cdb.get_name(ci['cui']), ci.get('count_train', 0)) 
    for ci in cat.cdb.cui2info.values() 
    if ci.get('count_train', 0) > 0
]
```

**Vocab에서 학습된 이름 확인**:
```python
trained_names = [
    (ni["name"], ni.get("count_train", 0)) 
    for ni in cat.cdb.name2info.values() 
    if ni.get("count_train", 0) > 0
]
```

### 한국어 코퍼스 처리

**번역 프로세스**:
1. 한국어 텍스트 감지
2. `KoreanTranslator`를 사용하여 영어로 번역
3. 번역된 텍스트로 학습
4. 번역 실패 시 원본 텍스트 사용 (폴백)

## [차트] 통합 전략

### 1. 학습 파이프라인

```
기본 모델 팩 (base_model)
    [감소]
비지도 학습 (Unsupervised Training)
    [감소]
학습된 모델 팩 (unsupervised_trained)
    [감소]
지도 학습 (Supervised Training) [선택]
    [감소]
최종 모델 팩 (final_model)
```

### 2. 스캐폴드 통합

**SlotsExtractor 연동**:
- 학습된 모델 팩을 `MedCAT2Adapter`에서 로드
- 엔티티 추출 성능 향상
- 한국어 질의 처리 지원

**자동화**:
- 코퍼스 업데이트 시 자동 재학습
- 학습 결과 모니터링 및 리포트
- 버전 관리 (학습 날짜/시간 포함)

## [실행] 다음 단계

1. **대용량 코퍼스 학습**
   - 전체 코퍼스(15,021개 문서)로 학습 실행
   - 학습 시간 및 메모리 사용량 모니터링

2. **한국어 코퍼스 학습**
   - `--enable-korean-translation` 옵션으로 한국어 코퍼스 학습
   - 번역 품질 평가 및 개선

3. **지도 학습 통합**
   - Supervised Training 튜토리얼 코드 분석 및 통합
   - MedCATtrainer export JSON 처리

4. **성능 평가**
   - 학습 전후 엔티티 추출 정확도 비교
   - 모호한 이름(DM 등)의 문맥 기반 구분 성능 측정

5. **자동화**
   - 학습 파이프라인 자동화 스크립트
   - 학습 결과 리포트 자동 생성

## [메모] 참고 자료

- **전략 문서**: `docs/MEDCAT2_UNSUPERVISED_TRAINING_STRATEGY.md`
- **스크립트**: `scripts/medcat2_train_unsupervised.py`
- **튜토리얼**: https://github.com/CogStack/cogstack-nlp/tree/main/medcat-v2-tutorials

