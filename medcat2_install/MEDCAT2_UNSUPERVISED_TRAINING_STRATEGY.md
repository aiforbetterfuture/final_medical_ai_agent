# MEDCAT2 비지도 학습 통합 전략

## [클립보드] 튜토리얼 코드 분석

### 핵심 프로세스

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
trained_concepts = [(ci['cui'], cat.cdb.get_name(ci['cui']), ci['count_train']) 
                    for ci in cat.cdb.cui2info.values() if ci['count_train']]
trained_names = [(ni["name"], ni["count_train"]) 
                 for ni in cat.cdb.name2info.values() if ni["count_train"]]

# 5. 모델 저장
cat.save_model_pack("models", pack_name="unsup_trained_model", add_hash_to_pack_name=False)
```

### 주요 특징

1. **모델 팩 기반**: CDB/Vocab/Config 개별 로드 대신 모델 팩에서 직접 로드
2. **간단한 API**: `cat.trainer.train_unsupervised(texts)` 한 줄로 학습
3. **학습 추적**: `count_train` 필드로 학습된 개념/이름 추적
4. **문맥 기반 학습**: 모호한 이름(DM)을 문맥으로 구분

## [목표] 스캐폴드 통합 전략

### 전략 1: 모델 팩 기반 학습 스크립트 개선

**현재 문제점**:
- 기존 스크립트는 CDB/Vocab/Config 개별 로드
- 튜토리얼은 모델 팩 직접 사용

**개선 방안**:
- 모델 팩 기반 로드 우선 지원
- 기존 방식도 호환성 유지

### 전략 2: 한국어 코퍼스 처리

**요구사항**:
- 한국어 의료 텍스트를 영어로 번역하여 학습
- 번역 품질 보장

**해결 방안**:
- `korean_translator` 모듈 활용
- 배치 번역 처리
- 번역 실패 시 원본 텍스트 사용 (폴백)

### 전략 3: 학습 진행 상황 모니터링

**요구사항**:
- 학습 진행률 표시
- 학습된 개념/이름 통계
- 성능 개선 측정

**해결 방안**:
- 진행률 표시 (tqdm 등)
- 학습 전후 비교 리포트
- 샘플 테스트 케이스로 성능 검증

### 전략 4: 자동화된 학습 파이프라인

**요구사항**:
- 코퍼스 자동 탐지 및 로드
- 학습 -> 검증 -> 저장 자동화

**해결 방안**:
- 코퍼스 디렉토리 자동 스캔
- 학습 후 자동 검증
- 버전 관리 (학습 날짜/시간 포함)

## [차트] 구현 계획

### Phase 1: 기본 학습 스크립트 개선

1. 모델 팩 기반 로드 지원
2. 학습 결과 확인 및 리포트
3. 한국어 코퍼스 번역 처리

### Phase 2: 고급 기능 추가

1. 학습 진행률 모니터링
2. 학습 전후 성능 비교
3. 자동 검증 및 리포트

### Phase 3: 통합 및 자동화

1. 학습 파이프라인 자동화
2. 스캐폴드 통합 (SlotsExtractor 연동)
3. 지속적 학습 지원

## [수정] 구현 세부 사항

### 1. 모델 팩 기반 학습

```python
# 모델 팩에서 직접 로드
cat = CAT.load_model_pack("models/medcat2/base_model")

# 학습 실행
cat.trainer.train_unsupervised(texts)

# 학습 결과 확인
trained_concepts = [
    (ci['cui'], cat.cdb.get_name(ci['cui']), ci['count_train']) 
    for ci in cat.cdb.cui2info.values() 
    if ci.get('count_train', 0) > 0
]
```

### 2. 한국어 코퍼스 처리

```python
from nlp.korean_translator import KoreanTranslator

translator = KoreanTranslator()

# 한국어 텍스트 번역
english_texts = []
for korean_text in korean_corpus:
    english_text = translator.translate_to_english(korean_text)
    english_texts.append(english_text)

# 영어 텍스트로 학습
cat.trainer.train_unsupervised(english_texts)
```

### 3. 학습 진행 모니터링

```python
from tqdm import tqdm

# 배치 단위 학습
batch_size = 1000
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    cat.trainer.train_unsupervised(batch)
    
    # 중간 결과 확인
    if i % (batch_size * 10) == 0:
        trained_count = sum(1 for ci in cat.cdb.cui2info.values() 
                           if ci.get('count_train', 0) > 0)
        print(f"학습된 개념 수: {trained_count}")
```

### 4. 학습 결과 리포트

```python
def generate_training_report(cat):
    """학습 결과 리포트 생성"""
    trained_concepts = [
        (ci['cui'], cat.cdb.get_name(ci['cui']), ci['count_train']) 
        for ci in cat.cdb.cui2info.values() 
        if ci.get('count_train', 0) > 0
    ]
    
    trained_names = [
        (ni["name"], ni["count_train"]) 
        for ni in cat.cdb.name2info.values() 
        if ni.get("count_train", 0) > 0
    ]
    
    print(f"학습된 개념: {len(trained_concepts)}개")
    print(f"학습된 이름: {len(trained_names)}개")
    
    return {
        "concepts": trained_concepts,
        "names": trained_names
    }
```

## [실행] 통합 시나리오

### 시나리오 1: 기본 학습

```bash
# 모델 팩 기반 학습
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained
```

### 시나리오 2: 한국어 코퍼스 학습

```bash
# 한국어 코퍼스 자동 번역 후 학습
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --enable-korean-translation \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained_korean
```

### 시나리오 3: 학습 후 자동 검증

```bash
# 학습 -> 검증 -> 리포트
python scripts/medcat2_train_unsupervised.py \
    --model-pack models/medcat2/base_model \
    --corpus-dir data/corpus/train_source \
    --output-dir models/medcat2 \
    --pack-name medcat2_unsupervised_trained \
    --validate-after-training \
    --test-cases data/test_cases.json
```

## [완료] 구현 완료

### 1. 모델 팩 기반 학습 스크립트 [완료]
- **파일**: `scripts/medcat2_train_unsupervised.py`
- **기능**:
  - 모델 팩에서 직접 로드 (`CAT.load_model_pack`)
  - 공식 튜토리얼 방식: `cat.trainer.train_unsupervised(texts)`
  - 학습 전후 상태 비교 및 리포트
  - 한국어 코퍼스 자동 번역 지원 (`--enable-korean-translation`)
  - 배치 단위 학습 (대용량 데이터 처리)

### 2. 학습 결과 확인 [완료]
- 학습된 개념 수 추적 (`count_train`)
- 학습된 이름 수 추적
- 학습 전후 비교 리포트

### 3. 테스트 결과 [완료]
```
[INFO] 학습 전 상태:
  - 학습된 개념: 0개
  - 학습된 이름: 0개

[INFO] 학습 후 상태:
  - 학습된 개념: 1개
    * Asthma (CUI: C0004096, 학습 횟수: 5)
  - 학습된 이름: 1개
    * asthma (학습 횟수: 5)

[INFO] 학습 효과:
  - 새로 학습된 개념: 1개
  - 새로 학습된 이름: 1개
```

## [메모] 다음 단계

1. **대용량 코퍼스 학습**: 전체 코퍼스로 학습 실행
2. **한국어 코퍼스 학습**: `--enable-korean-translation` 옵션 테스트
3. **지도 학습**: Supervised Training 튜토리얼 통합
4. **성능 평가**: 학습 전후 엔티티 추출 정확도 비교

