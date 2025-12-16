# MEDCAT2 튜토리얼 방식 통합 구현 요약

## [클립보드] 구현 완료 항목

### 1. CDB/VCB 생성 스크립트 [완료]
**파일**: `scripts/medcat2_build_cdb_vocab.py`

**기능**:
- UMLS CSV 파일에서 Concept Database(CDB) 및 Vocabulary(VCB) 생성
- UMLS API 키 및 라이센스 코드 지원
- 의미 유형 필터링 (T047: Disease, T184: Symptom 등)
- 샘플 CSV 생성 기능 (테스트용)

**사용법**:
```bash
python scripts/medcat2_build_cdb_vocab.py \
    --umls-csv data/umls/umls_terms.csv \
    --output-dir models/medcat2 \
    --semantic-types T047 T184 T121 T200
```

### 2. Unsupervised Training 스크립트 [완료]
**파일**: `scripts/medcat2_train_unsupervised.py`

**기능**:
- 도메인 코퍼스에 대한 비지도 학습
- .txt, .jsonl, .json 파일 자동 인식
- 병렬 처리 지원 (n_workers)
- 문서 수 제한 (max_docs)
- 모델 팩 자동 저장

**사용법**:
```bash
python scripts/medcat2_train_unsupervised.py \
    --cdb-path models/medcat2/cdb.dat \
    --corpus-dir data/corpus \
    --output-dir models/medcat2 \
    --n-workers 4 \
    --max-docs 100000
```

### 3. Supervised Training 스크립트 [완료]
**파일**: `scripts/medcat2_train_supervised.py`

**기능**:
- MedCATtrainer export JSON을 이용한 지도 학습
- 에포크 수 조정 가능
- 프로젝트 필터 지원

**사용법**:
```bash
python scripts/medcat2_train_supervised.py \
    --model-pack models/medcat2/medcat2_umls_symptom_disease.zip \
    --train-json data/medcattrainer_export/project_train.json \
    --n-epochs 2
```

### 4. 한국어 번역 모듈 [완료]
**파일**: `nlp/korean_translator.py`

**기능**:
- 한국어 의료 용어를 영어로 번역
- MEDCAT2 엔티티 추출 결과를 한국어로 매핑
- UTF-8 인코딩 처리
- 의료 용어 사전 기반 번역 (확장 가능)

**주요 클래스**:
- `KoreanTranslator`: 한국어-영어 번역 및 엔티티 매핑

### 5. MEDCAT2 어댑터 업데이트 [완료]
**파일**: `nlp/medcat2_adapter.py`

**개선 사항**:
- 한국어 텍스트 자동 감지
- 한국어 번역 기능 통합
- UTF-8 인코딩 처리
- 튜토리얼 방식 모델 팩 지원

**주요 메서드**:
- `_is_korean_text()`: 한국어 텍스트 감지
- `extract_entities()`: 한국어/영어 자동 처리

### 6. 통합 문서 [완료]
**파일**: `docs/MEDCAT2_TUTORIAL_INTEGRATION.md`

**내용**:
- 단계별 구현 가이드
- 한국어 처리 방법
- 스캐폴드 통합 방법
- 문제 해결 가이드

### 7. 테스트 스크립트 [완료]
**파일**: `examples/test_medcat2_tutorial.py`

**기능**:
- 한국어 번역 테스트
- MEDCAT2 엔티티 추출 테스트
- SlotsExtractor 통합 테스트

## [새로고침] 전체 워크플로우

```
1. UMLS 데이터 준비
   [감소]
2. CDB/VCB 생성 (scripts/medcat2_build_cdb_vocab.py)
   [감소]
3. Unsupervised Training (scripts/medcat2_train_unsupervised.py)
   [감소]
4. (선택) Supervised Training (scripts/medcat2_train_supervised.py)
   [감소]
5. 모델 팩 생성 완료
   [감소]
6. 스캐폴드 통합 (SlotsExtractor)
   [감소]
7. 한국어 질의 처리 (자동)
```

## [이모지][이모지] 한국어 처리 전략

### 자동 감지
- 한글 비율이 30% 이상이면 한국어로 간주
- UTF-8 인코딩 자동 처리

### 번역 프로세스
1. **의료 용어 사전 매칭**: `MEDICAL_TERM_DICT`에서 용어 검색
2. **영어 번역**: 매칭된 용어를 영어로 치환
3. **MEDCAT2 추출**: 번역된 텍스트로 엔티티 추출
4. **한국어 매핑**: 추출된 엔티티를 다시 한국어로 매핑

### 확장 가능성
- `nlp/korean_translator.py`의 `MEDICAL_TERM_DICT`에 용어 추가
- LLM 기반 번역 지원 (향후 구현 가능)

## [수정] 스캐폴드 통합

### SlotsExtractor 통합
`agent/nodes/slots_extract.py`에서 이미 MEDCAT2를 지원:

```python
extractor = SlotsExtractor(cfg_paths={}, use_medcat2=True)
result = extractor.extract("당뇨병 환자가 흉통을 호소합니다.")
```

### 환경 변수
`.env` 파일 설정:
```env
MEDCAT2_MODEL_PATH=models/medcat2/medcat2_umls_symptom_disease.zip
MEDCAT2_LICENSE_CODE=NLM-10000060827
MEDCAT2_API_KEY=84605af4-35bb-4292-90e7-19f906c2d38f
```

## [차트] 주요 파일 구조

```
medical_ai_agent_scaffold_v3_lite_upgrade/
├── scripts/
│   ├── medcat2_build_cdb_vocab.py      # CDB/VCB 생성
│   ├── medcat2_train_unsupervised.py  # 비지도 학습
│   └── medcat2_train_supervised.py    # 지도 학습
├── nlp/
│   ├── medcat2_adapter.py             # MEDCAT2 어댑터 (업데이트)
│   └── korean_translator.py           # 한국어 번역 모듈 (신규)
├── docs/
│   ├── MEDCAT2_TUTORIAL_INTEGRATION.md # 통합 가이드
│   └── MEDCAT2_IMPLEMENTATION_SUMMARY.md # 이 문서
└── examples/
    └── test_medcat2_tutorial.py        # 테스트 스크립트
```

## [완료] 체크리스트

### 구현 완료
- [x] CDB/VCB 생성 스크립트
- [x] Unsupervised Training 스크립트
- [x] Supervised Training 스크립트
- [x] 한국어 번역 모듈
- [x] MEDCAT2 어댑터 업데이트
- [x] 통합 문서 작성
- [x] 테스트 스크립트 작성

### 다음 단계 (사용자 작업)
- [ ] UMLS 데이터 준비 (CSV 또는 RRF)
- [ ] CDB/VCB 생성 실행
- [ ] 코퍼스 준비 (data/corpus/)
- [ ] Unsupervised Training 실행
- [ ] 모델 팩 생성 확인
- [ ] 한국어 질의 테스트
- [ ] 전체 파이프라인 테스트

## [목표] 사용 예시

### 1. CDB/VCB 생성
```bash
# 샘플 CSV 생성
python scripts/medcat2_build_cdb_vocab.py --create-sample

# 실제 데이터로 생성
python scripts/medcat2_build_cdb_vocab.py \
    --umls-csv data/umls/umls_terms.csv \
    --output-dir models/medcat2
```

### 2. 학습
```bash
# Unsupervised Training
python scripts/medcat2_train_unsupervised.py \
    --cdb-path models/medcat2/cdb.dat \
    --corpus-dir data/corpus \
    --output-dir models/medcat2
```

### 3. 사용
```python
from nlp.medcat2_adapter import MedCAT2Adapter

adapter = MedCAT2Adapter(
    model_path="models/medcat2/medcat2_umls_symptom_disease.zip"
)

# 한국어 질의 자동 처리
entities = adapter.extract_entities("당뇨병 환자가 혈압 140/90을 보이고 있습니다.")
```

## [메모] 참고 사항

1. **UMLS 라이센스**: UMLS 데이터 사용 시 라이센스 준수 필요
2. **모델 크기**: 학습된 모델 팩은 수백 MB ~ 수 GB 가능
3. **학습 시간**: 코퍼스 크기에 따라 수 시간 ~ 수일 소요 가능
4. **메모리 요구사항**: 대용량 코퍼스 처리 시 충분한 RAM 필요

## [링크] 관련 문서

- [MEDCAT2 튜토리얼 통합 가이드](MEDCAT2_TUTORIAL_INTEGRATION.md)
- [MEDCAT2 빠른 시작](MEDCAT2_QUICK_START.md)
- [MEDCAT2 통합 가이드](MEDCAT2_INTEGRATION_GUIDE.md)

