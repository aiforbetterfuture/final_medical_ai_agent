# MEDCAT2 UMLS RRF 파일 처리 가이드

## [클립보드] 개요

이 가이드는 UMLS RRF 파일(MRCONSO.RRF, MRSTY.RRF)을 직접 읽어서 MEDCAT2의 CDB/VCB를 생성하는 방법을 설명합니다.

공식 튜토리얼 코드와 ChatGPT 분석 결과를 반영하여 구현되었습니다.

## [이모지] UMLS RRF 파일 구조

### MRCONSO.RRF
UMLS 개념 용어 파일. 주요 컬럼:
- `CUI`: Concept Unique Identifier
- `LAT`: 언어 (ENG, KOR 등)
- `ISPREF`: Preferred term 여부 (Y/N)
- `STR`: 실제 용어 문자열
- `SAB`: 소스 vocabulary (SNOMEDCT_US, ICD10CM 등)
- `CODE`: 소스 vocabulary의 코드

### MRSTY.RRF
UMLS 의미 유형 파일. 주요 컬럼:
- `CUI`: Concept Unique Identifier
- `TUI`: Semantic Type ID (예: T047)
- `STY`: Semantic Type 이름 (예: "Disease or Syndrome")

## [실행] 사용 방법

### 방법 1: RRF 파일에서 직접 생성 (권장)

```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso /path/to/MRCONSO.RRF \
    --mrsty /path/to/MRSTY.RRF \
    --output-dir models/medcat2 \
    --sources SNOMEDCT_US ICD10CM \
    --semantic-types "Disease or Syndrome" "Sign or Symptom"
```

### 방법 2: 기존 스크립트 사용 (RRF 지원)

```bash
python scripts/medcat2_build_cdb_vocab.py \
    --umls-rrf-dir /path/to/umls_rrf_files \
    --output-dir models/medcat2 \
    --semantic-types T047 T184
```

## [차트] 필터링 옵션

### 언어 필터링
```bash
--languages ENG KOR  # 영어와 한국어만
```

### 소스 Vocabulary 필터링
```bash
--sources SNOMEDCT_US ICD10CM RXNORM  # 특정 vocabulary만
```

### Semantic Type 필터링
```bash
--semantic-types "Disease or Syndrome" "Sign or Symptom" "Pharmacologic Substance"
```

또는 TUI 코드로:
```bash
--semantic-types T047 T184 T121  # Disease, Symptom, Drug
```

### Preferred Term만 사용
기본적으로 Preferred term만 사용합니다. 모든 용어를 포함하려면:
```bash
--no-preferred-only
```

## [수정] CDBMaker 사용 (공식 튜토리얼 방식)

기본적으로 CDBMaker를 사용합니다 (공식 튜토리얼 방식):

```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso MRCONSO.RRF \
    --mrsty MRSTY.RRF \
    --use-cdbmaker  # 기본값: True
```

CDBMaker를 사용하지 않으려면:
```bash
--no-use-cdbmaker  # 간단한 방식 사용
```

## [메모] 중간 CSV 저장

RRF -> CSV 변환 후 CSV를 저장할 수 있습니다:

```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso MRCONSO.RRF \
    --mrsty MRSTY.RRF \
    --output-csv umls_concepts_for_medcat.csv \
    --output-dir models/medcat2
```

생성된 CSV는 다음 형식입니다:
```csv
CUI,name,type_id,semantic_type,SAB,CODE
C0011849,Diabetes Mellitus,T047,Disease or Syndrome,SNOMEDCT_US,73211009
C0020538,Hypertension,T047,Disease or Syndrome,SNOMEDCT_US,38341003
```

## [이모지] 테스트용 제한

대용량 파일 테스트 시 행 수 제한:

```bash
--nrows 10000  # 처음 10,000행만 처리
```

## [차트] 출력 파일

생성되는 파일:
- `models/medcat2/cdb.dat` - Concept Database
- `models/medcat2/vocab.dat` - Vocabulary
- `models/medcat2/config.json` - Configuration

## [새로고침] 전체 워크플로우

```
1. UMLS RRF 파일 다운로드
   [감소]
2. RRF 파일에서 CDB/VCB 생성
   python scripts/medcat2_build_from_umls_rrf.py --mrconso MRCONSO.RRF --mrsty MRSTY.RRF
   [감소]
3. Unsupervised Training
   python scripts/medcat2_train_unsupervised.py --cdb-path models/medcat2/cdb.dat
   [감소]
4. (선택) Supervised Training
   python scripts/medcat2_train_supervised.py --model-pack models/medcat2/medcat2_umls_symptom_disease.zip
   [감소]
5. 모델 팩 사용
```

## [팁] 주요 특징

### 1. CDBMaker 사용 (공식 튜토리얼)
- 공식 튜토리얼과 동일한 방식
- Name 전처리 자동화 (공백을 ~로 변환)
- 일관된 CDB 구조

### 2. 유연한 필터링
- 언어, 소스, semantic type 필터링
- Preferred term 선택
- 테스트용 행 수 제한

### 3. 중간 CSV 저장
- RRF -> CSV 변환 결과 저장
- 재사용 가능
- 다른 도구와 호환

## [이모지] 문제 해결

### 메모리 부족
**문제**: 대용량 RRF 파일 처리 시 메모리 부족

**해결책**:
1. `--nrows` 옵션으로 테스트
2. 소스 vocabulary 필터링으로 데이터 축소
3. Semantic type 필터링으로 데이터 축소

### 파일 인코딩 오류
**문제**: RRF 파일 읽기 오류

**해결책**:
1. 파일이 손상되지 않았는지 확인
2. 파일 경로 확인 (절대 경로 권장)
3. 파일 권한 확인

### CDBMaker Import 오류
**문제**: `ImportError: cannot import name 'CDBMaker'`

**해결책**:
1. `medcat>=2.0` 버전 확인
2. `--no-use-cdbmaker` 옵션 사용 (간단한 방식)

## [참고] 참고 자료

- [UMLS RRF 파일 형식](https://www.ncbi.nlm.nih.gov/books/NBK9685/)
- [MEDCAT2 공식 튜토리얼](https://github.com/CogStack/cogstack-nlp/tree/main/medcat-v2-tutorials)
- [ChatGPT 분석 결과](#) (사용자 제공)

## [완료] 체크리스트

- [ ] UMLS RRF 파일 다운로드 완료
- [ ] MRCONSO.RRF 파일 경로 확인
- [ ] MRSTY.RRF 파일 경로 확인
- [ ] 필터링 옵션 결정 (언어, 소스, semantic type)
- [ ] CDB/VCB 생성 실행
- [ ] 출력 파일 확인
- [ ] 다음 단계 (Unsupervised Training) 준비

