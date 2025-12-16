# MEDCAT2 UMLS RRF 통합 구현 요약

## [클립보드] 구현 완료 항목

### 1. UMLS RRF 직접 처리 스크립트 [완료]
**파일**: `scripts/medcat2_build_from_umls_rrf.py`

**기능**:
- MRCONSO.RRF 파일 직접 읽기
- MRSTY.RRF 파일 직접 읽기
- 언어, 소스, semantic type 필터링
- CDBMaker 사용 (공식 튜토리얼 방식)
- 중간 CSV 저장 옵션

**사용법**:
```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso MRCONSO.RRF \
    --mrsty MRSTY.RRF \
    --output-dir models/medcat2 \
    --sources SNOMEDCT_US ICD10CM \
    --semantic-types "Disease or Syndrome" "Sign or Symptom"
```

### 2. 기존 스크립트 업데이트 [완료]
**파일**: `scripts/medcat2_build_cdb_vocab.py`

**개선 사항**:
- UMLS RRF 파일 직접 읽기 지원 추가
- CDBMaker 사용 옵션 추가
- 공식 튜토리얼 방식 지원

### 3. 통합 문서 [완료]
**파일**: `docs/MEDCAT2_UMLS_RRF_GUIDE.md`

**내용**:
- UMLS RRF 파일 구조 설명
- 사용 방법 및 옵션
- 필터링 가이드
- 문제 해결

## [새로고침] 전체 워크플로우

```
UMLS RRF 파일 (MRCONSO.RRF, MRSTY.RRF)
    [감소]
[방법 1] medcat2_build_from_umls_rrf.py (직접 처리)
    [감소]
[방법 2] medcat2_build_cdb_vocab.py (RRF 지원)
    [감소]
CDB/VCB 생성 (CDBMaker 사용)
    [감소]
Unsupervised Training
    [감소]
모델 팩 생성
```

## [목표] 주요 특징

### 1. 공식 튜토리얼 방식
- CDBMaker 사용 (공식 튜토리얼과 동일)
- Name 전처리 자동화
- 일관된 CDB 구조

### 2. 유연한 필터링
- 언어: ENG, KOR 등
- 소스: SNOMEDCT_US, ICD10CM, RXNORM 등
- Semantic Type: "Disease or Syndrome", "Sign or Symptom" 등
- Preferred Term 선택

### 3. 두 가지 방식 지원
- **방법 1**: RRF 파일 직접 처리 (`medcat2_build_from_umls_rrf.py`)
- **방법 2**: 기존 스크립트 사용 (`medcat2_build_cdb_vocab.py`)

## [차트] UMLS RRF 파일 구조

### MRCONSO.RRF 컬럼
- `CUI`: Concept Unique Identifier
- `LAT`: 언어 (ENG, KOR 등)
- `ISPREF`: Preferred term 여부 (Y/N)
- `STR`: 실제 용어 문자열
- `SAB`: 소스 vocabulary
- `CODE`: 소스 vocabulary 코드

### MRSTY.RRF 컬럼
- `CUI`: Concept Unique Identifier
- `TUI`: Semantic Type ID
- `STY`: Semantic Type 이름

## [수정] 사용 예시

### 예시 1: 기본 사용
```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso MRCONSO.RRF \
    --mrsty MRSTY.RRF \
    --output-dir models/medcat2
```

### 예시 2: 필터링 적용
```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso MRCONSO.RRF \
    --mrsty MRSTY.RRF \
    --output-dir models/medcat2 \
    --languages ENG \
    --sources SNOMEDCT_US ICD10CM \
    --semantic-types "Disease or Syndrome" "Sign or Symptom" \
    --output-csv umls_concepts_for_medcat.csv
```

### 예시 3: 테스트용 (행 수 제한)
```bash
python scripts/medcat2_build_from_umls_rrf.py \
    --mrconso MRCONSO.RRF \
    --mrsty MRSTY.RRF \
    --output-dir models/medcat2 \
    --nrows 10000
```

## [메모] 생성되는 파일

1. **CDB**: `models/medcat2/cdb.dat`
   - Concept Database
   - CUI와 name 매핑

2. **Vocab**: `models/medcat2/vocab.dat`
   - Vocabulary
   - 단어 임베딩 (학습 후 업데이트)

3. **Config**: `models/medcat2/config.json`
   - 설정 파일

4. **CSV** (선택): `umls_concepts_for_medcat.csv`
   - 중간 결과물
   - 재사용 가능

## [완료] 체크리스트

### 구현 완료
- [x] UMLS RRF 파일 읽기
- [x] 필터링 기능
- [x] CDBMaker 통합
- [x] 중간 CSV 저장
- [x] 기존 스크립트 업데이트
- [x] 문서 작성

### 사용자 작업
- [ ] UMLS RRF 파일 다운로드
- [ ] 파일 경로 확인
- [ ] 필터링 옵션 결정
- [ ] CDB/VCB 생성 실행
- [ ] 출력 파일 확인

## [링크] 관련 문서

- [MEDCAT2 UMLS RRF 가이드](MEDCAT2_UMLS_RRF_GUIDE.md)
- [MEDCAT2 튜토리얼 통합 가이드](MEDCAT2_TUTORIAL_INTEGRATION.md)
- [MEDCAT2 구현 요약](MEDCAT2_IMPLEMENTATION_SUMMARY.md)

## [팁] 참고 사항

1. **파일 크기**: UMLS RRF 파일은 매우 큽니다 (수 GB)
2. **메모리**: 충분한 RAM 필요 (최소 8GB 권장)
3. **처리 시간**: 전체 파일 처리 시 수 시간 소요 가능
4. **테스트**: `--nrows` 옵션으로 먼저 테스트 권장

