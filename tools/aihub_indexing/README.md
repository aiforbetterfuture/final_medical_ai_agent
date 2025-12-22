# AIHub 71874 Chunking & Embedding (TS/TL/VL) - VSCode/Windows ready

이 폴더는 **AIHub 전문 의학지식 데이터(71874)**를 대상으로,
- TS(원천)=근거 말뭉치 → **문서대표(coarse) + 근거청크(fine)** 인덱스
- TL(라벨링/트레이닝)=문항 → **MCQ stem/options 분해 + 다중 인덱스**
- VL(라벨링/검증)=평가 → **question만 임베딩(선택), answer는 절대 인덱싱 금지**

까지 “한 번에” 끝내는 배치 파이프라인입니다.

## 0) 권장 환경
- Windows 10/11
- Python 3.10+ (권장 3.11)
- GPU 있으면 빨라집니다(없어도 동작)

## 1) 설치 (가상환경 권장)
```powershell
cd <final_medical_ai_agent 레포 루트>
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install -U pip
python -m pip install -r tools/aihub_indexing/requirements_indexing.txt
```

> **중요 (Windows / Python 3.13 설치 에러 회피)**
> - 기본 `requirements_indexing.txt`에는 `hnswlib`가 **포함되어 있지 않습니다.**
> - `2_embed_index.py`는 기본값 `--backend auto`로 동작하며,
>   - `hnswlib`가 설치되어 있으면 `*_hnsw.bin`(HNSW) 인덱스를 만들고
>   - 없으면 자동으로 `*_flat.npy`(flat embeddings) 인덱스를 만듭니다.
>
> **HNSW 인덱스가 꼭 필요**하면(추천: Windows에서는 Python 3.11 venv):
> ```powershell
> python -m pip install -r tools/aihub_indexing/requirements_hnsw_optional.txt
> ```
>
> **한 방 설치(권장)**: Python 3.11이 설치된 경우
> ```powershell
> powershell -ExecutionPolicy Bypass -File tools/aihub_indexing/windows_setup.ps1
> ```


## 2) 경로 설정 (필수)
`configs/aihub_71874_paths.yaml`를 열고, **본인 PC의 폴더 경로**가 맞는지 확인하세요.

## 3) (추천) ZIP 자동 해제 사용
데이터가 ZIP으로 들어있으면 **한 번만 해제**하고(캐시) 그 이후는 해제된 파일로 읽는 것이 보통 더 빠르고 안정적입니다.
본 파이프라인은 기본적으로 `--auto-extract`를 켜면,
`data/_extracted_cache/` 아래로 ZIP을 풀어두고 재사용합니다.

## 4) 실행 순서
### (1) 경로/파일 개수 점검
```powershell
python tools/aihub_indexing/0_validate_paths.py --config configs/aihub_71874_paths.yaml
```

### (2) 코퍼스 생성 (청킹 포함)
- TS: coarse/fine 생성
- TL: stem/options 분해 + view 생성
- VL: question-only view 생성(선택)
```powershell
python tools/aihub_indexing/1_build_corpus.py --config configs/aihub_71874_paths.yaml --auto-extract
```

### (3) 임베딩 + 벡터 인덱스 생성 (HNSWLIB)
기본 모델: `intfloat/multilingual-e5-base` (multilingual, 비용 0, 안정적)

```powershell
python tools/aihub_indexing/2_embed_index.py --config configs/aihub_71874_paths.yaml --device auto
```

### (4) 스모크 테스트 (검색이 되는지 확인)
```powershell
python tools/aihub_indexing/3_smoke_test.py --config configs/aihub_71874_paths.yaml --query "급성 신우신염 경구 항생제 용량"
```

## 5) 산출물 위치
- processed corpus(jsonl): `data/aihub_71874/processed/...`
- indexes:
  - `data/aihub_71874/indexes/ts_fine_hnsw.bin` 등
  - `data/aihub_71874/indexes/*_meta.jsonl` (id→메타)
  - `data/aihub_71874/indexes/*_texts.jsonl` (id→원문 텍스트; 선택)

## 6) 런타임에서 검색할 때(중요)
E5 계열은 권장 포맷이 있습니다.
- 문서 임베딩: `"passage: ..."`
- 쿼리 임베딩: `"query: ..."`
본 파이프라인은 문서 측에 `"passage:"`를 자동으로 붙입니다.
스캐폴드에서 검색 쿼리 임베딩 시엔 `"query:"`를 붙이세요.

---
