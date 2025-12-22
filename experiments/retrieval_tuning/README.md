# Retrieval Tuning Experiments

순차전략 기반 검색 파라미터 튜닝 파이프라인입니다.

## 실행 순서

### 0. 평가셋 준비
```bash
python experiments/retrieval_tuning/0_build_vl_evalset.py \
    --config configs/aihub_71874_paths.yaml \
    --out experiments/retrieval_tuning/eval_vl.jsonl \
    --limit 200
```

### 1. TS-only 튜닝
```bash
python experiments/retrieval_tuning/1_run_ts_tuning.py \
    --eval experiments/retrieval_tuning/eval_vl.jsonl \
    --limit 200 \
    --outdir experiments/retrieval_tuning/runs_ts
```

결과: `runs_ts/summary_ts.csv`에서 최적 `topk_coarse_docs`, `topk_ts_final` 확인

### 2. TS+TL Fusion 튜닝
```bash
python experiments/retrieval_tuning/2_run_fusion_tuning.py \
    --eval experiments/retrieval_tuning/eval_vl.jsonl \
    --limit 200 \
    --outdir experiments/retrieval_tuning/runs_fusion \
    --kdoc 30 \
    --kts 8
```

결과: `runs_fusion/summary_fusion.csv`에서 최적 `weight_tl`, `rrf_k` 확인

### 3. 최종 설정 저장
```bash
python experiments/retrieval_tuning/3_pick_best_and_write_yaml.py \
    --ts_summary experiments/retrieval_tuning/runs_ts/summary_ts.csv \
    --fusion_summary experiments/retrieval_tuning/runs_fusion/summary_fusion.csv \
    --out_yaml configs/aihub_retrieval_runtime.yaml
```

## 평가 지표

- **keyword_recall**: 답변의 키워드가 검색된 컨텍스트에 포함된 비율
- **mean_lat_ms**: 평균 검색 지연시간 (밀리초)

## 출력 파일

- `runs_ts/*.jsonl`: 각 설정별 상세 검색 결과
- `runs_ts/summary_ts.csv`: TS 튜닝 요약
- `runs_fusion/*.jsonl`: 각 설정별 상세 검색 결과
- `runs_fusion/summary_fusion.csv`: Fusion 튜닝 요약
- `configs/aihub_retrieval_runtime.yaml`: 최종 최적 설정

