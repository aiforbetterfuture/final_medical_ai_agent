from __future__ import annotations
from pathlib import Path
import argparse, json, time
import pandas as pd

from retrieval.aihub_flat.config import AIHubIndexConfig
from retrieval.aihub_flat.fused_retriever import AIHubFusedRetriever
from experiments.retrieval_tuning.metrics import keyword_recall
from retrieval.aihub_flat.runtime import build_retriever_from_runtime_yaml


def load_eval(path: Path, limit: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="experiments/retrieval_tuning/eval_vl.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--outdir", default="experiments/retrieval_tuning/runs_ts")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval(Path(args.eval), args.limit)

    # 추천 그리드
    grid_topk_coarse = [10, 20, 30, 50]
    grid_topk_ts = [5, 8, 12]

    summary = []

    for kdoc in grid_topk_coarse:
        for kts in grid_topk_ts:
            cfg = AIHubIndexConfig.default()
            cfg.topk_coarse_docs = kdoc
            cfg.topk_ts_final = kts

            # TS-only
            cfg.topk_tl_final = 0
            cfg.weight_tl = 0.0

            r = AIHubFusedRetriever.build(cfg)

            run_id = f"TS_kdoc{kdoc}_kts{kts}"
            out_jsonl = outdir / f"{run_id}.jsonl"

            t0 = time.perf_counter()
            latencies = []
            recalls = []

            with out_jsonl.open("w", encoding="utf-8") as w:
                for ex in eval_rows:
                    q = ex["question"]
                    a = ex["answer"]
                    dom = ex.get("domain_id")

                    s = time.perf_counter()
                    out = r.retrieve_fused(q, domain_id=dom)  # TL 힌트는 비어있음
                    e = time.perf_counter()

                    lat_ms = (e - s) * 1000.0
                    latencies.append(lat_ms)

                    ctx = [x.get("text", "") or "" for x in out.get("ts_context", [])]
                    rec = keyword_recall(a, ctx)
                    recalls.append(rec)

                    w.write(json.dumps({
                        "qa_id": ex.get("qa_id"),
                        "domain_id": dom,
                        "q": q,
                        "lat_ms": lat_ms,
                        "recall": rec,
                        "cfg": {"topk_coarse_docs": kdoc, "topk_ts_final": kts},
                        "ts_context": out.get("ts_context", []),
                    }, ensure_ascii=False) + "\n")

            t1 = time.perf_counter()

            summary.append({
                "run_id": run_id,
                "topk_coarse_docs": kdoc,
                "topk_ts_final": kts,
                "n": len(eval_rows),
                "mean_recall": sum(recalls) / max(1, len(recalls)),
                "mean_lat_ms": sum(latencies) / max(1, len(latencies)),
                "total_sec": (t1 - t0),
            })
            print(f"[DONE] {run_id} mean_recall={summary[-1]['mean_recall']:.3f} mean_lat_ms={summary[-1]['mean_lat_ms']:.1f}")

    df = pd.DataFrame(summary).sort_values(["mean_recall", "mean_lat_ms"], ascending=[False, True])
    out_csv = outdir / "summary_ts.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
