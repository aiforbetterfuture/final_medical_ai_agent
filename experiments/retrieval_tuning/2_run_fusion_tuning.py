from __future__ import annotations
from pathlib import Path
import argparse, json, time
import pandas as pd

from retrieval.aihub_flat.config import AIHubIndexConfig
from retrieval.aihub_flat.fused_retriever import AIHubFusedRetriever
from experiments.retrieval_tuning.metrics import keyword_recall

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
    ap.add_argument("--outdir", default="experiments/retrieval_tuning/runs_fusion")
    ap.add_argument("--kdoc", type=int, default=30, help="TS 튜닝 결과로 고정 권장")
    ap.add_argument("--kts", type=int, default=8, help="TS 튜닝 결과로 고정 권장")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval(Path(args.eval), args.limit)

    # TL은 힌트이므로 topK 작게
    grid_k_tl = [3, 5, 8]
    grid_weight_tl = [0.3, 0.5, 0.6, 0.8]
    grid_rrf_k = [20, 60, 100]

    summary = []

    for ktl in grid_k_tl:
        for wtl in grid_weight_tl:
            for rrfk in grid_rrf_k:
                cfg = AIHubIndexConfig.default()
                cfg.topk_coarse_docs = args.kdoc
                cfg.topk_ts_final = args.kts
                cfg.topk_tl_final = ktl
                cfg.weight_tl = wtl
                cfg.rrf_k = rrfk

                r = AIHubFusedRetriever.build(cfg)

                run_id = f"FUS_kdoc{args.kdoc}_kts{args.kts}_ktl{ktl}_wtl{wtl}_rrfk{rrfk}"
                out_jsonl = outdir / f"{run_id}.jsonl"

                latencies = []
                recalls = []

                with out_jsonl.open("w", encoding="utf-8") as w:
                    for ex in eval_rows:
                        q = ex["question"]
                        a = ex["answer"]
                        dom = ex.get("domain_id")

                        s = time.perf_counter()
                        out = r.retrieve_fused(q, domain_id=dom)
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
                            "cfg": {
                                "topk_coarse_docs": cfg.topk_coarse_docs,
                                "topk_ts_final": cfg.topk_ts_final,
                                "topk_tl_final": cfg.topk_tl_final,
                                "weight_tl": cfg.weight_tl,
                                "rrf_k": cfg.rrf_k,
                            },
                            "ts_context": out.get("ts_context", []),
                            "tl_hints": out.get("tl_hints", []),
                            "use_opts": out.get("use_opts", False),
                        }, ensure_ascii=False) + "\n")

                summary.append({
                    "run_id": run_id,
                    "topk_coarse_docs": cfg.topk_coarse_docs,
                    "topk_ts_final": cfg.topk_ts_final,
                    "topk_tl_final": cfg.topk_tl_final,
                    "weight_tl": cfg.weight_tl,
                    "rrf_k": cfg.rrf_k,
                    "n": len(eval_rows),
                    "mean_recall": sum(recalls) / max(1, len(recalls)),
                    "mean_lat_ms": sum(latencies) / max(1, len(latencies)),
                })
                print(f"[DONE] {run_id} mean_recall={summary[-1]['mean_recall']:.3f} mean_lat_ms={summary[-1]['mean_lat_ms']:.1f}")

    df = pd.DataFrame(summary).sort_values(["mean_recall", "mean_lat_ms"], ascending=[False, True])
    out_csv = outdir / "summary_fusion.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
