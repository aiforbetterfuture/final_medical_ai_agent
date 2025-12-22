from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from retrieval.aihub_flat.config import AIHubIndexConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts_summary", default="experiments/retrieval_tuning/runs_ts/summary_ts.csv")
    ap.add_argument("--fusion_summary", default="experiments/retrieval_tuning/runs_fusion/summary_fusion.csv")
    ap.add_argument("--out_yaml", default="configs/aihub_retrieval_runtime.yaml")
    args = ap.parse_args()

    ts_df = pd.read_csv(args.ts_summary)
    best_ts = ts_df.sort_values(["mean_recall", "mean_lat_ms"], ascending=[False, True]).iloc[0]

    f_df = pd.read_csv(args.fusion_summary)
    best_f = f_df.sort_values(["mean_recall", "mean_lat_ms"], ascending=[False, True]).iloc[0]

    cfg = AIHubIndexConfig.default()
    cfg.topk_coarse_docs = int(best_f["topk_coarse_docs"])
    cfg.topk_ts_final = int(best_f["topk_ts_final"])
    cfg.topk_tl_final = int(best_f["topk_tl_final"])
    cfg.weight_tl = float(best_f["weight_tl"])
    cfg.rrf_k = int(best_f["rrf_k"])

    cfg.save_yaml(Path(args.out_yaml))
    print("Best TS:", dict(best_ts))
    print("Best FUSION:", dict(best_f))
    print("Wrote:", args.out_yaml)

if __name__ == "__main__":
    main()
