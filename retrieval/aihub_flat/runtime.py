# retrieval/aihub_flat/runtime.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Tuple

import yaml

from .config import AIHubIndexConfig
from .fused_retriever import AIHubFusedRetriever

DEFAULT_RUNTIME_YAML = Path("configs/aihub_retrieval_runtime.yaml")

def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def build_retriever_from_runtime_yaml(
    runtime_yaml: str | None = None,
) -> Tuple[AIHubFusedRetriever, Dict[str, Any]]:
    """
    Build retriever using fixed runtime YAML.
    Priority:
      1) explicit runtime_yaml arg
      2) env AIHUB_RETRIEVAL_RUNTIME_YAML
      3) configs/aihub_retrieval_runtime.yaml
    """
    y = runtime_yaml or os.getenv("AIHUB_RETRIEVAL_RUNTIME_YAML") or str(DEFAULT_RUNTIME_YAML)
    ypath = Path(y)

    if not ypath.exists():
        raise FileNotFoundError(f"Runtime YAML not found: {ypath}")

    cfgd = _read_yaml(ypath)

    # 기존 실험 코드에서 쓰던 AIHubIndexConfig가 이미 존재하므로,
    # 여기서는 '필드명이 달라도' 최소한으로 안전하게 세팅합니다.
    # (필드명이 정확히 일치하면 그대로 반영되고,
    #  일부가 다르면 아래 set-attr 구간에서 넘어갑니다.)
    cfg = AIHubIndexConfig.default()

    # --- safest generic setattr mapping ---
    mapping = {
        # indices/backends
        "ts_index": ["ts_index", "ts_name", "ts_index_name", "ts"],
        "tl_index": ["tl_index", "tl_name", "tl_index_name", "tl"],
        "backend": ["backend"],

        # pools/topk
        "kdoc": ["kdoc", "topk_doc", "topk_coarse_docs", "topk_docs"],
        "kts_pool": ["kts_pool", "kts", "topk_ts", "topk_ts_final", "topk_ts_pool"],

        # fusion selection
        "fusion_mode": ["fusion_mode"],
        "out_k": ["out_k", "topk_out", "fused_topk"],
        "tl_quota": ["tl_quota"],
        "ts_quota": ["ts_quota"],
        "quota_strategy": ["quota_strategy"],

        # rrf
        "rrf_k": ["rrf_k"],

        # self-hit
        "no_self_hit_tl": ["no_self_hit_tl", "no_self_hit", "no_self_hit(TL)"],
    }

    def _try_set(key: str, value: Any) -> None:
        for cand in mapping.get(key, []):
            if hasattr(cfg, cand):
                setattr(cfg, cand, value)
                return
        # 못 찾으면 조용히 무시(레포 필드명이 다른 경우 대비)
        return

    for k, v in cfgd.items():
        if k in mapping:
            _try_set(k, v)

    # paths_config는 retriever가 내부에서 index_dir 등을 읽을 때 필요하면 사용
    # (AIHubFusedRetriever.build가 config 파일을 직접 받는 구조라면 여기를 맞춰주면 됩니다.)
    if "paths_config" in cfgd and hasattr(cfg, "paths_config"):
        setattr(cfg, "paths_config", cfgd["paths_config"])

    r = AIHubFusedRetriever.build(cfg)
    return r, cfgd
