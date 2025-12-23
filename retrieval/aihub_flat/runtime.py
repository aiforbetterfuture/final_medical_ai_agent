from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml

from .config import AIHubIndexConfig
from .fused_retriever import AIHubFusedRetriever

PathLike = Union[str, Path]


def _repo_root_from(path: Path) -> Path:
    # .../final_medical_ai_agent/retrieval/aihub_flat/runtime.py  -> repo_root
    return path.resolve().parents[2]


def read_runtime_yaml(runtime_yaml: Optional[PathLike] = "auto",
                      *,
                      repo_root: Optional[Path] = None) -> Tuple[dict[str, Any], Optional[Path]]:
    """
    Returns: (runtime_dict, resolved_path_or_None)

    - runtime_yaml="auto" or None: tries <repo_root>/configs/aihub_retrieval_runtime.yaml
    - runtime_yaml as str/Path: resolves relative paths under repo_root
    """
    root = repo_root or _repo_root_from(Path(__file__))

    if runtime_yaml is None or str(runtime_yaml).strip().lower() in ("auto", ""):
        cand = root / "configs" / "aihub_retrieval_runtime.yaml"
    else:
        cand = Path(runtime_yaml)
        if not cand.is_absolute():
            cand = (root / cand)

    cand = cand.resolve()

    if not cand.exists():
        return {}, None

    with cand.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        return {}, cand
    return data, cand


def _set(cfg: AIHubIndexConfig, field: str, value: Any) -> None:
    # AIHubIndexConfig is typically a frozen dataclass -> use object.__setattr__
    if hasattr(cfg, field):
        object.__setattr__(cfg, field, value)


def apply_runtime_to_cfg(cfg: AIHubIndexConfig,
                         runtime: dict[str, Any],
                         *,
                         index_dir_override: Optional[PathLike] = None) -> AIHubIndexConfig:
    """
    Maps runtime-yaml knobs into cfg fields where they exist.
    Anything not representable in cfg stays in retriever.runtime dict.
    """
    if index_dir_override:
        _set(cfg, "index_dir", Path(index_dir_override))

    # Common knobs used in your repo
    if "kdoc" in runtime:
        kdoc = int(runtime["kdoc"])
        _set(cfg, "topk_coarse_docs", kdoc)
        # TL pool is aligned to kdoc (kept large); quota will cut later.
        _set(cfg, "topk_tl_final", kdoc)

    if "kts_pool" in runtime:
        _set(cfg, "topk_ts_final", int(runtime["kts_pool"]))

    if "rrf_k" in runtime:
        _set(cfg, "rrf_k", int(runtime["rrf_k"]))

    # Optional weights (if present in yaml)
    if "weight_ts" in runtime:
        _set(cfg, "weight_ts", float(runtime["weight_ts"]))
    if "weight_tl" in runtime:
        _set(cfg, "weight_tl", float(runtime["weight_tl"]))

    return cfg


def build_retriever_from_runtime_yaml(runtime_yaml: Optional[PathLike] = "auto",
                                     *args: Any,
                                     **kwargs: Any) -> AIHubFusedRetriever:
    """
    SSOT builder (backward compatible):
    - Reads configs/aihub_retrieval_runtime.yaml (or provided path)
    - Applies cfg-representable knobs to AIHubIndexConfig
    - Stores the full runtime dict on retriever.runtime

    Supported kwargs:
      - cfg_base: Optional[AIHubIndexConfig]
      - index_dir_override: Optional[str|Path]
    """
    cfg_base: Optional[AIHubIndexConfig] = kwargs.get("cfg_base", None)
    index_dir_override = kwargs.get("index_dir_override", None)

    runtime, resolved = read_runtime_yaml(runtime_yaml)

    cfg = cfg_base or AIHubIndexConfig.default()
    cfg = apply_runtime_to_cfg(cfg, runtime, index_dir_override=index_dir_override)

    r = AIHubFusedRetriever.build(cfg)
    r.runtime = runtime
    r.runtime_yaml_path = resolved
    return r
