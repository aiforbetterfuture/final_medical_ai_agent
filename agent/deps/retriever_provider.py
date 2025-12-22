# agent/deps/retriever_provider.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from retrieval.aihub_flat.runtime import build_retriever_from_runtime_yaml
from retrieval.aihub_flat.fused_retriever import AIHubFusedRetriever


@lru_cache(maxsize=1)
def get_retriever(runtime_yaml: Optional[str] = None) -> Tuple[AIHubFusedRetriever, Dict[str, Any]]:
    """
    runtime YAML 기반 retriever를 프로세스당 1회만 생성해서 공유합니다.
    - runtime_yaml 미지정 시: env -> 기본값 순으로 runtime.py가 처리하도록 위임
    """
    r, runtime_cfg = build_retriever_from_runtime_yaml(runtime_yaml)
    return r, runtime_cfg
