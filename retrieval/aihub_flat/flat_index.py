from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import orjson


@dataclass
class FlatIndex:
    name: str
    emb: np.ndarray              # (N, D) mmap 가능
    meta: list[dict[str, Any]]   # length N


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Our meta jsonl rows look like:
      {"id": ..., "lid": ..., "meta": {...}}
    We want a single flat dict that includes:
      - id, lid
      - all fields inside meta (if dict)
      - also keep origin_path, doc_id, chunk_id, qa_id, domain_id, q_type, title, ...
    """
    out: dict[str, Any] = {}

    # keep id/lid at top
    if "id" in row:
        out["id"] = row.get("id")
    if "lid" in row:
        out["lid"] = row.get("lid")

    m = row.get("meta")
    if isinstance(m, dict):
        # meta fields become top-level
        out.update(m)
    else:
        # if meta is not dict, keep raw
        out["meta_raw"] = m

    return out


def load_flat_index(
    npy_path: Path,
    meta_path: Path,
    *,
    name: str,
    keep_fields: Optional[Sequence[str]] = None,
    mmap: bool = True,
) -> FlatIndex:
    """
    Load embeddings + meta jsonl.
    - Supports mmap loading for big npy
    - Flattens {"id","lid","meta":{...}} into one dict per row
    - If keep_fields is provided, keeps only those + id/lid (and always keeps origin_path if exists)
    """
    npy_path = Path(npy_path)
    meta_path = Path(meta_path)

    if not npy_path.exists():
        raise FileNotFoundError(f"[{name}] Missing npy: {npy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"[{name}] Missing meta jsonl: {meta_path}")

    emb = np.load(str(npy_path), mmap_mode="r" if mmap else None)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)

    metas: list[dict[str, Any]] = []
    for row in _iter_jsonl(meta_path):
        flat = _flatten_row(row)

        if keep_fields is not None:
            kept: dict[str, Any] = {}
            # always keep id/lid if present
            if "id" in flat:
                kept["id"] = flat.get("id")
            if "lid" in flat:
                kept["lid"] = flat.get("lid")

            # requested keys
            for k in keep_fields:
                if k in flat:
                    kept[k] = flat.get(k)

            # origin_path는 self-hit 필터/디버깅에 매우 유용하므로 있으면 보존
            if "origin_path" in flat and "origin_path" not in kept:
                kept["origin_path"] = flat.get("origin_path")

            flat = kept

        metas.append(flat)

    if emb.shape[0] != len(metas):
        raise RuntimeError(
            f"[{name}] Row mismatch: emb rows={emb.shape[0]} vs meta rows={len(metas)} "
            f"({npy_path.name} / {meta_path.name})"
        )

    return FlatIndex(name=name, emb=emb, meta=metas)
