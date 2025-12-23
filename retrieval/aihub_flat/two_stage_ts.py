from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import AIHubIndexConfig
from .io import resolve_path
from .flat_index import load_flat_index, FlatIndex
from .cosine import topk_dot_blockscan


def _meta_view(x: dict[str, Any]) -> dict[str, Any]:
    """
    idx.meta[i]가
      - 이미 평탄화된 dict 일 수도 있고
      - {"id","lid","meta":{...}} 형태의 row 일 수도 있습니다.
    어떤 형태든 안전하게 '필드 딕셔너리'를 돌려줍니다.
    """
    m = x.get("meta")
    if isinstance(m, dict):
        out = dict(m)
        # id/lid 같은 row-level 키도 보존(있으면)
        for k in ("id", "lid"):
            if k in x and k not in out:
                out[k] = x[k]
        return out
    return x


@dataclass
class TwoStageTSRetriever:
    """
    TS 2-stage:
      1) coarse에서 doc 후보(topk_coarse_docs)
      2) fine에서 후보 doc들 chunk 중 topk_fine 반환
    """
    cfg: AIHubIndexConfig
    coarse: FlatIndex
    fine: FlatIndex
    fine_doc2idx: dict[str, np.ndarray]

    @classmethod
    def build(cls, cfg: AIHubIndexConfig) -> "TwoStageTSRetriever":
        idx = cfg.index_dir

        # 기본 네이밍을 config가 들고 있다고 가정(현재 프로젝트 구조와 일치)
        coarse_npy = resolve_path(idx, cfg.ts_coarse_npy, exts=(".npy",))
        coarse_meta = resolve_path(idx, cfg.ts_coarse_meta, exts=(".jsonl",))
        fine_npy = resolve_path(idx, cfg.ts_fine_npy, exts=(".npy",))
        fine_meta = resolve_path(idx, cfg.ts_fine_meta, exts=(".jsonl",))

        coarse = load_flat_index(
            coarse_npy,
            coarse_meta,
            name="ts_coarse",
            keep_fields=("doc_id", "language", "origin_path", "source", "source_type", "title", "id", "lid"),
        )
        fine = load_flat_index(
            fine_npy,
            fine_meta,
            name="ts_fine",
            keep_fields=("doc_id", "chunk_id", "text", "language", "origin_path", "source", "source_type", "title", "id", "lid"),
        )

        # fine에서 doc_id -> row indices 매핑(한 번만 만들고 재사용)
        doc2: dict[str, list[int]] = {}
        for i, raw in enumerate(fine.meta):
            m = _meta_view(raw)
            did = m.get("doc_id")
            if not did:
                continue
            did = str(did)
            doc2.setdefault(did, []).append(i)

        fine_doc2idx: dict[str, np.ndarray] = {
            did: np.asarray(ixs, dtype=np.int64) for did, ixs in doc2.items()
        }

        return cls(cfg=cfg, coarse=coarse, fine=fine, fine_doc2idx=fine_doc2idx)

    def retrieve(
        self,
        q_emb: np.ndarray,
        *,
        query_text: Optional[str] = None,
        topk_fine: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        kdoc = int(getattr(self.cfg, "topk_coarse_docs", 30))
        topk = int(topk_fine if topk_fine is not None else getattr(self.cfg, "topk_ts_final", 4))
        block = int(getattr(self.cfg, "block_size", 200_000))

        # 1) coarse doc 후보
        coarse_ids, _ = topk_dot_blockscan(self.coarse.emb, q_emb, k=kdoc, block=block)

        cand_doc_ids: list[str] = []
        for i in coarse_ids:
            if i < 0:
                continue
            m = _meta_view(self.coarse.meta[int(i)])
            did = m.get("doc_id")
            if did:
                cand_doc_ids.append(str(did))

        # 후보 doc의 fine row indices 모으기
        cand_rows: list[np.ndarray] = []
        for did in cand_doc_ids:
            arr = self.fine_doc2idx.get(did)
            if arr is not None and arr.size > 0:
                cand_rows.append(arr)

        if cand_rows:
            rows = np.unique(np.concatenate(cand_rows))
        else:
            # 극단 케이스: coarse에서 doc_id를 못 얻었으면 fine 전체에서 검색(느림)
            rows = np.arange(self.fine.emb.shape[0], dtype=np.int64)

        # 2) fine에서 topk
        emb = self.fine.emb[rows].astype(np.float32, copy=False)
        sims = emb @ q_emb.astype(np.float32, copy=False)

        k = min(topk, sims.shape[0])
        if k <= 0:
            return []

        if sims.shape[0] <= k:
            local = np.arange(sims.shape[0], dtype=np.int64)
        else:
            local = np.argpartition(sims, -k)[-k:].astype(np.int64)

        local = local[np.argsort(sims[local])[::-1]]
        picked = rows[local]

        out: list[dict[str, Any]] = []
        for ridx in picked:
            raw = self.fine.meta[int(ridx)]
            m = _meta_view(raw)

            out.append(
                {
                    "source": "ts_fine",
                    "score": float(sims[np.where(rows == ridx)[0][0]]) if rows.size < 2_000 else float((self.fine.emb[int(ridx)] @ q_emb).item()),
                    "id": m.get("id"),
                    "lid": m.get("lid"),
                    "doc_id": m.get("doc_id"),
                    "chunk_id": m.get("chunk_id"),
                    "language": m.get("language"),
                    "origin_path": m.get("origin_path"),
                    "source_type": m.get("source_type"),
                    "title": m.get("title"),
                    "text": m.get("text"),
                }
            )

        return out
