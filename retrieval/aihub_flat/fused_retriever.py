# retrieval/aihub_flat/fused_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import AIHubIndexConfig
from .io import resolve_path
from .flat_index import load_flat_index, FlatIndex
from .two_stage_ts import TwoStageTSRetriever
from .cosine import topk_dot_blockscan
from .rrf import rrf_fuse
from .rules import should_use_tl_opts


@dataclass
class AIHubFusedRetriever:
    """
    - TS: coarse→fine 2단계 (근거 evidence)
    - TL: stem / stem_opts (case pattern hints)
    - 결합: RRF (가중치로 TS>TL 유지)
    """
    cfg: AIHubIndexConfig
    encoder: SentenceTransformer
    ts: TwoStageTSRetriever
    tl_stem: FlatIndex
    tl_stem_opts: FlatIndex

    @classmethod
    def build(cls, cfg: AIHubIndexConfig) -> "AIHubFusedRetriever":
        encoder = SentenceTransformer(cfg.embed_model_name)
        ts = TwoStageTSRetriever.build(cfg)

        idx = cfg.index_dir
        tl_stem_npy = resolve_path(idx, cfg.tl_stem_npy, exts=(".npy",))
        tl_stem_meta = resolve_path(idx, cfg.tl_stem_meta, exts=(".jsonl",))
        tl_stem_opts_npy = resolve_path(idx, cfg.tl_stem_opts_npy, exts=(".npy",))
        tl_stem_opts_meta = resolve_path(idx, cfg.tl_stem_opts_meta, exts=(".jsonl",))

        # IMPORTANT:
        # load_flat_index() now flattens {"meta": {...}} rows,
        # so keep_fields can refer to qa_id/domain_id/q_type directly.
        tl_keep = ("qa_id", "domain", "domain_id", "q_type", "question", "answer", "origin_path", "source")

        tl_stem = load_flat_index(
            tl_stem_npy, tl_stem_meta,
            name="tl_stem",
            keep_fields=tl_keep,
        )
        tl_stem_opts = load_flat_index(
            tl_stem_opts_npy, tl_stem_opts_meta,
            name="tl_stem_opts",
            keep_fields=tl_keep,
        )
        return cls(cfg=cfg, encoder=encoder, ts=ts, tl_stem=tl_stem, tl_stem_opts=tl_stem_opts)

    def embed_query(self, query: str) -> np.ndarray:
        q = f"query: {query}"
        v = self.encoder.encode(q, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def retrieve_ts_only(self, query: str) -> list[dict[str, Any]]:
        q_emb = self.embed_query(query)
        return self.ts.retrieve(q_emb, query_text=query, topk_fine=self.cfg.topk_ts_final)

    def retrieve_tl(
        self,
        query: str,
        *,
        use_opts: bool,
        topk: int,
        domain_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        q_emb = self.embed_query(query)
        idx = self.tl_stem_opts if use_opts else self.tl_stem

        # oversample then filter
        k_pool = max(topk * 5, topk)
        ids, scores = topk_dot_blockscan(idx.emb, q_emb, k=k_pool, block=self.cfg.block_size)

        out: list[dict[str, Any]] = []
        for idx_pos, i in enumerate(ids):
            if i < 0:
                continue
            m = idx.meta[int(i)]

            mid = m.get("domain_id", m.get("domain"))
            if domain_id is not None and mid is not None:
                try:
                    if int(mid) != int(domain_id):
                        continue
                except Exception:
                    pass

            out.append(
                {
                    "source": "tl_stem_opts" if use_opts else "tl_stem",
                    "lid": m.get("lid"),
                    "id": m.get("id"),
                    "score": float(scores[idx_pos]),
                    "qa_id": m.get("qa_id"),
                    "domain_id": mid,
                    "q_type": m.get("q_type"),
                    "question": m.get("question"),
                    "answer": m.get("answer"),
                    "origin_path": m.get("origin_path"),
                }
            )
            if len(out) >= topk:
                break
        return out

    def retrieve_fused(self, query: str, domain_id: Optional[int] = None) -> dict[str, Any]:
        ts_hits = self.retrieve_ts_only(query)

        use_opts = should_use_tl_opts(query)
        tl_hits = self.retrieve_tl(query, use_opts=use_opts, topk=self.cfg.topk_tl_final, domain_id=domain_id)

        def key_fn(item: dict[str, Any]) -> str:
            src = str(item.get("source", ""))
            if src.startswith("ts_"):
                return f"TS::{item.get('doc_id')}::{item.get('chunk_id')}"
            return f"TL::{item.get('qa_id')}"

        fused = rrf_fuse(
            [ts_hits, tl_hits],
            key_fn=key_fn,
            weights=[self.cfg.weight_ts, self.cfg.weight_tl],
            rrf_k=self.cfg.rrf_k,
            topk=(self.cfg.topk_ts_final + self.cfg.topk_tl_final),
        )
        return {
            "fused_ranked": fused,
            "ts_context": ts_hits,
            "tl_hints": tl_hits,
            "use_opts": use_opts,
        }
