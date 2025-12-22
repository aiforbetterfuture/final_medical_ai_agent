from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

import numpy as np

from .config import AIHubIndexConfig
from .io import resolve_path
from .flat_index import FlatIndex, load_flat_index
from .cosine import topk_dot_blockscan
from .rules import detect_lang, preferred_source_types


def _meta_pass_lang(meta: dict, lang: str) -> bool:
    if lang == "any":
        return True
    return (meta.get("language") == lang)


def _meta_pass_source(meta: dict, allowed: Optional[List[str]]) -> bool:
    if not allowed:
        return True
    return meta.get("source_type") in set(allowed)


@dataclass
class TwoStageTSRetriever:
    cfg: AIHubIndexConfig
    ts_coarse: FlatIndex
    ts_fine: FlatIndex
    doc2rows: dict[str, list[int]]

    @classmethod
    def build(cls, cfg: AIHubIndexConfig) -> "TwoStageTSRetriever":
        idx = cfg.index_dir

        ts_coarse_npy = resolve_path(idx, cfg.ts_coarse_npy, exts=(".npy",))
        ts_coarse_meta = resolve_path(idx, cfg.ts_coarse_meta, exts=(".jsonl",))
        ts_fine_npy = resolve_path(idx, cfg.ts_fine_npy, exts=(".npy",))
        ts_fine_meta = resolve_path(idx, cfg.ts_fine_meta, exts=(".jsonl",))

        ts_coarse = load_flat_index(
            ts_coarse_npy,
            ts_coarse_meta,
            name="ts_coarse",
            keep_fields=("doc_id", "language", "source_type", "title", "origin_path"),
        )
        ts_fine = load_flat_index(
            ts_fine_npy,
            ts_fine_meta,
            name="ts_fine",
            keep_fields=("doc_id", "chunk_id", "text", "language", "source_type", "title", "origin_path"),
        )

        doc2rows: dict[str, list[int]] = {}
        for i, m in enumerate(ts_fine.meta):
            doc_id = str(m.get("doc_id"))
            doc2rows.setdefault(doc_id, []).append(i)

        return cls(cfg=cfg, ts_coarse=ts_coarse, ts_fine=ts_fine, doc2rows=doc2rows)

    def _coarse_search_with_filter(
        self,
        q_emb: np.ndarray,
        topk: int,
        lang: str,
        allowed_sources: Optional[List[str]],
    ) -> list[str]:
        """
        coarse 전체를 스캔하되, meta 필터 통과만 후보로 사용.
        속도는 다소 떨어질 수 있으나 coarse는 N이 TS_fine보다 작아 감당 가능.
        """
        ids, scores = topk_dot_blockscan(self.ts_coarse.emb, q_emb, k=max(topk * 5, topk), block=self.cfg.block_size)

        doc_ids: list[str] = []
        for i in ids:
            if i < 0:
                continue
            m = self.ts_coarse.meta[int(i)]
            if not _meta_pass_lang(m, lang):
                continue
            if not _meta_pass_source(m, allowed_sources):
                continue
            doc_ids.append(str(m["doc_id"]))
            if len(doc_ids) >= topk:
                break
        return doc_ids

    def retrieve(self, q_emb: np.ndarray, query_text: str = "", topk_coarse_docs: int | None = None, topk_fine: int | None = None) -> list[dict[str, Any]]:
        """
        1) coarse → doc 후보 (룰 기반 필터 + 부족하면 fallback)
        2) fine → 후보 doc에 속한 chunk만으로 topK
        """
        topk_coarse_docs = topk_coarse_docs or self.cfg.topk_coarse_docs
        topk_fine = topk_fine or self.cfg.topk_ts_final

        q_emb = q_emb.astype(np.float32, copy=False)  # 이미 normalize_embeddings=True라 가정 (dot-only)

        # 룰: 언어/출처 타입
        lang = detect_lang(query_text) if query_text else "any"
        pref_sources = preferred_source_types(query_text) if query_text else None

        # 1차: preferred source로 좁혀서 coarse 후보
        doc_ids = self._coarse_search_with_filter(q_emb, topk_coarse_docs, lang=lang, allowed_sources=pref_sources)

        # fallback: 너무 적으면 source 제한 해제
        if len(doc_ids) < max(5, topk_coarse_docs // 3):
            doc_ids = self._coarse_search_with_filter(q_emb, topk_coarse_docs, lang=lang, allowed_sources=None)

        # 후보 fine rows
        cand_rows: list[int] = []
        for doc_id in doc_ids:
            cand_rows.extend(self.doc2rows.get(doc_id, []))

        if not cand_rows:
            return []

        # fine에서 언어/출처 필터도 적용(후보 rows 자체를 줄임)
        if lang != "any" or pref_sources:
            filtered = []
            for r in cand_rows:
                m = self.ts_fine.meta[r]
                if not _meta_pass_lang(m, lang):
                    continue
                if not _meta_pass_source(m, pref_sources):
                    continue
                filtered.append(r)
            cand_rows = filtered

        if not cand_rows:
            return []

        # 후보만 점수 계산
        cand_idx = np.array(cand_rows, dtype=np.int64)
        cand_emb = self.ts_fine.emb[cand_idx].astype(np.float32, copy=False)
        scores = cand_emb @ q_emb  # dot == cosine(정규화 가정)

        if scores.shape[0] <= topk_fine:
            sel = np.argsort(scores)[::-1]
        else:
            sel = np.argpartition(scores, -topk_fine)[-topk_fine:]
            sel = sel[np.argsort(scores[sel])[::-1]]

        out: list[dict[str, Any]] = []
        for j in sel[:topk_fine]:
            row = int(cand_idx[int(j)])
            m = self.ts_fine.meta[row]
            out.append(
                {
                    "source": "ts_fine",
                    "score": float(scores[int(j)]),
                    "doc_id": m.get("doc_id"),
                    "chunk_id": m.get("chunk_id"),
                    "text": m.get("text"),
                    "language": m.get("language"),
                    "source_type": m.get("source_type"),
                    "title": m.get("title"),
                    "origin_path": m.get("origin_path"),
                }
            )
        return out

