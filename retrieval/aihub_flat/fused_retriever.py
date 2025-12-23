from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from .config import AIHubIndexConfig
from .io import resolve_path
from .flat_index import load_flat_index, FlatIndex
from .two_stage_ts import TwoStageTSRetriever
from .cosine import topk_dot_blockscan
from .rrf import rrf_fuse
from .rules import should_use_tl_opts


def _repo_root_from(path: Path) -> Path:
    # .../final_medical_ai_agent/retrieval/aihub_flat/fused_retriever.py
    return path.resolve().parents[2]


def _auto_runtime_yaml() -> Path:
    root = _repo_root_from(Path(__file__))
    return root / "configs" / "aihub_retrieval_runtime.yaml"


def _load_runtime_yaml(runtime_yaml: str | None) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    runtime_yaml:
      - None: 런타임 미사용
      - "auto": configs/aihub_retrieval_runtime.yaml 자동 탐색
      - path str: 지정 경로
    """
    if runtime_yaml is None:
        return None, {}

    p = _auto_runtime_yaml() if str(runtime_yaml).lower() == "auto" else Path(str(runtime_yaml))
    if not p.exists():
        return p, {}

    with p.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        d = {}
    return p, d


def _apply_runtime_to_cfg(cfg: AIHubIndexConfig, rt: Dict[str, Any]) -> None:
    """
    runtime.yaml의 kdoc/kts_pool/rrf_k 등을 cfg의 pool 크기/파라미터로 반영.
    - kdoc: TS coarse 후보 문서 수(= topk_coarse_docs) + TL pool 크기(= topk_tl_final)로도 같이 사용
    - kts_pool: TS pool 크기(= topk_ts_final)
    """
    def _as_int(x, default=0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    kdoc = _as_int(rt.get("kdoc", 0), 0)
    kts_pool = _as_int(rt.get("kts_pool", 0), 0)
    rrf_k = _as_int(rt.get("rrf_k", 0), 0)

    if kdoc > 0:
        # TS coarse 후보 수
        try:
            object.__setattr__(cfg, "topk_coarse_docs", kdoc)
        except Exception:
            cfg.topk_coarse_docs = kdoc

        # TL pool 크기(= 힌트 후보를 몇 개 뽑을지). quota를 쓰려면 여기(pool)가 충분히 커야 함.
        try:
            object.__setattr__(cfg, "topk_tl_final", kdoc)
        except Exception:
            cfg.topk_tl_final = kdoc

    if kts_pool > 0:
        try:
            object.__setattr__(cfg, "topk_ts_final", kts_pool)
        except Exception:
            cfg.topk_ts_final = kts_pool

    if rrf_k > 0:
        try:
            object.__setattr__(cfg, "rrf_k", rrf_k)
        except Exception:
            cfg.rrf_k = rrf_k


def _quota_fuse(
    ts_hits: List[Dict[str, Any]],
    tl_hits: List[Dict[str, Any]],
    *,
    out_k: int,
    tl_quota: int,
    ts_quota: int,
    quota_strategy: str,
    key_fn,
) -> List[Dict[str, Any]]:
    """
    quota 기반 결합:
      - tl_first / ts_first: quota 먼저 채운 후, out_k까지 '남은 후보로 fill' (핵심 수정)
      - interleave: 번갈아 채우되 quota 소진/부족 시 남은 후보로 fill
    """
    out_k = max(0, int(out_k))
    if out_k <= 0:
        return []

    # quota가 out_k보다 커도 out_k를 넘길 수 없음
    tl_quota = max(0, min(int(tl_quota), out_k))
    ts_quota = max(0, min(int(ts_quota), out_k))

    # 둘 다 0이면: 사실상 TL 우선으로 out_k 채움
    if tl_quota == 0 and ts_quota == 0:
        tl_quota = out_k

    used = set()
    fused: List[Dict[str, Any]] = []
    tl_cnt = 0
    ts_cnt = 0

    def _push(seq: List[Dict[str, Any]], limit: Optional[int], src_is_ts: bool) -> None:
        nonlocal tl_cnt, ts_cnt
        if len(fused) >= out_k:
            return
        for item in seq:
            if len(fused) >= out_k:
                break
            k = key_fn(item)
            if k in used:
                continue
            # quota 체크
            if limit is not None:
                if src_is_ts and ts_cnt >= limit:
                    continue
                if (not src_is_ts) and tl_cnt >= limit:
                    continue

            used.add(k)
            fused.append(item)
            if src_is_ts:
                ts_cnt += 1
            else:
                tl_cnt += 1

    strat = (quota_strategy or "interleave").strip().lower()

    if strat == "tl_first":
        # 1) TL quota
        _push(tl_hits, tl_quota, src_is_ts=False)
        # 2) TS quota
        _push(ts_hits, ts_quota, src_is_ts=True)
        # 3) fill to out_k (중요!)
        _push(tl_hits, None, src_is_ts=False)
        _push(ts_hits, None, src_is_ts=True)
        return fused

    if strat == "ts_first":
        _push(ts_hits, ts_quota, src_is_ts=True)
        _push(tl_hits, tl_quota, src_is_ts=False)
        _push(ts_hits, None, src_is_ts=True)
        _push(tl_hits, None, src_is_ts=False)
        return fused

    # interleave(기본)
    i_tl = 0
    i_ts = 0
    while len(fused) < out_k:
        progressed = False

        # TL 한 개
        if tl_cnt < tl_quota and i_tl < len(tl_hits):
            item = tl_hits[i_tl]
            i_tl += 1
            k = key_fn(item)
            if k not in used:
                used.add(k)
                fused.append(item)
                tl_cnt += 1
                progressed = True

        if len(fused) >= out_k:
            break

        # TS 한 개
        if ts_cnt < ts_quota and i_ts < len(ts_hits):
            item = ts_hits[i_ts]
            i_ts += 1
            k = key_fn(item)
            if k not in used:
                used.add(k)
                fused.append(item)
                ts_cnt += 1
                progressed = True

        if not progressed:
            break

    # quota 다 못 채웠거나 중복 등으로 out_k에 못 미치면 남은 것들로 채움
    _push(tl_hits, None, src_is_ts=False)
    _push(ts_hits, None, src_is_ts=True)
    return fused


@dataclass
class AIHubFusedRetriever:
    """
    - TS: coarse→fine 2단계 (근거)
    - TL: stem / stem_opts (케이스 힌트)
    - 결합: RRF 또는 quota
    """
    cfg: AIHubIndexConfig
    encoder: SentenceTransformer
    ts: TwoStageTSRetriever
    tl_stem: FlatIndex
    tl_stem_opts: FlatIndex

    runtime_yaml_path: Optional[Path] = None
    runtime: Dict[str, Any] = None  # runtime.yaml 로드 결과(원문 dict)

    @classmethod
    def build(cls, cfg: AIHubIndexConfig, runtime_yaml: str | None = "auto") -> "AIHubFusedRetriever":
        runtime_path, rt = _load_runtime_yaml(runtime_yaml)
        # runtime이 있으면 cfg에 반영(특히 kdoc/kts_pool)
        if rt:
            _apply_runtime_to_cfg(cfg, rt)

        encoder = SentenceTransformer(cfg.embed_model_name)
        ts = TwoStageTSRetriever.build(cfg)

        idx = cfg.index_dir
        tl_stem_npy = resolve_path(idx, cfg.tl_stem_npy, exts=(".npy",))
        tl_stem_meta = resolve_path(idx, cfg.tl_stem_meta, exts=(".jsonl",))
        tl_stem_opts_npy = resolve_path(idx, cfg.tl_stem_opts_npy, exts=(".npy",))
        tl_stem_opts_meta = resolve_path(idx, cfg.tl_stem_opts_meta, exts=(".jsonl",))

        tl_stem = load_flat_index(
            tl_stem_npy, tl_stem_meta, name="tl_stem",
            keep_fields=("qa_id", "domain", "domain_id", "q_type", "question", "answer", "origin_path"),
        )
        tl_stem_opts = load_flat_index(
            tl_stem_opts_npy, tl_stem_opts_meta, name="tl_stem_opts",
            keep_fields=("qa_id", "domain", "domain_id", "q_type", "question", "answer", "origin_path"),
        )
        return cls(
            cfg=cfg,
            encoder=encoder,
            ts=ts,
            tl_stem=tl_stem,
            tl_stem_opts=tl_stem_opts,
            runtime_yaml_path=runtime_path,
            runtime=rt or {},
        )

    def embed_query(self, query: str) -> np.ndarray:
        q = f"query: {query}"
        v = self.encoder.encode(q, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def retrieve_ts_only(self, query: str) -> list[dict[str, Any]]:
        q_emb = self.embed_query(query)
        # cfg.topk_ts_final = "TS pool"(kts_pool)
        return self.ts.retrieve(q_emb, query_text=query, topk_fine=self.cfg.topk_ts_final)

    def retrieve_tl(
        self,
        query: str,
        *,
        use_opts: bool,
        topk: int,
        domain_id: Optional[int] = None,
        query_origin_path: Optional[str] = None,
        no_self_hit_tl: bool = False,
    ) -> list[dict[str, Any]]:
        q_emb = self.embed_query(query)
        idx = self.tl_stem_opts if use_opts else self.tl_stem

        ids, scores = topk_dot_blockscan(idx.emb, q_emb, k=max(topk * 5, topk), block=self.cfg.block_size)

        out: list[dict[str, Any]] = []
        qop = (str(query_origin_path).strip() if query_origin_path else None)

        for idx_pos, i in enumerate(ids):
            if i < 0:
                continue
            m = idx.meta[int(i)]

            # domain filter
            mid = m.get("domain_id", m.get("domain"))
            if domain_id is not None and mid is not None:
                try:
                    if int(mid) != int(domain_id):
                        continue
                except Exception:
                    pass

            # self-hit filter (origin_path 동일하면 제외)
            if no_self_hit_tl and qop:
                mop = m.get("origin_path")
                if mop and str(mop).strip() == qop:
                    continue

            out.append(
                {
                    "source": "tl_stem_opts" if use_opts else "tl_stem",
                    "score": float(scores[idx_pos]),
                    "qa_id": m.get("qa_id"),
                    "domain_id": mid,
                    "q_type": m.get("q_type"),
                    "question": m.get("question"),
                    "answer": m.get("answer"),
                    "origin_path": m.get("origin_path"),
                    "lid": m.get("lid"),
                    "id": m.get("id"),
                }
            )
            if len(out) >= topk:
                break
        return out

    def retrieve_fused(
        self,
        query: str,
        domain_id: Optional[int] = None,
        *,
        query_origin_path: Optional[str] = None,
    ) -> dict[str, Any]:
        ts_hits = self.retrieve_ts_only(query)

        use_opts = should_use_tl_opts(query)

        # runtime 옵션
        rt = self.runtime or {}
        fusion_mode = (rt.get("fusion_mode") or "rrf").strip().lower()
        out_k = int(rt.get("out_k") or 0)
        tl_quota = int(rt.get("tl_quota") or 0)
        ts_quota = int(rt.get("ts_quota") or 0)
        quota_strategy = (rt.get("quota_strategy") or "interleave").strip().lower()
        no_self_hit_tl = bool(rt.get("no_self_hit_tl") or rt.get("no_self_hit(TL)") or False)

        tl_hits = self.retrieve_tl(
            query,
            use_opts=use_opts,
            topk=self.cfg.topk_tl_final,      # TL pool(=kdoc로 맞춤)
            domain_id=domain_id,
            query_origin_path=query_origin_path,
            no_self_hit_tl=no_self_hit_tl,
        )

        def key_fn(item: dict[str, Any]) -> str:
            src = str(item.get("source", ""))
            if src.startswith("ts_") or src.startswith("ts"):
                return f"TS::{item.get('doc_id')}::{item.get('chunk_id')}"
            return f"TL::{item.get('qa_id')}"

        if fusion_mode == "quota":
            # out_k가 0이면 기본값: tl_pool + ts_pool
            if out_k <= 0:
                out_k = int(self.cfg.topk_tl_final + self.cfg.topk_ts_final)
            fused = _quota_fuse(
                ts_hits,
                tl_hits,
                out_k=out_k,
                tl_quota=tl_quota,
                ts_quota=ts_quota,
                quota_strategy=quota_strategy,
                key_fn=key_fn,
            )
        else:
            # rrf(기본)
            topk = int(self.cfg.topk_ts_final + self.cfg.topk_tl_final)
            fused = rrf_fuse(
                [ts_hits, tl_hits],
                key_fn=key_fn,
                weights=[self.cfg.weight_ts, self.cfg.weight_tl],
                rrf_k=self.cfg.rrf_k,
                topk=topk,
            )

        fused_tl = sum(1 for x in fused if str(x.get("source", "")).startswith("tl_") or str(x.get("source", "")).startswith("tl"))
        fused_ts = len(fused) - fused_tl

        return {
            "fusion_mode": fusion_mode,
            "out_k": out_k,
            "tl_quota": tl_quota,
            "ts_quota": ts_quota,
            "quota_strategy": quota_strategy,
            "no_self_hit_tl": no_self_hit_tl,
            "use_opts": use_opts,
            "fused_ranked": fused,
            "ts_context": ts_hits,
            "tl_hints": tl_hits,
            "fused_counts": {"total": len(fused), "tl": fused_tl, "ts": fused_ts},
        }
