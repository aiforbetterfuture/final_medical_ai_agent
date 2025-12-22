from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import orjson
import yaml
from sentence_transformers import SentenceTransformer


# -------------------------
# robust IO
# -------------------------
def _read_text_robust(p: Path) -> str:
    b = p.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("utf-8-sig", errors="replace")


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(_read_text_robust(path))


def load_json_robust(path: Path) -> Any:
    b = path.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
    try:
        return orjson.loads(b)
    except orjson.JSONDecodeError:
        return json.loads(b.decode("utf-8-sig"))


def load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with meta_path.open("rb") as f:
        for line in f:
            if line.strip():
                metas.append(orjson.loads(line))
    return metas


def norm_path(p: Any) -> str:
    s = str(p).strip()
    if not s:
        return ""
    s = s.replace("/", "\\")
    s = os.path.normpath(s)
    s = os.path.normcase(s)
    return s


# -------------------------
# cosine topk by full scan (fast; memmap OK)
# -------------------------
def topk_cosine_scan(emb_mmap: np.ndarray, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # embeddings are normalized, so dot == cosine
    k = min(int(k), int(emb_mmap.shape[0]))
    q = q.astype(np.float32, copy=False).reshape(-1)
    scores = emb_mmap.dot(q)  # (N,)

    if scores.shape[0] <= k:
        idx = np.argsort(scores)[::-1]
    else:
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

    return idx.astype(np.int64), scores[idx].astype(np.float32, copy=False)


# -------------------------
# TS support: cheap heuristic
# -------------------------
def _normalize_for_contains(s: str) -> str:
    s = s.lower()
    drop = " \t\r\n\"'`~!@#$%^&*()-_=+[]{}|\\;:,./<>?"
    for ch in drop:
        s = s.replace(ch, "")
    return s


def _tokenize_simple(s: str) -> List[str]:
    toks = [t.strip().lower() for t in s.split() if t.strip()]
    return [t for t in toks if len(t) >= 2]


def is_supported_by_context(answer: str, ctx: str, min_overlap: int = 3, overlap_ratio: float = 0.20) -> bool:
    if not answer or not ctx:
        return False

    a_norm = _normalize_for_contains(answer)
    c_norm = _normalize_for_contains(ctx)

    if len(a_norm) >= 6 and a_norm in c_norm:
        return True

    a_toks = set(_tokenize_simple(answer))
    c_toks = set(_tokenize_simple(ctx))
    if not a_toks or not c_toks:
        return False

    inter = a_toks.intersection(c_toks)
    if len(inter) >= min_overlap:
        ratio = len(inter) / max(1, len(a_toks))
        return ratio >= overlap_ratio

    return False


def extract_text_from_obj(obj: Any, max_chars: int) -> str:
    if obj is None:
        return ""

    if isinstance(obj, dict):
        for k in ("text", "content", "contents", "body", "passage", "article", "document", "answer"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:max_chars]

        for k in ("paragraphs", "sentences", "chunks", "sections", "data"):
            v = obj.get(k)
            if isinstance(v, list) and v:
                parts: List[str] = []
                total = 0
                for it in v[:60]:
                    s = ""
                    if isinstance(it, str) and it.strip():
                        s = it.strip()
                    elif isinstance(it, dict):
                        for kk in ("text", "content", "sentence", "passage", "chunk", "answer"):
                            vv = it.get(kk)
                            if isinstance(vv, str) and vv.strip():
                                s = vv.strip()
                                break
                    if s:
                        parts.append(s)
                        total += len(s) + 1
                    if total >= max_chars:
                        break
                out = " ".join(parts).strip()
                return out[:max_chars]
        return ""

    if isinstance(obj, list) and obj:
        parts: List[str] = []
        total = 0
        for it in obj[:60]:
            s = extract_text_from_obj(it, max_chars=max_chars)
            if s:
                parts.append(s)
                total += len(s) + 1
            if total >= max_chars:
                break
        out = " ".join(parts).strip()
        return out[:max_chars]

    return ""


def get_origin_text(meta: Dict[str, Any], cache: Dict[str, str], max_chars: int) -> str:
    origin = meta.get("origin_path")
    if not origin:
        return ""
    p = Path(str(origin))
    if not p.exists():
        return ""

    key = norm_path(str(p))
    if key in cache:
        return cache[key][:max_chars]

    try:
        obj = load_json_robust(p)
        txt = extract_text_from_obj(obj, max_chars=max_chars).strip()
        if txt:
            cache[key] = txt
            return txt[:max_chars]
    except Exception:
        return ""
    return ""


def read_evalset(eval_path: Path, limit: int) -> List[Dict[str, Any]]:
    if not eval_path.exists():
        raise SystemExit(f"Evalset not found: {eval_path}")
    rows: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def make_prefix_any(flags: List[bool]) -> List[bool]:
    # prefix_any[n] = any(flags[:n])
    out = [False]
    cur = False
    for b in flags:
        cur = cur or bool(b)
        out.append(cur)
    return out


@dataclass
class SweepRow:
    out_k: int
    ts_quota: int
    tl_quota: int
    kts_pool: int
    kdoc_pool: int
    ts_support_rate: float
    tl_domqt_rate: float
    fused_any_rate: float
    fused_domqt_rate: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aihub_71874_paths.yaml")
    ap.add_argument("--evalset", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--limit", type=int, default=200)

    ap.add_argument("--ts_index", default="ts_coarse", choices=["ts_fine", "ts_coarse"])
    ap.add_argument("--tl_index", default="tl_coarse", choices=["tl_stem", "tl_stem_opts", "tl_q_full", "tl_coarse"])

    ap.add_argument("--kts_pool", type=int, default=4, help="TS 후보 topK(지원률 계산용/TS quota 상한보다 크면 좋음)")
    ap.add_argument("--kdoc_pool", type=int, default=30, help="TL 후보 topK(최소 max(out_k) 이상 권장)")

    ap.add_argument("--out_ks", default="20,30,40,60")
    ap.add_argument("--ts_quotas", default="0,1,2")

    ap.add_argument("--no_self_hit", action="store_true", default=True)
    ap.add_argument("--query_prefix", default="query: ")

    ap.add_argument("--max_ctx_chars", type=int, default=1200)
    ap.add_argument("--min_overlap", type=int, default=3)
    ap.add_argument("--overlap_ratio", type=float, default=0.20)

    ap.add_argument("--csv_out", default="experiments/retrieval_tuning/sweep_summary.csv")
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    out_ks = [int(x.strip()) for x in args.out_ks.split(",") if x.strip()]
    ts_quotas = [int(x.strip()) for x in args.ts_quotas.split(",") if x.strip()]
    max_outk = max(out_ks) if out_ks else 20
    max_tsq = max(ts_quotas) if ts_quotas else 0

    # out_k=60을 의미있게 보려면 TL 후보가 최소 60 필요(SELF-HIT 1개 제거 대비로 +5)
    kdoc_pool = max(int(args.kdoc_pool), int(max_outk))
    kdoc_raw = kdoc_pool + 5
    kts_pool = max(int(args.kts_pool), int(max_tsq), 1)

    cfg = load_yaml(Path(args.config))
    idx_dir = Path(cfg["output"]["index_dir"])
    model_name = cfg["embedding"]["model_name"]

    ts_vec = np.load(str(idx_dir / f"{args.ts_index}_flat.npy"), mmap_mode="r")
    tl_vec = np.load(str(idx_dir / f"{args.tl_index}_flat.npy"), mmap_mode="r")

    ts_meta = load_meta(idx_dir / f"{args.ts_index}_meta.jsonl")
    tl_meta = load_meta(idx_dir / f"{args.tl_index}_meta.jsonl")

    rows = read_evalset(Path(args.evalset), limit=int(args.limit) if args.limit else 0)
    if not rows:
        raise SystemExit("Empty evalset.")

    # encode all queries in batch (one-time)
    model = SentenceTransformer(model_name)
    queries = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        if args.query_prefix:
            q = f"{args.query_prefix}{q}"
        queries.append(q)
    qvecs = model.encode(queries, normalize_embeddings=True, batch_size=32).astype(np.float32)

    # one-time retrieval + per-query prefix flags
    ts_text_cache: Dict[str, str] = {}

    ts_prefix_support: List[List[bool]] = []
    tl_prefix_domqt: List[List[bool]] = []

    # also baseline rates
    baseline_tl_domqt_hits = 0
    baseline_ts_support_hits = 0

    t0 = time.time()
    for i, r in enumerate(rows):
        qv = qvecs[i]
        answer = str(r.get("answer", "")).strip()
        domain_id = str(r.get("domain_id", "")).strip()
        q_type = str(r.get("q_type", "")).strip()
        eval_origin = norm_path(r.get("origin_path"))

        # TS topK
        ts_ids, _ = topk_cosine_scan(ts_vec, qv, k=kts_pool)
        ts_ranked = ts_ids.tolist()

        # TL topK (raw larger, to survive self-hit drop)
        tl_ids_raw, _ = topk_cosine_scan(tl_vec, qv, k=kdoc_raw)
        tl_ranked_raw = tl_ids_raw.tolist()

        # TL self-hit filter -> keep kdoc_pool
        tl_ranked: List[int] = []
        if args.no_self_hit and eval_origin:
            for lid in tl_ranked_raw:
                meta = tl_meta[lid].get("meta", {}) if isinstance(tl_meta[lid].get("meta"), dict) else {}
                if norm_path(meta.get("origin_path")) == eval_origin:
                    continue
                tl_ranked.append(lid)
                if len(tl_ranked) >= kdoc_pool:
                    break
        else:
            tl_ranked = tl_ranked_raw[:kdoc_pool]

        # TL dom+qtype flags + prefix
        domqt_flags: List[bool] = []
        for lid in tl_ranked:
            meta = tl_meta[lid].get("meta", {}) if isinstance(tl_meta[lid].get("meta"), dict) else {}
            got_dom = str(meta.get("domain_id", "")).strip()
            got_qt = str(meta.get("q_type", "")).strip()
            domqt_flags.append(bool(domain_id and q_type and got_dom == domain_id and got_qt == q_type))
        tl_pref = make_prefix_any(domqt_flags)
        tl_prefix_domqt.append(tl_pref)
        if tl_pref[min(len(domqt_flags), kdoc_pool)]:
            baseline_tl_domqt_hits += 1

        # TS support flags + prefix (support within top kts_pool)
        support_flags: List[bool] = []
        for lid in ts_ranked:
            meta = ts_meta[lid].get("meta", {}) if isinstance(ts_meta[lid].get("meta"), dict) else {}
            ctx = get_origin_text(meta, cache=ts_text_cache, max_chars=int(args.max_ctx_chars))
            support_flags.append(
                is_supported_by_context(
                    answer=answer,
                    ctx=ctx,
                    min_overlap=int(args.min_overlap),
                    overlap_ratio=float(args.overlap_ratio),
                )
            )
        ts_pref = make_prefix_any(support_flags)
        ts_prefix_support.append(ts_pref)
        if ts_pref[min(len(support_flags), kts_pool)]:
            baseline_ts_support_hits += 1

        if args.progress_every and (i + 1) % int(args.progress_every) == 0:
            dt = time.time() - t0
            print(f"[prep] {i+1}/{len(rows)} queries prepared (elapsed {dt:.1f}s)")

    n = len(rows)
    baseline_tl_domqt_rate = baseline_tl_domqt_hits / max(1, n)
    baseline_ts_support_rate = baseline_ts_support_hits / max(1, n)

    # sweep quickly using prefixes
    results: List[SweepRow] = []
    for out_k in out_ks:
        for tsq in ts_quotas:
            if tsq < 0 or tsq > out_k:
                continue

            # quota: TL = out_k - TS (단, 풀 크기가 부족하면 자동으로 남는 슬롯은 TL로 간주)
            tlq = out_k - tsq

            fused_any_hits = 0
            fused_domqt_hits = 0

            for i in range(n):
                # 실제 포함 가능한 TS 개수(풀 제한)
                include_ts = min(tsq, kts_pool, out_k)
                # 나머지는 TL로 채움(풀 제한)
                include_tl = min(kdoc_pool, out_k - include_ts)
                # TS 풀이 부족하면(현재 tsq<=2라 보통 없음) include_ts 줄어들고 include_tl 늘어나는 효과는 out_k-include_ts로 이미 반영됨

                domqt_hit = tl_prefix_domqt[i][include_tl]
                ts_hit = ts_prefix_support[i][include_ts]
                fused_domqt_hits += 1 if domqt_hit else 0
                fused_any_hits += 1 if (domqt_hit or ts_hit) else 0

            results.append(
                SweepRow(
                    out_k=out_k,
                    ts_quota=tsq,
                    tl_quota=tlq,
                    kts_pool=kts_pool,
                    kdoc_pool=kdoc_pool,
                    ts_support_rate=baseline_ts_support_rate,
                    tl_domqt_rate=baseline_tl_domqt_rate,
                    fused_any_rate=fused_any_hits / max(1, n),
                    fused_domqt_rate=fused_domqt_hits / max(1, n),
                )
            )
            r = results[-1]
            print(
                f"[OK] out_k={r.out_k:>2} tsq={r.ts_quota} tlq={r.tl_quota} | "
                f"FUSED(any)={r.fused_any_rate:.4f} FUSED(dom+qt)={r.fused_domqt_rate:.4f} "
                f"TS_support(base@kts_pool)={r.ts_support_rate:.4f}"
            )

    # save csv
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "out_k",
                "ts_quota",
                "tl_quota",
                "kts_pool",
                "kdoc_pool",
                "ts_support_rate_base",
                "tl_domqt_rate_base",
                "fused_any_rate",
                "fused_domqt_rate",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.out_k,
                    r.ts_quota,
                    r.tl_quota,
                    r.kts_pool,
                    r.kdoc_pool,
                    f"{r.ts_support_rate:.6f}",
                    f"{r.tl_domqt_rate:.6f}",
                    f"{r.fused_any_rate:.6f}",
                    f"{r.fused_domqt_rate:.6f}",
                ]
            )

    # best per out_k
    print("\n==== BEST per out_k (by fused_dom+qtype, then fused_any) ====")
    for ok in out_ks:
        cand = [x for x in results if x.out_k == ok]
        if not cand:
            continue
        cand.sort(key=lambda x: (x.fused_domqt_rate, x.fused_any_rate), reverse=True)
        b = cand[0]
        print(
            f"out_k={ok:>2} -> tsq={b.ts_quota}, tlq={b.tl_quota} | "
            f"FUSED(dom+qt)={b.fused_domqt_rate:.4f} FUSED(any)={b.fused_any_rate:.4f}"
        )

    print(f"\nSaved CSV: {csv_out}")


if __name__ == "__main__":
    main()
