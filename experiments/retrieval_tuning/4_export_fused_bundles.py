from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import yaml
import orjson
from sentence_transformers import SentenceTransformer


def _read_text_robust(p: Path) -> str:
    b = p.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("utf-8-sig", errors="replace")


def load_json_robust(p: Path) -> Any:
    b = p.read_bytes()
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


def topk_cosine_flat(emb_mmap: np.ndarray, q: np.ndarray, k: int, block: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    n = emb_mmap.shape[0]
    k = min(k, n)
    best_scores = np.full((k,), -1e9, dtype=np.float32)
    best_idx = np.full((k,), -1, dtype=np.int64)

    q = q.astype(np.float32, copy=False).reshape(-1)

    for start in range(0, n, block):
        end = min(n, start + block)
        scores = emb_mmap[start:end].dot(q)
        if scores.shape[0] <= k:
            local = np.argsort(scores)[::-1]
        else:
            local = np.argpartition(scores, -k)[-k:]
            local = local[np.argsort(scores[local])[::-1]]

        cand_scores = scores[local].astype(np.float32, copy=False)
        cand_idx = (local + start).astype(np.int64, copy=False)

        merged_scores = np.concatenate([best_scores, cand_scores])
        merged_idx = np.concatenate([best_idx, cand_idx])

        top = np.argpartition(merged_scores, -k)[-k:]
        top = top[np.argsort(merged_scores[top])[::-1]]
        best_scores = merged_scores[top]
        best_idx = merged_idx[top]

    return best_idx, best_scores


def extract_text_from_obj(obj: Any, max_chars: int) -> str:
    if obj is None:
        return ""
    if isinstance(obj, dict):
        for k in ("text", "content", "contents", "body", "passage", "article", "document", "answer", "question", "title"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:max_chars]
        # list-like
        for k in ("paragraphs", "sentences", "chunks", "sections", "data"):
            v = obj.get(k)
            if isinstance(v, list) and v:
                parts: List[str] = []
                total = 0
                for it in v[:50]:
                    if isinstance(it, str) and it.strip():
                        s = it.strip()
                    elif isinstance(it, dict):
                        s = ""
                        for kk in ("text", "content", "sentence", "passage", "chunk", "answer", "question"):
                            vv = it.get(kk)
                            if isinstance(vv, str) and vv.strip():
                                s = vv.strip()
                                break
                    else:
                        s = ""
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
        for it in obj[:50]:
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
    p = Path(str(origin)) if origin else None
    if not p or not p.exists():
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


def read_evalset(eval_path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def quota_fuse(ts_ranked: List[int], tl_ranked: List[int], out_k: int, tl_quota: int, ts_quota: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    out_k = int(out_k)
    tl_quota = int(tl_quota)
    ts_quota = int(ts_quota)

    ti = si = 0
    while len(out) < out_k and (tl_quota > 0 or ts_quota > 0):
        if tl_quota > 0 and ti < len(tl_ranked) and len(out) < out_k:
            out.append(("TL", tl_ranked[ti])); ti += 1; tl_quota -= 1
        if ts_quota > 0 and si < len(ts_ranked) and len(out) < out_k:
            out.append(("TS", ts_ranked[si])); si += 1; ts_quota -= 1
        if (ti >= len(tl_ranked)) and (si >= len(ts_ranked)):
            break
    # fill remaining TL then TS
    while len(out) < out_k and ti < len(tl_ranked):
        out.append(("TL", tl_ranked[ti])); ti += 1
    while len(out) < out_k and si < len(ts_ranked):
        out.append(("TS", ts_ranked[si])); si += 1
    return out[:out_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aihub_71874_paths.yaml")
    ap.add_argument("--evalset", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--limit", type=int, default=200)

    ap.add_argument("--ts_index", default="ts_coarse")
    ap.add_argument("--tl_index", default="tl_coarse")
    ap.add_argument("--backend", default="flat", choices=["flat"])

    ap.add_argument("--kts", type=int, default=4)
    ap.add_argument("--kdoc", type=int, default=30)

    ap.add_argument("--out_k", type=int, default=20)
    ap.add_argument("--tl_quota", type=int, default=19)
    ap.add_argument("--ts_quota", type=int, default=1)

    ap.add_argument("--no_self_hit", action="store_true", default=True)
    ap.add_argument("--query_prefix", default="query: ")
    ap.add_argument("--max_chars", type=int, default=1000)

    ap.add_argument("--out", default="experiments/retrieval_tuning/judge_bundles.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(_read_text_robust(Path(args.config)))
    idx_dir = Path(cfg["output"]["index_dir"])
    model_name = cfg["embedding"]["model_name"]
    model = SentenceTransformer(model_name)

    ts_vec = np.load(str(idx_dir / f"{args.ts_index}_flat.npy"), mmap_mode="r")
    tl_vec = np.load(str(idx_dir / f"{args.tl_index}_flat.npy"), mmap_mode="r")

    ts_meta = load_meta(idx_dir / f"{args.ts_index}_meta.jsonl")
    tl_meta = load_meta(idx_dir / f"{args.tl_index}_meta.jsonl")

    rows = read_evalset(Path(args.evalset), limit=args.limit if args.limit else 0)
    if not rows:
        raise SystemExit("Empty evalset.")

    ts_cache: Dict[str, str] = {}
    tl_cache: Dict[str, str] = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    written = 0
    with out_path.open("w", encoding="utf-8") as wf:
        for r in rows:
            q = str(r.get("question", "")).strip()
            a = str(r.get("answer", "")).strip()
            origin = norm_path(r.get("origin_path"))
            if not q or not a:
                continue

            q_model = f"{args.query_prefix}{q}" if args.query_prefix else q
            qv = model.encode([q_model], normalize_embeddings=True).astype(np.float32)[0]

            ts_ids, _ = topk_cosine_flat(ts_vec, qv, k=int(args.kts))
            tl_ids, _ = topk_cosine_flat(tl_vec, qv, k=int(args.kdoc))

            ts_ranked = ts_ids.tolist()
            tl_ranked_raw = tl_ids.tolist()

            # self-hit filter for TL
            tl_ranked: List[int] = []
            if args.no_self_hit and origin:
                for lid in tl_ranked_raw:
                    m = tl_meta[lid].get("meta", {}) if isinstance(tl_meta[lid].get("meta"), dict) else {}
                    if norm_path(m.get("origin_path")) == origin:
                        continue
                    tl_ranked.append(lid)
            else:
                tl_ranked = tl_ranked_raw

            fused = quota_fuse(
                ts_ranked=ts_ranked[: int(args.kts)],
                tl_ranked=tl_ranked[: int(args.kdoc)],
                out_k=int(args.out_k),
                tl_quota=int(args.tl_quota),
                ts_quota=int(args.ts_quota),
            )

            contexts: List[Dict[str, Any]] = []
            for src, lid in fused:
                if src == "TS":
                    meta = ts_meta[lid].get("meta", {}) if isinstance(ts_meta[lid].get("meta"), dict) else {}
                    txt = get_origin_text(meta, ts_cache, max_chars=int(args.max_chars))
                    contexts.append({"src": "TS", "lid": lid, "meta": meta, "text": txt})
                else:
                    meta = tl_meta[lid].get("meta", {}) if isinstance(tl_meta[lid].get("meta"), dict) else {}
                    txt = get_origin_text(meta, tl_cache, max_chars=int(args.max_chars))
                    contexts.append({"src": "TL", "lid": lid, "meta": meta, "text": txt})

            rec = {
                "question": q,
                "gold_answer": a,
                "origin_path": r.get("origin_path"),
                "domain_id": r.get("domain_id"),
                "q_type": r.get("q_type"),
                "qa_id": r.get("qa_id"),
                "params": {
                    "out_k": args.out_k,
                    "tl_quota": args.tl_quota,
                    "ts_quota": args.ts_quota,
                    "kts": args.kts,
                    "kdoc": args.kdoc,
                    "no_self_hit": args.no_self_hit,
                    "ts_index": args.ts_index,
                    "tl_index": args.tl_index,
                    "model": model_name,
                },
                "contexts": contexts,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved judge bundles: {out_path} (rows={written:,})")


if __name__ == "__main__":
    main()