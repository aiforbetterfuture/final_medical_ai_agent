from __future__ import annotations

from pathlib import Path
import argparse
import sys
import orjson
import numpy as np

try:
    import hnswlib  # type: ignore
    _HNSW_AVAILABLE = True
except Exception:
    hnswlib = None  # type: ignore
    _HNSW_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from common import load_yaml


def load_meta(meta_path: Path) -> list[dict]:
    metas: list[dict] = []
    with meta_path.open("rb") as f:
        for line in f:
            if line.strip():
                metas.append(orjson.loads(line))
    return metas


def topk_cosine_flat(
    emb_mmap: np.ndarray,
    q: np.ndarray,
    k: int,
    block: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, scores) for top-k cosine similarity using block scan (memory safe).
    Assumes embeddings are L2-normalized, so dot(q, x) == cosine(q, x).
    """
    n = emb_mmap.shape[0]
    k = min(k, n)

    best_scores = np.full((k,), -1e9, dtype=np.float32)
    best_idx = np.full((k,), -1, dtype=np.int64)

    # Ensure q is float32 vector shape (dim,)
    q = q.astype(np.float32, copy=False).reshape(-1)

    for start in range(0, n, block):
        end = min(n, start + block)

        # (end-start, dim) dot (dim,) -> (end-start,)
        scores = emb_mmap[start:end].dot(q)

        # get local top-k indices
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


def _pick_meta_path(idx_dir: Path, index_name: str, backend: str) -> Path | None:
    """
    Prefer backend-specific meta file first:
      - flat:  {index}_flat_meta.jsonl
      - hnsw:  {index}_hnsw_meta.jsonl
    Fallback:
      - {index}_meta.jsonl
    """
    candidates: list[Path] = []
    if backend == "flat":
        candidates.append(idx_dir / f"{index_name}_flat_meta.jsonl")
    elif backend == "hnsw":
        candidates.append(idx_dir / f"{index_name}_hnsw_meta.jsonl")
    candidates.append(idx_dir / f"{index_name}_meta.jsonl")

    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_backend(idx_dir: Path, index_name: str, backend_arg: str) -> tuple[str, Path]:
    """
    Decide backend and return (backend, index_path).
    backend_arg: auto|flat|hnsw
    """
    hnsw_path = idx_dir / f"{index_name}_hnsw.bin"
    flat_path = idx_dir / f"{index_name}_flat.npy"

    if backend_arg == "auto":
        if hnsw_path.exists():
            if not _HNSW_AVAILABLE:
                raise SystemExit(
                    f"Found HNSW index file but hnswlib is not installed:\n"
                    f"  - {hnsw_path}\n"
                    f"Install hnswlib or run with --backend flat (and ensure flat index exists)."
                )
            return "hnsw", hnsw_path
        if flat_path.exists():
            return "flat", flat_path
        raise SystemExit(
            f"No index file found under {idx_dir} for '{index_name}'. Tried:\n"
            f"  - {hnsw_path.name}\n"
            f"  - {flat_path.name}"
        )

    if backend_arg == "hnsw":
        if not _HNSW_AVAILABLE:
            raise SystemExit("You set --backend hnsw but hnswlib is not installed.")
        if not hnsw_path.exists():
            raise SystemExit(f"Missing HNSW index: {hnsw_path}")
        return "hnsw", hnsw_path

    # backend_arg == "flat"
    if not flat_path.exists():
        raise SystemExit(f"Missing flat index: {flat_path}")
    return "flat", flat_path


def _pretty_meta(m: dict) -> dict:
    """
    Print-friendly meta subset.
    Handles both {meta:{...}} and top-level fields.
    """
    meta = m.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    keep_keys = [
        "source",
        "language",
        "source_type",
        "title",
        "question_id",
        "q_type",
        "origin_path",
        "doc_id",
        "chunk_id",
        "page",
        "section",
    ]

    out = {}
    for k in keep_keys:
        if k in meta and meta.get(k) is not None:
            out[k] = meta.get(k)

    # also surface some common top-level identifiers if present
    for k in ["id", "doc_id", "chunk_id"]:
        if k in m and m.get(k) is not None and k not in out:
            out[k] = m.get(k)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument(
        "--index",
        default="ts_fine",
        choices=["ts_fine", "ts_coarse", "tl_stem", "tl_stem_opts", "tl_q_full", "tl_coarse", "vl_question"],
    )
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--backend", default="auto", choices=["auto", "hnsw", "flat"], help="auto=use existing index file")
    ap.add_argument("--block", type=int, default=200_000, help="flat block size (rows) for scanning")
    ap.add_argument("--show_text_chars", type=int, default=260, help="print up to N chars if text/content exists")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    # ---- paths & model ----
    idx_dir = Path(cfg["output"]["index_dir"])
    if not idx_dir.exists():
        raise SystemExit(f"Index dir does not exist: {idx_dir}")

    model_name = cfg["embedding"]["model_name"]
    model = SentenceTransformer(model_name)

    backend, index_path = _resolve_backend(idx_dir, args.index, args.backend)
    meta_path = _pick_meta_path(idx_dir, args.index, backend)
    if meta_path is None:
        raise SystemExit(
            f"Missing meta jsonl for index '{args.index}' (backend={backend}). Tried:\n"
            f"  - {args.index}_{backend}_meta.jsonl\n"
            f"  - {args.index}_meta.jsonl\n"
            f"in dir: {idx_dir}"
        )

    metas = load_meta(meta_path)

    # ---- query embedding ----
    q_text = args.query.strip()
    q_for_model = f"query: {q_text}"
    qv = model.encode([q_for_model], normalize_embeddings=True).astype(np.float32)[0]  # (dim,)

    # ---- search ----
    if backend == "hnsw":
        space = str(cfg.get("indexing", {}).get("space", "cosine"))
        dim = int(model.get_sentence_embedding_dimension())

        if not _HNSW_AVAILABLE or hnswlib is None:
            raise SystemExit("backend=hnsw but hnswlib is not available (unexpected).")

        index = hnswlib.Index(space=space, dim=dim)
        index.load_index(str(index_path))

        ef_search = int(cfg.get("indexing", {}).get("ef_search", 100))
        try:
            index.set_ef(ef_search)
        except Exception:
            pass

        labels, distances = index.knn_query(qv.reshape(1, -1), k=int(args.topk))

        labels = labels[0]
        distances = distances[0]

        # hnswlib distance meaning depends on space.
        # For cosine space, hnswlib returns distance = 1 - cosine_similarity (common convention).
        # We'll convert to similarity score when space == "cosine".
        if space.lower() == "cosine":
            scores = 1.0 - distances.astype(np.float32)
        else:
            # For l2/ip etc, just print raw distance as "score" (best-effort)
            scores = (-distances).astype(np.float32)

        idxs = labels.astype(np.int64)

    else:
        # flat: memory map + block scan
        flat_path = index_path  # *_flat.npy
        if not flat_path.exists():
            raise SystemExit(f"Missing flat index: {flat_path}")

        # memmap mode: read-only, float32 expected
        emb_mmap = np.load(str(flat_path), mmap_mode="r")
        if emb_mmap.dtype != np.float32:
            # still works, but warn
            print(f"[WARN] flat embeddings dtype is {emb_mmap.dtype}, expected float32", file=sys.stderr)

        idxs, scores = topk_cosine_flat(emb_mmap, qv, int(args.topk), block=int(args.block))

    # ---- print ----
    print("\n================ SMOKE TEST ================")
    print(f"Query   : {q_text}")
    print(f"Index   : {args.index}")
    print(f"Backend : {backend}")
    print(f"IndexFn : {index_path.name}")
    print(f"MetaFn  : {meta_path.name}")
    print("===========================================\n")

    for rank, (lid, score) in enumerate(zip(idxs, scores), start=1):
        if lid < 0:
            continue
        lid_int = int(lid)
        if lid_int >= len(metas):
            print(f"#{rank} score={float(score):.4f} [WARN] meta out of range lid={lid_int}")
            continue

        m = metas[lid_int]
        print(f"#{rank} score={float(score):.4f} lid={lid_int} id={m.get('id', None)}")

        keep = _pretty_meta(m)
        if keep:
            print(" meta:", keep)

        # show text/content if exists
        text = None
        for key in ["text", "content", "chunk", "passage"]:
            if key in m and isinstance(m.get(key), str) and m.get(key).strip():
                text = m.get(key).strip()
                break
        if text:
            n = max(0, int(args.show_text_chars))
            snippet = text[:n] + ("..." if len(text) > n else "")
            print(" text:", snippet)

        print("")

    print("Done.\n")


if __name__ == "__main__":
    main()
