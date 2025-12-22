
from __future__ import annotations
from pathlib import Path
import argparse, json, os, math
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import orjson
from tqdm import tqdm

try:
    import hnswlib  # type: ignore
    _HNSW_AVAILABLE = True
except Exception:
    hnswlib = None  # type: ignore
    _HNSW_AVAILABLE = False

from sentence_transformers import SentenceTransformer

from common import load_yaml, ensure_dir

def iter_jsonl(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        with p.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                yield orjson.loads(line)

def collect_shards(processed_dir: Path, prefix: str) -> list[Path]:
    return sorted(processed_dir.glob(f"{prefix}_shard*.jsonl"))

def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine space
    )
    return emb.astype(np.float32)

def build_hnsw_index(index_path: Path, meta_path: Path, shards: list[Path], model: SentenceTransformer,
                    batch_size: int, efC: int, M: int, efS: int, space: str = "cosine", resume: bool = True):
    """
    shards(jsonl) -> hnsw index + meta jsonl
    """
    ensure_dir(index_path.parent)
    dim = model.get_sentence_embedding_dimension()

    # count total rows
    total = 0
    for p in shards:
        with p.open("rb") as f:
            for _ in f:
                total += 1

    # resume support (simple): if index exists, skip
    if resume and index_path.exists() and meta_path.exists():
        print(f"[SKIP] {index_path.name} already exists.")
        return

    print(f"Building {index_path.name} (rows={total:,}, dim={dim})")

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=total, ef_construction=efC, M=M)
    index.set_ef(efS)

    # meta writer
    if meta_path.exists():
        meta_path.unlink()
    meta_f = meta_path.open("ab")

    cur = 0
    batch_texts = []
    batch_ids = []
    batch_metas = []

    def flush():
        nonlocal cur, batch_texts, batch_ids, batch_metas
        if not batch_texts:
            return
        emb = embed_texts(model, batch_texts, batch_size=batch_size)
        # numeric labels for hnsw
        labels = np.arange(cur, cur + len(batch_texts))
        index.add_items(emb, labels)

        for uid, meta in zip(batch_ids, batch_metas):
            meta_f.write(orjson.dumps({"lid": int(cur), "id": uid, "meta": meta}))
            meta_f.write(b"\n")
            cur += 1

        batch_texts, batch_ids, batch_metas = [], [], []

    for row in tqdm(iter_jsonl(shards), total=total, desc=index_path.stem):
        uid = row["id"]
        text = row["text"]
        meta = row.get("meta", {})
        batch_ids.append(uid)
        batch_texts.append(text)
        batch_metas.append(meta)
        if len(batch_texts) >= batch_size:
            flush()

    flush()
    meta_f.close()

    index.save_index(str(index_path))
    print(f"Saved: {index_path} / {meta_path}")


def build_flat_index(index_path: Path, meta_path: Path, shards: list[Path], model: SentenceTransformer,
                    batch_size: int, space: str = "cosine", resume: bool = True):
    """
    shards(jsonl) -> flat embeddings (.npy, memmap) + meta jsonl (+ small index info json)
    - Windows/Python 3.13 환경에서 hnswlib 빌드가 어려울 때 fallback 용도
    - embeddings는 normalize_embeddings=True로 저장되므로, dot product == cosine similarity
    """
    ensure_dir(index_path.parent)
    dim = model.get_sentence_embedding_dimension()

    # count total rows
    total = 0
    for p in shards:
        with p.open("rb") as f:
            for _ in f:
                total += 1

    if resume and index_path.exists() and meta_path.exists():
        print(f"[SKIP] {index_path.name} already exists.")
        return

    print(f"Building {index_path.name} (flat rows={total:,}, dim={dim})")

    # create .npy memmap
    emb_mm = np.lib.format.open_memmap(str(index_path), mode="w+", dtype=np.float32, shape=(total, dim))
    meta_f = meta_path.open("wb")

    cur = 0
    batch_texts: list[str] = []
    batch_ids: list[str] = []
    batch_metas: list[dict] = []

    def flush():
        nonlocal cur, batch_texts, batch_ids, batch_metas
        if not batch_texts:
            return
        emb = embed_texts(model, batch_texts, batch_size)
        n = len(batch_texts)
        emb_mm[cur:cur+n] = emb

        for uid, meta in zip(batch_ids, batch_metas):
            meta_f.write(orjson.dumps({"lid": int(cur), "id": uid, "meta": meta}))
            meta_f.write(b"\n")
            cur += 1

        batch_texts, batch_ids, batch_metas = [], [], []

    for row in tqdm(iter_jsonl(shards), total=total, desc=index_path.stem):
        uid = row["id"]
        text = row["text"]
        meta = row.get("meta", {})
        batch_ids.append(uid)
        batch_texts.append(text)
        batch_metas.append(meta)
        if len(batch_texts) >= batch_size:
            flush()

    flush()
    meta_f.close()
    emb_mm.flush()

    info = {"type": "flat", "space": space, "dim": dim, "rows": total}
    info_path = index_path.with_suffix(index_path.suffix + ".json")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {index_path} / {meta_path} (+ {info_path.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--backend", default="auto", choices=["auto","hnsw","flat"], help="auto=hnsw if available else flat")
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "hnsw" if _HNSW_AVAILABLE else "flat"
    if backend == "hnsw" and not _HNSW_AVAILABLE:
        raise SystemExit("hnswlib is not installed. Use --backend flat, or install optional: python -m pip install -r tools/aihub_indexing/requirements_hnsw_optional.txt (recommended: Python 3.11 on Windows).")

    cfg = load_yaml(Path(args.config))
    processed_dir = Path(cfg["output"]["processed_dir"])
    index_dir = Path(cfg["output"]["index_dir"])
    ensure_dir(index_dir)

    model_name = cfg["embedding"]["model_name"]
    batch_size = int(cfg["embedding"]["batch_size"])
    resume = bool(cfg["build"].get("resume", True))

    device = args.device
    if device == "auto":
        # sentence-transformers가 알아서 선택. 강제로 하고 싶으면 cpu/cuda 사용.
        device = None

    model = SentenceTransformer(model_name, device=device)

    idx_cfg = cfg["indexing"]
    efC = int(idx_cfg.get("ef_construction", 200))
    M = int(idx_cfg.get("M", 32))
    efS = int(idx_cfg.get("ef_search", 100))
    space = idx_cfg.get("space", "cosine")

    targets = [
        ("ts_coarse", collect_shards(processed_dir, "ts_coarse")),
        ("ts_fine", collect_shards(processed_dir, "ts_fine")),
        ("tl_stem", collect_shards(processed_dir, "tl_stem")),
        ("tl_stem_opts", collect_shards(processed_dir, "tl_stem_opts")),
        ("tl_q_full", collect_shards(processed_dir, "tl_q_full")),
        ("tl_coarse", collect_shards(processed_dir, "tl_coarse")),
        ("vl_question", collect_shards(processed_dir, "vl_question")),
    ]

    for name, shards in targets:
        if not shards:
            print(f"[WARN] No shards found for {name}. Skipping.")
            continue
        index_path = index_dir / (f"{name}_hnsw.bin" if backend == "hnsw" else f"{name}_flat.npy")
        meta_path = index_dir / f"{name}_meta.jsonl"
        if backend == "hnsw":
            build_hnsw_index(index_path, meta_path, shards, model, batch_size, efC, M, efS, space=space, resume=resume)
        else:
            build_flat_index(index_path, meta_path, shards, model, batch_size, space=space, resume=resume)

    print(f"\nAll indexes built under: {index_dir} (backend={backend})")

if __name__ == "__main__":
    main()
