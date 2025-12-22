from __future__ import annotations
import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def topk_dot_blockscan(
    emb_mmap: np.ndarray,
    q: np.ndarray,
    k: int,
    block: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    emb와 q가 이미 L2-normalized되어 있다고 가정하면 cosine = dot 입니다.
    -> denom 계산 없이 매우 빨라집니다.
    """
    n = emb_mmap.shape[0]
    k = min(k, n)

    best_scores = np.full((k,), -1e9, dtype=np.float32)
    best_idx = np.full((k,), -1, dtype=np.int64)

    q = q.astype(np.float32, copy=False)

    for start in range(0, n, block):
        end = min(start + block, n)
        chunk = emb_mmap[start:end].astype(np.float32, copy=False)
        scores = chunk @ q  # dot == cosine(정규화 가정)

        if scores.shape[0] <= k:
            local_idx = np.arange(scores.shape[0], dtype=np.int64)
        else:
            local_idx = np.argpartition(scores, -k)[-k:].astype(np.int64)

        local_scores = scores[local_idx]
        local_global_idx = local_idx + start

        merged_scores = np.concatenate([best_scores, local_scores])
        merged_idx = np.concatenate([best_idx, local_global_idx])

        sel = np.argpartition(merged_scores, -k)[-k:]
        best_scores = merged_scores[sel]
        best_idx = merged_idx[sel]

    order = np.argsort(best_scores)[::-1]
    return best_idx[order], best_scores[order]

