from __future__ import annotations

from typing import Any


def rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    key_fn,
    weights: list[float] | None = None,
    rrf_k: int = 60,
    topk: int = 20,
) -> list[dict[str, Any]]:
    """
    ranked_lists: 이미 각 리스트가 (score 상관없이) rank 순서로 정렬되어 있다고 가정
    key_fn: 아이템을 유니크하게 식별하는 키(중복 제거용)
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length mismatch")

    fused: dict[str, dict[str, Any]] = {}
    score_map: dict[str, float] = {}

    for li, items in enumerate(ranked_lists):
        w = float(weights[li])
        for rank0, item in enumerate(items):
            rank = rank0 + 1
            key = str(key_fn(item))
            score_map[key] = score_map.get(key, 0.0) + w * (1.0 / (rrf_k + rank))
            if key not in fused:
                fused[key] = dict(item)  # shallow copy

    out = list(fused.values())
    out.sort(key=lambda x: score_map[str(key_fn(x))], reverse=True)

    # fused score도 같이 넣어주면 디버깅/논문 로그에 매우 유용
    for x in out:
        x["_rrf"] = float(score_map[str(key_fn(x))])

    return out[:topk]

