from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from med_entity_ab.schema import Entity
from med_entity_ab.metrics.ner_metrics import _iou

@dataclass
class LinkMetrics:
    accuracy_at_1: float
    mrr: float
    n: int

def _get_candidate_list(e: Entity) -> List[str]:
    # Return list of candidate codes (CUIs), best first
    cands = []
    if e.code:
        cands.append(e.code)
    meta = e.metadata or {}
    if "candidates" in meta and isinstance(meta["candidates"], list):
        for c in meta["candidates"]:
            code = c.get("cui") or c.get("code")
            if code and code not in cands:
                cands.append(code)
    return cands

def evaluate_linking(
    gold: List[Entity],
    pred: List[Entity],
    match_mode: str = "strict",  # "strict" or "overlap"
    iou_threshold: float = 0.0,
    label_sensitive: bool = True,
    k: int = 5,
) -> LinkMetrics:
    # Align gold↔pred by span (strict or overlap). Compute Accuracy@1 and MRR using candidate lists.
    gold_sorted = sorted(gold, key=lambda x: (x.start, x.end))
    pred_sorted = sorted(pred, key=lambda x: (x.start, x.end))

    used_pred = set()
    ranks = []
    total = 0

    for g in gold_sorted:
        if not g.code:
            continue  # linking gold 없으면 스킵
        best_pi = None
        best_score = -1.0
        for pi, p in enumerate(pred_sorted):
            if pi in used_pred:
                continue
            if label_sensitive and (g.label != p.label):
                continue

            if match_mode == "strict":
                ok = (g.start == p.start and g.end == p.end)
                score = 1.0 if ok else 0.0
            else:
                score = _iou(g, p)

            if score > best_score:
                best_score = score
                best_pi = pi

        if best_pi is None:
            continue
        if match_mode != "strict" and best_score <= iou_threshold:
            continue
        if match_mode == "strict" and best_score < 1.0:
            continue

        used_pred.add(best_pi)
        total += 1
        p = pred_sorted[best_pi]
        cand_codes = _get_candidate_list(p)[:k] if k > 0 else _get_candidate_list(p)
        # rank in candidates (1-indexed)
        rank = None
        for idx, code in enumerate(cand_codes, start=1):
            if code == g.code:
                rank = idx
                break
        if rank is None:
            ranks.append(0.0)
        else:
            ranks.append(1.0 / rank)

    if total == 0:
        return LinkMetrics(accuracy_at_1=0.0, mrr=0.0, n=0)

    acc1 = sum(1 for r in ranks if r == 1.0) / total
    mrr = sum(ranks) / total
    return LinkMetrics(accuracy_at_1=acc1, mrr=mrr, n=total)
