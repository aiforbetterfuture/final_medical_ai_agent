from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from med_entity_ab.schema import Entity

def _overlap(a: Entity, b: Entity) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))

def _iou(a: Entity, b: Entity) -> float:
    inter = _overlap(a, b)
    if inter <= 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0

@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
    return PRF(p, r, f, tp, fp, fn)

def strict_span_match(gold: List[Entity], pred: List[Entity], label_sensitive: bool = True) -> PRF:
    gold_set = set()
    for e in gold:
        key = (e.start, e.end, e.label) if label_sensitive else (e.start, e.end)
        gold_set.add(key)

    pred_set = set()
    for e in pred:
        key = (e.start, e.end, e.label) if label_sensitive else (e.start, e.end)
        pred_set.add(key)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return _prf(tp, fp, fn)

def overlap_match(
    gold: List[Entity],
    pred: List[Entity],
    label_sensitive: bool = True,
    iou_threshold: float = 0.0,
) -> PRF:
    # Greedy bipartite matching: each gold matched at most once, each pred at most once
    used_pred = set()
    tp = 0

    # sort for deterministic behavior
    gold_sorted = sorted(gold, key=lambda x: (x.start, x.end))
    pred_sorted = sorted(pred, key=lambda x: (x.start, x.end))

    for gi, g in enumerate(gold_sorted):
        best_pi = None
        best_score = 0.0
        for pi, p in enumerate(pred_sorted):
            if pi in used_pred:
                continue
            if label_sensitive and (g.label != p.label):
                continue
            score = _iou(g, p)
            if score > best_score:
                best_score = score
                best_pi = pi
        if best_pi is not None and best_score > iou_threshold:
            used_pred.add(best_pi)
            tp += 1

    fp = len(pred_sorted) - tp
    fn = len(gold_sorted) - tp
    return _prf(tp, fp, fn)

def boundary_iou_mean(gold: List[Entity], pred: List[Entity], label_sensitive: bool = True) -> float:
    # Mean IoU of matched pairs under greedy matching; unmatched contribute 0
    gold_sorted = sorted(gold, key=lambda x: (x.start, x.end))
    pred_sorted = sorted(pred, key=lambda x: (x.start, x.end))
    used_pred = set()
    ious = []
    for g in gold_sorted:
        best_pi = None
        best = 0.0
        for pi, p in enumerate(pred_sorted):
            if pi in used_pred:
                continue
            if label_sensitive and (g.label != p.label):
                continue
            score = _iou(g, p)
            if score > best:
                best = score
                best_pi = pi
        if best_pi is not None and best > 0:
            used_pred.add(best_pi)
            ious.append(best)
        else:
            ious.append(0.0)
    return sum(ious) / len(ious) if ious else 0.0
