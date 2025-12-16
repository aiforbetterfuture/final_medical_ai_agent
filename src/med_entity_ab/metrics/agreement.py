from __future__ import annotations
from typing import List, Tuple, Set
from med_entity_ab.schema import Entity

def span_set(ents: List[Entity], label_sensitive: bool = True) -> Set[Tuple[int,int,str]]:
    s = set()
    for e in ents:
        lab = e.label or ""
        if label_sensitive:
            s.add((e.start, e.end, lab))
        else:
            s.add((e.start, e.end, ""))
    return s

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def agreement_jaccard(a_ents: List[Entity], b_ents: List[Entity], label_sensitive: bool = True) -> float:
    return jaccard(span_set(a_ents, label_sensitive), span_set(b_ents, label_sensitive))
