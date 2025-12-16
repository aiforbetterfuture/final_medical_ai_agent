from __future__ import annotations
from typing import List
from med_entity_ab.schema import Entity
from med_entity_ab.extractors.base import BaseExtractor

class QuickUMLSExtractor(BaseExtractor):
    name = "quickumls"

    def __init__(
        self,
        index_dir: str,
        threshold: float = 0.7,
        similarity_name: str = "jaccard",
        window: int = 5,
        best_match: bool = True,
    ):
        from quickumls import QuickUMLS
        self.matcher = QuickUMLS(
            index_dir,
            threshold=threshold,
            similarity_name=similarity_name,
            window=window,
        )
        self.best_match = best_match

    def extract(self, text: str) -> List[Entity]:
        matches = self.matcher.match(text, best_match=self.best_match)
        ents: List[Entity] = []
        for group in matches:
            # group: list of candidate dicts for the same span/ngram
            # keep best candidate as entity, but keep candidates list in metadata
            if not group:
                continue
            # group already sorted by similarity in many cases, but sort defensively
            group_sorted = sorted(group, key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
            best = group_sorted[0]
            ents.append(Entity(
                start=int(best.get("start", -1)),
                end=int(best.get("end", -1)),
                text=str(best.get("ngram", "")),
                label=",".join(best.get("semtypes", [])) if best.get("semtypes") else None,
                code=best.get("cui"),
                score=float(best.get("similarity")) if best.get("similarity") is not None else None,
                source=self.name,
                metadata={
                    "preferred": best.get("preferred"),
                    "term": best.get("term"),
                    "candidates": [
                        {
                            "cui": c.get("cui"),
                            "similarity": c.get("similarity"),
                            "preferred": c.get("preferred"),
                            "term": c.get("term"),
                            "semtypes": c.get("semtypes"),
                        }
                        for c in group_sorted[:10]
                    ],
                }
            ))
        ents.sort(key=lambda x: (x.start, x.end))
        return ents
