from __future__ import annotations
from typing import List
from med_entity_ab.schema import Entity
from med_entity_ab.extractors.base import BaseExtractor

class MedCATExtractor(BaseExtractor):
    name = "medcat"

    def __init__(self, modelpack_path: str):
        from medcat.cat import CAT
        self.cat = CAT.load_model_pack(modelpack_path)

    def extract(self, text: str) -> List[Entity]:
        res = self.cat.get_entities(text)
        ents: List[Entity] = []
        for _, e in res.get("entities", {}).items():
            types = e.get("types") or []
            ents.append(Entity(
                start=int(e.get("start", -1)),
                end=int(e.get("end", -1)),
                text=str(e.get("detected_name", "")),
                label=str(types[0]) if types else None,
                code=e.get("cui"),
                score=float(e.get("acc")) if e.get("acc") is not None else None,
                source=self.name,
                metadata={
                    "pretty_name": e.get("pretty_name"),
                    "context_similarity": e.get("context_similarity"),
                    "icd10": e.get("icd10"),
                }
            ))
        ents.sort(key=lambda x: (x.start, x.end))
        return ents
