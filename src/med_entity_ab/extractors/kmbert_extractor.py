from __future__ import annotations
from typing import List
from med_entity_ab.schema import Entity
from med_entity_ab.extractors.base import BaseExtractor

class KMBERTNERExtractor(BaseExtractor):
    name = "kmbert_ner"

    def __init__(self, model_dir: str, aggregation_strategy: str = "simple"):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.nlp = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=self.device,
        )

    def extract(self, text: str) -> List[Entity]:
        out = self.nlp(text)
        ents: List[Entity] = []
        for o in out:
            ents.append(Entity(
                start=int(o.get("start", -1)),
                end=int(o.get("end", -1)),
                text=str(o.get("word", "")).strip(),
                label=o.get("entity_group"),
                code=None,  # NER only
                score=float(o.get("score", 0.0)),
                source=self.name,
                metadata={}
            ))
        ents.sort(key=lambda x: (x.start, x.end))
        return ents
