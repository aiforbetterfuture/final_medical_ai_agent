from __future__ import annotations
import os
import time
import yaml
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from med_entity_ab.schema import Entity

def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    if isinstance(obj, str):
        # ${VAR} -> env
        if obj.startswith("${") and obj.endswith("}"):
            key = obj[2:-1]
            return os.environ.get(key, obj)
        return os.path.expandvars(obj)
    return obj

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env_vars(cfg)

@dataclass
class ExtractResult:
    entities: List[Entity]
    latency_ms: float

class EntityABPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.extractors = {}

        if cfg.get("medcat", {}).get("enabled"):
            from med_entity_ab.extractors.medcat_extractor import MedCATExtractor
            mp = cfg["medcat"]["modelpack_path"]
            self.extractors["medcat"] = MedCATExtractor(mp)

        if cfg.get("quickumls", {}).get("enabled"):
            from med_entity_ab.extractors.quickumls_extractor import QuickUMLSExtractor
            q = cfg["quickumls"]
            self.extractors["quickumls"] = QuickUMLSExtractor(
                index_dir=q["index_dir"],
                threshold=float(q.get("threshold", 0.7)),
                similarity_name=str(q.get("similarity_name", "jaccard")),
                window=int(q.get("window", 5)),
                best_match=bool(q.get("best_match", True)),
            )

        if cfg.get("kmbert_ner", {}).get("enabled"):
            from med_entity_ab.extractors.kmbert_extractor import KMBERTNERExtractor
            k = cfg["kmbert_ner"]
            self.extractors["kmbert_ner"] = KMBERTNERExtractor(
                model_dir=k["model_dir"],
                aggregation_strategy=str(k.get("aggregation_strategy", "simple")),
            )

    def extract_all(self, text: str) -> Dict[str, ExtractResult]:
        outputs: Dict[str, ExtractResult] = {}
        for name, ext in self.extractors.items():
            t0 = time.perf_counter()
            ents = ext.extract(text)
            t1 = time.perf_counter()
            outputs[name] = ExtractResult(entities=ents, latency_ms=(t1 - t0) * 1000.0)
        return outputs
