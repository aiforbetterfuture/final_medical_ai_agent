from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

@dataclass
class Entity:
    start: int
    end: int
    text: str
    label: Optional[str] = None
    code: Optional[str] = None   # e.g., UMLS CUI
    score: Optional[float] = None
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

@dataclass
class DocumentPrediction:
    id: str
    text: str
    entities: List[Entity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
        }
