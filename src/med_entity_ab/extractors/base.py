from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from med_entity_ab.schema import Entity

class BaseExtractor(ABC):
    name: str

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        raise NotImplementedError
