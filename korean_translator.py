# -*- coding: utf-8 -*-
"""
Dictionary-based Korean ↔ English translator used by MultilingualMedCAT.

This keeps a lightweight, dependency-free fallback so initialization never fails
when the neural translator or external packages are unavailable.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


class KoreanTranslator:
    """Rule-based translator for medical terms (Korean -> English)."""

    MEDICAL_TERM_DICT: Dict[str, str] = {
        # diseases
        "당뇨병": "diabetes mellitus",
        "고혈압": "hypertension",
        "뇌졸중": "stroke",
        "허혈성 심질환": "ischemic heart disease",
        "고지혈증": "hyperlipidemia",
        "심부전": "heart failure",
        "천식": "asthma",
        "COPD": "COPD",
        # symptoms
        "흉통": "chest pain",
        "호흡곤란": "dyspnea",
        "두통": "headache",
        "복통": "abdominal pain",
        "어지럼증": "dizziness",
        "발열": "fever",
        "기침": "cough",
        "가래": "sputum",
        # medications
        "메트포르민": "metformin",
        "리시노프릴": "lisinopril",
        "아스피린": "aspirin",
        "스타틴": "statin",
        "인슐린": "insulin",
        # measurements / labs
        "혈압": "blood pressure",
        "공복혈당": "fasting blood glucose",
        "A1c": "HbA1c",
        "당화혈색소": "HbA1c",
        "심박수": "heart rate",
        "산소포화도": "oxygen saturation",
    }

    def __init__(self, *, extra_terms: Dict[str, str] | None = None):
        # copy to avoid mutating the class-level constant
        self.term_dict: Dict[str, str] = dict(self.MEDICAL_TERM_DICT)
        if extra_terms:
            self.term_dict.update(extra_terms)

        # Pre-sort keys (longest first) to avoid partial replacements.
        self._sorted_terms: Tuple[Tuple[str, str], ...] = tuple(
            sorted(self.term_dict.items(), key=lambda kv: len(kv[0]), reverse=True)
        )
        # Pre-compute reverse mapping for English -> Korean conversions.
        self._reverse_terms: Tuple[Tuple[str, str], ...] = tuple(
            sorted(
                ((v.lower(), k) for k, v in self.term_dict.items()),
                key=lambda kv: len(kv[0]),
                reverse=True,
            )
        )

    def translate_to_english(self, korean_text: str) -> str:
        """Replace known Korean medical terms with their English equivalents."""
        if not korean_text:
            return korean_text

        translated = korean_text
        for korean_term, english_term in self._sorted_terms:
            if korean_term in translated:
                translated = translated.replace(korean_term, english_term)
        return translated

    def translate_to_korean(self, english_text: str) -> str:
        """Replace known English medical terms with their Korean equivalents (case-insensitive)."""
        if not english_text:
            return english_text

        translated = english_text
        for english_term, korean_term in self._reverse_terms:
            pattern = re.compile(re.escape(english_term), flags=re.IGNORECASE)
            translated = pattern.sub(korean_term, translated)
        return translated

    def add_terms(self, pairs: Dict[str, str] | Iterable[Tuple[str, str]]) -> None:
        """Dynamically extend the term dictionary."""
        if isinstance(pairs, dict):
            pairs = pairs.items()
        for k, v in pairs:
            self.term_dict[k] = v
        # refresh caches without losing newly added terms
        self._sorted_terms = tuple(
            sorted(self.term_dict.items(), key=lambda kv: len(kv[0]), reverse=True)
        )
        self._reverse_terms = tuple(
            sorted(
                ((v.lower(), k) for k, v in self.term_dict.items()),
                key=lambda kv: len(kv[0]),
                reverse=True,
            )
        )


__all__ = ["KoreanTranslator"]
