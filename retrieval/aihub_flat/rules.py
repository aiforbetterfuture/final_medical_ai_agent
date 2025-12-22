from __future__ import annotations
import re
from typing import Optional, List


def detect_lang(query: str) -> str:
    has_ko = bool(re.search(r"[가-힣]", query))
    has_en = bool(re.search(r"[A-Za-z]", query))
    if has_ko and not has_en:
        return "ko"
    if has_en and not has_ko:
        return "en"
    return "any"


def preferred_source_types(query: str) -> Optional[List[str]]:
    q = query.lower()

    dose_kw = ["용량", "투여", "mg", "mcg", "g ", " q", "bid", "tid", "qid", "q12", "q8", "q6", "po", "iv", "항생제", "처방"]
    emerg_kw = ["응급", "쇼크", "호흡곤란", "의식저하", "심정지", "급성", "즉시", "중환자"]

    if any(k in query for k in dose_kw) or any(k in query for k in emerg_kw):
        return ["guideline", "textbook"]

    # 기본은 제한 없이
    return None


def should_use_tl_opts(query: str) -> bool:
    # 선지/보기 형식 감지
    return any(x in query for x in [")", "①", "②", "③", "④", "⑤", "보기", "선지"])

