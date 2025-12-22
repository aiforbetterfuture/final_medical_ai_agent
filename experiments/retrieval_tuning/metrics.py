from __future__ import annotations
import re
from typing import List

STOP = set(["그리고", "또한", "때문에", "대한", "환자", "필요", "치료", "사용", "경우", "있다", "한다"])

def extract_keywords(text: str) -> List[str]:
    """
    매우 단순한 키워드 추출(휴리스틱):
    - 숫자/단위(mg, g, ml, %, 회, 일 등)
    - 한글/영문 토큰(길이>=2)
    - 너무 흔한 단어 제거
    """
    text = (text or "").lower()
    tokens = re.findall(
        r"[a-z]{2,}|[가-힣]{2,}|\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|l|%|회|일|시간|day|hr|h)",
        text
    )
    out: List[str] = []
    for t in tokens:
        t = t.strip()
        if t in STOP:
            continue
        if len(t) < 2:
            continue
        out.append(t)

    # 중복 제거(순서 유지)
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def keyword_recall(answer: str, contexts: List[str]) -> float:
    """
    VL의 answer에서 뽑은 키워드가 검색 context(=TS chunk들) 안에 얼마나 포함되는지.
    - 검색/생성엔 answer를 쓰지 않고 평가에만 씀.
    """
    keys = extract_keywords(answer)
    if not keys:
        return 0.0
    blob = " ".join(contexts or []).lower()
    hit = sum(1 for k in keys if k in blob)
    return hit / max(1, len(keys))
