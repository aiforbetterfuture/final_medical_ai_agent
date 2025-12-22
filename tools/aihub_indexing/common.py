
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import re, os, json, hashlib, zipfile

import yaml
import orjson

DOMAIN_NAME_TO_ID = {
    # 가이드라인 도메인 정의 fileciteturn7file0L16-L19 에 기반(코드에선 숫자만 사용)
    "외과": 1,
    "예방의학": 2,
    "정신건강의학과": 3,
    "신경과신경외과": 4,      # 폴더명이 "신경과신경외과"로 들어온 케이스
    "신경과/신경외과": 4,
    "피부과": 5,
    "안과": 6,
    "이비인후과": 7,
    "비뇨의학과": 8,
    "방사선종양학과": 9,
    "병리과": 10,
    "마취통증의학과": 11,
    "의료법규": 12,
    "기타": 13,
    "산부인과": 14,
    "소아청소년과": 15,
    "응급의학과": 16,
    "내과": 17,
}

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def iter_files_recursive(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def is_json_file(p: Path) -> bool:
    return p.suffix.lower() == ".json"

def is_zip_file(p: Path) -> bool:
    return p.suffix.lower() == ".zip"

def guess_ts_language_and_source_type(folder_name: str) -> tuple[str, str]:
    # folder_name like "TS_국문_학술 논문 및 저널"
    lang = "ko" if "TS_국문" in folder_name else ("en" if "TS_영문" in folder_name else "unknown")
    if "가이드라인" in folder_name:
        source_type = "guideline"
    elif "의학 교과서" in folder_name:
        source_type = "textbook"
    elif "학술 논문" in folder_name or "저널" in folder_name:
        source_type = "paper_journal"
    elif "온라인 의료 정보" in folder_name:
        source_type = "web_medinfo"
    else:
        source_type = "other"
    return lang, source_type

_MCQLINE = re.compile(r"^\s*([1-5])\)\s*(.+)$")

def split_mcq_stem_options(question: str) -> tuple[str, list[tuple[str, str]]]:
    """
    TL의 q_type=1(객관식) question 문자열에서 stem과 options를 분리.
    질문 텍스트는 줄바꿈 기반으로 1)~5) 패턴을 탐색합니다.
    """
    lines = question.splitlines()
    stem_lines = []
    options: list[tuple[str, str]] = []
    in_options = False
    for line in lines:
        m = _MCQLINE.match(line)
        if m:
            in_options = True
            options.append((m.group(1), m.group(2).strip()))
        else:
            if not in_options:
                stem_lines.append(line)
            else:
                # 보기 문장이 줄바꿈으로 이어지는 경우: 마지막 option에 이어붙임
                if options and line.strip():
                    k, v = options[-1]
                    options[-1] = (k, (v + " " + line.strip()).strip())
    stem = "\n".join([x.strip() for x in stem_lines]).strip()
    return stem, options

def format_options(options: list[tuple[str, str]]) -> str:
    return "\n".join([f"{k}) {v}" for k, v in options]).strip()

def extract_choice_from_answer(answer: str) -> Optional[str]:
    # "1) ...", "정답: 3)" 같은 케이스 대응(보수적으로)
    m = re.search(r"\b([1-5])\)", answer)
    return m.group(1) if m else None

def flatten_text_from_json(obj: Any, *, max_chars: int = 2_000_000) -> str:
    """
    TS 원천 JSON 구조가 문서마다 다를 수 있으니,
    문자열들을 재귀적으로 모아 하나의 텍스트로 합칩니다.
    - id/url/짧은 메타는 최대한 배제(길이 기준)
    """
    parts: list[str] = []
    def rec(x: Any):
        if isinstance(x, str):
            s = x.strip()
            # 너무 짧은 값(예: id, url)은 버림
            if len(s) >= 25 and not s.lower().startswith("http"):
                parts.append(s)
        elif isinstance(x, dict):
            for k, v in x.items():
                if k in {"id", "doc_id", "qa_id", "url", "source", "date", "created_at"}:
                    continue
                rec(v)
        elif isinstance(x, list):
            for it in x:
                rec(it)
        else:
            return
    rec(obj)
    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text

def read_json_bytes(b: bytes) -> Any:
    try:
        # Strip UTF-8 BOM if present
        if b.startswith(b"\xef\xbb\xbf"):
            b = b[3:]
        return orjson.loads(b)
    except Exception:
        return json.loads(b.decode("utf-8-sig", errors="ignore"))

def read_json_path(p: Path) -> Any:
    b = p.read_bytes()
    return read_json_bytes(b)

def iter_json_objects_from_path(root_or_file: Path) -> Iterator[tuple[str, Any, dict]]:
    """
    입력 폴더/파일에서 JSON 객체를 순회합니다.
    - JSON 파일이면 그대로 yield
    - ZIP이면 내부 JSON 파일을 순회
    반환: (origin_path, obj, extra_meta)
    """
    if root_or_file.is_file():
        if is_json_file(root_or_file):
            yield (str(root_or_file), read_json_path(root_or_file), {})
        elif is_zip_file(root_or_file):
            with zipfile.ZipFile(root_or_file, "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".json"):
                        b = zf.read(name)
                        yield (f"{root_or_file}!{name}", read_json_bytes(b), {"zip_path": str(root_or_file), "zip_member": name})
        return

    # directory
    for p in iter_files_recursive(root_or_file):
        if is_json_file(p) or is_zip_file(p):
            yield from iter_json_objects_from_path(p)
