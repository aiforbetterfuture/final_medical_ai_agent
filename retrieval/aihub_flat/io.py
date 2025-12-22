from __future__ import annotations

from pathlib import Path
import json

try:
    import orjson  # type: ignore
    _HAS_ORJSON = True
except Exception:
    orjson = None
    _HAS_ORJSON = False


def resolve_path(index_dir: Path, name: str, exts: tuple[str, ...]) -> Path:
    """
    name이 'ts_fine_meta'처럼 확장자 없이 올 수도 있어 자동 탐색.
    exts 후보 중 실제 존재하는 파일을 찾음.
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

    # index_dir/name 이 그대로 존재하면 우선 사용
    candidate = index_dir / name
    if candidate.exists():
        return candidate

    # 확장자 보정
    for ext in exts:
        candidate2 = index_dir / f"{name}{ext}"
        if candidate2.exists():
            return candidate2

    # 마지막: name이 이미 확장자일 수도 있으니 그대로 한 번 더
    if (index_dir / name).exists():
        return index_dir / name

    raise FileNotFoundError(f"File not found under {index_dir}: {name} (tried exts={exts})")


def read_jsonl(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if _HAS_ORJSON:
                yield orjson.loads(line)
            else:
                yield json.loads(line.decode("utf-8"))

