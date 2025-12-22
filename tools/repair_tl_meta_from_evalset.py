from __future__ import annotations
from pathlib import Path
import argparse
import json
from typing import Any, Dict, Optional

import yaml

try:
    import orjson  # type: ignore
except Exception:
    orjson = None


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def jloads(b: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(b)
    return json.loads(b.decode("utf-8"))


def jdumps(obj: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj)
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def flatten_meta_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust flatten:
    - row may be {"id":..,"lid":..,"meta":{...}}
    - meta may be nested again: {"meta": {"meta": {...}}}
    """
    out: Dict[str, Any] = {}
    m = row.get("meta")
    if isinstance(m, dict):
        out.update(m)
        # handle meta.meta (double nested)
        mm = m.get("meta")
        if isinstance(mm, dict):
            out.update(mm)

    # keep top-level too (do not overwrite existing)
    for k, v in row.items():
        if k == "meta":
            continue
        if k not in out:
            out[k] = v
    return out


def set_if_empty(dst: Dict[str, Any], key: str, val: Any):
    cur = dst.get(key)
    if cur is None:
        dst[key] = val
        return
    if isinstance(cur, str) and cur.strip() == "":
        dst[key] = val


def build_eval_map(evalset: Path) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    with evalset.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            qa_id = str(o.get("qa_id", "")).strip()
            if not qa_id:
                continue
            mp[qa_id] = {
                "qa_id": qa_id,
                "domain_id": o.get("domain_id"),
                "q_type": o.get("q_type"),
                "question": o.get("question"),
                "answer": o.get("answer"),
                "origin_path": o.get("origin_path"),
            }
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/aihub_71874_paths.yaml")
    ap.add_argument("--index", required=True, help="e.g. tl_coarse / tl_stem / tl_stem_opts")
    ap.add_argument("--evalset", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--inplace", action="store_true", help="원본 meta를 .bak로 백업 후 교체")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    index_dir = Path(cfg["output"]["index_dir"])

    meta_path = index_dir / f"{args.index}_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    eval_path = Path(args.evalset)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evalset not found: {eval_path}")

    eval_map = build_eval_map(eval_path)
    print(f"[eval] loaded {len(eval_map):,} qa rows from {eval_path}")

    out_path = meta_path.with_suffix(".repaired.jsonl")

    total = 0
    hit = 0
    missing_qa = 0

    with meta_path.open("rb") as r, out_path.open("wb") as w:
        for line in r:
            if not line.strip():
                continue
            row = jloads(line)
            if not isinstance(row, dict):
                w.write(line)
                continue

            flat = flatten_meta_row(row)
            qa_id = flat.get("qa_id")
            qa_id = str(qa_id).strip() if qa_id is not None else ""
            total += 1

            if not qa_id:
                missing_qa += 1
                w.write(jdumps(row) + (b"" if orjson is not None else b""))
                continue

            src = eval_map.get(qa_id)
            if src is None:
                w.write(jdumps(row) + (b"" if orjson is not None else b""))
                continue

            # 어디에 쓸지: row["meta"]가 dict면 거기에 채우는 걸 우선
            target = row.get("meta") if isinstance(row.get("meta"), dict) else row
            if isinstance(target, dict):
                set_if_empty(target, "qa_id", src.get("qa_id"))
                set_if_empty(target, "domain_id", src.get("domain_id"))
                set_if_empty(target, "q_type", src.get("q_type"))
                set_if_empty(target, "question", src.get("question"))
                set_if_empty(target, "answer", src.get("answer"))
                set_if_empty(target, "origin_path", src.get("origin_path"))

            hit += 1
            w.write(jdumps(row) + (b"" if orjson is not None else b""))

    print(f"[meta] rows={total:,} repaired_by_eval={hit:,} missing_qa_id={missing_qa:,}")
    print(f"[out ] {out_path}")

    if args.inplace:
        bak = meta_path.with_suffix(".bak.jsonl")
        if not bak.exists():
            meta_path.rename(bak)
        else:
            # 이미 bak가 있으면 원본을 .bak2로
            meta_path.rename(meta_path.with_suffix(".bak2.jsonl"))
        out_path.rename(meta_path)
        print(f"[inplace] replaced meta. backup={bak} -> {meta_path}")


if __name__ == "__main__":
    main()
