from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import json
import orjson


def load_eval_map(eval_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build map: qa_id(str) -> {"question":..., "answer":..., "domain_id":..., "q_type":..., "origin_path":...}
    """
    mp: Dict[str, Dict[str, Any]] = {}
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            qa_id = str(o.get("qa_id", "")).strip()
            if not qa_id:
                continue
            # 마지막 값이 이겨도 무방(qa_id는 보통 유일)
            mp[qa_id] = {
                "question": o.get("question"),
                "answer": o.get("answer"),
                "domain_id": o.get("domain_id"),
                "q_type": o.get("q_type"),
                "origin_path": o.get("origin_path"),
            }
    return mp


def patch_meta(meta_in: Path, meta_out: Path, mp: Dict[str, Dict[str, Any]], verbose: bool = False) -> Tuple[int, int]:
    """
    Read jsonl rows: {"id","lid","meta":{...}}
    If meta.qa_id matches eval map, inject meta.question/meta.answer (+domain_id/q_type if missing)
    """
    updated = 0
    total = 0

    with meta_in.open("rb") as r, meta_out.open("wb") as w:
        for line in r:
            if not line.strip():
                continue
            total += 1
            row = orjson.loads(line)
            m = row.get("meta")
            if not isinstance(m, dict):
                w.write(line if line.endswith(b"\n") else line + b"\n")
                continue

            qa_id = str(m.get("qa_id", "")).strip()
            if qa_id and qa_id in mp:
                src = mp[qa_id]
                # inject if missing
                if "question" not in m or not m.get("question"):
                    m["question"] = src.get("question")
                if "answer" not in m or not m.get("answer"):
                    m["answer"] = src.get("answer")

                # domain/q_type 보강(없을 때만)
                if ("domain_id" not in m or m.get("domain_id") is None) and src.get("domain_id") is not None:
                    m["domain_id"] = src.get("domain_id")
                if ("q_type" not in m or m.get("q_type") is None) and src.get("q_type") is not None:
                    m["q_type"] = src.get("q_type")

                row["meta"] = m
                updated += 1

                if verbose and updated <= 3:
                    print(f"[patch] qa_id={qa_id} question_len={len((m.get('question') or ''))}")

                w.write(orjson.dumps(row) + b"\n")
            else:
                w.write(orjson.dumps(row) + b"\n")

    return total, updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--meta", required=True, help="data/aihub_71874/indexes/tl_stem_meta.jsonl (or tl_stem_opts_meta.jsonl)")
    ap.add_argument("--backup", action="store_true", help="create .bak copy")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    eval_path = Path(args.eval)
    meta_path = Path(args.meta)
    if not eval_path.exists():
        raise SystemExit(f"Missing eval: {eval_path}")
    if not meta_path.exists():
        raise SystemExit(f"Missing meta: {meta_path}")

    mp = load_eval_map(eval_path)

    out_path = meta_path.with_suffix(meta_path.suffix + ".patched")
    total, updated = patch_meta(meta_path, out_path, mp, verbose=args.verbose)

    if args.backup:
        bak = meta_path.with_suffix(meta_path.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(meta_path.read_bytes())

    # replace
    meta_path.write_bytes(out_path.read_bytes())
    out_path.unlink(missing_ok=True)

    print(f"[OK] patched: {meta_path}")
    print(f"  total rows   : {total:,}")
    print(f"  updated rows : {updated:,}")


if __name__ == "__main__":
    main()
