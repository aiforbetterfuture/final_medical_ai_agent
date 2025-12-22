from __future__ import annotations
from pathlib import Path
import argparse, json
from typing import Any, Dict, List

import yaml
import orjson


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        yield p


def load_json(p: Path) -> Any:
    b = p.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
    try:
        return orjson.loads(b)
    except orjson.JSONDecodeError:
        try:
            return json.loads(b.decode("utf-8-sig"))
        except json.JSONDecodeError:
            try:
                return json.loads(b.decode("utf-16"))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON: {p}") from e


def normalize_qas(obj: Any) -> List[Dict[str, Any]]:
    # AIHub 라벨링은 파일마다 구조가 달라질 수 있어 방어적으로 처리
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return [x for x in obj["data"] if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/aihub_71874_paths.yaml")
    ap.add_argument("--out", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # ✅ 여기 키 이름이 config에 맞아야 합니다.
    # 보통 당신이 쓰던 패턴대로라면 local_paths 아래에 TL_dirs가 있어야 합니다.
    tl_dirs = [Path(x) for x in cfg["local_paths"]["TL_dirs"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    files_seen = 0
    bad_json = 0
    skipped_noqa = 0
    skipped_empty = 0

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for d in tl_dirs:
            for jp in iter_json_files(d):
                files_seen += 1
                try:
                    obj = load_json(jp)
                except Exception:
                    bad_json += 1
                    continue

                qas = normalize_qas(obj)
                if not qas:
                    skipped_noqa += 1
                    continue

                wrote_any = False
                for qa in qas:
                    q = str(qa.get("question", "")).strip()
                    a = str(qa.get("answer", "")).strip()
                    qa_id = qa.get("qa_id") if qa.get("qa_id") is not None else qa.get("id")

                    if not q or not a:
                        skipped_empty += 1
                        continue

                    rec = {
                        "qa_id": qa_id,
                        "domain_id": qa.get("domain") or qa.get("domain_id"),
                        "q_type": qa.get("q_type"),
                        "question": q,
                        "answer": a,
                        "origin_path": str(jp),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    wrote_any = True

                    if args.limit and n >= args.limit:
                        break

                if not wrote_any:
                    skipped_noqa += 1

                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break

    print(f"Saved eval set: {out_path} (rows={n:,})")
    if args.verbose:
        print("---- stats ----")
        print(f"TL dirs        : {len(tl_dirs)}")
        print(f"JSON files seen: {files_seen:,}")
        print(f"Bad JSON files : {bad_json:,}")
        print(f"Skipped(no QA) : {skipped_noqa:,}")
        print(f"Skipped(empty) : {skipped_empty:,}")


if __name__ == "__main__":
    main()
