from __future__ import annotations
from pathlib import Path
import argparse, json
from typing import Any, List, Dict

import yaml

def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        yield p

def load_json(p: Path) -> Any:
    b = p.read_bytes()
    try:
        import orjson
        return orjson.loads(b)
    except Exception:
        return json.loads(b.decode("utf-8", errors="ignore"))

def normalize_qas(obj: Any) -> List[Dict]:
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
    ap.add_argument("--out", default="experiments/retrieval_tuning/eval_vl.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # ✅ 여기 키는 사용 중인 YAML에 맞춰야 합니다.
    # 제안본은 cfg["local_paths"]["VL_dirs"]를 가정합니다.
    vl_dirs = [Path(x) for x in cfg["local_paths"]["VL_dirs"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for d in vl_dirs:
            for jp in iter_json_files(d):
                obj = load_json(jp)
                qas = normalize_qas(obj)
                for qa in qas:
                    q = str(qa.get("question", "")).strip()
                    a = str(qa.get("answer", "")).strip()
                    if not q or not a:
                        continue
                    rec = {
                        "qa_id": qa.get("qa_id"),
                        "domain_id": qa.get("domain"),
                        "q_type": qa.get("q_type"),
                        "question": q,
                        "answer": a,
                        "origin_path": str(jp),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if args.limit and n >= args.limit:
                        break
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break

    print(f"Saved eval set: {out_path} (rows={n:,})")

if __name__ == "__main__":
    main()
