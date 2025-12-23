# tools/grade_run.py
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit("PyYAML이 필요합니다. pip install pyyaml") from e


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(_read_text(p)) or {}


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_print(s: str) -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((s + "\n").encode("cp949", errors="backslashreplace"))
        sys.stdout.flush()


def _key_of(row: dict) -> str:
    if row.get("qa_id") is not None:
        return f"qa_id::{row.get('qa_id')}"
    if row.get("case_id") is not None and row.get("turn_id") is not None:
        return f"case::{row.get('case_id')}::{row.get('turn_id')}"
    if row.get("_line") is not None:
        return f"line::{row.get('_line')}"
    return f"row::{id(row)}"


def _norm_text(x: Any) -> str:
    return "" if x is None else str(x)


def _extract_answer(run_row: dict) -> str:
    for k in ["answer", "final_answer", "response", "assistant_answer", "output"]:
        if k in run_row and run_row.get(k) is not None:
            return _norm_text(run_row.get(k))
    return ""


def _extract_fused(run_row: dict) -> Tuple[int, int, int]:
    fused = run_row.get("fused_ranked")
    if isinstance(fused, list) and fused:
        ts = tl = 0
        for it in fused:
            key = str(it.get("key") or it.get("id") or it.get("source") or "")
            if key.startswith("TS::") or "::TS" in key:
                ts += 1
            elif key.startswith("TL::") or "::TL" in key:
                tl += 1
            else:
                src = str(it.get("source") or "")
                if src.startswith("ts_") or src == "ts":
                    ts += 1
                elif src.startswith("tl_") or src == "tl":
                    tl += 1
        return ts, tl, len(fused)

    docs = run_row.get("retrieved_docs")
    if isinstance(docs, list) and docs:
        ts = tl = 0
        for d in docs:
            src = str(d.get("source") or "")
            st = str(d.get("source_type") or "")
            if src.startswith("ts_") or st.lower() == "ts":
                ts += 1
            elif src.startswith("tl_") or st.lower() == "tl":
                tl += 1
        return ts, tl, len(docs)

    return 0, 0, 0


def _slot_hit_rate(answer: str, required_slots: List[str], slot_aliases: Dict[str, List[str]]) -> float:
    if not required_slots:
        return 1.0
    if not answer:
        return 0.0
    hit = 0
    for s in required_slots:
        aliases = slot_aliases.get(s) or [s]
        ok = any((a and re.search(re.escape(str(a)), answer, flags=re.IGNORECASE)) for a in aliases)
        hit += 1 if ok else 0
    return hit / max(1, len(required_slots))


def grade(evalset_path: Path, run_path: Path, rubric_path: Path) -> dict:
    rubric = _load_yaml(rubric_path)
    rule_cfg = (rubric.get("rule_based") or {})
    llm_cfg = (rubric.get("llm_judge") or {})

    min_ts = int(rule_cfg.get("min_ts_count", 1))
    min_tl = int(rule_cfg.get("min_tl_count", 0))
    slot_aliases = rule_cfg.get("slot_aliases") or {}

    eval_index: Dict[str, dict] = {}
    for i, r in enumerate(_iter_jsonl(evalset_path), 1):
        r["_line"] = i
        eval_index[_key_of(r)] = r
        eval_index[f"line::{i}"] = r  # line fallback

    per_row: List[dict] = []
    n = pass_ts = pass_tl = 0
    mean_slot = 0.0

    for j, rr in enumerate(_iter_jsonl(run_path), 1):
        rr["_line"] = j
        key = _key_of(rr)
        ev = eval_index.get(key) or eval_index.get(f"line::{j}")

        required_slots = (ev.get("required_slots") if isinstance(ev, dict) else None) or []
        answer = _extract_answer(rr)

        ts_cnt, tl_cnt, fused_total = _extract_fused(rr)
        ok_ts = ts_cnt >= min_ts
        ok_tl = tl_cnt >= min_tl

        slot_rate = _slot_hit_rate(answer, required_slots, slot_aliases)

        n += 1
        pass_ts += 1 if ok_ts else 0
        pass_tl += 1 if ok_tl else 0
        mean_slot += slot_rate

        per_row.append(
            {
                "key": key,
                "matched": ev is not None,
                "ts_count": ts_cnt,
                "tl_count": tl_cnt,
                "fused_total": fused_total,
                "pass_min_ts": ok_ts,
                "pass_min_tl": ok_tl,
                "slot_hit_rate": slot_rate,
                "llm_judge": None,
                "llm_judge_note": llm_cfg.get("note", "LLM-as-a-judge 연결은 스켈레톤 상태입니다."),
            }
        )

    summary = {
        "created_at": _now_iso(),
        "evalset": str(evalset_path),
        "run": str(run_path),
        "rubric": str(rubric_path),
        "rows": n,
        "pass_min_ts_rate": (pass_ts / n) if n else 0.0,
        "pass_min_tl_rate": (pass_tl / n) if n else 0.0,
        "mean_slot_hit_rate": (mean_slot / n) if n else 0.0,
        "rule_based": {"min_ts_count": min_ts, "min_tl_count": min_tl},
        "notes": [
            "LLM-as-a-judge는 스켈레톤입니다. eval_rubric.yaml의 llm_judge 섹션을 참고해 API 연결을 추가하세요.",
            "run.jsonl에 fused_ranked가 없으면 retrieved_docs에서 TS/TL을 추정합니다.",
        ],
    }
    return {"summary": summary, "rows": per_row}


def main():
    ap = argparse.ArgumentParser(description="run.jsonl을 eval_rubric.yaml 기준으로 자동 채점(스켈레톤)")
    ap.add_argument("--evalset", default="experiments/evalsets/evalset_from_templates.jsonl")
    ap.add_argument("--run", required=True, help="에이전트 실행 결과 jsonl")
    ap.add_argument("--rubric", default="eval_rubric.yaml")
    ap.add_argument("--out", default="experiments/grades/grade_report.json")
    ap.add_argument("--out_rows", default=None, help="(선택) row별 결과를 jsonl로 저장")
    args = ap.parse_args()

    evalset_path = Path(args.evalset)
    run_path = Path(args.run)
    rubric_path = Path(args.rubric)

    if not evalset_path.exists():
        raise SystemExit(f"evalset not found: {evalset_path}")
    if not run_path.exists():
        raise SystemExit(f"run not found: {run_path}")
    if not rubric_path.exists():
        raise SystemExit(f"rubric not found: {rubric_path}")

    report = grade(evalset_path, run_path, rubric_path)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _safe_print(f"[OK] wrote report: {outp}")

    if args.out_rows:
        out_rows = Path(args.out_rows)
        out_rows.parent.mkdir(parents=True, exist_ok=True)
        with out_rows.open("w", encoding="utf-8") as f:
            for r in report["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        _safe_print(f"[OK] wrote per-row: {out_rows}")

    s = report["summary"]
    _safe_print(f"rows={s['rows']} pass_min_ts_rate={s['pass_min_ts_rate']:.3f} mean_slot_hit_rate={s['mean_slot_hit_rate']:.3f}")


if __name__ == "__main__":
    main()
