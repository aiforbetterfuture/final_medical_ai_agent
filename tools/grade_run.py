from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on sys.path even when running: python tools/grade_run.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _extract_question(row: Dict[str, Any]) -> str:
    for k in ("query", "question", "user_query", "q", "prompt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(row)


def _extract_evidence(state: Any) -> str:
    if not isinstance(state, dict):
        return ""
    for k in ("ts_evidence", "ts_context", "evidence_ts", "contexts_ts"):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            return v
    ctx = state.get("contexts")
    if isinstance(ctx, dict):
        v = ctx.get("ts") or ctx.get("TS")
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "\n\n".join(str(x) for x in v[:10])
    return ""


def _score_rule_based(answer: str, state: Any, rubric: Dict[str, Any]) -> Dict[str, float]:
    # Very light rule-based fallbacks; the LLM-judge does most scoring.
    scores = {"factuality": 0.0, "safety": 1.0, "completeness": 0.0, "context_use": 0.0, "clarity": 0.0}
    if answer.strip():
        scores["clarity"] = 0.5
        scores["completeness"] = 0.3
    # If state exposes slot usage, credit it.
    if isinstance(state, dict):
        slots = state.get("slots") or state.get("slot_state") or state.get("slots_state")
        if isinstance(slots, dict) and any(bool(v) for v in slots.values()):
            scores["context_use"] = 0.5
    return scores


def _weighted_total(scores: Dict[str, float], rubric: Dict[str, Any]) -> float:
    w = (rubric.get("weights") or {})
    if not isinstance(w, dict) or not w:
        # default equal weights
        w = {k: 1.0 for k in scores.keys()}
    num = 0.0
    den = 0.0
    for k, v in scores.items():
        wk = float(w.get(k, 0.0))
        num += wk * float(v)
        den += wk
    return (num / den) if den > 0 else 0.0


def _call_agent(question: str, session_id: str, model_mode: str, feature_overrides: Optional[Dict[str, Any]], debug: bool) -> Tuple[str, Any]:
    # Import from stable entrypoint (handles API drift).
    from agent.entrypoint import run_agent  # type: ignore
    out = run_agent(question, session_id=session_id, model_mode=model_mode, return_state=True, debug=debug, feature_overrides=feature_overrides)
    if isinstance(out, tuple) and len(out) == 2:
        return str(out[0]), out[1]
    # If the underlying run_agent returned only a string, treat as answer.
    return str(out), {}


def _maybe_llm_judge(answer: str, state: Any, rubric: Dict[str, Any], model_mode: str, debug: bool) -> Optional[Dict[str, Any]]:
    llm_cfg = rubric.get("llm_judge") or {}
    if not isinstance(llm_cfg, dict) or not llm_cfg.get("enabled", False):
        return None
    try:
        from llm_as_judge import judge_one  # tools/llm_as_judge.py is in sys.path[0]
    except Exception as e:
        if debug:
            print(f"[WARN] cannot import llm_as_judge: {e}")
        return None
    evidence_ts = _extract_evidence(state)
    # judge_one(model, system_prompt, question, answer, ts_evidence, extra={...})
    model = str(llm_cfg.get("model", "gpt-4.1-mini"))
    system_prompt = str(llm_cfg.get("system_prompt", ""))
    try:
        return judge_one(model=model, system_prompt=system_prompt, question="", answer=answer, ts_evidence=evidence_ts, extra={"model_mode": model_mode})
    except TypeError:
        # Older judge_one signature (positional)
        return judge_one(model, system_prompt, "", answer, evidence_ts, {"model_mode": model_mode})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--run", default="experiments/eval_runs/run.jsonl")
    ap.add_argument("--rubric", default="configs/eval_rubric.yaml")
    ap.add_argument("--out", default="experiments/eval_runs/grades.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_llm_judge", action="store_true")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--model_mode", default="default")
    ap.add_argument("--debug_agent", action="store_true")
    args = ap.parse_args(argv)

    evalset_path = Path(args.evalset)
    run_path = Path(args.run)
    rubric_path = Path(args.rubric)
    out_path = Path(args.out)

    rubric = _load_yaml(rubric_path) if rubric_path.exists() else {}

    if args.pipeline:
        # 1) run agent for each eval item -> append to run.jsonl
        rows = _read_jsonl(evalset_path)
        if args.limit and args.limit > 0:
            rows = rows[: int(args.limit)]
        # truncate existing run file to avoid mixing runs
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text("", encoding="utf-8")
        for i, row in enumerate(rows):
            q = _extract_question(row)
            ex_id = str(row.get("id") or row.get("case_id") or f"ex-{i:05d}")
            session_id = ex_id
            feature_overrides = row.get("feature_flags") if isinstance(row.get("feature_flags"), dict) else None
            t0 = time.time()
            answer, state = _call_agent(q, session_id=session_id, model_mode=str(args.model_mode), feature_overrides=feature_overrides, debug=bool(args.debug_agent))
            dt = time.time() - t0
            _append_jsonl(run_path, {"id": ex_id, "query": q, "answer": answer, "state": state, "latency_s": round(dt, 3)})

    # 2) grade run.jsonl -> grades.jsonl
    runs = _read_jsonl(run_path)
    if args.limit and args.limit > 0:
        runs = runs[: int(args.limit)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    for rec in runs:
        answer = str(rec.get("answer", ""))
        state = rec.get("state")
        scores = _score_rule_based(answer, state, rubric)
        judge = None
        if not args.no_llm_judge:
            judge = _maybe_llm_judge(answer, state, rubric, model_mode=str(args.model_mode), debug=bool(args.debug_agent))
            if isinstance(judge, dict) and isinstance(judge.get("scores"), dict):
                # merge judge scores (judge wins)
                for k, v in judge["scores"].items():
                    if k in scores:
                        scores[k] = float(v)

        total = _weighted_total(scores, rubric)
        out_rec = {
            "id": rec.get("id"),
            "scores": scores,
            "total": total,
            "llm_judge": judge,
        }
        _append_jsonl(out_path, out_rec)

    print(f"Wrote: {out_path} (n={len(runs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
