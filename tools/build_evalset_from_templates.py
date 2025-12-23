# tools/build_evalset_from_templates.py
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit("PyYAML이 필요합니다. pip install pyyaml") from e


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(_read_text(p)) or {}


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _jsonl_write(path: Path, rows: List[dict]) -> None:
    _ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_format(template: str, vars: Dict[str, Any]) -> str:
    """템플릿 문자열에 {age}, {sex} 같은 placeholder가 있으면 치환합니다."""
    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    try:
        return template.format_map(_Safe(vars))
    except Exception:
        return template


@dataclass
class PhaseSpec:
    name: str
    required_slots: List[str]
    next_questions: List[str]


def _parse_phases(y: dict) -> List[PhaseSpec]:
    phases: List[PhaseSpec] = []
    default_req = (y.get("default") or {}).get("required_slots") or []
    for name, block in y.items():
        if name == "default":
            continue
        if not isinstance(block, dict):
            continue
        req = list(dict.fromkeys(list(default_req) + list(block.get("required_slots") or [])))
        qs = block.get("next_questions") or []
        if isinstance(qs, str):
            qs = [qs]
        phases.append(PhaseSpec(name=name, required_slots=req, next_questions=list(qs)))

    order = {"diagnosis": 0, "treatment": 1, "follow_up": 2}
    phases.sort(key=lambda p: (order.get(p.name, 999), p.name))
    return phases


def _make_patient(rng: random.Random, i: int) -> dict:
    sex = rng.choice(["M", "F"])
    age = rng.randint(20, 85)
    pool = ["당뇨", "고혈압", "고지혈증", "천식", "간질환", "신장질환", "우울증"]
    cond = rng.sample(pool, k=rng.randint(1, 2))
    return {"patient_id": f"P{i:04d}", "age": age, "sex": sex, "conditions": cond}


def build_evalset(templates_yaml: Path, *, cases: int, per_phase: int, seed: int) -> List[dict]:
    y = _load_yaml(templates_yaml)
    phases = _parse_phases(y)
    if not phases:
        raise SystemExit("question_templates.yaml에 default 외 phase(diagnosis/treatment/follow_up 등)가 없습니다.")

    rng = random.Random(seed)
    rows: List[dict] = []

    patients = y.get("patients")
    use_patients = patients if isinstance(patients, list) and patients else None

    for ci in range(cases):
        patient = use_patients[ci % len(use_patients)] if use_patients else _make_patient(rng, ci + 1)
        case_id = f"C{ci+1:05d}"

        turn_id = 0
        for phase in phases:
            if not phase.next_questions:
                continue
            qs = phase.next_questions[:]
            rng.shuffle(qs)
            pick = qs[: max(1, per_phase)]
            for q in pick:
                turn_id += 1
                q_rendered = _safe_format(str(q), patient)
                rows.append(
                    {
                        "case_id": case_id,
                        "turn_id": turn_id,
                        "phase": phase.name,
                        "question": q_rendered,
                        "required_slots": phase.required_slots,
                        "patient": patient,
                        "created_at": _now_iso(),
                    }
                )
    return rows


def main():
    ap = argparse.ArgumentParser(description="question_templates.yaml로부터 evalset.jsonl 생성")
    ap.add_argument("--templates", default="question_templates.yaml", help="YAML 템플릿 경로")
    ap.add_argument("--out", default="experiments/evalsets/evalset_from_templates.jsonl", help="생성할 evalset jsonl")
    ap.add_argument("--cases", type=int, default=100, help="케이스(환자 시나리오) 수")
    ap.add_argument("--per_phase", type=int, default=1, help="phase당 질문 샘플 개수")
    ap.add_argument("--seed", type=int, default=13, help="재현성 시드")
    args = ap.parse_args()

    templates_yaml = Path(args.templates)
    if not templates_yaml.exists():
        raise SystemExit(f"templates YAML not found: {templates_yaml}")

    rows = build_evalset(templates_yaml, cases=int(args.cases), per_phase=int(args.per_phase), seed=int(args.seed))
    outp = Path(args.out)
    _jsonl_write(outp, rows)
    print(f"[OK] wrote evalset: {outp} (rows={len(rows)})")
    print("Tip) 다음 단계: 에이전트 실행 결과(run.jsonl)를 만든 뒤 grade_run.py로 평가하세요.")


if __name__ == "__main__":
    main()
