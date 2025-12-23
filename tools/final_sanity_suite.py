from __future__ import annotations

import argparse
import compileall
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REQUIRED_RUNTIME_KEYS = [
    "ts_index", "tl_index", "backend", "kdoc", "kts_pool",
    "rrf_k", "fusion_mode", "out_k", "tl_quota", "ts_quota",
    "quota_strategy", "no_self_hit_tl"
]


def _run(cmd: list[str], cwd: Path, timeout: int = 180) -> tuple[int, str]:
    # decoding 에러 때문에 스위트가 "죽지 않게" errors=replace
    proc = subprocess.run(
        cmd, cwd=str(cwd),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out.strip()


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="final_medical_ai_agent repo root")
    ap.add_argument("--runtime_yaml", default="configs/aihub_retrieval_runtime.yaml")
    ap.add_argument("--paths_yaml", default="configs/aihub_71874_paths.yaml")
    ap.add_argument("--evalset", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--query", default="당뇨 환자 HbA1c 목표 범위")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    runtime_yaml = (root / args.runtime_yaml).resolve()
    paths_yaml = (root / args.paths_yaml).resolve()
    evalset = (root / args.evalset).resolve()

    print("== FINAL SANITY SUITE ==")
    print("repo_root   :", root)
    print("runtime_yaml:", runtime_yaml)
    print("paths_yaml  :", paths_yaml)
    print("evalset     :", evalset)
    print("limit       :", args.limit)

    # 1) compileall
    ok = compileall.compile_dir(str(root / "retrieval"), quiet=1) \
        and compileall.compile_dir(str(root / "experiments"), quiet=1) \
        and compileall.compile_dir(str(root / "agent"), quiet=1)
    print("[PASS] compileall (retrieval/experiments/agent)" if ok else "[FAIL] compileall (retrieval/experiments/agent)")

    # 2) runtime yaml keys
    rt = _read_yaml(runtime_yaml)
    missing = [k for k in REQUIRED_RUNTIME_KEYS if k not in rt]
    if missing:
        print("[FAIL] runtime_yaml missing keys:", missing)
    else:
        print("[PASS] runtime_yaml has required keys")
        print(f"      fusion_mode={rt.get('fusion_mode')} out_k={rt.get('out_k')} tlq={rt.get('tl_quota')} tsq={rt.get('ts_quota')}")

    # 3) demo_cli run
    rc, out = _run(
        [sys.executable, "-m", "retrieval.aihub_flat.demo_cli", "--query", args.query, "--show_cfg", "--debug_keys"],
        cwd=root,
        timeout=180
    )
    if rc == 0:
        print("[PASS] demo_cli runs")
    else:
        print("[FAIL] demo_cli runs")
        print(out)

    # 4) meta jsonl required fields
    # (간단 체크: meta 안에 필요한 키가 있는지)
    import orjson  # local import
    idx_dir = None
    try:
        paths = _read_yaml(paths_yaml)
        idx_dir = Path(paths["output"]["index_dir"])
    except Exception:
        idx_dir = root / "data" / "aihub_71874" / "indexes"

    required_meta = {
        "ts_fine_meta.jsonl": ["doc_id", "chunk_id", "origin_path"],
        "ts_coarse_meta.jsonl": ["doc_id", "origin_path"],
        "tl_stem_meta.jsonl": ["qa_id", "domain_id", "q_type", "origin_path"],
        "tl_stem_opts_meta.jsonl": ["qa_id", "domain_id", "q_type", "origin_path"],
    }
    meta_ok = True
    for fn, keys in required_meta.items():
        p = idx_dir / fn
        if not p.exists():
            meta_ok = False
            print(f"[FAIL] meta jsonl missing file: {p}")
            continue
        try:
            row = orjson.loads(p.open("rb").readline())
            meta = row.get("meta", {})
            if not isinstance(meta, dict):
                meta_ok = False
                print(f"[FAIL] {fn}: meta is not dict")
                continue
            miss = [k for k in keys if k not in meta]
            if miss:
                meta_ok = False
                print(f"[FAIL] {fn}: missing meta keys {miss}")
        except Exception as e:
            meta_ok = False
            print(f"[FAIL] {fn}: read error {e}")

    if meta_ok:
        print("[PASS] meta jsonl contains required fields (prevents None)")
        for fn in required_meta:
            print(f"      {fn}: OK")

    # 5) 2_run_fusion_tuning runs (no_self_hit)
    # NOTE: 스크립트 실행은 sys.path 문제를 일으킬 수 있으므로,
    # 2_run_fusion_tuning.py 내부에서 repo root를 sys.path에 추가하도록 고쳐둠.
    rc2, out2 = _run(
        [
            sys.executable,
            str(root / "experiments" / "retrieval_tuning" / "2_run_fusion_tuning.py"),
            "--evalset", str(evalset),
            "--limit", str(args.limit),
            "--kdoc", "30",
            "--kts", "4",
            "--no_self_hit",
            "--fusion_mode", str(rt.get("fusion_mode", "quota")),
            "--out_k", str(rt.get("out_k", 22)),
            "--tl_quota", str(rt.get("tl_quota", 20)),
            "--ts_quota", str(rt.get("ts_quota", 2)),
            "--quota_strategy", str(rt.get("quota_strategy", "tl_first")),
        ],
        cwd=root,
        timeout=180
    )
    if rc2 == 0:
        print("[PASS] 2_run_fusion_tuning runs (no_self_hit)")
    else:
        print("[FAIL] 2_run_fusion_tuning runs (no_self_hit)")
        print(out2)

    # 6) SSOT check: direct build/default usage outside tuning
    # - runtime.py는 "SSOT builder" 파일이므로 예외로 둠
    # - 이 파일(final_sanity_suite.py)도 예외로 둠
    # - experiments/retrieval_tuning은 예외(튜닝 스크립트)
    pattern = r"AIHubFusedRetriever" + r"\.build|AIHubIndexConfig" + r"\.default"
    offenders = []
    allow_files = {
        (root / "retrieval" / "aihub_flat" / "runtime.py").resolve(),
        Path(__file__).resolve(),
    }
    for p in root.rglob("*.py"):
        rp = p.resolve()
        if rp in allow_files:
            continue
        s = p.as_posix().replace("\\", "/")
        if "experiments/retrieval_tuning" in s:
            continue
        if "retrieval/aihub_flat/__pycache__" in s:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if re.search(pattern, txt):
            offenders.append(s)

    if offenders:
        print("[FAIL] No direct build/default usage outside tuning (SSOT 유지)")
        for o in offenders[:30]:
            print("     ", o)
    else:
        print("[PASS] No direct build/default usage outside tuning (SSOT 유지)")

    print("\n== Done ==")


if __name__ == "__main__":
    main()
