"""tools.final_ready_check

목적:
- final_sanity_suite 실행
- agent retrieve 경로가 runtime yaml(quota)로 동작하는지 점검
- assemble_context가 TS evidence만 프롬프트에 넣는지 확인
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any

from agent.nodes.retrieve import retrieve_node
from agent.nodes.assemble_context import assemble_context_node


def run_sanity_suite() -> bool:
    print("[CHECK] running tools/final_sanity_suite.py ...")
    try:
        res = subprocess.run(
            ["python", "tools/final_sanity_suite.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("[FAIL] final_sanity_suite.py")
        print(e.stdout)
        print(e.stderr)
        return False


async def run_agent_retrieve_and_context(query: str, runtime_yaml: Path) -> Dict[str, Any]:
    # 최소 state
    state: Dict[str, Any] = {
        "user_text": query,
        "query_for_retrieval": query,
        "feature_flags": {
            "retrieval_mode": "aihub_quota",
            "aihub_runtime_yaml": str(runtime_yaml),
            "include_tl_hints": True,
            "tl_hints_k": 5,
        },
        "agent_config": {},
    }

    state = await retrieve_node(state)  # async node
    state = await assemble_context_node(state)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="당뇨 환자 HbA1c 목표 범위")
    ap.add_argument("--runtime_yaml", default="configs/aihub_retrieval_runtime.yaml")
    args = ap.parse_args()

    ok_suite = run_sanity_suite()

    runtime_yaml = Path(args.runtime_yaml)
    if not runtime_yaml.exists():
        raise SystemExit(f"runtime yaml not found: {runtime_yaml}")

    print(f"[CHECK] retrieve+assemble with runtime_yaml={runtime_yaml}")
    state = asyncio.run(run_agent_retrieve_and_context(args.query, runtime_yaml))

    ts = state.get("retrieved_docs", [])
    tl = state.get("tl_hints", [])
    fused = state.get("fused_ranked", [])
    print(f"TS used (evidence): {len(ts)} items")
    print(f"TL hints (pattern only): {len(tl)} items")
    print(f"Fused ranked total: {len(fused)}")
    print("fusion_mode:", state.get("fusion_mode"))

    # evidence와 힌트 사용 여부 로그
    pe_ids = state.get("prompt_evidence_ts_ids", [])
    print("Prompt evidence = TS only :", bool(ts))
    print("prompt_evidence_ts_ids   :", pe_ids[:5])
    print("TL used_as_hint          :", state.get("tl_used_as_hint"))

    # 프롬프트 검사
    sys_prompt = state.get("system_prompt", "")
    has_tl_block = "[TL HINTS" in sys_prompt
    print("TL block in prompt       :", has_tl_block)

    if not ok_suite:
        raise SystemExit("final_sanity_suite failed")

    print("\n[READY] final_ready_check passed.")


if __name__ == "__main__":
    main()

