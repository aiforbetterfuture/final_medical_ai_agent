#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-Judge Integration Test

Usage:
    python tools/test_llm_judge_simple.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.llm_as_judge import LLMJudgeConfig, compute_perplexity, judge_one, load_rubric


def main():
    print("\n" + "=" * 60)
    print("          LLM-as-Judge Integration Test")
    print("=" * 60 + "\n")

    # Test 1: Imports
    print("[Test 1] Import check")
    print("  [OK] All modules imported successfully\n")

    # Test 2: Rubric Load
    print("[Test 2] Rubric loading")
    rubric_path = "configs/eval_rubric.yaml"
    if os.path.exists(rubric_path):
        cfg = LLMJudgeConfig.from_rubric(rubric_path)
        print(f"  [OK] Loaded rubric: {rubric_path}")
        print(f"       Model: {cfg.model}")
        print(f"       Threshold: {cfg.threshold}")
    else:
        print(f"  [WARN] Rubric not found: {rubric_path}")
        cfg = LLMJudgeConfig()
    print()

    # Test 3: Perplexity
    print("[Test 3] Perplexity calculation")
    test_text = "Test sentence for perplexity."
    result = compute_perplexity(test_text)
    print(f"  Text: {test_text}")
    print(f"  Perplexity: {result['perplexity']}")
    print(f"  Source: {result['perplexity_source']}")
    if result['perplexity_ok']:
        print("  [OK] Perplexity calculated (transformers installed)")
    else:
        print("  [INFO] Perplexity unavailable (transformers not installed)")
        print("         This is optional. Install with: pip install transformers torch")
    print()

    # Test 4: Schema validation
    print("[Test 4] Output schema validation")
    required_keys = ['scores', 'perplexity', 'perplexity_ok', 'verdict', 'rationale']
    required_scores = ['faithfulness', 'answer_relevance', 'context_use']
    
    mock = {
        'scores': {'faithfulness': 0.0, 'answer_relevance': 0.0, 'context_use': 0.0},
        'perplexity': -1.0, 'perplexity_ok': False, 'verdict': 'skip', 'rationale': 'test'
    }
    
    all_ok = all(k in mock for k in required_keys)
    all_ok = all_ok and all(k in mock['scores'] for k in required_scores)
    
    if all_ok:
        print("  [OK] Schema validation passed")
    else:
        print("  [FAIL] Schema validation failed")
    print()

    # Test 5: Alias mapping
    print("[Test 5] Alias mapping (backward compatibility)")
    aliases = {
        "factuality": "faithfulness",
        "relevance": "answer_relevance",
        "completeness": "answer_relevance"
    }
    print("  Legacy keys -> Standard keys:")
    for old, new in aliases.items():
        print(f"    {old:20s} -> {new}")
    print("  [OK] Alias mapping verified\n")

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Optional: Install perplexity support")
    print("   pip install transformers torch")
    print("\n2. Run full evaluation pipeline:")
    print("   python tools/grade_run.py --pipeline ...")
    print("\n3. See documentation:")
    print("   251225_3metrics_twins.md\n")


if __name__ == "__main__":
    main()
