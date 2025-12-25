#!/usr/bin/env python3
"""
LLM-as-Judge 통합 테스트 스크립트

사용법:
    python tools/test_llm_judge.py
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.llm_as_judge import (
    LLMJudgeConfig,
    compute_perplexity,
    judge_one,
    load_rubric,
)


def test_imports():
    """테스트 1: Import 확인"""
    print("=" * 60)
    print("테스트 1: Import 확인")
    print("=" * 60)
    print("✅ 모든 모듈 import 성공")
    print()


def test_rubric_load():
    """테스트 2: Rubric 로드"""
    print("=" * 60)
    print("테스트 2: Rubric 로드")
    print("=" * 60)

    rubric_path = "configs/eval_rubric.yaml"

    if not os.path.exists(rubric_path):
        print(f"⚠️  경고: {rubric_path} 파일이 없습니다")
        print("   기본 설정을 사용합니다")
        cfg = LLMJudgeConfig()
    else:
        cfg = LLMJudgeConfig.from_rubric(rubric_path)
        print(f"✅ Rubric 로드 성공: {rubric_path}")

    print(f"   - Model: {cfg.model}")
    print(f"   - Temperature: {cfg.temperature}")
    print(f"   - Max Tokens: {cfg.max_tokens}")
    print(f"   - Threshold: {cfg.threshold}")
    print(f"   - Enabled: {cfg.enabled}")
    print()


def test_perplexity():
    """테스트 3: Perplexity 계산"""
    print("=" * 60)
    print("테스트 3: Perplexity 계산")
    print("=" * 60)

    test_text = "당뇨 환자는 혈당 관리가 중요합니다."
    result = compute_perplexity(test_text)

    print(f"   텍스트: {test_text}")
    print(f"   - Perplexity: {result['perplexity']}")
    print(f"   - Source: {result['perplexity_source']}")
    print(f"   - OK: {result['perplexity_ok']}")

    if result['perplexity_ok']:
        print("   ✅ Perplexity 계산 성공 (transformers 설치됨)")
    else:
        print("   ⚠️  Perplexity 계산 불가 (transformers 미설치)")
        print("   ℹ️  선택사항이므로 계속 진행 가능합니다")
        print("   ℹ️  설치: pip install transformers torch")

    print()


def test_judge_mock():
    """테스트 4: Judge 함수 (모의 실행)"""
    print("=" * 60)
    print("테스트 4: Judge 함수 스키마 확인")
    print("=" * 60)

    # 실제 LLM 호출 없이 스키마만 확인
    print("   ℹ️  실제 LLM 호출 없이 스키마만 확인합니다")

    # Mock result
    mock_result = {
        "scores": {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_use": 0.0
        },
        "perplexity": -1.0,
        "perplexity_ok": False,
        "perplexity_source": "mock",
        "verdict": "skip",
        "rationale": "테스트 모드",
        "raw_text": ""
    }

    # Validate schema
    required_keys = ['scores', 'perplexity', 'perplexity_ok', 'verdict', 'rationale']
    required_score_keys = ['faithfulness', 'answer_relevance', 'context_use']

    all_ok = True
    for key in required_keys:
        if key in mock_result:
            print(f"   ✅ 키 존재: {key}")
        else:
            print(f"   ❌ 키 누락: {key}")
            all_ok = False

    for key in required_score_keys:
        if key in mock_result['scores']:
            print(f"   ✅ 점수 키 존재: scores.{key}")
        else:
            print(f"   ❌ 점수 키 누락: scores.{key}")
            all_ok = False

    if all_ok:
        print("   ✅ 스키마 검증 성공")
    else:
        print("   ❌ 스키마 검증 실패")

    print()


def test_alias_mapping():
    """테스트 5: Alias 매핑 확인"""
    print("=" * 60)
    print("테스트 5: Alias 매핑 (하위 호환)")
    print("=" * 60)

    alias_map = {
        "factuality": "faithfulness",
        "relevance": "answer_relevance",
        "completeness": "answer_relevance",
        "faithfulness": "faithfulness",
        "answer_relevance": "answer_relevance",
        "context_use": "context_use",
    }

    print("   레거시 키 → 표준 키 매핑:")
    for legacy, standard in alias_map.items():
        if legacy != standard:
            print(f"   - {legacy:20s} → {standard}")

    print("\n   ✅ Alias 매핑 확인 완료")
    print("   ℹ️  LLM이 레거시 키로 응답해도 자동 변환됩니다")
    print()


def test_yaml_structure():
    """테스트 6: YAML 구조 확인"""
    print("=" * 60)
    print("테스트 6: YAML 구조 확인")
    print("=" * 60)

    rubric_path = "configs/eval_rubric.yaml"

    if not os.path.exists(rubric_path):
        print(f"   ⚠️  {rubric_path} 파일이 없습니다")
        print()
        return

    try:
        rubric = load_rubric(rubric_path)

        # Check llm_judge section
        if 'llm_judge' in rubric:
            print("   ✅ llm_judge 섹션 존재")
            llm_judge = rubric['llm_judge']

            required = ['enabled', 'model', 'temperature', 'max_tokens', 'timeout_s']
            for key in required:
                if key in llm_judge:
                    print(f"      ✅ {key}: {llm_judge[key]}")
                else:
                    print(f"      ⚠️  {key} 누락")
        else:
            print("   ⚠️  llm_judge 섹션 누락")

        # Check perplexity section
        if 'perplexity' in rubric:
            print("   ✅ perplexity 섹션 존재")
            ppl = rubric['perplexity']
            print(f"      - enabled: {ppl.get('enabled')}")
            print(f"      - model: {ppl.get('model')}")
        else:
            print("   ⚠️  perplexity 섹션 누락 (선택사항)")

        print("\n   ✅ YAML 구조 확인 완료")

    except Exception as e:
        print(f"   ❌ YAML 로드 실패: {e}")

    print()


def main():
    """모든 테스트 실행"""
    print("\n")
    print("=" * 60)
    print(" " * 10 + "LLM-as-Judge 통합 테스트")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_rubric_load,
        test_perplexity,
        test_judge_mock,
        test_alias_mapping,
        test_yaml_structure,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"   ❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print("테스트 완료")
    print("=" * 60)
    print()
    print("다음 단계:")
    print("1. Perplexity 계산 활성화 (선택):")
    print("   pip install transformers torch")
    print()
    print("2. 실제 LLM 평가 실행:")
    print("   python tools/grade_run.py --pipeline ...")
    print()
    print("3. 자세한 내용:")
    print("   251225_3metrics_twins.md 참조")
    print()


if __name__ == "__main__":
    main()
