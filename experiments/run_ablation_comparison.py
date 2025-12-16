"""
다중 Ablation 프로파일 비교 실험

여러 프로파일을 동일한 쿼리로 테스트하여 성능 비교

Usage:
    python experiments/run_ablation_comparison.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import time

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.graph import run_agent
from config.ablation_config import ABLATION_PROFILES, get_ablation_profile

# ============================================================
# 설정 섹션
# ============================================================

# 비교할 프로파일 목록 (config/ablation_config.py 참조)
PROFILES_TO_TEST = [
    "baseline",
    "self_refine_heuristic",
    "self_refine_llm_quality",
    "self_refine_dynamic_query",
    "full_context_engineering",
]

# 테스트 쿼리 (빠른 비교를 위해 적은 수 사용)
TEST_QUERIES = [
    "당뇨병 환자에게 메트포르민의 부작용은?",
    "고혈압 환자의 식이요법은?",
    "아스피린 복용 시 피해야 할 음식은?",
    "임신 중 복용 가능한 진통제는?",
    "간 질환 환자에게 금기인 약물은?",
]

# ============================================================
# 실행 섹션
# ============================================================

def main():
    print("=" * 80)
    print("Ablation Study - 프로파일 비교 실험")
    print("=" * 80)
    print(f"비교 프로파일 수: {len(PROFILES_TO_TEST)}")
    print(f"테스트 쿼리 수: {len(TEST_QUERIES)}")
    print(f"총 실행 횟수: {len(PROFILES_TO_TEST) * len(TEST_QUERIES)}")
    print("=" * 80)
    print()

    # 프로파일별 결과 저장
    all_results = {}

    # 각 프로파일 테스트
    for profile_idx, profile_name in enumerate(PROFILES_TO_TEST, 1):
        print(f"\n{'='*80}")
        print(f"[{profile_idx}/{len(PROFILES_TO_TEST)}] 프로파일: {profile_name}")
        print(f"{'='*80}")

        # 프로파일 로드
        try:
            features = get_ablation_profile(profile_name)
            print(f"설명: {ABLATION_PROFILES[profile_name]['description']}")
        except ValueError as e:
            print(f"❌ 오류: {e}")
            continue

        # 캐시 비활성화 (순수 성능 측정)
        features['response_cache_enabled'] = False

        profile_results = []

        # 각 쿼리 실행
        for query_idx, query in enumerate(TEST_QUERIES, 1):
            print(f"  [{query_idx}/{len(TEST_QUERIES)}] {query[:40]}...")

            query_start = time.time()

            try:
                result = run_agent(
                    user_text=query,
                    mode="ai_agent",
                    feature_overrides=features,
                    return_state=True
                )

                query_elapsed = time.time() - query_start

                metrics = {
                    'query_id': query_idx,
                    'query': query,
                    'quality_score': result.get('quality_score', 0.0),
                    'iteration_count': result.get('iteration_count', 0),
                    'num_docs': len(result.get('retrieved_docs', [])),
                    'elapsed_sec': query_elapsed,
                    'total_tokens': result.get('total_tokens', 0),
                    'estimated_cost_usd': result.get('estimated_cost_usd', 0.0),
                }

                profile_results.append(metrics)

                print(f"    ✓ Q={metrics['quality_score']:.3f}, "
                      f"Iter={metrics['iteration_count']}, "
                      f"Docs={metrics['num_docs']}, "
                      f"Time={metrics['elapsed_sec']:.1f}s")

            except Exception as e:
                print(f"    ✗ 오류: {str(e)}")
                profile_results.append({
                    'query_id': query_idx,
                    'query': query,
                    'error': str(e),
                })

        # 프로파일별 통계 계산
        successful = [r for r in profile_results if 'error' not in r]

        if successful:
            summary = {
                'total_queries': len(TEST_QUERIES),
                'successful': len(successful),
                'avg_quality': sum(r['quality_score'] for r in successful) / len(successful),
                'avg_iterations': sum(r['iteration_count'] for r in successful) / len(successful),
                'avg_docs': sum(r['num_docs'] for r in successful) / len(successful),
                'avg_time_sec': sum(r['elapsed_sec'] for r in successful) / len(successful),
                'total_tokens': sum(r.get('total_tokens', 0) for r in successful),
                'total_cost_usd': sum(r.get('estimated_cost_usd', 0.0) for r in successful),
            }

            all_results[profile_name] = {
                'description': ABLATION_PROFILES[profile_name]['description'],
                'feature_config': features,
                'results': profile_results,
                'summary': summary,
            }

            print(f"\n  📊 요약: Q={summary['avg_quality']:.3f}, "
                  f"Iter={summary['avg_iterations']:.1f}, "
                  f"Docs={summary['avg_docs']:.1f}, "
                  f"Time={summary['avg_time_sec']:.1f}s")
        else:
            print(f"\n  ⚠️ 모든 쿼리 실패")
            all_results[profile_name] = {
                'description': ABLATION_PROFILES[profile_name]['description'],
                'feature_config': features,
                'results': profile_results,
                'summary': {'error': '모든 쿼리 실패'},
            }

    # ============================================================
    # 전체 비교 테이블 출력
    # ============================================================
    print(f"\n\n{'='*80}")
    print("전체 비교 결과")
    print(f"{'='*80}")

    # 헤더
    print(f"{'프로파일':<30} {'품질':>8} {'반복':>6} {'문서':>6} {'시간(s)':>8} {'비용($)':>10}")
    print(f"{'-'*80}")

    # 각 프로파일 통계
    for profile_name in PROFILES_TO_TEST:
        if profile_name not in all_results:
            continue

        data = all_results[profile_name]
        if 'error' in data['summary']:
            print(f"{profile_name:<30} {'ERROR'}")
            continue

        s = data['summary']
        print(f"{profile_name:<30} "
              f"{s['avg_quality']:>8.3f} "
              f"{s['avg_iterations']:>6.1f} "
              f"{s['avg_docs']:>6.1f} "
              f"{s['avg_time_sec']:>8.1f} "
              f"{s['total_cost_usd']:>10.6f}")

    print(f"{'='*80}")

    # ============================================================
    # 결과 저장
    # ============================================================
    output_dir = project_root / "runs" / "ablation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"

    output_data = {
        'experiment_type': 'ablation_comparison',
        'timestamp': datetime.now().isoformat(),
        'profiles_tested': PROFILES_TO_TEST,
        'num_queries': len(TEST_QUERIES),
        'test_queries': TEST_QUERIES,
        'results': all_results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 결과 저장: {output_file}")

    # CSV 요약 저장
    csv_file = output_dir / f"summary_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8-sig') as f:
        f.write("Profile,Description,Avg_Quality,Avg_Iterations,Avg_Docs,Avg_Time_Sec,Total_Cost_USD\n")

        for profile_name in PROFILES_TO_TEST:
            if profile_name not in all_results:
                continue
            data = all_results[profile_name]
            if 'error' in data['summary']:
                continue

            s = data['summary']
            desc = data['description'].replace(',', ';')  # CSV 안전

            f.write(f"{profile_name},"
                   f"\"{desc}\","
                   f"{s['avg_quality']:.4f},"
                   f"{s['avg_iterations']:.2f},"
                   f"{s['avg_docs']:.2f},"
                   f"{s['avg_time_sec']:.2f},"
                   f"{s['total_cost_usd']:.6f}\n")

    print(f"   CSV 요약: {csv_file}")

    print("\n실험 완료! 🎉")
    print(f"총 {len(all_results)}개 프로파일 비교 완료")


if __name__ == "__main__":
    main()