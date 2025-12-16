"""
단일 Ablation 테스트 실행 스크립트

Usage:
    python experiments/run_ablation_single.py

커스터마이징:
    - ABLATION_NAME: 실험 이름
    - FEATURE_CONFIG: 테스트할 feature flags
    - TEST_QUERIES: 테스트 쿼리 목록
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

# ============================================================
# 설정 섹션 (여기를 수정하세요)
# ============================================================

ABLATION_NAME = "self_refine_off"  # 실험 이름 (파일명에 사용됨)

# Feature flags 설정 (테스트하고 싶은 설정)
FEATURE_CONFIG = {
    'self_refine_enabled': False,  # ⭐ 주요 테스트 변수
    'retrieval_mode': 'hybrid',
    'active_retrieval_enabled': False,
    'response_cache_enabled': False,  # 캐시 비활성화 (순수 성능 측정)
}

# 테스트 쿼리 목록
TEST_QUERIES = [
    "당뇨병 환자에게 메트포르민의 부작용은 무엇인가요?",
    "고혈압 환자의 식이요법은 어떻게 해야 하나요?",
    "아스피린을 복용하는 환자가 피해야 할 음식은?",
    "임신 중 복용 가능한 진통제는 무엇인가요?",
    "간 질환 환자에게 금기인 약물은?",
]

# ============================================================
# 실행 섹션
# ============================================================

def main():
    print("=" * 80)
    print(f"Ablation Test: {ABLATION_NAME}")
    print("=" * 80)
    print(f"Feature Config: {json.dumps(FEATURE_CONFIG, indent=2)}")
    print(f"Test Queries: {len(TEST_QUERIES)}개")
    print("=" * 80)
    print()

    results = []
    total_start_time = time.time()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] 실행 중: {query[:50]}...")

        query_start_time = time.time()

        try:
            result = run_agent(
                user_text=query,
                mode="ai_agent",
                feature_overrides=FEATURE_CONFIG,
                return_state=True
            )

            query_elapsed = time.time() - query_start_time

            # 메트릭 추출
            metrics = {
                'query_id': i,
                'query': query,
                'answer': result.get('answer', ''),
                'quality_score': result.get('quality_score', 0.0),
                'iteration_count': result.get('iteration_count', 0),
                'num_docs': len(result.get('retrieved_docs', [])),
                'cache_hit': result.get('cache_hit', False),
                'elapsed_sec': query_elapsed,
                'total_tokens': result.get('total_tokens', 0),
                'estimated_cost_usd': result.get('estimated_cost_usd', 0.0),
            }

            results.append(metrics)

            print(f"  ✓ 품질: {metrics['quality_score']:.3f}, "
                  f"반복: {metrics['iteration_count']}, "
                  f"문서: {metrics['num_docs']}, "
                  f"시간: {metrics['elapsed_sec']:.1f}s")

        except Exception as e:
            print(f"  ✗ 오류 발생: {str(e)}")
            results.append({
                'query_id': i,
                'query': query,
                'error': str(e),
                'elapsed_sec': time.time() - query_start_time,
            })

        print()

    total_elapsed = time.time() - total_start_time

    # ============================================================
    # 통계 요약
    # ============================================================
    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        summary = {
            'total_queries': len(TEST_QUERIES),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'avg_quality': sum(r['quality_score'] for r in successful_results) / len(successful_results),
            'avg_iterations': sum(r['iteration_count'] for r in successful_results) / len(successful_results),
            'avg_docs': sum(r['num_docs'] for r in successful_results) / len(successful_results),
            'avg_time_sec': sum(r['elapsed_sec'] for r in successful_results) / len(successful_results),
            'total_time_sec': total_elapsed,
            'total_tokens': sum(r.get('total_tokens', 0) for r in successful_results),
            'total_cost_usd': sum(r.get('estimated_cost_usd', 0.0) for r in successful_results),
            'cache_hit_rate': sum(r['cache_hit'] for r in successful_results) / len(successful_results) if successful_results else 0.0,
        }

        print("=" * 80)
        print("통계 요약")
        print("=" * 80)
        print(f"성공/전체: {summary['successful']}/{summary['total_queries']}")
        print(f"평균 품질 점수: {summary['avg_quality']:.3f}")
        print(f"평균 반복 횟수: {summary['avg_iterations']:.2f}")
        print(f"평균 검색 문서: {summary['avg_docs']:.1f}")
        print(f"평균 실행 시간: {summary['avg_time_sec']:.2f}초")
        print(f"총 실행 시간: {summary['total_time_sec']:.1f}초")
        print(f"총 토큰 사용: {summary['total_tokens']:,}")
        print(f"총 예상 비용: ${summary['total_cost_usd']:.4f}")
        print(f"캐시 히트율: {summary['cache_hit_rate']:.1%}")
        print("=" * 80)
    else:
        summary = {'error': '모든 쿼리 실패'}
        print("\n⚠️ 모든 쿼리가 실패했습니다.")

    # ============================================================
    # 결과 저장
    # ============================================================
    output_dir = project_root / "runs" / f"ablation_{ABLATION_NAME}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    output_data = {
        'ablation_name': ABLATION_NAME,
        'feature_config': FEATURE_CONFIG,
        'timestamp': datetime.now().isoformat(),
        'num_queries': len(TEST_QUERIES),
        'results': results,
        'summary': summary,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 결과 저장됨: {output_file}")
    print(f"   디렉토리: {output_dir}")

    # 간단한 CSV도 저장 (엑셀로 열기 쉽게)
    if successful_results:
        csv_file = output_dir / f"results_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8-sig') as f:
            # Header
            f.write("ID,Query,Quality,Iterations,Docs,Time(s),Tokens,Cost($)\n")
            # Rows
            for r in successful_results:
                f.write(f"{r['query_id']},"
                       f"\"{r['query']}\","
                       f"{r['quality_score']:.3f},"
                       f"{r['iteration_count']},"
                       f"{r['num_docs']},"
                       f"{r['elapsed_sec']:.2f},"
                       f"{r.get('total_tokens', 0)},"
                       f"{r.get('estimated_cost_usd', 0.0):.6f}\n")
        print(f"   CSV 파일: {csv_file}")

    print("\n실험 완료! 🎉")


if __name__ == "__main__":
    main()