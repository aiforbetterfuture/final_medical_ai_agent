"""
Ablation 결과 분석 및 시각화

저장된 ablation 실험 결과를 분석하고 차트 생성

Usage:
    python experiments/analyze_ablation_results.py [results_file.json]
"""
import json
import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(file_path):
    """결과 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_comparison_results(data):
    """비교 실험 결과 분석"""
    print("=" * 80)
    print("Ablation Comparison 결과 분석")
    print("=" * 80)
    print(f"실험 시간: {data['timestamp']}")
    print(f"프로파일 수: {len(data['profiles_tested'])}")
    print(f"쿼리 수: {data['num_queries']}")
    print("=" * 80)

    # DataFrame 생성
    rows = []
    for profile_name, profile_data in data['results'].items():
        if 'error' in profile_data['summary']:
            continue

        summary = profile_data['summary']
        rows.append({
            'Profile': profile_name,
            'Description': profile_data.get('description', ''),
            'Avg Quality': summary['avg_quality'],
            'Avg Iterations': summary['avg_iterations'],
            'Avg Docs': summary['avg_docs'],
            'Avg Time (s)': summary['avg_time_sec'],
            'Total Tokens': summary.get('total_tokens', 0),
            'Total Cost ($)': summary.get('total_cost_usd', 0.0),
        })

    if not rows:
        print("❌ 분석할 데이터 없음")
        return

    df = pd.DataFrame(rows)

    # 통계 출력
    print("\n📊 프로파일별 성능 비교")
    print(df.to_string(index=False))

    # 순위 계산
    print("\n\n🏆 순위 분석")
    print("-" * 80)

    print("\n1. 품질 점수 (Quality) - 높을수록 좋음")
    quality_rank = df.sort_values('Avg Quality', ascending=False)
    for i, row in quality_rank.iterrows():
        print(f"  {quality_rank.index.get_loc(i)+1}위: {row['Profile']:<30} {row['Avg Quality']:.4f}")

    print("\n2. 실행 시간 (Time) - 낮을수록 좋음")
    time_rank = df.sort_values('Avg Time (s)', ascending=True)
    for i, row in time_rank.iterrows():
        print(f"  {time_rank.index.get_loc(i)+1}위: {row['Profile']:<30} {row['Avg Time (s)']:.2f}초")

    print("\n3. 비용 (Cost) - 낮을수록 좋음")
    cost_rank = df.sort_values('Total Cost ($)', ascending=True)
    for i, row in cost_rank.iterrows():
        print(f"  {cost_rank.index.get_loc(i)+1}위: {row['Profile']:<30} ${row['Total Cost ($)']:.6f}")

    # 효율성 분석 (품질/시간 비율)
    df['Quality/Time'] = df['Avg Quality'] / df['Avg Time (s)']
    df['Quality/Cost'] = df['Avg Quality'] / (df['Total Cost ($)'] + 0.0001)  # 0 방지

    print("\n4. 효율성 (Quality/Time) - 높을수록 좋음")
    eff_rank = df.sort_values('Quality/Time', ascending=False)
    for i, row in eff_rank.iterrows():
        print(f"  {eff_rank.index.get_loc(i)+1}위: {row['Profile']:<30} {row['Quality/Time']:.4f}")

    print("\n5. 비용 효율성 (Quality/Cost) - 높을수록 좋음")
    cost_eff_rank = df.sort_values('Quality/Cost', ascending=False)
    for i, row in cost_eff_rank.iterrows():
        print(f"  {cost_eff_rank.index.get_loc(i)+1}위: {row['Profile']:<30} {row['Quality/Cost']:.1f}")

    # 개선 효과 분석 (baseline 대비)
    if 'baseline' in df['Profile'].values:
        baseline = df[df['Profile'] == 'baseline'].iloc[0]
        print("\n\n📈 Baseline 대비 개선율")
        print("-" * 80)

        for i, row in df.iterrows():
            if row['Profile'] == 'baseline':
                continue

            quality_improve = ((row['Avg Quality'] - baseline['Avg Quality']) / baseline['Avg Quality']) * 100
            time_change = ((row['Avg Time (s)'] - baseline['Avg Time (s)']) / baseline['Avg Time (s)']) * 100
            cost_change = ((row['Total Cost ($)'] - baseline['Total Cost ($)']) / (baseline['Total Cost ($)'] + 0.0001)) * 100

            print(f"\n{row['Profile']}:")
            print(f"  품질: {quality_improve:+.1f}%")
            print(f"  시간: {time_change:+.1f}%")
            print(f"  비용: {cost_change:+.1f}%")

    return df


def analyze_single_results(data):
    """단일 실험 결과 분석"""
    print("=" * 80)
    print(f"Ablation Test: {data['ablation_name']}")
    print("=" * 80)
    print(f"실험 시간: {data['timestamp']}")
    print(f"쿼리 수: {data['num_queries']}")
    print("=" * 80)

    # Feature config 출력
    print("\n⚙️ Feature Configuration:")
    for key, value in data['feature_config'].items():
        print(f"  {key}: {value}")

    # Summary 출력
    print("\n📊 통계 요약:")
    summary = data['summary']
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 쿼리별 결과
    print("\n📝 쿼리별 결과:")
    results = data['results']
    successful = [r for r in results if 'error' not in r]

    if successful:
        df = pd.DataFrame(successful)
        print(df[['query_id', 'quality_score', 'iteration_count', 'num_docs', 'elapsed_sec']].to_string(index=False))
    else:
        print("  ⚠️ 성공한 쿼리 없음")


def main():
    # 결과 파일 경로 확인
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        # 가장 최근 comparison 결과 찾기
        comparison_dir = project_root / "runs" / "ablation_comparison"
        if comparison_dir.exists():
            json_files = list(comparison_dir.glob("comparison_*.json"))
            if json_files:
                results_file = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📂 가장 최근 파일 사용: {results_file.name}\n")
            else:
                print("❌ 결과 파일을 찾을 수 없습니다.")
                print("Usage: python experiments/analyze_ablation_results.py [results_file.json]")
                return
        else:
            print("❌ ablation_comparison 디렉토리가 없습니다.")
            return

    if not results_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {results_file}")
        return

    # 결과 로드
    data = load_results(results_file)

    # 타입에 따라 분석
    if 'profiles_tested' in data:
        # Comparison 결과
        df = analyze_comparison_results(data)

        # 차트 생성 시도 (matplotlib 있으면)
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # GUI 없이 저장만

            if df is not None and not df.empty:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # 1. 품질 비교
                axes[0, 0].barh(df['Profile'], df['Avg Quality'], color='skyblue')
                axes[0, 0].set_xlabel('Average Quality Score')
                axes[0, 0].set_title('Quality Comparison')
                axes[0, 0].grid(axis='x', alpha=0.3)

                # 2. 반복 횟수
                axes[0, 1].barh(df['Profile'], df['Avg Iterations'], color='lightcoral')
                axes[0, 1].set_xlabel('Average Iterations')
                axes[0, 1].set_title('Self-Refine Iterations')
                axes[0, 1].grid(axis='x', alpha=0.3)

                # 3. 실행 시간
                axes[1, 0].barh(df['Profile'], df['Avg Time (s)'], color='lightgreen')
                axes[1, 0].set_xlabel('Average Time (seconds)')
                axes[1, 0].set_title('Execution Time')
                axes[1, 0].grid(axis='x', alpha=0.3)

                # 4. 비용
                axes[1, 1].barh(df['Profile'], df['Total Cost ($)'], color='gold')
                axes[1, 1].set_xlabel('Total Cost (USD)')
                axes[1, 1].set_title('API Cost')
                axes[1, 1].grid(axis='x', alpha=0.3)

                plt.tight_layout()

                chart_file = results_file.parent / f"charts_{results_file.stem}.png"
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                print(f"\n📊 차트 저장: {chart_file}")

        except ImportError:
            print("\n⚠️ matplotlib이 설치되지 않아 차트를 생성할 수 없습니다.")
            print("   설치: pip install matplotlib")

    else:
        # Single 결과
        analyze_single_results(data)

    print("\n" + "=" * 80)
    print("분석 완료!")


if __name__ == "__main__":
    main()