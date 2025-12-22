"""
RAG 변형 비교 결과에 대한 RAGAS 평가 (피드백 반영)

RAGAS 3축 평가:
1. Faithfulness (근거 충실도)
2. Answer Relevancy (답변 관련성)
3. Context Precision (문맥 정확도)

Usage:
    python experiments/evaluate_rag_variants.py runs/rag_variants_comparison/comparison_P001_20251216_143022.json
"""
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.evaluation.ragas_metrics import calculate_ragas_metrics_full


def load_comparison_results(json_path: Path) -> Dict[str, Any]:
    """비교 실험 결과 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_variant(variant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """단일 RAG 변형에 대한 RAGAS 평가
    
    Args:
        variant_data: 변형 실험 결과 데이터
    
    Returns:
        턴별 RAGAS 점수 리스트
    """
    variant_name = variant_data['variant_name']
    turns = variant_data['turns']
    
    print(f"\n{'='*80}")
    print(f"[RAGAS 평가] {variant_name}")
    print(f"{'='*80}")
    
    turn_scores = []
    
    for turn_data in turns:
        if 'error' in turn_data:
            print(f"  Turn {turn_data['turn_id']}: 스킵 (오류 발생)")
            continue
        
        turn_id = turn_data['turn_id']
        question = turn_data['user_query']
        answer = turn_data['answer']
        contexts = turn_data['contexts']
        
        # 빈 contexts 처리
        if not contexts or all(not ctx.strip() for ctx in contexts):
            contexts = ["No context retrieved"]
        
        print(f"  Turn {turn_id}: {question[:50]}...")
        
        # RAGAS 메트릭 계산 (LLM as a Judge)
        ragas_scores = calculate_ragas_metrics_full(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=None  # ground_truth 없으면 context_recall은 계산 안 됨
        )
        
        if ragas_scores:
            turn_score = {
                'turn_id': turn_id,
                'user_query': question,
                'faithfulness': ragas_scores.get('faithfulness', 0.0),
                'answer_relevancy': ragas_scores.get('answer_relevancy', 0.0),
                'context_precision': ragas_scores.get('context_precision', 0.0),
                'context_relevancy': ragas_scores.get('context_relevancy', 0.0),
                'quality_score': turn_data.get('quality_score', 0.0),
                'iteration_count': turn_data.get('iteration_count', 0),
                'num_docs': turn_data.get('num_docs', 0),
                'elapsed_sec': turn_data.get('elapsed_sec', 0.0),
            }
            
            turn_scores.append(turn_score)
            
            print(f"    ✓ Faithfulness={turn_score['faithfulness']:.3f}, "
                  f"Relevancy={turn_score['answer_relevancy']:.3f}, "
                  f"Precision={turn_score['context_precision']:.3f}")
        else:
            print(f"    ✗ RAGAS 평가 실패")
    
    return turn_scores


def calculate_statistics(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """통계 계산 (평균, 표준편차)"""
    if not scores:
        return {}
    
    df = pd.DataFrame(scores)
    
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_relevancy']
    
    stats = {
        'count': len(scores),
    }
    
    for metric in metrics:
        if metric in df.columns:
            stats[f'{metric}_mean'] = float(df[metric].mean())
            stats[f'{metric}_std'] = float(df[metric].std())
            stats[f'{metric}_min'] = float(df[metric].min())
            stats[f'{metric}_max'] = float(df[metric].max())
    
    return stats


def compare_variants_statistical(
    variant_scores: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """변형 간 통계적 비교 (t-test)"""
    from scipy import stats as scipy_stats
    
    comparison_results = {}
    
    variants = list(variant_scores.keys())
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision']
    
    # 쌍별 비교
    for i, variant_a in enumerate(variants):
        for variant_b in variants[i+1:]:
            scores_a = variant_scores[variant_a]
            scores_b = variant_scores[variant_b]
            
            if not scores_a or not scores_b:
                continue
            
            df_a = pd.DataFrame(scores_a)
            df_b = pd.DataFrame(scores_b)
            
            pair_key = f"{variant_a}_vs_{variant_b}"
            comparison_results[pair_key] = {}
            
            for metric in metrics:
                if metric not in df_a.columns or metric not in df_b.columns:
                    continue
                
                values_a = df_a[metric].dropna()
                values_b = df_b[metric].dropna()
                
                if len(values_a) < 2 or len(values_b) < 2:
                    continue
                
                # t-test (양측 검정)
                t_stat, p_value = scipy_stats.ttest_ind(values_a, values_b)
                
                # 효과 크기 (Cohen's d)
                mean_a = values_a.mean()
                mean_b = values_b.mean()
                std_pooled = ((values_a.std() ** 2 + values_b.std() ** 2) / 2) ** 0.5
                cohens_d = (mean_a - mean_b) / std_pooled if std_pooled > 0 else 0
                
                comparison_results[pair_key][metric] = {
                    'mean_a': float(mean_a),
                    'mean_b': float(mean_b),
                    'diff': float(mean_a - mean_b),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': p_value < 0.05,
                }
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="RAG 변형 비교 결과 RAGAS 평가")
    parser.add_argument('comparison_file', type=str,
                        help='비교 실험 결과 JSON 파일 경로')
    
    args = parser.parse_args()
    
    comparison_file = Path(args.comparison_file)
    
    if not comparison_file.exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {comparison_file}")
        return
    
    print("=" * 80)
    print("RAG 변형 RAGAS 평가 (LLM as a Judge)")
    print("=" * 80)
    print(f"입력 파일: {comparison_file}")
    print("=" * 80)
    
    # 결과 로드
    data = load_comparison_results(comparison_file)
    
    patient_id = data.get('patient_id', 'unknown')
    patient_name = data.get('patient_name', '')
    results = data.get('results', {})
    
    print(f"환자: {patient_name} ({patient_id})")
    print(f"변형 수: {len(results)}")
    
    # 각 변형 평가
    all_scores = {}
    
    for variant_name, variant_data in results.items():
        scores = evaluate_variant(variant_data)
        all_scores[variant_name] = scores
    
    # ============================================================
    # 통계 계산
    # ============================================================
    print(f"\n\n{'='*80}")
    print("RAGAS 메트릭 통계")
    print(f"{'='*80}")
    
    all_stats = {}
    
    for variant_name, scores in all_scores.items():
        stats = calculate_statistics(scores)
        all_stats[variant_name] = stats
        
        if stats:
            print(f"\n[{variant_name}]")
            print(f"  샘플 수: {stats['count']}")
            print(f"  Faithfulness:      {stats.get('faithfulness_mean', 0):.3f} ± {stats.get('faithfulness_std', 0):.3f}")
            print(f"  Answer Relevancy:  {stats.get('answer_relevancy_mean', 0):.3f} ± {stats.get('answer_relevancy_std', 0):.3f}")
            print(f"  Context Precision: {stats.get('context_precision_mean', 0):.3f} ± {stats.get('context_precision_std', 0):.3f}")
    
    # ============================================================
    # 통계적 비교 (t-test)
    # ============================================================
    print(f"\n\n{'='*80}")
    print("통계적 유의성 검정 (t-test)")
    print(f"{'='*80}")
    
    comparison_stats = compare_variants_statistical(all_scores)
    
    for pair_key, pair_stats in comparison_stats.items():
        print(f"\n[{pair_key}]")
        
        for metric, metric_stats in pair_stats.items():
            sig_marker = "***" if metric_stats['significant'] else ""
            
            print(f"  {metric}:")
            print(f"    Δ = {metric_stats['diff']:+.3f} "
                  f"(p={metric_stats['p_value']:.4f}, "
                  f"d={metric_stats['cohens_d']:.2f}) {sig_marker}")
    
    # ============================================================
    # 비교 테이블 출력
    # ============================================================
    print(f"\n\n{'='*80}")
    print("RAGAS 메트릭 비교 테이블")
    print(f"{'='*80}")
    
    # 헤더
    print(f"{'변형':<20} {'Faithfulness':>14} {'Relevancy':>14} {'Precision':>14}")
    print(f"{'-'*80}")
    
    # 각 변형 통계
    for variant_name in results.keys():
        if variant_name not in all_stats:
            continue
        
        stats = all_stats[variant_name]
        
        if stats:
            print(f"{variant_name:<20} "
                  f"{stats.get('faithfulness_mean', 0):>8.3f}±{stats.get('faithfulness_std', 0):.3f} "
                  f"{stats.get('answer_relevancy_mean', 0):>8.3f}±{stats.get('answer_relevancy_std', 0):.3f} "
                  f"{stats.get('context_precision_mean', 0):>8.3f}±{stats.get('context_precision_std', 0):.3f}")
    
    print(f"{'='*80}")
    
    # ============================================================
    # 결과 저장
    # ============================================================
    output_dir = comparison_file.parent / "ragas_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ragas_{patient_id}_{timestamp}.json"
    
    output_data = {
        'experiment_type': 'ragas_evaluation',
        'timestamp': datetime.now().isoformat(),
        'source_file': str(comparison_file),
        'patient_id': patient_id,
        'patient_name': patient_name,
        'variant_scores': all_scores,
        'variant_statistics': all_stats,
        'statistical_comparison': comparison_stats,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ RAGAS 평가 결과 저장: {output_file}")
    
    # CSV 저장 (논문/보고서용)
    csv_file = output_dir / f"ragas_summary_{patient_id}_{timestamp}.csv"
    
    with open(csv_file, 'w', encoding='utf-8-sig') as f:
        f.write("Variant,Faithfulness_Mean,Faithfulness_Std,Relevancy_Mean,Relevancy_Std,Precision_Mean,Precision_Std\n")
        
        for variant_name in results.keys():
            if variant_name not in all_stats:
                continue
            
            stats = all_stats[variant_name]
            
            if stats:
                f.write(f"{variant_name},"
                       f"{stats.get('faithfulness_mean', 0):.4f},"
                       f"{stats.get('faithfulness_std', 0):.4f},"
                       f"{stats.get('answer_relevancy_mean', 0):.4f},"
                       f"{stats.get('answer_relevancy_std', 0):.4f},"
                       f"{stats.get('context_precision_mean', 0):.4f},"
                       f"{stats.get('context_precision_std', 0):.4f}\n")
    
    print(f"   CSV 요약: {csv_file}")
    
    print("\n평가 완료! 🎉")
    print(f"총 {len(all_scores)}개 변형 평가 완료")


if __name__ == "__main__":
    main()

