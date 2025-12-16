"""
LLM vs RAG 비교 평가 러너

목적:
- 저장된 대화 로그를 읽어 RAGAS 평가 수행
- 3가지 시스템 간 통계적 비교
- 결과 시각화 및 저장

사용법:
    python experiments/evaluate_llm_vs_rag.py --log-dir experiments/comparison_logs/20251216_llm_vs_rag
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.evaluation.ragas_metrics import calculate_ragas_metrics_full

# ============================================================
# 대화 로그 읽기
# ============================================================

def read_jsonl(file_path: Path) -> List[Dict]:
    """JSONL 파일 읽기"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_comparison_logs(log_dir: Path) -> Dict[str, List[Dict]]:
    """
    비교 로그 로드
    
    Args:
        log_dir: 로그 디렉토리 (experiments/comparison_logs/{experiment_id})
    
    Returns:
        {variant_name: [turn_logs]}
    """
    variants = ['llm_only', 'basic_rag', 'corrective_rag']
    logs = {}
    
    for variant_name in variants:
        variant_dir = log_dir / variant_name
        if not variant_dir.exists():
            print(f"경고: {variant_name} 디렉토리를 찾을 수 없습니다: {variant_dir}")
            continue
        
        # 모든 환자 로그 읽기
        variant_logs = []
        for log_file in variant_dir.glob('*.jsonl'):
            patient_logs = read_jsonl(log_file)
            variant_logs.extend(patient_logs)
        
        logs[variant_name] = variant_logs
        print(f"[{variant_name}] {len(variant_logs)}개 턴 로드")
    
    return logs


# ============================================================
# RAGAS 평가
# ============================================================

def evaluate_comparison_logs(comparison_logs: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    저장된 대화 로그를 읽어 RAGAS 평가 수행
    
    Args:
        comparison_logs: {variant_name: [turn_logs]}
    
    Returns:
        {variant_name: {metrics}}
    """
    results = {}
    
    for variant_name, turn_logs in comparison_logs.items():
        print(f"\n[{variant_name.upper()}] RAGAS 평가 중...")
        
        # 각 턴별 RAGAS 메트릭 계산
        variant_metrics = []
        
        for turn_idx, turn_data in enumerate(turn_logs):
            print(f"  턴 {turn_idx + 1}/{len(turn_logs)}: {turn_data['question'][:50]}...")
            
            try:
                metrics = calculate_ragas_metrics_full(
                    question=turn_data['question'],
                    answer=turn_data['answer'],
                    contexts=turn_data['contexts']
                )
                
                if metrics:
                    variant_metrics.append(metrics)
                    print(f"    ✓ 완료: faithfulness={metrics.get('faithfulness', 0):.3f}")
                else:
                    print(f"    ✗ 실패: 메트릭 계산 불가")
            
            except Exception as e:
                print(f"    ✗ 오류: {e}")
        
        # 평균 계산
        if variant_metrics:
            results[variant_name] = {
                'faithfulness_avg': np.mean([m.get('faithfulness', 0) for m in variant_metrics]),
                'faithfulness_std': np.std([m.get('faithfulness', 0) for m in variant_metrics]),
                'answer_relevancy_avg': np.mean([m.get('answer_relevancy', 0) for m in variant_metrics]),
                'answer_relevancy_std': np.std([m.get('answer_relevancy', 0) for m in variant_metrics]),
                'context_precision_avg': np.mean([m.get('context_precision', 0) for m in variant_metrics]),
                'context_precision_std': np.std([m.get('context_precision', 0) for m in variant_metrics]),
                'context_relevancy_avg': np.mean([m.get('context_relevancy', 0) for m in variant_metrics]),
                'context_relevancy_std': np.std([m.get('context_relevancy', 0) for m in variant_metrics]),
                'per_turn_metrics': variant_metrics,
                'num_turns': len(variant_metrics)
            }
            
            print(f"  평균 faithfulness: {results[variant_name]['faithfulness_avg']:.3f}")
            print(f"  평균 answer_relevancy: {results[variant_name]['answer_relevancy_avg']:.3f}")
        else:
            print(f"  ✗ 평가 실패: 유효한 메트릭 없음")
            results[variant_name] = None
    
    return results


# ============================================================
# 통계 분석
# ============================================================

def statistical_comparison(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    3가지 시스템 간 통계적 유의성 검정
    
    Args:
        results: {variant_name: {metrics}}
    
    Returns:
        통계 분석 결과
    """
    print(f"\n{'='*80}")
    print("통계 분석")
    print(f"{'='*80}\n")
    
    comparisons = {}
    
    # LLM vs Basic RAG
    if 'llm_only' in results and 'basic_rag' in results:
        if results['llm_only'] and results['basic_rag']:
            print("[LLM Only vs Basic RAG]")
            
            llm_faithfulness = [m.get('faithfulness', 0) for m in results['llm_only']['per_turn_metrics']]
            rag_faithfulness = [m.get('faithfulness', 0) for m in results['basic_rag']['per_turn_metrics']]
            
            t_stat, p_value = stats.ttest_ind(llm_faithfulness, rag_faithfulness)
            
            comparisons['llm_vs_basic_rag'] = {
                'faithfulness': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            }
            
            print(f"  Faithfulness: t={t_stat:.3f}, p={p_value:.4f} {'✓ 유의함' if p_value < 0.05 else '✗ 유의하지 않음'}")
    
    # Basic RAG vs Corrective RAG
    if 'basic_rag' in results and 'corrective_rag' in results:
        if results['basic_rag'] and results['corrective_rag']:
            print("\n[Basic RAG vs Corrective RAG]")
            
            basic_faithfulness = [m.get('faithfulness', 0) for m in results['basic_rag']['per_turn_metrics']]
            crag_faithfulness = [m.get('faithfulness', 0) for m in results['corrective_rag']['per_turn_metrics']]
            
            t_stat2, p_value2 = stats.ttest_ind(basic_faithfulness, crag_faithfulness)
            
            comparisons['basic_vs_corrective'] = {
                'faithfulness': {
                    't_statistic': t_stat2,
                    'p_value': p_value2,
                    'significant': p_value2 < 0.05
                }
            }
            
            print(f"  Faithfulness: t={t_stat2:.3f}, p={p_value2:.4f} {'✓ 유의함' if p_value2 < 0.05 else '✗ 유의하지 않음'}")
    
    return comparisons


# ============================================================
# 결과 저장
# ============================================================

def save_results(
    evaluation_results: Dict[str, Dict],
    statistical_results: Dict[str, Any],
    output_dir: Path
):
    """결과 저장"""
    # 평가 결과 저장
    eval_file = output_dir / 'evaluation_results.json'
    
    # per_turn_metrics는 저장하지 않음 (용량 절약)
    eval_results_summary = {}
    for variant_name, metrics in evaluation_results.items():
        if metrics:
            eval_results_summary[variant_name] = {
                k: v for k, v in metrics.items() if k != 'per_turn_metrics'
            }
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n평가 결과 저장: {eval_file}")
    
    # 통계 결과 저장
    stats_file = output_dir / 'statistical_results.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistical_results, f, ensure_ascii=False, indent=2)
    
    print(f"통계 결과 저장: {stats_file}")


def print_summary(evaluation_results: Dict[str, Dict]):
    """결과 요약 출력"""
    print(f"\n{'='*80}")
    print("평가 요약")
    print(f"{'='*80}\n")
    
    # 테이블 형식 출력
    print(f"{'Variant':<20} {'Faithfulness':<15} {'Answer Relevancy':<18} {'Context Precision':<18}")
    print(f"{'-'*80}")
    
    for variant_name, metrics in evaluation_results.items():
        if metrics:
            print(f"{variant_name:<20} "
                  f"{metrics['faithfulness_avg']:.3f} ± {metrics['faithfulness_std']:.3f}    "
                  f"{metrics['answer_relevancy_avg']:.3f} ± {metrics['answer_relevancy_std']:.3f}    "
                  f"{metrics['context_precision_avg']:.3f} ± {metrics['context_precision_std']:.3f}")
        else:
            print(f"{variant_name:<20} N/A")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description='LLM vs RAG 비교 평가')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='로그 디렉토리 (experiments/comparison_logs/{experiment_id})')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"오류: 로그 디렉토리를 찾을 수 없습니다: {log_dir}")
        return
    
    print(f"{'='*80}")
    print("LLM vs RAG 비교 평가")
    print(f"{'='*80}")
    print(f"로그 디렉토리: {log_dir}\n")
    
    # 1. 대화 로그 로드
    comparison_logs = load_comparison_logs(log_dir)
    
    if not comparison_logs:
        print("오류: 로드된 로그가 없습니다.")
        return
    
    # 2. RAGAS 평가
    evaluation_results = evaluate_comparison_logs(comparison_logs)
    
    # 3. 통계 분석
    statistical_results = statistical_comparison(evaluation_results)
    
    # 4. 결과 저장
    save_results(evaluation_results, statistical_results, log_dir)
    
    # 5. 요약 출력
    print_summary(evaluation_results)
    
    print(f"\n{'='*80}")
    print("✓ 평가 완료!")
    print(f"{'='*80}")
    print(f"\n결과 위치: {log_dir}")


if __name__ == '__main__':
    main()

