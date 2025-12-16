"""
RAGAS 수동 설문조사 결과 분석

목적:
- Markdown 체크박스 형식의 설문 결과 파싱
- RAGAS 메트릭 형식으로 변환
- 통계 분석 및 시각화

사용법:
    python experiments/survey/analyze_survey_results.py --survey-dir experiments/survey/forms
"""

import re
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from scipy import stats

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 설문 결과 파싱
# ============================================================

def parse_survey_file(survey_file: Path) -> List[Dict]:
    """
    설문 결과 파싱 (Markdown 체크박스 기반)
    
    Args:
        survey_file: 설문 파일 경로
    
    Returns:
        [{'turn': 1, 'faithfulness': 4, 'answer_relevancy': 5, 'context_precision': 3}, ...]
    """
    with open(survey_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 턴별로 분할
    turns = re.split(r'# RAGAS 평가 설문 \(턴 \d+\)', content)[1:]
    
    results = []
    
    for turn_idx, turn_content in enumerate(turns):
        turn_num = turn_idx + 1
        
        # 각 메트릭별 점수 추출
        faithfulness = extract_score(turn_content, '### 1. Faithfulness')
        answer_relevancy = extract_score(turn_content, '### 2. Answer Relevancy')
        context_precision = extract_score(turn_content, '### 3. Context Precision')
        
        if faithfulness is not None or answer_relevancy is not None or context_precision is not None:
            results.append({
                'turn': turn_num,
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision
            })
    
    return results


def extract_score(text: str, section_header: str) -> float:
    """
    특정 섹션에서 체크된 점수 추출
    
    Args:
        text: 설문 텍스트
        section_header: 섹션 헤더 (예: '### 1. Faithfulness')
    
    Returns:
        점수 (1-5) 또는 None
    """
    # 섹션 추출
    section_pattern = re.escape(section_header) + r'(.*?)(?=###|\Z)'
    section_match = re.search(section_pattern, text, re.DOTALL)
    
    if not section_match:
        return None
    
    section_text = section_match.group(1)
    
    # [x] 표시된 항목 찾기
    checked_pattern = r'\[x\]\s*(\d+)점'
    checked_match = re.search(checked_pattern, section_text, re.IGNORECASE)
    
    if checked_match:
        return int(checked_match.group(1))
    
    return None


# ============================================================
# 메트릭 계산
# ============================================================

def calculate_survey_metrics(survey_results: List[Dict]) -> Dict:
    """
    설문 결과를 RAGAS 메트릭 형식으로 변환
    
    Args:
        survey_results: 파싱된 설문 결과
    
    Returns:
        {
            'faithfulness_avg': 0.75,
            'answer_relevancy_avg': 0.80,
            'context_precision_avg': 0.70,
            ...
        }
    """
    # 유효한 점수만 추출
    faithfulness_scores = [r['faithfulness'] for r in survey_results if r['faithfulness'] is not None]
    answer_relevancy_scores = [r['answer_relevancy'] for r in survey_results if r['answer_relevancy'] is not None]
    context_precision_scores = [r['context_precision'] for r in survey_results if r['context_precision'] is not None]
    
    # 5점 만점을 1.0 만점으로 변환
    metrics = {}
    
    if faithfulness_scores:
        metrics['faithfulness_avg'] = np.mean(faithfulness_scores) / 5.0
        metrics['faithfulness_std'] = np.std(faithfulness_scores) / 5.0
    
    if answer_relevancy_scores:
        metrics['answer_relevancy_avg'] = np.mean(answer_relevancy_scores) / 5.0
        metrics['answer_relevancy_std'] = np.std(answer_relevancy_scores) / 5.0
    
    if context_precision_scores:
        metrics['context_precision_avg'] = np.mean(context_precision_scores) / 5.0
        metrics['context_precision_std'] = np.std(context_precision_scores) / 5.0
    
    metrics['num_turns'] = len(survey_results)
    metrics['num_valid_faithfulness'] = len(faithfulness_scores)
    metrics['num_valid_answer_relevancy'] = len(answer_relevancy_scores)
    metrics['num_valid_context_precision'] = len(context_precision_scores)
    
    return metrics


# ============================================================
# 통계 분석
# ============================================================

def statistical_comparison(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    시스템 간 통계적 비교
    
    Args:
        results: {variant_name: metrics}
    
    Returns:
        통계 분석 결과
    """
    print(f"\n{'='*80}")
    print("통계 분석")
    print(f"{'='*80}\n")
    
    comparisons = {}
    
    variants = list(results.keys())
    
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            v1, v2 = variants[i], variants[j]
            
            if v1 in results and v2 in results:
                print(f"[{v1.upper()} vs {v2.upper()}]")
                
                # Faithfulness 비교 (간단한 평균 차이)
                if 'faithfulness_avg' in results[v1] and 'faithfulness_avg' in results[v2]:
                    diff = results[v2]['faithfulness_avg'] - results[v1]['faithfulness_avg']
                    print(f"  Faithfulness 차이: {diff:+.3f}")
                
                # Answer Relevancy 비교
                if 'answer_relevancy_avg' in results[v1] and 'answer_relevancy_avg' in results[v2]:
                    diff = results[v2]['answer_relevancy_avg'] - results[v1]['answer_relevancy_avg']
                    print(f"  Answer Relevancy 차이: {diff:+.3f}")
                
                print()
    
    return comparisons


# ============================================================
# 결과 저장
# ============================================================

def save_results(results: Dict[str, Dict], output_file: Path):
    """결과 저장"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {output_file}")


def print_summary(results: Dict[str, Dict]):
    """결과 요약 출력"""
    print(f"\n{'='*80}")
    print("평가 요약")
    print(f"{'='*80}\n")
    
    # 테이블 형식 출력
    print(f"{'Variant':<20} {'Faithfulness':<15} {'Answer Relevancy':<18} {'Context Precision':<18}")
    print(f"{'-'*80}")
    
    for variant_name, metrics in results.items():
        if metrics:
            faith_str = f"{metrics.get('faithfulness_avg', 0):.3f} ± {metrics.get('faithfulness_std', 0):.3f}" if 'faithfulness_avg' in metrics else "N/A"
            ans_str = f"{metrics.get('answer_relevancy_avg', 0):.3f} ± {metrics.get('answer_relevancy_std', 0):.3f}" if 'answer_relevancy_avg' in metrics else "N/A"
            ctx_str = f"{metrics.get('context_precision_avg', 0):.3f} ± {metrics.get('context_precision_std', 0):.3f}" if 'context_precision_avg' in metrics else "N/A"
            
            print(f"{variant_name:<20} {faith_str:<15} {ans_str:<18} {ctx_str:<18}")
        else:
            print(f"{variant_name:<20} N/A")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description='RAGAS 수동 설문조사 결과 분석')
    parser.add_argument('--survey-dir', type=str, required=True,
                        help='설문지 디렉토리')
    parser.add_argument('--output-file', type=str,
                        default='experiments/survey/survey_results.json',
                        help='결과 파일')
    
    args = parser.parse_args()
    
    survey_dir = Path(args.survey_dir)
    output_file = Path(args.output_file)
    
    if not survey_dir.exists():
        print(f"오류: 설문지 디렉토리를 찾을 수 없습니다: {survey_dir}")
        return
    
    print(f"{'='*80}")
    print("RAGAS 수동 설문조사 결과 분석")
    print(f"{'='*80}")
    print(f"설문지 디렉토리: {survey_dir}\n")
    
    # 각 변형별 설문 결과 파싱
    results = {}
    
    for survey_file in survey_dir.glob('survey_*.md'):
        variant_name = survey_file.stem.replace('survey_', '')
        
        print(f"[{variant_name.upper()}] 파싱 중...")
        
        survey_results = parse_survey_file(survey_file)
        metrics = calculate_survey_metrics(survey_results)
        
        results[variant_name] = metrics
        
        print(f"  유효한 턴 수: {metrics['num_valid_faithfulness']}/{metrics['num_turns']}")
    
    # 통계 분석
    statistical_comparison(results)
    
    # 결과 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_file)
    
    # 요약 출력
    print_summary(results)
    
    print(f"\n{'='*80}")
    print("✓ 분석 완료!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

