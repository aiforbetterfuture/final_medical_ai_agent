"""
LLM vs RAG 시스템 비교 실험 러너

목적:
- LLM 단독 vs Basic RAG vs Corrective RAG 3가지 시스템 비교
- 동일한 환자/질문으로 대화 로그 생성
- RAGAS 평가를 위한 데이터 준비

실험 설정:
- LLM Only: mode='llm' (검색 없음)
- Basic RAG: mode='ai_agent', refine_strategy='basic_rag'
- Corrective RAG: mode='ai_agent', refine_strategy='corrective_rag'
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.graph import run_agent

# ============================================================
# 실험 변형 정의
# ============================================================

EXPERIMENT_VARIANTS = {
    'llm_only': {
        'mode': 'llm',
        'feature_overrides': {},
        'description': 'Pure LLM without retrieval'
    },
    'basic_rag': {
        'mode': 'ai_agent',
        'feature_overrides': {
            'refine_strategy': 'basic_rag',
            'self_refine_enabled': False,
            'quality_check_enabled': False,
            'response_cache_enabled': False
        },
        'description': 'Basic RAG (1-shot retrieval)'
    },
    'corrective_rag': {
        'mode': 'ai_agent',
        'feature_overrides': {
            'refine_strategy': 'corrective_rag',
            'self_refine_enabled': True,
            'llm_based_quality_check': True,
            'dynamic_query_rewrite': True,
            'duplicate_detection': True,
            'progress_monitoring': True,
            'max_refine_iterations': 2,
            'quality_threshold': 0.5,
            'response_cache_enabled': False
        },
        'description': 'Corrective RAG (Self-Refine)'
    }
}

# ============================================================
# 테스트 질문 (5턴 대화)
# ============================================================

DEFAULT_QUESTIONS = [
    "당뇨병이란 무엇인가요?",
    "당뇨병의 주요 증상은 무엇인가요?",
    "당뇨병 환자가 피해야 할 음식은 무엇인가요?",
    "메트포르민의 부작용은 무엇인가요?",
    "당뇨병 환자의 운동 방법을 알려주세요."
]

# ============================================================
# 비교 실험 실행
# ============================================================

def run_comparison_experiment(
    patient_id: str,
    questions: List[str],
    experiment_id: str,
    output_dir: Path
) -> Dict[str, List[Dict]]:
    """
    3가지 시스템으로 동일한 대화 수행
    
    Args:
        patient_id: 환자 ID
        questions: 질문 리스트
        experiment_id: 실험 ID
        output_dir: 출력 디렉토리
    
    Returns:
        {variant_name: [turn_logs]}
    """
    results = {}
    
    print(f"\n{'='*80}")
    print(f"실험 시작: {experiment_id}")
    print(f"환자 ID: {patient_id}")
    print(f"질문 수: {len(questions)}")
    print(f"{'='*80}\n")
    
    for variant_name, config in EXPERIMENT_VARIANTS.items():
        print(f"\n[{variant_name.upper()}] 실행 중...")
        print(f"  설명: {config['description']}")
        print(f"  모드: {config['mode']}")
        
        conversation_log = []
        session_id = f"{experiment_id}_{variant_name}_{patient_id}"
        
        for turn_idx, question in enumerate(questions):
            turn_num = turn_idx + 1
            print(f"  턴 {turn_num}/{len(questions)}: {question[:50]}...")
            
            try:
                # run_agent 호출
                start_time = time.time()
                result = run_agent(
                    user_text=question,
                    mode=config['mode'],
                    feature_overrides=config.get('feature_overrides', {}),
                    session_id=session_id,
                    return_state=True
                )
                elapsed_time = time.time() - start_time
                
                # 턴 로그 생성
                turn_log = {
                    'experiment_id': experiment_id,
                    'patient_id': patient_id,
                    'variant': variant_name,
                    'turn': turn_num,
                    'question': question,
                    'answer': result.get('answer', ''),
                    'contexts': [doc.get('text', '') for doc in result.get('retrieved_docs', [])],
                    'metadata': {
                        'iteration_count': result.get('iteration_count', 0),
                        'quality_score': result.get('quality_score', 0.0),
                        'elapsed_time': elapsed_time,
                        'mode': config['mode'],
                        'refine_strategy': result.get('refine_strategy', 'N/A')
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                conversation_log.append(turn_log)
                print(f"    ✓ 완료 ({elapsed_time:.2f}초)")
                
            except Exception as e:
                print(f"    ✗ 오류: {e}")
                # 오류 로그도 저장
                turn_log = {
                    'experiment_id': experiment_id,
                    'patient_id': patient_id,
                    'variant': variant_name,
                    'turn': turn_num,
                    'question': question,
                    'answer': '',
                    'contexts': [],
                    'metadata': {
                        'error': str(e)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                conversation_log.append(turn_log)
        
        results[variant_name] = conversation_log
        
        # 변형별 로그 저장
        variant_log_file = output_dir / variant_name / f"{patient_id}.jsonl"
        variant_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(variant_log_file, 'w', encoding='utf-8') as f:
            for turn_log in conversation_log:
                f.write(json.dumps(turn_log, ensure_ascii=False) + '\n')
        
        print(f"  저장: {variant_log_file}")
    
    return results


def save_summary(
    results: Dict[str, List[Dict]],
    experiment_id: str,
    output_dir: Path
):
    """실험 요약 저장"""
    summary = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'variants': list(results.keys()),
        'num_turns': len(results[list(results.keys())[0]]),
        'statistics': {}
    }
    
    for variant_name, conversation_log in results.items():
        # 통계 계산
        total_time = sum(turn.get('metadata', {}).get('elapsed_time', 0) for turn in conversation_log)
        avg_time = total_time / len(conversation_log) if conversation_log else 0
        
        num_contexts = [len(turn.get('contexts', [])) for turn in conversation_log]
        avg_contexts = sum(num_contexts) / len(num_contexts) if num_contexts else 0
        
        summary['statistics'][variant_name] = {
            'total_time': total_time,
            'avg_time_per_turn': avg_time,
            'avg_contexts_per_turn': avg_contexts,
            'num_turns': len(conversation_log)
        }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n요약 저장: {summary_file}")
    
    # 콘솔 출력
    print(f"\n{'='*80}")
    print("실험 요약")
    print(f"{'='*80}")
    for variant_name, stats in summary['statistics'].items():
        print(f"\n[{variant_name.upper()}]")
        print(f"  총 시간: {stats['total_time']:.2f}초")
        print(f"  평균 시간/턴: {stats['avg_time_per_turn']:.2f}초")
        print(f"  평균 문서 수/턴: {stats['avg_contexts_per_turn']:.1f}")


def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description='LLM vs RAG 비교 실험')
    parser.add_argument('--patients', type=int, default=1,
                        help='테스트할 환자 수 (기본: 1)')
    parser.add_argument('--turns', type=int, default=5,
                        help='턴 수 (기본: 5)')
    parser.add_argument('--patient-id', type=str, default='TEST_001',
                        help='환자 ID (기본: TEST_001)')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/comparison_logs',
                        help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 실험 ID 생성
    experiment_id = f"llm_vs_rag_{datetime.now():%Y%m%d_%H%M%S}"
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 질문 준비
    questions = DEFAULT_QUESTIONS[:args.turns]
    
    # 실험 실행
    results = run_comparison_experiment(
        patient_id=args.patient_id,
        questions=questions,
        experiment_id=experiment_id,
        output_dir=output_dir
    )
    
    # 요약 저장
    save_summary(results, experiment_id, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ 실험 완료!")
    print(f"{'='*80}")
    print(f"\n결과 위치: {output_dir}")
    print(f"\n다음 단계:")
    print(f"  python experiments/evaluate_llm_vs_rag.py --log-dir {output_dir}")


if __name__ == '__main__':
    main()

