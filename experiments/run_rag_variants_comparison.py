"""
RAG 시스템 간 비교 실험 (피드백 반영)

비교 대상:
1. Basic RAG (baseline)
2. Modular RAG (self_refine_llm_quality)
3. Corrective RAG (full_context_engineering)

Usage:
    python experiments/run_rag_variants_comparison.py --patient-id P001 --turns 5
"""
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict, Any

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.graph import run_agent
from config.ablation_config import get_ablation_profile, ABLATION_PROFILES

# ============================================================
# RAG 시스템 변형 정의 (논문/보고서용)
# ============================================================

RAG_VARIANTS = {
    "basic_rag": {
        "profile": "baseline",
        "description": "Basic RAG: 단순 검색-생성 (Self-Refine 없음)",
        "color": "#3498db"
    },
    "modular_rag": {
        "profile": "self_refine_llm_quality",
        "description": "Modular RAG: LLM 기반 품질 평가 + Self-Refine",
        "color": "#e74c3c"
    },
    "corrective_rag": {
        "profile": "full_context_engineering",
        "description": "Corrective RAG (Agentic): 동적 검색 + 품질 제어 + 메모리",
        "color": "#2ecc71"
    }
}

# ============================================================
# 테스트 시나리오 (환자 케이스)
# ============================================================

PATIENT_SCENARIOS = {
    "P001": {
        "name": "당뇨병 환자 (메트포르민 복용)",
        "turns": [
            "당뇨병 환자인데 메트포르민을 복용하고 있어요. 부작용이 궁금합니다.",
            "메트포르민 복용 시 피해야 할 음식이 있나요?",
            "메트포르민과 함께 복용하면 안 되는 약물은?",
            "메트포르민 복용 중 운동은 어떻게 해야 하나요?",
            "메트포르민 복용을 잊었을 때 어떻게 해야 하나요?"
        ]
    },
    "P002": {
        "name": "고혈압 환자 (임신 계획)",
        "turns": [
            "고혈압 환자인데 임신을 계획하고 있습니다.",
            "임신 중에도 고혈압 약을 계속 복용해야 하나요?",
            "임신 중 복용 가능한 고혈압 약물은 무엇인가요?",
            "임신 중 혈압 관리를 위한 식이요법은?",
            "임신 중 혈압이 갑자기 오르면 어떻게 해야 하나요?"
        ]
    },
    "P003": {
        "name": "간 질환 환자 (약물 복용)",
        "turns": [
            "간 질환이 있는데 진통제를 복용해도 되나요?",
            "간 질환 환자에게 금기인 약물은 무엇인가요?",
            "간 질환 환자의 식이요법은?",
            "간 질환 환자가 피해야 할 음식은?",
            "간 질환 환자의 알코올 섭취는 절대 금지인가요?"
        ]
    }
}


def run_variant_experiment(
    variant_name: str,
    patient_id: str,
    turns: List[str],
    session_id: str
) -> Dict[str, Any]:
    """단일 RAG 변형 실험 실행
    
    Args:
        variant_name: RAG 변형 이름 (basic_rag, modular_rag, corrective_rag)
        patient_id: 환자 ID
        turns: 대화 턴 리스트
        session_id: 세션 ID
    
    Returns:
        실험 결과 딕셔너리
    """
    variant_config = RAG_VARIANTS[variant_name]
    profile_name = variant_config["profile"]
    
    print(f"\n{'='*80}")
    print(f"[{variant_name}] {variant_config['description']}")
    print(f"{'='*80}")
    
    # Ablation 프로파일 로드
    features = get_ablation_profile(profile_name)
    
    # 캐시 비활성화 (순수 성능 측정)
    features['response_cache_enabled'] = False
    
    # 턴별 결과 저장
    turn_results = []
    conversation_history = ""
    
    # 세션 상태 (메모리 유지)
    session_state = None
    
    for turn_idx, user_query in enumerate(turns, 1):
        print(f"  Turn {turn_idx}/{len(turns)}: {user_query[:60]}...")
        
        turn_start = time.time()
        
        try:
            # Agent 실행
            result = run_agent(
                user_text=user_query,
                mode="ai_agent",
                conversation_history=conversation_history,
                session_state=session_state,
                feature_overrides=features,
                return_state=True,
                session_id=session_id,
                user_id=patient_id
            )
            
            turn_elapsed = time.time() - turn_start
            
            # 메트릭 수집
            answer = result.get('answer', '')
            retrieved_docs = result.get('retrieved_docs', [])
            contexts = [doc.get('text', '') for doc in retrieved_docs]
            
            turn_data = {
                'turn_id': turn_idx,
                'user_query': user_query,
                'answer': answer,
                'contexts': contexts,
                'quality_score': result.get('quality_score', 0.0),
                'iteration_count': result.get('iteration_count', 0),
                'num_docs': len(retrieved_docs),
                'elapsed_sec': turn_elapsed,
                'profile_summary': result.get('profile_summary', ''),
                'slot_out': result.get('slot_out', {}),
            }
            
            turn_results.append(turn_data)
            
            # 대화 이력 업데이트
            conversation_history += f"\nUser: {user_query}\nAssistant: {answer}\n"
            
            # 세션 상태 업데이트 (메모리 유지)
            session_state = {
                'profile_store': result.get('profile_store'),
                'hierarchical_memory': result.get('hierarchical_memory'),
            }
            
            print(f"    ✓ Q={turn_data['quality_score']:.3f}, "
                  f"Iter={turn_data['iteration_count']}, "
                  f"Docs={turn_data['num_docs']}, "
                  f"Time={turn_data['elapsed_sec']:.1f}s")
            
        except Exception as e:
            print(f"    ✗ 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            
            turn_results.append({
                'turn_id': turn_idx,
                'user_query': user_query,
                'error': str(e),
            })
    
    # 통계 계산
    successful_turns = [t for t in turn_results if 'error' not in t]
    
    summary = {
        'total_turns': len(turns),
        'successful_turns': len(successful_turns),
        'avg_quality': sum(t['quality_score'] for t in successful_turns) / len(successful_turns) if successful_turns else 0,
        'avg_iterations': sum(t['iteration_count'] for t in successful_turns) / len(successful_turns) if successful_turns else 0,
        'avg_docs': sum(t['num_docs'] for t in successful_turns) / len(successful_turns) if successful_turns else 0,
        'avg_time_sec': sum(t['elapsed_sec'] for t in successful_turns) / len(successful_turns) if successful_turns else 0,
        'total_time_sec': sum(t['elapsed_sec'] for t in successful_turns),
    }
    
    print(f"\n  📊 요약: Q={summary['avg_quality']:.3f}, "
          f"Iter={summary['avg_iterations']:.1f}, "
          f"Docs={summary['avg_docs']:.1f}, "
          f"Time={summary['avg_time_sec']:.1f}s")
    
    return {
        'variant_name': variant_name,
        'profile_name': profile_name,
        'description': variant_config['description'],
        'patient_id': patient_id,
        'turns': turn_results,
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 시스템 간 비교 실험")
    parser.add_argument('--patient-id', type=str, default='P001',
                        help='환자 시나리오 ID (P001, P002, P003)')
    parser.add_argument('--turns', type=int, default=5,
                        help='실행할 대화 턴 수 (기본: 5)')
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['basic_rag', 'modular_rag', 'corrective_rag'],
                        help='비교할 RAG 변형 (기본: 모두)')
    
    args = parser.parse_args()
    
    # 환자 시나리오 로드
    if args.patient_id not in PATIENT_SCENARIOS:
        print(f"❌ 오류: 존재하지 않는 환자 ID '{args.patient_id}'")
        print(f"   사용 가능한 ID: {list(PATIENT_SCENARIOS.keys())}")
        return
    
    scenario = PATIENT_SCENARIOS[args.patient_id]
    turns = scenario['turns'][:args.turns]
    
    print("=" * 80)
    print("RAG 시스템 간 비교 실험 (피드백 반영)")
    print("=" * 80)
    print(f"환자 시나리오: {scenario['name']}")
    print(f"대화 턴 수: {len(turns)}")
    print(f"비교 변형: {', '.join(args.variants)}")
    print("=" * 80)
    
    # 세션 ID 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{args.patient_id}_{timestamp}"
    
    # 각 변형 실험 실행
    all_results = {}
    
    for variant_name in args.variants:
        if variant_name not in RAG_VARIANTS:
            print(f"⚠️ 경고: 알 수 없는 변형 '{variant_name}' 건너뜀")
            continue
        
        result = run_variant_experiment(
            variant_name=variant_name,
            patient_id=args.patient_id,
            turns=turns,
            session_id=f"{session_id}_{variant_name}"
        )
        
        all_results[variant_name] = result
    
    # ============================================================
    # 비교 테이블 출력
    # ============================================================
    print(f"\n\n{'='*80}")
    print("RAG 시스템 비교 결과")
    print(f"{'='*80}")
    
    # 헤더
    print(f"{'변형':<20} {'품질':>8} {'반복':>6} {'문서':>6} {'시간(s)':>8} {'성공률':>8}")
    print(f"{'-'*80}")
    
    # 각 변형 통계
    for variant_name in args.variants:
        if variant_name not in all_results:
            continue
        
        data = all_results[variant_name]
        s = data['summary']
        success_rate = s['successful_turns'] / s['total_turns'] * 100
        
        print(f"{variant_name:<20} "
              f"{s['avg_quality']:>8.3f} "
              f"{s['avg_iterations']:>6.1f} "
              f"{s['avg_docs']:>6.1f} "
              f"{s['avg_time_sec']:>8.1f} "
              f"{success_rate:>7.0f}%")
    
    print(f"{'='*80}")
    
    # ============================================================
    # 결과 저장 (RAGAS 평가용)
    # ============================================================
    output_dir = project_root / "runs" / "rag_variants_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"comparison_{args.patient_id}_{timestamp}.json"
    
    output_data = {
        'experiment_type': 'rag_variants_comparison',
        'timestamp': datetime.now().isoformat(),
        'patient_id': args.patient_id,
        'patient_name': scenario['name'],
        'num_turns': len(turns),
        'variants_tested': args.variants,
        'results': all_results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과 저장: {output_file}")
    print(f"   다음 단계: python experiments/evaluate_rag_variants.py {output_file}")
    
    print("\n실험 완료! 🎉")


if __name__ == "__main__":
    main()

