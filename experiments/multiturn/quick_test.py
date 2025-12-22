"""
멀티턴 시스템 빠른 테스트
전체 파이프라인 검증용
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.multiturn import (
    PatientScenarioGenerator,
    QuestionBank,
    MultiTurnSimulator,
    PatientResponseGenerator,
    EvaluationRubric
)


def test_patient_scenarios():
    """환자 시나리오 생성 테스트"""
    print("=" * 60)
    print("1. 환자 시나리오 생성 테스트")
    print("=" * 60)
    
    generator = PatientScenarioGenerator()
    scenarios = generator.generate_all_scenarios()
    
    print(f"[OK] 총 {len(scenarios)}개 시나리오 생성")
    
    # 코호트별 개수
    for cohort in ["Full", "No-Meds", "No-Trend"]:
        count = len([s for s in scenarios if s.cohort == cohort])
        print(f"  - {cohort}: {count}개")
    
    # 샘플 출력
    sample = scenarios[0]
    print(f"\n[샘플: {sample.patient_id}]")
    print(f"  이름: {sample.name}, 나이: {sample.age}세")
    print(f"  진단: {', '.join(sample.diagnosis)}")
    print(f"  약물: {len(sample.medications)}개")
    print(f"  검사: {len(sample.lab_results)}개")
    print(f"  코호트: {sample.cohort}, 난이도: {sample.scenario_level}")
    
    return scenarios


def test_question_bank():
    """질문은행 테스트"""
    print("\n" + "=" * 60)
    print("2. 질문은행 테스트")
    print("=" * 60)
    
    bank = QuestionBank()
    
    print(f"[OK] 총 {len(bank.templates)}개 템플릿 로드")
    
    # 턴 타입별 개수
    from experiments.multiturn.question_bank import TurnType
    for turn_type in TurnType:
        templates = bank.get_templates_by_turn_type(turn_type)
        print(f"  - {turn_type.value}: {len(templates)}개")
    
    # 샘플 템플릿
    t1 = bank.get_template("T1_basic_info")
    print(f"\n[샘플: T1_basic_info]")
    print(f"  목적: {t1.goal}")
    print(f"  Prerequisites: {t1.prerequisites}")
    print(f"  Disclosed slots: {t1.disclosed_slots}")
    print(f"  템플릿 (첫 100자): {t1.template_text[:100]}...")
    
    return bank


def test_simulator(scenarios, bank):
    """시뮬레이터 테스트"""
    print("\n" + "=" * 60)
    print("3. 대화 시뮬레이터 테스트")
    print("=" * 60)
    
    response_gen = PatientResponseGenerator(temperature=0.0, seed=42)
    simulator = MultiTurnSimulator(bank, response_gen, protocol="P6")
    
    # 첫 번째 환자로 대화 생성
    patient = scenarios[0]
    print(f"환자: {patient.name} ({patient.patient_id})")
    
    dialogue = simulator.generate_dialogue(patient, mode="cooperative")
    
    print(f"[OK] 대화 ID: {dialogue.dialogue_id}")
    print(f"[OK] 총 {len(dialogue.turns)}개 턴 생성\n")
    
    # 각 턴 요약
    for turn in dialogue.turns:
        print(f"[Turn {turn.turn_idx}: {turn.turn_type}]")
        print(f"  템플릿: {turn.template_id}")
        print(f"  질문 (첫 80자): {turn.question[:80]}...")
        print(f"  환자 응답: {turn.patient_response[:80] if turn.patient_response else 'N/A'}...")
        if turn.gold_answer:
            print(f"  정답: {turn.gold_answer[:80]}...")
        print()
    
    return dialogue


def test_evaluation(dialogue):
    """평가 루브릭 테스트"""
    print("=" * 60)
    print("4. 평가 루브릭 테스트")
    print("=" * 60)
    
    rubric = EvaluationRubric(model_type="basic")
    
    # 첫 번째 턴 평가
    turn = dialogue.turns[0]
    
    # 더미 모델 답변
    model_answer = "진단명: Type 2 Diabetes, Hypertension. 복용약: Metformin 500mg BID, Amlodipine 5mg QD. 알레르기: 없음."
    
    result = rubric.evaluate_turn(
        turn_data=turn.to_dict(),
        model_answer=model_answer,
        gold_answer=turn.gold_answer
    )
    
    print(f"[OK] Turn {result.turn_idx} ({result.turn_type}) 평가 완료")
    print(f"  총점: {result.total_score:.2f} / {len(result.subscores) * 2:.0f}")
    print(f"  가중 점수: {result.weighted_score:.2f}\n")
    
    print("서브스코어:")
    for subscore in result.subscores:
        print(f"  - {subscore.metric}: {subscore.score:.1f}/2.0")
        print(f"    이유: {subscore.reason}")
    
    return result


def test_cohort_gating(scenarios, bank):
    """코호트별 게이팅 테스트"""
    print("\n" + "=" * 60)
    print("5. 코호트별 게이팅 테스트")
    print("=" * 60)
    
    from experiments.multiturn.question_bank import TurnType
    
    # 각 코호트별로 T2 템플릿 선택 확인
    for cohort in ["Full", "No-Trend"]:
        cohort_scenarios = [s for s in scenarios if s.cohort == cohort]
        if not cohort_scenarios:
            continue
        
        patient = cohort_scenarios[0]
        patient_data = patient.to_dict()
        
        template = bank.select_template_with_gating(
            turn_type=TurnType.T2_COMPARISON,
            patient_data=patient_data,
            cohort=cohort
        )
        
        print(f"\n[{cohort} Cohort]")
        print(f"  환자: {patient.patient_id}")
        print(f"  검사 횟수: {len(patient.lab_results)}회")
        print(f"  선택된 템플릿: {template.template_id}")
        print(f"  목적: {template.goal}")


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 60)
    print("멀티턴 시스템 빠른 테스트")
    print("=" * 60 + "\n")
    
    try:
        # 1. 환자 시나리오
        scenarios = test_patient_scenarios()
        
        # 2. 질문은행
        bank = test_question_bank()
        
        # 3. 시뮬레이터
        dialogue = test_simulator(scenarios, bank)
        
        # 4. 평가
        test_evaluation(dialogue)
        
        # 5. 코호트 게이팅
        test_cohort_gating(scenarios, bank)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 테스트 통과!")
        print("=" * 60)
        print("\n다음 단계:")
        print("1. 실제 RAG 시스템 연결: run_multiturn_experiment.py 수정")
        print("2. 전체 실험 실행: python run_multiturn_experiment.py")
        print("3. 결과 분석: results/multiturn/ 확인")
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

