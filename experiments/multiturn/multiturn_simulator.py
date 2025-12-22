"""
멀티턴 대화 시뮬레이터
재현성 확보를 위한 프로토콜 기반 대화 생성
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import copy

try:
    from .patient_scenarios import PatientScenario
    from .question_bank import QuestionBank, TurnType, QuestionTemplate
except ImportError:
    from patient_scenarios import PatientScenario
    from question_bank import QuestionBank, TurnType, QuestionTemplate


@dataclass
class Turn:
    """단일 턴 데이터"""
    turn_idx: int
    turn_type: str
    template_id: str
    question: str
    patient_response: Optional[str] = None
    model_answer: Optional[str] = None
    disclosed_slots: List[str] = None
    required_slots: List[str] = None
    canonical_state_snapshot: Dict[str, Any] = None
    gold_answer: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiTurnDialogue:
    """멀티턴 대화 전체"""
    dialogue_id: str
    patient_id: str
    cohort: str
    scenario_level: str
    protocol_id: str  # "P6"
    turns: List[Turn]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['turns'] = [t.to_dict() for t in self.turns]
        return data


class PatientResponseGenerator:
    """
    환자 응답 생성기
    4블록 구조 강제: Direct Answer + Disclosed Slots + Unknown Handling + Follow-up
    """
    
    def __init__(self, temperature: float = 0.0, seed: int = 42):
        self.temperature = temperature
        self.seed = seed
    
    def generate_response(
        self, 
        patient: PatientScenario,
        turn: Turn,
        mode: str = "cooperative"
    ) -> str:
        """
        환자 응답 생성
        
        Args:
            patient: 환자 시나리오
            turn: 현재 턴
            mode: "cooperative", "minimal", "noisy"
        """
        # 실제 구현에서는 LLM 호출 (temperature=0, seed 고정)
        # 여기서는 규칙 기반 생성 (재현성 극대화)
        
        response_blocks = []
        
        # 1. Direct Answer (필수)
        direct_answer = self._generate_direct_answer(patient, turn)
        response_blocks.append(direct_answer)
        
        # 2. Disclosed Slots (규칙 기반)
        if turn.disclosed_slots:
            slot_info = self._generate_slot_disclosure(patient, turn.disclosed_slots)
            if slot_info:
                response_blocks.append(slot_info)
        
        # 3. Unknown Handling (해당 시)
        unknown_info = self._generate_unknown_handling(patient, turn)
        if unknown_info:
            response_blocks.append(unknown_info)
        
        # 4. Follow-up Question (선택적)
        if mode == "cooperative" and turn.turn_type in ["T1", "T4"]:
            followup = self._generate_followup(patient, turn)
            if followup:
                response_blocks.append(followup)
        
        return "\n".join(response_blocks)
    
    def _generate_direct_answer(self, patient: PatientScenario, turn: Turn) -> str:
        """직답 생성"""
        turn_type = turn.turn_type
        
        if turn_type == "T1":
            return "네, 제 기록을 확인해 주세요."
        elif turn_type == "T2":
            return "네, 검사 수치를 비교해 드릴게요."
        elif turn_type == "T3":
            return "네, 다시 정리해 드리겠습니다."
        elif turn_type == "T4":
            return "네, 제 상황에 맞는 권고를 부탁드립니다."
        elif turn_type == "T5":
            return "죄송합니다. 정정 사항을 반영해 주세요."
        elif turn_type == "T6":
            return "네, 전체적으로 정리해 주세요."
        else:
            return "네, 알겠습니다."
    
    def _generate_slot_disclosure(self, patient: PatientScenario, slots: List[str]) -> str:
        """슬롯 공개 정보 생성"""
        disclosures = []
        
        for slot in slots:
            if slot == "diagnosis":
                if patient.diagnosis:
                    disclosures.append(f"진단명: {', '.join(patient.diagnosis)}")
                else:
                    disclosures.append("진단명: 없음")
            
            elif slot == "medications":
                if patient.medications:
                    med_strs = [
                        f"{m.name} {m.dosage} {m.frequency}" 
                        for m in patient.medications
                    ]
                    disclosures.append(f"복용약: {', '.join(med_strs)}")
                else:
                    disclosures.append("복용약: 없음")
            
            elif slot == "allergy":
                disclosures.append(f"알레르기: {patient.allergy}")
            
            elif slot == "lab_results":
                if patient.lab_results:
                    lab_strs = [
                        f"{lr.date} {lr.test} {lr.value}{lr.unit}"
                        for lr in patient.lab_results[-2:]  # 최근 2개
                    ]
                    disclosures.append(f"검사 결과: {', '.join(lab_strs)}")
                else:
                    disclosures.append("검사 결과: 없음")
            
            elif slot == "age":
                disclosures.append(f"나이: {patient.age}세")
        
        return "\n".join(disclosures) if disclosures else ""
    
    def _generate_unknown_handling(self, patient: PatientScenario, turn: Turn) -> str:
        """모름/없음 처리"""
        # 실제로는 더 정교한 로직 필요
        # 여기서는 간단히 처리
        return ""
    
    def _generate_followup(self, patient: PatientScenario, turn: Turn) -> str:
        """후속 질문 생성"""
        if turn.turn_type == "T1":
            return "이 정보가 맞나요?"
        elif turn.turn_type == "T4":
            return "추가로 주의할 점이 있을까요?"
        return ""


class MultiTurnSimulator:
    """멀티턴 대화 시뮬레이터"""
    
    def __init__(
        self, 
        question_bank: QuestionBank,
        patient_response_generator: PatientResponseGenerator,
        protocol: str = "P6"
    ):
        self.question_bank = question_bank
        self.response_generator = patient_response_generator
        self.protocol = protocol
        
        # P6 프로토콜 정의
        self.protocol_sequence = [
            TurnType.T1_FACT_RETRIEVAL,
            TurnType.T2_COMPARISON,
            TurnType.T3_CONTEXT_DEPENDENT,
            TurnType.T4_COMPLEX_REASONING,
            TurnType.T5_CORRECTION,
            TurnType.T6_CONSISTENCY
        ]
    
    def generate_dialogue(
        self, 
        patient: PatientScenario,
        mode: str = "cooperative"
    ) -> MultiTurnDialogue:
        """
        단일 환자에 대한 멀티턴 대화 생성
        
        Args:
            patient: 환자 시나리오
            mode: 환자 응답 모드
        """
        dialogue_id = f"D-{patient.patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        turns = []
        
        # 환자 상태 (정정 반영용)
        current_state = self._patient_to_dict(patient)
        
        for turn_idx, turn_type in enumerate(self.protocol_sequence, start=1):
            # 템플릿 선택 (게이팅)
            template = self.question_bank.select_template_with_gating(
                turn_type=turn_type,
                patient_data=current_state,
                cohort=patient.cohort
            )
            
            # 질문 생성
            question = self._format_question(template, current_state, turn_idx)
            
            # 턴 객체 생성
            turn = Turn(
                turn_idx=turn_idx,
                turn_type=turn_type.value,
                template_id=template.template_id,
                question=question,
                disclosed_slots=template.disclosed_slots,
                required_slots=template.required_slots_in_answer,
                canonical_state_snapshot=copy.deepcopy(current_state)
            )
            
            # 환자 응답 생성
            turn.patient_response = self.response_generator.generate_response(
                patient, turn, mode
            )
            
            # 정정 턴이면 상태 업데이트
            if turn_type == TurnType.T5_CORRECTION:
                current_state = self._apply_correction(current_state, template, patient)
            
            # Gold Answer 생성 (가능한 경우)
            if template.gold_answer_generator:
                turn.gold_answer = self._generate_gold_answer(
                    template.gold_answer_generator, 
                    patient, 
                    current_state,
                    turn
                )
            
            turns.append(turn)
        
        return MultiTurnDialogue(
            dialogue_id=dialogue_id,
            patient_id=patient.patient_id,
            cohort=patient.cohort,
            scenario_level=patient.scenario_level,
            protocol_id=self.protocol,
            turns=turns,
            metadata={
                "patient_name": patient.name,
                "generation_time": datetime.now().isoformat(),
                "mode": mode,
                "temperature": self.response_generator.temperature,
                "seed": self.response_generator.seed
            }
        )
    
    def _patient_to_dict(self, patient: PatientScenario) -> Dict[str, Any]:
        """환자 객체를 딕셔너리로 변환"""
        return patient.to_dict()
    
    def _format_question(
        self, 
        template: QuestionTemplate, 
        patient_data: Dict[str, Any],
        turn_idx: int
    ) -> str:
        """템플릿을 환자 데이터로 채워서 질문 생성"""
        # 템플릿에 필요한 변수 추출
        template_vars = {}
        
        # 기본 정보
        template_vars['age'] = patient_data.get('age', '알 수 없음')
        template_vars['diagnosis_list'] = ', '.join(patient_data.get('diagnosis', []))
        
        # 약물 정보
        meds = patient_data.get('medications', [])
        if meds:
            template_vars['medication_list'] = ', '.join([m['name'] for m in meds])
        else:
            template_vars['medication_list'] = '없음'
        
        # 검사 결과 (최근 2개)
        labs = patient_data.get('lab_results', [])
        if labs:
            latest = labs[-1]
            template_vars['test_name'] = latest['test']
            template_vars['latest_value'] = latest['value']
            template_vars['unit'] = latest.get('unit', '')
            template_vars['date'] = latest['date']
            
            if len(labs) >= 2:
                prev = labs[-2]
                template_vars['date1'] = prev['date']
                template_vars['value1'] = prev['value']
                template_vars['date2'] = latest['date']
                template_vars['value2'] = latest['value']
        
        # 목표 범위 (하드코딩, 실제로는 DB에서)
        template_vars['target_range'] = self._get_target_range(
            template_vars.get('test_name', '')
        )
        
        # T5 정정 턴용 변수
        if template.turn_type == TurnType.T5_CORRECTION:
            if 'lab' in template.template_id:
                template_vars['old_value'] = template_vars.get('value2', 0)
                template_vars['new_value'] = template_vars.get('value2', 0) + 1.0  # 임시
            elif 'medication' in template.template_id:
                template_vars['old_med'] = template_vars.get('medication_list', '').split(',')[0]
                template_vars['new_med'] = "수정된약물"
            elif 'allergy' in template.template_id:
                template_vars['new_allergy'] = "Sulfa"
        
        try:
            return template.template_text.format(**template_vars)
        except KeyError as e:
            # 변수 누락 시 템플릿 그대로 반환 (디버깅용)
            print(f"Warning: Missing template variable {e} in {template.template_id}")
            return template.template_text
    
    def _get_target_range(self, test_name: str) -> str:
        """검사 항목별 목표 범위 (하드코딩)"""
        ranges = {
            "HbA1c": "7.0% 미만",
            "BP": "140/90 mmHg 미만",
            "LDL": "100 mg/dL 미만",
            "TSH": "0.5-5.0 mIU/L",
            "FEV1": "80% 이상"
        }
        return ranges.get(test_name, "정상 범위")
    
    def _apply_correction(
        self, 
        current_state: Dict[str, Any], 
        template: QuestionTemplate,
        patient: PatientScenario
    ) -> Dict[str, Any]:
        """정정 사항 반영 (T5)"""
        # 실제 정정값은 환자 시나리오에 미리 정의되어 있어야 함
        # 여기서는 간단히 lab 값 수정
        if 'lab' in template.template_id and current_state.get('lab_results'):
            # 최근 검사 수치 수정 (예: HbA1c 7.2 → 8.2)
            labs = current_state['lab_results']
            if labs:
                # P-F003 시나리오의 경우 8.2 → 9.2로 정정
                if patient.patient_id == "P-F003":
                    labs[-1]['value'] = 9.2
        
        return current_state
    
    def _generate_gold_answer(
        self, 
        generator_name: str, 
        patient: PatientScenario,
        current_state: Dict[str, Any],
        turn: Turn
    ) -> str:
        """Gold Answer 생성 (결정론적)"""
        # 실제 구현에서는 각 generator 함수 호출
        # 여기서는 간단히 처리
        
        if generator_name == "generate_t1_gold_answer":
            parts = []
            parts.append(f"1. 진단명: {', '.join(patient.diagnosis) if patient.diagnosis else '없음'}")
            if patient.medications:
                med_strs = [f"{m.name} {m.dosage} {m.frequency}" for m in patient.medications]
                parts.append(f"2. 복용약: {', '.join(med_strs)}")
            else:
                parts.append("2. 복용약: 없음")
            parts.append(f"3. 알레르기: {patient.allergy}")
            return "\n".join(parts)
        
        elif generator_name == "generate_t2_comparison_gold":
            labs = current_state.get('lab_results', [])
            if len(labs) >= 2:
                prev = labs[-2]
                latest = labs[-1]
                diff = latest['value'] - prev['value']
                direction = "증가" if diff > 0 else ("감소" if diff < 0 else "유지")
                return (
                    f"{prev['date']} {prev['value']}{prev.get('unit', '')}에서 "
                    f"{latest['date']} {latest['value']}{latest.get('unit', '')}로 "
                    f"{direction} (변화량: {abs(diff)}{latest.get('unit', '')})"
                )
        
        elif generator_name == "generate_t3_lab_recall_gold":
            labs = current_state.get('lab_results', [])
            if len(labs) >= 2:
                prev = labs[-2]
                latest = labs[-1]
                diff = latest['value'] - prev['value']
                return (
                    f"이전값 {prev['value']}{prev.get('unit', '')}, "
                    f"최근값 {latest['value']}{latest.get('unit', '')}, "
                    f"변화량 {abs(diff)}{latest.get('unit', '')}"
                )
        
        elif generator_name == "generate_t6_recall_gold":
            # T6: 전체 정보 요약 (정정 반영)
            parts = []
            parts.append(f"1. 진단명: {', '.join(current_state.get('diagnosis', []))}")
            meds = current_state.get('medications', [])
            if meds:
                med_strs = [f"{m['name']} {m['dosage']} {m['frequency']}" for m in meds]
                parts.append(f"2. 복용약: {', '.join(med_strs)}")
            else:
                parts.append("2. 복용약: 없음")
            parts.append(f"3. 알레르기: {current_state.get('allergy', '없음')}")
            labs = current_state.get('lab_results', [])
            if labs:
                latest = labs[-1]
                parts.append(f"4. 최근 검사: {latest['test']} {latest['value']}{latest.get('unit', '')}")
            return "\n".join(parts)
        
        return ""


def main():
    """시뮬레이터 테스트"""
    from patient_scenarios import PatientScenarioGenerator
    
    # 환자 시나리오 로드
    generator = PatientScenarioGenerator()
    scenarios = generator.generate_all_scenarios()
    
    # 질문은행 초기화
    question_bank = QuestionBank()
    
    # 응답 생성기 초기화
    response_gen = PatientResponseGenerator(temperature=0.0, seed=42)
    
    # 시뮬레이터 초기화
    simulator = MultiTurnSimulator(question_bank, response_gen, protocol="P6")
    
    # 첫 번째 환자로 대화 생성
    patient = scenarios[0]
    dialogue = simulator.generate_dialogue(patient, mode="cooperative")
    
    print(f"대화 ID: {dialogue.dialogue_id}")
    print(f"환자: {patient.name} ({patient.patient_id})")
    print(f"코호트: {dialogue.cohort}, 난이도: {dialogue.scenario_level}")
    print(f"\n총 {len(dialogue.turns)}개 턴:\n")
    
    for turn in dialogue.turns:
        print(f"=== Turn {turn.turn_idx}: {turn.turn_type} ({turn.template_id}) ===")
        print(f"질문: {turn.question[:100]}...")
        print(f"환자 응답: {turn.patient_response[:100] if turn.patient_response else 'N/A'}...")
        if turn.gold_answer:
            print(f"정답: {turn.gold_answer[:100]}...")
        print()
    
    # JSON 저장
    output_path = "data/multiturn/sample_dialogue.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()

