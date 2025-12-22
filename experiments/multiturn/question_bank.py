"""
질문은행 (Question Bank)
턴 타입별 템플릿 + prerequisites + fallback 정의
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TurnType(Enum):
    """턴 타입"""
    T1_FACT_RETRIEVAL = "T1"
    T2_COMPARISON = "T2"
    T3_CONTEXT_DEPENDENT = "T3"
    T4_COMPLEX_REASONING = "T4"
    T5_CORRECTION = "T5"
    T6_CONSISTENCY = "T6"


@dataclass
class QuestionTemplate:
    """질문 템플릿"""
    template_id: str
    turn_type: TurnType
    goal: str  # 목적 설명
    prerequisites: List[str]  # 필요 슬롯 (예: ["medications", "lab_results"])
    fallback_id: Optional[str]  # prerequisites 불충족 시 대체 템플릿 ID
    disclosed_slots: List[str]  # 이번 턴에서 공개되는 슬롯
    required_slots_in_answer: List[str]  # 답변에 반드시 포함되어야 하는 슬롯
    template_text: str  # 템플릿 문자열 (f-string 형식)
    gold_answer_generator: Optional[str] = None  # 정답 생성 함수명
    cohort_specific: Optional[str] = None  # 특정 코호트 전용 (None이면 공통)
    
    def format(self, patient_data: Dict[str, Any]) -> str:
        """환자 데이터로 템플릿 채우기"""
        return self.template_text.format(**patient_data)


class QuestionBank:
    """질문은행 관리"""
    
    def __init__(self):
        self.templates: Dict[str, QuestionTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """템플릿 초기화"""
        
        # ===== T1: 단순 사실 조회 =====
        self.add_template(QuestionTemplate(
            template_id="T1_basic_info",
            turn_type=TurnType.T1_FACT_RETRIEVAL,
            goal="기본 슬롯 정확히 추출 (정확성/환각 억제)",
            prerequisites=[],
            fallback_id=None,
            disclosed_slots=["diagnosis", "medications", "allergy"],
            required_slots_in_answer=["diagnosis", "medications", "allergy"],
            template_text=(
                "제 기록을 기준으로 다음을 정리해 주세요:\n"
                "1. 진단명 (있으면 나열, 없으면 '없음')\n"
                "2. 현재 복용 중인 약물 (약물명, 용량, 빈도 포함. 없으면 '복용약 없음')\n"
                "3. 알레르기 (있으면 나열, 없으면 '알레르기 없음')\n\n"
                "없는 항목은 반드시 '없음'이라고 명시해 주세요."
            ),
            gold_answer_generator="generate_t1_gold_answer"
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T1_medication_details",
            turn_type=TurnType.T1_FACT_RETRIEVAL,
            goal="약물 상세 정보 확인",
            prerequisites=["medications"],
            fallback_id="T1_basic_info",
            disclosed_slots=["medications"],
            required_slots_in_answer=["medications"],
            template_text=(
                "제가 현재 복용 중인 약물의 이름, 용량, 복용 빈도를 "
                "각각 정리해 주세요. 복용약이 없다면 '복용약 없음'이라고 답해 주세요."
            ),
            gold_answer_generator="generate_t1_medication_gold"
        ))
        
        # ===== T2: 수치 비교/집계 =====
        self.add_template(QuestionTemplate(
            template_id="T2_lab_comparison_full",
            turn_type=TurnType.T2_COMPARISON,
            goal="시계열 변화 계산/비교",
            prerequisites=["lab_results_trend"],  # 2회 이상
            fallback_id="T2_lab_interpretation_single",
            disclosed_slots=["lab_results"],
            required_slots_in_answer=["lab_results"],
            template_text=(
                "최근 2회 {test_name} 검사 결과 "
                "({date1}: {value1}{unit}, {date2}: {value2}{unit})를 비교했을 때, "
                "수치가 어떻게 변화했습니까? (증가/감소/유지 + 변화량 포함)"
            ),
            gold_answer_generator="generate_t2_comparison_gold",
            cohort_specific="Full"
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T2_lab_interpretation_single",
            turn_type=TurnType.T2_COMPARISON,
            goal="최근값 해석 (트렌드 없음)",
            prerequisites=["lab_results"],
            fallback_id=None,
            disclosed_slots=["lab_results"],
            required_slots_in_answer=["lab_results"],
            template_text=(
                "최근 {test_name} 검사 결과 ({date}: {value}{unit})가 "
                "목표 범위({target_range})에 속하는지 판단하고 이유를 설명해 주세요."
            ),
            gold_answer_generator="generate_t2_interpretation_gold",
            cohort_specific="No-Trend"
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T2_bp_trend",
            turn_type=TurnType.T2_COMPARISON,
            goal="혈압 변화 추이",
            prerequisites=["lab_results_trend"],
            fallback_id="T2_lab_interpretation_single",
            disclosed_slots=["lab_results"],
            required_slots_in_answer=["lab_results"],
            template_text=(
                "{date1}과 {date2}의 혈압 수치를 비교했을 때, "
                "혈압 조절이 개선되었습니까? 수치 변화량과 함께 설명해 주세요."
            ),
            gold_answer_generator="generate_t2_bp_gold",
            cohort_specific="Full"
        ))
        
        # ===== T3: 문맥 의존 (지시대명사 해소) =====
        self.add_template(QuestionTemplate(
            template_id="T3_anaphora_lab",
            turn_type=TurnType.T3_CONTEXT_DEPENDENT,
            goal="이전 턴 수치 정확히 참조",
            prerequisites=["lab_results"],
            fallback_id=None,
            disclosed_slots=[],  # 이전 턴 재사용
            required_slots_in_answer=["lab_results"],
            template_text=(
                "방금 비교한 그 수치({test_name})의 '이전값 / 최근값 / 변화량'을 "
                "한 문장으로 다시 요약해 주세요."
            ),
            gold_answer_generator="generate_t3_lab_recall_gold"
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T3_anaphora_medication",
            turn_type=TurnType.T3_CONTEXT_DEPENDENT,
            goal="이전 턴 약물 정확히 참조",
            prerequisites=["medications"],
            fallback_id="T3_anaphora_diagnosis",
            disclosed_slots=[],
            required_slots_in_answer=["medications"],
            template_text=(
                "아까 정리한 그 약물 (첫 번째로 언급된 약)의 "
                "복용 목적과 복용법(용량, 빈도)을 다시 설명해 주세요."
            ),
            gold_answer_generator="generate_t3_medication_recall_gold"
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T3_anaphora_diagnosis",
            turn_type=TurnType.T3_CONTEXT_DEPENDENT,
            goal="이전 턴 진단명 참조 (약물 없는 경우)",
            prerequisites=["diagnosis"],
            fallback_id=None,
            disclosed_slots=[],
            required_slots_in_answer=["diagnosis"],
            template_text=(
                "제가 처음에 말한 그 진단명(주 진단)에 대해 "
                "간단히 설명하고, 일반적인 관리 방법을 알려주세요."
            ),
            gold_answer_generator=None,  # 외부 지식 필요 (정답 고정 어려움)
            cohort_specific="No-Meds"
        ))
        
        # ===== T4: 복합 계획/권고 =====
        self.add_template(QuestionTemplate(
            template_id="T4_guideline_based_plan",
            turn_type=TurnType.T4_COMPLEX_REASONING,
            goal="가이드라인 기반 추론 + 개인화",
            prerequisites=["diagnosis", "age", "lab_results"],
            fallback_id=None,
            disclosed_slots=["age"],
            required_slots_in_answer=["diagnosis", "age", "lab_results"],
            template_text=(
                "제 나이({age}세), 동반 질환({diagnosis_list}), "
                "최근 {test_name} 수치({latest_value}{unit})를 고려할 때, "
                "관련 진료지침 기준으로 현재 약물 치료를 유지해야 할까요, "
                "아니면 조정(증량/계열 변경)이 필요할까요?\n\n"
                "**반드시 근거 문서에서 찾은 구체적인 기준(예: 목표 범위, 나이별 권고)을 "
                "1~2개 인용하여** 판단 근거를 설명해 주세요."
            ),
            gold_answer_generator=None  # 가이드라인 의존 (정답 고정 어려움)
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T4_personalized_lifestyle",
            turn_type=TurnType.T4_COMPLEX_REASONING,
            goal="개인화된 생활습관 권고",
            prerequisites=["diagnosis", "medications"],
            fallback_id="T4_general_lifestyle",
            disclosed_slots=[],
            required_slots_in_answer=["diagnosis", "medications"],
            template_text=(
                "제 진단({diagnosis_list})과 복용 중인 약물({medication_list})을 고려할 때, "
                "특히 주의해야 할 생활습관이나 식이 조절 사항이 있나요? "
                "약물과 상호작용할 수 있는 음식이나 활동도 포함해 주세요."
            ),
            gold_answer_generator=None
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T4_general_lifestyle",
            turn_type=TurnType.T4_COMPLEX_REASONING,
            goal="일반적 생활습관 권고 (약물 없음)",
            prerequisites=["diagnosis"],
            fallback_id=None,
            disclosed_slots=[],
            required_slots_in_answer=["diagnosis"],
            template_text=(
                "제 진단({diagnosis_list})을 고려할 때, "
                "비약물적 관리 방법(식이, 운동, 생활습관)을 "
                "우선순위대로 3가지 추천해 주세요."
            ),
            gold_answer_generator=None,
            cohort_specific="No-Meds"
        ))
        
        # ===== T5: 정정/모순 =====
        self.add_template(QuestionTemplate(
            template_id="T5_correction_lab",
            turn_type=TurnType.T5_CORRECTION,
            goal="검사 수치 정정 후 재추론",
            prerequisites=["lab_results"],
            fallback_id=None,
            disclosed_slots=[],  # 업데이트
            required_slots_in_answer=["lab_results"],
            template_text=(
                "죄송합니다. 아까 {test_name} 최근값이 {old_value}{unit}라고 했는데, "
                "사실 {new_value}{unit}였어요. 그렇다면 이전 턴에서 내린 결론"
                "(유지 vs 조정)이 어떻게 바뀌나요?"
            ),
            gold_answer_generator=None  # 논리 평가 중심
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T5_correction_medication",
            turn_type=TurnType.T5_CORRECTION,
            goal="약물 정보 정정",
            prerequisites=["medications"],
            fallback_id="T5_correction_allergy",
            disclosed_slots=[],
            required_slots_in_answer=["medications"],
            template_text=(
                "죄송합니다. 제가 복용 중인 약이 {old_med}가 아니라 {new_med}였어요. "
                "이 경우 주의사항이나 권고사항이 달라지나요?"
            ),
            gold_answer_generator=None
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T5_correction_allergy",
            turn_type=TurnType.T5_CORRECTION,
            goal="알레르기 정보 추가",
            prerequisites=["allergy"],
            fallback_id=None,
            disclosed_slots=[],
            required_slots_in_answer=["allergy"],
            template_text=(
                "아까 알레르기가 없다고 했는데, 사실 {new_allergy} 알레르기가 있어요. "
                "이 경우 피해야 할 약물이나 주의사항이 있나요?"
            ),
            gold_answer_generator=None,
            cohort_specific="No-Meds"
        ))
        
        # ===== T6: 일관성/회상 =====
        self.add_template(QuestionTemplate(
            template_id="T6_comprehensive_plan",
            turn_type=TurnType.T6_CONSISTENCY,
            goal="전체 컨텍스트 반영 + 일관성",
            prerequisites=[],
            fallback_id=None,
            disclosed_slots=[],  # 전체 누적
            required_slots_in_answer=["diagnosis", "medications", "lab_results"],
            template_text=(
                "지금까지 제 정보 (나이, 진단, 복용약, 알레르기, 정정된 검사 수치 포함)를 "
                "모두 반영해서, 다음 1주일 동안의 행동 계획을 3단계로 정리해 주세요.\n"
                "(예: 1. 약물 복용, 2. 식이 조절, 3. 재검사 일정)"
            ),
            gold_answer_generator=None
        ))
        
        self.add_template(QuestionTemplate(
            template_id="T6_recall_all_info",
            turn_type=TurnType.T6_CONSISTENCY,
            goal="전체 정보 회상 + 일관성 검증",
            prerequisites=[],
            fallback_id=None,
            disclosed_slots=[],
            required_slots_in_answer=["diagnosis", "medications", "allergy", "lab_results"],
            template_text=(
                "제가 지금까지 말한 정보를 요약해 주세요:\n"
                "1. 진단명\n"
                "2. 복용약 (정정 사항 반영)\n"
                "3. 알레르기 (정정 사항 반영)\n"
                "4. 최근 검사 수치 (정정 사항 반영)\n\n"
                "각 항목이 정확한지 확인해 주세요."
            ),
            gold_answer_generator="generate_t6_recall_gold"
        ))
    
    def add_template(self, template: QuestionTemplate):
        """템플릿 추가"""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[QuestionTemplate]:
        """템플릿 조회"""
        return self.templates.get(template_id)
    
    def get_templates_by_turn_type(self, turn_type: TurnType) -> List[QuestionTemplate]:
        """턴 타입별 템플릿 조회"""
        return [t for t in self.templates.values() if t.turn_type == turn_type]
    
    def select_template_with_gating(
        self, 
        turn_type: TurnType, 
        patient_data: Dict[str, Any],
        cohort: str
    ) -> QuestionTemplate:
        """
        게이팅 로직으로 템플릿 선택
        prerequisites 충족 여부 확인 + fallback
        """
        candidates = self.get_templates_by_turn_type(turn_type)
        
        # 코호트 필터링
        candidates = [
            t for t in candidates 
            if t.cohort_specific is None or t.cohort_specific == cohort
        ]
        
        if not candidates:
            raise ValueError(f"No templates found for {turn_type} (after cohort filtering)")
        
        # prerequisites 충족하는 템플릿 찾기
        for template in candidates:
            if self._check_prerequisites(template.prerequisites, patient_data):
                return template
        
        # 모두 불충족 시 fallback
        for template in candidates:
            if template.fallback_id:
                fallback = self.get_template(template.fallback_id)
                if fallback and self._check_prerequisites(fallback.prerequisites, patient_data):
                    return fallback
        
        # 최후의 수단: prerequisites 없는 템플릿
        for template in candidates:
            if not template.prerequisites:
                return template
        
        raise ValueError(f"No suitable template for {turn_type} in cohort {cohort}")
    
    def _check_prerequisites(self, prerequisites: List[str], patient_data: Dict[str, Any]) -> bool:
        """prerequisites 충족 여부 확인"""
        for prereq in prerequisites:
            if prereq == "lab_results_trend":
                # 특수 조건: 검사 2회 이상
                if "lab_results" not in patient_data or len(patient_data["lab_results"]) < 2:
                    return False
            elif prereq == "medications":
                # 약물이 있고 비어있지 않은지
                if "medications" not in patient_data or not patient_data["medications"]:
                    return False
            else:
                # 일반 슬롯 존재 여부
                if prereq not in patient_data or patient_data[prereq] in [None, "", [], "없음"]:
                    return False
        return True


def main():
    """질문은행 테스트"""
    bank = QuestionBank()
    
    print(f"총 {len(bank.templates)}개 템플릿 로드됨\n")
    
    # 턴 타입별 개수
    for turn_type in TurnType:
        templates = bank.get_templates_by_turn_type(turn_type)
        print(f"{turn_type.value}: {len(templates)}개")
    
    # 샘플 템플릿 출력
    print("\n[샘플 템플릿: T1_basic_info]")
    t1 = bank.get_template("T1_basic_info")
    if t1:
        print(f"목적: {t1.goal}")
        print(f"Prerequisites: {t1.prerequisites}")
        print(f"Disclosed slots: {t1.disclosed_slots}")
        print(f"템플릿:\n{t1.template_text}")


if __name__ == "__main__":
    main()

