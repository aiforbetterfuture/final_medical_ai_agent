"""
5턴 멀티턴 질문 스크립트 자동 생성
슬롯을 받아서 5턴 질문 템플릿으로 스크립트 생성
3·4턴은 의도적으로 맥락이 필요하지만 질문에는 드러나지 않게 설계
"""

from typing import Dict, List, Any


class SyntheaScriptGenerator:
    """5턴 멀티턴 질문 스크립트 생성기"""
    
    def __init__(self):
        pass
    
    def generate_5turn_script(self, slots: Dict[str, Any]) -> List[str]:
        """
        5턴 질문 스크립트 생성
        
        Args:
            slots: build_slots()로 생성된 슬롯 딕셔너리
        
        Returns:
            5개 질문 리스트 (턴 1~5)
        """
        age = slots.get('age', 0)
        sex = slots.get('sex', '남성')
        primary_condition = slots.get('primary_condition', '기저질환')
        chief_symptom = slots.get('chief_symptom', '불편감')
        key_meds = slots.get('key_meds', [])
        key_labs = slots.get('key_labs', {})
        key_vitals = slots.get('key_vitals', {})
        major_procedures = slots.get('major_procedures', [])
        
        # Turn 1: 필수 명시 (인구통계+질환+증상)
        t1 = self._generate_turn1(age, sex, primary_condition, chief_symptom)
        
        # Turn 2: 명시 (약+검사/바이탈+시술)
        t2 = self._generate_turn2(key_meds, key_labs, key_vitals, major_procedures)
        
        # Turn 3: 운동 (의도적으로 맥락 비명시)
        t3 = self._generate_turn3()
        
        # Turn 4: 식단 (의도적으로 맥락 비명시)
        t4 = self._generate_turn4()
        
        # Turn 5: 통합 플랜
        t5 = self._generate_turn5()
        
        return [t1, t2, t3, t4, t5]
    
    def _generate_turn1(self, age: int, sex: str, primary_condition: str, chief_symptom: str) -> str:
        """Turn 1: 인구통계 + 주요 질환 + 증상"""
        return f"{age}세 {sex}인데 {primary_condition}이(가) 있고, 최근 {chief_symptom}이(가) 심해졌어요. 무엇부터 점검해야 하나요?"
    
    def _generate_turn2(self, key_meds: List[str], key_labs: Dict[str, str], 
                        key_vitals: Dict[str, str], major_procedures: List[str]) -> str:
        """Turn 2: 약물 + 검사/바이탈 + 시술"""
        # 약물 텍스트
        if key_meds:
            meds_txt = ", ".join(key_meds[:3])
        else:
            meds_txt = "처방약"
        
        # 검사 결과 텍스트
        lab_items = []
        for k, v in list(key_labs.items())[:4]:
            lab_items.append(f"{k} {v}")
        labs_txt = ", ".join(lab_items) if lab_items else "몇몇 검사 수치"
        
        # 시술 텍스트
        if major_procedures:
            proc_txt = major_procedures[0]
        else:
            proc_txt = "시술/검사를"
        
        return f"현재 {meds_txt}를 복용 중이고, 최근 검사에서 {labs_txt}가 나왔어요. (예전에 {proc_txt}도 했습니다.) 제 상황에서 관리 우선순위가 뭘까요?"
    
    def _generate_turn3(self) -> str:
        """
        Turn 3: 운동 (의도적으로 맥락 비명시)
        질환/수치/약을 질문에 재언급하지 않음
        Agent는 Turn 1, 2의 정보를 메모리에서 끌어와야 함
        """
        return "운동을 시작하려고 하는데, 제 상태에서 '해도 되는 강도'와 '피해야 할 신호'를 어떻게 정하면 좋을까요? 걷기/근력/인터벌 중 우선순위도 알려주세요."
    
    def _generate_turn4(self) -> str:
        """
        Turn 4: 식단 (의도적으로 맥락 비명시)
        질환/수치/약을 질문에 재언급하지 않음
        Agent는 Turn 1, 2의 정보를 메모리에서 끌어와야 함
        """
        return "식단도 바꾸라고 하는데 복잡한 건 못 하겠어요. 제 상황 기준으로 '가장 효과 큰 규칙 3개'만 정해주면 뭐가 될까요?"
    
    def _generate_turn5(self) -> str:
        """Turn 5: 통합 4주 플랜"""
        return "지금까지 제 얘기를 바탕으로 4주 실행계획(우선순위/모니터링 지표/재검 타이밍)을 만들어 주세요."

