"""
환자 시나리오 카드 생성기
재현성 확보를 위한 구조화된 환자 데이터 생성
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json


@dataclass
class Medication:
    """약물 정보"""
    name: str
    dosage: str
    frequency: str  # QD, BID, TID 등


@dataclass
class LabResult:
    """검사 결과"""
    date: str  # YYYY-MM-DD
    test: str
    value: float
    unit: str = ""


@dataclass
class PatientScenario:
    """환자 시나리오 카드 (Canonical State)"""
    patient_id: str
    name: str
    age: int
    gender: str
    diagnosis: List[str]
    medications: List[Medication]
    allergy: str  # "없음" 또는 구체적 알레르기
    lab_results: List[LabResult]
    last_visit_note: str
    preferences: str
    
    # 메타데이터
    cohort: str  # "Full", "No-Meds", "No-Trend"
    scenario_level: str  # "L1", "L2", "L3"
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화"""
        data = asdict(self)
        data['medications'] = [asdict(m) for m in self.medications]
        data['lab_results'] = [asdict(lr) for lr in self.lab_results]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatientScenario':
        """JSON 역직렬화"""
        data['medications'] = [Medication(**m) for m in data['medications']]
        data['lab_results'] = [LabResult(**lr) for lr in data['lab_results']]
        return cls(**data)


class PatientScenarioGenerator:
    """환자 시나리오 생성기"""
    
    def __init__(self):
        self.scenarios: List[PatientScenario] = []
    
    def generate_full_cohort_scenarios(self, count: int = 7) -> List[PatientScenario]:
        """Full Cohort 시나리오 생성 (모든 슬롯 있음)"""
        scenarios = []
        
        # 시나리오 1: 당뇨병 + 고혈압 (L2 - 중간)
        scenarios.append(PatientScenario(
            patient_id="P-F001",
            name="김철수",
            age=58,
            gender="Male",
            diagnosis=["Type 2 Diabetes", "Hypertension"],
            medications=[
                Medication("Metformin", "500mg", "BID"),
                Medication("Amlodipine", "5mg", "QD")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-01-15", "HbA1c", 7.8, "%"),
                LabResult("2024-04-20", "HbA1c", 7.2, "%"),
                LabResult("2024-04-20", "BP", 135, "mmHg (수축기)")
            ],
            last_visit_note="환자가 최근 불규칙한 식사로 인해 속쓰림을 호소함.",
            preferences="짧고 명확한 설명 선호",
            cohort="Full",
            scenario_level="L2"
        ))
        
        # 시나리오 2: 고혈압 단독 (L1 - 쉬움)
        scenarios.append(PatientScenario(
            patient_id="P-F002",
            name="이영희",
            age=62,
            gender="Female",
            diagnosis=["Hypertension"],
            medications=[
                Medication("Losartan", "50mg", "QD")
            ],
            allergy="Penicillin",
            lab_results=[
                LabResult("2024-03-10", "BP", 145, "mmHg (수축기)"),
                LabResult("2024-05-15", "BP", 132, "mmHg (수축기)")
            ],
            last_visit_note="혈압 조절 양호. 약물 순응도 우수.",
            preferences="자세한 설명 선호",
            cohort="Full",
            scenario_level="L1"
        ))
        
        # 시나리오 3: 당뇨병 + 고지혈증 (L3 - 어려움, 정정 턴 포함)
        scenarios.append(PatientScenario(
            patient_id="P-F003",
            name="박민수",
            age=55,
            gender="Male",
            diagnosis=["Type 2 Diabetes", "Dyslipidemia"],
            medications=[
                Medication("Metformin", "1000mg", "BID"),
                Medication("Atorvastatin", "20mg", "QD")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-02-01", "HbA1c", 8.5, "%"),
                LabResult("2024-05-01", "HbA1c", 8.2, "%"),  # 정정 대상: 실제 9.2%
                LabResult("2024-05-01", "LDL", 110, "mg/dL")
            ],
            last_visit_note="혈당 조절 불량. 식이 조절 필요.",
            preferences="단계별 계획 선호",
            cohort="Full",
            scenario_level="L3"
        ))
        
        # 시나리오 4: 천식 + 알레르기 비염 (L2)
        scenarios.append(PatientScenario(
            patient_id="P-F004",
            name="최지은",
            age=34,
            gender="Female",
            diagnosis=["Asthma", "Allergic Rhinitis"],
            medications=[
                Medication("Fluticasone", "250mcg", "BID (흡입)"),
                Medication("Montelukast", "10mg", "QD")
            ],
            allergy="Aspirin",
            lab_results=[
                LabResult("2024-03-20", "FEV1", 78, "% predicted"),
                LabResult("2024-05-20", "FEV1", 85, "% predicted")
            ],
            last_visit_note="최근 호흡곤란 증상 개선됨. 흡입기 사용법 교육 완료.",
            preferences="부작용 정보 중요시",
            cohort="Full",
            scenario_level="L2"
        ))
        
        # 시나리오 5: 갑상선기능저하증 (L1)
        scenarios.append(PatientScenario(
            patient_id="P-F005",
            name="정수민",
            age=45,
            gender="Female",
            diagnosis=["Hypothyroidism"],
            medications=[
                Medication("Levothyroxine", "100mcg", "QD (아침 공복)")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-01-10", "TSH", 8.5, "mIU/L"),
                LabResult("2024-04-10", "TSH", 3.2, "mIU/L")
            ],
            last_visit_note="TSH 수치 정상화. 약물 용량 유지.",
            preferences="복용 시간 중요시",
            cohort="Full",
            scenario_level="L1"
        ))
        
        # 시나리오 6: 심부전 + 당뇨병 (L3 - 복잡)
        scenarios.append(PatientScenario(
            patient_id="P-F006",
            name="강대호",
            age=68,
            gender="Male",
            diagnosis=["Heart Failure (NYHA Class II)", "Type 2 Diabetes"],
            medications=[
                Medication("Furosemide", "40mg", "QD"),
                Medication("Carvedilol", "12.5mg", "BID"),
                Medication("Metformin", "500mg", "BID")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-03-01", "BNP", 450, "pg/mL"),
                LabResult("2024-05-01", "BNP", 320, "pg/mL"),
                LabResult("2024-05-01", "HbA1c", 7.5, "%")
            ],
            last_visit_note="부종 감소. 호흡곤란 개선. 체중 감소 권고.",
            preferences="금기사항 중요시",
            cohort="Full",
            scenario_level="L3"
        ))
        
        # 시나리오 7: 류마티스 관절염 (L2)
        scenarios.append(PatientScenario(
            patient_id="P-F007",
            name="윤서연",
            age=52,
            gender="Female",
            diagnosis=["Rheumatoid Arthritis"],
            medications=[
                Medication("Methotrexate", "15mg", "주 1회"),
                Medication("Folic acid", "5mg", "주 1회 (MTX 다음날)")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-02-15", "CRP", 2.8, "mg/dL"),
                LabResult("2024-05-15", "CRP", 1.2, "mg/dL"),
                LabResult("2024-05-15", "ESR", 22, "mm/hr")
            ],
            last_visit_note="관절 통증 감소. 아침 경직 개선.",
            preferences="생활 습관 조언 선호",
            cohort="Full",
            scenario_level="L2"
        ))
        
        return scenarios[:count]
    
    def generate_no_meds_cohort_scenarios(self, count: int = 3) -> List[PatientScenario]:
        """No-Meds Cohort 시나리오 생성 (복용약 없음)"""
        scenarios = []
        
        # 시나리오 1: 경계성 고혈압 (약물 치료 전)
        scenarios.append(PatientScenario(
            patient_id="P-NM001",
            name="홍길동",
            age=48,
            gender="Male",
            diagnosis=["Prehypertension"],
            medications=[],  # 빈 리스트
            allergy="없음",
            lab_results=[
                LabResult("2024-03-01", "BP", 138, "mmHg (수축기)"),
                LabResult("2024-05-01", "BP", 135, "mmHg (수축기)")
            ],
            last_visit_note="생활습관 개선 중. 약물 치료는 보류.",
            preferences="비약물적 치료 선호",
            cohort="No-Meds",
            scenario_level="L1"
        ))
        
        # 시나리오 2: 경증 불면증
        scenarios.append(PatientScenario(
            patient_id="P-NM002",
            name="김수진",
            age=38,
            gender="Female",
            diagnosis=["Mild Insomnia"],
            medications=[],
            allergy="없음",
            lab_results=[
                LabResult("2024-04-01", "Sleep Quality Score", 45, "점 (0-100)")
            ],
            last_visit_note="수면 위생 교육 시행. 인지행동치료 권고.",
            preferences="수면제 회피 희망",
            cohort="No-Meds",
            scenario_level="L1"
        ))
        
        # 시나리오 3: 건강검진 이상 소견 (경계성 혈당)
        scenarios.append(PatientScenario(
            patient_id="P-NM003",
            name="이준호",
            age=42,
            gender="Male",
            diagnosis=["Prediabetes"],
            medications=[],
            allergy="없음",
            lab_results=[
                LabResult("2024-01-15", "Fasting Glucose", 115, "mg/dL"),
                LabResult("2024-04-15", "Fasting Glucose", 108, "mg/dL")
            ],
            last_visit_note="체중 감량 5kg 달성. 식이·운동 요법 지속 권고.",
            preferences="예방적 관리 중요시",
            cohort="No-Meds",
            scenario_level="L2"
        ))
        
        return scenarios[:count]
    
    def generate_no_trend_cohort_scenarios(self, count: int = 3) -> List[PatientScenario]:
        """No-Trend Cohort 시나리오 생성 (검사 1회만)"""
        scenarios = []
        
        # 시나리오 1: 신규 진단 당뇨병
        scenarios.append(PatientScenario(
            patient_id="P-NT001",
            name="박지훈",
            age=51,
            gender="Male",
            diagnosis=["Type 2 Diabetes (신규 진단)"],
            medications=[
                Medication("Metformin", "500mg", "BID")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-05-20", "HbA1c", 8.2, "%")  # 1회만
            ],
            last_visit_note="신규 진단. 약물 치료 시작. 3개월 후 재검 예정.",
            preferences="초기 교육 필요",
            cohort="No-Trend",
            scenario_level="L2"
        ))
        
        # 시나리오 2: 급성 요로감염 (항생제 치료 중)
        scenarios.append(PatientScenario(
            patient_id="P-NT002",
            name="최은지",
            age=29,
            gender="Female",
            diagnosis=["Acute Cystitis"],
            medications=[
                Medication("Ciprofloxacin", "500mg", "BID (5일)")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-05-18", "Urinalysis", 1, "WBC 다수")  # 정성적
            ],
            last_visit_note="항생제 치료 시작. 증상 호전 시 재검 불필요.",
            preferences="빠른 회복 희망",
            cohort="No-Trend",
            scenario_level="L1"
        ))
        
        # 시나리오 3: 고혈압 초진
        scenarios.append(PatientScenario(
            patient_id="P-NT003",
            name="정민철",
            age=56,
            gender="Male",
            diagnosis=["Hypertension (초진)"],
            medications=[
                Medication("Amlodipine", "5mg", "QD")
            ],
            allergy="없음",
            lab_results=[
                LabResult("2024-05-22", "BP", 152, "mmHg (수축기)")
            ],
            last_visit_note="약물 치료 시작. 2주 후 혈압 재측정 예정.",
            preferences="부작용 우려",
            cohort="No-Trend",
            scenario_level="L1"
        ))
        
        return scenarios[:count]
    
    def generate_all_scenarios(self) -> List[PatientScenario]:
        """전체 시나리오 생성 (20개)"""
        scenarios = []
        scenarios.extend(self.generate_full_cohort_scenarios(7))
        scenarios.extend(self.generate_no_meds_cohort_scenarios(3))
        scenarios.extend(self.generate_no_trend_cohort_scenarios(3))
        
        # 추가 다양성 확보 (필요시 확장)
        # scenarios.extend(self.generate_edge_case_scenarios(7))
        
        self.scenarios = scenarios
        return scenarios
    
    def save_to_json(self, filepath: str):
        """JSON 파일로 저장"""
        data = [s.to_dict() for s in self.scenarios]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> List[PatientScenario]:
        """JSON 파일에서 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [PatientScenario.from_dict(d) for d in data]


def main():
    """시나리오 생성 및 저장"""
    generator = PatientScenarioGenerator()
    scenarios = generator.generate_all_scenarios()
    
    print(f"총 {len(scenarios)}개 시나리오 생성:")
    print(f"  - Full Cohort: {len([s for s in scenarios if s.cohort == 'Full'])}개")
    print(f"  - No-Meds Cohort: {len([s for s in scenarios if s.cohort == 'No-Meds'])}개")
    print(f"  - No-Trend Cohort: {len([s for s in scenarios if s.cohort == 'No-Trend'])}개")
    
    # 저장
    output_path = "data/multiturn/patient_scenarios.json"
    generator.save_to_json(output_path)
    print(f"\n저장 완료: {output_path}")
    
    # 샘플 출력
    print("\n[샘플 시나리오]")
    sample = scenarios[0]
    print(f"ID: {sample.patient_id}, 이름: {sample.name}, 코호트: {sample.cohort}")
    print(f"진단: {', '.join(sample.diagnosis)}")
    print(f"약물: {len(sample.medications)}개")
    print(f"검사: {len(sample.lab_results)}개")


if __name__ == "__main__":
    main()

