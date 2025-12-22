"""
멀티턴 대화 테스트 시스템
재현성과 객관성을 확보한 프로토콜 기반 평가 프레임워크
"""

from .patient_scenarios import (
    PatientScenario,
    PatientScenarioGenerator,
    Medication,
    LabResult
)

from .question_bank import (
    QuestionBank,
    QuestionTemplate,
    TurnType
)

from .multiturn_simulator import (
    MultiTurnSimulator,
    PatientResponseGenerator,
    MultiTurnDialogue,
    Turn
)

from .evaluation_rubric import (
    EvaluationRubric,
    EvaluationResult,
    SubScore,
    EvaluationMetric,
    LLMJudge
)

__all__ = [
    # Patient scenarios
    'PatientScenario',
    'PatientScenarioGenerator',
    'Medication',
    'LabResult',
    
    # Question bank
    'QuestionBank',
    'QuestionTemplate',
    'TurnType',
    
    # Simulator
    'MultiTurnSimulator',
    'PatientResponseGenerator',
    'MultiTurnDialogue',
    'Turn',
    
    # Evaluation
    'EvaluationRubric',
    'EvaluationResult',
    'SubScore',
    'EvaluationMetric',
    'LLMJudge',
]

__version__ = '0.1.0'

