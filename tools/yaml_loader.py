"""
YAML 설정 로더 (question_templates.yaml, eval_rubric.yaml)
SSOT(Single Source of Truth) 기반 설정 관리
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import json
import re


class QuestionTemplateLoader:
    """
    question_templates.yaml 로더
    멀티턴 케이스 정의를 로드하고 검증
    """

    def __init__(self, yaml_path: str = "configs/question_templates.yaml"):
        self.yaml_path = Path(yaml_path)
        self.config = self._load_yaml()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        """YAML 파일 로드"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"question_templates.yaml not found at {self.yaml_path}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """설정 검증"""
        required_keys = ['version', 'language', 'slot_schema', 'cases']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key: {key}")

        # 케이스 검증
        for case in self.config['cases']:
            self._validate_case(case)

    def _validate_case(self, case: Dict[str, Any]):
        """케이스 검증"""
        required_case_keys = ['case_id', 'domain_id', 'q_type', 'turns']
        for key in required_case_keys:
            if key not in case:
                raise ValueError(f"Case missing required key: {key} in case {case.get('case_id', 'unknown')}")

        # 턴 검증
        for turn in case['turns']:
            self._validate_turn(turn, case['case_id'])

    def _validate_turn(self, turn: Dict[str, Any], case_id: str):
        """턴 검증"""
        required_turn_keys = ['turn_id', 'role', 'utterance', 'required_slots', 'expected_slot_updates']
        for key in required_turn_keys:
            if key not in turn:
                raise ValueError(f"Turn missing required key: {key} in case {case_id}, turn {turn.get('turn_id', 'unknown')}")

    def get_all_cases(self) -> List[Dict[str, Any]]:
        """모든 케이스 반환"""
        return self.config['cases']

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """케이스 ID로 조회"""
        for case in self.config['cases']:
            if case['case_id'] == case_id:
                return case
        return None

    def get_cases_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """난이도별 케이스 조회"""
        return [case for case in self.config['cases'] if case.get('difficulty') == difficulty]

    def get_cases_by_domain(self, domain_id: int) -> List[Dict[str, Any]]:
        """도메인별 케이스 조회"""
        return [case for case in self.config['cases'] if case['domain_id'] == domain_id]

    def get_slot_schema(self) -> Dict[str, Any]:
        """슬롯 스키마 반환"""
        return self.config['slot_schema']

    def get_global_policies(self) -> Dict[str, Any]:
        """전역 정책 반환"""
        return self.config.get('global_policies', {})

    def export_to_jsonl(self, output_path: str):
        """
        멀티턴 케이스를 JSONL로 내보내기 (평가용)
        각 턴을 개별 레코드로 저장
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for case in self.config['cases']:
                # 누적 슬롯 추적
                accumulated_slots = {}

                for turn in case['turns']:
                    # 슬롯 업데이트 적용
                    if turn.get('expected_slot_updates'):
                        accumulated_slots.update(turn['expected_slot_updates'])

                    # JSONL 레코드 생성
                    record = {
                        'case_id': case['case_id'],
                        'domain_id': case['domain_id'],
                        'q_type': case['q_type'],
                        'difficulty': case.get('difficulty', 'medium'),
                        'turn_id': turn['turn_id'],
                        'role': turn['role'],
                        'utterance': turn['utterance'],
                        'required_slots': turn['required_slots'],
                        'expected_slot_updates': turn.get('expected_slot_updates', {}),
                        'accumulated_slots': accumulated_slots.copy(),
                        'update_key': turn.get('update_key', 'none'),
                        'trap': turn.get('trap', False),
                        'expected_agent_behavior': turn.get('expected_agent_behavior', []),
                        'retrieval_plan': case.get('retrieval_plan', {})
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Exported {len(self.config['cases'])} cases to {output_path}")


class EvalRubricLoader:
    """
    eval_rubric.yaml 로더
    평가 기준 로드 및 적용
    """

    def __init__(self, yaml_path: str = "configs/eval_rubric.yaml"):
        self.yaml_path = Path(yaml_path)
        self.config = self._load_yaml()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        """YAML 파일 로드"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"eval_rubric.yaml not found at {self.yaml_path}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """설정 검증"""
        required_keys = ['version', 'gates', 'rule_based', 'llm_judge', 'reporting']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key: {key}")

    def get_gates(self) -> Dict[str, Any]:
        """게이트 설정 반환"""
        return self.config['gates']

    def get_rule_based_config(self) -> Dict[str, Any]:
        """규칙 기반 평가 설정 반환"""
        return self.config['rule_based']

    def get_llm_judge_config(self) -> Dict[str, Any]:
        """LLM Judge 설정 반환"""
        return self.config['llm_judge']

    def get_reporting_config(self) -> Dict[str, Any]:
        """리포팅 설정 반환"""
        return self.config['reporting']

    def get_slot_validation_rules(self) -> Dict[str, Any]:
        """슬롯 검증 규칙 반환"""
        return self.config.get('slot_validation', {})

    def get_turn_evaluation_criteria(self, turn_type: str = "normal_turn") -> Dict[str, float]:
        """턴 타입별 평가 기준 가중치 반환"""
        criteria = self.config.get('turn_evaluation_criteria', {})
        return criteria.get(turn_type, criteria.get('normal_turn', {}))

    def apply_gates(self, state: Dict[str, Any], answer: str) -> Dict[str, Any]:
        """
        게이트 적용 (fail-fast)

        Returns:
            gate_result: {
                'pass': bool,
                'reasons': List[str],
                'capped_factuality': Optional[float]
            }
        """
        gates = self.get_gates()
        result = {
            'pass': True,
            'reasons': [],
            'capped_factuality': None
        }

        # 1. High stakes 키워드 체크
        high_stakes_keywords = gates.get('high_stakes_keywords', [])
        has_high_stakes = any(kw in answer for kw in high_stakes_keywords)

        if has_high_stakes:
            # 안전 문구 확인
            rule_based = self.get_rule_based_config()
            safety_phrases = rule_based.get('Safety_Phrases', {})
            required_any = safety_phrases.get('required_any', [])

            has_safety_phrase = any(phrase in answer for phrase in required_any)

            if not has_safety_phrase and gates.get('require_emergency_action_if_redflag'):
                result['pass'] = False
                result['reasons'].append('high_stakes_without_safety_phrase')

        # 2. TS 근거 부족 체크
        ts_count = state.get('qc_ts_count', 0)
        ts_chars = state.get('qc_ts_chars', 0)

        min_ts_count = gates.get('min_ts_count', 2)
        min_ts_chars = gates.get('min_ts_chars_total', 240)
        cap_factuality = gates.get('cap_factuality_if_ts_insufficient', 0.4)

        if ts_count < min_ts_count or ts_chars < min_ts_chars:
            result['capped_factuality'] = cap_factuality
            result['reasons'].append(f'ts_insufficient (count={ts_count}, chars={ts_chars})')

        return result

    def calculate_rule_based_scores(
        self,
        answer: str,
        required_slots: List[str],
        extracted_slots: Dict[str, Any],
        ts_context: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        규칙 기반 점수 계산

        Returns:
            scores: {
                'CUS': float (0~1),
                'TS_Use': float (0~1),
                'Safety': float (0~1)
            }
        """
        rule_config = self.get_rule_based_config()
        scores = {}

        # 1. CUS (Context Use Score)
        if rule_config.get('CUS', {}).get('enabled'):
            cus_score = self._calculate_cus(answer, required_slots, extracted_slots, rule_config['CUS'])
            scores['CUS'] = cus_score

        # 2. TS Use
        if rule_config.get('TS_Use', {}).get('enabled'):
            ts_use_score = self._calculate_ts_use(answer, ts_context, rule_config['TS_Use'])
            scores['TS_Use'] = ts_use_score

        # 3. Safety Phrases
        if rule_config.get('Safety_Phrases', {}).get('enabled'):
            safety_score = self._calculate_safety(answer, rule_config['Safety_Phrases'])
            scores['Safety'] = safety_score

        return scores

    def _calculate_cus(
        self,
        answer: str,
        required_slots: List[str],
        extracted_slots: Dict[str, Any],
        cus_config: Dict[str, Any]
    ) -> float:
        """CUS (Context Use Score) 계산"""
        if not required_slots:
            return 1.0

        weights = cus_config.get('weights', {})
        hit_scores = cus_config.get('hit_scores', {'exact': 1.0, 'partial': 0.5, 'miss': 0.0})

        total_weight = 0.0
        weighted_score = 0.0

        for slot in required_slots:
            weight = weights.get(slot, 0.5)  # 기본 가중치
            total_weight += weight

            # 슬롯 값 확인
            slot_value = self._get_nested_value(extracted_slots, slot)

            if slot_value is not None:
                # 답변에서 슬롯 값 언급 확인
                slot_str = str(slot_value)
                if slot_str.lower() in answer.lower():
                    weighted_score += weight * hit_scores['exact']
                elif any(word in answer.lower() for word in slot_str.lower().split()):
                    weighted_score += weight * hit_scores['partial']
                else:
                    weighted_score += weight * hit_scores['miss']
            else:
                weighted_score += weight * hit_scores['miss']

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _calculate_ts_use(
        self,
        answer: str,
        ts_context: List[Dict[str, Any]],
        ts_config: Dict[str, Any]
    ) -> float:
        """TS 근거 사용 점수 계산"""
        if not ts_context:
            return 0.0

        min_overlap_ratio = ts_config.get('min_overlap_ratio', 0.08)

        # TS 텍스트 추출
        ts_texts = []
        for doc in ts_context:
            text = doc.get('text', '')
            if isinstance(text, str):
                ts_texts.append(text)

        if not ts_texts:
            return 0.0

        # 토큰 오버랩 계산
        answer_tokens = set(answer.lower().split())
        ts_tokens = set(' '.join(ts_texts).lower().split())

        overlap = len(answer_tokens & ts_tokens)
        total_answer_tokens = len(answer_tokens)

        if total_answer_tokens == 0:
            return 0.0

        overlap_ratio = overlap / total_answer_tokens

        # 최소 비율 이상이면 1.0, 아니면 비율 스코어
        return 1.0 if overlap_ratio >= min_overlap_ratio else overlap_ratio / min_overlap_ratio

    def _calculate_safety(
        self,
        answer: str,
        safety_config: Dict[str, Any]
    ) -> float:
        """안전 문구 점수 계산"""
        required_any = safety_config.get('required_any', [])
        forbidden_any = safety_config.get('forbidden_any', [])

        # 필수 문구 확인
        has_required = any(phrase in answer for phrase in required_any)

        # 금지 문구 확인
        has_forbidden = any(phrase in answer for phrase in forbidden_any)

        if has_forbidden:
            return 0.0
        elif has_required:
            return 1.0
        else:
            return 0.5  # 중립

    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """중첩된 키 경로에서 값 추출 (예: 'patient.age')"""
        keys = key_path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value

    def validate_slot_value(self, slot_name: str, value: Any) -> Dict[str, Any]:
        """
        슬롯 값 검증

        Returns:
            {
                'valid': bool,
                'error_msg': Optional[str]
            }
        """
        validation_rules = self.get_slot_validation_rules()

        if slot_name not in validation_rules:
            return {'valid': True, 'error_msg': None}

        rule = validation_rules[slot_name]
        value_type = rule.get('type')
        error_msg = rule.get('error_msg', f'Invalid value for {slot_name}')

        # 타입별 검증
        if value_type == 'int':
            try:
                int_value = int(value)
                range_min, range_max = rule.get('range', [None, None])
                if range_min is not None and int_value < range_min:
                    return {'valid': False, 'error_msg': error_msg}
                if range_max is not None and int_value > range_max:
                    return {'valid': False, 'error_msg': error_msg}
                return {'valid': True, 'error_msg': None}
            except (ValueError, TypeError):
                return {'valid': False, 'error_msg': error_msg}

        elif value_type == 'enum':
            valid_values = rule.get('values', [])
            if value not in valid_values:
                return {'valid': False, 'error_msg': error_msg}
            return {'valid': True, 'error_msg': None}

        elif value_type == 'float_string':
            try:
                float_value = float(value)
                range_min, range_max = rule.get('range', [None, None])
                if range_min is not None and float_value < range_min:
                    return {'valid': False, 'error_msg': error_msg}
                if range_max is not None and float_value > range_max:
                    return {'valid': False, 'error_msg': error_msg}
                return {'valid': True, 'error_msg': None}
            except (ValueError, TypeError):
                return {'valid': False, 'error_msg': error_msg}

        elif value_type == 'bp_string':
            pattern = rule.get('pattern', '')
            if not re.match(pattern, str(value)):
                return {'valid': False, 'error_msg': error_msg}
            return {'valid': True, 'error_msg': None}

        return {'valid': True, 'error_msg': None}


def main():
    """테스트"""
    print("=== QuestionTemplateLoader 테스트 ===")
    qt_loader = QuestionTemplateLoader()

    print(f"\n총 케이스 수: {len(qt_loader.get_all_cases())}")

    for case in qt_loader.get_all_cases():
        print(f"\n케이스: {case['case_id']}")
        print(f"  도메인: {case['domain_id']}, 난이도: {case.get('difficulty', 'N/A')}")
        print(f"  턴 수: {len(case['turns'])}")

    # JSONL 내보내기
    qt_loader.export_to_jsonl("experiments/multiturn/eval_cases.jsonl")

    print("\n=== EvalRubricLoader 테스트 ===")
    rubric_loader = EvalRubricLoader()

    gates = rubric_loader.get_gates()
    print(f"\n게이트 설정:")
    print(f"  최소 TS 개수: {gates['min_ts_count']}")
    print(f"  최소 TS 문자: {gates['min_ts_chars_total']}")

    rule_based = rubric_loader.get_rule_based_config()
    print(f"\n규칙 기반 평가:")
    print(f"  CUS 활성화: {rule_based['CUS']['enabled']}")
    print(f"  TS_Use 활성화: {rule_based['TS_Use']['enabled']}")
    print(f"  Safety_Phrases 활성화: {rule_based['Safety_Phrases']['enabled']}")

    # 슬롯 검증 테스트
    print("\n=== 슬롯 검증 테스트 ===")
    test_cases = [
        ('patient.age', 25, True),
        ('patient.age', 150, False),
        ('patient.sex', 'M', True),
        ('patient.sex', 'X', False),
        ('labs.hba1c', '7.5', True),
        ('labs.hba1c', '20.0', False),
        ('labs.bp', '120/80', True),
        ('labs.bp', '120', False),
    ]

    for slot_name, value, expected in test_cases:
        result = rubric_loader.validate_slot_value(slot_name, value)
        status = "✓" if result['valid'] == expected else "✗"
        print(f"  {status} {slot_name}={value}: {result['valid']} (expected {expected})")


if __name__ == "__main__":
    main()
