"""
YAML 기반 통합 평가기
question_templates.yaml과 eval_rubric.yaml을 사용한 멀티턴 평가
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.yaml_loader import QuestionTemplateLoader, EvalRubricLoader


class YAMLBasedEvaluator:
    """
    YAML 설정 기반 멀티턴 평가기
    - question_templates.yaml로 테스트 케이스 로드
    - eval_rubric.yaml로 평가 기준 적용
    - Rule-based + LLM-as-judge 하이브리드 평가
    """

    def __init__(
        self,
        template_yaml: str = "configs/question_templates.yaml",
        rubric_yaml: str = "configs/eval_rubric.yaml",
        output_dir: str = "experiments/multiturn/yaml_eval_results"
    ):
        self.template_loader = QuestionTemplateLoader(template_yaml)
        self.rubric_loader = EvalRubricLoader(rubric_yaml)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_case(
        self,
        case_id: str,
        agent_runner: Any,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        단일 케이스 평가

        Args:
            case_id: 평가할 케이스 ID
            agent_runner: 에이전트 실행 함수 (question -> answer + state)
            verbose: 상세 로그 출력

        Returns:
            case_result: {
                'case_id': str,
                'turns': List[turn_result],
                'aggregate_scores': Dict[str, float],
                'metadata': Dict[str, Any]
            }
        """
        case = self.template_loader.get_case_by_id(case_id)
        if not case:
            raise ValueError(f"Case not found: {case_id}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"평가 케이스: {case_id}")
            print(f"설명: {case.get('description', 'N/A')}")
            print(f"난이도: {case.get('difficulty', 'N/A')}")
            print(f"턴 수: {len(case['turns'])}")
            print(f"{'='*60}\n")

        # 누적 슬롯 추적
        accumulated_slots = {}
        turn_results = []

        for turn in case['turns']:
            if verbose:
                print(f"\n[Turn {turn['turn_id']}]")
                print(f"사용자: {turn['utterance']}")

            # 슬롯 업데이트 적용
            if turn.get('expected_slot_updates'):
                accumulated_slots.update(turn['expected_slot_updates'])

            # 에이전트 실행
            try:
                agent_output = agent_runner(
                    question=turn['utterance'],
                    context=accumulated_slots,
                    case_metadata=case
                )

                answer = agent_output.get('answer', '')
                state = agent_output.get('state', {})

                if verbose:
                    print(f"에이전트: {answer[:200]}...")

            except Exception as e:
                if verbose:
                    print(f"에러 발생: {e}")
                answer = ""
                state = {}

            # 턴 평가
            turn_result = self.evaluate_turn(
                turn=turn,
                case=case,
                answer=answer,
                state=state,
                accumulated_slots=accumulated_slots,
                verbose=verbose
            )

            turn_results.append(turn_result)

        # 케이스 전체 집계
        aggregate_scores = self._aggregate_case_scores(turn_results)

        case_result = {
            'case_id': case_id,
            'domain_id': case['domain_id'],
            'q_type': case['q_type'],
            'difficulty': case.get('difficulty', 'medium'),
            'description': case.get('description', ''),
            'turns': turn_results,
            'aggregate_scores': aggregate_scores,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_turns': len(turn_results),
                'pass': aggregate_scores['weighted_mean'] >= 0.75
            }
        }

        return case_result

    def evaluate_turn(
        self,
        turn: Dict[str, Any],
        case: Dict[str, Any],
        answer: str,
        state: Dict[str, Any],
        accumulated_slots: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        단일 턴 평가

        Returns:
            turn_result: {
                'turn_id': int,
                'utterance': str,
                'answer': str,
                'rule_scores': Dict[str, float],
                'gate_result': Dict[str, Any],
                'slot_validation': Dict[str, Any],
                'overall_score': float
            }
        """
        # 1. 게이트 적용
        gate_result = self.rubric_loader.apply_gates(state, answer)

        if verbose and not gate_result['pass']:
            print(f"  ⚠ 게이트 실패: {gate_result['reasons']}")

        # 2. 슬롯 검증
        slot_validation = self._validate_slots(turn, accumulated_slots)

        if verbose and not slot_validation['all_valid']:
            print(f"  ⚠ 슬롯 검증 실패: {slot_validation['errors']}")

        # 3. 규칙 기반 평가
        rule_scores = self.rubric_loader.calculate_rule_based_scores(
            answer=answer,
            required_slots=turn['required_slots'],
            extracted_slots=accumulated_slots,
            ts_context=state.get('ts_context', [])
        )

        if verbose:
            print(f"  규칙 기반 점수:")
            for metric, score in rule_scores.items():
                print(f"    {metric}: {score:.3f}")

        # 4. 턴 타입별 가중치 적용
        turn_type = self._detect_turn_type(turn)
        weights = self.rubric_loader.get_turn_evaluation_criteria(turn_type)

        # 5. 종합 점수 계산
        overall_score = self._calculate_overall_score(
            rule_scores=rule_scores,
            gate_result=gate_result,
            slot_validation=slot_validation,
            weights=weights
        )

        if verbose:
            print(f"  종합 점수: {overall_score:.3f}")

        turn_result = {
            'turn_id': turn['turn_id'],
            'utterance': turn['utterance'],
            'answer': answer,
            'required_slots': turn['required_slots'],
            'expected_slot_updates': turn.get('expected_slot_updates', {}),
            'accumulated_slots': accumulated_slots.copy(),
            'rule_scores': rule_scores,
            'gate_result': gate_result,
            'slot_validation': slot_validation,
            'turn_type': turn_type,
            'weights': weights,
            'overall_score': overall_score,
            'qc_metadata': {
                'qc_pass': state.get('qc_pass', False),
                'qc_reasons': state.get('qc_reasons', []),
                'qc_ts_count': state.get('qc_ts_count', 0),
                'qc_ts_chars': state.get('qc_ts_chars', 0)
            }
        }

        return turn_result

    def _validate_slots(
        self,
        turn: Dict[str, Any],
        accumulated_slots: Dict[str, Any]
    ) -> Dict[str, Any]:
        """슬롯 검증"""
        errors = []
        all_valid = True

        for slot_name in turn['required_slots']:
            value = self._get_nested_value(accumulated_slots, slot_name)

            if value is None:
                errors.append(f"{slot_name}: missing")
                all_valid = False
                continue

            # YAML 규칙으로 검증
            validation = self.rubric_loader.validate_slot_value(slot_name, value)

            if not validation['valid']:
                errors.append(f"{slot_name}: {validation['error_msg']}")
                all_valid = False

        return {
            'all_valid': all_valid,
            'errors': errors
        }

    def _detect_turn_type(self, turn: Dict[str, Any]) -> str:
        """턴 타입 감지"""
        if turn.get('trap'):
            return 'trap_turn'

        if 'symptoms.red_flags' in turn.get('required_slots', []):
            return 'redflag_turn'

        return 'normal_turn'

    def _calculate_overall_score(
        self,
        rule_scores: Dict[str, float],
        gate_result: Dict[str, Any],
        slot_validation: Dict[str, Any],
        weights: Dict[str, float]
    ) -> float:
        """종합 점수 계산"""
        # 게이트 실패시 자동 0점
        if not gate_result['pass']:
            return 0.0

        # 슬롯 검증 실패시 감점
        slot_penalty = 0.0 if slot_validation['all_valid'] else 0.2

        # 규칙 점수들을 평균 (CUS, TS_Use, Safety)
        rule_avg = sum(rule_scores.values()) / len(rule_scores) if rule_scores else 0.5

        # Factuality cap 적용
        if gate_result.get('capped_factuality') is not None:
            rule_avg = min(rule_avg, gate_result['capped_factuality'])

        # 슬롯 감점 적용
        final_score = max(0.0, rule_avg - slot_penalty)

        return final_score

    def _aggregate_case_scores(self, turn_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """케이스 전체 집계 점수"""
        if not turn_results:
            return {'mean': 0.0, 'weighted_mean': 0.0, 'min': 0.0, 'max': 0.0}

        scores = [tr['overall_score'] for tr in turn_results]

        # 난이도별 가중치 (나중 턴일수록 중요)
        weights = [1.0 + (i * 0.2) for i in range(len(scores))]
        total_weight = sum(weights)

        weighted_mean = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return {
            'mean': sum(scores) / len(scores),
            'weighted_mean': weighted_mean,
            'min': min(scores),
            'max': max(scores),
            'pass_rate': sum(1 for s in scores if s >= 0.75) / len(scores)
        }

    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """중첩된 키에서 값 추출"""
        keys = key_path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value

    def export_results(self, results: List[Dict[str, Any]], filename: str = "eval_results.jsonl"):
        """평가 결과 저장"""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\n평가 결과 저장: {output_path}")

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """요약 리포트 생성"""
        if not results:
            return "평가 결과 없음"

        total_cases = len(results)
        total_turns = sum(len(r['turns']) for r in results)

        pass_cases = sum(1 for r in results if r['metadata']['pass'])
        pass_rate = pass_cases / total_cases if total_cases > 0 else 0

        avg_score = sum(r['aggregate_scores']['weighted_mean'] for r in results) / total_cases

        difficulty_breakdown = {}
        for result in results:
            diff = result.get('difficulty', 'medium')
            if diff not in difficulty_breakdown:
                difficulty_breakdown[diff] = {'count': 0, 'pass': 0, 'scores': []}

            difficulty_breakdown[diff]['count'] += 1
            if result['metadata']['pass']:
                difficulty_breakdown[diff]['pass'] += 1
            difficulty_breakdown[diff]['scores'].append(result['aggregate_scores']['weighted_mean'])

        report = f"""
{'='*60}
평가 요약 리포트
{'='*60}

전체 통계:
  - 평가 케이스 수: {total_cases}
  - 총 턴 수: {total_turns}
  - 통과율: {pass_rate:.1%} ({pass_cases}/{total_cases})
  - 평균 점수: {avg_score:.3f}

난이도별 통계:
"""

        for diff, stats in difficulty_breakdown.items():
            avg = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            pass_r = stats['pass'] / stats['count'] if stats['count'] > 0 else 0
            report += f"  [{diff.upper()}]\n"
            report += f"    케이스 수: {stats['count']}\n"
            report += f"    통과율: {pass_r:.1%} ({stats['pass']}/{stats['count']})\n"
            report += f"    평균 점수: {avg:.3f}\n\n"

        report += f"{'='*60}\n"

        return report


def dummy_agent_runner(question: str, context: Dict[str, Any], case_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    더미 에이전트 (테스트용)
    실제 사용시 agent.graph를 통한 실행으로 교체
    """
    return {
        'answer': f"질문 '{question}'에 대한 답변입니다. (더미)",
        'state': {
            'ts_context': [{'text': '더미 TS 근거 문서', 'source': 'ts_coarse'}],
            'qc_pass': True,
            'qc_ts_count': 2,
            'qc_ts_chars': 300,
            'qc_reasons': []
        }
    }


def main():
    """테스트 실행"""
    print("YAML 기반 평가기 테스트\n")

    evaluator = YAMLBasedEvaluator()

    # 모든 케이스 가져오기
    all_cases = evaluator.template_loader.get_all_cases()
    print(f"총 {len(all_cases)}개 케이스 로드됨\n")

    # 각 케이스 평가
    results = []

    for case in all_cases[:2]:  # 처음 2개만 테스트
        case_id = case['case_id']

        try:
            result = evaluator.evaluate_case(
                case_id=case_id,
                agent_runner=dummy_agent_runner,
                verbose=True
            )
            results.append(result)

        except Exception as e:
            print(f"케이스 {case_id} 평가 실패: {e}")

    # 결과 저장
    evaluator.export_results(results, filename="test_eval_results.jsonl")

    # 요약 리포트
    summary = evaluator.generate_summary_report(results)
    print(summary)


if __name__ == "__main__":
    main()
