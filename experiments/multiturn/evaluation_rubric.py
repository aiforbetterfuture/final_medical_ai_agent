"""
평가 루브릭 (Evaluation Rubric)
서브스코어 기반 객관적 평가 + LLM-as-a-Judge
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from enum import Enum


class EvaluationMetric(Enum):
    """평가 항목"""
    ACCURACY = "accuracy"
    CONTEXT = "context"
    REASONING = "reasoning"
    HALLUCINATION = "hallucination"
    EVIDENCE = "evidence"  # Agentic RAG 전용


@dataclass
class SubScore:
    """서브스코어"""
    metric: str
    score: float  # 0~2
    max_score: float = 2.0
    reason: str = ""


@dataclass
class EvaluationResult:
    """평가 결과"""
    turn_idx: int
    turn_type: str
    subscores: List[SubScore]
    total_score: float
    weighted_score: float
    judge_reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['subscores'] = [asdict(s) for s in self.subscores]
        return data


class EvaluationRubric:
    """평가 루브릭"""
    
    # 가중치 (Basic RAG)
    WEIGHTS_BASIC = {
        EvaluationMetric.ACCURACY: 0.35,
        EvaluationMetric.CONTEXT: 0.25,
        EvaluationMetric.REASONING: 0.25,
        EvaluationMetric.HALLUCINATION: 0.15
    }
    
    # 가중치 (Agentic RAG)
    WEIGHTS_AGENTIC = {
        EvaluationMetric.ACCURACY: 0.30,
        EvaluationMetric.CONTEXT: 0.20,
        EvaluationMetric.REASONING: 0.20,
        EvaluationMetric.HALLUCINATION: 0.10,
        EvaluationMetric.EVIDENCE: 0.20
    }
    
    def __init__(self, model_type: str = "basic"):
        """
        Args:
            model_type: "basic" 또는 "agentic"
        """
        self.model_type = model_type
        self.weights = (
            self.WEIGHTS_AGENTIC if model_type == "agentic" 
            else self.WEIGHTS_BASIC
        )
    
    def evaluate_turn(
        self,
        turn_data: Dict[str, Any],
        model_answer: str,
        gold_answer: Optional[str] = None,
        retrieved_context: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        단일 턴 평가
        
        Args:
            turn_data: 턴 정보 (질문, 환자 상태 등)
            model_answer: 모델 답변
            gold_answer: 정답 (있는 경우)
            retrieved_context: 검색된 문서 (Agentic RAG)
        """
        subscores = []
        
        # 1. Accuracy (정확성)
        accuracy_score = self._evaluate_accuracy(
            turn_data, model_answer, gold_answer
        )
        subscores.append(accuracy_score)
        
        # 2. Context Awareness (문맥 파악)
        context_score = self._evaluate_context(
            turn_data, model_answer
        )
        subscores.append(context_score)
        
        # 3. Reasoning Logic (추론 논리)
        reasoning_score = self._evaluate_reasoning(
            turn_data, model_answer, gold_answer
        )
        subscores.append(reasoning_score)
        
        # 4. Hallucination (환각 억제)
        hallucination_score = self._evaluate_hallucination(
            turn_data, model_answer
        )
        subscores.append(hallucination_score)
        
        # 5. Evidence Grounding (근거 인용, Agentic만)
        if self.model_type == "agentic":
            evidence_score = self._evaluate_evidence(
                turn_data, model_answer, retrieved_context
            )
            subscores.append(evidence_score)
        
        # 총점 계산
        total_score = sum(s.score for s in subscores)
        
        # 가중 점수 계산
        weighted_score = sum(
            s.score * self.weights[EvaluationMetric(s.metric)]
            for s in subscores
        )
        
        return EvaluationResult(
            turn_idx=turn_data.get('turn_idx', 0),
            turn_type=turn_data.get('turn_type', ''),
            subscores=subscores,
            total_score=total_score,
            weighted_score=weighted_score,
            judge_reason="Rule-based evaluation"
        )
    
    def _evaluate_accuracy(
        self, 
        turn_data: Dict[str, Any], 
        model_answer: str,
        gold_answer: Optional[str]
    ) -> SubScore:
        """
        정확성 평가 (0~2점)
        - 2점: 모든 사실 정확
        - 1점: 사소한 오류 1개
        - 0점: 주요 오류 2개 이상
        """
        # 실제 구현에서는 엔티티 추출 + GT 비교
        # 여기서는 간단히 처리
        
        if not gold_answer:
            # Gold answer 없으면 중립 점수
            return SubScore(
                metric=EvaluationMetric.ACCURACY.value,
                score=1.0,
                reason="No gold answer available"
            )
        
        # 간단한 토큰 기반 비교 (실제로는 더 정교한 로직 필요)
        gold_tokens = set(gold_answer.lower().split())
        answer_tokens = set(model_answer.lower().split())
        
        overlap = len(gold_tokens & answer_tokens)
        total = len(gold_tokens)
        
        if total == 0:
            score = 1.0
        else:
            ratio = overlap / total
            if ratio >= 0.8:
                score = 2.0
            elif ratio >= 0.5:
                score = 1.0
            else:
                score = 0.0
        
        return SubScore(
            metric=EvaluationMetric.ACCURACY.value,
            score=score,
            reason=f"Token overlap: {overlap}/{total} ({ratio:.2%})"
        )
    
    def _evaluate_context(
        self, 
        turn_data: Dict[str, Any], 
        model_answer: str
    ) -> SubScore:
        """
        문맥 파악 평가 (0~2점)
        - 2점: 지시대명사 해소 100% 정확
        - 1점: 부분적으로 맞춤
        - 0점: 지시 대상 오인식
        """
        turn_type = turn_data.get('turn_type', '')
        
        # T3 (문맥 의존) 턴에서만 중요
        if turn_type != "T3":
            return SubScore(
                metric=EvaluationMetric.CONTEXT.value,
                score=2.0,
                reason="Not applicable for this turn type"
            )
        
        # 지시대명사 존재 여부 확인
        anaphora_keywords = ["그", "아까", "방금", "이전", "최근"]
        has_anaphora = any(kw in turn_data.get('question', '') for kw in anaphora_keywords)
        
        if not has_anaphora:
            return SubScore(
                metric=EvaluationMetric.CONTEXT.value,
                score=2.0,
                reason="No anaphora in question"
            )
        
        # 실제로는 엔티티 링킹 + 이전 턴 참조 확인
        # 여기서는 간단히 키워드 존재 여부로 판단
        required_slots = turn_data.get('required_slots', [])
        slot_mentions = sum(
            1 for slot in required_slots 
            if self._check_slot_mention(slot, model_answer, turn_data)
        )
        
        if len(required_slots) == 0:
            score = 2.0
        else:
            ratio = slot_mentions / len(required_slots)
            if ratio >= 0.8:
                score = 2.0
            elif ratio >= 0.5:
                score = 1.0
            else:
                score = 0.0
        
        return SubScore(
            metric=EvaluationMetric.CONTEXT.value,
            score=score,
            reason=f"Slot mentions: {slot_mentions}/{len(required_slots)}"
        )
    
    def _evaluate_reasoning(
        self, 
        turn_data: Dict[str, Any], 
        model_answer: str,
        gold_answer: Optional[str]
    ) -> SubScore:
        """
        추론 논리 평가 (0~2점)
        - 2점: 계산 정확 + 논리 타당
        - 1점: 계산 맞지만 설명 부족
        - 0점: 계산 오류 또는 논리 비약
        """
        turn_type = turn_data.get('turn_type', '')
        
        # T2 (비교), T4 (복합 추론)에서 중요
        if turn_type not in ["T2", "T4"]:
            return SubScore(
                metric=EvaluationMetric.REASONING.value,
                score=2.0,
                reason="Not applicable for this turn type"
            )
        
        # T2: 변화량 계산 확인
        if turn_type == "T2" and gold_answer:
            # 변화량 추출 (간단한 정규식)
            import re
            gold_numbers = re.findall(r'\d+\.?\d*', gold_answer)
            answer_numbers = re.findall(r'\d+\.?\d*', model_answer)
            
            if gold_numbers and answer_numbers:
                # 주요 수치가 포함되어 있는지 확인
                key_numbers_found = sum(
                    1 for gn in gold_numbers 
                    if any(abs(float(gn) - float(an)) < 0.1 for an in answer_numbers)
                )
                ratio = key_numbers_found / len(gold_numbers)
                
                if ratio >= 0.8:
                    score = 2.0
                elif ratio >= 0.5:
                    score = 1.0
                else:
                    score = 0.0
                
                return SubScore(
                    metric=EvaluationMetric.REASONING.value,
                    score=score,
                    reason=f"Key numbers found: {key_numbers_found}/{len(gold_numbers)}"
                )
        
        # T4: 논리 구조 확인 (키워드 기반)
        if turn_type == "T4":
            reasoning_keywords = ["기준", "권고", "고려", "판단", "따라서", "그러므로"]
            keyword_count = sum(1 for kw in reasoning_keywords if kw in model_answer)
            
            if keyword_count >= 3:
                score = 2.0
            elif keyword_count >= 1:
                score = 1.0
            else:
                score = 0.0
            
            return SubScore(
                metric=EvaluationMetric.REASONING.value,
                score=score,
                reason=f"Reasoning keywords: {keyword_count}"
            )
        
        return SubScore(
            metric=EvaluationMetric.REASONING.value,
            score=1.0,
            reason="Default score"
        )
    
    def _evaluate_hallucination(
        self, 
        turn_data: Dict[str, Any], 
        model_answer: str
    ) -> SubScore:
        """
        환각 억제 평가 (0~2점)
        - 2점: 환각 없음
        - 1점: 사소한 추측 1개
        - 0점: 명백한 허위 정보
        """
        # 실제로는 GT와 비교 + 엔티티 검증
        # 여기서는 간단히 불확실성 표현 확인
        
        uncertainty_phrases = [
            "확실하지 않", "정확하지 않", "알 수 없", "추정", "가능성",
            "아마도", "~것으로 보임", "~것 같습니다"
        ]
        
        has_uncertainty = any(phrase in model_answer for phrase in uncertainty_phrases)
        
        # 환자 상태에 없는 정보 언급 확인 (간단한 휴리스틱)
        canonical_state = turn_data.get('canonical_state_snapshot', {})
        
        # 약물명 환각 체크
        known_meds = [m['name'] for m in canonical_state.get('medications', [])]
        # 실제로는 NER로 추출해야 하지만 여기서는 생략
        
        # 기본 점수
        score = 2.0
        reason = "No obvious hallucination detected"
        
        # 불확실성 표현이 있으면 좋은 신호
        if has_uncertainty:
            reason = "Appropriate uncertainty expression"
        
        return SubScore(
            metric=EvaluationMetric.HALLUCINATION.value,
            score=score,
            reason=reason
        )
    
    def _evaluate_evidence(
        self, 
        turn_data: Dict[str, Any], 
        model_answer: str,
        retrieved_context: Optional[List[str]]
    ) -> SubScore:
        """
        근거 인용 평가 (0~2점, Agentic RAG 전용)
        - 2점: 근거 문서 ID + 구체적 문장 인용
        - 1점: 근거 언급했으나 모호
        - 0점: 근거 없음 또는 잘못된 인용
        """
        turn_type = turn_data.get('turn_type', '')
        
        # T4 (복합 추론)에서 중요
        if turn_type != "T4":
            return SubScore(
                metric=EvaluationMetric.EVIDENCE.value,
                score=2.0,
                reason="Not applicable for this turn type"
            )
        
        # 근거 인용 키워드 확인
        citation_keywords = ["근거", "기준", "지침", "권고", "문서", "출처", "참고"]
        has_citation = any(kw in model_answer for kw in citation_keywords)
        
        if not has_citation:
            return SubScore(
                metric=EvaluationMetric.EVIDENCE.value,
                score=0.0,
                reason="No citation found"
            )
        
        # 검색 문서와의 일치도 확인
        if retrieved_context:
            # 간단한 토큰 오버랩 확인
            answer_tokens = set(model_answer.lower().split())
            context_tokens = set(' '.join(retrieved_context).lower().split())
            
            overlap = len(answer_tokens & context_tokens)
            if overlap >= 10:  # 충분한 오버랩
                score = 2.0
                reason = f"Strong evidence grounding (overlap: {overlap} tokens)"
            elif overlap >= 5:
                score = 1.0
                reason = f"Weak evidence grounding (overlap: {overlap} tokens)"
            else:
                score = 0.0
                reason = "Citation without actual grounding"
        else:
            # 검색 문서 없으면 키워드만으로 판단
            score = 1.0
            reason = "Citation mentioned but no context provided"
        
        return SubScore(
            metric=EvaluationMetric.EVIDENCE.value,
            score=score,
            reason=reason
        )
    
    def _check_slot_mention(
        self, 
        slot: str, 
        answer: str, 
        turn_data: Dict[str, Any]
    ) -> bool:
        """슬롯 언급 여부 확인"""
        canonical_state = turn_data.get('canonical_state_snapshot', {})
        
        if slot == "medications":
            meds = canonical_state.get('medications', [])
            return any(m['name'].lower() in answer.lower() for m in meds)
        
        elif slot == "lab_results":
            labs = canonical_state.get('lab_results', [])
            return any(str(lr['value']) in answer for lr in labs)
        
        elif slot == "diagnosis":
            diagnoses = canonical_state.get('diagnosis', [])
            return any(d.lower() in answer.lower() for d in diagnoses)
        
        return False


class LLMJudge:
    """LLM-as-a-Judge 평가기"""
    
    JUDGE_PROMPT_TEMPLATE = """당신은 의료 AI 시스템의 답변 품질을 평가하는 전문가입니다.
제공된 [환자 데이터(Ground Truth)], [질문], [모델의 답변]을 바탕으로 
아래 기준에 따라 각 항목을 0~2점으로 평가하세요.

**평가 기준**:
1. Accuracy (0~2): GT 값/날짜/수치 오류 개수
   - 2점: 모든 사실 정확
   - 1점: 사소한 오류 1개
   - 0점: 주요 오류 2개 이상

2. Context (0~2): 지시대명사('그 약', '아까 말한 수치') 해소 정확도
   - 2점: 지시 대상 100% 정확
   - 1점: 부분적으로 맞춤
   - 0점: 지시 대상 오인식

3. Reasoning (0~2): 변화량 계산/비교 논리의 타당성
   - 2점: 계산 정확 + 논리 타당
   - 1점: 계산 맞지만 설명 부족
   - 0점: 계산 오류 또는 논리 비약

4. Hallucination (0~2): GT/근거 없는 주장 수
   - 2점: 환각 없음
   - 1점: 사소한 추측 1개
   - 0점: 명백한 허위 정보

{evidence_section}

**입력 데이터**:
- 환자 데이터 (Ground Truth): {patient_data}
- 질문: {question}
- 모델 답변: {model_answer}
{gold_answer_section}
{context_section}

**출력 형식** (JSON만):
{{
  "accuracy": <0~2>,
  "context": <0~2>,
  "reasoning": <0~2>,
  "hallucination": <0~2>,
  {evidence_field}
  "reason": "<각 항목별 채점 근거 1~2문장>"
}}
"""
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM 클라이언트 (core.llm_client.LLMClient)
        """
        self.llm_client = llm_client
    
    def evaluate_turn(
        self,
        turn_data: Dict[str, Any],
        model_answer: str,
        gold_answer: Optional[str] = None,
        retrieved_context: Optional[List[str]] = None,
        model_type: str = "basic"
    ) -> EvaluationResult:
        """LLM을 사용한 턴 평가"""
        
        # 프롬프트 구성
        evidence_section = ""
        evidence_field = ""
        if model_type == "agentic":
            evidence_section = """
5. Evidence (0~2, Agentic만): 근거 문서 인용 정확도
   - 2점: 근거 문서 ID + 구체적 문장 인용
   - 1점: 근거 언급했으나 모호
   - 0점: 근거 없음 또는 잘못된 인용
"""
            evidence_field = '"evidence": <0~2>,'
        
        gold_answer_section = ""
        if gold_answer:
            gold_answer_section = f"- 정답 (참고용): {gold_answer}"
        
        context_section = ""
        if retrieved_context:
            context_section = f"- 검색된 문서: {retrieved_context[:3]}"  # 최대 3개
        
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            evidence_section=evidence_section,
            evidence_field=evidence_field,
            patient_data=json.dumps(turn_data.get('canonical_state_snapshot', {}), ensure_ascii=False),
            question=turn_data.get('question', ''),
            model_answer=model_answer,
            gold_answer_section=gold_answer_section,
            context_section=context_section
        )
        
        # LLM 호출 (실제 구현에서)
        # response = self.llm_client.generate(prompt, temperature=0.0)
        # judge_result = json.loads(response)
        
        # 여기서는 더미 결과 반환
        judge_result = {
            "accuracy": 2.0,
            "context": 2.0,
            "reasoning": 1.5,
            "hallucination": 2.0,
            "reason": "LLM judge evaluation (dummy)"
        }
        
        if model_type == "agentic":
            judge_result["evidence"] = 1.5
        
        # SubScore 객체로 변환
        subscores = [
            SubScore(metric=k, score=v, reason=judge_result['reason'])
            for k, v in judge_result.items()
            if k != 'reason'
        ]
        
        # 가중치
        weights = (
            EvaluationRubric.WEIGHTS_AGENTIC if model_type == "agentic"
            else EvaluationRubric.WEIGHTS_BASIC
        )
        
        total_score = sum(s.score for s in subscores)
        weighted_score = sum(
            s.score * weights[EvaluationMetric(s.metric)]
            for s in subscores
        )
        
        return EvaluationResult(
            turn_idx=turn_data.get('turn_idx', 0),
            turn_type=turn_data.get('turn_type', ''),
            subscores=subscores,
            total_score=total_score,
            weighted_score=weighted_score,
            judge_reason=judge_result['reason']
        )


def main():
    """평가 루브릭 테스트"""
    
    # Basic RAG 평가
    rubric_basic = EvaluationRubric(model_type="basic")
    
    # 샘플 턴 데이터
    turn_data = {
        "turn_idx": 1,
        "turn_type": "T1",
        "question": "진단명과 복용약을 정리해 주세요.",
        "canonical_state_snapshot": {
            "diagnosis": ["Type 2 Diabetes"],
            "medications": [{"name": "Metformin", "dosage": "500mg", "frequency": "BID"}],
            "allergy": "없음"
        },
        "required_slots": ["diagnosis", "medications"]
    }
    
    model_answer = "진단명: Type 2 Diabetes, 복용약: Metformin 500mg BID, 알레르기: 없음"
    gold_answer = "1. 진단명: Type 2 Diabetes\n2. 복용약: Metformin 500mg BID\n3. 알레르기: 없음"
    
    result = rubric_basic.evaluate_turn(turn_data, model_answer, gold_answer)
    
    print("=== 평가 결과 ===")
    print(f"Turn {result.turn_idx} ({result.turn_type})")
    print(f"총점: {result.total_score:.2f} / {len(result.subscores) * 2:.0f}")
    print(f"가중 점수: {result.weighted_score:.2f}\n")
    
    print("서브스코어:")
    for subscore in result.subscores:
        print(f"  - {subscore.metric}: {subscore.score:.1f}/2.0 ({subscore.reason})")


if __name__ == "__main__":
    main()

