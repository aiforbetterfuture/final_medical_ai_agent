"""
Quality Evaluator: LLM-based quality assessment

Evaluates answer quality based on grounding, completeness, and accuracy.
"""

from typing import Dict, Any, List, Optional
import json


class QualityEvaluator:
    """
    LLM-based quality evaluator for generated answers.

    Uses LLM to assess:
    - Grounding: Is the answer supported by retrieved documents?
    - Completeness: Does it address all aspects of the query?
    - Accuracy: Is the medical information correct?
    """

    def __init__(self, llm_client: Any):
        """
        Initialize quality evaluator.

        Args:
            llm_client: LLM client for evaluation
        """
        self.llm_client = llm_client

    def evaluate(
        self,
        user_query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        profile_summary: str = "",
        previous_feedback: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate answer quality.

        Args:
            user_query: Original user query
            answer: Generated answer
            retrieved_docs: Retrieved documents used for answer
            profile_summary: User profile summary
            previous_feedback: Previous evaluation feedback (if iterating)

        Returns:
            Dictionary with evaluation scores and feedback
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            user_query, answer, retrieved_docs, profile_summary, previous_feedback
        )

        try:
            # Call LLM for evaluation
            response = self.llm_client.generate(prompt)

            # Parse evaluation response
            quality_feedback = self._parse_evaluation_response(response)

        except Exception as e:
            print(f"[ERROR] Quality evaluation failed: {e}")
            # Fallback to simple heuristic
            quality_feedback = self._fallback_evaluation(
                answer, retrieved_docs, profile_summary
            )

        return quality_feedback

    def _build_evaluation_prompt(
        self,
        user_query: str,
        answer: str,
        retrieved_docs: List[Dict],
        profile_summary: str,
        previous_feedback: Optional[Dict]
    ) -> str:
        """Build evaluation prompt for LLM."""
        # Format retrieved documents
        doc_texts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5 docs
            text = doc.get('text', '')[:300]  # First 300 chars
            doc_texts.append(f"[Doc {i}] {text}")

        docs_str = "\n".join(doc_texts) if doc_texts else "(No documents retrieved)"

        # Build prompt
        prompt = f"""의료 AI 에이전트가 생성한 답변의 품질을 평가해주세요.

**사용자 질문:**
{user_query}

**생성된 답변:**
{answer}

**검색된 문서:**
{docs_str}
"""

        if profile_summary:
            prompt += f"""
**환자 프로필:**
{profile_summary}
"""

        if previous_feedback:
            prompt += f"""
**이전 피드백:**
{previous_feedback.get('improvement_suggestions', [])}
"""

        prompt += """
다음 기준으로 평가하고 JSON 형식으로 응답해주세요:

1. **grounding_score** (0.0-1.0): 답변이 검색된 문서에 근거하는가?
2. **completeness_score** (0.0-1.0): 질문의 모든 측면을 다루는가?
3. **accuracy_score** (0.0-1.0): 의료 정보가 정확한가?
4. **overall_score** (0.0-1.0): 전체 품질 점수
5. **missing_info** (list): 누락된 정보 목록
6. **improvement_suggestions** (list): 개선 제안 목록
7. **needs_retrieval** (boolean): 추가 검색이 필요한가?
8. **reason** (string): 평가 근거

응답 형식:
```json
{
  "grounding_score": 0.8,
  "completeness_score": 0.7,
  "accuracy_score": 0.9,
  "overall_score": 0.8,
  "missing_info": ["부작용 정보"],
  "improvement_suggestions": ["복용 시기 추가"],
  "needs_retrieval": false,
  "reason": "답변이 잘 근거되어 있으나 부작용 정보 보완 필요"
}
```
"""

        return prompt

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response."""
        # Try to extract JSON from response
        try:
            # Find JSON block
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                evaluation = json.loads(json_str)

                # Validate required fields
                required_fields = [
                    'overall_score', 'grounding_score',
                    'completeness_score', 'accuracy_score'
                ]

                for field in required_fields:
                    if field not in evaluation:
                        evaluation[field] = 0.5  # Default

                # Ensure boolean and list fields
                evaluation.setdefault('needs_retrieval', False)
                evaluation.setdefault('missing_info', [])
                evaluation.setdefault('improvement_suggestions', [])
                evaluation.setdefault('reason', 'LLM evaluation')

                return evaluation

        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse evaluation JSON: {e}")

        # Fallback
        return {
            'overall_score': 0.5,
            'grounding_score': 0.5,
            'completeness_score': 0.5,
            'accuracy_score': 0.5,
            'missing_info': [],
            'improvement_suggestions': [],
            'needs_retrieval': False,
            'reason': 'Parse error, using fallback'
        }

    def _fallback_evaluation(
        self,
        answer: str,
        retrieved_docs: List[Dict],
        profile_summary: str
    ) -> Dict[str, Any]:
        """Simple heuristic evaluation fallback."""
        # Length-based completeness
        length_score = min(len(answer) / 500, 1.0)

        # Evidence-based grounding
        evidence_score = min(len(retrieved_docs) / 3, 1.0)

        # Personalization
        personalization_score = 1.0 if profile_summary else 0.5

        overall = (
            length_score * 0.3 +
            evidence_score * 0.4 +
            personalization_score * 0.3
        )

        return {
            'overall_score': overall,
            'grounding_score': evidence_score,
            'completeness_score': length_score,
            'accuracy_score': 0.7,  # Assume decent accuracy
            'missing_info': [],
            'improvement_suggestions': [],
            'needs_retrieval': overall < 0.5,
            'reason': 'Heuristic evaluation (fallback)'
        }
