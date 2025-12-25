"""
Query Rewriter: Dynamic query rewriting based on quality feedback

Rewrites queries to improve retrieval quality when initial results are insufficient.
"""

from typing import Dict, Any, Optional
import json


class QueryRewriter:
    """
    LLM-based query rewriter.

    Rewrites queries based on quality feedback to improve retrieval.
    """

    def __init__(self, llm_client: Any):
        """
        Initialize query rewriter.

        Args:
            llm_client: LLM client for rewriting
        """
        self.llm_client = llm_client

    def rewrite(
        self,
        original_query: str,
        quality_feedback: Dict[str, Any],
        previous_answer: str = "",
        profile_summary: str = "",
        slot_out: Optional[Dict] = None,
        iteration_count: int = 0
    ) -> str:
        """
        Rewrite query based on quality feedback.

        Args:
            original_query: Original user query
            quality_feedback: Quality evaluation feedback
            previous_answer: Previously generated answer
            profile_summary: User profile summary
            slot_out: Extracted slots
            iteration_count: Current iteration number

        Returns:
            Rewritten query string
        """
        # Build rewriting prompt
        prompt = self._build_rewriting_prompt(
            original_query,
            quality_feedback,
            previous_answer,
            profile_summary,
            slot_out,
            iteration_count
        )

        try:
            # Call LLM for rewriting
            response = self.llm_client.generate(prompt)

            # Parse rewritten query
            rewritten_query = self._parse_rewriting_response(response, original_query)

        except Exception as e:
            print(f"[ERROR] Query rewriting failed: {e}")
            # Fallback: use original query with expanded terms
            rewritten_query = self._fallback_rewrite(original_query, quality_feedback)

        return rewritten_query

    def _build_rewriting_prompt(
        self,
        original_query: str,
        quality_feedback: Dict,
        previous_answer: str,
        profile_summary: str,
        slot_out: Optional[Dict],
        iteration_count: int
    ) -> str:
        """Build query rewriting prompt."""
        missing_info = quality_feedback.get('missing_info', [])
        suggestions = quality_feedback.get('improvement_suggestions', [])

        prompt = f"""의료 정보 검색을 위한 질의를 개선해주세요.

**원래 질문:**
{original_query}

**이전 답변의 문제점:**
- 품질 점수: {quality_feedback.get('overall_score', 0):.2f}
"""

        if missing_info:
            prompt += f"- 누락된 정보: {', '.join(missing_info)}\n"

        if suggestions:
            prompt += f"- 개선 제안: {', '.join(suggestions)}\n"

        if profile_summary:
            prompt += f"""
**환자 정보:**
{profile_summary}
"""

        if slot_out:
            prompt += f"""
**추출된 정보:**
{json.dumps(slot_out, ensure_ascii=False, indent=2)}
"""

        prompt += f"""
**현재 반복 횟수:** {iteration_count + 1}회차

**요구사항:**
1. 누락된 정보를 명시적으로 요청하도록 질의 확장
2. 의료 전문 용어를 포함하여 검색 정확도 향상
3. 환자 프로필을 반영하여 개인화된 검색
4. 원래 질문의 의도를 유지하면서 개선

**응답 형식:**
개선된 질의만 출력하세요 (JSON이나 다른 형식 없이 질의 텍스트만).

개선된 질의:
"""

        return prompt

    def _parse_rewriting_response(self, response: str, original_query: str) -> str:
        """Parse LLM rewriting response."""
        # Clean up response
        rewritten = response.strip()

        # Remove common prefixes
        prefixes = [
            "개선된 질의:",
            "질의:",
            "Rewritten query:",
            "Query:",
        ]

        for prefix in prefixes:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        # Validate rewritten query
        if not rewritten or len(rewritten) < 5:
            print("[WARNING] Invalid rewritten query, using original")
            return original_query

        # Ensure it's not too different from original
        if len(rewritten) > len(original_query) * 3:
            print("[WARNING] Rewritten query too long, truncating")
            rewritten = rewritten[:len(original_query) * 2]

        return rewritten

    def _fallback_rewrite(
        self,
        original_query: str,
        quality_feedback: Dict
    ) -> str:
        """Simple fallback query rewriting."""
        missing_info = quality_feedback.get('missing_info', [])

        if not missing_info:
            # Just return original
            return original_query

        # Append missing info as additional terms
        additional_terms = ' '.join(missing_info)
        rewritten = f"{original_query} {additional_terms}"

        return rewritten.strip()
