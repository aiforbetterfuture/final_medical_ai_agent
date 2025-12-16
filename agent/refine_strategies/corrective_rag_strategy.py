"""
Corrective RAG Strategy (CRAG)

현재 구현의 CRAG 로직을 Strategy 패턴으로 캡슐화
- LLM 기반 품질 평가
- 동적 질의 재작성
- 조건부 재검색
"""

import hashlib
from typing import Dict, Any
from agent.state import AgentState
from agent.refine_strategies.base_strategy import BaseRefineStrategy
from agent.quality_evaluator import QualityEvaluator
from agent.query_rewriter import QueryRewriter
from core.llm_client import get_llm_client
from core.config import get_llm_config


class CorrectiveRAGStrategy(BaseRefineStrategy):
    """
    Corrective RAG 전략 (기본값)

    특징:
    - LLM 기반 품질 평가 (Grounding + Self-Critique)
    - 동적 질의 재작성 (피드백 반영)
    - 조건부 재검색 (품질 임계값 기반)
    - 이력 추적 (Ablation 연구용)
    """

    def get_strategy_name(self) -> str:
        return 'corrective_rag'

    def refine(self, state: AgentState) -> Dict[str, Any]:
        """
        CRAG 품질 평가 및 재검색 판단
        """
        print(f"[{self.get_strategy_name().upper()}] Refine 수행 중...")

        answer = state.get('answer', '')
        retrieved_docs = state.get('retrieved_docs', [])
        profile_summary = state.get('profile_summary', '')
        iteration_count = state.get('iteration_count', 0)

        # LLM 기반 또는 휴리스틱 품질 평가
        llm_based_quality_check = self.feature_flags.get('llm_based_quality_check', True)

        if llm_based_quality_check:
            quality_feedback = self._llm_based_evaluation(
                state=state,
                answer=answer,
                retrieved_docs=retrieved_docs,
                profile_summary=profile_summary
            )
            quality_score = quality_feedback.get('overall_score', 0.5)
            needs_retrieval_by_quality = quality_feedback.get('needs_retrieval', False)
        else:
            quality_feedback = self._heuristic_evaluation(
                answer=answer,
                retrieved_docs=retrieved_docs,
                profile_summary=profile_summary
            )
            quality_score = quality_feedback.get('overall_score', 0.5)
            threshold = self.feature_flags.get('quality_threshold', 0.5)
            needs_retrieval_by_quality = quality_score < threshold

        print(f"[{self.get_strategy_name().upper()}] 품질 점수: {quality_score:.2f} (Iteration: {iteration_count + 1})")

        # 재검색 필요 여부 결정
        max_iter = self.feature_flags.get('max_refine_iterations', 2)
        threshold = self.feature_flags.get('quality_threshold', 0.5)

        needs_retrieval = (
            needs_retrieval_by_quality and
            quality_score < threshold and
            iteration_count < max_iter
        )

        # 이력 추적
        quality_score_history = state.get('quality_score_history') or []
        quality_score_history.append(quality_score)

        # 동적 질의 재작성
        new_query = state.get('user_text', '')
        query_rewrite_history = state.get('query_rewrite_history') or []

        if needs_retrieval and self.feature_flags.get('dynamic_query_rewrite', True):
            new_query = self._rewrite_query(
                state=state,
                quality_feedback=quality_feedback,
                answer=answer
            )
            query_rewrite_history.append(new_query)
            print(f"[{self.get_strategy_name().upper()}] 질의 재작성: {new_query[:100]}...")
        else:
            query_rewrite_history.append(new_query)

        # Iteration 로그
        refine_iteration_logs = state.get('refine_iteration_logs') or []
        refine_iteration_logs.append({
            'iteration': iteration_count + 1,
            'strategy': self.get_strategy_name(),
            'quality_score': quality_score,
            'quality_feedback': quality_feedback,
            'needs_retrieval': needs_retrieval,
            'rewritten_query': new_query,
            'num_docs': len(retrieved_docs)
        })

        return {
            'quality_score': quality_score,
            'quality_feedback': quality_feedback,
            'needs_retrieval': needs_retrieval,
            'query_for_retrieval': new_query,
            'quality_score_history': quality_score_history,
            'query_rewrite_history': query_rewrite_history,
            'refine_iteration_logs': refine_iteration_logs,
            'refine_strategy': self.get_strategy_name(),
        }

    def should_retrieve(self, state: AgentState) -> bool:
        """
        CRAG 재검색 판단 (quality_check_node에서 사용)
        """
        needs_retrieval = state.get('needs_retrieval', False)
        iteration_count = state.get('iteration_count', 0)
        max_iter = self.feature_flags.get('max_refine_iterations', 2)

        # 기본 조건
        if not needs_retrieval or iteration_count >= max_iter:
            return False

        # 안전장치 1: 동일 문서 재검색 방지
        if self.feature_flags.get('duplicate_detection', True):
            if self._check_duplicate_docs(state):
                print(f"[{self.get_strategy_name().upper()}] 동일 문서 재검색 감지: 조기 종료")
                return False

        # 안전장치 2: 품질 점수 진행도 모니터링
        if self.feature_flags.get('progress_monitoring', True):
            if self._check_progress_stagnation(state):
                print(f"[{self.get_strategy_name().upper()}] 품질 개선 없음: 조기 종료")
                return False

        return True

    def _llm_based_evaluation(self, state, answer, retrieved_docs, profile_summary) -> dict:
        """LLM 기반 품질 평가"""
        # LLM 클라이언트 초기화
        if 'llm_client' not in state:
            llm_config = get_llm_config()
            llm_client = get_llm_client(
                provider=llm_config.get('provider', 'openai'),
                model=llm_config.get('model', 'gpt-4o-mini'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 1000)
            )
            state['llm_client'] = llm_client
        else:
            llm_client = state['llm_client']

        # Quality Evaluator 초기화
        if 'quality_evaluator' not in state:
            evaluator = QualityEvaluator(llm_client=llm_client)
            state['quality_evaluator'] = evaluator
        else:
            evaluator = state['quality_evaluator']

        # 평가 실행
        try:
            quality_feedback = evaluator.evaluate(
                user_query=state.get('user_text', ''),
                answer=answer,
                retrieved_docs=retrieved_docs,
                profile_summary=profile_summary,
                previous_feedback=state.get('quality_feedback')
            )
        except Exception as e:
            print(f"[ERROR] LLM 평가 실패, 휴리스틱으로 폴백: {e}")
            quality_feedback = self._heuristic_evaluation(answer, retrieved_docs, profile_summary)

        return quality_feedback

    def _heuristic_evaluation(self, answer, retrieved_docs, profile_summary) -> dict:
        """휴리스틱 평가 (폴백)"""
        length_score = min(len(answer) / 500, 1.0)
        evidence_score = 1.0 if len(retrieved_docs) > 0 else 0.0
        personalization_score = 1.0 if profile_summary else 0.0

        overall_score = (
            length_score * 0.3 +
            evidence_score * 0.4 +
            personalization_score * 0.3
        )

        return {
            'overall_score': overall_score,
            'grounding_score': evidence_score,
            'completeness_score': length_score,
            'accuracy_score': 0.7,
            'missing_info': [],
            'improvement_suggestions': [],
            'needs_retrieval': overall_score < 0.5,
            'reason': '휴리스틱 평가'
        }

    def _rewrite_query(self, state, quality_feedback, answer) -> str:
        """동적 질의 재작성"""
        llm_client = state.get('llm_client')
        if not llm_client:
            llm_config = get_llm_config()
            llm_client = get_llm_client(
                provider=llm_config.get('provider', 'openai'),
                model=llm_config.get('model', 'gpt-4o-mini'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 1000)
            )
            state['llm_client'] = llm_client

        if 'query_rewriter' not in state:
            rewriter = QueryRewriter(llm_client=llm_client)
            state['query_rewriter'] = rewriter
        else:
            rewriter = state['query_rewriter']

        try:
            rewritten_query = rewriter.rewrite(
                original_query=state.get('user_text', ''),
                quality_feedback=quality_feedback,
                previous_answer=answer,
                profile_summary=state.get('profile_summary', ''),
                slot_out=state.get('slot_out', {}),
                iteration_count=state.get('iteration_count', 0)
            )
        except Exception as e:
            print(f"[ERROR] 질의 재작성 실패: {e}")
            rewritten_query = state.get('user_text', '')

        return rewritten_query

    def _check_duplicate_docs(self, state) -> bool:
        """동일 문서 재검색 방지"""
        retrieved_docs = state.get('retrieved_docs', [])
        retrieved_docs_history = state.get('retrieved_docs_history') or []

        if not retrieved_docs or len(retrieved_docs_history) < 2:
            return False

        current_hashes = self._compute_doc_hashes(retrieved_docs)
        previous_hashes = retrieved_docs_history[-2]

        current_set = set(current_hashes)
        previous_set = set(previous_hashes)

        if len(current_set | previous_set) == 0:
            return False

        similarity = len(current_set & previous_set) / len(current_set | previous_set)
        return similarity >= 0.8

    def _check_progress_stagnation(self, state) -> bool:
        """품질 점수 진행도 모니터링"""
        quality_score_history = state.get('quality_score_history') or []

        if len(quality_score_history) < 2:
            return False

        improvement = quality_score_history[-1] - quality_score_history[-2]
        return improvement < 0.05  # 최소 0.05 개선 필요

    def _compute_doc_hashes(self, retrieved_docs) -> list:
        """문서 해시 계산"""
        return [
            hashlib.md5(doc.get('text', '').encode('utf-8')).hexdigest()
            for doc in retrieved_docs if doc.get('text')
        ]

    def get_metrics(self, state: AgentState) -> Dict[str, Any]:
        """CRAG 특화 메트릭"""
        base_metrics = super().get_metrics(state)

        crag_metrics = {
            'llm_evaluations': len([
                log for log in state.get('refine_iteration_logs', [])
                if log.get('quality_feedback', {}).get('reason') != '휴리스틱 평가'
            ]),
            'query_rewrites': len(state.get('query_rewrite_history', [])),
            'quality_improvements': self._calculate_quality_improvements(state),
            'duplicate_detections': 0,  # TODO: 추적 필요 시 추가
            'early_terminations': 0,  # TODO: 추적 필요 시 추가
        }

        return {**base_metrics, **crag_metrics}

    def _calculate_quality_improvements(self, state) -> float:
        """품질 개선폭 계산"""
        history = state.get('quality_score_history', [])
        if len(history) < 2:
            return 0.0
        return history[-1] - history[0]
