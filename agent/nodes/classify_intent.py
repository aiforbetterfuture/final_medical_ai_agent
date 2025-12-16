"""
Active Retrieval: Intent Classification Node

이 노드는 쿼리의 의도를 분석하여:
1. 검색이 필요한지 판단 (needs_retrieval)
2. 필요한 문서 개수를 동적으로 결정 (dynamic_k)
3. 쿼리 복잡도를 추정 (query_complexity)

Ablation Study를 위한 메트릭을 수집합니다.
"""

import time
from typing import Dict, Any, Tuple
from agent.state import AgentState


class IntentClassifier:
    """
    의도 분류기 (Rule-based + Slot-based)

    안전성:
    - 모든 메서드에 try-except
    - Fallback to conservative defaults
    - 메트릭 수집으로 성능 추적
    """

    def __init__(self, feature_flags: Dict[str, Any]):
        self.feature_flags = feature_flags
        self.enabled = feature_flags.get('active_retrieval_enabled', False)

        # 설정값 (기본값 포함)
        self.default_k = feature_flags.get('default_k', 8)
        self.simple_k = feature_flags.get('simple_query_k', 3)
        self.moderate_k = feature_flags.get('moderate_query_k', 8)
        self.complex_k = feature_flags.get('complex_query_k', 15)

        # 메트릭 수집
        self.metrics = {
            'total_queries': 0,
            'skipped_retrieval': 0,
            'simple_queries': 0,
            'moderate_queries': 0,
            'complex_queries': 0,
            'classification_errors': 0,
            'avg_classification_time_ms': 0.0
        }

    def classify(
        self,
        query: str,
        slot_out: Dict[str, Any],
        conversation_history: str = None
    ) -> Tuple[bool, int, str]:
        """
        쿼리 분류

        Returns:
            (needs_retrieval, dynamic_k, complexity)
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1

        try:
            # 비활성화 시 기본값 반환
            if not self.enabled:
                return True, self.default_k, "default"

            # Stage 1: Rule-based filtering (빠른 패턴 매칭)
            is_greeting = self._is_greeting(query)
            if is_greeting:
                self.metrics['skipped_retrieval'] += 1
                return False, 0, "greeting"

            is_acknowledgment = self._is_acknowledgment(query)
            if is_acknowledgment:
                self.metrics['skipped_retrieval'] += 1
                return False, 0, "acknowledgment"

            # Stage 2: Slot-based analysis (의료 정보 기반)
            has_medical_info = self._has_medical_entities(slot_out)

            if has_medical_info:
                # 복잡도 추정
                complexity = self._estimate_complexity(slot_out, query)
                k = self._map_complexity_to_k(complexity)

                # 메트릭 업데이트
                self._update_complexity_metrics(complexity)

                return True, k, complexity

            # Stage 3: Query length and content analysis
            needs_retrieval, complexity = self._analyze_query_content(query, conversation_history)

            if needs_retrieval:
                k = self._map_complexity_to_k(complexity)
                self._update_complexity_metrics(complexity)
                return True, k, complexity
            else:
                self.metrics['skipped_retrieval'] += 1
                return False, 0, "simple_conversational"

        except Exception as e:
            # 에러 발생 시 안전한 기본값 반환
            print(f"[WARNING] Intent classification failed: {e}")
            self.metrics['classification_errors'] += 1
            return True, self.default_k, "error_fallback"

        finally:
            # 분류 시간 측정
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_avg_time(elapsed_ms)

    def _is_greeting(self, query: str) -> bool:
        """인사 감지"""
        greeting_patterns = [
            "안녕", "hello", "hi", "반가워", "처음", "만나서",
            "안녕하세요", "안녕하십니까", "good morning", "good afternoon"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in greeting_patterns) and len(query) < 30

    def _is_acknowledgment(self, query: str) -> bool:
        """단순 응답 감지 (네, 알겠습니다 등)"""
        acknowledgment_patterns = [
            "네", "예", "알겠습니다", "알겠어요", "감사합니다", "고맙습니다",
            "yes", "okay", "ok", "sure", "thanks", "thank you"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in acknowledgment_patterns) and len(query) < 20

    def _has_medical_entities(self, slot_out: Dict[str, Any]) -> bool:
        """의료 엔티티 존재 여부"""
        if not slot_out:
            return False

        return bool(
            slot_out.get('symptoms') or
            slot_out.get('conditions') or
            slot_out.get('medications') or
            slot_out.get('vitals') or
            slot_out.get('labs')
        )

    def _estimate_complexity(self, slot_out: Dict[str, Any], query: str) -> str:
        """
        쿼리 복잡도 추정

        기준:
        - simple: 1개 개념, 단순 질문 (20자 이하)
        - moderate: 2-3개 개념, 일반 질문
        - complex: 4개 이상 개념, 복합 질문 (50자 이상)
        """
        # 의료 개념 수 계산
        concept_count = (
            len(slot_out.get('symptoms', [])) +
            len(slot_out.get('conditions', [])) +
            len(slot_out.get('medications', [])) +
            len(slot_out.get('vitals', [])) +
            len(slot_out.get('labs', []))
        )

        # 쿼리 길이
        query_length = len(query)

        # 복잡도 결정
        if concept_count <= 1 and query_length <= 20:
            return "simple"
        elif concept_count <= 3 and query_length <= 50:
            return "moderate"
        else:
            return "complex"

    def _analyze_query_content(
        self,
        query: str,
        conversation_history: str = None
    ) -> Tuple[bool, str]:
        """
        쿼리 내용 분석 (의료 엔티티 없는 경우)

        Returns:
            (needs_retrieval, complexity)
        """
        query_lower = query.lower()
        query_length = len(query)

        # 사실 기반 질문 패턴
        factual_patterns = [
            "뭐", "무엇", "어떻", "어떤", "왜", "어디", "누구",
            "what", "how", "why", "where", "who", "when",
            "?", "정상", "범위", "치료", "원인", "증상", "약", "방법"
        ]

        is_factual = any(pattern in query_lower for pattern in factual_patterns)

        # Follow-up 질문 (대화 이력 참조)
        is_followup = False
        if conversation_history:
            followup_patterns = ["그", "그거", "그게", "그건", "이거", "저거", "that", "it"]
            is_followup = any(pattern in query_lower for pattern in followup_patterns)

        if is_factual:
            if query_length <= 20:
                return True, "simple"
            elif query_length <= 50:
                return True, "moderate"
            else:
                return True, "complex"
        elif is_followup:
            return True, "simple"  # Follow-up은 간단하게
        else:
            # 대화형 질문 - 검색 불필요
            return False, "conversational"

    def _map_complexity_to_k(self, complexity: str) -> int:
        """복잡도를 k 값으로 매핑"""
        mapping = {
            "simple": self.simple_k,
            "moderate": self.moderate_k,
            "complex": self.complex_k,
            "error_fallback": self.default_k,
            "default": self.default_k
        }
        return mapping.get(complexity, self.default_k)

    def _update_complexity_metrics(self, complexity: str):
        """복잡도 메트릭 업데이트"""
        if complexity == "simple":
            self.metrics['simple_queries'] += 1
        elif complexity == "moderate":
            self.metrics['moderate_queries'] += 1
        elif complexity == "complex":
            self.metrics['complex_queries'] += 1

    def _update_avg_time(self, elapsed_ms: float):
        """평균 분류 시간 업데이트"""
        n = self.metrics['total_queries']
        current_avg = self.metrics['avg_classification_time_ms']
        # Incremental average
        self.metrics['avg_classification_time_ms'] = (
            (current_avg * (n - 1) + elapsed_ms) / n
        )

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 반환 (Ablation study용)"""
        total = self.metrics['total_queries']
        if total == 0:
            return self.metrics.copy()

        # 비율 계산
        return {
            **self.metrics,
            'skip_rate': self.metrics['skipped_retrieval'] / total,
            'simple_rate': self.metrics['simple_queries'] / total,
            'moderate_rate': self.metrics['moderate_queries'] / total,
            'complex_rate': self.metrics['complex_queries'] / total,
            'error_rate': self.metrics['classification_errors'] / total
        }

    def reset_metrics(self):
        """메트릭 초기화 (새 실험 시작 시)"""
        for key in self.metrics:
            if isinstance(self.metrics[key], int):
                self.metrics[key] = 0
            else:
                self.metrics[key] = 0.0


def classify_intent_node(state: AgentState) -> AgentState:
    """
    의도 분류 노드

    안전성:
    - Feature flag로 활성화/비활성화
    - 에러 시 기본 동작으로 fallback
    - 메트릭 수집으로 성능 추적

    Returns:
        업데이트된 state (needs_retrieval, dynamic_k, query_complexity 추가)
    """
    print("[Node] classify_intent")

    try:
        # Feature flags 로드
        feature_flags = state.get('feature_flags', {})
        active_retrieval_enabled = feature_flags.get('active_retrieval_enabled', False)

        # 비활성화 시 기본 동작
        if not active_retrieval_enabled:
            print("[INFO] Active Retrieval disabled - using default behavior")
            return {
                **state,
                'needs_retrieval': True,
                'dynamic_k': None,  # None = 기존 로직 사용
                'query_complexity': "default",
                'classification_skipped': True
            }

        # Classifier 초기화 (캐싱)
        if 'intent_classifier' not in state or state['intent_classifier'] is None:
            classifier = IntentClassifier(feature_flags)
            state['intent_classifier'] = classifier
        else:
            classifier = state['intent_classifier']
        
        # Classifier 유효성 검증
        if classifier is None:
            print("[WARNING] IntentClassifier가 None입니다. 기본값을 사용합니다.")
            return {
                **state,
                'needs_retrieval': True,
                'dynamic_k': None,
                'query_complexity': "error_fallback",
                'classification_skipped': True,
                'classification_error': "IntentClassifier is None"
            }

        # 분류 수행
        needs_retrieval, dynamic_k, complexity = classifier.classify(
            query=state['user_text'],
            slot_out=state.get('slot_out', {}),
            conversation_history=state.get('conversation_history')
        )

        # 메트릭 로깅
        print(f"[Active Retrieval] needs_retrieval={needs_retrieval}, k={dynamic_k}, complexity={complexity}")

        # 상태 업데이트
        return {
            **state,
            'needs_retrieval': needs_retrieval,
            'dynamic_k': dynamic_k,
            'query_complexity': complexity,
            'classification_skipped': False,
            'classification_time_ms': classifier.metrics['avg_classification_time_ms']
        }

    except Exception as e:
        # 치명적 에러 시 안전한 기본값
        print(f"[ERROR] classify_intent_node failed: {e}")
        return {
            **state,
            'needs_retrieval': True,
            'dynamic_k': None,
            'query_complexity': "error_fallback",
            'classification_skipped': True,
            'classification_error': str(e)
        }
