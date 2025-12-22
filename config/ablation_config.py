"""
Ablation Study 설정 관리

Self-Refine과 Quality Check의 on/off 실험을 위한 설정 프로파일을 제공합니다.
각 프로파일은 특정 기능 조합을 활성화/비활성화하여 성능 영향을 측정할 수 있습니다.
"""

from typing import Dict, Any


# ============================================================
# Ablation Study 프로파일
# ============================================================

ABLATION_PROFILES = {
    # === 베이스라인 (모든 기능 비활성화) ===
    "baseline": {
        "description": "베이스라인: Self-Refine 없음, 1회 검색-생성만",
        "features": {
            "self_refine_enabled": False,
            "quality_check_enabled": False,
            "llm_based_quality_check": False,
            "dynamic_query_rewrite": False,
            "duplicate_detection": False,
            "progress_monitoring": False,
        }
    },

    # === Self-Refine 활성화 (휴리스틱 품질 평가) ===
    "self_refine_heuristic": {
        "description": "Self-Refine + 휴리스틱 품질 평가 (LLM 평가 없음)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": False,  # 휴리스틱 평가만 사용
            "dynamic_query_rewrite": False,  # 정적 질의
            "duplicate_detection": False,
            "progress_monitoring": False,
            "max_refine_iterations": 2,
            "quality_threshold": 0.5,
        }
    },

    # === Self-Refine + LLM 품질 평가 ===
    "self_refine_llm_quality": {
        "description": "Self-Refine + LLM 기반 품질 평가 (동적 질의 재작성 없음)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,  # LLM 평가 활성화
            "dynamic_query_rewrite": False,  # 정적 질의
            "duplicate_detection": False,
            "progress_monitoring": False,
            "max_refine_iterations": 2,
            "quality_threshold": 0.5,
        }
    },

    # === Self-Refine + 동적 질의 재작성 ===
    "self_refine_dynamic_query": {
        "description": "Self-Refine + LLM 품질 평가 + 동적 질의 재작성",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,  # 동적 질의 재작성 활성화
            "duplicate_detection": False,
            "progress_monitoring": False,
            "max_refine_iterations": 2,
            "quality_threshold": 0.5,
        }
    },

    # === Self-Refine + Quality Check (2중 안전장치) ===
    "self_refine_full_safety": {
        "description": "Self-Refine + Quality Check (2중 안전장치: 중복 검색 방지 + 진행도 모니터링)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,  # 중복 검색 방지
            "progress_monitoring": True,  # 진행도 모니터링
            "max_refine_iterations": 2,
            "quality_threshold": 0.5,
        }
    },

    # === 전체 활성화 (Context Engineering 기반 최종 버전) ===
    "full_context_engineering": {
        "description": "Context Engineering 기반 전체 활성화 (최종 버전)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 3,  # 더 많은 iteration 허용
            "quality_threshold": 0.6,  # 더 높은 품질 기준
        }
    },

    # === Quality Check만 활성화 (Self-Refine 없음) ===
    "quality_check_only": {
        "description": "Quality Check만 활성화 (Self-Refine 비활성화)",
        "features": {
            "self_refine_enabled": False,  # Self-Refine 비활성화
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": False,
            "duplicate_detection": True,
            "progress_monitoring": True,
        }
    },

    # === Self-Refine만 활성화 (Quality Check 없음) ===
    "self_refine_no_safety": {
        "description": "Self-Refine만 활성화 (Quality Check 안전장치 없음)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": False,  # Quality Check 비활성화
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": False,  # 안전장치 없음
            "progress_monitoring": False,  # 안전장치 없음
            "max_refine_iterations": 2,
            "quality_threshold": 0.5,
        }
    },

    # ============================================================
    # 고도화 프로파일 (개인화 강화)
    # ============================================================

    # === 슬롯 기반 메모리 강화 ===
    "personalized_slot_memory": {
        "description": "슬롯 기반 구조화 메모리 (confidence/provenance/TTL 포함)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.6,
            # 메모리 강화 설정
            "memory_mode": "structured",  # structured/hierarchical/none
            "profile_update_enabled": True,
            "temporal_weight_enabled": True,
            "slot_confidence_tracking": True,  # 슬롯별 신뢰도 추적
            "slot_provenance_tracking": True,  # 근거 추적
            "slot_conflict_detection": True,  # 모순 감지
        }
    },

    # === 개인화 정책 레이어 ===
    "personalized_policy_layer": {
        "description": "컨텍스트 완전성 기반 질문/답변 라우팅",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.6,
            # 정책 레이어 설정
            "context_completeness_check": True,  # 컨텍스트 완전성 점수
            "personalization_gate": True,  # 개인화 안전성 판단
            "action_routing": True,  # ASK_CLARIFY/RETRIEVE/ANSWER 선택
            "required_slots_check": True,  # 필수 슬롯 확인
        }
    },

    # === 컨텍스트 기반 쿼리 재작성 ===
    "contextual_query_rewrite": {
        "description": "사용자 슬롯을 반영한 동적 쿼리 재작성",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.6,
            # 쿼리 재작성 강화
            "slot_aware_query_expansion": True,  # 슬롯 기반 쿼리 확장
            "query_expansion_count": 3,  # 2~4개 쿼리로 확장
            "retrieval_diversity_constraint": True,  # MMR 기반 중복 억제
            "user_context_reranking": True,  # 사용자 적합도 재랭킹
        }
    },

    # === 컨텍스트 패킷 표준화 ===
    "context_packet_standard": {
        "description": "토큰 예산 기반 컨텍스트 주입 통제",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.6,
            # 컨텍스트 패킷 설정
            "use_context_manager": True,
            "budget_aware_retrieval": True,
            "context_packet_priority": True,  # A(확정) > B(불확실) > C(이력) > D(근거)
            "context_conflict_resolution": True,  # 충돌 시 사용자 확인
            "include_history": True,
            "include_profile": True,
            "include_evidence": True,
            "include_personalization": True,
        }
    },

    # === 조건부 Refine 실행 ===
    "conditional_refine": {
        "description": "리스크 기반 조건부 Refine 실행 (비용 절감)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.7,  # 더 높은 임계값
            # 조건부 실행 설정
            "refine_risk_detection": True,  # 리스크 탐지기
            "refine_skip_on_pass": True,  # 통과 시 Refine 생략
            "refine_early_termination": True,  # 명확한 종료 조건
            "refine_checklist": [
                "citation_missing",  # 근거 인용 누락
                "contradiction",  # 모순
                "question_unanswered",  # 질문 미응답
                "medical_warning_missing",  # 의료 경고 누락
            ],
        }
    },

    # === 검증 가능 개인화 ===
    "verifiable_personalization": {
        "description": "개인화 근거를 답변에 명시 (검증 가능)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.6,
            # 검증 가능성 설정
            "include_personalization_evidence": True,  # "당신이 이전에 말한 ○○..."
            "include_information_status": True,  # "현재 알려진 정보는 A/B..."
            "include_confirmation_needed": True,  # "C는 아직 확인이 필요합니다"
            "privacy_aware": True,  # 민감정보 노출 최소화
        }
    },

    # === 의료 안전 트리아지 ===
    "medical_safety_triage": {
        "description": "경고증상 감지 시 답변 모드 전환 (안전 우선)",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 2,
            "quality_threshold": 0.6,
            # 안전 트리아지 설정
            "red_flag_detection": True,  # 경고증상 감지
            "severity_classification": True,  # 중증도 분류
            "emergency_mode_switch": True,  # 응급 모드 전환
            "diagnostic_prohibition": True,  # 진단 단정 금지
            "uncertainty_disclosure": True,  # 불확실성 고지
            "specialist_referral": True,  # 전문의 상담 권고
        }
    },

    # === 최종 고도화 (모든 개선 포함) ===
    "advanced_personalized_rag": {
        "description": "최종 고도화: 슬롯 메모리 + 정책 레이어 + 조건부 Refine + 안전 트리아지",
        "features": {
            "self_refine_enabled": True,
            "quality_check_enabled": True,
            "llm_based_quality_check": True,
            "dynamic_query_rewrite": True,
            "duplicate_detection": True,
            "progress_monitoring": True,
            "max_refine_iterations": 3,
            "quality_threshold": 0.7,
            # 메모리 강화
            "memory_mode": "structured",
            "profile_update_enabled": True,
            "temporal_weight_enabled": True,
            "slot_confidence_tracking": True,
            "slot_provenance_tracking": True,
            "slot_conflict_detection": True,
            # 정책 레이어
            "context_completeness_check": True,
            "personalization_gate": True,
            "action_routing": True,
            "required_slots_check": True,
            # 쿼리 재작성
            "slot_aware_query_expansion": True,
            "query_expansion_count": 3,
            "retrieval_diversity_constraint": True,
            "user_context_reranking": True,
            # 컨텍스트 패킷
            "use_context_manager": True,
            "budget_aware_retrieval": True,
            "context_packet_priority": True,
            "context_conflict_resolution": True,
            # 조건부 Refine
            "refine_risk_detection": True,
            "refine_skip_on_pass": True,
            "refine_early_termination": True,
            # 검증 가능성
            "include_personalization_evidence": True,
            "include_information_status": True,
            "include_confirmation_needed": True,
            "privacy_aware": True,
            # 안전 트리아지
            "red_flag_detection": True,
            "severity_classification": True,
            "emergency_mode_switch": True,
            "diagnostic_prohibition": True,
            "uncertainty_disclosure": True,
            "specialist_referral": True,
        }
    },
}


# ============================================================
# 헬퍼 함수
# ============================================================

def get_ablation_profile(profile_name: str) -> Dict[str, Any]:
    """
    Ablation 프로파일 가져오기

    Args:
        profile_name: 프로파일 이름 (예: "baseline", "full_context_engineering")

    Returns:
        프로파일 설정 딕셔너리

    Raises:
        ValueError: 존재하지 않는 프로파일
    """
    if profile_name not in ABLATION_PROFILES:
        available = ", ".join(ABLATION_PROFILES.keys())
        raise ValueError(
            f"존재하지 않는 ablation 프로파일: '{profile_name}'. "
            f"사용 가능한 프로파일: {available}"
        )

    profile = ABLATION_PROFILES[profile_name]
    return profile["features"]


def list_ablation_profiles() -> Dict[str, str]:
    """
    사용 가능한 Ablation 프로파일 목록과 설명 반환

    Returns:
        {profile_name: description} 딕셔너리
    """
    return {
        name: profile["description"]
        for name, profile in ABLATION_PROFILES.items()
    }


def print_ablation_profiles():
    """Ablation 프로파일 목록을 콘솔에 출력"""
    print("=" * 80)
    print("사용 가능한 Ablation Study 프로파일")
    print("=" * 80)

    for name, profile in ABLATION_PROFILES.items():
        print(f"\n[{name}]")
        print(f"  설명: {profile['description']}")
        print(f"  설정:")
        for key, value in profile["features"].items():
            print(f"    - {key}: {value}")

    print("\n" + "=" * 80)


# ============================================================
# 사용 예제
# ============================================================

if __name__ == "__main__":
    # 프로파일 목록 출력
    print_ablation_profiles()

    # 특정 프로파일 로드
    print("\n\n=== 'full_context_engineering' 프로파일 로드 ===")
    full_features = get_ablation_profile("full_context_engineering")
    print(full_features)

    # Agent 실행 시 사용 예제
    print("\n\n=== Agent 실행 예제 (코드) ===")
    print("""
    from agent.graph import run_agent
    from config.ablation_config import get_ablation_profile

    # Ablation 프로파일 선택
    ablation_features = get_ablation_profile("full_context_engineering")

    # Agent 실행 (feature_overrides로 전달)
    result = run_agent(
        user_text="당뇨병 환자에게 메트포르민의 부작용은 무엇인가요?",
        mode="ai_agent",
        feature_overrides=ablation_features,
        return_state=True  # 상세 로그를 위해 전체 상태 반환
    )

    # Iteration 로그 확인
    refine_logs = result.get('refine_iteration_logs', [])
    for log in refine_logs:
        print(f"Iteration {log['iteration']}: Quality Score = {log['quality_score']:.2f}")
    """)
