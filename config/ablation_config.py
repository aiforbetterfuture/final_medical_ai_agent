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
