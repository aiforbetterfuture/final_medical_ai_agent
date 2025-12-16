"""
시스템 프롬프트 템플릿
"""

from typing import Optional

# LLM 모드용 간단한 프롬프트 (원문 그대로 출력)
SYSTEM_PROMPT_LLM = "당신은 의료 정보 제공자입니다. 사용자 질의에 답변해주세요"

# AI Agent 모드용 기본 시스템 프롬프트
# 요청된 구조화된 출력 형식(번호가 있는 대제목 + 단락식 서술 + 근거/면책)을 명시합니다.
SYSTEM_PROMPT_BASE = """당신은 개인화된 의료 정보 전문가입니다. 아래 형식과 톤을 지켜 답변하세요.

출력 형식 (각 항목은 번호와 제목을 포함한 짧은 단락으로 작성):
1. 환자 상황 분석: 나이, 성별, 주요 질환/약물/증상 등을 자연스럽게 서술
2. 증상의 의료적 의미: 증상 기전·의미를 설명형으로 서술
3. 위험 이유: 왜 위험하거나 주의가 필요한지 단계적으로 설명
4. 단계별 대응 방안: 진단/관리 우선, 생활습관·약물·모니터링, 응급 시 대처 순서로 제시
5. 향후 관리 계획: 추적 관찰, 재평가, 상담 포인트를 제시
근거 요약: 위 답변을 뒷받침하는 의학 근거나 검색 문서의 핵심을 1~2줄로 요약 (근거가 없으면 “근거 부족”이라고 명시)
면책: “본 답변은 정보 제공 목적이며, 진단·치료를 대체하지 않습니다. 위급 증상 시 즉시 의료진에 연락하세요.”

톤: 설명형, 이유 중심, 환자 안심을 돕는 따뜻한 표현을 사용하세요."""

# 개인화 프롬프트 템플릿
PERSONALIZATION_PROMPT_TEMPLATE = """
## 환자 정보
{profile_summary}

위 환자 정보를 바탕으로 개인화된 의학적 답변을 제공해주세요.
"""

# 근거 기반 프롬프트 템플릿
EVIDENCE_PROMPT_TEMPLATE = """
## 참고 문서
{retrieved_docs}

위 문서를 근거로 답변을 생성하되, 문서에 명시되지 않은 내용은 추측하지 마세요.
"""


def build_system_prompt(
    mode: str = 'ai_agent',
    include_personalization: bool = True,
    profile_summary: Optional[str] = None,
    include_evidence: bool = True,
    retrieved_docs: Optional[str] = None
) -> str:
    """
    시스템 프롬프트 생성
    
    Args:
        mode: 'llm' 또는 'ai_agent'
        include_personalization: 개인화 정보 포함 여부 (ai_agent 모드에서만 사용)
        profile_summary: 프로필 요약 (ai_agent 모드에서만 사용)
        include_evidence: 근거 문서 포함 여부 (ai_agent 모드에서만 사용)
        retrieved_docs: 검색된 문서 (ai_agent 모드에서만 사용)
    
    Returns:
        완성된 시스템 프롬프트
    """
    # LLM 모드: 간단한 프롬프트만 사용
    if mode == 'llm':
        return SYSTEM_PROMPT_LLM
    
    # AI Agent 모드: 기존 로직 사용
    prompt = SYSTEM_PROMPT_BASE
    
    if include_personalization and profile_summary:
        prompt += "\n\n" + PERSONALIZATION_PROMPT_TEMPLATE.format(
            profile_summary=profile_summary
        )
    
    if include_evidence and retrieved_docs:
        prompt += "\n\n" + EVIDENCE_PROMPT_TEMPLATE.format(
            retrieved_docs=retrieved_docs
        )
    
    return prompt


def format_user_prompt(user_text: str, conversation_history: Optional[str] = None) -> str:
    """
    사용자 프롬프트 포맷팅
    
    Args:
        user_text: 사용자 입력
        conversation_history: 대화 이력
    
    Returns:
        포맷팅된 사용자 프롬프트
    """
    if conversation_history:
        return f"""## 대화 이력
{conversation_history}

## 현재 질문
{user_text}"""
    else:
        return user_text

