"""
노드 5: LLM 답변 생성
"""

from agent.state import AgentState
from core.llm_client import get_llm_client
from core.config import get_llm_config


def generate_answer_node(state: AgentState) -> AgentState:
    """
    답변 생성 노드
    
    LLM을 사용하여 답변을 생성합니다.
    """
    print("[Node] generate_answer")
    
    # LLM 설정 로드 (재현성을 위해 고정 파라미터 사용)
    llm_config = get_llm_config()
    temperature = 0.0
    top_p = 1.0
    max_tokens = llm_config.get('max_tokens', 1000)
    model = llm_config.get('model', 'gpt-4o-mini')
    provider = llm_config.get('provider', 'openai')
    
    # LLM 클라이언트 초기화
    if 'llm_client' not in state:
        llm_client = get_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        state['llm_client'] = llm_client
    else:
        llm_client = state['llm_client']
    
    # 답변 생성
    try:
        # context_prompt가 있으면 사용자 프롬프트 앞에 붙여 맥락을 전달
        combined_user_prompt = "\n\n".join(filter(None, [
            state.get('context_prompt', ''),
            state.get('user_prompt', '')
        ]))
        
        answer = llm_client.generate(
            prompt=combined_user_prompt,
            system_prompt=state['system_prompt'],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"[ERROR] 답변 생성 실패: {e}")
        answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    return {
        **state,
        'answer': answer,
        # 로그 스키마 고정 일부
        'generation_params': {
            'provider': provider,
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
        },
    }

