"""
LLM 클라이언트 통합
- OpenAI API
- Google Gemini API
- 통일된 인터페이스
"""

import os
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """LLM 클라이언트 추상 클래스"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """임베딩 생성 (선택적)"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI 클라이언트"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4o-mini', embedding_model: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.embedding_model = embedding_model  # 임베딩 모델 저장
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai 패키지가 설치되지 않았습니다. pip install openai")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """텍스트 생성"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("LLM 응답이 비어있습니다")
            
            return response.choices[0].message.content
        except Exception as e:
            # 에러를 그대로 전파 (상위에서 처리)
            raise
    
    def embed(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        """
        임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            embedding_model: 임베딩 모델명 (없으면 설정에서 읽음)
        
        Returns:
            임베딩 벡터 (3072차원 for text-embedding-3-large)
        """
        # 모델명 결정: 인자 > 인스턴스 속성 > 기본값
        if embedding_model is None:
            embedding_model = getattr(self, 'embedding_model', None)
        
        if embedding_model is None:
            # 설정에서 읽기
            from core.config import get_embedding_config
            embedding_config = get_embedding_config()
            embedding_model = embedding_config.get('model', 'text-embedding-3-large')
        
        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=text
            )
            
            if not response.data or not response.data[0].embedding:
                raise ValueError("임베딩 응답이 비어있습니다")
            
            return response.data[0].embedding
        except Exception as e:
            # 에러를 그대로 전파 (상위에서 처리)
            raise


class GeminiClient(LLMClient):
    """Google Gemini 클라이언트"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-2.0-flash-exp', **kwargs):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """텍스트 생성"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.generate_content(
            full_prompt,
            generation_config={
                'temperature': kwargs.get('temperature', self.temperature),
                'max_output_tokens': kwargs.get('max_tokens', self.max_tokens)
            }
        )
        
        return response.text
    
    def embed(self, text: str) -> List[float]:
        """임베딩 생성 (Gemini는 임베딩 API가 다를 수 있음)"""
        # Gemini 임베딩은 별도 구현 필요
        raise NotImplementedError("Gemini 임베딩은 별도 구현이 필요합니다.")


def get_llm_client(provider: str = 'openai', **kwargs) -> LLMClient:
    """
    LLM 클라이언트 팩토리 함수
    
    Args:
        provider: 'openai' 또는 'gemini'
        **kwargs: 클라이언트별 설정
    
    Returns:
        LLMClient 인스턴스
    """
    if provider.lower() == 'openai':
        return OpenAIClient(**kwargs)
    elif provider.lower() == 'gemini':
        return GeminiClient(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")

