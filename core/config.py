"""
설정 관리 모듈
- YAML 파일 로드
- 환경 변수 관리
- 설정 검증
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# .env 파일 자동 로드
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# 설정 캐시 (성능 최적화)
_config_cache = None
_llm_config_cache = None
_retrieval_config_cache = None
_embedding_config_cache = None
_agent_config_cache = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드 (캐싱)
    
    Args:
        config_path: 설정 파일 경로 (없으면 기본 경로 사용)
    
    Returns:
        설정 딕셔너리
    """
    global _config_cache
    
    # 캐시된 설정이 있고 기본 경로를 사용하는 경우
    if config_path is None and _config_cache is not None:
        return _config_cache
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'corpus_config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 기본 경로인 경우 캐시에 저장
    if config_path == Path(__file__).parent.parent / 'config' / 'corpus_config.yaml':
        _config_cache = config
    
    return config


def get_llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM 설정 반환 (캐싱)
    
    Args:
        config: 전체 설정 (없으면 자동 로드)
    
    Returns:
        LLM 설정 딕셔너리
    """
    global _llm_config_cache
    
    # 캐시된 설정이 있으면 반환
    if _llm_config_cache is not None:
        return _llm_config_cache
    
    if config is None:
        config = load_config()
    
    model_config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
    else:
        # 기본 설정
        model_config = {
            'llm': {
                'provider': os.getenv('LLM_PROVIDER', 'openai'),
                'model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
    
    llm_config = model_config.get('llm', {})
    _llm_config_cache = llm_config
    return llm_config


def get_retrieval_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    검색 설정 반환 (캐싱)
    
    Args:
        config: 전체 설정 (없으면 자동 로드)
    
    Returns:
        검색 설정 딕셔너리
    """
    global _retrieval_config_cache
    
    if _retrieval_config_cache is not None:
        return _retrieval_config_cache
    
    if config is None:
        config = load_config()
    
    retrieval_config = config.get('retrieval', {})
    _retrieval_config_cache = retrieval_config
    return retrieval_config


def get_embedding_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    임베딩 설정 반환 (캐싱)
    
    Args:
        config: 전체 설정 (없으면 자동 로드)
    
    Returns:
        임베딩 설정 딕셔너리
    """
    global _embedding_config_cache
    
    if _embedding_config_cache is not None:
        return _embedding_config_cache
    
    if config is None:
        config = load_config()
    
    embedding_config = config.get('embedding', {})
    _embedding_config_cache = embedding_config
    return embedding_config


def get_agent_config() -> Dict[str, Any]:
    """
    에이전트 기능 플래그/라우팅 설정 반환 (캐싱)
    """
    global _agent_config_cache

    if _agent_config_cache is not None:
        return _agent_config_cache

    agent_config_path = Path(__file__).parent.parent / 'config' / 'agent_config.yaml'
    if agent_config_path.exists():
        with open(agent_config_path, 'r', encoding='utf-8') as f:
            _agent_config_cache = yaml.safe_load(f) or {}
    else:
        _agent_config_cache = {}

    return _agent_config_cache

