"""
노드: 질문 유사도 검사 및 캐시 확인
------------------------------------
This node checks if the current question is similar to previously asked questions
and retrieves cached responses if similarity is high enough.
"""

from typing import Dict, Any
from agent.state import AgentState
from memory.response_cache import ResponseCache, ResponseStyleVariator
from core.utils import is_llm_mode
import time


# Global cache instances (singleton pattern for session persistence)
_response_cache = None
_style_variator = None


def get_response_cache() -> ResponseCache:
    """Get or create global response cache instance"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(
            max_cache_size=100,
            similarity_threshold=0.85,  # 85% 유사도 이상일 때만 재사용
            cache_ttl_minutes=60,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _response_cache


def get_style_variator() -> ResponseStyleVariator:
    """Get or create global style variator instance"""
    global _style_variator
    if _style_variator is None:
        _style_variator = ResponseStyleVariator()
    return _style_variator


def check_similarity_node(state: AgentState) -> AgentState:
    """
    Check similarity with cached questions and retrieve response if similar

    This node:
    1. Checks if the current question is semantically similar to previous ones
    2. If similar (>85% similarity), retrieves the cached response
    3. Applies style variations to avoid repetitive output
    4. Skips the entire generation pipeline if cache hit occurs
    """
    print("[Node] check_similarity")

    # Skip in LLM mode
    if is_llm_mode(state):
        return {**state, 'cache_hit': False}

    # Get feature flags
    feature_flags = state.get('feature_flags', {})
    cache_enabled = feature_flags.get('response_cache_enabled', True)
    similarity_threshold = feature_flags.get('cache_similarity_threshold', 0.85)
    style_variation_level = feature_flags.get('style_variation_level', 0.3)

    if not cache_enabled:
        return {**state, 'cache_hit': False}

    # Get cache and variator instances
    cache = get_response_cache()
    variator = get_style_variator()

    # Update cache threshold if different
    cache.similarity_threshold = similarity_threshold

    # Get current query
    user_query = state.get('user_text', '')

    # Check for similar cached query
    start_time = time.time()
    cache_result = cache.find_similar(user_query)

    if cache_result:
        cached_response, similarity_score = cache_result
        print(f"[Cache Hit] Found similar question with {similarity_score:.2%} similarity")

        # 캐시된 응답 확인
        original_response = cached_response.response
        if not original_response or not original_response.strip():
            print(f"[WARNING] 캐시된 응답이 비어있습니다. 원본 질문: {cached_response.query[:50]}...")
            return {
                **state,
                'cache_hit': False,
                'skip_pipeline': False,
                'cache_stats': cache.get_stats()
            }

        # Apply style variations to avoid repetition
        try:
            varied_response = variator.vary_response(
                original_response,
                variation_level=style_variation_level
            )
            
            # 변형된 응답이 비어있는 경우 원본 사용
            if not varied_response or not varied_response.strip():
                print(f"[WARNING] 변형된 응답이 비어있어 원본 사용")
                varied_response = original_response
        except Exception as e:
            print(f"[WARNING] 스타일 변형 실패: {e}, 원본 응답 사용")
            varied_response = original_response

        # Calculate savings (approximate)
        # Assume average tokens: query processing (50) + retrieval (200) + generation (500) = 750 tokens
        estimated_tokens_saved = 750
        time_saved_ms = int((time.time() - start_time) * 1000)

        # Update cache statistics
        cache.update_stats(
            tokens_saved=estimated_tokens_saved,
            time_saved_ms=time_saved_ms
        )

        # Update state to skip remaining pipeline
        return {
            **state,
            'cache_hit': True,
            'cached_response': varied_response,
            'cache_similarity_score': similarity_score,
            'answer': varied_response,  # Set answer directly
            'skip_pipeline': True,  # Signal to skip remaining nodes
            'cache_stats': cache.get_stats()
        }

    print("[Cache Miss] No similar question found in cache")
    return {
        **state,
        'cache_hit': False,
        'skip_pipeline': False,
        'cache_stats': cache.get_stats()
    }


def store_response_node(state: AgentState) -> AgentState:
    """
    Store generated response in cache for future reuse

    This node runs after answer generation to cache the response
    """
    print("[Node] store_response")

    # Skip if this was a cache hit (already cached)
    if state.get('cache_hit', False):
        # 캐시 히트인 경우 answer가 설정되어 있는지 확인
        answer = state.get('answer', '')
        cached_response = state.get('cached_response', '')
        
        # answer가 비어있고 cached_response가 있으면 사용
        if not answer or not answer.strip():
            if cached_response:
                print(f"[INFO] 캐시 히트: cached_response를 answer로 설정 ({len(cached_response)}자)")
                return {
                    **state,
                    'answer': cached_response
                }
            else:
                print(f"[WARNING] 캐시 히트인데 answer와 cached_response가 모두 비어있습니다")
        
        return state

    # Skip in LLM mode
    if is_llm_mode(state):
        return state

    # Get feature flags
    feature_flags = state.get('feature_flags', {})
    cache_enabled = feature_flags.get('response_cache_enabled', True)

    if not cache_enabled:
        return state

    # Get cache instance
    cache = get_response_cache()

    # Store the generated response
    user_query = state.get('user_text', '')
    answer = state.get('answer', '')

    if user_query and answer:
        metadata = {
            'quality_score': state.get('quality_score', 0.0),
            'iteration_count': state.get('iteration_count', 0),
            'session_id': state.get('session_id', ''),
            'user_id': state.get('user_id', ''),
            'mode': state.get('mode', 'ai_agent')
        }

        cache.add(
            query=user_query,
            response=answer,
            metadata=metadata
        )

        print(f"[Cache Store] Response cached. Cache size: {len(cache.cache)}")

    return state