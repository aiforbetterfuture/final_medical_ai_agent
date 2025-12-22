"""Repository integration layer for entity extraction A/B tests.

현재 스캐폴드 개선사항:
- Agent 코드가 개별 extractor를 직접 import하지 않고 이 라우터를 통해 호출
- 환경 변수로 extractor 교체 가능 (코드 수정 불필요)
- 캐싱을 통한 성능 최적화
- 현재 스캐폴드의 기존 구조(src/med_entity_ab/) 완전 호환

사용 예시:
    from extraction.entity_ab_router import extract_entities
    
    # 단일 extractor 사용
    result = extract_entities(user_text, extractor="medcat")
    
    # 환경 변수로 제어
    result = extract_entities(user_text, extractor=os.getenv("ENTITY_EXTRACTOR", "medcat"))
    
    # 모든 extractor 동시 실행 (비교 모드)
    all_results = extract_entities_all(user_text)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

from med_entity_ab.pipeline import EntityABPipeline, load_config
from med_entity_ab.schema import Entity

# 기본 설정 파일 경로 (환경 변수로 오버라이드 가능)
DEFAULT_CONFIG_PATH = os.getenv("ENTITY_AB_CONFIG", "configs/default.yaml")


@lru_cache(maxsize=1)
def _get_pipeline(config_path: str = DEFAULT_CONFIG_PATH) -> EntityABPipeline:
    """파이프라인 싱글톤 (성능 최적화)"""
    cfg = load_config(config_path)
    return EntityABPipeline(cfg)


def extract_entities(
    text: str,
    extractor: str = "medcat",
    *,
    config_path: Optional[str] = None,
) -> List[Entity]:
    """단일 extractor로 엔티티 추출
    
    Args:
        text: 입력 텍스트
        extractor: 사용할 extractor 이름
            - "medcat": MedCAT (UMLS 기반 concept extraction/linking)
            - "quickumls": QuickUMLS (UMLS 문자열 매칭)
            - "kmbert_ner": KM-BERT NER (한국어 의료 NER)
        config_path: 설정 파일 경로 (선택, 기본값: configs/default.yaml)
    
    Returns:
        Entity 객체 리스트
    
    Raises:
        ValueError: 알 수 없는 extractor 이름
    
    Example:
        >>> from extraction.entity_ab_router import extract_entities
        >>> entities = extract_entities("당뇨병 환자입니다", extractor="medcat")
        >>> for ent in entities:
        ...     print(f"{ent.text} ({ent.label}): {ent.code}")
    """
    pipe = _get_pipeline(config_path or DEFAULT_CONFIG_PATH)
    results = pipe.extract_all(text)
    
    if extractor not in results:
        available = list(results.keys())
        raise ValueError(
            f"Unknown extractor: '{extractor}'. "
            f"Available extractors: {available}. "
            f"Check configs/default.yaml to enable extractors."
        )
    
    return results[extractor].entities


def extract_entities_all(
    text: str,
    *,
    config_path: Optional[str] = None,
) -> Dict[str, List[Entity]]:
    """모든 활성화된 extractor로 엔티티 추출 (비교 모드)
    
    Args:
        text: 입력 텍스트
        config_path: 설정 파일 경로 (선택)
    
    Returns:
        {extractor_name: [Entity, ...]} 딕셔너리
    
    Example:
        >>> from extraction.entity_ab_router import extract_entities_all
        >>> all_results = extract_entities_all("당뇨병 환자입니다")
        >>> for name, entities in all_results.items():
        ...     print(f"{name}: {len(entities)} entities")
    """
    pipe = _get_pipeline(config_path or DEFAULT_CONFIG_PATH)
    results = pipe.extract_all(text)
    
    # ExtractResult → List[Entity] 변환
    return {
        name: result.entities
        for name, result in results.items()
    }


def get_available_extractors(config_path: Optional[str] = None) -> List[str]:
    """현재 활성화된 extractor 목록 반환
    
    Args:
        config_path: 설정 파일 경로 (선택)
    
    Returns:
        활성화된 extractor 이름 리스트
    
    Example:
        >>> from extraction.entity_ab_router import get_available_extractors
        >>> extractors = get_available_extractors()
        >>> print(f"Available: {extractors}")
    """
    pipe = _get_pipeline(config_path or DEFAULT_CONFIG_PATH)
    return list(pipe.extractors.keys())


def extract_entities_with_metadata(
    text: str,
    extractor: str = "medcat",
    *,
    config_path: Optional[str] = None,
) -> Dict[str, any]:
    """엔티티 추출 + 메타데이터 (latency 등) 반환
    
    Args:
        text: 입력 텍스트
        extractor: 사용할 extractor 이름
        config_path: 설정 파일 경로 (선택)
    
    Returns:
        {
            'entities': [Entity, ...],
            'latency_ms': float,
            'extractor': str
        }
    
    Example:
        >>> result = extract_entities_with_metadata("당뇨병", extractor="medcat")
        >>> print(f"Found {len(result['entities'])} entities in {result['latency_ms']:.1f}ms")
    """
    pipe = _get_pipeline(config_path or DEFAULT_CONFIG_PATH)
    results = pipe.extract_all(text)
    
    if extractor not in results:
        available = list(results.keys())
        raise ValueError(
            f"Unknown extractor: '{extractor}'. "
            f"Available: {available}"
        )
    
    result = results[extractor]
    
    return {
        'entities': result.entities,
        'latency_ms': result.latency_ms,
        'extractor': extractor,
        'num_entities': len(result.entities)
    }


# Agent 통합을 위한 편의 함수
def extract_for_agent(
    text: str,
    extractor: Optional[str] = None,
) -> List[Entity]:
    """Agent에서 사용하기 위한 간편 함수
    
    환경 변수 ENTITY_EXTRACTOR로 extractor 선택 가능
    기본값: medcat
    
    Args:
        text: 입력 텍스트
        extractor: extractor 이름 (None이면 환경 변수 사용)
    
    Returns:
        Entity 객체 리스트
    
    Example:
        >>> # .env 파일에 ENTITY_EXTRACTOR=quickumls 설정
        >>> from extraction.entity_ab_router import extract_for_agent
        >>> entities = extract_for_agent("당뇨병 환자")
    """
    if extractor is None:
        extractor = os.getenv("ENTITY_EXTRACTOR", "medcat")
    
    return extract_entities(text, extractor=extractor)

