"""
싱글톤 캐시 시스템

목적:
- 코퍼스, FAISS 인덱스, 임베딩 모델 등을 1번만 로드
- 멀티턴 실험에서 반복 로딩으로 인한 시간/토큰 낭비 방지
- 메모리 효율성 향상

사용법:
    from retrieval.singleton_cache import get_corpus, get_faiss_index, get_embedding_model
    
    corpus = get_corpus()  # 첫 호출: 로드, 이후: 캐시 반환
    index = get_faiss_index()  # 첫 호출: 로드, 이후: 캐시 반환
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading

logger = logging.getLogger(__name__)

# 전역 캐시 (싱글톤)
_CACHE = {
    'corpus': None,
    'corpus_metadata': None,
    'faiss_index': None,
    'faiss_metadata': None,
    'embedding_model': None,
    'bm25_index': None,
}

# 스레드 안전성을 위한 락
_CACHE_LOCK = threading.Lock()

# 캐시 통계
_CACHE_STATS = {
    'corpus_loads': 0,
    'corpus_hits': 0,
    'faiss_loads': 0,
    'faiss_hits': 0,
    'embedding_loads': 0,
    'embedding_hits': 0,
    'bm25_loads': 0,
    'bm25_hits': 0,
}


def get_corpus(corpus_path: Optional[str] = None) -> Tuple[List[str], List[Dict]]:
    """
    코퍼스 로드 (싱글톤)
    
    Args:
        corpus_path: 코퍼스 파일 경로 (기본값: data/corpus/train_source_data.jsonl)
    
    Returns:
        (corpus_texts, corpus_metadata)
    """
    with _CACHE_LOCK:
        if _CACHE['corpus'] is not None:
            _CACHE_STATS['corpus_hits'] += 1
            logger.debug(f"[캐시 HIT] 코퍼스 (hits: {_CACHE_STATS['corpus_hits']})")
            return _CACHE['corpus'], _CACHE['corpus_metadata']
        
        # 첫 로드
        _CACHE_STATS['corpus_loads'] += 1
        logger.info(f"[캐시 MISS] 코퍼스 로드 중... (loads: {_CACHE_STATS['corpus_loads']})")
        
        if corpus_path is None:
            corpus_path = "data/corpus/train_source_data.jsonl"
        
        corpus_texts, corpus_metadata = _load_corpus_internal(corpus_path)
        
        _CACHE['corpus'] = corpus_texts
        _CACHE['corpus_metadata'] = corpus_metadata
        
        logger.info(f"[캐시 저장] 코퍼스 로드 완료: {len(corpus_texts)}개 문서")
        
        return corpus_texts, corpus_metadata


def get_faiss_index(index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> Tuple[Any, List[Dict]]:
    """
    FAISS 인덱스 로드 (싱글톤)
    
    Args:
        index_path: FAISS 인덱스 파일 경로
        metadata_path: 메타데이터 파일 경로
    
    Returns:
        (faiss_index, metadata)
    """
    with _CACHE_LOCK:
        if _CACHE['faiss_index'] is not None:
            _CACHE_STATS['faiss_hits'] += 1
            logger.debug(f"[캐시 HIT] FAISS 인덱스 (hits: {_CACHE_STATS['faiss_hits']})")
            return _CACHE['faiss_index'], _CACHE['faiss_metadata']
        
        # 첫 로드
        _CACHE_STATS['faiss_loads'] += 1
        logger.info(f"[캐시 MISS] FAISS 인덱스 로드 중... (loads: {_CACHE_STATS['faiss_loads']})")
        
        if index_path is None:
            index_path = "data/index/train_source/train_source_data.index.faiss"
        if metadata_path is None:
            metadata_path = "data/index/train_source/train_source_data.index.metadata.json"
        
        faiss_index, metadata = _load_faiss_index_internal(index_path, metadata_path)
        
        _CACHE['faiss_index'] = faiss_index
        _CACHE['faiss_metadata'] = metadata
        
        logger.info(f"[캐시 저장] FAISS 인덱스 로드 완료: {len(metadata)}개 문서")
        
        return faiss_index, metadata


def get_embedding_model(model_name: Optional[str] = None):
    """
    임베딩 모델 로드 (싱글톤)
    
    Args:
        model_name: 임베딩 모델명 (기본값: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
    
    Returns:
        SentenceTransformer 모델
    """
    with _CACHE_LOCK:
        if _CACHE['embedding_model'] is not None:
            _CACHE_STATS['embedding_hits'] += 1
            logger.debug(f"[캐시 HIT] 임베딩 모델 (hits: {_CACHE_STATS['embedding_hits']})")
            return _CACHE['embedding_model']
        
        # 첫 로드
        _CACHE_STATS['embedding_loads'] += 1
        logger.info(f"[캐시 MISS] 임베딩 모델 로드 중... (loads: {_CACHE_STATS['embedding_loads']})")
        
        if model_name is None:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        model = _load_embedding_model_internal(model_name)
        
        _CACHE['embedding_model'] = model
        
        logger.info(f"[캐시 저장] 임베딩 모델 로드 완료: {model_name}")
        
        return model


def get_bm25_index(corpus_texts: Optional[List[str]] = None):
    """
    BM25 인덱스 로드 (싱글톤)
    
    Args:
        corpus_texts: 코퍼스 텍스트 리스트
    
    Returns:
        BM25Okapi 인덱스
    """
    with _CACHE_LOCK:
        if _CACHE['bm25_index'] is not None:
            _CACHE_STATS['bm25_hits'] += 1
            logger.debug(f"[캐시 HIT] BM25 인덱스 (hits: {_CACHE_STATS['bm25_hits']})")
            return _CACHE['bm25_index']
        
        # 첫 로드
        _CACHE_STATS['bm25_loads'] += 1
        logger.info(f"[캐시 MISS] BM25 인덱스 생성 중... (loads: {_CACHE_STATS['bm25_loads']})")
        
        if corpus_texts is None:
            corpus_texts, _ = get_corpus()
        
        bm25_index = _build_bm25_index_internal(corpus_texts)
        
        _CACHE['bm25_index'] = bm25_index
        
        logger.info(f"[캐시 저장] BM25 인덱스 생성 완료: {len(corpus_texts)}개 문서")
        
        return bm25_index


def clear_cache():
    """캐시 초기화"""
    with _CACHE_LOCK:
        logger.info("[캐시 초기화] 모든 캐시 삭제")
        _CACHE['corpus'] = None
        _CACHE['corpus_metadata'] = None
        _CACHE['faiss_index'] = None
        _CACHE['faiss_metadata'] = None
        _CACHE['embedding_model'] = None
        _CACHE['bm25_index'] = None


def get_cache_stats() -> Dict[str, int]:
    """캐시 통계 반환"""
    return _CACHE_STATS.copy()


def print_cache_stats():
    """캐시 통계 출력"""
    stats = get_cache_stats()
    logger.info("=" * 80)
    logger.info("[캐시 통계]")
    logger.info(f"  코퍼스:     로드 {stats['corpus_loads']}회, 히트 {stats['corpus_hits']}회")
    logger.info(f"  FAISS:      로드 {stats['faiss_loads']}회, 히트 {stats['faiss_hits']}회")
    logger.info(f"  임베딩:     로드 {stats['embedding_loads']}회, 히트 {stats['embedding_hits']}회")
    logger.info(f"  BM25:       로드 {stats['bm25_loads']}회, 히트 {stats['bm25_hits']}회")
    logger.info("=" * 80)


# ============================================================================
# 내부 로딩 함수들
# ============================================================================

def _load_corpus_internal(corpus_path: str) -> Tuple[List[str], List[Dict]]:
    """코퍼스 로드 (내부 함수)"""
    import json
    
    corpus_texts = []
    corpus_metadata = []
    
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"코퍼스 파일을 찾을 수 없습니다: {corpus_path}")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus_texts.append(doc.get('text', ''))
                corpus_metadata.append(doc)
    
    return corpus_texts, corpus_metadata


def _load_faiss_index_internal(index_path: str, metadata_path: str) -> Tuple[Any, List[Dict]]:
    """FAISS 인덱스 로드 (내부 함수)"""
    import faiss
    import json
    
    index_file = Path(index_path)
    metadata_file = Path(metadata_path)
    
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {index_path}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    
    # FAISS 인덱스 로드
    faiss_index = faiss.read_index(str(index_file))
    
    # 메타데이터 로드
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return faiss_index, metadata


def _load_embedding_model_internal(model_name: str):
    """임베딩 모델 로드 (내부 함수)"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    return model


def _build_bm25_index_internal(corpus_texts: List[str]):
    """BM25 인덱스 생성 (내부 함수)"""
    from rank_bm25 import BM25Okapi
    
    # 토큰화 (간단한 공백 기반)
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    return bm25_index

