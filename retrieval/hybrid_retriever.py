"""
하이브리드 검색기
- BM25 키워드 검색
- FAISS 의미 검색
- RRF 융합
- 전역 싱글톤 캐시로 멀티턴 대화에서 재사용
"""

import os
import re
import json
import heapq
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

from .faiss_index import FAISSIndex
from .rrf_fusion import rrf_fusion


# 전역 캐시: 같은 경로에 대해서는 한 번만 로드
_BM25_RETRIEVER_CACHE: Dict[str, 'BM25Retriever'] = {}
_HYBRID_RETRIEVER_CACHE: Dict[str, 'HybridRetriever'] = {}


def tokenize_ko_en(text: str) -> List[str]:
    """한국어/영어 토큰화"""
    return re.findall(r"[A-Za-z가-힣0-9]+", text.lower())


class BM25Retriever:
    """BM25 키워드 검색기"""
    
    def __init__(self, corpus_path: str):
        """
        Args:
            corpus_path: 코퍼스 파일 경로 (JSONL)
        
        Note:
            같은 경로에 대해서는 전역 캐시에서 재사용합니다.
            멀티턴 대화에서 매 턴마다 재로딩하는 것을 방지합니다.
        """
        # 정규화된 경로를 키로 사용
        corpus_path = os.path.abspath(corpus_path)
        cache_key = corpus_path
        
        # 캐시에 있으면 재사용
        if cache_key in _BM25_RETRIEVER_CACHE:
            cached = _BM25_RETRIEVER_CACHE[cache_key]
            self.corpus_path = cached.corpus_path
            self.corpus_docs = cached.corpus_docs
            self.bm25_index = cached.bm25_index
            return
        
        # 새로 생성
        self.corpus_path = corpus_path
        self.corpus_docs = []
        self.bm25_index = None
        self._load_corpus()
        
        # 캐시에 저장
        _BM25_RETRIEVER_CACHE[cache_key] = self
    
    def _load_corpus(self):
        """코퍼스 로드 (캐시에서 재사용 중이면 로드하지 않음)"""
        # 이미 로드되어 있으면 스킵
        if self.bm25_index is not None:
            return
            
        if not HAS_BM25:
            print("[WARNING] rank-bm25가 설치되지 않았습니다.")
            return
        
        if not os.path.exists(self.corpus_path):
            print(f"[WARNING] 코퍼스 파일을 찾을 수 없습니다: {self.corpus_path}")
            return
        
        try:
            self.corpus_docs = []
            texts = []
            
            # JSONL 파일 읽기
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        self.corpus_docs.append(doc)
                        texts.append(doc.get('text', ''))
            
            # BM25 인덱스 구축
            tokenized = [tokenize_ko_en(text) for text in texts]
            self.bm25_index = BM25Okapi(tokenized)
            
            print(f"[BM25] 코퍼스 로드 완료: {len(self.corpus_docs)}개 문서")
        except Exception as e:
            print(f"[ERROR] 코퍼스 로드 실패: {e}")
            self.bm25_index = None
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25 검색
        
        Args:
            query: 쿼리 텍스트
            k: 반환할 상위 k개
        
        Returns:
            검색 결과 리스트
        """
        if not HAS_BM25 or self.bm25_index is None:
            return []
        
        try:
            query_tokens = tokenize_ko_en(query)
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 상위 k개만 선택 (O(n log k) - 전체 정렬 대신)
            top_indices = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
            
            results = []
            for rank, idx in enumerate(top_indices, start=1):
                doc = self.corpus_docs[idx].copy()
                doc['score'] = float(scores[idx])
                doc['rank'] = rank
                results.append(doc)
            
            return results
        except Exception as e:
            print(f"[ERROR] BM25 검색 실패: {e}")
            return []


class HybridRetriever:
    """
    하이브리드 검색기
    
    BM25와 FAISS를 결합하여 검색합니다.
    전역 캐시를 사용하여 멀티턴 대화에서 재사용됩니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 검색 설정
                - bm25_corpus_path: BM25 코퍼스 경로
                - faiss_index_path: FAISS 인덱스 경로
                - faiss_meta_path: FAISS 메타데이터 경로
        
        Note:
            같은 설정에 대해서는 전역 캐시에서 재사용합니다.
            멀티턴 대화에서 매 턴마다 재생성하는 것을 방지합니다.
        """
        # 캐시 키 생성 (설정을 기반으로)
        bm25_path = config.get('bm25_corpus_path', '')
        faiss_path = config.get('faiss_index_path', '')
        faiss_meta = config.get('faiss_meta_path', '')
        rrf_k = config.get('rrf_k', 60)
        
        # 정규화된 경로 사용
        if bm25_path:
            bm25_path = os.path.abspath(bm25_path)
        if faiss_path:
            faiss_path = os.path.abspath(faiss_path)
        if faiss_meta:
            faiss_meta = os.path.abspath(faiss_meta)
        
        cache_key = f"{bm25_path}::{faiss_path}::{faiss_meta}::{rrf_k}"
        
        # 캐시에 있으면 재사용
        if cache_key in _HYBRID_RETRIEVER_CACHE:
            cached = _HYBRID_RETRIEVER_CACHE[cache_key]
            self.config = cached.config
            self.bm25_retriever = cached.bm25_retriever
            self.faiss_index = cached.faiss_index
            return
        
        # 새로 생성
        self.config = config
        
        # BM25 검색기 초기화 (전역 캐시 사용)
        self.bm25_retriever = BM25Retriever(bm25_path) if bm25_path else None
        
        # FAISS 인덱스 초기화 (전역 캐시 사용)
        self.faiss_index = FAISSIndex(faiss_path, faiss_meta) if faiss_path else None
        
        # 캐시에 저장
        _HYBRID_RETRIEVER_CACHE[cache_key] = self
    
    def search(self, query: str, query_vector: Optional[List[float]] = None, k: int = 10) -> List[Dict[str, Any]]:
        """
        하이브리드 검색
        
        Args:
            query: 쿼리 텍스트
            query_vector: 쿼리 벡터 (FAISS용, 없으면 None)
            k: 반환할 상위 k개
        
        Returns:
            검색 결과 리스트
        """
        results_list = []
        
        # BM25 검색
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.search(query, k=k)
            if bm25_results:
                results_list.append(bm25_results)
        
        # FAISS 검색
        if self.faiss_index and query_vector:
            faiss_results = self.faiss_index.search(query_vector, k=k)
            if faiss_results:
                results_list.append(faiss_results)
        
        # RRF 융합
        if len(results_list) > 1:
            fused_results = rrf_fusion(results_list, k=self.config.get('rrf_k', 60))
            return fused_results[:k]
        elif results_list:
            return results_list[0][:k]
        else:
            return []

