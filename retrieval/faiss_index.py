"""
FAISS 인덱스 관리
- 인덱스 로드/저장
- 벡터 검색
- 전역 싱글톤 캐시로 멀티턴 대화에서 재사용
"""

import os
from typing import List, Dict, Any, Optional
import json

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None
    np = None


# 전역 캐시: 같은 경로에 대해서는 한 번만 로드
_FAISS_INDEX_CACHE: Dict[str, 'FAISSIndex'] = {}


class FAISSIndex:
    """
    FAISS 인덱스 관리 클래스
    """
    
    def __init__(self, index_path: str, meta_path: Optional[str] = None):
        """
        Args:
            index_path: FAISS 인덱스 파일 경로
            meta_path: 메타데이터 파일 경로 (선택)
        
        Note:
            같은 경로에 대해서는 전역 캐시에서 재사용합니다.
            멀티턴 대화에서 매 턴마다 재로딩하는 것을 방지합니다.
        """
        # 정규화된 경로를 키로 사용
        index_path = os.path.abspath(index_path)
        cache_key = index_path
        
        # 캐시에 있으면 재사용
        if cache_key in _FAISS_INDEX_CACHE:
            cached = _FAISS_INDEX_CACHE[cache_key]
            self.index_path = cached.index_path
            self.meta_path = cached.meta_path
            self.index = cached.index
            self.metadata = cached.metadata
            return
        
        # 새로 생성
        self.index_path = index_path
        self.meta_path = meta_path or index_path.replace('.faiss', '.meta.jsonl').replace('.index', '.meta.jsonl')
        self.meta_path = os.path.abspath(self.meta_path)
        self.index = None
        self.metadata = []
        
        if os.path.exists(index_path):
            self.load()
        
        # 캐시에 저장
        _FAISS_INDEX_CACHE[cache_key] = self
    
    def load(self):
        """인덱스 로드 (캐시에서 재사용 중이면 로드하지 않음)"""
        # 이미 로드되어 있으면 스킵
        if self.index is not None:
            return
            
        if not HAS_FAISS:
            print("[WARNING] faiss가 설치되지 않았습니다.")
            return
        
        if not os.path.exists(self.index_path):
            print(f"[WARNING] 인덱스 파일을 찾을 수 없습니다: {self.index_path}")
            return
        
        try:
            self.index = faiss.read_index(self.index_path)
            print(f"[FAISS] 인덱스 로드 완료: {self.index_path}")
            
            # 메타데이터 로드
            if os.path.exists(self.meta_path):
                self._load_metadata()
        except Exception as e:
            print(f"[ERROR] 인덱스 로드 실패: {e}")
            self.index = None
    
    def _load_metadata(self):
        """메타데이터 로드"""
        try:
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = [json.loads(line) for line in f if line.strip()]
            print(f"[FAISS] 메타데이터 로드 완료: {len(self.metadata)}개 문서")
        except Exception as e:
            print(f"[WARNING] 메타데이터 로드 실패: {e}")
            self.metadata = []
    
    def search(self, query_vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """
        벡터 검색
        
        Args:
            query_vector: 쿼리 벡터
            k: 반환할 상위 k개
        
        Returns:
            검색 결과 리스트
        """
        if not HAS_FAISS or self.index is None:
            return []
        
        if query_vector is None:
            print("[ERROR] FAISS 검색 실패: query_vector가 None입니다 (임베딩 생성 실패)")
            return []
        
        try:
            # 벡터를 numpy 배열로 변환
            query_vec = np.array([query_vector], dtype=np.float32)
            
            # 차원 검증 (이제 3072차원으로 통일되어야 함)
            query_dim = query_vec.shape[1]
            index_dim = self.index.d
            
            if query_dim != index_dim:
                print(f"[ERROR] FAISS 검색 실패: 차원 불일치 (쿼리: {query_dim}, 인덱스: {index_dim})")
                print(f"[INFO] 쿼리 벡터 길이: {len(query_vector)}, 인덱스 차원: {index_dim}")
                print(f"[INFO] 인덱스 파일: {self.index_path}")
                print(f"[ERROR] 쿼리와 인덱스의 차원이 일치하지 않습니다.")
                print(f"[INFO] 예상 차원: 3072 (text-embedding-3-large)")
                print(f"[INFO] 임베딩 모델 설정을 확인하세요: config/corpus_config.yaml")
                return []
            
            # L2 정규화 (Inner Product를 위한 코사인 유사도)
            if HAS_FAISS and faiss is not None:
                faiss.normalize_L2(query_vec)
            
            # 검색 실행
            scores, indices = self.index.search(query_vec, k)
            
            # 결과 구성
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                doc = self.metadata[idx].copy()
                doc['score'] = float(score)
                doc['rank'] = i + 1
                results.append(doc)
            
            return results
        except Exception as e:
            print(f"[ERROR] FAISS 검색 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def build(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], metric: str = 'ip'):
        """
        인덱스 구축
        
        Args:
            vectors: 벡터 리스트
            metadata: 메타데이터 리스트
            metric: 거리 메트릭 ('ip' 또는 'l2')
        """
        if not HAS_FAISS:
            print("[WARNING] faiss가 설치되지 않았습니다.")
            return
        
        try:
            vecs = np.array(vectors, dtype=np.float32)
            dim = vecs.shape[1]
            
            # 인덱스 생성
            if metric == 'ip':
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
            
            # 벡터 정규화 (IP의 경우)
            if metric == 'ip':
                faiss.normalize_L2(vecs)
            
            index.add(vecs)
            
            # 저장
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(index, self.index_path)
            
            # 메타데이터 저장
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                for meta in metadata:
                    f.write(json.dumps(meta, ensure_ascii=False) + '\n')
            
            self.index = index
            self.metadata = metadata
            
            print(f"[FAISS] 인덱스 구축 완료: {len(vectors)}개 벡터")
        except Exception as e:
            print(f"[ERROR] 인덱스 구축 실패: {e}")

