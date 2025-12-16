"""
RRF (Reciprocal Rank Fusion) 구현
"""

from typing import List, Dict, Any


def rrf_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion) 융합
    
    Args:
        results_list: 여러 검색 결과 리스트
        k: RRF 상수 (기본값: 60)
    
    Returns:
        융합된 결과 리스트
    """
    if not results_list:
        return []
    
    # 문서별 RRF 점수 계산
    doc_scores = {}
    
    for rank_list in results_list:
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = doc.get('text', '')  # 텍스트를 ID로 사용
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'score': 0.0
                }
            # RRF 점수: 1 / (rank + k)
            doc_scores[doc_id]['score'] += 1.0 / (rank + k)
    
    # 점수 기준 정렬
    fused_results = sorted(
        doc_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    
    # 원본 문서 반환 (점수 포함)
    return [
        {
            **result['doc'],
            'rrf_score': result['score']
        }
        for result in fused_results
    ]


