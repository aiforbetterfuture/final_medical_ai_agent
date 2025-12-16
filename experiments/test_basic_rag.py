"""
Basic RAG 테스트 스크립트

목적: 새 스캐폴드에서 Basic RAG가 정상 작동하는지 확인
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.graph import run_agent
from config.ablation_config import get_ablation_profile

# 테스트 쿼리
TEST_QUERIES = [
    "당뇨병이란 무엇인가요?",
    "메트포르민의 부작용은 무엇인가요?",
    "고혈압 환자의 식이요법은?",
]

print("="*80)
print("Basic RAG 테스트")
print("="*80)

# Basic RAG 설정 (ablation_config의 baseline 프로파일 사용)
basic_config = get_ablation_profile("baseline")
basic_config['refine_strategy'] = 'basic_rag'  # Basic RAG 전략 명시

print("\n[설정]")
print(f"  - refine_strategy: {basic_config.get('refine_strategy', 'N/A')}")
print(f"  - self_refine_enabled: {basic_config.get('self_refine_enabled', False)}")
print(f"  - quality_check_enabled: {basic_config.get('quality_check_enabled', False)}")
print("="*80)

# 각 쿼리 테스트
for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n[{i}/{len(TEST_QUERIES)}] 쿼리: {query}")
    print("-"*80)
    
    try:
        result = run_agent(
            user_text=query,
            mode='ai_agent',
            feature_overrides=basic_config,
            return_state=True
        )
        
        print(f"  [✓] 성공")
        print(f"    - 전략: {result.get('refine_strategy', 'N/A')}")
        print(f"    - 품질 점수: {result.get('quality_score', 0.0):.3f}")
        print(f"    - 반복 횟수: {result.get('iteration_count', 0)}")
        print(f"    - 검색 문서 수: {len(result.get('retrieved_docs', []))}")
        print(f"    - 답변 길이: {len(result.get('answer', ''))}자")
        print(f"    - 답변 미리보기: {result.get('answer', '')[:100]}...")
        
    except Exception as e:
        print(f"  [✗] 실패: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("테스트 완료")
print("="*80)

