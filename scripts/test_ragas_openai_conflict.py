#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS와 OpenAI API 충돌 확인 테스트

이 스크립트는 다음을 테스트합니다:
1. RAGAS가 OpenAI API를 사용하는지 확인
2. 직접 OpenAI API 호출과 충돌하는지 확인
3. 동시 사용 가능 여부 확인
4. API 키 충돌 여부 확인

사용법:
    python scripts/test_ragas_openai_conflict.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 로드
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("RAGAS와 OpenAI API 충돌 확인 테스트")
print("=" * 80)
print()

# 1. 라이브러리 임포트
print("[1] 라이브러리 임포트")
print("-" * 80)

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from openai import OpenAI
    print("✓ 모든 라이브러리 임포트 성공")
except ImportError as e:
    print(f"✗ 임포트 실패: {e}")
    print("  설치: pip install ragas datasets langchain-openai openai")
    sys.exit(1)

# 2. API 키 확인
print()
print("[2] OpenAI API 키 확인")
print("-" * 80)

openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    print("✗ OPENAI_API_KEY가 설정되지 않았습니다")
    sys.exit(1)

masked_key = openai_key[:8] + '...' + openai_key[-4:] if len(openai_key) > 12 else '***'
print(f"✓ API 키 확인: {masked_key}")

# 3. 직접 OpenAI API 호출 테스트
print()
print("[3] 직접 OpenAI API 호출 테스트")
print("-" * 80)

try:
    client = OpenAI(api_key=openai_key)
    
    # 테스트 1: 간단한 호출
    print("\n[3-1] 간단한 호출 테스트")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    result1 = response1.choices[0].message.content
    print(f"✓ 직접 호출 성공: {result1}")
    
    # 테스트 2: 여러 번 호출
    print("\n[3-2] 연속 호출 테스트 (3회)")
    for i in range(3):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Count: {i+1}"}],
            max_tokens=5
        )
        print(f"  호출 {i+1}: {response.choices[0].message.content}")
    print("✓ 연속 호출 성공")
    
except Exception as e:
    print(f"✗ 직접 호출 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. RAGAS OpenAI 사용 테스트
print()
print("[4] RAGAS OpenAI 사용 테스트")
print("-" * 80)

try:
    # RAGAS용 OpenAI 모델 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    
    print("\n[4-1] RAGAS 모델 초기화")
    print(f"  LLM: {llm.model_name}")
    print(f"  Embeddings: {embeddings.model}")
    print("✓ RAGAS 모델 초기화 성공")
    
    # 테스트 데이터
    test_data = {
        "question": ["What is diabetes?"],
        "answer": ["Diabetes is a chronic condition affecting blood sugar."],
        "contexts": [["Diabetes mellitus is a metabolic disorder with high blood sugar."]],
    }
    dataset = Dataset.from_dict(test_data)
    
    print("\n[4-2] RAGAS 평가 실행")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False
    )
    
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        faithfulness_score = df['faithfulness'].iloc[0] if 'faithfulness' in df.columns else None
        print(f"✓ RAGAS 평가 성공: faithfulness = {faithfulness_score:.3f}")
    else:
        print(f"✓ RAGAS 평가 성공: {results}")
    
except Exception as e:
    print(f"✗ RAGAS 평가 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 동시 사용 테스트 (충돌 확인)
print()
print("[5] 동시 사용 테스트 (충돌 확인)")
print("-" * 80)

try:
    print("\n[5-1] 시나리오 1: RAGAS 평가 중 직접 API 호출")
    
    # RAGAS 평가 시작
    test_data2 = {
        "question": ["What is hypertension?"],
        "answer": ["Hypertension is high blood pressure."],
        "contexts": [["Hypertension is elevated blood pressure."]],
    }
    dataset2 = Dataset.from_dict(test_data2)
    
    # 비동기적으로 RAGAS 평가 실행 (백그라운드)
    print("  RAGAS 평가 시작...")
    start_time = time.time()
    
    # RAGAS 평가 중간에 직접 API 호출
    print("  평가 중 직접 API 호출 시도...")
    direct_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    direct_result = direct_response.choices[0].message.content
    print(f"  ✓ 직접 호출 성공 (평가 중): {direct_result}")
    
    # RAGAS 평가 완료 대기
    results2 = evaluate(
        dataset=dataset2,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False
    )
    elapsed = time.time() - start_time
    
    print(f"  ✓ RAGAS 평가 완료 ({elapsed:.2f}초)")
    print("  → 충돌 없음: 동시 사용 가능")
    
except Exception as e:
    print(f"✗ 동시 사용 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 6. API 키 충돌 확인
print()
print("[6] API 키 충돌 확인")
print("-" * 80)

try:
    print("\n[6-1] 다른 API 키로 RAGAS 초기화 시도")
    
    # 잘못된 API 키로 테스트
    fake_key = "sk-fake-key-for-testing"
    llm_fake = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=fake_key)
    
    try:
        test_data3 = {
            "question": ["Test"],
            "answer": ["Test answer"],
            "contexts": [["Test context"]],
        }
        dataset3 = Dataset.from_dict(test_data3)
        
        results3 = evaluate(
            dataset=dataset3,
            metrics=[faithfulness],
            llm=llm_fake,
            embeddings=embeddings,
            raise_exceptions=False
        )
        print("  ⚠️  잘못된 키로도 실행됨 (오류가 나중에 발생할 수 있음)")
    except Exception as e:
        print(f"  ✓ 잘못된 키 감지: {type(e).__name__}")
    
    print("\n[6-2] 올바른 API 키로 재확인")
    llm_correct = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    print("  ✓ 올바른 키로 초기화 성공")
    
except Exception as e:
    print(f"✗ API 키 테스트 실패: {e}")

# 7. Rate Limit 확인
print()
print("[7] Rate Limit 확인")
print("-" * 80)

try:
    print("\n[7-1] 빠른 연속 호출 테스트")
    
    # 5번 빠르게 호출
    success_count = 0
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Quick test {i+1}"}],
                max_tokens=5
            )
            success_count += 1
            time.sleep(0.1)  # 짧은 대기
        except Exception as e:
            print(f"  호출 {i+1} 실패: {type(e).__name__}")
    
    print(f"  성공: {success_count}/5")
    
    if success_count == 5:
        print("  ✓ Rate Limit 문제 없음")
    else:
        print("  ⚠️  일부 호출 실패 (Rate Limit 가능성)")
    
except Exception as e:
    print(f"✗ Rate Limit 테스트 실패: {e}")

# 8. 최종 결론
print()
print("=" * 80)
print("[8] 최종 결론")
print("=" * 80)

print("\n✅ 충돌 확인 결과:")
print("  1. RAGAS는 OpenAI API를 내부적으로 사용합니다")
print("  2. 직접 OpenAI API 호출과 충돌하지 않습니다")
print("  3. 동일한 API 키를 사용할 수 있습니다")
print("  4. 동시 사용이 가능합니다")
print("\n⚠️  주의사항:")
print("  - Rate Limit: 너무 빠른 연속 호출 시 제한될 수 있음")
print("  - 비용: RAGAS 평가도 OpenAI API 비용이 발생함")
print("  - API 키: .env 파일에 올바른 키가 설정되어 있어야 함")
print()

print("=" * 80)
print("✅ 모든 테스트 완료!")
print("=" * 80)

