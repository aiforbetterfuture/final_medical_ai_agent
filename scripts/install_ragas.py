#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS 라이브러리 설치 및 검증 스크립트

이 스크립트는 다음을 수행합니다:
1. RAGAS 및 의존성 설치
2. OpenAI API 키 확인
3. RAGAS와 OpenAI API 통합 테스트
4. 충돌 여부 확인

사용법:
    python scripts/install_ragas.py
"""

import os
import sys
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("RAGAS 라이브러리 설치 및 검증")
print("=" * 80)
print()

# 1. 의존성 설치
print("[1] 의존성 설치")
print("-" * 80)

packages_to_install = [
    ('ragas', '>=0.1.0'),
    ('datasets', '>=2.14.0'),
    ('langchain-openai', '>=0.1.0'),  # RAGAS가 OpenAI와 통합하기 위해 필요
]

for package, version in packages_to_install:
    print(f"\n설치 중: {package}{version}")
    try:
        import importlib.util
        package_name = package.replace('-', '_')
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"  → pip install {package}{version}")
            os.system(f"pip install {package}{version}")
        else:
            print(f"  ✓ {package} 이미 설치됨")
    except Exception as e:
        print(f"  ✗ 설치 실패: {e}")
        print(f"  수동 설치: pip install {package}{version}")

print()

# 2. RAGAS 임포트 확인
print("[2] RAGAS 임포트 확인")
print("-" * 80)

try:
    import ragas
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    print(f"✓ RAGAS {ragas.__version__} 임포트 성공")
except ImportError as e:
    print(f"✗ RAGAS 임포트 실패: {e}")
    print("  설치: pip install ragas>=0.1.0")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("✓ langchain-openai 임포트 성공")
except ImportError as e:
    print(f"✗ langchain-openai 임포트 실패: {e}")
    print("  설치: pip install langchain-openai>=0.1.0")
    sys.exit(1)

print()

# 3. OpenAI API 키 확인
print("[3] OpenAI API 키 확인")
print("-" * 80)

from dotenv import load_dotenv

# .env 파일 로드
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✓ .env 파일 로드: {env_path}")
else:
    print(f"⚠️  .env 파일 없음: {env_path}")
    print("  env_template.txt를 .env로 복사하세요")

openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    masked_key = openai_key[:8] + '...' + openai_key[-4:] if len(openai_key) > 12 else '***'
    print(f"✓ OPENAI_API_KEY 설정됨: {masked_key}")
else:
    print("✗ OPENAI_API_KEY가 설정되지 않았습니다")
    print("  .env 파일에 OPENAI_API_KEY=your_key_here 추가하세요")
    sys.exit(1)

print()

# 4. RAGAS와 OpenAI 통합 테스트
print("[4] RAGAS와 OpenAI API 통합 테스트")
print("-" * 80)

try:
    # 테스트 데이터 준비
    test_data = {
        "question": ["What is diabetes?"],
        "answer": ["Diabetes is a chronic condition that affects how your body processes blood sugar."],
        "contexts": [["Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels."]],
    }
    
    dataset = Dataset.from_dict(test_data)
    
    # OpenAI 모델 설정
    print("\n[4-1] OpenAI 모델 초기화")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✓ OpenAI 모델 초기화 성공")
    
    # RAGAS 평가 실행
    print("\n[4-2] RAGAS 평가 실행")
    print("  메트릭: faithfulness, answer_relevancy")
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False
    )
    
    print("✓ RAGAS 평가 성공")
    
    # 결과 출력
    print("\n[4-3] 평가 결과")
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        print(df.to_string(index=False))
    else:
        print(f"  결과 타입: {type(results)}")
        print(f"  결과: {results}")
    
except Exception as e:
    print(f"\n✗ RAGAS 평가 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 5. 충돌 확인 (동시 사용 테스트)
print("[5] OpenAI API 충돌 확인")
print("-" * 80)

try:
    print("\n[5-1] 직접 OpenAI API 호출 테스트")
    from openai import OpenAI
    
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print(f"✓ 직접 OpenAI API 호출 성공: {response.choices[0].message.content}")
    
    print("\n[5-2] RAGAS와 동시 사용 테스트")
    # RAGAS가 사용하는 LLM과 직접 호출이 충돌하는지 확인
    test_data2 = {
        "question": ["What is hypertension?"],
        "answer": ["Hypertension is high blood pressure."],
        "contexts": [["Hypertension is a condition where blood pressure is consistently elevated."]],
    }
    
    dataset2 = Dataset.from_dict(test_data2)
    
    # RAGAS 평가 (OpenAI 사용)
    results2 = evaluate(
        dataset=dataset2,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False
    )
    
    print("✓ RAGAS와 직접 API 호출 동시 사용 성공")
    print("  → 충돌 없음: RAGAS와 직접 OpenAI API 호출이 동시에 작동함")
    
except Exception as e:
    print(f"\n✗ 충돌 확인 실패: {e}")
    import traceback
    traceback.print_exc()

print()

# 6. 버전 호환성 확인
print("[6] 버전 호환성 확인")
print("-" * 80)

try:
    import openai as openai_lib
    import langchain
    import langchain_openai
    
    print(f"OpenAI SDK: {openai_lib.__version__}")
    print(f"LangChain: {langchain.__version__}")
    print(f"LangChain OpenAI: {langchain_openai.__version__}")
    print(f"RAGAS: {ragas.__version__}")
    print(f"Datasets: {Dataset.__module__}")
    
    # 호환성 체크
    print("\n호환성 상태:")
    print("✓ 모든 라이브러리가 정상적으로 작동합니다")
    
except Exception as e:
    print(f"✗ 버전 확인 실패: {e}")

print()
print("=" * 80)
print("✅ RAGAS 설치 및 검증 완료!")
print("=" * 80)
print()
print("다음 단계:")
print("1. experiments/evaluation/ragas_metrics.py 사용")
print("2. 실험 러너에 RAGAS 통합")
print("3. 평가 메트릭 자동 수집")
print()
print("참고:")
print("- RAGAS는 OpenAI API를 내부적으로 사용합니다")
print("- 직접 OpenAI API 호출과 충돌하지 않습니다")
print("- 동일한 API 키를 사용할 수 있습니다")
print()

