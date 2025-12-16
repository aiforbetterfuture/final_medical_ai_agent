#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
번역 모델 테스트 스크립트

Helsinki-NLP/opus-mt-ko-en 모델이 정상 작동하는지 테스트합니다.
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
print("Helsinki-NLP 번역 모델 테스트")
print("=" * 80)
print()

# 1. transformers 확인
try:
    import transformers
    print(f"[1] transformers {transformers.__version__} 확인 ✓")
except ImportError:
    print("[1] ✗ transformers 패키지가 설치되지 않았습니다")
    print("     설치: pip install transformers torch")
    sys.exit(1)

# 2. 번역기 임포트
try:
    from extraction.neural_translator import NeuralTranslator, neural_translate_ko2en
    print("[2] NeuralTranslator 임포트 성공 ✓")
except ImportError as e:
    print(f"[2] ✗ NeuralTranslator 임포트 실패: {e}")
    sys.exit(1)

# 3. 번역기 초기화
print()
print("[3] 번역기 초기화")
print("-" * 80)

try:
    translator = NeuralTranslator(lazy_load=False)  # 즉시 로드
    print(f"✓ 번역기 초기화 완료")
    print(f"  - Device: {translator.device}")
    print(f"  - 한영 모델: {translator.KO2EN_MODEL}")
    print(f"  - 영한 모델: {translator.EN2KO_MODEL or '설정되지 않음'}")
except Exception as e:
    print(f"✗ 번역기 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 한영 번역 테스트
print()
print("[4] 한영 번역 테스트")
print("-" * 80)

test_cases_ko2en = [
    "안녕하세요",
    "환자는 당뇨병이 있습니다",
    "65세 남성 환자로 고혈압과 당뇨병이 있고 메트포르민을 복용 중입니다",
    "가슴 통증과 호흡곤란이 있습니다",
    "혈압은 140/90 mmHg입니다"
]

for i, text_ko in enumerate(test_cases_ko2en, 1):
    try:
        result = translator.translate_ko2en(text_ko)
        print(f"\n[{i}] 입력: {text_ko}")
        print(f"    출력: {result}")
        
        if result == text_ko:
            print("    ⚠️  번역되지 않음 (원본 반환)")
        else:
            print("    ✓ 번역 성공")
    except Exception as e:
        print(f"\n[{i}] ✗ 번역 실패: {e}")

# 5. 배치 번역 테스트
print()
print("[5] 배치 번역 테스트")
print("-" * 80)

try:
    batch_texts = [
        "당뇨병",
        "고혈압",
        "메트포르민"
    ]
    
    results = translator.batch_translate_ko2en(batch_texts)
    
    print(f"입력: {batch_texts}")
    print(f"출력: {results}")
    
    if all(r != orig for r, orig in zip(results, batch_texts)):
        print("✓ 배치 번역 성공")
    else:
        print("⚠️  일부 번역 실패")
        
except Exception as e:
    print(f"✗ 배치 번역 실패: {e}")

# 6. 성능 테스트
print()
print("[6] 성능 테스트")
print("-" * 80)

import time

test_text = "환자는 당뇨병과 고혈압이 있고 메트포르민을 복용 중입니다"

# 첫 번째 번역 (모델 로드 시간 포함)
start = time.time()
result1 = translator.translate_ko2en(test_text)
time1 = time.time() - start

# 두 번째 번역 (캐시된 모델 사용)
start = time.time()
result2 = translator.translate_ko2en(test_text)
time2 = time.time() - start

print(f"첫 번째 번역 (모델 로드 포함): {time1:.2f}초")
print(f"두 번째 번역 (캐시 사용): {time2:.2f}초")
print(f"속도 향상: {time1/time2:.1f}배")

# 7. 편의 함수 테스트
print()
print("[7] 편의 함수 테스트")
print("-" * 80)

try:
    result = neural_translate_ko2en("안녕하세요")
    print(f"입력: '안녕하세요'")
    print(f"출력: '{result}'")
    print("✓ 편의 함수 작동 확인")
except Exception as e:
    print(f"✗ 편의 함수 실패: {e}")

print()
print("=" * 80)
print("✅ 모든 테스트 완료!")
print("=" * 80)
print()
print("다음 단계:")
print("1. MedCAT과 통합 테스트: python test_multilingual.py")
print("2. 실제 사용: extraction/medcat2_adapter.py에서 자동 사용됨")
print()

