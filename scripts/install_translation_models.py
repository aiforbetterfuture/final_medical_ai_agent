#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helsinki-NLP 번역 모델 설치 스크립트

이 스크립트는 다음 모델을 다운로드하고 캐시합니다:
- Helsinki-NLP/opus-mt-ko-en (한영 번역)
- Helsinki-NLP/opus-mt-en-ko (영한 번역, 대안 모델)

사용법:
    python scripts/install_translation_models.py
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("Helsinki-NLP 번역 모델 설치")
print("=" * 80)
print()

# 1. transformers 설치 확인
try:
    import transformers
    print(f"✓ transformers {transformers.__version__} 설치됨")
except ImportError:
    print("✗ transformers 패키지가 설치되지 않았습니다")
    print("  설치 중...")
    os.system("pip install transformers torch")
    import transformers
    print(f"✓ transformers {transformers.__version__} 설치 완료")

print()

# 2. 모델 다운로드
from transformers import pipeline

models_to_install = [
    {
        'name': 'Helsinki-NLP/opus-mt-ko-en',
        'description': '한영 번역 모델',
        'direction': 'ko2en'
    },
    {
        'name': 'Helsinki-NLP/opus-mt-en-ko',
        'description': '영한 번역 모델 (대안)',
        'direction': 'en2ko',
        'optional': True  # 이 모델이 없을 수 있음
    }
]

print("[1] 모델 다운로드 시작")
print("-" * 80)

for model_info in models_to_install:
    model_name = model_info['name']
    description = model_info['description']
    direction = model_info['direction']
    is_optional = model_info.get('optional', False)
    
    print(f"\n[{direction.upper()}] {model_name}")
    print(f"  설명: {description}")
    
    try:
        print(f"  다운로드 중... (처음 실행 시 시간이 걸릴 수 있습니다)")
        
        # 모델 다운로드 (자동으로 Hugging Face 캐시에 저장됨)
        pipe = pipeline(
            "translation",
            model=model_name,
            device=-1  # CPU 사용 (GPU는 자동 감지)
        )
        
        print(f"  ✓ {model_name} 다운로드 완료")
        
        # 간단한 테스트
        if direction == 'ko2en':
            test_text = "안녕하세요"
            result = pipe(test_text)
            print(f"  테스트: '{test_text}' → '{result[0]['translation_text']}'")
        elif direction == 'en2ko':
            test_text = "Hello"
            result = pipe(test_text)
            print(f"  테스트: '{test_text}' → '{result[0]['translation_text']}'")
        
    except Exception as e:
        if is_optional:
            print(f"  ⚠️  {model_name} 다운로드 실패 (선택적 모델)")
            print(f"     오류: {str(e)}")
            print(f"     참고: 이 모델은 Hugging Face에 없을 수 있습니다")
        else:
            print(f"  ✗ {model_name} 다운로드 실패")
            print(f"     오류: {str(e)}")
            print(f"     해결: 인터넷 연결 확인 또는 수동 설치 필요")
            sys.exit(1)

print()
print("=" * 80)
print("[2] 모델 캐시 위치 확인")
print("-" * 80)

# Hugging Face 캐시 위치 확인
from transformers import file_utils

cache_dir = file_utils.default_cache_path
print(f"모델 캐시 위치: {cache_dir}")

# 다운로드된 모델 확인
if cache_dir.exists():
    model_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    print(f"\n다운로드된 모델 수: {len(model_dirs)}")
    
    # Helsinki-NLP 모델만 필터링
    helsinki_models = [d for d in model_dirs if 'opus-mt' in d.name.lower()]
    if helsinki_models:
        print("\nHelsinki-NLP 모델:")
        for model_dir in helsinki_models:
            print(f"  - {model_dir.name}")

print()
print("=" * 80)
print("[3] 번역 테스트")
print("-" * 80)

try:
    from extraction.neural_translator import NeuralTranslator
    
    translator = NeuralTranslator(lazy_load=False)
    
    # 한영 번역 테스트
    print("\n[한영 번역 테스트]")
    test_ko = "환자는 당뇨병과 고혈압이 있습니다"
    result_ko2en = translator.translate_ko2en(test_ko)
    print(f"  입력: {test_ko}")
    print(f"  출력: {result_ko2en}")
    
    # 영한 번역 테스트 (모델이 있는 경우)
    if translator.EN2KO_MODEL:
        print("\n[영한 번역 테스트]")
        test_en = "The patient has diabetes and hypertension"
        result_en2ko = translator.translate_en2ko(test_en)
        print(f"  입력: {test_en}")
        print(f"  출력: {result_en2ko}")
    else:
        print("\n[영한 번역]")
        print("  ⚠️  영한 번역 모델이 설정되지 않았습니다")
        print("     (Helsinki-NLP/opus-mt-en-ko는 Hugging Face에 없을 수 있음)")
    
except Exception as e:
    print(f"\n✗ 번역 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✅ 설치 완료!")
print("=" * 80)
print()
print("다음 단계:")
print("1. extraction/neural_translator.py가 정상 작동하는지 확인")
print("2. test_multilingual.py 실행하여 전체 시스템 테스트")
print()
print("모델은 Hugging Face 캐시에 저장되었습니다.")
print(f"캐시 위치: {cache_dir}")
print()

