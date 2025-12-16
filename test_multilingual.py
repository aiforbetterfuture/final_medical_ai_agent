# -*- coding: utf-8 -*-
"""
다국어 MedCAT2 파이프라인 테스트
"""

import os
import sys

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from extraction.multilingual_medcat import MultilingualMedCAT, _detect_language
from extraction.medcat2_adapter import MedCAT2Adapter, medcat2_extract_korean
from extraction.slot_extractor import SlotExtractor


def test_language_detection():
    """언어 감지 테스트"""
    print("=" * 60)
    print("테스트 1: 언어 감지")
    print("=" * 60)
    
    texts = [
        ("55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중", "ko"),
        ("55 year old male with hypertension and diabetes, taking metformin", "en"),
        ("I have 고혈압 and 당뇨", "mixed"),
    ]
    
    for text, expected in texts:
        detected = _detect_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"{status} [{detected}] {text[:50]}...")
    
    print()


def test_dictionary_translation():
    """사전 기반 번역 테스트"""
    print("=" * 60)
    print("테스트 2: 사전 기반 번역 (의료 용어)")
    print("=" * 60)
    
    from korean_translator import KoreanTranslator
    
    translator = KoreanTranslator(use_neural_translation=False)
    
    test_cases = [
        "고혈압과 당뇨가 있습니다",
        "메트포르민 복용 중입니다",
        "가슴이 답답하고 어지러워요",
        "55세 남성 환자입니다",
    ]
    
    for text in test_cases:
        translated = translator.translate_to_english(text)
        print(f"원본: {text}")
        print(f"번역: {translated}")
        print()


def test_multilingual_extraction():
    """다국어 엔티티 추출 테스트"""
    print("=" * 60)
    print("테스트 3: 다국어 MedCAT2 엔티티 추출")
    print("=" * 60)
    
    # 사전 기반만 사용 (신경망 번역 없이)
    medcat = MultilingualMedCAT(
        use_neural_translation=False,
        use_dict_translation=True
    )
    
    test_text = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중이며 가슴이 답답하고 어지러운 환자입니다"
    
    print(f"입력: {test_text}")
    print()
    
    result = medcat.extract_entities(test_text)
    
    # 메타데이터
    metadata = result.get("metadata", {})
    print(f"감지된 언어: {metadata.get('detected_language', 'unknown')}")
    print(f"번역 방법: {metadata.get('translation_method', 'none')}")
    print(f"번역된 텍스트: {metadata.get('translated_text', '')[:100]}")
    print()
    
    # 추출된 엔티티
    print("추출된 엔티티:")
    
    conditions = result.get("conditions", [])
    print(f"  Conditions ({len(conditions)}개):")
    for c in conditions[:5]:
        name = c.get("name", "")
        cui = c.get("cui", "")
        conf = c.get("confidence", 0)
        print(f"    - {name} (CUI: {cui}, conf: {conf:.2f})")
    
    symptoms = result.get("symptoms", [])
    print(f"  Symptoms ({len(symptoms)}개):")
    for s in symptoms[:5]:
        name = s.get("name", "")
        cui = s.get("cui", "")
        conf = s.get("confidence", 0)
        print(f"    - {name} (CUI: {cui}, conf: {conf:.2f})")
    
    medications = result.get("medications", [])
    print(f"  Medications ({len(medications)}개):")
    for m in medications[:5]:
        name = m.get("name", "")
        cui = m.get("cui", "")
        conf = m.get("confidence", 0)
        print(f"    - {name} (CUI: {cui}, conf: {conf:.2f})")
    
    vitals = result.get("vitals", [])
    print(f"  Vitals ({len(vitals)}개):")
    for v in vitals[:5]:
        name = v.get("name", "")
        value = v.get("value", "")
        unit = v.get("unit", "")
        print(f"    - {name}: {value} {unit}")
    
    print()


def test_slot_extractor():
    """슬롯 추출기 테스트"""
    print("=" * 60)
    print("테스트 4: 슬롯 추출기 (다국어 통합)")
    print("=" * 60)
    
    extractor = SlotExtractor(
        use_medcat2=True,
        use_multilingual=True,
        use_neural_translation=False,  # 사전 기반만
        use_dict_translation=True
    )
    
    test_text = "55세 남성, 고혈압과 당뇨가 있고 메트포르민 복용 중이며 혈압 140/90 mmHg, A1c 7.5% 입니다"
    
    print(f"입력: {test_text}")
    print()
    
    slots = extractor.extract(test_text)
    
    # 인구통계학적 정보
    demo = slots.get("demographics", {})
    print(f"Demographics:")
    print(f"  나이: {demo.get('age', 'N/A')}")
    print(f"  성별: {demo.get('gender', 'N/A')}")
    
    # 메타데이터
    metadata = slots.get("metadata", {})
    print(f"\n메타데이터:")
    print(f"  감지된 언어: {metadata.get('detected_language', 'unknown')}")
    print(f"  번역된 텍스트: {metadata.get('translated_text', '')[:80]}...")
    
    # 슬롯
    print(f"\n추출된 슬롯:")
    print(f"  Conditions: {[c.get('name') for c in slots.get('conditions', [])]}")
    print(f"  Symptoms: {[s.get('name') for s in slots.get('symptoms', [])]}")
    print(f"  Medications: {[m.get('name') for m in slots.get('medications', [])]}")
    print(f"  Vitals: {[(v.get('name'), v.get('value'), v.get('unit')) for v in slots.get('vitals', [])]}")
    print(f"  Labs: {[(l.get('name'), l.get('value'), l.get('unit')) for l in slots.get('labs', [])]}")
    
    print()


def test_direct_english():
    """영어 직접 입력 테스트"""
    print("=" * 60)
    print("테스트 5: 영어 직접 입력")
    print("=" * 60)
    
    adapter = MedCAT2Adapter()
    
    test_text = "55 year old male with hypertension and diabetes mellitus, taking metformin, experiencing chest tightness and dizziness"
    
    print(f"입력: {test_text}")
    print()
    
    result = adapter.extract_entities(test_text)
    
    print("추출된 엔티티:")
    print(f"  Conditions: {[c.get('name') for c in result.get('conditions', [])]}")
    print(f"  Symptoms: {[s.get('name') for s in result.get('symptoms', [])]}")
    print(f"  Medications: {[m.get('name') for m in result.get('medications', [])]}")
    
    print()


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 60)
    print("다국어 MedCAT2 파이프라인 테스트")
    print("=" * 60 + "\n")
    
    # MedCAT2 모델 경로 확인
    model_path = os.getenv("MEDCAT2_MODEL_PATH")
    if model_path:
        print(f"MedCAT2 모델 경로: {model_path}")
    else:
        print("⚠️ MEDCAT2_MODEL_PATH 환경 변수가 설정되지 않았습니다.")
    print()
    
    # 테스트 실행
    test_language_detection()
    test_dictionary_translation()
    test_multilingual_extraction()
    test_slot_extractor()
    test_direct_english()
    
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

