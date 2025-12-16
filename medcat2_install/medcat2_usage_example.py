"""
MEDCAT2 사용 예제

이 스크립트는 MEDCAT2를 사용하여 의료 텍스트에서 엔티티를 추출하는 방법을 보여줍니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

from nlp.medcat2_adapter import MedCAT2Adapter


def example_basic_usage():
    """기본 사용 예제"""
    print("=" * 60)
    print("MEDCAT2 기본 사용 예제")
    print("=" * 60)
    
    try:
        # 어댑터 초기화
        adapter = MedCAT2Adapter()
        
        # 테스트 텍스트
        text = "당뇨병 환자가 혈압 140/90을 보이고 있습니다. HbA1c는 8.1%입니다."
        
        print(f"\n입력 텍스트: {text}\n")
        
        # 엔티티 추출
        entities = adapter.extract_entities(text)
        
        print("추출된 엔티티:")
        print(f"  - 질환 (Conditions): {entities['conditions']}")
        print(f"  - 증상 (Symptoms): {entities['symptoms']}")
        print(f"  - 활력징후 (Vitals): {entities['vitals']}")
        print(f"  - 검사결과 (Labs): {entities['labs']}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n해결 방법:")
        print("1. .env 파일에 MEDCAT2_MODEL_PATH를 설정하세요")
        print("2. MEDCAT2 모델 팩을 다운로드하여 지정한 경로에 저장하세요")
        print("3. 'pip install medcat>=2.0'을 실행하여 패키지를 설치하세요")


def example_multiple_texts():
    """여러 텍스트 처리 예제"""
    print("\n" + "=" * 60)
    print("여러 텍스트 처리 예제")
    print("=" * 60)
    
    try:
        adapter = MedCAT2Adapter()
        
        texts = [
            "고혈압 환자가 흉통을 호소합니다.",
            "당뇨병과 고지혈증이 있는 65세 남성 환자입니다.",
            "신장 기능 저하로 eGFR이 45 ml/min/1.73m²입니다."
        ]
        
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}] 입력: {text}")
            entities = adapter.extract_entities(text)
            
            if entities['conditions']:
                print(f"    질환: {[c['name'] for c in entities['conditions']]}")
            if entities['symptoms']:
                print(f"    증상: {[s['name'] for s in entities['symptoms']]}")
            if entities['labs']:
                print(f"    검사: {[l['name'] for l in entities['labs']]}")
                
    except Exception as e:
        print(f"오류 발생: {e}")


def example_with_slots_extractor():
    """SlotsExtractor와 함께 사용하는 예제"""
    print("\n" + "=" * 60)
    print("SlotsExtractor 통합 예제")
    print("=" * 60)
    
    try:
        from agent.nodes.slots_extract import SlotsExtractor
        
        # MEDCAT2를 사용하는 SlotsExtractor 생성
        cfg_paths = {}  # 실제 사용 시 적절한 설정 경로 제공
        extractor = SlotsExtractor(cfg_paths, use_medcat2=True)
        
        text = "당뇨병 환자가 혈압 140/90을 보이고 있습니다."
        result = extractor.extract(text)
        
        print(f"\n입력: {text}")
        print(f"\n추출된 슬롯:")
        print(f"  - Raw Slots: {result['raw_slots']}")
        print(f"  - Pending Changes: {len(result['pending_changes'])}개")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def example_training():
    """비지도 학습 예제 (선택적)"""
    print("\n" + "=" * 60)
    print("비지도 학습 예제 (참고용)")
    print("=" * 60)
    
    print("\n비지도 학습을 수행하려면:")
    print("""
    from nlp.medcat2_adapter import MedCAT2Adapter
    
    adapter = MedCAT2Adapter()
    
    # 학습 데이터 준비
    data_iterator = [
        "Patient has diabetes and hypertension.",
        "Kidney failure with elevated creatinine.",
        # ... 더 많은 문서들
    ]
    
    # 학습 수행
    adapter.train(data_iterator)
    
    # 학습된 모델 저장
    adapter.save_model('path/to/saved_model_pack.zip')
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MEDCAT2 사용 예제")
    print("=" * 60)
    
    # 기본 사용 예제
    example_basic_usage()
    
    # 여러 텍스트 처리 예제
    example_multiple_texts()
    
    # SlotsExtractor 통합 예제
    example_with_slots_extractor()
    
    # 학습 예제 (참고용)
    example_training()
    
    print("\n" + "=" * 60)
    print("예제 실행 완료")
    print("=" * 60)



















