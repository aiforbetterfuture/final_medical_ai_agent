#!/usr/bin/env python3
"""
MEDCAT2 Supervised Training 스크립트 (공식 튜토리얼 방식)

튜토리얼 3번: MedCATtrainer에서 export한 JSON을 이용한 지도 학습

공식 튜토리얼 방식:
    1. 비지도 학습 모델 로드
    2. 새로운 개념 추가 (CDBMaker, 선택사항)
    3. 학습 전 테스트 (선택사항)
    4. MedCATtrainer export JSON 로드
    5. cat.trainer.train_supervised_raw(data, use_filters=True) 실행
    6. 학습 후 테스트 (선택사항)
    7. 모델 팩 저장

사용법:
    # 기본 사용법
    python scripts/medcat2_train_supervised.py \
        --model-pack models/medcat2/medcat2_unsupervised_trained \
        --train-json data/medcattrainer_export/sample_train.json \
        --output-dir models/medcat2 \
        --pack-name medcat2_supervised_trained
    
    # 튜토리얼 방식 (use_filters=True)
    python scripts/medcat2_train_supervised.py \
        --model-pack models/medcat2/medcat2_unsupervised_trained \
        --train-json data/medcattrainer_export/sample_train.json \
        --use-filters
"""

import argparse
import os
import sys
import json
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from medcat.cat import CAT
    from medcat.model_creation.cdb_maker import CDBMaker
    import pandas as pd
except ImportError:
    print("ERROR: medcat 패키지가 설치되지 않았습니다.")
    print("설치: pip install medcat>=2.0")
    sys.exit(1)


def load_train_json(json_path: str) -> dict:
    """
    MedCATtrainer export JSON 로드
    
    형식:
        {
            "projects": [...],
            "documents": [...],
            "annotations": [...]
        }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="MEDCAT2 Supervised Training (튜토리얼 방식)"
    )
    parser.add_argument(
        "--model-pack",
        type=str,
        required=True,
        help="모델 팩 경로 (.zip 파일)"
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="MedCATtrainer export JSON 파일 경로"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/medcat2",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--pack-name",
        type=str,
        default="medcat2_umls_symptom_disease_supervised",
        help="저장할 모델 팩 이름"
    )
    parser.add_argument(
        "--use-filters",
        action="store_true",
        default=True,
        help="필터 사용 (공식 튜토리얼 방식, 기본값: True)"
    )
    parser.add_argument(
        "--add-concepts",
        type=str,
        help="새로운 개념 추가 (CSV 형식: name,cui 또는 JSON 파일 경로)"
    )
    parser.add_argument(
        "--test-before-training",
        action="store_true",
        help="학습 전 테스트 실행"
    )
    parser.add_argument(
        "--test-after-training",
        action="store_true",
        default=True,
        help="학습 후 테스트 실행 (기본값: True)"
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        help="테스트 케이스 JSON 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 모델 팩 로드
    model_pack_path = Path(args.model_pack)
    if not model_pack_path.exists():
        raise FileNotFoundError(f"모델 팩을 찾을 수 없습니다: {model_pack_path}")
    
    print(f"[INFO] 모델 팩 로드 (공식 튜토리얼 방식): {model_pack_path}")
    cat = CAT.load_model_pack(str(model_pack_path))
    print("[SUCCESS] 모델 팩 로드 완료")
    
    # 새로운 개념 추가 (선택사항)
    if args.add_concepts:
        print(f"\n[INFO] 새로운 개념 추가: {args.add_concepts}")
        try:
            cdb_maker = CDBMaker(cat.config, cat.cdb)
            
            # CSV 파일 또는 JSON 파일 처리
            if args.add_concepts.endswith('.csv'):
                df = pd.read_csv(args.add_concepts)
            elif args.add_concepts.endswith('.json'):
                concepts_data = json.load(open(args.add_concepts, 'r', encoding='utf-8'))
                df = pd.DataFrame(concepts_data)
            else:
                # 직접 CSV 형식 문자열 파싱
                import io
                df = pd.read_csv(io.StringIO(args.add_concepts))
            
            print(f"[INFO] 추가할 개념: {len(df)}개")
            print(df)
            cdb_maker.prepare_csvs([df])
            
            print("[SUCCESS] 개념 추가 완료")
            print(f"  - 총 CUI 수: {len(cat.cdb.cui2info)}")
            
        except Exception as e:
            print(f"[WARNING] 개념 추가 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 학습 전 테스트 (선택사항)
    if args.test_before_training and args.test_cases:
        print("\n[INFO] 학습 전 테스트 실행...")
        test_cases = json.load(open(args.test_cases, 'r', encoding='utf-8'))
        for i, test_case in enumerate(test_cases):
            text = test_case.get('text', '')
            expected_cui = test_case.get('expected_cui', None)
            entities = cat.get_entities(text)['entities']
            print(f"  테스트 {i+1}: {len(entities)}개 엔티티 발견")
            if entities:
                for ent_id, ent in entities.items():
                    print(f"    - CUI: {ent.get('cui')}, 이름: {ent.get('pretty_name')}")
            else:
                print(f"    - 엔티티 없음 (예상: {expected_cui})")
    
    # 학습 데이터 로드
    train_json_path = Path(args.train_json)
    if not train_json_path.exists():
        raise FileNotFoundError(f"학습 JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    print(f"[INFO] 학습 데이터 로드: {train_json_path}")
    train_data = load_train_json(str(train_json_path))
    
    # 데이터 검증
    if not isinstance(train_data, dict):
        raise ValueError("학습 데이터는 딕셔너리 형식이어야 합니다.")
    
    print(f"[INFO] 학습 데이터 구조:")
    print(f"  - Keys: {list(train_data.keys())}")
    if "documents" in train_data:
        print(f"  - Documents: {len(train_data.get('documents', []))}")
    if "annotations" in train_data:
        print(f"  - Annotations: {len(train_data.get('annotations', []))}")
    
    # Supervised Training (공식 튜토리얼 방식)
    print(f"\n[INFO] Supervised Training 시작...")
    print(f"  - 필터 사용: {args.use_filters}")
    
    try:
        trainer = cat.trainer
        
        # 공식 튜토리얼 방식: cat.trainer.train_supervised_raw(data, use_filters=True)
        trainer.train_supervised_raw(
            data=train_data,
            use_filters=args.use_filters,
        )
        
        print("[SUCCESS] Supervised Training 완료!")
        
        # 학습 결과 확인 (공식 튜토리얼 방식)
        print("\n[INFO] 학습 결과:")
        trained_concepts = [
            (ci['cui'], cat.cdb.get_name(ci['cui']), ci.get('count_train', 0)) 
            for ci in cat.cdb.cui2info.values() 
            if ci.get('count_train', 0) > 0
        ]
        print(f"  - 학습된 개념: {len(trained_concepts)}개")
        for cui, name, count in trained_concepts[:10]:  # 상위 10개만 표시
            print(f"    * {name} (CUI: {cui}, 학습 횟수: {count})")
        if len(trained_concepts) > 10:
            print(f"    ... 외 {len(trained_concepts) - 10}개")
    
    except Exception as e:
        print(f"[ERROR] Training 오류: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 모델 팩 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] 모델 팩 저장: {output_dir}")
    try:
        cat.save_model_pack(
            target_folder=str(output_dir),
            pack_name=args.pack_name,
            make_archive=True,
        )
        
        pack_path = output_dir / f"{args.pack_name}.zip"
        print(f"[SUCCESS] 모델 팩 저장 완료: {pack_path}")
        
        # 학습 후 테스트 (선택사항)
        if args.test_after_training and args.test_cases:
            print("\n[INFO] 학습 후 테스트 실행...")
            test_cases = json.load(open(args.test_cases, 'r', encoding='utf-8'))
            for i, test_case in enumerate(test_cases):
                text = test_case.get('text', '')
                expected_cui = test_case.get('expected_cui', None)
                entities = cat.get_entities(text)['entities']
                print(f"  테스트 {i+1}: {len(entities)}개 엔티티 발견")
                if entities:
                    for ent_id, ent in entities.items():
                        predicted_cui = ent.get('cui')
                        correct = predicted_cui == expected_cui if expected_cui else None
                        status = "[확인]" if correct else ("[취소]" if correct is False else "?")
                        print(f"    {status} CUI: {predicted_cui}, 이름: {ent.get('pretty_name')}, 정확도: {ent.get('acc', 'N/A')}")
                        if expected_cui:
                            print(f"      예상: {expected_cui}")
                else:
                    print(f"    - 엔티티 없음 (예상: {expected_cui})")
        
        print("\n이제 스캐폴드에서 사용할 수 있습니다!")
    
    except Exception as e:
        print(f"[ERROR] 모델 팩 저장 오류: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

