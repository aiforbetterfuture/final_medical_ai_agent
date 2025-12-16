#!/usr/bin/env python3
"""
MEDCAT2 학습 전후 성능 평가 스크립트

학습 전후 모델의 Precision, Recall, F1 Score를 계산합니다.

사용법:
    python scripts/medcat2_evaluate_training.py \
        --model-before models/medcat2/medcat2_unsupervised_trained \
        --model-after models/medcat2/medcat2_supervised_trained \
        --test-cases data/test_cases.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from medcat.cat import CAT
except ImportError:
    print("ERROR: medcat 패키지가 설치되지 않았습니다.")
    print("설치: pip install medcat>=2.0")
    sys.exit(1)


def load_test_cases(json_path: str) -> List[Dict[str, Any]]:
    """테스트 케이스 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'test_cases' in data:
        return data['test_cases']
    else:
        raise ValueError("테스트 케이스는 리스트 형식이어야 합니다.")


def extract_entities(cat: CAT, text: str) -> List[Dict[str, Any]]:
    """텍스트에서 엔티티 추출"""
    result = cat.get_entities(text)
    entities = result.get('entities', {})
    
    # 딕셔너리를 리스트로 변환
    entity_list = []
    for ent_id, ent in entities.items():
        entity_list.append({
            'cui': ent.get('cui'),
            'name': ent.get('pretty_name'),
            'source_value': ent.get('source_value'),
            'start': ent.get('start'),
            'end': ent.get('end'),
            'acc': ent.get('acc'),
            'context_similarity': ent.get('context_similarity'),
        })
    
    return entity_list


def evaluate_model(cat: CAT, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """모델 성능 평가"""
    results = {
        'total': len(test_cases),
        'correct': 0,
        'incorrect': 0,
        'missed': 0,
        'details': []
    }
    
    for i, test_case in enumerate(test_cases):
        text = test_case.get('text', '')
        expected_cui = test_case.get('expected_cui', None)
        expected_name = test_case.get('expected_name', None)
        
        # 엔티티 추출
        entities = extract_entities(cat, text)
        
        # 결과 분석
        if expected_cui:
            # CUI 기반 평가
            predicted_cui = entities[0]['cui'] if entities else None
            correct = predicted_cui == expected_cui
            
            if correct:
                results['correct'] += 1
            elif predicted_cui:
                results['incorrect'] += 1
            else:
                results['missed'] += 1
        elif expected_name:
            # 이름 기반 평가
            predicted_names = [e['name'] for e in entities]
            correct = expected_name in predicted_names if predicted_names else False
            
            if correct:
                results['correct'] += 1
            elif entities:
                results['incorrect'] += 1
            else:
                results['missed'] += 1
        else:
            # 엔티티 존재 여부만 평가
            correct = len(entities) > 0
            if correct:
                results['correct'] += 1
            else:
                results['missed'] += 1
        
        results['details'].append({
            'index': i + 1,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'expected_cui': expected_cui,
            'expected_name': expected_name,
            'predicted_entities': entities,
            'correct': correct
        })
    
    # Precision, Recall, F1 계산
    if results['total'] > 0:
        precision = results['correct'] / (results['correct'] + results['incorrect']) if (results['correct'] + results['incorrect']) > 0 else 0.0
        recall = results['correct'] / results['total']
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
    else:
        results['precision'] = 0.0
        results['recall'] = 0.0
        results['f1'] = 0.0
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MEDCAT2 학습 전후 성능 평가"
    )
    parser.add_argument(
        "--model-before",
        type=str,
        help="학습 전 모델 팩 경로"
    )
    parser.add_argument(
        "--model-after",
        type=str,
        help="학습 후 모델 팩 경로"
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        required=True,
        help="테스트 케이스 JSON 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="결과 저장 경로 (JSON 형식)"
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="상세 결과 표시"
    )
    
    args = parser.parse_args()
    
    # 테스트 케이스 로드
    test_cases_path = Path(args.test_cases)
    if not test_cases_path.exists():
        raise FileNotFoundError(f"테스트 케이스 파일을 찾을 수 없습니다: {test_cases_path}")
    
    print(f"[INFO] 테스트 케이스 로드: {test_cases_path}")
    test_cases = load_test_cases(str(test_cases_path))
    print(f"[INFO] 총 {len(test_cases)}개 테스트 케이스")
    
    results = {}
    
    # 학습 전 모델 평가
    if args.model_before:
        model_before_path = Path(args.model_before)
        if not model_before_path.exists():
            raise FileNotFoundError(f"학습 전 모델을 찾을 수 없습니다: {model_before_path}")
        
        print(f"\n[INFO] 학습 전 모델 로드: {model_before_path}")
        cat_before = CAT.load_model_pack(str(model_before_path))
        
        print("[INFO] 학습 전 모델 평가 중...")
        results['before'] = evaluate_model(cat_before, test_cases)
        
        print(f"\n[결과] 학습 전 모델:")
        print(f"  - 정확도 (Precision): {results['before']['precision']:.4f}")
        print(f"  - 재현율 (Recall): {results['before']['recall']:.4f}")
        print(f"  - F1 Score: {results['before']['f1']:.4f}")
        print(f"  - 정답: {results['before']['correct']}/{results['before']['total']}")
        print(f"  - 오답: {results['before']['incorrect']}")
        print(f"  - 미검출: {results['before']['missed']}")
    
    # 학습 후 모델 평가
    if args.model_after:
        model_after_path = Path(args.model_after)
        if not model_after_path.exists():
            raise FileNotFoundError(f"학습 후 모델을 찾을 수 없습니다: {model_after_path}")
        
        print(f"\n[INFO] 학습 후 모델 로드: {model_after_path}")
        cat_after = CAT.load_model_pack(str(model_after_path))
        
        print("[INFO] 학습 후 모델 평가 중...")
        results['after'] = evaluate_model(cat_after, test_cases)
        
        print(f"\n[결과] 학습 후 모델:")
        print(f"  - 정확도 (Precision): {results['after']['precision']:.4f}")
        print(f"  - 재현율 (Recall): {results['after']['recall']:.4f}")
        print(f"  - F1 Score: {results['after']['f1']:.4f}")
        print(f"  - 정답: {results['after']['correct']}/{results['after']['total']}")
        print(f"  - 오답: {results['after']['incorrect']}")
        print(f"  - 미검출: {results['after']['missed']}")
    
    # 향상도 계산
    if 'before' in results and 'after' in results:
        improvement = {
            'precision': results['after']['precision'] - results['before']['precision'],
            'recall': results['after']['recall'] - results['before']['recall'],
            'f1': results['after']['f1'] - results['before']['f1'],
        }
        
        print(f"\n[향상도] 학습 전후 비교:")
        print(f"  - Precision 향상: {improvement['precision']:+.4f} ({improvement['precision']/results['before']['precision']*100:+.2f}%)")
        print(f"  - Recall 향상: {improvement['recall']:+.4f} ({improvement['recall']/results['before']['recall']*100:+.2f}%)")
        print(f"  - F1 Score 향상: {improvement['f1']:+.4f} ({improvement['f1']/results['before']['f1']*100:+.2f}%)")
        
        results['improvement'] = improvement
    
    # 상세 결과 표시
    if args.show_details and 'after' in results:
        print(f"\n[상세 결과] 학습 후 모델:")
        for detail in results['after']['details']:
            status = "[확인]" if detail['correct'] else "[취소]"
            print(f"  {status} 테스트 {detail['index']}: {detail['text']}")
            if detail['predicted_entities']:
                for ent in detail['predicted_entities']:
                    print(f"    - CUI: {ent['cui']}, 이름: {ent['name']}, 정확도: {ent.get('acc', 'N/A')}")
            else:
                print(f"    - 엔티티 없음 (예상: {detail.get('expected_cui') or detail.get('expected_name')})")
    
    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 상세 정보 제거 (파일 크기 절감)
        results_to_save = {
            'before': {
                'precision': results.get('before', {}).get('precision'),
                'recall': results.get('before', {}).get('recall'),
                'f1': results.get('before', {}).get('f1'),
                'correct': results.get('before', {}).get('correct'),
                'total': results.get('before', {}).get('total'),
            } if 'before' in results else None,
            'after': {
                'precision': results.get('after', {}).get('precision'),
                'recall': results.get('after', {}).get('recall'),
                'f1': results.get('after', {}).get('f1'),
                'correct': results.get('after', {}).get('correct'),
                'total': results.get('after', {}).get('total'),
            } if 'after' in results else None,
            'improvement': results.get('improvement'),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] 결과 저장: {output_path}")


if __name__ == "__main__":
    main()

