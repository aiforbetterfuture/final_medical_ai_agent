#!/usr/bin/env python3
"""
MEDCAT2 Unsupervised Training 스크립트 (공식 튜토리얼 방식)

튜토리얼 2번: 도메인 코퍼스에 대한 비지도 학습 수행

공식 튜토리얼 방식:
    1. 모델 팩에서 직접 로드 (CAT.load_model_pack)
    2. cat.trainer.train_unsupervised() 실행
    3. 학습 결과 확인 (count_train)
    4. 모델 팩 저장

사용법:
    # 모델 팩 기반 (권장)
    python scripts/medcat2_train_unsupervised.py \
        --model-pack models/medcat2/base_model \
        --corpus-dir data/corpus/train_source \
        --output-dir models/medcat2 \
        --pack-name medcat2_unsupervised_trained
    
    # CDB/Vocab 기반 (호환성)
    python scripts/medcat2_train_unsupervised.py \
        --cdb-path models/medcat2/cdb.dat \
        --corpus-dir data/corpus \
        --output-dir models/medcat2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List
import json

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from medcat.cdb import CDB
    from medcat.vocab import Vocab
    from medcat.config import Config
    from medcat.cat import CAT
except ImportError:
    print("ERROR: medcat 패키지가 설치되지 않았습니다.")
    print("설치: pip install medcat>=2.0")
    sys.exit(1)


def iter_texts_from_dir(corpus_dir: str) -> Iterator[str]:
    """
    코퍼스 디렉토리에서 텍스트 파일들을 순회
    
    지원 형식:
        - .txt 파일
        - .jsonl 파일 (각 줄이 JSON 객체, 'text' 필드 포함)
        - .json 파일 (리스트 또는 단일 객체)
    """
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"코퍼스 디렉토리를 찾을 수 없습니다: {corpus_dir}")
    
    # .txt 파일 처리
    for txt_file in corpus_path.rglob("*.txt"):
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
            if text:
                yield text
        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({txt_file}): {e}")
            continue
    
    # .jsonl 파일 처리
    for jsonl_file in corpus_path.rglob("*.jsonl"):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get("text", "") or data.get("content", "") or data.get("body", "")
                        if text and isinstance(text, str):
                            yield text.strip()
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({jsonl_file}): {e}")
            continue
    
    # .json 파일 처리
    for json_file in corpus_path.rglob("*.json"):
        if json_file.name.endswith(".jsonl"):
            continue  # 이미 처리됨
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "") or item.get("content", "") or item.get("body", "")
                        if text and isinstance(text, str):
                            yield text.strip()
                elif isinstance(data, dict):
                    text = data.get("text", "") or data.get("content", "") or data.get("body", "")
                    if text and isinstance(text, str):
                        yield text.strip()
        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({json_file}): {e}")
            continue


def iter_texts_from_list(texts: List[str]) -> Iterator[str]:
    """텍스트 리스트에서 순회"""
    for text in texts:
        if text and isinstance(text, str) and text.strip():
            yield text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="MEDCAT2 Unsupervised Training (공식 튜토리얼 방식)"
    )
    parser.add_argument(
        "--model-pack",
        type=str,
        help="모델 팩 경로 (권장: models/medcat2/base_model 또는 .zip)"
    )
    parser.add_argument(
        "--cdb-path",
        type=str,
        help="CDB 파일 경로 (호환성: models/medcat2/cdb.dat, --model-pack 우선)"
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        help="Vocab 파일 경로 (선택사항, CDB와 같은 디렉토리에서 자동 탐색)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Config 파일 경로 (선택사항, CDB와 같은 디렉토리에서 자동 탐색)"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="코퍼스 디렉토리 경로 (data/corpus)"
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        help="코퍼스 파일 경로 (단일 파일, .txt 또는 .jsonl)"
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
        default="medcat2_umls_symptom_disease",
        help="모델 팩 이름"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="병렬 처리 프로세스 수"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=100000,
        help="최대 문서 수 (시간 제한 고려)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="배치 크기"
    )
    parser.add_argument(
        "--enable-korean-translation",
        action="store_true",
        help="한국어 코퍼스 자동 번역 (기본값: False)"
    )
    parser.add_argument(
        "--show-training-results",
        action="store_true",
        default=True,
        help="학습 결과 표시 (기본값: True)"
    )
    
    args = parser.parse_args()
    
    # 모델 팩 기반 로드 (공식 튜토리얼 방식, 권장)
    if args.model_pack:
        model_pack_path = Path(args.model_pack)
        if not model_pack_path.exists():
            raise FileNotFoundError(f"모델 팩을 찾을 수 없습니다: {model_pack_path}")
        
        print(f"[INFO] 모델 팩 로드 (공식 튜토리얼 방식): {model_pack_path}")
        cat = CAT.load_model_pack(str(model_pack_path))
        print("[SUCCESS] 모델 팩 로드 완료")
    
    # CDB/Vocab/Config 기반 로드 (호환성)
    elif args.cdb_path:
        cdb_path = Path(args.cdb_path)
        if not cdb_path.exists():
            raise FileNotFoundError(f"CDB 파일을 찾을 수 없습니다: {cdb_path}")
        
        print(f"[INFO] CDB 로드: {cdb_path}")
        cdb = CDB()
        cdb.load_dict(str(cdb_path))
        
        # Vocab 로드
        vocab_path = Path(args.vocab_path) if args.vocab_path else cdb_path.parent / "vocab.dat"
        if vocab_path.exists():
            print(f"[INFO] Vocab 로드: {vocab_path}")
            vocab = Vocab()
            vocab.load_dict(str(vocab_path))
        else:
            print("[WARNING] Vocab 파일을 찾을 수 없습니다. 빈 Vocab을 사용합니다.")
            vocab = Vocab()
        
        # Config 로드
        config_path = Path(args.config_path) if args.config_path else cdb_path.parent / "config.json"
        if config_path.exists():
            print(f"[INFO] Config 로드: {config_path}")
            config = Config()
            config.load(str(config_path))
        else:
            print("[WARNING] Config 파일을 찾을 수 없습니다. 기본 Config를 사용합니다.")
            config = Config()
        
        # CAT 생성
        print("[INFO] CAT 초기화...")
        cat = CAT(cdb=cdb, config=config, vocab=vocab)
    
    else:
        print("[ERROR] --model-pack 또는 --cdb-path 중 하나를 지정해야 합니다.")
        return
    
    # 코퍼스 로드
    texts_iter = None
    doc_count = 0
    
    if args.corpus_dir:
        print(f"[INFO] 코퍼스 디렉토리 로드: {args.corpus_dir}")
        texts_iter = iter_texts_from_dir(args.corpus_dir)
        # 문서 수 미리 계산 (선택적)
        print("[INFO] 코퍼스 문서 수 계산 중...")
        temp_iter = iter_texts_from_dir(args.corpus_dir)
        doc_count = sum(1 for _ in temp_iter)
        print(f"[INFO] 총 {doc_count}개 문서 발견")
        texts_iter = iter_texts_from_dir(args.corpus_dir)  # 다시 생성
    
    elif args.corpus_file:
        corpus_file = Path(args.corpus_file)
        if not corpus_file.exists():
            raise FileNotFoundError(f"코퍼스 파일을 찾을 수 없습니다: {corpus_file}")
        
        print(f"[INFO] 코퍼스 파일 로드: {corpus_file}")
        if corpus_file.suffix == ".txt":
            text = corpus_file.read_text(encoding='utf-8', errors='ignore')
            # 줄 단위로 분할
            texts = [line.strip() for line in text.split('\n') if line.strip()]
            texts_iter = iter_texts_from_list(texts)
            doc_count = len(texts)
        elif corpus_file.suffix == ".jsonl":
            texts_iter = iter_texts_from_dir(str(corpus_file.parent))
            # 파일 하나만 처리하도록 필터링
            # (실제로는 iter_texts_from_dir이 이미 처리함)
            temp_iter = iter_texts_from_dir(str(corpus_file.parent))
            doc_count = sum(1 for _ in temp_iter)
            texts_iter = iter_texts_from_dir(str(corpus_file.parent))
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {corpus_file.suffix}")
    
    else:
        print("[ERROR] --corpus-dir 또는 --corpus-file 중 하나를 지정해야 합니다.")
        return
    
    if not texts_iter:
        print("[ERROR] 코퍼스를 로드할 수 없습니다.")
        return
    
    # 문서 수 제한
    if args.max_docs and doc_count > args.max_docs:
        print(f"[INFO] 문서 수 제한: {doc_count}개 -> {args.max_docs}개")
        doc_count = args.max_docs
    
    # 한국어 번역 처리
    if args.enable_korean_translation:
        print("[INFO] 한국어 번역 모드 활성화")
        try:
            from nlp.korean_translator import KoreanTranslator
            translator = KoreanTranslator()
            
            # 텍스트를 리스트로 변환하여 번역
            original_texts = []
            for i, text in enumerate(texts_iter):
                if args.max_docs and i >= args.max_docs:
                    break
                original_texts.append(text)
            
            print(f"[INFO] 한국어 텍스트 번역 중: {len(original_texts)}개")
            translated_texts = []
            for text in original_texts:
                english_text = translator.translate_to_english(text)
                translated_texts.append(english_text)
            
            texts_for_training = translated_texts
            print(f"[INFO] 번역 완료: {len(translated_texts)}개")
        except Exception as e:
            print(f"[WARNING] 한국어 번역 실패, 원본 텍스트 사용: {e}")
            texts_for_training = list(texts_iter)[:args.max_docs] if args.max_docs else list(texts_iter)
    else:
        # 문서 제한을 위한 리스트 변환
        texts_for_training = []
        for i, text in enumerate(texts_iter):
            if args.max_docs and i >= args.max_docs:
                break
            texts_for_training.append(text)
    
    print(f"[INFO] 실제 학습 문서 수: {len(texts_for_training)}")
    
    # 학습 전 상태 확인 (공식 튜토리얼 방식)
    if args.show_training_results:
        print("\n[INFO] 학습 전 상태:")
        trained_concepts_before = [
            (ci['cui'], cat.cdb.get_name(ci['cui']), ci.get('count_train', 0)) 
            for ci in cat.cdb.cui2info.values() 
            if ci.get('count_train', 0) > 0
        ]
        trained_names_before = [
            (ni["name"], ni.get("count_train", 0)) 
            for ni in cat.cdb.name2info.values() 
            if ni.get("count_train", 0) > 0
        ]
        print(f"  - 학습된 개념: {len(trained_concepts_before)}개")
        print(f"  - 학습된 이름: {len(trained_names_before)}개")
    
    # Unsupervised Training (공식 튜토리얼 방식)
    print("\n[INFO] Unsupervised Training 시작...")
    print(f"  - 학습 문서 수: {len(texts_for_training)}")
    print(f"  - 병렬 프로세스: {args.n_workers}")
    
    try:
        # 공식 튜토리얼: cat.trainer.train_unsupervised(texts)
        trainer = cat.trainer
        
        # 공식 튜토리얼 방식: 위치 인자로 텍스트 리스트 전달
        # cat.trainer.train_unsupervised(unsup_train_texts)
        # 배치 단위 학습 (대용량 데이터 처리)
        if len(texts_for_training) > args.batch_size:
            print(f"[INFO] 배치 단위 학습: 배치 크기 {args.batch_size}")
            for i in range(0, len(texts_for_training), args.batch_size):
                batch = texts_for_training[i:i+args.batch_size]
                # 위치 인자로 전달 (튜토리얼 방식)
                trainer.train_unsupervised(batch)
                if (i // args.batch_size + 1) % 10 == 0:
                    print(f"[INFO] 진행: {i+len(batch)}/{len(texts_for_training)}개 문서 처리됨")
        else:
            # 전체 한 번에 학습 (튜토리얼 방식)
            trainer.train_unsupervised(texts_for_training)
        
        print("[SUCCESS] Unsupervised Training 완료!")
    
    except Exception as e:
        print(f"[ERROR] Training 오류: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 학습 결과 확인 (공식 튜토리얼 방식)
    if args.show_training_results:
        print("\n[INFO] 학습 후 상태:")
        trained_concepts_after = [
            (ci['cui'], cat.cdb.get_name(ci['cui']), ci.get('count_train', 0)) 
            for ci in cat.cdb.cui2info.values() 
            if ci.get('count_train', 0) > 0
        ]
        trained_names_after = [
            (ni["name"], ni.get("count_train", 0)) 
            for ni in cat.cdb.name2info.values() 
            if ni.get("count_train", 0) > 0
        ]
        
        print(f"  - 학습된 개념: {len(trained_concepts_after)}개")
        for cui, name, count in trained_concepts_after[:10]:  # 상위 10개만 표시
            print(f"    * {name} (CUI: {cui}, 학습 횟수: {count})")
        if len(trained_concepts_after) > 10:
            print(f"    ... 외 {len(trained_concepts_after) - 10}개")
        
        print(f"  - 학습된 이름: {len(trained_names_after)}개")
        for name, count in trained_names_after[:10]:  # 상위 10개만 표시
            print(f"    * {name} (학습 횟수: {count})")
        if len(trained_names_after) > 10:
            print(f"    ... 외 {len(trained_names_after) - 10}개")
        
        # 학습 전후 비교
        new_concepts = len(trained_concepts_after) - len(trained_concepts_before)
        new_names = len(trained_names_after) - len(trained_names_before)
        print(f"\n[INFO] 학습 효과:")
        print(f"  - 새로 학습된 개념: {new_concepts}개")
        print(f"  - 새로 학습된 이름: {new_names}개")
    
    # 모델 팩 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] 모델 팩 저장: {output_dir}")
    try:
        cat.save_model_pack(
            target_folder=str(output_dir),
            pack_name=args.pack_name,
            make_archive=True,  # zip 생성
        )
        
        pack_path = output_dir / f"{args.pack_name}.zip"
        print(f"[SUCCESS] 모델 팩 저장 완료: {pack_path}")
        print("\n다음 단계 (선택): python scripts/medcat2_train_supervised.py")
    
    except Exception as e:
        print(f"[ERROR] 모델 팩 저장 오류: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

