#!/usr/bin/env python3
"""
MEDCAT2 CDB & Vocab 생성 스크립트 (UMLS RRF 직접 처리)

공식 튜토리얼 방식: UMLS RRF 파일(MRCONSO.RRF, MRSTY.RRF)을 직접 읽어서
CDB/VCB를 생성합니다.

사용법:
    # RRF 파일에서 직접 생성
    python scripts/medcat2_build_from_umls_rrf.py \
        --mrconso MRCONSO.RRF \
        --mrsty MRSTY.RRF \
        --output-dir models/medcat2
    
    # RRF -> CSV 변환 후 생성
    python scripts/medcat2_build_from_umls_rrf.py \
        --mrconso MRCONSO.RRF \
        --mrsty MRSTY.RRF \
        --output-csv umls_concepts_for_medcat.csv \
        --output-dir models/medcat2

참고:
    - MRCONSO.RRF: UMLS 개념 용어 파일
    - MRSTY.RRF: UMLS 의미 유형 파일
    - 공식 튜토리얼의 CDBMaker를 사용합니다
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from medcat.cdb import CDB
    from medcat.vocab import Vocab
    from medcat.config import Config
    from medcat.model_creation.cdb_maker import CDBMaker
    from medcat.cat import CAT
except ImportError:
    print("ERROR: medcat 패키지가 설치되지 않았습니다.")
    print("설치: pip install medcat>=2.0")
    sys.exit(1)


# UMLS RRF 파일 컬럼 정의
CONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "STR", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
    "CODE", "SRL", "SUPPRESS", "CVF"
]

STY_COLS = [
    "CUI", "TUI", "STN", "STY", "ATUI", "CVF"
]


def load_mrconso_rrf(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    MRCONSO.RRF 파일 로드
    
    Args:
        file_path: MRCONSO.RRF 파일 경로
        nrows: 읽을 행 수 제한 (테스트용)
    
    Returns:
        DataFrame with CONSO columns
    """
    print(f"[INFO] MRCONSO.RRF 로드: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MRCONSO.RRF 파일을 찾을 수 없습니다: {file_path}")
    
    # RRF 파일은 파이프(|) 구분자, 헤더 없음
    df = pd.read_csv(
        file_path,
        sep="|",
        names=CONSO_COLS,
        dtype=str,
        engine="python",
        nrows=nrows
    )
    
    # 마지막 빈 컬럼 제거 (RRF 파일 끝의 | 때문에)
    df = df.iloc[:, :-1]
    
    print(f"[INFO] MRCONSO.RRF 로드 완료: {len(df)}개 행")
    return df


def load_mrsty_rrf(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    MRSTY.RRF 파일 로드
    
    Args:
        file_path: MRSTY.RRF 파일 경로
        nrows: 읽을 행 수 제한 (테스트용)
    
    Returns:
        DataFrame with STY columns
    """
    print(f"[INFO] MRSTY.RRF 로드: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MRSTY.RRF 파일을 찾을 수 없습니다: {file_path}")
    
    df = pd.read_csv(
        file_path,
        sep="|",
        names=STY_COLS,
        dtype=str,
        engine="python",
        nrows=nrows
    )
    
    # 마지막 빈 컬럼 제거
    df = df.iloc[:, :-1]
    
    print(f"[INFO] MRSTY.RRF 로드 완료: {len(df)}개 행")
    return df


def filter_umls_data(
    mrconso: pd.DataFrame,
    mrsty: pd.DataFrame,
    languages: List[str] = ["ENG"],
    sources: Optional[List[str]] = None,
    preferred_only: bool = True,
    semantic_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    UMLS 데이터 필터링 및 병합
    
    Args:
        mrconso: MRCONSO DataFrame
        mrsty: MRSTY DataFrame
        languages: 포함할 언어 리스트 (기본값: ["ENG"])
        sources: 포함할 소스 vocabulary 리스트 (예: ["SNOMEDCT_US", "ICD10CM"])
        preferred_only: Preferred term만 사용할지 여부
        semantic_types: 포함할 semantic type 이름 리스트 (예: ["Disease or Syndrome"])
    
    Returns:
        필터링된 DataFrame (CUI, name, TUI, semantic_type, SAB, CODE 포함)
    """
    print("[INFO] UMLS 데이터 필터링 시작...")
    
    # 1) 언어 필터링
    if languages:
        mrconso_filtered = mrconso[mrconso["LAT"].isin(languages)]
        print(f"[INFO] 언어 필터링 ({languages}): {len(mrconso_filtered)}개 행")
    else:
        mrconso_filtered = mrconso
    
    # 2) 소스 vocabulary 필터링
    if sources:
        mrconso_filtered = mrconso_filtered[mrconso_filtered["SAB"].isin(sources)]
        print(f"[INFO] 소스 필터링 ({sources}): {len(mrconso_filtered)}개 행")
    
    # 3) Preferred term만
    if preferred_only:
        mrconso_filtered = mrconso_filtered[mrconso_filtered["ISPREF"] == "Y"]
        print(f"[INFO] Preferred term 필터링: {len(mrconso_filtered)}개 행")
    
    # 4) MRSTY와 semantic type 정보 병합
    umls_df = mrconso_filtered.merge(
        mrsty[["CUI", "TUI", "STY"]],
        on="CUI",
        how="left"
    )
    
    # 5) Semantic type 필터링
    if semantic_types:
        umls_df = umls_df[umls_df["STY"].isin(semantic_types)]
        print(f"[INFO] Semantic type 필터링 ({semantic_types}): {len(umls_df)}개 행")
    
    # 6) 컬럼명 정리 및 선택
    umls_df = umls_df.rename(columns={
        "STR": "name",
        "STY": "semantic_type",
        "TUI": "type_id"
    })
    
    # 필요한 컬럼만 선택
    result_df = umls_df[["CUI", "name", "type_id", "semantic_type", "SAB", "CODE"]].copy()
    
    # 중복 제거 (같은 CUI+name 조합)
    result_df = result_df.drop_duplicates(subset=["CUI", "name"])
    
    print(f"[INFO] 필터링 완료: {len(result_df)}개 고유 개념")
    return result_df


def build_cdb_vocab_with_cdbmaker(
    umls_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[Config] = None
) -> tuple[CDB, Vocab, Config]:
    """
    CDBMaker를 사용하여 CDB/Vocab 생성 (공식 튜토리얼 방식)
    
    Args:
        umls_df: UMLS DataFrame (CUI, name 컬럼 포함)
        output_dir: 출력 디렉토리
        config: Config 객체 (없으면 새로 생성)
    
    Returns:
        (cdb, vocab, config) 튜플
    """
    print("[INFO] CDB/Vocab 생성 시작 (CDBMaker 사용)...")
    
    # Config 생성 (공식 튜토리얼: 기본 Config 사용)
    if config is None:
        config = Config()
        # 공식 튜토리얼에서는 기본 Config를 그대로 사용
        # 필요시 설정 변경 가능하지만 기본값으로 시작
    
    # CDB 생성
    cdb = CDB(config)
    
    # CDBMaker 생성 (공식 튜토리얼 방식)
    maker = CDBMaker(config, cdb)
    
    # CUI별로 그룹화하여 처리
    print("[INFO] 개념 추가 중...")
    added_count = 0
    
    for cui, group in umls_df.groupby("CUI"):
        # 각 CUI에 대한 모든 name을 리스트로
        name_list = group["name"].dropna().unique().tolist()
        
        if not name_list:
            continue
        
        # DataFrame 생성 (CDBMaker 형식)
        cui_df = pd.DataFrame({
            "cui": [cui] * len(name_list),
            "name": name_list
        })
        
        # CDBMaker로 추가
        try:
            maker.prepare_csvs([cui_df])
            added_count += 1
            
            if added_count % 1000 == 0:
                print(f"[INFO] 진행: {added_count}개 개념 추가됨")
        except Exception as e:
            print(f"[WARNING] CUI {cui} 처리 오류: {e}")
            continue
    
    print(f"[INFO] CDB 생성 완료: {added_count}개 개념 추가됨")
    print(f"[INFO] CDB CUI 수: {len(cdb.cui2info)}")
    print(f"[INFO] CDB Name 수: {len(cdb.name2info)}")
    
    # Vocab 생성
    print("[INFO] Vocab 생성 시작...")
    vocab = Vocab()
    
    # CDB의 모든 name을 Vocab에 추가
    vocab_words = set()
    for name_info in cdb.name2info.values():
        # name은 ~로 구분된 형태일 수 있음 (CDBMaker 전처리)
        name = name_info.get("name", "")
        if name:
            # 단어 단위로 분리하여 추가
            words = name.replace("~", " ").split()
            vocab_words.update(words)
    
    # Vocab에 단어 추가 (공식 튜토리얼 방식: vocab.add_word("word", count, vector))
    for word in vocab_words:
        if len(word) >= 2:  # 최소 길이
            # 기본 임베딩 벡터 (나중에 학습으로 업데이트)
            # 공식 튜토리얼: vocab.add_word("severe", 10000, np.array((1.0, 0, 0, 1, 0, 0, 0)))
            vocab.add_word(
                word,
                1000,  # 기본 빈도 (위치 인자)
                np.random.rand(300)  # 기본 벡터 (실제로는 학습 필요)
            )
    
    print(f"[INFO] Vocab 생성 완료: {len(vocab.vocab)}개 단어")
    
    return cdb, vocab, config


def build_cdb_vocab_simple(
    umls_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[Config] = None
) -> tuple[CDB, Vocab, Config]:
    """
    간단한 방식으로 CDB/Vocab 생성 (CDBMaker 없이)
    
    Args:
        umls_df: UMLS DataFrame
        output_dir: 출력 디렉토리
        config: Config 객체
    
    Returns:
        (cdb, vocab, config) 튜플
    """
    print("[INFO] CDB/Vocab 생성 시작 (간단한 방식)...")
    
    if config is None:
        config = Config()
        # 공식 튜토리얼: 기본 Config 사용
    
    cdb = CDB(config)
    vocab = Vocab()
    
    added_count = 0
    for _, row in umls_df.iterrows():
        cui = str(row["CUI"]).strip()
        name = str(row["name"]).strip()
        
        if not cui or not name:
            continue
        
        try:
            # CDB에 개념 추가
            cdb.add_concept(cui=cui, name=name)
            
            # Vocab에 단어 추가 (공식 튜토리얼 방식)
            words = name.split()
            for word in words:
                if len(word) >= 2:
                    vocab.add_word(
                        word,
                        1000,  # 위치 인자
                        np.random.rand(300)
                    )
            
            added_count += 1
            
            if added_count % 1000 == 0:
                print(f"[INFO] 진행: {added_count}개 개념 추가됨")
        
        except Exception as e:
            print(f"[WARNING] CUI {cui} 처리 오류: {e}")
            continue
    
    print(f"[INFO] CDB/Vocab 생성 완료: {added_count}개 개념")
    return cdb, vocab, config


def main():
    parser = argparse.ArgumentParser(
        description="MEDCAT2 CDB & Vocab 생성 (UMLS RRF 직접 처리)"
    )
    parser.add_argument(
        "--mrconso",
        type=str,
        required=True,
        help="MRCONSO.RRF 파일 경로"
    )
    parser.add_argument(
        "--mrsty",
        type=str,
        required=True,
        help="MRSTY.RRF 파일 경로"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/medcat2",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="중간 CSV 파일 저장 경로 (선택사항)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["ENG"],
        help="포함할 언어 (기본값: ENG)"
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        help="포함할 소스 vocabulary (예: SNOMEDCT_US ICD10CM RXNORM)"
    )
    parser.add_argument(
        "--semantic-types",
        type=str,
        nargs="+",
        help="포함할 semantic type 이름 (예: 'Disease or Syndrome' 'Sign or Symptom')"
    )
    parser.add_argument(
        "--no-preferred-only",
        action="store_true",
        help="Preferred term만 사용하지 않음 (모든 용어 포함)"
    )
    parser.add_argument(
        "--use-cdbmaker",
        action="store_true",
        default=True,
        help="CDBMaker 사용 (공식 튜토리얼 방식, 기본값: True)"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        help="테스트용: 읽을 행 수 제한"
    )
    parser.add_argument(
        "--pack-name",
        type=str,
        default="base_model",
        help="모델 팩 이름 (기본값: base_model)"
    )
    parser.add_argument(
        "--create-model-pack",
        action="store_true",
        default=True,
        help="모델 팩 생성 (기본값: True)"
    )
    parser.add_argument(
        "--no-create-model-pack",
        action="store_false",
        dest="create_model_pack",
        help="모델 팩 생성하지 않음"
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) RRF 파일 로드
    mrconso = load_mrconso_rrf(args.mrconso, nrows=args.nrows)
    mrsty = load_mrsty_rrf(args.mrsty, nrows=args.nrows)
    
    # 2) 필터링 및 병합
    umls_df = filter_umls_data(
        mrconso=mrconso,
        mrsty=mrsty,
        languages=args.languages,
        sources=args.sources,
        preferred_only=not args.no_preferred_only,
        semantic_types=args.semantic_types
    )
    
    if umls_df.empty:
        print("[ERROR] 필터링 후 데이터가 비어있습니다.")
        return
    
    # 3) 중간 CSV 저장 (선택사항)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        umls_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"[INFO] 중간 CSV 저장: {csv_path}")
    
    # 4) CDB/Vocab 생성
    if args.use_cdbmaker:
        cdb, vocab, config = build_cdb_vocab_with_cdbmaker(
            umls_df=umls_df,
            output_dir=output_dir,
            config=None
        )
    else:
        cdb, vocab, config = build_cdb_vocab_simple(
            umls_df=umls_df,
            output_dir=output_dir,
            config=None
        )
    
    print("\n[SUCCESS] CDB & Vocab 생성 완료!")
    print(f"  - CDB: {len(cdb.cui2info)}개 CUI, {len(cdb.name2info)}개 Name")
    print(f"  - Vocab: {len(vocab.vocab)}개 단어")
    
    # 5) 모델 팩 생성 (공식 튜토리얼 방식)
    if args.create_model_pack:
        print("\n[INFO] 모델 팩 생성 시작...")
        try:
            # CAT 생성 (공식 튜토리얼: cat = CAT(cdb, vocab, cnf))
            print("[INFO] CAT 객체 생성 중...")
            cat = CAT(cdb, vocab, config)
            
            # 모델 팩 저장 (공식 튜토리얼: cat.save_model_pack(save_path, pack_name="base_model", add_hash_to_pack_name=False))
            print(f"[INFO] 모델 팩 저장: {output_dir}")
            mpp = cat.save_model_pack(
                str(output_dir),
                pack_name=args.pack_name,
                add_hash_to_pack_name=False
            )
            
            print(f"[SUCCESS] 모델 팩 저장 완료: {mpp}")
            
            # 모델 팩 테스트 (공식 튜토리얼 방식)
            print(f"\n[INFO] 모델 팩 테스트:")
            test_text = "Patient was diagnosed with diabetes last year."
            print(f"  테스트 텍스트: '{test_text}'")
            entities = cat.get_entities(test_text)
            print(f"  추출된 엔티티: {len(entities.get('entities', {}))}개")
            if entities.get('entities'):
                for ent_id, ent in entities['entities'].items():
                    print(f"    - {ent.get('pretty_name', 'N/A')} (CUI: {ent.get('cui', 'N/A')})")
            else:
                print("    (엔티티 없음 - 학습이 필요할 수 있습니다)")
            
            print(f"\n[SUCCESS] 전체 프로세스 완료!")
            print(f"  모델 팩 경로: {mpp}")
            print(f"  사용 방법: cat = CAT.load_model_pack('{mpp}')")
            
        except Exception as e:
            print(f"[ERROR] 모델 팩 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            print("\n[INFO] CDB/Vocab는 생성되었지만 모델 팩 저장에 실패했습니다.")
    else:
        print("\n[INFO] 모델 팩 생성 건너뜀 (--no-create-model-pack 옵션)")
        print("\n다음 단계: python scripts/medcat2_train_unsupervised.py")


if __name__ == "__main__":
    main()

