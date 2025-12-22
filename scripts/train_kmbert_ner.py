"""Fine-tune KM-BERT for Korean medical NER (CoNLL format support).

현재 스캐폴드 개선사항:
- CoNLL 형식 데이터셋 지원 (dataset-agnostic)
- seqeval 기반 정확한 NER 평가
- 서브워드 토큰 처리 개선
- 테스트 세트 평가 지원

사용 예시:
    # 기본 학습
    python scripts/train_kmbert_ner.py \
      --base_model madatnlp/km-bert \
      --train_file data/kbmc/train.conll \
      --valid_file data/kbmc/valid.conll \
      --output_dir models/kmbert_ner_kbmc
    
    # 테스트 세트 포함
    python scripts/train_kmbert_ner.py \
      --base_model madatnlp/km-bert \
      --train_file data/kbmc/train.conll \
      --valid_file data/kbmc/valid.conll \
      --test_file data/kbmc/test.conll \
      --output_dir models/kmbert_ner_kbmc \
      --epochs 10 \
      --batch_size 16

CoNLL 형식:
    당뇨\tB-DISEASE
    병\tI-DISEASE
    환자\tO
    입니다\tO
    
    (빈 줄로 문장 구분)

주의사항:
- KM-BERT는 사전학습 모델이므로 NER을 위해서는 fine-tuning 필요
- 이미 fine-tuned 체크포인트가 있다면 KMBERT_NER_DIR 환경 변수로 지정
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """CoNLL 형식 파일 읽기
    
    Args:
        path: CoNLL 파일 경로
    
    Returns:
        (sentences, labels) 튜플
        - sentences: [[token, ...], ...]
        - labels: [[label, ...], ...]
    
    CoNLL 형식:
        token\tlabel
        token\tlabel
        (빈 줄로 문장 구분)
    """
    sentences: List[List[str]] = []
    labels: List[List[str]] = []
    cur_tokens: List[str] = []
    cur_labels: List[str] = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # 빈 줄: 문장 구분
            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    labels.append(cur_labels)
                    cur_tokens, cur_labels = [], []
                continue
            
            # 주석 무시
            if line.startswith("#"):
                continue
            
            # 토큰과 라벨 분리 (탭 또는 공백)
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()
            
            if len(parts) < 2:
                print(f"Warning: 잘못된 형식 무시: {line}")
                continue
            
            tok = parts[0]
            lab = parts[1]
            
            cur_tokens.append(tok)
            cur_labels.append(lab)
    
    # 마지막 문장 처리
    if cur_tokens:
        sentences.append(cur_tokens)
        labels.append(cur_labels)
    
    print(f"Loaded {len(sentences)} sentences from {path}")
    return sentences, labels


def build_label_map(all_labels: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """라벨 → ID 매핑 생성
    
    Args:
        all_labels: [[label, ...], ...]
    
    Returns:
        (label2id, id2label) 튜플
    """
    # 모든 고유 라벨 수집
    unique_labels = sorted({l for sent in all_labels for l in sent})
    
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    return label2id, id2label


def tokenize_and_align_labels(
    examples: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int]
) -> Dict[str, Any]:
    """토큰화 및 라벨 정렬
    
    서브워드 토큰 처리:
    - 첫 번째 서브워드: 원래 라벨 사용
    - 나머지 서브워드: -100 (loss 계산 시 무시)
    
    Args:
        examples: {"tokens": [[token, ...], ...], "labels": [[label, ...], ...]}
        tokenizer: HuggingFace tokenizer
        label2id: 라벨 → ID 매핑
    
    Returns:
        토큰화된 examples (input_ids, attention_mask, labels 포함)
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        return_offsets_mapping=False,
    )
    
    all_labels = []
    
    for i, word_labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        
        for word_id in word_ids:
            # 특수 토큰 ([CLS], [SEP] 등): -100
            if word_id is None:
                label_ids.append(-100)
            # 첫 번째 서브워드: 원래 라벨
            elif word_id != prev_word_id:
                label_ids.append(label2id[word_labels[word_id]])
            # 나머지 서브워드: -100 (무시)
            else:
                label_ids.append(-100)
            
            prev_word_id = word_id
        
        all_labels.append(label_ids)
    
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(p):
    """seqeval 기반 NER 평가 메트릭 계산
    
    Args:
        p: (predictions, labels) 튜플
    
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    predictions, labels = p
    preds = np.argmax(predictions, axis=-1)
    
    # -100 제거 및 라벨 문자열로 변환
    true_labels = []
    true_preds = []
    
    for pred_row, label_row in zip(preds, labels):
        sent_true = []
        sent_pred = []
        
        for p_i, l_i in zip(pred_row, label_row):
            if l_i == -100:
                continue
            sent_true.append(l_i)
            sent_pred.append(p_i)
        
        true_labels.append(sent_true)
        true_preds.append(sent_pred)
    
    # ID → 라벨 문자열 변환
    id2label = compute_metrics.id2label  # type: ignore[attr-defined]
    true_labels_str = [[id2label[i] for i in row] for row in true_labels]
    true_preds_str = [[id2label[i] for i in row] for row in true_preds]
    
    # seqeval로 평가
    return {
        "precision": precision_score(true_labels_str, true_preds_str),
        "recall": recall_score(true_labels_str, true_preds_str),
        "f1": f1_score(true_labels_str, true_preds_str),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune KM-BERT for Korean medical NER"
    )
    
    # 모델 및 데이터
    ap.add_argument(
        "--base_model",
        default="madatnlp/km-bert",
        help="Base model (default: madatnlp/km-bert)"
    )
    ap.add_argument(
        "--train_file",
        required=True,
        help="Training data (CoNLL format)"
    )
    ap.add_argument(
        "--valid_file",
        required=True,
        help="Validation data (CoNLL format)"
    )
    ap.add_argument(
        "--test_file",
        default=None,
        help="Test data (CoNLL format, optional)"
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for model checkpoint"
    )
    
    # 학습 하이퍼파라미터
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    
    args = ap.parse_args()
    
    # 시드 고정
    set_seed(args.seed)
    
    # 데이터 로드
    print(f"\n{'='*80}")
    print("데이터 로드 중...")
    print(f"{'='*80}\n")
    
    train_tokens, train_labels = read_conll(args.train_file)
    valid_tokens, valid_labels = read_conll(args.valid_file)
    
    # 라벨 매핑 생성
    all_labels = train_labels + valid_labels
    label2id, id2label = build_label_map(all_labels)
    
    # 데이터셋 생성
    ds = DatasetDict({
        "train": Dataset.from_dict({
            "tokens": train_tokens,
            "labels": train_labels
        }),
        "validation": Dataset.from_dict({
            "tokens": valid_tokens,
            "labels": valid_labels
        }),
    })
    
    # 토크나이저 로드
    print(f"\n토크나이저 로드: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    
    # 토큰화 및 라벨 정렬
    print("토큰화 중...")
    
    def _tokenize(examples):
        return tokenize_and_align_labels(examples, tokenizer, label2id)
    
    ds_tok = ds.map(
        _tokenize,
        batched=True,
        remove_columns=["tokens", "labels"]
    )
    
    # 모델 로드
    print(f"\n모델 로드: {args.base_model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # 학습 설정
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )
    
    # compute_metrics에 id2label 전달
    compute_metrics.id2label = id2label  # type: ignore[attr-defined]
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 학습 실행
    print(f"\n{'='*80}")
    print("학습 시작...")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    # 모델 저장
    print(f"\n모델 저장: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # 테스트 세트 평가 (선택)
    if args.test_file:
        print(f"\n{'='*80}")
        print("테스트 세트 평가...")
        print(f"{'='*80}\n")
        
        test_tokens, test_labels = read_conll(args.test_file)
        test_ds = Dataset.from_dict({
            "tokens": test_tokens,
            "labels": test_labels
        })
        test_tok = test_ds.map(
            _tokenize,
            batched=True,
            remove_columns=["tokens", "labels"]
        )
        
        metrics = trainer.evaluate(test_tok)
        
        print("\n테스트 세트 결과:")
        print(f"  Precision: {metrics.get('eval_precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('eval_recall', 0):.4f}")
        print(f"  F1:        {metrics.get('eval_f1', 0):.4f}")
        
        # 결과 저장
        metrics_file = output_dir / "test_metrics.txt"
        with open(metrics_file, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"\n테스트 결과 저장: {metrics_file}")
    
    print(f"\n{'='*80}")
    print("✓ 학습 완료!")
    print(f"{'='*80}")
    print(f"\n모델 위치: {output_dir}")
    print(f"\n다음 단계:")
    print(f"  1. .env 파일에 KMBERT_NER_DIR={output_dir} 설정")
    print(f"  2. python cli/run_compare.py --text \"테스트 텍스트\"")


if __name__ == "__main__":
    main()

