import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./models/kmbert_kbmc_ner")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--base_model", type=str, default="madatnlp/km-bert")
    return p.parse_args()

def main():
    args = parse_args()

    ds = load_dataset("SungJoo/KBMC")  # KBMC

    # tokenizer: try KR-BERT char tokenizer first, then fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424", use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Infer column names
    ex = ds[list(ds.keys())[0]][0]
    sent_col = "sentence" if "sentence" in ex else ("tokens" if "tokens" in ex else None)
    tag_col  = "tags" if "tags" in ex else ("ner_tags" if "ner_tags" in ex else None)
    if sent_col is None or tag_col is None:
        raise ValueError(f"Unexpected KBMC columns. Example keys: {list(ex.keys())}")

    def normalize(ex):
        if isinstance(ex[sent_col], str):
            tokens = ex[sent_col].split()
        else:
            tokens = ex[sent_col]
        if isinstance(ex[tag_col], str):
            labels = ex[tag_col].split()
        else:
            labels = ex[tag_col]
        m = min(len(tokens), len(labels))
        return {"tokens": tokens[:m], "labels_str": labels[:m]}

    ds2 = {split: ds[split].map(normalize, remove_columns=ds[split].column_names) for split in ds.keys()}

    # Label list
    all_labels = set()
    for split in ds2.keys():
        for row in ds2[split]:
            all_labels.update(row["labels_str"])
    label_list = sorted(list(all_labels))
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    def tokenize_and_align(ex):
        tok = tokenizer(ex["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_length)
        word_ids = tok.word_ids()
        labels = []
        prev = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(label2id[ex["labels_str"][wid]])
            else:
                labels.append(-100)
            prev = wid
        tok["labels"] = labels
        return tok

    ds_tok = {split: ds2[split].map(tokenize_and_align) for split in ds2.keys()}

    if "validation" not in ds_tok and "train" in ds_tok:
        tmp = ds_tok["train"].train_test_split(test_size=0.1, seed=args.seed)
        ds_tok["train"] = tmp["train"]
        ds_tok["validation"] = tmp["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        true_preds, true_labels = [], []
        for pred_seq, lab_seq in zip(preds, labels):
            cp, cl = [], []
            for pr, la in zip(pred_seq, lab_seq):
                if la == -100:
                    continue
                cp.append(id2label[int(pr)])
                cl.append(id2label[int(la)])
            true_preds.append(cp)
            true_labels.append(cl)
        out = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": out.get("overall_precision", 0.0),
            "recall": out.get("overall_recall", 0.0),
            "f1": out.get("overall_f1", 0.0),
        }

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved KM-BERT NER model to:", args.output_dir)

if __name__ == "__main__":
    main()
