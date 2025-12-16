import argparse
from med_entity_ab.utils.io import read_jsonl
from med_entity_ab.schema import Entity
from med_entity_ab.metrics.ner_metrics import strict_span_match, overlap_match, boundary_iou_mean
from med_entity_ab.metrics.linking_metrics import evaluate_linking

def to_entities(rows):
    ents = []
    for e in rows:
        ents.append(Entity(
            start=int(e["start"]),
            end=int(e["end"]),
            text=str(e.get("text","")),
            label=e.get("label"),
            code=e.get("code"),
            score=e.get("score"),
            source=str(e.get("source","")),
            metadata=e.get("metadata") or {}
        ))
    return ents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_jsonl", type=str, required=True)
    ap.add_argument("--pred_jsonl", type=str, required=True)
    ap.add_argument("--mode", type=str, default="strict", choices=["strict","overlap"])
    ap.add_argument("--label_sensitive", action="store_true", default=False, help="default False for cross-model comparison")
    ap.add_argument("--iou_threshold", type=float, default=0.0)
    ap.add_argument("--linking", action="store_true", default=False, help="evaluate linking (requires gold code fields)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold_jsonl)
    pred_rows = read_jsonl(args.pred_jsonl)
    pred_map = {r["id"]: r for r in pred_rows}

    tp = fp = fn = 0
    iou_sum = 0.0
    n_docs = 0

    # linking accumulators
    link_acc1_sum = 0.0
    link_mrr_sum = 0.0
    link_n_sum = 0

    for gdoc in gold_rows:
        doc_id = gdoc["id"]
        pdoc = pred_map.get(doc_id, {"entities":[]})
        gents = to_entities(gdoc.get("entities", []))
        pents = to_entities(pdoc.get("entities", []))

        if args.mode == "strict":
            prf = strict_span_match(gents, pents, label_sensitive=args.label_sensitive)
        else:
            prf = overlap_match(gents, pents, label_sensitive=args.label_sensitive, iou_threshold=args.iou_threshold)

        tp += prf.tp
        fp += prf.fp
        fn += prf.fn
        iou_sum += boundary_iou_mean(gents, pents, label_sensitive=args.label_sensitive)
        n_docs += 1

        if args.linking:
            lm = evaluate_linking(
                gents, pents,
                match_mode=args.mode,
                iou_threshold=args.iou_threshold,
                label_sensitive=args.label_sensitive,
                k=args.k
            )
            # weighted by n
            link_acc1_sum += lm.accuracy_at_1 * lm.n
            link_mrr_sum += lm.mrr * lm.n
            link_n_sum += lm.n

    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    mean_iou = iou_sum/n_docs if n_docs else 0.0

    print("=== NER Metrics ===")
    print("mode:", args.mode, "| label_sensitive:", args.label_sensitive, "| iou_threshold:", args.iou_threshold)
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    print(f"MeanBoundaryIoU={mean_iou:.4f}")

    if args.linking:
        acc1 = (link_acc1_sum/link_n_sum) if link_n_sum else 0.0
        mrr = (link_mrr_sum/link_n_sum) if link_n_sum else 0.0
        print("\n=== Linking Metrics ===")
        print(f"n_matched_with_gold_code={link_n_sum}")
        print(f"Accuracy@1={acc1:.4f}  MRR@{args.k}={mrr:.4f}")

if __name__ == "__main__":
    main()
