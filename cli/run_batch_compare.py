import argparse
import os
from dotenv import load_dotenv
from med_entity_ab.pipeline import load_config, EntityABPipeline
from med_entity_ab.utils.io import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True, help='Each line: {"id":"...","text":"..."}')
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    pipe = EntityABPipeline(cfg)

    rows = read_jsonl(args.input_jsonl)
    os.makedirs(args.out_dir, exist_ok=True)

    # Prepare per-model outputs
    per_model = {name: [] for name in pipe.extractors.keys()}
    per_model_meta = []

    for row in rows:
        doc_id = str(row.get("id", ""))
        text = str(row.get("text", ""))
        out = pipe.extract_all(text)
        per_model_meta.append({
            "id": doc_id,
            "text": text,
            "latency_ms": {k: v.latency_ms for k, v in out.items()}
        })
        for name, r in out.items():
            per_model[name].append({
                "id": doc_id,
                "text": text,
                "entities": [e.to_dict() for e in r.entities]
            })

    # Save
    for name, preds in per_model.items():
        write_jsonl(os.path.join(args.out_dir, f"pred_{name}.jsonl"), preds)

    write_jsonl(os.path.join(args.out_dir, "latency.jsonl"), per_model_meta)

    print("Saved outputs:")
    for name in per_model.keys():
        print(" -", os.path.join(args.out_dir, f"pred_{name}.jsonl"))
    print(" -", os.path.join(args.out_dir, "latency.jsonl"))

if __name__ == "__main__":
    main()
