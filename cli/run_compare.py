import argparse
import os
from dotenv import load_dotenv

from med_entity_ab.pipeline import load_config, EntityABPipeline
from med_entity_ab.utils.io import write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out_json", type=str, default="outputs/compare_single.json")
    args = ap.parse_args()

    load_dotenv()

    cfg = load_config(args.config)
    pipe = EntityABPipeline(cfg)
    out = pipe.extract_all(args.text)

    payload = {
        "text": args.text,
        "results": {
            name: {
                "latency_ms": r.latency_ms,
                "entities": [e.to_dict() for e in r.entities]
            }
            for name, r in out.items()
        }
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    write_json(args.out_json, payload)

    # pretty print
    try:
        import pandas as pd
        from rich import print as rprint
        rprint(f"\n[bold]Input:[/bold] {args.text}\n")
        for name, r in out.items():
            rprint(f"[bold]=== {name} (latency {r.latency_ms:.1f}ms) ===[/bold]")
            df = pd.DataFrame([e.to_dict() for e in r.entities])
            rprint(df if not df.empty else "(no entities)")
            rprint("")
    except Exception:
        pass

    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
