
from __future__ import annotations
from pathlib import Path
import argparse
from collections import Counter

from common import load_yaml, iter_files_recursive, is_json_file, is_zip_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    lp = cfg["local_paths"]

    def scan_dirs(label: str, dirs: list[str]):
        print(f"\n[{label}]")
        total_json = 0
        total_zip = 0
        total_files = 0
        missing = []
        for d in dirs:
            p = Path(d)
            if not p.exists():
                missing.append(d)
                continue
            for f in iter_files_recursive(p):
                total_files += 1
                if is_json_file(f):
                    total_json += 1
                elif is_zip_file(f):
                    total_zip += 1
        print(f"- dirs: {len(dirs)}")
        print(f"- files: {total_files:,}")
        print(f"- json:  {total_json:,}")
        print(f"- zip:   {total_zip:,}")
        if missing:
            print("\n! Missing paths:")
            for m in missing:
                print("  -", m)

    scan_dirs("TS", lp["TS_dirs"])
    scan_dirs("TL", lp["TL_dirs"])
    scan_dirs("VL", lp["VL_dirs"])

if __name__ == "__main__":
    main()
