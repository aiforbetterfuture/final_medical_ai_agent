#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 /path/to/UMLS_INSTALL /path/to/output_index_dir"
  exit 1
fi

UMLS_DIR="$1"
OUT_DIR="$2"

python -m quickumls.install "$UMLS_DIR" "$OUT_DIR" -d unqlite -U
echo "QuickUMLS index built at: $OUT_DIR"
