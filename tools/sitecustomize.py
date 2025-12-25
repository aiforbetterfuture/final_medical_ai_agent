"""Auto-add repo root to sys.path when running scripts from tools/.

Python automatically tries to import `sitecustomize` on startup (unless `-S` is
used). Because `python tools/xxx.py` sets sys.path[0] = tools/, placing this
file here lets us fix imports without requiring users to set PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
