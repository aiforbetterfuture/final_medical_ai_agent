"""Top-level `memory` compatibility package.

Some parts of this repository import modules like:
    from memory.profile_store import ProfileStore

In the current scaffold, the real implementation is expected to live under:
    agent/memory/

This shim keeps older absolute imports working without touching every file.
If your real implementation is NOT under agent.memory, change
AGENT_MEMORY_PREFIX below.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

AGENT_MEMORY_PREFIX = "agent.memory"

def _forward(module: str) -> Any:
    """Import and return a module from the real memory package."""
    try:
        return import_module(f"{AGENT_MEMORY_PREFIX}.{module}")
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"Failed to import '{AGENT_MEMORY_PREFIX}.{module}'. "
            "Fix by either: (1) creating agent/memory/ modules, or "
            "(2) updating AGENT_MEMORY_PREFIX in memory/__init__.py."
        ) from e
