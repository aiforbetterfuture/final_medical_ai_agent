"""Forwarder for `memory.profile_store`.

Exposes ProfileStore from agent.memory.profile_store.
"""

from __future__ import annotations

from importlib import import_module

_mod = import_module("agent.memory.profile_store")
ProfileStore = getattr(_mod, "ProfileStore")
__all__ = ["ProfileStore"]
