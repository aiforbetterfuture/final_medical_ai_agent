# memory/ (compat shim)

This folder exists only to satisfy imports like:

- `from memory.profile_store import ProfileStore`

The real code should live under `agent/memory/`.

If your real memory code is elsewhere, edit `memory/__init__.py`
and change `AGENT_MEMORY_PREFIX`.
