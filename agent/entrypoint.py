from __future__ import annotations

"""Stable agent entrypoint.

Use from tooling scripts:
    from agent.entrypoint import run_agent

Rationale
- Scripts executed under `tools/` are often launched as `python tools/<script>.py`.
  On Windows, that can make `import agent` fragile because `sys.path[0]` becomes
  the `tools/` folder.
- Cursor/Claude edits sometimes change the API of `agent.graph`.

This module provides a stable import path and a compatibility wrapper.

Behavior
- Prefer `agent.graph.run_agent` if present.
- Otherwise, fall back to building the graph via `agent.graph.get_agent_graph()`
  and invoking it.

Note
- The fallback path is best-effort: it tries to find common state keys.
"""

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def _ensure_repo_root_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _call_compat(fn, question: str, *, session_id: Optional[str], model_mode: str,
                return_state: bool, debug: bool,
                feature_overrides: Optional[Dict[str, Any]] = None,
                feature_flags: Optional[Dict[str, Any]] = None,
                **extra: Any):
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}

    # session/thread id
    if session_id is not None:
        if 'session_id' in sig.parameters:
            kwargs['session_id'] = session_id
        elif 'thread_id' in sig.parameters:
            kwargs['thread_id'] = session_id

    # model mode
    if 'model_mode' in sig.parameters:
        kwargs['model_mode'] = model_mode

    # state return
    if 'return_state' in sig.parameters:
        kwargs['return_state'] = return_state

    # debug
    if 'debug' in sig.parameters:
        kwargs['debug'] = debug

    # feature overrides/flags
    if feature_overrides is not None and 'feature_overrides' in sig.parameters:
        kwargs['feature_overrides'] = feature_overrides
    if feature_flags is not None and 'feature_flags' in sig.parameters:
        kwargs['feature_flags'] = feature_flags

    # any additional supported kwargs
    for k, v in extra.items():
        if k in sig.parameters:
            kwargs[k] = v

    return fn(question, **kwargs)


def run_agent(
    question: str,
    *,
    session_id: Optional[str] = None,
    model_mode: str = 'default',
    return_state: bool = False,
    debug: bool = False,
    feature_overrides: Optional[Dict[str, Any]] = None,
    feature_flags: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Run the agent with a stable API.

    Returns:
      - answer (str) when return_state=False
      - (answer, state_dict) when return_state=True
    """
    _ensure_repo_root_on_path()

    import importlib

    g = importlib.import_module('agent.graph')

    # 1) Preferred path: graph.run_agent
    if hasattr(g, 'run_agent') and callable(getattr(g, 'run_agent')):
        fn = getattr(g, 'run_agent')
        return _call_compat(
            fn,
            question,
            session_id=session_id,
            model_mode=model_mode,
            return_state=return_state,
            debug=debug,
            feature_overrides=feature_overrides,
            feature_flags=feature_flags,
            **extra,
        )

    # 2) Fallback path: build graph via get_agent_graph() and invoke
    if not hasattr(g, 'get_agent_graph') or not callable(getattr(g, 'get_agent_graph')):
        raise ImportError('agent.graph has no run_agent and no get_agent_graph')

    get_graph = getattr(g, 'get_agent_graph')

    # get_agent_graph may or may not accept args
    try:
        graph_obj = _call_compat(
            get_graph,
            '',
            session_id=None,
            model_mode=model_mode,
            return_state=False,
            debug=debug,
            feature_overrides=feature_overrides,
            feature_flags=feature_flags,
            **extra,
        )
    except TypeError:
        graph_obj = get_graph()

    # build a minimal initial state
    initial_state: Dict[str, Any] = {
        'user_text': question,
        'query': question,
    }
    if session_id is not None:
        initial_state['session_id'] = session_id
        initial_state['thread_id'] = session_id

    if feature_overrides is not None:
        initial_state['feature_overrides'] = feature_overrides
    if feature_flags is not None:
        initial_state['feature_flags'] = feature_flags

    final_state = graph_obj.invoke(initial_state)

    # guess answer key
    answer = (
        final_state.get('final_answer')
        or final_state.get('answer')
        or final_state.get('response')
        or ''
    )

    if return_state:
        return answer, final_state
    return answer
