This drop-in patch fixes:
1) LangGraph error: "No synchronous function provided to 'retrieve'" by wrapping retrieve_node with a sync adapter.
2) grade_run pipeline error: removes thread_id kwarg dependency and calls agent.graph.run_agent with signature-compat mapping.

Files:
- agent/graph.py
- tools/grade_run.py
