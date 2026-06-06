"""Workflow execution engine.

Runs a validated workflow script: the script runtime, per-run execution context
(token/agent accounting, concurrency, deadlines), the agent()/parallel()/
pipeline() API surface, the AST sandbox, and the resume cache. Depends on
``core`` and the ``child`` execution primitives; never on ``run`` or ``adapters``.
"""
