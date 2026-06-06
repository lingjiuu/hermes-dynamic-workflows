"""Background workflow run orchestration.

The run manager (lifecycle: start/stop/pause/resume/restart, threading,
persistence, notifications) and the live child-transcript subsystem. Drives
``engine.run_workflow`` and depends on ``child``/``storage``/``view``/``host``;
nothing here is imported by ``engine`` or any lower layer.
"""
