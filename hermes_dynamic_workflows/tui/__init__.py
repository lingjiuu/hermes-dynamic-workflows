"""Standalone terminal UI for dynamic workflow runs.

The `hermes-workflows` console app: a self-contained curses monitor (model +
render + app loop) that reads persisted run snapshots, journals, and live child
transcripts. Depends on ``storage`` for data; separate from the in-session
``view`` renderers.
"""
