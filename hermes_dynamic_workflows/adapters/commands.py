"""Slash commands for workflow run inspection and control."""

from __future__ import annotations

from ..engine.manager import get_run_manager


def workflows_command(raw_args: str = "") -> str:
    arg = (raw_args or "").strip()
    if arg:
        return "Usage: /workflows\nFor live monitoring and controls, run `hermes-workflows` in a terminal."
    return get_run_manager().format_agent_overview(limit=12)
