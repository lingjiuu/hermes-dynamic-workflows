"""Slash commands for workflow run inspection and control."""

from __future__ import annotations

import os
from typing import Any

from ..run.manager import get_run_manager


def workflows_command(raw_args: str = "", *, plugin_context: Any = None) -> str:
    arg = (raw_args or "").strip()
    if arg:
        return "Usage: /workflows\nFor live monitoring and controls, run `hermes-workflows` in a terminal."
    return get_run_manager().format_agent_overview(
        limit=12,
        session_id=_current_session_id(plugin_context) or None,
    )


def _current_session_id(plugin_context: Any = None) -> str:
    for attr in ("session_id", "sessionId"):
        value = getattr(plugin_context, attr, None) if plugin_context is not None else None
        if value:
            return str(value)
    for method_name in ("get_session_id", "current_session_id"):
        method = getattr(plugin_context, method_name, None) if plugin_context is not None else None
        if callable(method):
            try:
                value = method()
            except Exception:
                value = None
            if value:
                return str(value)

    cli_ref = _plugin_context_cli_ref(plugin_context)
    for value in (
        getattr(getattr(cli_ref, "agent", None), "session_id", None),
        getattr(cli_ref, "session_id", None),
    ):
        if value:
            return str(value)

    for name in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
        value = _session_env(name)
        if value:
            return value
    return ""


def _plugin_context_cli_ref(plugin_context: Any) -> Any:
    manager = getattr(plugin_context, "_manager", None) if plugin_context is not None else None
    if manager is not None:
        return getattr(manager, "_cli_ref", None)
    return None


def _session_env(name: str) -> str:
    try:
        from ..host import gateway as host_gateway

        value = str(host_gateway.raw_session_env(name, "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, "").strip()
