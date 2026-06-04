"""Holds the live Hermes PluginContext so saved workflows can register a
``/<name>`` slash command at save time, not just at plugin load.

Hermes resolves plugin slash commands by reading the manager's command dict
live (``get_plugin_command_handler``), so a command registered after load is
dispatchable immediately in the same session.
"""

from __future__ import annotations

from typing import Any, Callable

_CTX: Any = None


def set_plugin_context(ctx: Any) -> None:
    global _CTX
    _CTX = ctx


def has_context() -> bool:
    return _CTX is not None


def register_command(
    name: str,
    handler: Callable[..., Any],
    *,
    description: str = "",
    args_hint: str = "",
) -> bool:
    """Register a slash command through the captured context.

    Returns True if the context was present and the call did not raise. The
    underlying ``ctx.register_command`` silently skips names that collide with
    built-in commands, so a True return does not guarantee the command is live
    — callers should phrase user messaging accordingly.
    """
    if _CTX is None:
        return False
    try:
        _CTX.register_command(
            name=name,
            handler=handler,
            description=description,
            args_hint=args_hint,
        )
        return True
    except Exception:
        return False
