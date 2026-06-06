"""Host port: Hermes gateway session context (``gateway.*``).

The only module that imports ``gateway.session_context`` / ``gateway.run``.
Raw pass-throughs: callers keep their existing strip/fallback/try-except so the
relocation is behavior-preserving. Used to inherit the launching agent's
runtime and to route a child's mid-run approval back to the originating user.
"""

from __future__ import annotations

from typing import Any


def raw_session_env(name: str, default: str = "") -> str:
    """Read a gateway session env var. Raises if ``gateway`` is unavailable."""
    from gateway.session_context import get_session_env

    return get_session_env(name, default)


def set_session_vars(**values: str) -> None:
    """Re-apply launching gateway session vars on the current worker thread."""
    from gateway.session_context import set_session_vars as _set

    _set(**values)


def gateway_runner_ref() -> Any:
    """Return the gateway runner singleton. Raises if ``gateway.run`` is absent."""
    from gateway.run import _gateway_runner_ref

    return _gateway_runner_ref()
