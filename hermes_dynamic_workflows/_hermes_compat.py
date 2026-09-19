"""Resolve Hermes core symbols that moved in the 2026-09-14 decomposition.

Hermes core relocated a number of modules (approval session context, the smart
approval classifier, the terminal VM lifecycle). Two things follow from that:

* Importing the *old* path still works for some symbols through a re-export
  facade, but it raises ``HermesPluginCompatWarning`` -- and Hermes refuses to
  load a plugin that does it, so the plugin silently never comes up.
* For other symbols the old path is gone outright (``AttributeError``).

Both problems disappear if every symbol is resolved from the module that owns
it. This module keeps one import path working on either side of the move: it
tries the current module first and falls back to the legacy one, so the plugin
loads on a post-decomposition core *and* keeps working on older cores instead
of pinning itself to one of them.

Resolution happens per call on purpose. Binding the names at import time would
freeze whichever module happened to be importable first -- including under a
test's ``patch.dict(sys.modules, ...)``, where an early binding escapes the
mock entirely. Call ``resolve()`` at the point of use:

    from hermes_dynamic_workflows._hermes_compat import resolve

    try:
        mode = resolve("_get_approval_mode")()
    except Exception:
        return "deny"
"""

from __future__ import annotations

import importlib
from typing import Any

# symbol -> candidate modules, current first, legacy last
_MOVED: dict[str, tuple[str, ...]] = {
    # approval session context
    "set_current_session_key": ("tools.approval_context", "tools.approval"),
    "reset_current_session_key": ("tools.approval_context", "tools.approval"),
    "get_current_session_key": ("tools.approval_context", "tools.approval"),
    "_get_approval_mode": ("tools.approval_context", "tools.approval"),
    "_is_gateway_approval_context": ("tools.approval_context", "tools.approval"),
    # smart approval classifier
    "_smart_approve": ("tools.approval_smart", "tools.approval"),
    # terminal VM lifecycle
    "cleanup_vm": ("tools.terminal_tool_lifecycle", "tools.terminal_tool"),
}


def resolve(name: str) -> Any:
    """Return ``name`` from the first candidate module that provides it.

    Raises ``ImportError`` when no candidate provides it, and ``AttributeError``
    for a name that is not part of the moved set at all -- callers wrap this in
    the same ``try/except`` they used around the plain import.
    """
    try:
        candidates = _MOVED[name]
    except KeyError:
        raise AttributeError(f"{name!r} is not a relocated Hermes symbol") from None

    problems: list[str] = []
    for module_name in candidates:
        try:
            return getattr(importlib.import_module(module_name), name)
        except (ImportError, AttributeError) as exc:
            problems.append(f"{module_name}.{name}: {exc}")
    raise ImportError(f"cannot resolve {name!r} from {candidates}: {'; '.join(problems)}")
