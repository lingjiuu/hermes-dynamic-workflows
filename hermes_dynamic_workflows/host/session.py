"""Host port: Hermes session store (``hermes_state``) and home (``hermes_constants``).

The only module that imports those host packages. Callers keep their own
try/except + fallback around these (a faithful relocation of the previously
inline ``from hermes_state import SessionDB`` / ``from hermes_constants import
get_hermes_home`` calls), so behavior is unchanged.
"""

from __future__ import annotations

from typing import Any


def create_session_db() -> Any:
    """Return a fresh Hermes ``SessionDB``. Raises if ``hermes_state`` is absent."""
    from hermes_state import SessionDB

    return SessionDB()


def hermes_home() -> Any:
    """Return Hermes' home directory. Raises if ``hermes_constants`` is absent."""
    from hermes_constants import get_hermes_home

    return get_hermes_home()
