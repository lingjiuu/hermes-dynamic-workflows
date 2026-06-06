"""Hermes-native tool error helpers."""

from __future__ import annotations

import json


def tool_error(message: str) -> str:
    """Return a Hermes-style tool error result."""
    return json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":"))
