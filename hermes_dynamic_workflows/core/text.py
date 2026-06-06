"""Small text helpers shared across layers.

``preview`` lives in ``core`` (not in the presentation layer) so the execution
engine can truncate prompts/results for snapshots without importing ``view``.
"""

from __future__ import annotations

from typing import Any


def preview(value: Any, max_chars: int = 160) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "..."
