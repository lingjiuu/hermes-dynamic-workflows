"""Parse Claude-style token targets from the user's current task."""

from __future__ import annotations

import re


_ASCII_WORD_BOUNDARY = r"(?![A-Za-z0-9_])"
_SHORTHAND_START_RE = re.compile(
    rf"^\s*\+(\d+(?:\.\d+)?)\s*(k|m|b){_ASCII_WORD_BOUNDARY}",
    re.IGNORECASE,
)
_SHORTHAND_END_RE = re.compile(
    r"\s\+(\d+(?:\.\d+)?)\s*(k|m|b)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_VERBOSE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:use|spend)\s+(\d+(?:\.\d+)?)\s*(k|m|b)"
    rf"\s*tokens?{_ASCII_WORD_BOUNDARY}",
    re.IGNORECASE,
)
_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


def parse_token_budget(text: str | None) -> int | None:
    """Return the user's Claude-style token target, or ``None`` when absent."""
    if not text:
        return None
    for pattern in (_SHORTHAND_START_RE, _SHORTHAND_END_RE, _VERBOSE_RE):
        match = pattern.search(str(text))
        if match:
            return int(float(match.group(1)) * _MULTIPLIERS[match.group(2).lower()])
    return None
