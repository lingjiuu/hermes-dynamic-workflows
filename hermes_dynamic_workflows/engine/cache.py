"""Agent-call cache used by resumeFromRunId."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class ResumeCache:
    def __init__(self, previous: dict[str, Any] | None = None):
        self.previous = previous or {}
        self.current: dict[str, Any] = {}
        self._prefix_open = True

    @classmethod
    def from_run(cls, run_record: dict[str, Any] | None) -> "ResumeCache":
        if not run_record:
            return cls()
        cache = run_record.get("agentCache")
        return cls(cache if isinstance(cache, dict) else {})

    def get(self, sequence: int, fingerprint: str) -> Any:
        if not self._prefix_open:
            return _MISS
        key = str(sequence)
        item = self.previous.get(key)
        if not isinstance(item, dict) or item.get("fingerprint") != fingerprint:
            self._prefix_open = False
            return _MISS
        return item.get("result")

    def put(self, sequence: int, fingerprint: str, result: Any) -> None:
        self.current[str(sequence)] = {
            "fingerprint": fingerprint,
            "result": _jsonable(result),
        }


class _Miss:
    pass


_MISS = _Miss()


def is_cache_miss(value: Any) -> bool:
    return isinstance(value, _Miss)


def agent_fingerprint(prompt: str, opts: dict[str, Any]) -> str:
    payload = {
        "prompt": prompt,
        "opts": _jsonable(opts),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)
