"""Hermes tool handler for workflow."""

from __future__ import annotations

import json
import os
import traceback
from typing import Any

from ..engine.manager import get_run_manager


def workflow(params: dict[str, Any], *, plugin_context: Any = None, **_: Any) -> str:
    try:
        manager = get_run_manager()
        record = manager.start_from_params(
            params or {},
            cwd=os.environ.get("TERMINAL_CWD") or os.getcwd(),
            plugin_context=plugin_context,
        )
        return json.dumps(
            {
                "content": (
                    f"Workflow started: {record['runId']}. "
                    "Use /workflows to list runs or /workflows "
                    f"{record['runId']} to inspect progress."
                ),
                "runId": record["runId"],
                "status": record.get("status"),
                "scriptPath": record.get("scriptPath"),
                "source": record.get("source"),
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as exc:
        return json.dumps(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "trace": _short_traceback(),
            },
            ensure_ascii=False,
        )


def _short_traceback() -> str:
    lines = traceback.format_exc(limit=4).strip().splitlines()
    return "\n".join(lines[-8:])
