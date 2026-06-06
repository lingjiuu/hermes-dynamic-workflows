"""Claude-style background task stop tool for workflow runs."""

from __future__ import annotations

import json
from typing import Any

from ..engine.manager import get_run_manager
from ..core.tool_errors import tool_error

_DESCRIPTION = """
- Stop a running workflow by its Task ID
- Takes a task_id parameter identifying the workflow task to stop
- Returns a success or failure status
- Use this tool when you need to terminate a workflow task
"""

TASK_STOP_SCHEMA = {
    "description": _DESCRIPTION,
    "parameters": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the workflow task to stop",
            }
        },
        "required": ["task_id"],
    },
}


def task_stop(params: dict[str, Any] | None, **_: Any) -> str:
    task_id = ""
    if isinstance(params, dict):
        task_id = str(params.get("task_id") or "").strip()
    if not task_id:
        return tool_error("Missing required parameter: task_id")

    result = get_run_manager().stop_task(task_id)
    if result is None:
        return tool_error(f"No task found with ID: {task_id}")
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
