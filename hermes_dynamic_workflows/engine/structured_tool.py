"""Tool-channel structured output for workflow child agents.

Claude Code's ``agent(prompt, {schema})`` forces the subagent to emit its
answer through a StructuredOutput tool call that is validated at the tool
layer, so the model retries on a schema mismatch instead of the orchestrator
parsing free text after the fact.

This module is the Hermes analogue. A single global tool
(``workflow_submit_structured_output``) is registered in its own toolset. When
``agent()`` is called with a schema in tool mode, the child agent gets that
toolset and is told to call the tool with its final answer. The handler
validates the submitted value against the per-call schema (routed by the
child's ``task_id``) and either records it or returns the validation error so
the model corrects itself and calls again.

The orchestrator reads the recorded value back through the child runner's
metadata. If the model never calls the tool, the caller falls back to parsing
the child's final text, so this is strictly additive over the prompt path.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from .structured import validate_schema

STRUCTURED_OUTPUT_TOOL_NAME = "workflow_submit_structured_output"
STRUCTURED_OUTPUT_TOOLSET = "workflow_structured"

STRUCTURED_OUTPUT_TOOL_SCHEMA = {
    "description": (
        "Submit the final structured answer for a dynamic-workflow child agent. "
        "Call this exactly once when you are done, passing your complete final "
        "answer as the `result` argument. `result` is validated against the JSON "
        "Schema given in your task instructions; if validation fails you receive "
        "the error and must call again with a corrected `result`."
    ),
    "parameters": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "result": {
                "description": (
                    "The final answer value. Must validate against the JSON Schema "
                    "from the task instructions (object, array, or scalar as required)."
                )
            }
        },
        "required": ["result"],
    },
}


def build_tool_schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "\n\nStructured output required:\n"
        f"- When finished, call the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool exactly once.\n"
        "- Pass your complete final answer as the `result` argument; `result` must "
        "validate against the JSON Schema below.\n"
        "- Do not put the final answer in your normal message; put it only in the tool call.\n"
        "- If the tool rejects your answer, read the error and call it again with a corrected `result`.\n"
        "- If you cannot call the tool at all, then return exactly one JSON value matching the "
        "schema as your message instead.\n\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


class _StructuredOutputBroker:
    """Thread-safe per-task store of expected schemas and submitted results.

    Concurrent child agents each run under a distinct ``task_id``, so the
    schema and captured result are keyed by task to keep parallel agents from
    crossing wires through this one global tool.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._expect: dict[str, dict[str, Any]] = {}
        self._results: dict[str, Any] = {}
        self._attempts: dict[str, int] = {}

    def register(self, task_id: str, schema: dict[str, Any]) -> None:
        if not task_id:
            return
        with self._lock:
            self._expect[task_id] = schema
            self._attempts[task_id] = 0
            self._results.pop(task_id, None)

    def submit(self, task_id: str, value: Any) -> tuple[bool, str]:
        with self._lock:
            schema = self._expect.get(task_id)
            self._attempts[task_id] = self._attempts.get(task_id, 0) + 1
        if schema is None:
            return False, "no structured-output expectation is registered for this task"
        try:
            validate_schema(value, schema)
        except Exception as exc:
            return False, getattr(exc, "message", str(exc))
        with self._lock:
            self._results[task_id] = value
        return True, ""

    def pop(self, task_id: str) -> tuple[bool, Any, int]:
        with self._lock:
            attempts = self._attempts.get(task_id, 0)
            if task_id in self._results:
                return True, self._results[task_id], attempts
            return False, None, attempts

    def clear(self, task_id: str) -> None:
        with self._lock:
            self._expect.pop(task_id, None)
            self._results.pop(task_id, None)
            self._attempts.pop(task_id, None)


_BROKER = _StructuredOutputBroker()


def register_expectation(task_id: str, schema: dict[str, Any]) -> None:
    _BROKER.register(task_id, schema)


def pop_result(task_id: str) -> tuple[bool, Any, int]:
    return _BROKER.pop(task_id)


def clear_expectation(task_id: str) -> None:
    _BROKER.clear(task_id)


def submit_structured_output_handler(args: Any, *, task_id: str | None = None, **_: Any) -> str:
    """Registry handler for ``workflow_submit_structured_output``.

    Validates ``args["result"]`` against the schema registered for the calling
    child's ``task_id``. On success the value is recorded for the orchestrator
    to read back; on failure the validation error is returned so the model can
    correct and call again.
    """
    if not isinstance(args, dict) or "result" not in args:
        return json.dumps(
            {"status": "rejected", "error": "missing required `result` argument"},
            ensure_ascii=False,
        )
    ok, error = _BROKER.submit(str(task_id or ""), args["result"])
    if ok:
        return json.dumps(
            {
                "status": "accepted",
                "message": (
                    "Structured output recorded. Reply with a short confirmation; "
                    "do not repeat the JSON."
                ),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "rejected",
            "error": error,
            "hint": f"Call {STRUCTURED_OUTPUT_TOOL_NAME} again with a corrected `result`.",
        },
        ensure_ascii=False,
        default=str,
    )
