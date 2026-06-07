"""Structured output for workflow child agents."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import threading
from typing import Any, Callable, Iterable, Iterator

from ..core.schema import validate_json_schema, validate_schema
from ..core.tool_errors import tool_error

STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"
STRUCTURED_OUTPUT_TOOLSET = "workflow_structured"
STRUCTURED_OUTPUT_SUCCESS = "Structured output provided successfully"
STRUCTURED_OUTPUT_CONTINUE_MESSAGE = (
    "You MUST call the structured_output tool to complete this request. "
    "Call this tool now."
)
MAX_STRUCTURED_OUTPUT_RETRIES = 5

# The registered schema is only a process-global placeholder. Each workflow
# child gets an instance-local copy whose parameters are replaced with the
# schema passed to agent(..., {schema}).
STRUCTURED_OUTPUT_TOOL_SCHEMA = {
    "description": "Return structured output in the requested format",
    "parameters": {
        "type": "object",
        "additionalProperties": True,
        "properties": {},
    },
}


def build_tool_schema_instruction() -> str:
    return (
        f"\n\nUse the {STRUCTURED_OUTPUT_TOOL_NAME} tool to return your final response "
        "in the requested structured format. You MUST call this tool exactly once "
        "at the end of your response to provide the structured output."
    )


def specialize_structured_output_tool(
    tools: list[dict[str, Any]] | None,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return child-local tool definitions with the target schema installed."""
    validate_json_schema(schema)
    specialized = list(tools or [])
    for index, definition in enumerate(specialized):
        function = definition.get("function") if isinstance(definition, dict) else None
        if not isinstance(function, dict) or function.get("name") != STRUCTURED_OUTPUT_TOOL_NAME:
            continue
        replacement = deepcopy(definition)
        replacement_function = replacement["function"]
        replacement_function["description"] = "Return structured output in the requested format"
        replacement_function["parameters"] = deepcopy(schema)
        specialized[index] = replacement
        return specialized
    raise RuntimeError("structured_output tool is not available to the workflow child")


class _StructuredOutputBroker:
    """Thread-safe per-child store of expected schemas and accepted results."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._expect: dict[str, dict[str, Any]] = {}
        self._results: dict[str, Any] = {}
        self._attempts: dict[str, int] = {}
        self._on_exhausted: dict[str, Callable[[], Any]] = {}
        self._exhausted: set[str] = set()

    def register(
        self,
        task_id: str,
        schema: dict[str, Any],
        on_exhausted: Callable[[], Any] | None = None,
    ) -> None:
        if not task_id:
            return
        with self._lock:
            self._expect[task_id] = schema
            self._attempts[task_id] = 0
            self._results.pop(task_id, None)
            self._exhausted.discard(task_id)
            if on_exhausted is None:
                self._on_exhausted.pop(task_id, None)
            else:
                self._on_exhausted[task_id] = on_exhausted

    def submit(self, task_id: str, value: Any) -> tuple[bool, str]:
        with self._lock:
            schema = self._expect.get(task_id)
            attempts = self._attempts.get(task_id, 0) + 1
            self._attempts[task_id] = attempts
        if schema is None:
            return False, "root: no structured-output expectation is registered for this task"
        if attempts > MAX_STRUCTURED_OUTPUT_RETRIES:
            return False, (
                "root: maximum structured output attempts exceeded "
                f"({MAX_STRUCTURED_OUTPUT_RETRIES})"
            )

        errors = _validation_errors(value, schema)
        if errors:
            if attempts >= MAX_STRUCTURED_OUTPUT_RETRIES:
                self._interrupt_exhausted(task_id)
            return False, ", ".join(errors)

        with self._lock:
            self._results[task_id] = value
        return True, ""

    def peek(self, task_id: str) -> tuple[bool, Any, int]:
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
            self._on_exhausted.pop(task_id, None)
            self._exhausted.discard(task_id)

    def _interrupt_exhausted(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._exhausted:
                return
            self._exhausted.add(task_id)
            callback = self._on_exhausted.get(task_id)
        if callback is not None:
            try:
                callback()
            except Exception:
                pass


_BROKER = _StructuredOutputBroker()
_REGISTRY_LOCK = threading.RLock()
_REGISTRY_USERS = 0
_REGISTRY_OWNED = False


@contextmanager
def structured_output_tool_scope() -> Iterator[None]:
    """Expose the internal tool only while workflow schema children need it."""
    global _REGISTRY_USERS, _REGISTRY_OWNED

    registry = _tool_registry()
    with _REGISTRY_LOCK:
        if _REGISTRY_USERS == 0:
            existing = registry.get_entry(STRUCTURED_OUTPUT_TOOL_NAME)
            if existing is None:
                registry.register(
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    toolset=STRUCTURED_OUTPUT_TOOLSET,
                    schema=STRUCTURED_OUTPUT_TOOL_SCHEMA,
                    handler=structured_output_handler,
                    description="Return structured output in the requested format.",
                )
                existing = registry.get_entry(STRUCTURED_OUTPUT_TOOL_NAME)
                _REGISTRY_OWNED = True
            else:
                _REGISTRY_OWNED = False
            if existing is None or existing.handler is not structured_output_handler:
                raise RuntimeError(
                    "structured_output tool name is already registered by another tool"
                )
        _REGISTRY_USERS += 1

    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            _REGISTRY_USERS = max(0, _REGISTRY_USERS - 1)
            if _REGISTRY_USERS == 0 and _REGISTRY_OWNED:
                registry.deregister(STRUCTURED_OUTPUT_TOOL_NAME)
                _REGISTRY_OWNED = False


def register_expectation(
    task_id: str,
    schema: dict[str, Any],
    on_exhausted: Callable[[], Any] | None = None,
) -> None:
    _BROKER.register(task_id, schema, on_exhausted)


def peek_result(task_id: str) -> tuple[bool, Any, int]:
    return _BROKER.peek(task_id)


def clear_expectation(task_id: str) -> None:
    _BROKER.clear(task_id)


def structured_output_handler(args: Any, *, task_id: str | None = None, **_: Any) -> str:
    """Validate and capture a workflow child's final structured value."""
    ok, error = _BROKER.submit(str(task_id or ""), args)
    if ok:
        return STRUCTURED_OUTPUT_SUCCESS
    return tool_error(f"Output does not match required schema: {error}")


def _tool_registry() -> Any:
    from tools.registry import registry

    return registry


def _validation_errors(value: Any, schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(value),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                str(error.validator),
                error.message,
            ),
        )
        return [_format_validation_error(error) for error in errors]
    except ImportError:
        pass
    except Exception as exc:
        return [f"root: {getattr(exc, 'message', str(exc))}"]

    try:
        validate_schema(value, schema)
    except Exception as exc:
        message = getattr(exc, "message", str(exc))
        prefix = "structured output did not match schema: "
        if message.startswith(prefix):
            message = message[len(prefix) :]
        return [_format_fallback_error(message)]
    return []


def _format_validation_error(error: Any) -> str:
    path = _json_pointer(error.absolute_path)
    validator = str(getattr(error, "validator", "") or "")
    validator_value = getattr(error, "validator_value", None)

    if validator == "required":
        missing = _missing_required_property(error.message)
        if missing:
            return f"{path}: must have required property '{missing}'"
    if validator == "type":
        expected = validator_value
        if isinstance(expected, list):
            expected = ",".join(str(item) for item in expected)
        return f"{path}: must be {expected}"
    if validator == "additionalProperties":
        return f"{path}: must NOT have additional properties"
    if validator == "enum":
        return f"{path}: must be equal to one of the allowed values"
    if validator == "const":
        return f"{path}: must be equal to constant"
    if validator == "minItems":
        return f"{path}: must NOT have fewer than {validator_value} items"
    if validator == "maxItems":
        return f"{path}: must NOT have more than {validator_value} items"
    if validator == "minLength":
        return f"{path}: must NOT have fewer than {validator_value} characters"
    if validator == "maxLength":
        return f"{path}: must NOT have more than {validator_value} characters"
    if validator == "minimum":
        return f"{path}: must be >= {validator_value}"
    if validator == "maximum":
        return f"{path}: must be <= {validator_value}"
    if validator == "pattern":
        return f'{path}: must match pattern "{validator_value}"'
    return f"{path}: {error.message}"


def _json_pointer(path: Iterable[Any]) -> str:
    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in path]
    return "/" + "/".join(parts) if parts else "root"


def _missing_required_property(message: str) -> str:
    if not message.startswith("'"):
        return ""
    _, _, remainder = message.partition("'")
    return remainder.partition("'")[0]


def _format_fallback_error(message: str) -> str:
    if message.startswith("$: "):
        message = message[3:]
    if message.startswith("missing required key "):
        return f"root: must have required property {message.removeprefix('missing required key ')}"
    if message.startswith("unexpected key "):
        return "root: must NOT have additional properties"
    if message.startswith("expected type "):
        return f"root: must be {message.removeprefix('expected type ')}"
    return message if message.startswith(("root:", "/")) else f"root: {message}"
