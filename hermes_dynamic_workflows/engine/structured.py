"""Structured output helpers for workflow child agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .errors import ChildAgentError


class StructuredOutputError(ChildAgentError):
    """Raised when child-agent output cannot satisfy a requested JSON schema."""


@dataclass(frozen=True)
class StructuredAttempt:
    status: str
    mode: str
    attempts: int
    raw_preview: str = ""
    error: str = ""
    repaired: bool = False

    def snapshot(self) -> dict[str, Any]:
        data = {
            "status": self.status,
            "mode": self.mode,
            "attempts": self.attempts,
            "repaired": self.repaired,
        }
        if self.raw_preview:
            data["raw_preview"] = self.raw_preview
        if self.error:
            data["error"] = self.error
        return data


def build_schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "\n\nStructured output required:\n"
        "- Return exactly one JSON value that validates against the JSON Schema below.\n"
        "- Do not include prose, markdown fences, comments, or trailing text.\n"
        "- Preserve the schema's required keys and primitive value types.\n\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


def build_response_format_overrides(schema: dict[str, Any], *, name: str = "workflow_structured_output") -> dict[str, Any]:
    return {
        "extra_body": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _safe_schema_name(name),
                    "schema": schema,
                    "strict": False,
                },
            }
        }
    }


def parse_structured_output(value: Any, schema: dict[str, Any]) -> Any:
    parsed = value if not isinstance(value, str) else parse_jsonish(value)
    validate_schema(parsed, schema)
    return parsed


def parse_jsonish(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise StructuredOutputError("child agent returned empty structured output")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise StructuredOutputError("child agent did not return valid JSON")


def validate_schema(value: Any, schema: dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore[import-untyped]

        jsonschema.validate(value, schema)
        return
    except ImportError:
        pass
    except Exception as exc:
        message = getattr(exc, "message", str(exc))
        raise StructuredOutputError(f"structured output did not match schema: {message}") from exc

    _validate_fallback(value, schema, path="$")


def build_repair_prompt(raw_text: Any, error: str, schema: dict[str, Any]) -> str:
    raw = raw_text if isinstance(raw_text, str) else json.dumps(raw_text, ensure_ascii=False, default=str)
    return (
        "Repair the child agent's final answer into valid JSON only.\n"
        "Return exactly one JSON value. Do not include prose or markdown fences.\n\n"
        f"Validation error:\n{error}\n\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n\n"
        "Original answer:\n"
        f"{raw}"
    )


def looks_like_response_format_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "response_format" not in text and "json_schema" not in text:
        return False
    markers = (
        "unsupported",
        "not supported",
        "unknown",
        "unrecognized",
        "invalid",
        "400",
        "bad request",
        "extra_body",
    )
    return any(marker in text for marker in markers)


def _validate_fallback(value: Any, schema: Any, *, path: str) -> None:
    if schema is True or schema is None:
        return
    if schema is False:
        raise StructuredOutputError(f"{path}: schema does not allow this value")
    if not isinstance(schema, dict):
        return

    for keyword in ("allOf",):
        subschemas = schema.get(keyword)
        if isinstance(subschemas, list):
            for sub in subschemas:
                _validate_fallback(value, sub, path=path)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not _matches_any(value, any_of, path=path):
            raise StructuredOutputError(f"{path}: value did not match anyOf schemas")

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(1 for sub in one_of if _validates(value, sub, path=path))
        if matches != 1:
            raise StructuredOutputError(f"{path}: value matched {matches} oneOf schemas")

    if "const" in schema and value != schema["const"]:
        raise StructuredOutputError(f"{path}: value must equal {schema['const']!r}")
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        raise StructuredOutputError(f"{path}: value must be one of {schema['enum']!r}")

    expected = schema.get("type")
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, str(item)) for item in allowed):
            raise StructuredOutputError(f"{path}: expected type {expected!r}")

    if isinstance(value, dict):
        _validate_object(value, schema, path=path)
    if isinstance(value, list):
        _validate_array(value, schema, path=path)
    if isinstance(value, str):
        _validate_string(value, schema, path=path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path=path)


def _validate_object(value: dict[str, Any], schema: dict[str, Any], *, path: str) -> None:
    required = schema.get("required") or []
    if isinstance(required, list):
        for key in required:
            if key not in value:
                raise StructuredOutputError(f"{path}: missing required key {key!r}")

    properties = schema.get("properties") or {}
    if isinstance(properties, dict):
        for key, sub in properties.items():
            if key in value:
                _validate_fallback(value[key], sub, path=f"{path}.{key}")

    additional = schema.get("additionalProperties", True)
    if additional is False and isinstance(properties, dict):
        extras = [key for key in value if key not in properties]
        if extras:
            raise StructuredOutputError(f"{path}: unexpected key {extras[0]!r}")
    elif isinstance(additional, dict) and isinstance(properties, dict):
        for key, item in value.items():
            if key not in properties:
                _validate_fallback(item, additional, path=f"{path}.{key}")


def _validate_array(value: list[Any], schema: dict[str, Any], *, path: str) -> None:
    if "minItems" in schema and len(value) < int(schema["minItems"]):
        raise StructuredOutputError(f"{path}: expected at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > int(schema["maxItems"]):
        raise StructuredOutputError(f"{path}: expected at most {schema['maxItems']} items")
    items = schema.get("items")
    if isinstance(items, dict):
        for index, item in enumerate(value):
            _validate_fallback(item, items, path=f"{path}[{index}]")


def _validate_string(value: str, schema: dict[str, Any], *, path: str) -> None:
    if "minLength" in schema and len(value) < int(schema["minLength"]):
        raise StructuredOutputError(f"{path}: expected string length >= {schema['minLength']}")
    if "maxLength" in schema and len(value) > int(schema["maxLength"]):
        raise StructuredOutputError(f"{path}: expected string length <= {schema['maxLength']}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        raise StructuredOutputError(f"{path}: string did not match pattern {pattern!r}")


def _validate_number(value: int | float, schema: dict[str, Any], *, path: str) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise StructuredOutputError(f"{path}: expected number >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise StructuredOutputError(f"{path}: expected number <= {schema['maximum']}")


def _matches_any(value: Any, subschemas: list[Any], *, path: str) -> bool:
    return any(_validates(value, sub, path=path) for sub in subschemas)


def _validates(value: Any, schema: Any, *, path: str) -> bool:
    try:
        _validate_fallback(value, schema, path=path)
        return True
    except StructuredOutputError:
        return False


def _matches_type(value: Any, type_name: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(type_name, True)


def _safe_schema_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return clean[:64] or "workflow_structured_output"
