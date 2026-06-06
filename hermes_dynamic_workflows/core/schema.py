"""Structured output helpers for workflow child agents."""

from __future__ import annotations

import re
from typing import Any

from .errors import ChildAgentError


class StructuredOutputError(ChildAgentError):
    """Raised when child-agent output cannot satisfy a requested JSON schema."""


def validate_json_schema(schema: dict[str, Any]) -> None:
    """Validate a child-agent output schema before launching the child.

    With the optional ``jsonschema`` dependency installed, this uses the same
    Draft 2020-12 validator that runtime output validation uses. Directory
    plugin installs may not have dependencies installed, so the fallback accepts
    only the JSON Schema subset that this module can enforce itself.
    """
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

        Draft202012Validator.check_schema(schema)
        return
    except ImportError:
        pass
    except Exception as exc:
        message = getattr(exc, "message", str(exc))
        raise StructuredOutputError(f"invalid JSON Schema: {message}") from exc

    _validate_fallback_schema(schema, path="root")


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


_FALLBACK_UNSUPPORTED_KEYWORDS = {
    "$dynamicRef",
    "$recursiveRef",
    "$ref",
    "contains",
    "dependentRequired",
    "dependentSchemas",
    "else",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "if",
    "maxContains",
    "maxProperties",
    "minContains",
    "minProperties",
    "multipleOf",
    "not",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
    "uniqueItems",
}


def _validate_fallback_schema(schema: Any, *, path: str) -> None:
    if schema is True or schema is False:
        return
    if not isinstance(schema, dict):
        raise StructuredOutputError(
            f"invalid JSON Schema: {path}: schema must be an object or boolean"
        )

    for keyword in sorted(_FALLBACK_UNSUPPORTED_KEYWORDS.intersection(schema)):
        raise StructuredOutputError(
            "jsonschema package is not installed; "
            f"schema keyword {keyword!r} at {path} requires full JSON Schema validation. "
            'Install "jsonschema>=4,<5" in the Hermes Python environment or simplify the schema.'
        )

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not allowed_types or not all(isinstance(item, str) for item in allowed_types):
            raise StructuredOutputError(f"invalid JSON Schema: {path}.type must be a string or list")

    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list) or not all(isinstance(item, str) for item in required)
    ):
        raise StructuredOutputError(f"invalid JSON Schema: {path}.required must be a list of strings")

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise StructuredOutputError(f"invalid JSON Schema: {path}.properties must be an object")
        for name, subschema in properties.items():
            _validate_fallback_schema(subschema, path=f"{path}.properties.{name}")

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _validate_fallback_schema(additional, path=f"{path}.additionalProperties")
    elif additional is not None and not isinstance(additional, bool):
        raise StructuredOutputError(
            f"invalid JSON Schema: {path}.additionalProperties must be a boolean or object"
        )

    items = schema.get("items")
    if isinstance(items, dict):
        _validate_fallback_schema(items, path=f"{path}.items")
    elif items is not None and not isinstance(items, bool):
        raise StructuredOutputError(
            "jsonschema package is not installed; "
            f"{path}.items uses tuple/complex item validation. "
            'Install "jsonschema>=4,<5" in the Hermes Python environment or simplify the schema.'
        )

    for keyword in ("allOf", "anyOf", "oneOf"):
        subschemas = schema.get(keyword)
        if subschemas is None:
            continue
        if not isinstance(subschemas, list):
            raise StructuredOutputError(f"invalid JSON Schema: {path}.{keyword} must be a list")
        for index, subschema in enumerate(subschemas):
            _validate_fallback_schema(subschema, path=f"{path}.{keyword}[{index}]")


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
