"""AST validation for generated Python workflow scripts.

This is a guardrail, not a perfect Python sandbox. The runtime keeps the
available globals narrow and this validator rejects the most dangerous syntax.
The next hardening step is to execute scripts in a subprocess and expose
agent()/phase()/log() through RPC.
"""

from __future__ import annotations

import ast
from typing import Any

from .config import PluginConfig
from .errors import SandboxViolation, WorkflowParseError

FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
)

FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "type",
    "vars",
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "importlib",
}

MAX_AST_NODES = 2500
MAX_STRING_LITERAL_CHARS = 20000
MAX_ABS_INT_LITERAL = 10**9


def parse_script(script: str, config: PluginConfig) -> ast.Module:
    if not isinstance(script, str) or not script.strip():
        raise WorkflowParseError("workflow script must be a non-empty Python string")
    if len(script) > config.script_max_chars:
        raise WorkflowParseError(
            f"workflow script is too large ({len(script)} chars; max {config.script_max_chars})"
        )
    try:
        tree = ast.parse(script, filename="<workflow>", mode="exec")
    except SyntaxError as exc:
        raise WorkflowParseError(f"invalid Python workflow script: {exc.msg} at line {exc.lineno}") from exc
    validate_ast(tree)
    return tree


def validate_ast(tree: ast.AST) -> None:
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_AST_NODES:
            raise SandboxViolation(f"workflow script is too complex (>{MAX_AST_NODES} AST nodes)")

        if isinstance(node, FORBIDDEN_NODES):
            raise SandboxViolation(f"forbidden Python syntax: {type(node).__name__}")

        if isinstance(node, ast.Name):
            _validate_name(node.id)

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise SandboxViolation(f"forbidden attribute access: {node.attr}")

        if isinstance(node, ast.Constant):
            _validate_constant(node.value)

        if isinstance(node, ast.Call):
            _validate_call(node)


def extract_meta(tree: ast.Module) -> dict[str, Any]:
    """Extract a literal top-level ``meta = {...}`` assignment when present."""
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or target.id != "meta":
            continue
        try:
            value = ast.literal_eval(stmt.value)
        except (ValueError, TypeError, SyntaxError, MemoryError):
            raise WorkflowParseError("meta must be a literal dict")
        if not isinstance(value, dict):
            raise WorkflowParseError("meta must be a dict")
        return _normalize_meta(value)
    return {"name": "dynamic-workflow", "description": ""}


def _normalize_meta(value: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise WorkflowParseError("meta keys must be strings")
        if key.startswith("_") or key in {"__proto__", "constructor", "prototype"}:
            raise WorkflowParseError(f"forbidden meta key: {key}")
        if key in {"name", "description", "whenToUse"}:
            if item is not None and not isinstance(item, str):
                raise WorkflowParseError(f"meta.{key} must be a string")
            meta[key] = item or ""
        elif key == "phases":
            if not isinstance(item, list):
                raise WorkflowParseError("meta.phases must be a list")
            normalized = []
            for part in item:
                if isinstance(part, str):
                    normalized.append(part)
                    continue
                if isinstance(part, dict):
                    title = part.get("title")
                    if not isinstance(title, str) or not title.strip():
                        raise WorkflowParseError("meta.phases object entries require a title string")
                    entry = {"title": title.strip()}
                    for phase_key in ("detail", "model"):
                        value = part.get(phase_key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise WorkflowParseError(f"meta.phases.{phase_key} must be a string")
                            entry[phase_key] = value
                    normalized.append(entry)
                    continue
                raise WorkflowParseError("meta.phases entries must be strings or objects")
            meta[key] = normalized
        else:
            meta[key] = item
    name = str(meta.get("name") or "").strip()
    if not name:
        raise WorkflowParseError("meta.name must be a non-empty string")
    meta["name"] = name
    meta.setdefault("description", "")
    return meta


def _validate_name(name: str) -> None:
    if name.startswith("__") or name in FORBIDDEN_NAMES:
        raise SandboxViolation(f"forbidden name: {name}")


def _validate_constant(value: Any) -> None:
    if isinstance(value, str) and len(value) > MAX_STRING_LITERAL_CHARS:
        raise SandboxViolation("string literal is too large")
    if isinstance(value, int) and abs(value) > MAX_ABS_INT_LITERAL:
        raise SandboxViolation("integer literal is too large")


def _validate_call(node: ast.Call) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        _validate_name(func.id)
    elif isinstance(func, ast.Attribute):
        if func.attr.startswith("_"):
            raise SandboxViolation(f"forbidden method call: {func.attr}")
    else:
        raise SandboxViolation("dynamic call targets are not allowed")
