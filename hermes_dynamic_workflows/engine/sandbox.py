"""AST validation for generated Python workflow scripts.

This is a guardrail, not a perfect Python sandbox. The runtime keeps the
available globals narrow and this validator rejects the most dangerous syntax.
The next hardening step is to execute scripts in a subprocess and expose
agent()/phase()/log() through RPC.
"""

from __future__ import annotations

import ast
from typing import Any

from ..core.config import PluginConfig
from ..core.errors import SandboxViolation, WorkflowParseError

# We gate CAPABILITY (what a script can touch), not CONTROL FLOW (how it loops
# or branches). while/try/raise are pure control flow — harmless on their own
# and required by the documented loop-until-budget / loop-until-dry / catch-
# gracefully patterns — so they are allowed. Imports, file/process/network
# access, dunder traversal and dynamic eval stay forbidden; that is the real
# integrity+escape boundary (all world-access must go through child agents and
# Hermes' approval engine, never the orchestration script itself).
FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncWith,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
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

META_FIRST_ERROR = "Invalid workflow script: `meta = {...}` must be the FIRST statement in the script"
DETERMINISM_ERROR = (
    "Workflow scripts must be deterministic: current time and randomness are "
    "unavailable (breaks resume). Stamp results after the workflow returns, "
    "or pass timestamps via args."
)
MAX_AST_NODES = 2500
MAX_STRING_LITERAL_CHARS = 20000
MAX_ABS_INT_LITERAL = 10**9
ENTRYPOINT_NAME = "__workflow_main__"


def parse_script(script: str, config: PluginConfig) -> ast.Module:
    if not isinstance(script, str) or not script.strip():
        raise WorkflowParseError(META_FIRST_ERROR)
    if len(script) > config.script_max_chars:
        raise WorkflowParseError(
            f"Invalid workflow script: workflow script is too large "
            f"({len(script)} chars; max {config.script_max_chars})"
        )
    try:
        tree = ast.parse(script, filename="<workflow>", mode="exec")
    except SyntaxError as exc:
        line = exc.lineno or 0
        column = exc.offset or 0
        raise WorkflowParseError(
            "Invalid workflow script: Script parse error: "
            f"{exc.msg} at line {line}, column {column}. "
            "Workflow scripts must be plain Python."
        ) from exc
    extract_meta(tree)
    _validate_top_level_contract(tree)
    validate_ast(tree)
    return instrument_loops(tree)


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

        if isinstance(node, ast.ExceptHandler):
            _validate_except_handler(node)


def extract_meta(tree: ast.Module) -> dict[str, Any]:
    """Extract the required literal ``meta = {...}`` first statement."""
    if not tree.body:
        raise WorkflowParseError(META_FIRST_ERROR)
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        raise WorkflowParseError(META_FIRST_ERROR)
    target = stmt.targets[0]
    if not isinstance(target, ast.Name) or target.id != "meta":
        raise WorkflowParseError(META_FIRST_ERROR)
    if not isinstance(stmt.value, ast.Dict):
        raise WorkflowParseError(META_FIRST_ERROR)
    _validate_meta_literal(stmt.value)
    try:
        value = ast.literal_eval(stmt.value)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        raise WorkflowParseError("Invalid workflow script: meta must be a pure literal")
    if not isinstance(value, dict):
        raise WorkflowParseError(META_FIRST_ERROR)
    return _normalize_meta(value)


def _normalize_meta(value: dict[str, Any]) -> dict[str, Any]:
    if "name" not in value:
        raise WorkflowParseError("Invalid workflow script: meta.name must be a non-empty string")
    if "description" not in value:
        raise WorkflowParseError("Invalid workflow script: meta.description must be a non-empty string")
    meta: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise WorkflowParseError("Invalid workflow script: meta keys must be strings")
        if key.startswith("_") or key in {"__proto__", "constructor", "prototype"}:
            raise WorkflowParseError(f"Invalid workflow script: forbidden meta key: {key}")
        if key in {"name", "description", "whenToUse"}:
            if item is not None and not isinstance(item, str):
                raise WorkflowParseError(f"Invalid workflow script: meta.{key} must be a string")
            meta[key] = item or ""
        elif key == "phases":
            if not isinstance(item, list):
                raise WorkflowParseError("Invalid workflow script: meta.phases must be a list")
            normalized = []
            for part in item:
                if isinstance(part, str):
                    normalized.append(part)
                    continue
                if isinstance(part, dict):
                    title = part.get("title")
                    if not isinstance(title, str) or not title.strip():
                        raise WorkflowParseError(
                            "Invalid workflow script: meta.phases object entries require a title string"
                        )
                    entry = {"title": title.strip()}
                    for phase_key in ("detail", "model"):
                        value = part.get(phase_key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise WorkflowParseError(
                                    f"Invalid workflow script: meta.phases.{phase_key} must be a string"
                                )
                            entry[phase_key] = value
                    normalized.append(entry)
                    continue
                raise WorkflowParseError(
                    "Invalid workflow script: meta.phases entries must be strings or objects"
                )
            meta[key] = normalized
        else:
            meta[key] = item
    name = str(meta.get("name") or "").strip()
    if not name:
        raise WorkflowParseError("Invalid workflow script: meta.name must be a non-empty string")
    description = str(meta.get("description") or "").strip()
    if not description:
        raise WorkflowParseError(
            "Invalid workflow script: meta.description must be a non-empty string"
        )
    meta["name"] = name
    meta["description"] = description
    return meta


def _validate_meta_literal(node: ast.Dict) -> None:
    for key in node.keys:
        if key is None:
            raise WorkflowParseError(
                "Invalid workflow script: meta must be a pure literal: "
                "only plain properties allowed in meta"
            )
    for child in ast.walk(node):
        if isinstance(child, ast.JoinedStr):
            raise WorkflowParseError(
                "Invalid workflow script: meta must be a pure literal: "
                "template interpolation not allowed in meta"
            )
        if isinstance(child, _META_LITERAL_NODE_TYPES):
            continue
        if child is node:
            continue
        raise WorkflowParseError(
            "Invalid workflow script: meta must be a pure literal: "
            f"non-literal node type in meta: {type(child).__name__}"
        )


def _validate_top_level_contract(tree: ast.Module) -> None:
    for stmt in tree.body[1:]:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "workflow":
            raise WorkflowParseError(
                "do not define workflow(); the workflow script body is already async"
            )


def build_execution_tree(tree: ast.Module) -> ast.Module:
    """Wrap the post-meta script body in the runtime's private async entrypoint."""
    body = tree.body[1:] or [ast.Return(value=ast.Constant(value=None))]
    entrypoint = ast.AsyncFunctionDef(
        name=ENTRYPOINT_NAME,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    execution_tree = ast.Module(body=[tree.body[0], entrypoint], type_ignores=[])
    return ast.fix_missing_locations(execution_tree)


def _validate_name(name: str) -> None:
    if name.startswith("__") or name in FORBIDDEN_NAMES:
        raise SandboxViolation(f"forbidden name: {name}")


def _validate_constant(value: Any) -> None:
    if isinstance(value, str) and len(value) > MAX_STRING_LITERAL_CHARS:
        raise SandboxViolation("string literal is too large")
    if isinstance(value, int) and abs(value) > MAX_ABS_INT_LITERAL:
        raise SandboxViolation("integer literal is too large")


def _validate_call(node: ast.Call) -> None:
    if _is_nondeterministic_call(node):
        raise SandboxViolation(DETERMINISM_ERROR)
    func = node.func
    if isinstance(func, ast.Name):
        _validate_name(func.id)
    elif isinstance(func, ast.Attribute):
        if func.attr.startswith("_"):
            raise SandboxViolation(f"forbidden method call: {func.attr}")
    else:
        raise SandboxViolation("dynamic call targets are not allowed")


_META_LITERAL_NODE_TYPES = (
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Load,
    ast.Set,
    ast.Tuple,
)


def _is_nondeterministic_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if not isinstance(func.value, ast.Name):
        return False
    base = func.value.id
    attr = func.attr
    return (
        (base == "time" and attr in {"time", "monotonic", "perf_counter", "process_time"})
        or (base == "random" and attr in {"random", "randint", "randrange", "choice", "shuffle"})
        or (base == "uuid" and attr in {"uuid1", "uuid4"})
        or (base == "datetime" and attr in {"now", "utcnow", "today"})
        or (base == "date" and attr == "today")
    )


def _validate_except_handler(node: ast.ExceptHandler) -> None:
    """Forbid wildcard catches that could swallow a ``WorkflowHalt``.

    A ``WorkflowHalt`` (user stop / deadline / hard limit) derives from
    ``BaseException``, so ``except Exception`` cannot catch it — but a bare
    ``except:`` or ``except BaseException`` would. Reject those so a run stays
    cancellable and bounded no matter what the script catches. Scripts may
    still ``except Exception`` (or a specific exposed type) to handle
    recoverable failures gracefully.
    """
    if node.type is None:
        raise SandboxViolation(
            "bare 'except:' is not allowed; catch Exception or a specific type"
        )
    handlers = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
    for handler_type in handlers:
        if isinstance(handler_type, ast.Name) and handler_type.id == "BaseException":
            raise SandboxViolation(
                "'except BaseException' is not allowed; catch Exception instead"
            )


# Name of the guard call the loop instrumenter injects; the runtime binds it in
# the script namespace. Dunder-prefixed so a script cannot define or shadow it
# (the validator forbids names starting with "__").
LOOP_GUARD_NAME = "__wf_tick__"


class _LoopGuard(ast.NodeTransformer):
    """Inject ``__wf_tick__()`` into every while/for iteration.

    ``__wf_tick__()`` checks stop/deadline and the loop-iteration cap (raising a
    WorkflowHalt if exceeded). This makes the cooperative deadline fire inside
    a pure-compute loop that never calls agent().
    """

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        guard = ast.Call(
            func=ast.Name(id=LOOP_GUARD_NAME, ctx=ast.Load()), args=[], keywords=[]
        )
        node.test = ast.BoolOp(op=ast.And(), values=[guard, node.test])
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        guard = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=LOOP_GUARD_NAME, ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )
        node.body.insert(0, guard)
        return node


def instrument_loops(tree: ast.Module) -> ast.Module:
    """Inject the per-iteration loop guard into every ``while`` and ``for`` loop.

    Runs after ``validate_ast`` (the injected nodes are trusted and not
    re-validated).
    """
    _LoopGuard().visit(tree)
    ast.fix_missing_locations(tree)
    return tree
