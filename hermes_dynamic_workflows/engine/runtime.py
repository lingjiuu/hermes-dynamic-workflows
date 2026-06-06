"""Python workflow script runtime."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

from .api import WorkflowAPI
from .cache import ResumeCache
from .config import PluginConfig, load_config
from .context import WorkflowExecutionContext
from .sandbox import LOOP_GUARD_NAME, extract_meta, parse_script
from .types import ChildAgentRunner, WorkflowFrame, WorkflowState, normalize_phase_specs


@dataclass
class WorkflowOptions:
    args: Any = None
    cwd: str | None = None
    config: PluginConfig | None = None
    child_runner: ChildAgentRunner | None = None
    stop_event: threading.Event | None = None
    resume_cache: ResumeCache | None = None
    on_update: Callable[[WorkflowState], None] | None = None
    on_journal: Callable[[dict[str, Any]], None] | None = None
    context: WorkflowExecutionContext | None = None
    parent_frame: WorkflowFrame | None = None
    frame: WorkflowFrame | None = None
    depth: int = 0
    source_ref: str | None = None
    plugin_context: Any = None
    token_budget_total: int | None = None


@dataclass
class WorkflowResult:
    value: Any
    state: WorkflowState

    @property
    def agent_count(self) -> int:
        return int((self.state.snapshot().get("totals") or {}).get("agents") or 0)

    @property
    def error_count(self) -> int:
        return int((self.state.snapshot().get("totals") or {}).get("errors") or 0)


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    # Exception types a script may catch to handle recoverable failures (a
    # failed child agent, a subworkflow error, bad result indexing). Halt
    # signals (stop/deadline/limits) are BaseException and deliberately absent,
    # so `except Exception` cannot swallow them.
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
}


def run_workflow(script: str, options: WorkflowOptions | None = None) -> WorkflowResult:
    options = options or WorkflowOptions()
    config = options.config or load_config()
    args = options.args
    cwd = options.cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
    tree = parse_script(script, config)
    meta = extract_meta(tree)
    context = options.context
    frame = options.frame

    if context is None:
        if options.child_runner is None:
            from ..agents.runner import HermesChildAgentRunner

            child_runner = HermesChildAgentRunner(config)
        else:
            child_runner = options.child_runner
        stop_event = options.stop_event or threading.Event()
        root = WorkflowFrame(
            id="root",
            meta=meta,
            args=args,
            cwd=cwd,
            phases=normalize_phase_specs(meta.get("phases")),
            source_ref=options.source_ref,
        )
        context = WorkflowExecutionContext(
            config=config,
            runner=child_runner,
            stop_event=stop_event,
            resume_cache=options.resume_cache or ResumeCache(),
            deadline=monotonic() + config.workflow_timeout_seconds,
            root=root,
            on_update=options.on_update,
            on_journal=options.on_journal,
            plugin_context=options.plugin_context,
            token_budget_total=options.token_budget_total,
        )
        frame = root
    else:
        if frame is None:
            parent = options.parent_frame or context.root
            frame = context.create_child_frame(
                parent=parent,
                meta=meta,
                args=args,
                cwd=cwd,
                source_ref=options.source_ref,
            )
        else:
            frame.meta = meta
            frame.args = args
            frame.cwd = cwd
            frame.phases = normalize_phase_specs(meta.get("phases"))

    state = context.state
    api = WorkflowAPI(
        context=context,
        frame=frame,
        depth=options.depth,
    )
    namespace = _build_namespace(api)

    try:
        context.check_runtime()
        compiled = compile(tree, filename="<workflow>", mode="exec")
        exec(compiled, namespace, namespace)
        value = _resolve_workflow_value(namespace)
        if _frame_agent_count(frame) == 0:
            raise WorkflowRuntimeError("workflow must call agent() at least once")
        frame.status = "completed"
        return WorkflowResult(value=value, state=state)
    except BaseException as exc:
        # BaseException so a WorkflowHalt (stop/deadline/limit) still records
        # frame status before propagating to the run thread.
        frame.status = "stopped" if context.stop_event.is_set() else "error"
        frame.errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        frame.ended_at = monotonic()
        context.notify()


def _build_namespace(api: WorkflowAPI) -> dict[str, Any]:
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "json": json,
        "math": math,
        "True": True,
        "False": False,
        "None": None,
    }
    namespace.update(api.globals())
    # Per-iteration guard injected into every `while` test by the sandbox.
    namespace[LOOP_GUARD_NAME] = api.context.tick_loop
    return namespace


def _resolve_workflow_value(namespace: dict[str, Any]) -> Any:
    workflow = namespace.get("workflow")
    if callable(workflow):
        return workflow()
    if "return_value" in namespace:
        return namespace["return_value"]
    if "result" in namespace:
        return namespace["result"]
    raise WorkflowRuntimeError(
        "workflow script must define workflow(), return_value, or result"
    )


def _frame_agent_count(frame: WorkflowFrame) -> int:
    return len(frame.agents) + sum(_frame_agent_count(child) for child in frame.children)
