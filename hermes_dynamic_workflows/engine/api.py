"""Workflow globals exposed to generated Python scripts."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from .cache import agent_fingerprint, is_cache_miss
from ..core.text import preview
from ..core.errors import (
    ChildAgentError,
    ChildAgentSkipped,
    WorkflowHalt,
    WorkflowParseError,
    WorkflowRuntimeError,
)
from ..core.schema import StructuredOutputError, validate_json_schema
from ..core.types import (
    AgentRecord,
    ChildAgentRequest,
    ChildAgentResult,
    ResolvedAgentSpec,
    WorkflowFrame,
)

MAX_VM_ARRAY_ITEMS = 4096


class BudgetView:
    def __init__(self, context: Any):
        self._context = context

    @property
    def total(self) -> int | None:
        return self._context.token_budget_total

    def spent(self) -> int:
        return self._context.spent_tokens

    def remaining(self) -> float:
        return self._context.remaining_tokens


class WorkflowAPI:
    def __init__(
        self,
        *,
        context: Any,
        frame: WorkflowFrame,
        depth: int = 0,
    ):
        self.context = context
        self.frame = frame
        self.runner = context.runner
        self.config = context.config
        self.resume_cache = context.resume_cache
        self.depth = depth
        self._lock = threading.RLock()
        self.budget = BudgetView(context)

    def globals(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": self.phase,
            "log": self.log,
            "args": self.frame.args,
            "budget": self.budget,
            "workflow": self.workflow,
        }

    async def agent(self, prompt: str, opts: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(self._agent_sync, prompt, opts)

    def _agent_sync(self, prompt: str, opts: dict[str, Any] | None = None) -> Any:
        self._check_deadline()
        opts = opts or {}
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkflowRuntimeError("agent() expects a non-empty prompt string")
        if not isinstance(opts, dict):
            raise WorkflowRuntimeError("agent() options must be a dict")
        _validate_agent_opts(opts)

        schema = opts.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise WorkflowRuntimeError("agent() schema option must be a dict")
        if schema is not None:
            try:
                validate_json_schema(schema)
            except StructuredOutputError as exc:
                raise WorkflowRuntimeError(str(exc)) from exc
        phase_name = str(opts.get("phase") or self.frame.current_phase or "") or None
        resolved = _resolve_agent_spec(
            opts,
            cwd=self.frame.cwd,
            config=self.config,
            structured_output=schema is not None,
            phase_model=_phase_model(self.frame, phase_name),
        )

        with self._lock:
            agent_id = self.context.reserve_agent()
            label = str(opts.get("label") or f"agent-{agent_id}")
            record = AgentRecord(
                id=agent_id,
                label=label,
                phase=phase_name,
                prompt=prompt,
                prompt_preview=preview(prompt, 160),
                agent_type=resolved.agent_type_name,
                isolation=resolved.isolation or "shared",
                model=resolved.model,
            )
            self.frame.agents.append(record)
            self._notify()

        fingerprint = agent_fingerprint(
            prompt,
            {
                "schema": schema,
                **resolved.cache_inputs(),
            },
        )
        journal_key = f"v2:{fingerprint}"
        cached = self.resume_cache.get(fingerprint)
        if not is_cache_miss(cached):
            record.status = "done"
            record.result_preview = f"(cached) {preview(cached, 170)}"
            if schema:
                record.structured = {"status": "cached", "mode": "tool", "attempts": 0}
            record.started_at = monotonic()
            record.ended_at = record.started_at
            self.resume_cache.put(fingerprint, cached)
            self._journal(
                {
                    "type": "result",
                    "key": journal_key,
                    "agentId": str(agent_id),
                    "cached": True,
                    "result": cached,
                }
            )
            self._notify()
            return cached

        def on_child_start(metadata: dict[str, Any]) -> None:
            _apply_child_metadata(record, metadata)
            self._notify()

        request = ChildAgentRequest(
            id=agent_id,
            prompt=prompt,
            label=label,
            phase=phase_name,
            toolsets=list(resolved.toolsets),
            model=resolved.model,
            schema=schema,
            agent_type=resolved.agent_type_name,
            isolation=resolved.isolation,
            cwd=self.frame.cwd,
            structured_tool=bool(schema),
            on_start=on_child_start,
            resolved=resolved,
        )
        if schema:
            record.structured = {
                "status": "pending",
                "mode": "tool",
                "attempts": 0,
            }

        max_attempts = 1
        record.status = "running"
        record.started_at = monotonic()
        self._journal(
            {
                "type": "started",
                "key": journal_key,
                "agentId": str(agent_id),
            }
        )
        self._notify()

        accumulated_tokens = 0
        for attempt in range(max_attempts):
            try:
                with self.context.agent_slot():
                    raw_result = self._run_child(request, record)
                metadata = raw_result.metadata if isinstance(raw_result, ChildAgentResult) else {}
                result = raw_result.content if isinstance(raw_result, ChildAgentResult) else raw_result
                _apply_child_metadata(record, metadata)
                # Count every attempt's tokens toward the budget; record.tokens
                # reports the run total across attempts.
                self.context.record_tokens(record.tokens)
                accumulated_tokens += record.tokens
                if schema:
                    if not isinstance(metadata, dict) or not metadata.get("structured_captured"):
                        raise WorkflowRuntimeError(
                            "child agent did not submit valid structured output"
                        )
                    result = metadata.get("structured_result")
                    record.structured.update(
                        {
                            "status": "valid",
                            "mode": "tool",
                            "attempts": int(metadata.get("structured_attempts") or 1),
                            "error": "",
                        }
                    )
                record.status = "done"
                record.attempts = attempt + 1
                record.tokens = accumulated_tokens
                record.result_preview = preview(result, 180)
                self.resume_cache.put(fingerprint, result)
                self._journal(
                    {
                        "type": "result",
                        "key": journal_key,
                        "agentId": str(agent_id),
                        "result": result,
                    }
                )
                return result
            except WorkflowHalt:
                # A run-level halt (stop / deadline / token/agent/loop limit) is
                # not a child failure — never retry or swallow it.
                raise
            except ChildAgentSkipped:
                record.attempts = attempt + 1
                record.status = "skipped"
                record.tokens = accumulated_tokens
                record.result_preview = ""
                self._journal(
                    {
                        "type": "result",
                        "key": journal_key,
                        "agentId": str(agent_id),
                        "skipped": True,
                        "result": None,
                    }
                )
                return None
            except Exception as exc:
                record.attempts = attempt + 1
                record.status = "error"
                record.tokens = accumulated_tokens
                record.error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self.frame.errors.append(f"{label}: {record.error}")
                self._journal(
                    {
                        "type": "error",
                        "key": journal_key,
                        "agentId": str(agent_id),
                        "error": record.error,
                    }
                )
                if isinstance(exc, ChildAgentError):
                    raise
                raise ChildAgentError(str(exc)) from exc
            finally:
                record.ended_at = monotonic()
                self._notify()
        raise ChildAgentError("child agent failed without a result")

    def _run_child(self, request: ChildAgentRequest, record: AgentRecord) -> Any:
        return self.runner.run(request)

    async def parallel(self, thunks: list[Callable[[], Any]]) -> list[Any]:
        self._check_deadline()
        if not isinstance(thunks, list):
            raise WorkflowRuntimeError("parallel() expects a list of callables")
        _check_vm_array_length(thunks)
        if not all(callable(item) for item in thunks):
            raise WorkflowRuntimeError("parallel() entries must be callables, e.g. lambda: agent(...)")
        if not thunks:
            return []

        results = await asyncio.gather(
            *(self._run_parallel_thunk(index, thunk) for index, thunk in enumerate(thunks))
        )
        self._check_deadline()
        return results

    async def _run_parallel_thunk(self, index: int, thunk: Callable[[], Any]) -> Any:
        try:
            self._check_deadline()
            return await _maybe_await(thunk())
        except WorkflowHalt:
            raise
        except Exception as exc:
            message = f"parallel[{index}] failed: {type(exc).__name__}: {exc}"
            self.log(message)
            with self._lock:
                self.frame.errors.append(message)
            return None

    async def pipeline(self, items: list[Any], *stages: Callable[[Any, Any, int], Any]) -> list[Any]:
        self._check_deadline()
        if not isinstance(items, list):
            raise WorkflowRuntimeError("pipeline() expects a list as the first argument")
        _check_vm_array_length(items)
        if not stages or not all(callable(stage) for stage in stages):
            raise WorkflowRuntimeError("pipeline() expects one or more callable stages")

        async def run_one(index: int, original: Any) -> Any:
            current = original
            try:
                for stage in stages:
                    self._check_deadline()
                    current = await _maybe_await(stage(current, original, index))
            except WorkflowHalt:
                raise
            except Exception as exc:
                message = f"pipeline[{index}] failed: {type(exc).__name__}: {exc}"
                self.log(message)
                with self._lock:
                    self.frame.errors.append(message)
                return None
            return current

        return await asyncio.gather(*(run_one(i, item) for i, item in enumerate(items)))

    def phase(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise WorkflowRuntimeError("phase() expects a non-empty string")
        clean = name.strip()
        with self._lock:
            self.frame.current_phase = clean
            self.frame.ensure_phase(clean)
            self._notify()

    def log(self, message: Any) -> None:
        if not isinstance(message, str):
            raise WorkflowRuntimeError("log() expects a string")
        with self._lock:
            self.frame.logs.append(preview(message, 500))
            self._notify()

    async def workflow(self, name_or_ref: Any, args: Any = None) -> Any:
        return await asyncio.to_thread(self._workflow_sync, name_or_ref, args)

    def _workflow_sync(self, name_or_ref: Any, args: Any = None) -> Any:
        """Run a child workflow from async Python scripts."""
        self._check_deadline()
        if self.depth >= 1:
            raise WorkflowRuntimeError("nested workflows are limited to one level")
        from .runtime import WorkflowOptions, run_workflow
        from ..storage.store import WorkflowStore, resolve_workflow_source

        params = _normalize_workflow_ref(name_or_ref)
        store = self.context.store or WorkflowStore()
        try:
            source = resolve_workflow_source(params, store=store, cwd=self.frame.cwd)
        except WorkflowParseError as exc:
            if "name" in params:
                name = str(params.get("name") or "")
                available = ", ".join(_available_workflow_names(store, self.frame.cwd)) or "none"
                raise WorkflowRuntimeError(
                    f"workflow({name!r}): no workflow with that name. Available: {available}"
                ) from exc
            raise WorkflowRuntimeError(str(exc)) from exc
        result = run_workflow(
            source.script,
            WorkflowOptions(
                args=args,
                cwd=self.frame.cwd,
                config=self.config,
                child_runner=self.runner,
                context=self.context,
                parent_frame=self.frame,
                depth=self.depth + 1,
                source_ref=source.source_ref,
                store=store,
            ),
        )
        return result.value

    def _check_deadline(self) -> None:
        self.context.check_runtime()

    def _notify(self) -> None:
        self.context.notify()

    def _journal(self, event: dict[str, Any]) -> None:
        self.context.journal(event)


_PUBLIC_AGENT_OPT_KEYS = frozenset(
    {
        "label",
        "phase",
        "schema",
        "model",
        "isolation",
        "agentType",
    }
)


def _validate_agent_opts(opts: dict[str, Any]) -> None:
    unknown = sorted(str(key) for key in opts if str(key) not in _PUBLIC_AGENT_OPT_KEYS)
    if not unknown:
        return
    raise WorkflowRuntimeError(
        "unsupported agent() option(s): "
        + ", ".join(unknown)
        + ". Public workflow agent options are label, phase, schema, model, "
        "isolation, and agentType. Put tool access in agentType presets or "
        "plugin config; provider/runtime, timeout, and retry policy belong in "
        "Hermes/plugin configuration, not workflow scripts."
    )


def _check_vm_array_length(items: list[Any]) -> None:
    if len(items) > MAX_VM_ARRAY_ITEMS:
        raise WorkflowRuntimeError(
            f"array length {len(items)} exceeds the maximum of {MAX_VM_ARRAY_ITEMS} "
            "supported across the workflow VM boundary"
        )


def _normalize_agent_type(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    return clean or None


def _normalize_agent_model(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    if not clean or clean.lower() == "inherit":
        return None
    return clean


def _normalize_isolation(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    if clean == "worktree":
        return clean
    raise WorkflowRuntimeError("isolation must be 'worktree'")


def _resolve_agent_spec(
    opts: dict[str, Any],
    *,
    cwd: str,
    config: Any,
    structured_output: bool,
    phase_model: str | None = None,
) -> ResolvedAgentSpec:
    from ..agent.presets import list_agent_types, resolve_agent_type
    from ..agent.runner import (
        _prepare_mcp_tool_registry,
        _resolve_child_toolsets,
        build_child_system_prompt,
    )

    requested_type = _normalize_agent_type(opts.get("agentType"))
    agent_type_spec = resolve_agent_type(requested_type, cwd=cwd)
    if requested_type and agent_type_spec is None:
        available = ", ".join(spec.name for spec in list_agent_types(cwd=cwd)) or "none"
        raise WorkflowRuntimeError(
            f"agent({{agentType}}): agent type '{requested_type}' not found. "
            f"Available agents: {available}"
        )

    explicit_isolation = _normalize_isolation(opts.get("isolation"))
    agent_type_isolation = _normalize_agent_type_isolation(
        getattr(agent_type_spec, "isolation", None)
    )
    model = _normalize_agent_model(
        opts.get("model")
        if opts.get("model")
        else phase_model
        if phase_model
        else getattr(agent_type_spec, "model", None)
    )
    _prepare_mcp_tool_registry(config)
    toolsets = tuple(
        _resolve_child_toolsets(
            config,
            [],
            getattr(agent_type_spec, "toolsets", ()),
            include_discoverable=agent_type_spec is None,
        )
    )
    prompt = build_child_system_prompt(
        agent_type_spec,
        structured_output=structured_output,
    )
    return ResolvedAgentSpec(
        requested_agent_type=requested_type,
        agent_type_spec=agent_type_spec,
        model=model or None,
        isolation=explicit_isolation or agent_type_isolation,
        toolsets=toolsets,
        allowed_tools=tuple(getattr(agent_type_spec, "allowed_tools", ()) or ()),
        disallowed_tools=tuple(getattr(agent_type_spec, "disallowed_tools", ()) or ()),
        system_prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        workspace=str(Path(cwd).expanduser().resolve()),
    )


def _phase_model(frame: WorkflowFrame, phase_name: str | None) -> str | None:
    if not phase_name:
        return None
    for phase in frame.phases:
        if phase.title == phase_name:
            return _normalize_agent_model(phase.model)
    return None


def _normalize_workflow_ref(name_or_ref: Any) -> dict[str, str]:
    if isinstance(name_or_ref, str) and name_or_ref.strip():
        return {"name": name_or_ref.strip()}
    if isinstance(name_or_ref, dict) and set(name_or_ref) == {"scriptPath"}:
        script_path = name_or_ref.get("scriptPath")
        if isinstance(script_path, str) and script_path.strip():
            return {"scriptPath": script_path.strip()}
    raise WorkflowRuntimeError(
        "workflow() expects a non-empty workflow name or {'scriptPath': '<path>'}"
    )


def _available_workflow_names(store: Any, cwd: str) -> list[str]:
    from ..storage.store import _RESERVED_WORKFLOW_NAMES

    directories = [
        Path(cwd) / ".hermes" / "workflows",
        store.workflows_dir,
        Path(__file__).resolve().parent.parent / "workflows",
    ]
    names: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        try:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.py")):
                stem = path.stem
                if not stem or stem.startswith("_") or stem in _RESERVED_WORKFLOW_NAMES:
                    continue
                if stem not in seen:
                    seen.add(stem)
                    names.append(stem)
        except OSError:
            continue
    return names


def _normalize_agent_type_isolation(value: Any) -> str | None:
    if value in (None, "", "shared", "none"):
        return None
    clean = str(value).strip()
    if clean == "worktree":
        return clean
    raise WorkflowRuntimeError(
        f"agentType isolation must be 'worktree' when set, got {clean!r}"
    )


def _apply_child_metadata(record: AgentRecord, metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        return
    record.runner = str(metadata.get("runner") or record.runner)
    record.workspace = _optional_str(metadata.get("workspace"))
    record.model = _optional_str(metadata.get("model"))
    record.task_id = _optional_str(metadata.get("task_id"))
    record.hermes_session_id = _optional_str(
        metadata.get("hermes_session_id") or metadata.get("session_id")
    )
    record.transcript_path = _optional_str(metadata.get("transcript_path"))
    record.agent_type = _optional_str(metadata.get("agent_type")) or record.agent_type
    record.isolation = _optional_str(metadata.get("isolation")) or record.isolation
    record.tokens = _as_int_metadata(metadata.get("tokens"))
    record.cache_read_tokens = _as_int_metadata(metadata.get("cache_read_tokens"))
    record.cache_write_tokens = _as_int_metadata(metadata.get("cache_write_tokens"))
    record.tool_calls = _as_int_metadata(metadata.get("tool_calls"))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    return clean or None


def _as_int_metadata(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
