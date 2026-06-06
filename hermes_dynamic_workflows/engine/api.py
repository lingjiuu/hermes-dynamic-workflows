"""Workflow globals exposed to generated Python scripts."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic
from typing import Any, Callable

from .cache import agent_fingerprint, is_cache_miss
from ..plugin.structured_output import build_tool_schema_instruction
from ..ui.display import preview
from .errors import WorkflowHalt, WorkflowRuntimeError
from .structured import StructuredOutputError, validate_json_schema
from .types import AgentRecord, ChildAgentRequest, ChildAgentResult, WorkflowFrame


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
            "cwd": self.frame.cwd,
            "budget": self.budget,
            "print": self.log,
            "subworkflow": self.subworkflow,
        }

    def agent(self, prompt: str, opts: dict[str, Any] | None = None) -> Any:
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

        with self._lock:
            agent_id = self.context.reserve_agent()
            phase_name = str(opts.get("phase") or self.frame.current_phase or "") or None
            label = str(opts.get("label") or f"agent-{agent_id}")
            isolation = _normalize_isolation(opts.get("isolation"))
            agent_type = _normalize_agent_type(opts.get("agentType", opts.get("agent_type")))
            record = AgentRecord(
                id=agent_id,
                label=label,
                phase=phase_name,
                prompt=prompt,
                prompt_preview=preview(prompt, 160),
                agent_type=agent_type,
                isolation=isolation or "shared",
            )
            self.frame.agents.append(record)
            self._notify()

        task_prompt = prompt + build_tool_schema_instruction() if schema else prompt
        fingerprint = agent_fingerprint(
            prompt,
            {
                "label": label,
                "phase": phase_name,
                "schema": schema,
                "model": opts.get("model"),
                "agentType": agent_type,
                "isolation": isolation,
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
            prompt=task_prompt,
            label=label,
            phase=phase_name,
            toolsets=[],
            model=str(opts.get("model")).strip() if opts.get("model") else None,
            schema=schema,
            agent_type=agent_type,
            isolation=isolation,
            cwd=self.frame.cwd,
            structured_tool=bool(schema),
            on_start=on_child_start,
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
                return None
            finally:
                record.ended_at = monotonic()
                self._notify()
        return None

    def _run_child(self, request: ChildAgentRequest, record: AgentRecord) -> Any:
        return self.runner.run(request)

    def parallel(self, thunks: list[Callable[[], Any]]) -> list[Any]:
        self._check_deadline()
        if not isinstance(thunks, list):
            raise WorkflowRuntimeError("parallel() expects a list of callables")
        if not all(callable(item) for item in thunks):
            raise WorkflowRuntimeError("parallel() entries must be callables, e.g. lambda: agent(...)")
        if not thunks:
            return []

        results: list[Any] = [None] * len(thunks)
        max_workers = min(len(thunks), self.config.concurrency, self.config.max_concurrency)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dw-parallel") as pool:
            future_to_index = {pool.submit(thunk): index for index, thunk in enumerate(thunks)}
            for future in as_completed(future_to_index):
                self._check_deadline()
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    message = f"parallel[{index}] failed: {type(exc).__name__}: {exc}"
                    self.log(message)
                    with self._lock:
                        self.frame.errors.append(message)
                    results[index] = None
        return results

    def pipeline(self, items: list[Any], *stages: Callable[[Any, Any, int], Any]) -> list[Any]:
        self._check_deadline()
        if not isinstance(items, list):
            raise WorkflowRuntimeError("pipeline() expects a list as the first argument")
        if not stages or not all(callable(stage) for stage in stages):
            raise WorkflowRuntimeError("pipeline() expects one or more callable stages")

        def run_one(index: int, original: Any) -> Any:
            current = original
            for stage in stages:
                self._check_deadline()
                current = stage(current, original, index)
            return current

        return self.parallel([lambda i=i, item=item: run_one(i, item) for i, item in enumerate(items)])

    def phase(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise WorkflowRuntimeError("phase() expects a non-empty string")
        clean = name.strip()
        with self._lock:
            self.frame.current_phase = clean
            self.frame.ensure_phase(clean)
            self._notify()

    def log(self, message: Any) -> None:
        with self._lock:
            self.frame.logs.append(preview(message, 500))
            self._notify()

    def subworkflow(self, name_or_ref: Any, args: Any = None) -> Any:
        """Run a child workflow synchronously.

        Python workflows use def workflow() as their entrypoint, so the nested
        workflow API is named subworkflow() to avoid shadowing that function.
        """
        self._check_deadline()
        if self.depth >= 1:
            raise WorkflowRuntimeError("nested workflows are limited to one level")
        from .runtime import WorkflowOptions, run_workflow
        from ..storage.store import WorkflowStore, resolve_workflow_source

        if isinstance(name_or_ref, dict):
            params = dict(name_or_ref)
        else:
            params = {"name": str(name_or_ref)}
        source = resolve_workflow_source(params, store=WorkflowStore(), cwd=self.frame.cwd)
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
        # Compatibility with Python-style scripts. The model-facing API should
        # still prefer Claude-style ``agentType``.
        "agent_type",
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


def _normalize_agent_type(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    return clean or None


def _normalize_isolation(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    if clean in {"shared", "none"}:
        return "shared"
    if clean == "worktree":
        return clean
    raise WorkflowRuntimeError("isolation must be 'worktree' or 'shared'")


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
