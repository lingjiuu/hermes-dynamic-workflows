"""Workflow globals exposed to generated Python scripts."""

from __future__ import annotations

from dataclasses import replace
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic
from typing import Any, Callable

from .cache import agent_fingerprint, is_cache_miss
from .structured import (
    build_repair_prompt,
    build_response_format_overrides,
    build_schema_instruction,
    looks_like_response_format_error,
    parse_structured_output,
)
from ..ui.display import preview
from .errors import WorkflowRuntimeError
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

        schema = opts.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise WorkflowRuntimeError("agent() schema option must be a dict")

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

        task_prompt = prompt + build_schema_instruction(schema) if schema else prompt
        structured_mode = self.config.structured_output_mode if schema else ""
        request_overrides = _structured_request_overrides(schema, structured_mode, label, agent_id)
        fingerprint = agent_fingerprint(
            prompt,
            {
                "label": label,
                "phase": phase_name,
                "schema": schema,
                "structured_mode": structured_mode,
                "toolsets": opts.get("toolsets"),
                "model": opts.get("model"),
                "provider": opts.get("provider"),
                "agentType": agent_type,
                "isolation": isolation,
            },
        )
        cached = self.resume_cache.get(agent_id, fingerprint)
        if not is_cache_miss(cached):
            record.status = "done"
            record.result_preview = f"(cached) {preview(cached, 170)}"
            if schema:
                record.structured = {"status": "cached", "mode": structured_mode, "attempts": 0}
            record.started_at = monotonic()
            record.ended_at = record.started_at
            self.resume_cache.put(agent_id, fingerprint, cached)
            self._notify()
            return cached

        request = ChildAgentRequest(
            id=agent_id,
            prompt=task_prompt,
            label=label,
            phase=phase_name,
            toolsets=_normalize_toolsets(opts.get("toolsets")),
            model=str(opts.get("model")).strip() if opts.get("model") else None,
            provider=str(opts.get("provider")).strip() if opts.get("provider") else None,
            schema=schema,
            timeout_seconds=_as_timeout(opts.get("timeout_seconds")),
            agent_type=agent_type,
            isolation=isolation,
            cwd=self.frame.cwd,
            request_overrides=request_overrides,
        )
        if schema:
            record.structured = {
                "status": "pending",
                "mode": "response_format" if request_overrides else "prompt",
                "attempts": 0,
                "repaired": False,
            }

        record.status = "running"
        record.started_at = monotonic()
        self._notify()
        try:
            with self.context.agent_slot():
                raw_result = self._run_child(request, record)
            metadata = raw_result.metadata if isinstance(raw_result, ChildAgentResult) else {}
            result = raw_result.content if isinstance(raw_result, ChildAgentResult) else raw_result
            _apply_child_metadata(record, metadata)
            self.context.record_tokens(record.tokens)
            if schema:
                result = self._parse_or_repair_structured(result, schema, record, request)
            record.status = "done"
            record.result_preview = preview(result, 180)
            self.resume_cache.put(agent_id, fingerprint, result)
            return result
        except Exception as exc:
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.frame.errors.append(f"{label}: {record.error}")
            return None
        finally:
            record.ended_at = monotonic()
            self._notify()

    def _run_child(self, request: ChildAgentRequest, record: AgentRecord) -> Any:
        try:
            return self.runner.run(request)
        except Exception as exc:
            if (
                request.request_overrides
                and self.config.structured_output_mode == "auto"
                and looks_like_response_format_error(exc)
            ):
                record.structured.update(
                    {
                        "status": "response_format_fallback",
                        "mode": "prompt",
                        "response_format_error": preview(f"{type(exc).__name__}: {exc}", 240),
                    }
                )
                self._notify()
                return self.runner.run(replace(request, request_overrides=None))
            raise

    def _parse_or_repair_structured(
        self,
        raw: Any,
        schema: dict[str, Any],
        record: AgentRecord,
        request: ChildAgentRequest,
    ) -> Any:
        raw_preview = preview(raw, self.config.structured_raw_preview_chars)
        record.structured.update(
            {
                "status": "validating",
                "attempts": 1,
                "raw_preview": raw_preview,
            }
        )
        self._notify()
        try:
            parsed = parse_structured_output(raw, schema)
            record.structured.update({"status": "valid", "error": ""})
            return parsed
        except Exception as first_error:
            record.structured.update(
                {
                    "status": "repairing" if self.config.structured_retries > 0 else "failed",
                    "error": f"{type(first_error).__name__}: {first_error}",
                }
            )
            self._notify()
            if self.config.structured_retries <= 0:
                raise

            repaired = self._repair_structured_output(raw, schema, first_error, record, request)
            record.structured["attempts"] = 2
            try:
                parsed = parse_structured_output(repaired, schema)
            except Exception as second_error:
                record.structured.update(
                    {
                        "status": "failed",
                        "error": f"{type(second_error).__name__}: {second_error}",
                    }
                )
                raise
            record.structured.update(
                {
                    "status": "repaired",
                    "error": "",
                    "repaired": True,
                    "repair_preview": preview(repaired, self.config.structured_raw_preview_chars),
                }
            )
            return parsed

    def _repair_structured_output(
        self,
        raw: Any,
        schema: dict[str, Any],
        error: Exception,
        record: AgentRecord,
        request: ChildAgentRequest,
    ) -> Any:
        prompt = build_repair_prompt(raw, f"{type(error).__name__}: {error}", schema)
        if self.config.structured_repair_with_llm:
            repaired = self._repair_with_plugin_llm(prompt, schema, record, request)
            if repaired is not None:
                return repaired
        return self._repair_with_child_runner(prompt, schema, record, request)

    def _repair_with_plugin_llm(
        self,
        prompt: str,
        schema: dict[str, Any],
        record: AgentRecord,
        request: ChildAgentRequest,
    ) -> Any | None:
        plugin_context = getattr(self.context, "plugin_context", None)
        llm = getattr(plugin_context, "llm", None)
        complete = getattr(llm, "complete_structured", None)
        if not callable(complete):
            return None
        try:
            result = complete(
                instructions="Repair invalid workflow child-agent output into schema-valid JSON.",
                input=[{"type": "text", "text": prompt}],
                json_schema=schema,
                schema_name=f"workflow_agent_{request.id}_structured_repair",
                timeout=min(float(request.timeout_seconds or self.config.child_timeout_seconds), 60.0),
                purpose="dynamic_workflow_structured_repair",
            )
        except Exception as exc:
            record.structured["repair_error"] = preview(f"{type(exc).__name__}: {exc}", 240)
            self._notify()
            return None

        usage = getattr(result, "usage", None)
        total_tokens = _usage_total_tokens(usage)
        if total_tokens:
            record.tokens += total_tokens
            self.context.record_tokens(total_tokens)
            record.structured["repair_tokens"] = total_tokens

        parsed = getattr(result, "parsed", None)
        if parsed is not None:
            record.structured["repair_runner"] = "plugin_llm"
            return parsed
        text = getattr(result, "text", None)
        if text:
            record.structured["repair_runner"] = "plugin_llm"
            return text
        return None

    def _repair_with_child_runner(
        self,
        prompt: str,
        schema: dict[str, Any],
        record: AgentRecord,
        request: ChildAgentRequest,
    ) -> Any:
        repair_request = replace(
            request,
            prompt=prompt,
            label=f"{request.label}:structured-repair",
            toolsets=[],
            schema=schema,
            request_overrides=None,
        )
        with self.context.agent_slot():
            raw_result = self.runner.run(repair_request)
        metadata = raw_result.metadata if isinstance(raw_result, ChildAgentResult) else {}
        repair_tokens = _as_int_metadata(metadata.get("tokens")) if isinstance(metadata, dict) else 0
        if repair_tokens:
            record.tokens += repair_tokens
            self.context.record_tokens(repair_tokens)
            record.structured["repair_tokens"] = repair_tokens
        record.structured["repair_runner"] = "child_agent"
        return raw_result.content if isinstance(raw_result, ChildAgentResult) else raw_result

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


def _normalize_toolsets(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        raise WorkflowRuntimeError("toolsets must be a string or list of strings")
    return [str(item).strip() for item in raw if str(item).strip()]


def _as_timeout(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        raise WorkflowRuntimeError("timeout_seconds must be a number") from None


def _structured_request_overrides(
    schema: dict[str, Any] | None,
    mode: str,
    label: str,
    agent_id: int,
) -> dict[str, Any] | None:
    if not schema or mode not in {"auto", "response_format"}:
        return None
    return build_response_format_overrides(schema, name=f"workflow_agent_{agent_id}_{label}")


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
    record.agent_type = _optional_str(metadata.get("agent_type")) or record.agent_type
    record.isolation = _optional_str(metadata.get("isolation")) or record.isolation
    record.tokens = _as_int_metadata(metadata.get("tokens"))
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


def _usage_total_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    for name in ("total_tokens", "tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    if isinstance(usage, dict):
        for name in ("total_tokens", "tokens"):
            value = usage.get(name)
            if isinstance(value, (int, float)):
                return max(0, int(value))
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", input_tokens) or 0
        output_tokens = usage.get("output_tokens", output_tokens) or 0
    try:
        return max(0, int(input_tokens) + int(output_tokens))
    except (TypeError, ValueError):
        return 0
