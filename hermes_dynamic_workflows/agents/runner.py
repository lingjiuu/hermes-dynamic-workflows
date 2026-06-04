"""Standalone Hermes AIAgent runner used by workflow agent()."""

from __future__ import annotations

import inspect
import os
import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import replace
from typing import Any

from .presets import AgentTypeSpec, resolve_agent_type
from .worktree import WorkspaceLease, create_workspace_lease
from ..engine.config import PluginConfig
from ..engine.errors import ChildAgentError, WorkflowTimeout
from ..engine.structured_tool import (
    STRUCTURED_OUTPUT_TOOLSET,
    clear_expectation,
    pop_result,
    register_expectation,
)
from ..engine.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner

logger = logging.getLogger(__name__)


class HermesChildAgentRunner(ChildAgentRunner):
    """Create standalone Hermes AIAgent children without native delegation."""

    def __init__(self, config: PluginConfig):
        self.config = config
        self._active_children: list[Any] = []
        self._active_lock = threading.RLock()

    def run(self, request: ChildAgentRequest) -> ChildAgentResult:
        task_id = f"workflow-{uuid.uuid4().hex[:12]}"
        base_cwd = request.cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
        agent_type = resolve_agent_type(request.agent_type, cwd=base_cwd, task_id=task_id)
        request = _apply_agent_type_defaults(request, agent_type)
        lease = create_workspace_lease(
            cwd=base_cwd,
            isolation=request.isolation,
            label=request.label,
            task_id=task_id,
            keep_worktree=self.config.keep_worktrees,
        )
        runtime = self._resolve_runtime(request)
        toolsets = _resolve_child_toolsets(
            self.config,
            request.toolsets,
            agent_type.toolsets if agent_type else (),
        )
        structured_tool = bool(request.structured_tool and request.schema)
        if structured_tool and STRUCTURED_OUTPUT_TOOLSET not in toolsets:
            toolsets = toolsets + [STRUCTURED_OUTPUT_TOOLSET]
        _prepare_mcp_tool_registry(self.config)
        child = self._build_agent(request, runtime, toolsets, lease, agent_type)
        if structured_tool:
            register_expectation(lease.task_id, request.schema)
        try:
            result = self._run_child_with_timeout(child, request, lease, agent_type, toolsets)
            if structured_tool and isinstance(result, ChildAgentResult):
                captured, value, attempts = pop_result(lease.task_id)
                if captured:
                    result.metadata["structured_captured"] = True
                    result.metadata["structured_result"] = value
                    result.metadata["structured_attempts"] = attempts
            return result
        finally:
            if structured_tool:
                clear_expectation(lease.task_id)

    def supports_request_overrides(self) -> bool:
        try:
            from run_agent import AIAgent
        except Exception:
            return False
        return _callable_accepts_keyword(AIAgent, "request_overrides")

    def _resolve_runtime(self, request: ChildAgentRequest) -> dict[str, Any]:
        requested_model = (request.model or "").strip() or None
        requested_provider = (request.provider or "").strip() or None
        if requested_model and not self.config.allow_model_override:
            raise ChildAgentError("model override is disabled for workflow child agents")
        if requested_provider and not self.config.allow_provider_override:
            raise ChildAgentError("provider override is disabled for workflow child agents")
        if requested_provider and not requested_model:
            raise ChildAgentError("provider override requires an explicit model")

        try:
            from hermes_cli.config import load_config
            from hermes_cli.fallback_config import get_fallback_chain
            from hermes_cli.models import detect_provider_for_model
            from hermes_cli.runtime_provider import resolve_runtime_provider
        except Exception as exc:
            raise ChildAgentError(f"could not import Hermes runtime helpers: {exc}") from exc

        cfg = load_config() or {}
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, str):
            cfg_model = model_cfg
            cfg_provider = ""
        else:
            cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()

        env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
        effective_model = requested_model or env_model or cfg_model
        effective_provider = requested_provider
        explicit_base_url = None

        if effective_provider is None and (requested_model or env_model):
            explicit_model = requested_model or env_model
            direct = None
            try:
                from hermes_cli import model_switch as model_switch

                model_switch._ensure_direct_aliases()
                direct = model_switch.DIRECT_ALIASES.get(explicit_model.strip().lower())
            except Exception:
                direct = None
            if direct is not None:
                effective_model = direct.model
                effective_provider = direct.provider
                explicit_base_url = direct.base_url.rstrip("/") if direct.base_url else None
            else:
                current_provider = (
                    cfg_provider
                    or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
                    or "auto"
                )
                detected = detect_provider_for_model(explicit_model, current_provider)
                if detected:
                    effective_provider, effective_model = detected

        runtime = resolve_runtime_provider(
            requested=effective_provider,
            target_model=effective_model or None,
            explicit_base_url=explicit_base_url,
        )
        runtime["model"] = effective_model
        runtime["fallback_model"] = get_fallback_chain(cfg) or None
        return runtime

    def _build_agent(
        self,
        request: ChildAgentRequest,
        runtime: dict[str, Any],
        toolsets: list[str],
        lease: WorkspaceLease,
        agent_type: AgentTypeSpec | None,
    ):
        try:
            from run_agent import AIAgent
        except Exception as exc:
            raise ChildAgentError(f"could not import Hermes AIAgent: {exc}") from exc

        child_prompt = build_child_system_prompt(agent_type)
        try:
            session_db = _create_session_db()
        except Exception:
            session_db = None

        kwargs = {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "model": runtime.get("model"),
            "credential_pool": runtime.get("credential_pool"),
            "fallback_model": runtime.get("fallback_model"),
            "enabled_toolsets": toolsets,
            "disabled_toolsets": list(self.config.blocked_child_toolsets),
            "quiet_mode": True,
            "platform": "cli",
            "skip_context_files": True,
            "skip_memory": True,
            "clarify_callback": _child_clarify_callback,
            "ephemeral_system_prompt": child_prompt,
            "session_db": session_db,
            "session_id": lease.task_id,
        }
        if request.request_overrides and _callable_accepts_keyword(AIAgent, "request_overrides"):
            kwargs["request_overrides"] = request.request_overrides
        return AIAgent(**kwargs)

    def _run_child_with_timeout(
        self,
        child: Any,
        request: ChildAgentRequest,
        lease: WorkspaceLease,
        agent_type: AgentTypeSpec | None,
        toolsets: list[str],
    ) -> ChildAgentResult:
        timeout = request.timeout_seconds or self.config.child_timeout_seconds
        approval_callback = _make_child_approval_callback(self.config.child_approval_policy)

        def _init_worker() -> None:
            # Install the approval callback on the worker thread itself, so the
            # child's terminal tool has it when a flagged command would prompt.
            # Mirrors tools/delegate_tool.py's ThreadPoolExecutor(initializer=...)
            # pattern (see GHSA-qg5c-hvr5-hjgr).
            if approval_callback is None:
                return
            try:
                from tools.terminal_tool import set_approval_callback

                set_approval_callback(approval_callback)
            except Exception:
                pass

        def _run() -> dict[str, Any]:
            _register_task_cwd(lease.task_id, lease.cwd)
            return child.run_conversation(
                user_message=build_child_task_message(request, workspace=lease.cwd),
                task_id=lease.task_id,
            )

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dw-child-agent",
            initializer=_init_worker,
        )
        with self._active_lock:
            self._active_children.append(child)
        future = executor.submit(_run)
        result: dict[str, Any] | None = None
        try:
            result = future.result(timeout=timeout) or {}
            content = str(result.get("final_response") or "")
            # A hard child failure (e.g. a non-retryable API error) returns an
            # "error" string and no usable final_response. Surface it as a real
            # failure instead of a silent empty "success", so the agent shows as
            # error in /workflows rather than masking the failure.
            failure = _child_failure_message(result, content)
            if failure:
                raise ChildAgentError(failure)
            metadata = _child_metadata(child, result, lease, agent_type, toolsets)
            return ChildAgentResult(content=content, metadata=metadata)
        except FuturesTimeoutError as exc:
            try:
                if hasattr(child, "interrupt"):
                    child.interrupt()
            finally:
                raise WorkflowTimeout(f"child agent timed out after {timeout:.0f}s") from exc
        finally:
            with self._active_lock:
                if child in self._active_children:
                    self._active_children.remove(child)
            _cleanup_task_cwd(lease.task_id)
            lease.cleanup()
            executor.shutdown(wait=False, cancel_futures=True)

    def interrupt_all(self) -> None:
        with self._active_lock:
            children = list(self._active_children)
        for child in children:
            try:
                if hasattr(child, "interrupt"):
                    child.interrupt()
            except Exception:
                pass


def build_child_system_prompt(agent_type: AgentTypeSpec | None = None) -> str:
    """Stable, per-task-independent system prompt for a child agent.

    Kept byte-identical across children with the same agent_type so that, on
    cache-eligible models, the ``[tools + system]`` request prefix is shared and
    cached across a workflow's fan-out. Hermes' ``system_and_3`` caching places
    a cache_control breakpoint at the end of the system prompt; if the system
    prompt carried per-task data (label/phase/workspace) it would vary per child
    and defeat cross-child reuse of the (identical) tool definitions in front of
    it. Per-task context lives in the task message instead — see
    :func:`build_child_task_message`.
    """
    lines = [
        "You are a focused Hermes child agent spawned by a dynamic workflow.",
        "Work only on the delegated task. Do not ask the user questions.",
        "Use available tools when needed, then return a concise final answer.",
    ]
    if agent_type is not None:
        lines.extend(
            [
                "",
                f"Agent type: {agent_type.name}",
                f"Agent type source: {agent_type.source}",
                "Follow these agent-type instructions for this child task:",
                "",
                agent_type.instructions,
            ]
        )
    return "\n".join(lines)


def build_child_task_message(request: ChildAgentRequest, *, workspace: str) -> str:
    """Per-task context (the variable part) prepended to the child's task.

    This is the child's first user message — the part that legitimately differs
    per child (workspace, label, phase, worktree note) — kept out of the cached
    system prefix so it doesn't break cross-child cache reuse.
    """
    context = [f"- Workspace: {workspace}"]
    if request.label:
        context.append(f"- Task label: {request.label}")
    if request.phase:
        context.append(f"- Workflow phase: {request.phase}")
    if request.isolation == "worktree":
        context.append(
            "- You are running in an isolated git worktree; keep all file "
            "operations inside the workspace above."
        )
    return "Task context:\n" + "\n".join(context) + "\n\n" + request.prompt


def _apply_agent_type_defaults(
    request: ChildAgentRequest,
    agent_type: AgentTypeSpec | None,
) -> ChildAgentRequest:
    if agent_type is None:
        return request
    return replace(
        request,
        model=request.model or agent_type.model,
        provider=request.provider or agent_type.provider,
        isolation=request.isolation or agent_type.isolation,
    )


def _resolve_child_toolsets(
    config: PluginConfig,
    requested: list[str],
    agent_type_toolsets: tuple[str, ...] = (),
) -> list[str]:
    raw = requested or list(agent_type_toolsets) or list(config.default_child_toolsets)
    blocked = set(config.blocked_child_toolsets)
    cleaned: list[str] = []
    for item in raw:
        name = str(item).strip()
        if not name or name in blocked or name in cleaned:
            continue
        cleaned.append(name)
    return cleaned or list(config.default_child_toolsets)


def _prepare_mcp_tool_registry(config: PluginConfig) -> None:
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery, wait_for_mcp_discovery

        start_background_mcp_discovery(
            logger=logging.getLogger("hermes_dynamic_workflows.mcp"),
            thread_name="dw-mcp-discovery",
        )
        wait_for_mcp_discovery(timeout=config.mcp_discovery_wait_seconds)
    except Exception:
        pass


def _create_session_db():
    try:
        from hermes_state import SessionDB

        return SessionDB()
    except Exception:
        return None


def _child_clarify_callback(question: str, choices=None) -> str:
    if choices:
        return (
            "[dynamic workflow child agent: no user is available. "
            f"Pick the best option from {choices} and continue.]"
        )
    return (
        "[dynamic workflow child agent: no user is available. "
        "Make the most reasonable assumption and continue.]"
    )


def _child_failure_message(result: Any, content: str) -> str | None:
    """Return an error message when a child result signals a hard failure with
    no usable content, else None.

    Hermes' conversation loop returns ``{"final_response": None, "error": "..."}``
    (and completed=False / failed=True) when an API call aborts. Successful
    turns never set ``error``, so a truthy ``error`` with empty content is an
    unambiguous failure that should not be reported as an empty success.
    """
    if not isinstance(result, dict):
        return None
    error_msg = result.get("error")
    if error_msg and not content:
        return str(error_msg)
    return None


def _make_child_approval_callback(policy: str):
    """Build the non-interactive approval callback for child worker threads.

    Child agents run every command through Hermes' approval engine
    (tools/approval.py): hardline blocks, the permanent allowlist, yolo, and
    smart mode all still apply upstream. This callback only decides what to do
    when a *flagged* command would otherwise prompt a human who isn't present.

    Policy comes from the plugin's own config key
    ``dynamic_workflows.child_approval_policy`` (never delegation.*), so a
    workflow's blast radius is controlled independently of native delegation:

      deny    -> refuse flagged commands (safe default)
      approve -> allow flagged commands (hardline is still blocked upstream)
      smart   -> defer to Hermes' _smart_approve auxiliary-LLM guardian;
                 'escalate' (uncertain) resolves to deny since no human is present
    """
    clean = (policy or "deny").strip().lower()

    if clean == "approve":
        def _approve(command: str, description: str, **_: Any) -> str:
            logger.warning(
                "workflow child auto-approved flagged command (policy=approve): %s (%s)",
                command, description,
            )
            return "once"
        return _approve

    if clean == "smart":
        def _smart(command: str, description: str, **_: Any) -> str:
            try:
                from tools.approval import _smart_approve
            except Exception:
                return "deny"
            try:
                verdict = _smart_approve(command, description)
            except Exception:
                return "deny"
            if verdict == "approve":
                logger.warning(
                    "workflow child smart-approved flagged command: %s (%s)",
                    command, description,
                )
                return "once"
            # 'deny' and 'escalate' both refuse: no human is present to escalate to.
            return "deny"
        return _smart

    def _deny(command: str, description: str, **_: Any) -> str:
        logger.warning(
            "workflow child denied flagged command (policy=deny): %s (%s)",
            command, description,
        )
        return "deny"
    return _deny


def _register_task_cwd(task_id: str, cwd: str) -> None:
    try:
        from tools.terminal_tool import register_task_env_overrides

        register_task_env_overrides(task_id, {"cwd": cwd})
    except Exception:
        pass


def _cleanup_task_cwd(task_id: str) -> None:
    try:
        from tools.terminal_tool import cleanup_vm, clear_task_env_overrides

        clear_task_env_overrides(task_id)
        cleanup_vm(task_id)
    except Exception:
        try:
            from tools.terminal_tool import clear_task_env_overrides

            clear_task_env_overrides(task_id)
        except Exception:
            pass


def _child_metadata(
    child: Any,
    result: dict[str, Any],
    lease: WorkspaceLease,
    agent_type: AgentTypeSpec | None,
    toolsets: list[str],
) -> dict[str, Any]:
    prompt_tokens = _int_attr(child, "session_prompt_tokens")
    completion_tokens = _int_attr(child, "session_completion_tokens")
    reasoning_tokens = _int_attr(child, "session_reasoning_tokens")
    metadata = {
        "runner": "standalone",
        "workspace": lease.cwd,
        "isolation": lease.isolation or "shared",
        "worktree_path": lease.path,
        "worktree_branch": lease.branch,
        "agent_type": agent_type.name if agent_type else None,
        "agent_type_source": agent_type.source if agent_type else None,
        "model": getattr(child, "model", None),
        "toolsets": toolsets,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "tokens": prompt_tokens + completion_tokens + reasoning_tokens,
        # Hermes auto-enables Anthropic prompt caching for Claude-family models
        # (agent/prompt_caching.py). Children inherit it, so these counters show
        # how much each child reused vs wrote to the cache.
        "cache_read_tokens": _int_attr(child, "session_cache_read_tokens"),
        "cache_write_tokens": _int_attr(child, "session_cache_write_tokens"),
        "tool_calls": _tool_call_count(result),
    }
    return metadata


def _int_attr(obj: Any, name: str) -> int:
    value = getattr(obj, name, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _tool_call_count(result: dict[str, Any]) -> int:
    """Count actual tool invocations in the child's conversation.

    Hermes' get_activity_summary() exposes api_call_count (LLM round-trips), not
    a tool-call count, so it is not a valid source here — using it reported a
    nonzero "tool calls" even for toolset=[] agents that just answered. Count
    real tool calls from the result messages instead: OpenAI-style assistant
    `tool_calls`, plus Anthropic-style `tool_use` content blocks.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            )
    return count


def _callable_accepts_keyword(target: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False
