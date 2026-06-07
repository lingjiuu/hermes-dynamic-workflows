"""Standalone Hermes AIAgent runner used by workflow agent()."""

from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
import threading
import unicodedata
import uuid
import logging
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import replace
from typing import Any

from .presets import AgentTypeSpec, list_agent_types, resolve_agent_type
from .worktree import WorkspaceLease, create_workspace_lease
from ..core.config import PluginConfig
from ..core.errors import ChildAgentError, ChildAgentSkipped, WorkflowTimeout
from .structured_output import (
    MAX_STRUCTURED_OUTPUT_RETRIES,
    STRUCTURED_OUTPUT_CONTINUE_MESSAGE,
    STRUCTURED_OUTPUT_TOOL_NAME,
    STRUCTURED_OUTPUT_TOOLSET,
    build_tool_schema_instruction,
    clear_expectation,
    peek_result,
    register_expectation,
    specialize_structured_output_tool,
    structured_output_tool_scope,
)
from ..core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner

try:
    from wcwidth import wcswidth as _wcswidth
except ImportError:
    _wcswidth = None

logger = logging.getLogger(__name__)

_CHILD_EXCLUDED_TOOL_NAMES = frozenset({"skill_manage"})


class _WorkflowApprovalCoordinator:
    """Serialize workflow-child CLI approvals and reuse explicit session grants."""

    def __init__(self, callback: Any):
        self._callback = callback if callable(callback) else None
        self._lock = threading.RLock()
        self._session_approved_descriptions: set[str] = set()

    def callback_for(self, task_id: str, on_event: Any = None):
        if self._callback is None:
            return None

        def _callback(command: str, description: str, **kwargs: Any) -> str:
            emit = on_event if callable(on_event) else None
            if emit is not None:
                emit(
                    {
                        "type": "approval",
                        "status": "queued",
                        "task_id": task_id,
                        "command": command,
                        "description": description,
                    }
                )
            with self._lock:
                if description in self._session_approved_descriptions:
                    if emit is not None:
                        emit(
                            {
                                "type": "approval",
                                "status": "reused",
                                "task_id": task_id,
                                "command": command,
                                "description": description,
                                "choice": "session",
                            }
                        )
                    return "once"
                if emit is not None:
                    emit(
                        {
                            "type": "approval",
                            "status": "requested",
                            "task_id": task_id,
                            "command": command,
                            "description": description,
                        }
                    )
                choice = self._callback(command, description, **kwargs)
                if choice in ("session", "always"):
                    self._session_approved_descriptions.add(description)
                if emit is not None:
                    emit(
                        {
                            "type": "approval",
                            "status": "resolved",
                            "task_id": task_id,
                            "command": command,
                            "description": description,
                            "choice": choice,
                        }
                    )
                return choice

        return _callback


class HermesChildAgentRunner(ChildAgentRunner):
    """Create standalone Hermes AIAgent children without native delegation."""

    def __init__(
        self,
        config: PluginConfig,
        session_context: dict[str, str] | None = None,
        approval_callback: Any = None,
        approval_session_key: str | None = None,
        parent_runtime: dict[str, Any] | None = None,
    ):
        self.config = config
        # Captured gateway session vars (platform/session_key/chat_id/...) from
        # the launching session, re-applied on each child worker thread so a
        # child's dangerous command can route to the user for mid-run approval
        # (child_approval_policy="ask"). None outside gateway.
        self._session_context = session_context or None
        # Captured from the launching CLI thread. Hermes' CLI approval callback
        # is designed to serialize concurrent background requests.
        self._approval_callback = approval_callback if callable(approval_callback) else None
        self._approval_coordinator = _WorkflowApprovalCoordinator(self._approval_callback)
        self._approval_session_key = str(approval_session_key or "").strip()
        # Captured from the launching main agent. It stays in memory so default
        # children and model="inherit" use the active session runtime, including
        # a non-persisted /model switch, without leaking credentials to run data.
        self._parent_runtime = dict(parent_runtime) if parent_runtime else None
        self._active_children: dict[str, Any] = {}
        self._skipped_children: set[str] = set()
        self._active_lock = threading.RLock()
        self._progress_print_lock = threading.RLock()

    def run(self, request: ChildAgentRequest) -> ChildAgentResult:
        task_id = f"workflow-{uuid.uuid4().hex[:12]}"
        base_cwd = request.cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
        resolved = request.resolved
        agent_type = (
            resolved.agent_type_spec
            if resolved is not None
            else resolve_agent_type(request.agent_type, cwd=base_cwd)
        )
        if request.agent_type and agent_type is None:
            available = ", ".join(spec.name for spec in list_agent_types(cwd=base_cwd)) or "none"
            raise ChildAgentError(
                f"agent({{agentType}}): agent type '{request.agent_type}' not found. "
                f"Available agents: {available}"
            )
        if resolved is None:
            request = _apply_agent_type_defaults(request, agent_type)
        lease = create_workspace_lease(
            cwd=base_cwd,
            isolation=request.isolation,
            label=request.label,
            task_id=task_id,
            keep_worktree=self.config.keep_worktrees,
        )
        runtime = self._resolve_runtime(request)
        _prepare_mcp_tool_registry(self.config)
        toolsets = _resolve_child_toolsets(
            self.config,
            request.toolsets,
            agent_type.toolsets if agent_type else (),
            include_discoverable=(
                resolved is None
                and agent_type is None
                and not request.toolsets
            ),
        )
        structured_tool = bool(request.structured_tool and request.schema)
        if structured_tool and STRUCTURED_OUTPUT_TOOLSET not in toolsets:
            toolsets = toolsets + [STRUCTURED_OUTPUT_TOOLSET]
        tool_scope = structured_output_tool_scope() if structured_tool else nullcontext()
        with tool_scope:
            child = self._build_agent(request, runtime, toolsets, lease, agent_type)
            _configure_child_tools(
                child,
                toolsets=toolsets,
                blocked_toolsets=self.config.blocked_child_toolsets,
                allowed_tools=agent_type.allowed_tools if agent_type else (),
                disallowed_tools=agent_type.disallowed_tools if agent_type else (),
            )
            if structured_tool:
                child.tools = specialize_structured_output_tool(child.tools, request.schema)
                child.valid_tool_names.add(STRUCTURED_OUTPUT_TOOL_NAME)
            if request.on_start is not None:
                try:
                    request.on_start(_child_metadata(child, {}, lease, agent_type, toolsets))
                except Exception:
                    logger.debug("dynamic workflow child start callback failed", exc_info=True)
            if structured_tool:
                interrupt = getattr(child, "interrupt", None)
                register_expectation(
                    lease.task_id,
                    request.schema,
                    interrupt if callable(interrupt) else None,
                )
            try:
                result = self._run_child_with_timeout(child, request, lease, agent_type, toolsets)
                if self._consume_skipped(task_id):
                    raise ChildAgentSkipped(f"child agent {task_id} was skipped")
                if structured_tool and isinstance(result, ChildAgentResult):
                    captured, value, attempts = peek_result(lease.task_id)
                    if captured:
                        result.metadata["structured_captured"] = True
                        result.metadata["structured_result"] = value
                        result.metadata["structured_attempts"] = attempts
                return result
            except BaseException:
                if self._consume_skipped(task_id):
                    raise ChildAgentSkipped(f"child agent {task_id} was skipped") from None
                raise
            finally:
                self._clear_skipped(task_id)
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
        if requested_model and requested_model.lower() == "inherit":
            requested_model = None
        if requested_model and not self.config.allow_model_override:
            raise ChildAgentError("model override is disabled for workflow child agents")
        if requested_model is None and self._parent_runtime is not None:
            return dict(self._parent_runtime)

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
        effective_provider = None
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

        child_prompt = build_child_system_prompt(
            agent_type,
            structured_output=request.structured_tool,
        )
        try:
            session_db = _create_session_db()
        except Exception:
            session_db = None

        kwargs = {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "acp_command": runtime.get("acp_command"),
            "acp_args": runtime.get("acp_args"),
            "model": runtime.get("model"),
            "credential_pool": runtime.get("credential_pool"),
            "fallback_model": runtime.get("fallback_model"),
            "max_tokens": runtime.get("max_tokens"),
            "enabled_toolsets": toolsets,
            "disabled_toolsets": list(self.config.blocked_child_toolsets),
            "quiet_mode": True,
            "platform": "cli",
            "tool_progress_callback": self._make_tool_progress_callback(request, lease),
            "thinking_callback": _child_thinking_callback,
            "skip_context_files": True,
            "skip_memory": True,
            "clarify_callback": _child_clarify_callback,
            "ephemeral_system_prompt": child_prompt,
            "session_db": session_db,
            "session_id": lease.task_id,
        }
        request_overrides = request.request_overrides or runtime.get("request_overrides")
        if request_overrides and _callable_accepts_keyword(AIAgent, "request_overrides"):
            kwargs["request_overrides"] = request_overrides
        if runtime.get("reasoning_config") is not None:
            kwargs["reasoning_config"] = runtime["reasoning_config"]
        if runtime.get("service_tier") is not None:
            kwargs["service_tier"] = runtime["service_tier"]
        return AIAgent(**kwargs)

    def _make_tool_progress_callback(self, request: ChildAgentRequest, lease: WorkspaceLease):
        label = (request.label or "").strip() or lease.task_id
        print_lock = self._progress_print_lock

        def _callback(event: str, tool_name: str, _preview: Any = None, args: Any = None, **_: Any) -> None:
            if event != "tool.started":
                return
            if not _stdout_is_tty():
                return
            line = _compact_tool_progress_line(
                label,
                tool_name,
                args,
                max_width=_tool_progress_line_width(),
            )
            with print_lock:
                try:
                    _write_tool_progress_line(line)
                except (OSError, ValueError):
                    pass

        return _callback

    def _run_child_with_timeout(
        self,
        child: Any,
        request: ChildAgentRequest,
        lease: WorkspaceLease,
        agent_type: AgentTypeSpec | None,
        toolsets: list[str],
    ) -> ChildAgentResult:
        timeout = self.config.child_timeout_seconds
        live_lock = threading.RLock()
        live_tool_calls = 0

        def _emit_progress(event: dict[str, Any]) -> None:
            nonlocal live_tool_calls
            with live_lock:
                if event.get("type") == "tool_call":
                    live_tool_calls += 1
                metadata = _child_metadata(child, {}, lease, agent_type, toolsets)
                metadata["tool_calls"] = max(
                    int(metadata.get("tool_calls") or 0),
                    live_tool_calls,
                )
                if event.get("activity"):
                    metadata["activity"] = str(event["activity"])
                if event.get("type") == "approval":
                    metadata["approval"] = dict(event)
            _emit_request_update(request, metadata)

        interactive_callback = self._approval_coordinator.callback_for(
            lease.task_id,
            _emit_progress,
        )
        approval_callback = _make_child_approval_callback(
            self.config.child_approval_policy,
            getattr(self.config, "ask_fallback", "smart"),
            interactive_callback=interactive_callback,
        )

        def _init_worker() -> None:
            # Install the approval callback on the worker thread itself, so the
            # child's terminal tool has it when a flagged command would prompt.
            # Mirrors tools/delegate_tool.py's ThreadPoolExecutor(initializer=...)
            # pattern (see GHSA-qg5c-hvr5-hjgr).
            if approval_callback is not None:
                try:
                    from tools.terminal_tool import set_approval_callback

                    set_approval_callback(approval_callback)
                except Exception:
                    pass
            # Re-apply the launching gateway session context (contextvars don't
            # cross into detached threads) so check_all_command_guards can route
            # a flagged command to the originating user for mid-run approval.
            if self._session_context:
                try:
                    from ..host import gateway as host_gateway

                    host_gateway.set_session_vars(**self._session_context)
                except Exception:
                    pass

        def _run() -> dict[str, Any]:
            approval_token = None
            reset_current_session_key = None
            if self._approval_session_key:
                try:
                    from tools.approval import (
                        reset_current_session_key as _reset_current_session_key,
                        set_current_session_key,
                    )

                    approval_token = set_current_session_key(self._approval_session_key)
                    reset_current_session_key = _reset_current_session_key
                except Exception:
                    pass
            try:
                _register_task_cwd(lease.task_id, lease.cwd)
                message = build_child_task_message(request, workspace=lease.cwd)
                history = None
                stop_attempts = 0
                while True:
                    result = child.run_conversation(
                        user_message=message,
                        conversation_history=history,
                        task_id=lease.task_id,
                    )
                    if not request.structured_tool:
                        return result

                    captured, _value, tool_attempts = peek_result(lease.task_id)
                    if captured:
                        return result

                    stop_attempts += 1
                    if (
                        tool_attempts >= MAX_STRUCTURED_OUTPUT_RETRIES
                        or stop_attempts >= MAX_STRUCTURED_OUTPUT_RETRIES
                    ):
                        raise ChildAgentError(
                            "Failed to provide valid structured output after "
                            f"{MAX_STRUCTURED_OUTPUT_RETRIES} attempts"
                        )

                    previous_messages = result.get("messages") if isinstance(result, dict) else None
                    if isinstance(previous_messages, list):
                        history = previous_messages
                    message = STRUCTURED_OUTPUT_CONTINUE_MESSAGE
            finally:
                if approval_token is not None and reset_current_session_key is not None:
                    try:
                        reset_current_session_key(approval_token)
                    except Exception:
                        pass

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dw-child-agent",
            initializer=_init_worker,
        )
        try:
            from ..adapters.hooks import register_child_observer

            register_child_observer(lease.task_id, _emit_progress)
        except Exception:
            pass
        with self._active_lock:
            self._active_children[lease.task_id] = child
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
            try:
                from ..adapters.hooks import unregister_child_observer

                unregister_child_observer(lease.task_id)
            except Exception:
                pass
            with self._active_lock:
                self._active_children.pop(lease.task_id, None)
            _cleanup_task_cwd(lease.task_id)
            lease.cleanup()
            executor.shutdown(wait=False, cancel_futures=True)

    def interrupt_all(self) -> None:
        with self._active_lock:
            children = list(self._active_children.values())
        for child in children:
            try:
                if hasattr(child, "interrupt"):
                    child.interrupt()
            except Exception:
                pass

    def skip_child(self, task_id: str) -> bool:
        """Interrupt one active child and make its agent() call resolve to None."""
        wanted = str(task_id or "")
        if not wanted:
            return False
        with self._active_lock:
            child = self._active_children.get(wanted)
            if child is None:
                return False
            self._skipped_children.add(wanted)
        try:
            interrupt = getattr(child, "interrupt", None)
            if callable(interrupt):
                interrupt()
            return True
        except Exception:
            with self._active_lock:
                self._skipped_children.discard(wanted)
            return False

    def _consume_skipped(self, task_id: str) -> bool:
        with self._active_lock:
            if task_id not in self._skipped_children:
                return False
            self._skipped_children.discard(task_id)
            return True

    def _clear_skipped(self, task_id: str) -> None:
        with self._active_lock:
            self._skipped_children.discard(task_id)


def build_child_system_prompt(
    agent_type: AgentTypeSpec | None = None,
    *,
    structured_output: bool = False,
) -> str:
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
        "You are a subagent spawned by a workflow orchestration script.",
        "Use the tools available to complete the task.",
        "Your final text response is returned verbatim as a string to the calling script — it is your return value, not a message to a human.",
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
    if structured_output:
        lines.extend(["", build_tool_schema_instruction().strip()])
    return "\n".join(lines)


def build_child_task_message(request: ChildAgentRequest, *, workspace: str) -> str:
    """Per-task context (the variable part) prepended to the child's task.

    This is the child's first user message — the part that legitimately differs
    per child (workspace and worktree note) — kept out of the cached system
    prefix so it doesn't break cross-child cache reuse. Label and phase are
    display-only metadata and deliberately do not alter the child's prompt.
    """
    context = [f"- Workspace: {workspace}"]
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
    model = request.model or agent_type.model
    if model and model.strip().lower() == "inherit":
        model = None
    return replace(
        request,
        model=model,
        isolation=request.isolation or agent_type.isolation,
    )


def _resolve_child_toolsets(
    config: PluginConfig,
    requested: list[str],
    agent_type_toolsets: tuple[str, ...] = (),
    *,
    include_discoverable: bool = False,
) -> list[str]:
    raw = requested or list(agent_type_toolsets) or list(config.default_child_toolsets)
    if include_discoverable:
        raw = list(raw) + _discoverable_child_toolsets(config)
    blocked = set(config.blocked_child_toolsets)
    cleaned: list[str] = []
    for item in raw:
        name = str(item).strip()
        if not name or name in blocked or name in cleaned:
            continue
        cleaned.append(name)
    return cleaned


def _discoverable_child_toolsets(config: PluginConfig) -> list[str]:
    """Return installed MCP/plugin toolsets safe for the default workflow child."""
    blocked = set(config.blocked_child_toolsets)
    blocked.add(STRUCTURED_OUTPUT_TOOLSET)
    discovered: set[str] = set()
    plugin_tool_names: set[str] = set()
    try:
        from hermes_cli.plugins import get_plugin_manager

        plugin_tool_names = set(get_plugin_manager()._plugin_tool_names)
    except Exception:
        pass
    try:
        from tools.registry import registry
        from tools.tool_search import is_deferrable_tool_name

        for tool_name in registry.get_all_tool_names():
            if not is_deferrable_tool_name(tool_name):
                continue
            toolset = registry.get_toolset_for_tool(tool_name)
            if (
                toolset
                and toolset not in blocked
                and (toolset.startswith("mcp-") or tool_name in plugin_tool_names)
            ):
                discovered.add(toolset)
    except Exception:
        return []
    return sorted(discovered)


def _configure_child_tools(
    child: Any,
    *,
    toolsets: list[str],
    blocked_toolsets: tuple[str, ...],
    allowed_tools: tuple[str, ...] = (),
    disallowed_tools: tuple[str, ...] = (),
) -> None:
    """Apply the workflow-child tool surface and force native Tool Search."""
    try:
        import model_tools
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        definitions = model_tools.get_tool_definitions(
            enabled_toolsets=toolsets,
            disabled_toolsets=list(blocked_toolsets),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
        definitions = _filter_child_tool_definitions(
            definitions,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
        )
        direct: list[dict[str, Any]] = []
        searchable: list[dict[str, Any]] = []
        for definition in definitions:
            function = definition.get("function") if isinstance(definition, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if name in _CHILD_EXCLUDED_TOOL_NAMES:
                continue
            if name == STRUCTURED_OUTPUT_TOOL_NAME:
                direct.append(definition)
            else:
                searchable.append(definition)

        assembled = assemble_tool_defs(
            searchable,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        child.tools = list(assembled.tool_defs) + direct
    except Exception:
        child.tools = _filter_child_tool_definitions(
            [
                definition
                for definition in list(getattr(child, "tools", None) or [])
                if _tool_definition_name(definition) not in _CHILD_EXCLUDED_TOOL_NAMES
            ],
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
        )

    child.valid_tool_names = {
        name
        for definition in child.tools
        if (name := _tool_definition_name(definition))
    }
    # Keep structured_output directly callable but out of the deferred catalog.
    child.enabled_toolsets = [
        toolset for toolset in toolsets if toolset != STRUCTURED_OUTPUT_TOOLSET
    ]
    try:
        child._tool_search_scope_cache = None
    except Exception:
        pass


def _filter_child_tool_definitions(
    definitions: list[dict[str, Any]],
    *,
    allowed_tools: tuple[str, ...] = (),
    disallowed_tools: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    allowed = {name for item in allowed_tools if (name := str(item).strip())}
    disallowed = {name for item in disallowed_tools if (name := str(item).strip())}
    if allowed:
        allowed.add(STRUCTURED_OUTPUT_TOOL_NAME)
    filtered: list[dict[str, Any]] = []
    for definition in definitions:
        name = _tool_definition_name(definition)
        if not name:
            continue
        if allowed and name not in allowed:
            continue
        if name in disallowed:
            continue
        filtered.append(definition)
    return filtered


def _tool_definition_name(definition: Any) -> str:
    if not isinstance(definition, dict):
        return ""
    function = definition.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "")


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
        from ..host import session as host_session

        return host_session.create_session_db()
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


def _child_thinking_callback(_text: str) -> None:
    """Prevent quiet workflow children from starting raw stdout spinners."""


_TOOL_ARG_PRIORITY = (
    "query",
    "path",
    "url",
    "command",
    "action",
    "key",
    "name",
)
_TOOL_ARG_VALUE_LIMIT = 80
_TOOL_PROGRESS_FALLBACK_WIDTH = 120
_TOOL_PROGRESS_MAX_WIDTH = 140


def _stdout_is_tty() -> bool:
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _compact_tool_progress_line(
    label: str,
    tool_name: str,
    args: Any,
    *,
    max_width: int | None = None,
) -> str:
    clean_label = _single_line(label or "workflow-child")
    clean_tool = _single_line(tool_name or "tool")
    compact_args = _compact_tool_args(args)
    suffix = f"({compact_args})" if compact_args else ""
    line = f"↳ {clean_label} · {clean_tool}{suffix}"
    if max_width is None or _display_width(line) <= max_width:
        return line

    reserved_args_width = _display_width("(...)") if compact_args else 0
    static_width = _display_width(f"↳  · {clean_tool}") + reserved_args_width
    label_width = max(1, max_width - static_width)
    if label_width == 1 and static_width >= max_width:
        tool_width = max(
            1,
            max_width - _display_width("↳  · ") - reserved_args_width - 1,
        )
        clean_tool = _ellipsize_display_middle(clean_tool, tool_width)
        static_width = _display_width(f"↳  · {clean_tool}") + reserved_args_width
        label_width = max(1, max_width - static_width)

    clean_label = _ellipsize_display_middle(clean_label, label_width)
    prefix = f"↳ {clean_label} · {clean_tool}"
    if not compact_args:
        return _ellipsize_display_middle(prefix, max_width)

    args_width = max(1, max_width - _display_width(prefix) - _display_width("()"))
    compact_args = _ellipsize_display_middle(compact_args, args_width)
    return _ellipsize_display_middle(f"{prefix}({compact_args})", max_width)


def _tool_progress_line_width() -> int:
    try:
        columns = shutil.get_terminal_size((_TOOL_PROGRESS_FALLBACK_WIDTH, 24)).columns
    except (OSError, ValueError):
        columns = _TOOL_PROGRESS_FALLBACK_WIDTH
    return max(1, min(int(columns) - 2, _TOOL_PROGRESS_MAX_WIDTH))


def _write_tool_progress_line(line: str) -> None:
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return
    stream.write(f"{line}\n")
    stream.flush()


def _compact_tool_args(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for key in _TOOL_ARG_PRIORITY:
        if key in args:
            parts.append(f"{key}:{_compact_tool_arg_value(args[key])}")
            seen.add(key)
        if len(parts) >= 3:
            break
    if not parts:
        for key in sorted(str(k) for k in args.keys()):
            if key in seen:
                continue
            parts.append(f"{key}:{_compact_tool_arg_value(args.get(key))}")
            if len(parts) >= 3:
                break
    return ", ".join(parts)


def _compact_tool_arg_value(value: Any) -> str:
    if isinstance(value, str):
        return _quote_display_string(
            _ellipsize_display_middle(_single_line(value), _TOOL_ARG_VALUE_LIMIT)
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return "{...}"
    if isinstance(value, (list, tuple, set)):
        return "[...]"
    return _quote_display_string(
        _ellipsize_display_middle(_single_line(str(value)), _TOOL_ARG_VALUE_LIMIT)
    )


def _quote_display_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _display_width(value: str) -> int:
    if _wcswidth is not None:
        width = _wcswidth(value)
        if width >= 0:
            return width
    width = 0
    for char in value:
        category = unicodedata.category(char)
        if unicodedata.combining(char) or category in {"Cc", "Cf", "Mn", "Me"}:
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _take_display_prefix(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    end = 0
    for index in range(1, len(value) + 1):
        if _display_width(value[:index]) > limit:
            break
        end = index
    return value[:end]


def _take_display_suffix(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    start = len(value)
    for index in range(len(value) - 1, -1, -1):
        if _display_width(value[index:]) > limit:
            break
        start = index
    return value[start:]


def _ellipsize_display_middle(value: str, limit: int) -> str:
    if _display_width(value) <= limit:
        return value
    if limit <= 3:
        return "." * max(0, limit)
    keep = max(1, limit - 3)
    head_width = (keep + 1) // 2
    tail_width = keep // 2
    head = _take_display_prefix(value, head_width)
    tail = _take_display_suffix(value, tail_width)
    return f"{head}...{tail}"


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


def _emit_request_update(request: ChildAgentRequest, metadata: dict[str, Any]) -> None:
    if request.on_update is None:
        return
    try:
        request.on_update(metadata)
    except Exception:
        logger.debug("dynamic workflow child update callback failed", exc_info=True)


def _resolve_inherit_policy(policy: str) -> str:
    """Resolve ``inherit`` to Hermes' own approvals.mode (manual->ask,
    smart->smart, off->approve). Mirrors approval_hook._resolve_policy so the
    CLI per-thread callback and the non-CLI hook agree."""
    if policy != "inherit":
        return policy
    try:
        from tools.approval import _get_approval_mode

        mode = _get_approval_mode()
    except Exception:
        return "deny"
    return {"manual": "ask", "smart": "smart", "off": "approve"}.get(mode, "deny")


def _make_child_approval_callback(
    policy: str,
    ask_fallback: str = "smart",
    *,
    interactive_callback: Any = None,
):
    """Build the non-interactive approval callback for child worker threads.

    Child agents run every command through Hermes' approval engine
    (tools/approval.py): hardline blocks, the permanent allowlist, yolo, and
    smart mode all still apply upstream. This callback only decides what to do
    when a *flagged* command would otherwise prompt a human who isn't present.

    Policy comes from the plugin's own config key
    ``dynamic_workflows.child_approval_policy`` (never delegation.*), so a
    workflow's blast radius is controlled independently of native delegation:

      inherit -> follow Hermes' approvals.mode (resolved before we get here)
      deny    -> refuse flagged commands
      approve -> allow flagged commands (hardline is still blocked upstream)
      smart   -> defer to Hermes' _smart_approve auxiliary-LLM guardian;
                 'escalate' (uncertain) resolves to deny since no human is present
      ask     -> use the captured launching CLI approval callback when present;
                 otherwise degrade to ask_fallback (smart|deny|approve)
    """
    clean = _resolve_inherit_policy((policy or "deny").strip().lower())
    if callable(interactive_callback):
        interactive_callback = _with_read_only_fast_path(interactive_callback)
    if clean == "ask":
        if callable(interactive_callback):
            return interactive_callback
        clean = ask_fallback if ask_fallback in ("smart", "deny", "approve") else "smart"

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
            "workflow child denied flagged command (policy=%s): %s (%s)",
            clean, command, description,
        )
        return "deny"
    return _deny


def _with_read_only_fast_path(callback: Any):
    def _callback(command: str, description: str, **kwargs: Any) -> str:
        try:
            from ..adapters.hooks import is_obviously_read_only_terminal_command

            if is_obviously_read_only_terminal_command(command):
                logger.info(
                    "workflow child auto-approved read-only terminal command: %s (%s)",
                    command,
                    description,
                )
                return "once"
        except Exception:
            pass
        return callback(command, description, **kwargs)

    return _callback


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
        "task_id": lease.task_id,
        "session_id": lease.task_id,
        "hermes_session_id": lease.task_id,
        "workspace": lease.cwd,
        "isolation": lease.isolation or "shared",
        "worktree_path": lease.path,
        "worktree_branch": lease.branch,
        "agent_type": agent_type.name if agent_type else "workflow-subagent",
        "agent_type_source": agent_type.source if agent_type else None,
        "model": getattr(child, "model", None),
        "toolsets": toolsets,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "tokens": prompt_tokens + completion_tokens + reasoning_tokens,
        # Prompt caching is auto-enabled for supported models
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
    real tool calls from the result messages: assistant `tool_calls` and
    provider `tool_use` content blocks.
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
