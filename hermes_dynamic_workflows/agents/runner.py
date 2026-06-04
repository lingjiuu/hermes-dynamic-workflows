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
from ..engine.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner


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
        _prepare_mcp_tool_registry(self.config)
        child = self._build_agent(request, runtime, toolsets, lease, agent_type)
        return self._run_child_with_timeout(child, request, lease, agent_type, toolsets)

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

        child_prompt = build_child_system_prompt(request, workspace=lease.cwd, agent_type=agent_type)
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

        def _run() -> dict[str, Any]:
            _install_noninteractive_terminal_approval()
            _register_task_cwd(lease.task_id, lease.cwd)
            return child.run_conversation(
                user_message=request.prompt,
                task_id=lease.task_id,
            )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dw-child-agent")
        with self._active_lock:
            self._active_children.append(child)
        future = executor.submit(_run)
        result: dict[str, Any] | None = None
        try:
            result = future.result(timeout=timeout)
            content = str((result or {}).get("final_response") or "")
            metadata = _child_metadata(child, result or {}, lease, agent_type, toolsets)
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


def build_child_system_prompt(
    request: ChildAgentRequest,
    *,
    workspace: str,
    agent_type: AgentTypeSpec | None = None,
) -> str:
    lines = [
        "You are a focused Hermes child agent spawned by a dynamic workflow.",
        "Work only on the delegated task. Do not ask the user questions.",
        "Use available tools when needed, then return a concise final answer.",
        f"Workspace: {workspace}",
    ]
    if request.label:
        lines.append(f"Task label: {request.label}")
    if request.phase:
        lines.append(f"Workflow phase: {request.phase}")
    if request.isolation == "worktree":
        lines.append("You are running in an isolated git worktree. Keep all file operations inside the workspace above.")
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


def _install_noninteractive_terminal_approval() -> None:
    try:
        from tools.terminal_tool import set_approval_callback

        set_approval_callback(_workflow_child_auto_deny)
    except Exception:
        pass


def _workflow_child_auto_deny(command: str, description: str, **kwargs) -> str:
    return "deny"


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
        "tool_calls": _tool_call_count(child, result),
    }
    return metadata


def _int_attr(obj: Any, name: str) -> int:
    value = getattr(obj, name, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _tool_call_count(child: Any, result: dict[str, Any]) -> int:
    try:
        summary = child.get_activity_summary()
        value = summary.get("tool_call_count") or summary.get("api_call_count")
        if isinstance(value, (int, float)):
            return int(value)
    except Exception:
        pass
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if isinstance(message, dict):
            count += len(message.get("tool_calls") or [])
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
