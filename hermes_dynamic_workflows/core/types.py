"""Dataclasses shared by the workflow runtime and display code."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Literal

AgentStatus = Literal["queued", "running", "done", "error", "skipped"]


@dataclass
class AgentRecord:
    id: int
    label: str
    prompt: str
    prompt_preview: str
    phase: str | None = None
    status: AgentStatus = "queued"
    result_preview: str = ""
    error: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    runner: str = "standalone"
    agent_type: str | None = None
    isolation: str | None = None
    workspace: str | None = None
    model: str | None = None
    task_id: str | None = None
    hermes_session_id: str | None = None
    transcript_path: str | None = None
    tool_calls: int = 0
    tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    attempts: int = 0
    structured: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else monotonic()
        return round(max(0.0, end - self.started_at), 3)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "phase": self.phase,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "prompt": self.prompt,
            "prompt_preview": self.prompt_preview,
            "result_preview": self.result_preview,
            "error": self.error,
            "runner": self.runner,
            "agent_type": self.agent_type,
            "isolation": self.isolation,
            "workspace": self.workspace,
            "model": self.model,
            "task_id": self.task_id,
            "hermes_session_id": self.hermes_session_id,
            "transcript_path": self.transcript_path,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "attempts": self.attempts,
            "structured": dict(self.structured),
        }


@dataclass
class PhaseSpec:
    title: str
    detail: str = ""
    model: str | None = None

    def snapshot(self) -> dict[str, Any]:
        data: dict[str, Any] = {"title": self.title}
        if self.detail:
            data["detail"] = self.detail
        if self.model:
            data["model"] = self.model
        return data


@dataclass
class WorkflowFrame:
    id: str
    meta: dict[str, Any]
    args: Any
    cwd: str
    phases: list[PhaseSpec] = field(default_factory=list)
    current_phase: str | None = None
    logs: list[str] = field(default_factory=list)
    agents: list[AgentRecord] = field(default_factory=list)
    children: list["WorkflowFrame"] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    parent_id: str | None = None
    source_ref: str | None = None
    status: str = "running"
    started_at: float = field(default_factory=monotonic)
    ended_at: float | None = None

    @property
    def phase_titles(self) -> list[str]:
        return [phase.title for phase in self.phases]

    def ensure_phase(self, title: str) -> None:
        if title not in self.phase_titles:
            self.phases.append(PhaseSpec(title=title))

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else monotonic()
        return round(max(0.0, end - self.started_at), 3)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "meta": self.meta,
            "cwd": self.cwd,
            "phases": [phase.snapshot() for phase in self.phases],
            "current_phase": self.current_phase,
            "logs": list(self.logs),
            "errors": list(self.errors),
            "agents": [agent.snapshot() for agent in self.agents],
            "children": [child.snapshot() for child in self.children],
            "parent_id": self.parent_id,
            "source_ref": self.source_ref,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class WorkflowState:
    root: WorkflowFrame

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.root.snapshot()
        snapshot["totals"] = workflow_totals(snapshot)
        return snapshot

    @property
    def duration_seconds(self) -> float:
        return self.root.duration_seconds

    @property
    def meta(self) -> dict[str, Any]:
        return self.root.meta

    @property
    def args(self) -> Any:
        return self.root.args

    @property
    def cwd(self) -> str:
        return self.root.cwd

    @property
    def phases(self) -> list[str]:
        return self.root.phase_titles

    @property
    def current_phase(self) -> str | None:
        return self.root.current_phase

    @current_phase.setter
    def current_phase(self, value: str | None) -> None:
        self.root.current_phase = value

    @property
    def logs(self) -> list[str]:
        return self.root.logs

    @property
    def agents(self) -> list[AgentRecord]:
        return self.root.agents

    @property
    def errors(self) -> list[str]:
        return self.root.errors

    @property
    def ended_at(self) -> float | None:
        return self.root.ended_at

    @ended_at.setter
    def ended_at(self, value: float | None) -> None:
        self.root.ended_at = value


@dataclass(frozen=True)
class ChildAgentResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAgentSpec:
    requested_agent_type: str | None
    agent_type_spec: Any = None
    model: str | None = None
    isolation: str | None = None
    toolsets: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    system_prompt_hash: str = ""
    workspace: str = ""

    @property
    def agent_type_name(self) -> str | None:
        value = getattr(self.agent_type_spec, "name", None)
        return str(value) if value else self.requested_agent_type

    def cache_inputs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "isolation": self.isolation,
            "toolsets": list(self.toolsets),
            "allowedTools": list(self.allowed_tools),
            "disallowedTools": list(self.disallowed_tools),
            "agentType": self.agent_type_name,
            "systemPromptHash": self.system_prompt_hash,
            "workspace": self.workspace,
        }


@dataclass(frozen=True)
class ChildAgentRequest:
    id: int
    prompt: str
    label: str
    phase: str | None
    toolsets: list[str]
    model: str | None = None
    schema: dict[str, Any] | None = None
    agent_type: str | None = None
    isolation: str | None = None
    cwd: str | None = None
    request_overrides: dict[str, Any] | None = None
    structured_tool: bool = False
    on_start: Callable[[dict[str, Any]], None] | None = None
    on_update: Callable[[dict[str, Any]], None] | None = None
    resolved: ResolvedAgentSpec | None = None


class ChildAgentRunner:
    """Protocol-like base class for child agent runners."""

    def run(self, request: ChildAgentRequest) -> Any:
        raise NotImplementedError


def normalize_phase_specs(raw: Any) -> list[PhaseSpec]:
    if not raw:
        return []
    phases: list[PhaseSpec] = []
    if not isinstance(raw, list):
        return phases
    for item in raw:
        if isinstance(item, str):
            title = item.strip()
            if title:
                phases.append(PhaseSpec(title=title))
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            detail = str(item.get("detail") or "").strip()
            model = str(item.get("model") or "").strip() or None
            phases.append(PhaseSpec(title=title, detail=detail, model=model))
    return phases


def workflow_totals(snapshot: dict[str, Any]) -> dict[str, int]:
    totals = {
        "agents": 0,
        "done": 0,
        "running": 0,
        "errors": 0,
        "tokens": 0,
        "tool_calls": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    _accumulate_totals(snapshot, totals)
    return totals


def _accumulate_totals(snapshot: dict[str, Any], totals: dict[str, int]) -> None:
    errors = snapshot.get("errors") or []
    totals["errors"] += len(errors)
    for agent in snapshot.get("agents") or []:
        totals["agents"] += 1
        status = agent.get("status")
        if status == "done":
            totals["done"] += 1
        elif status == "running":
            totals["running"] += 1
        elif status == "error":
            totals["errors"] += 1
        try:
            totals["tokens"] += int(agent.get("tokens") or 0)
        except (TypeError, ValueError):
            pass
        try:
            totals["tool_calls"] += int(agent.get("tool_calls") or 0)
        except (TypeError, ValueError):
            pass
        for cache_key in ("cache_read_tokens", "cache_write_tokens"):
            try:
                totals[cache_key] += int(agent.get(cache_key) or 0)
            except (TypeError, ValueError):
                pass
    for child in snapshot.get("children") or []:
        if isinstance(child, dict):
            _accumulate_totals(child, totals)
