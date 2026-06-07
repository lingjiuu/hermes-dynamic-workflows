"""Read persisted workflow state into a TUI-friendly model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..storage.control import ControlClient
from ..storage.store import WorkflowStore


ACTIVE_STATUSES = frozenset({"queued", "running", "paused", "stopping"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "error", "stopped"})
JSONL_TAIL_BYTES = 256 * 1024


@dataclass(frozen=True)
class AgentView:
    id: str
    label: str
    status: str
    phase: str
    prompt: str
    outcome: str
    model: str
    tokens: int
    tool_calls: int
    duration_seconds: float | None
    transcript_path: str
    activity: tuple[str, ...]


@dataclass(frozen=True)
class PhaseView:
    title: str
    agents: tuple[AgentView, ...]

    @property
    def done(self) -> int:
        return sum(agent.status == "done" for agent in self.agents)


@dataclass(frozen=True)
class WorkflowView:
    run_id: str
    task_id: str
    name: str
    description: str
    status: str
    current_phase: str
    phases: tuple[PhaseView, ...]
    agents: tuple[AgentView, ...]
    agent_count: int
    tokens: int
    tool_calls: int
    duration_seconds: float
    record: dict[str, Any]

    @property
    def done(self) -> int:
        return sum(agent.status == "done" for agent in self.agents)

    @property
    def running(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def session_id(self) -> str:
        return str(self.record.get("workflowSessionId") or "").strip()

    @property
    def cwd(self) -> str:
        return str(self.record.get("cwd") or "").strip()

    @property
    def started_iso(self) -> str:
        return str(self.record.get("startedAt") or self.record.get("createdAt") or "")


@dataclass(frozen=True)
class SessionGroup:
    """A set of workflow runs that belong to the same Hermes session."""

    key: str
    cwd: str
    project: str
    ago: str
    latest: float
    workflows: tuple[WorkflowView, ...]
    running: int
    done: int
    failed: int
    is_current: bool = False


def group_sessions(workflows: list[WorkflowView], *, now: datetime | None = None) -> list[SessionGroup]:
    """Bucket runs by Hermes session, newest session first.

    Runs without a session id (legacy iterations) are dropped. The "current"
    session is HERMES_SESSION_ID if set, else the most recent session with an
    active run, else the most recently active session overall.
    """
    moment = now or datetime.now(timezone.utc)
    buckets: dict[str, list[WorkflowView]] = {}
    order: list[str] = []
    for workflow in workflows:
        key = workflow.session_id
        if not key:
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(workflow)

    groups: list[SessionGroup] = []
    for key in order:
        runs = buckets[key]
        latest_iso = max((run.started_iso for run in runs), default="")
        latest_dt = _parse_iso(latest_iso)
        groups.append(
            SessionGroup(
                key=key,
                cwd=runs[0].cwd,
                project=_project_name(runs[0].cwd) or "workflow",
                ago=_relative_time(latest_dt, moment),
                latest=latest_dt.timestamp() if latest_dt else 0.0,
                workflows=tuple(runs),
                running=sum(run.running for run in runs),
                done=sum(run.status == "completed" for run in runs),
                failed=sum(run.status in TERMINAL_FAILURE_STATUSES for run in runs),
            )
        )
    groups.sort(key=lambda group: group.latest, reverse=True)

    current = _current_session_key(groups)
    return [replace(group, is_current=group.key == current) for group in groups]


def list_items(groups: list[SessionGroup], expanded: frozenset[str]) -> list[tuple[str, int, int]]:
    """Flat selectable list for the accordion: ('group', gi, -1) and ('run', gi, ri)."""
    items: list[tuple[str, int, int]] = []
    for group_index, group in enumerate(groups):
        items.append(("group", group_index, -1))
        if group.key in expanded:
            for run_index in range(len(group.workflows)):
                items.append(("run", group_index, run_index))
    return items


def _current_session_key(groups: list[SessionGroup]) -> str:
    env = os.getenv("HERMES_SESSION_ID", "").strip()
    if env and any(group.key == env for group in groups):
        return env
    for group in groups:  # sorted newest-first
        if group.running:
            return group.key
    return groups[0].key if groups else ""


def _project_name(cwd: str) -> str:
    return os.path.basename(cwd.rstrip("/")) if cwd else ""


def _relative_time(when: datetime | None, now: datetime) -> str:
    if when is None:
        return "unknown"
    seconds = max(0, int((now - when).total_seconds()))
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        return f"{max(1, seconds // 60)}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


class WorkflowRepository:
    """Reload workflow files on demand so another process can update the TUI."""

    def __init__(
        self,
        store: WorkflowStore | None = None,
        control_client: ControlClient | None = None,
    ):
        self.store = store or WorkflowStore()
        self.control_client = control_client or ControlClient(self.store)
        self._jsonl_reader = _JsonlTailReader()
        # run_id -> (file signature, parsed view). Terminal runs never change, so
        # parsing them once and serving from cache keeps the refresh loop cheap;
        # active runs are always re-parsed so their live duration keeps ticking.
        self._view_cache: dict[str, tuple[tuple[int, int], WorkflowView]] = {}
        # Full (detail) views are heavier (agents/phases/journal) and only built
        # when a workflow is opened — cached separately, same invalidation rule.
        self._detail_cache: dict[str, tuple[tuple[int, int], WorkflowView]] = {}

    def world_version(self) -> int:
        """O(1) change token: the runs dir mtime bumps on every save_run (it
        writes via tmp+rename), so this detects add/remove/rewrite at any scale."""
        try:
            return self.store.runs_dir.stat().st_mtime_ns
        except OSError:
            return 0

    def load(self, limit: int = 50) -> list[WorkflowView]:
        views: list[WorkflowView] = []
        seen: set[str] = set()
        for path in self.store.list_run_paths(limit=limit):
            run_id = path.stem
            try:
                stat = path.stat()
            except OSError:
                continue
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = self._view_cache.get(run_id)
            if cached is not None and cached[0] == signature:
                seen.add(run_id)
                views.append(cached[1])
                continue
            record = self.store.load_run(run_id)
            if not record:
                continue
            view = workflow_view(record, jsonl_reader=self._jsonl_reader, detail=False)
            if view.status not in ACTIVE_STATUSES:
                self._view_cache[run_id] = (signature, view)
            seen.add(run_id)
            views.append(view)
        for stale in [run_id for run_id in self._view_cache if run_id not in seen]:
            self._view_cache.pop(stale, None)
        return views

    def detail(self, run_id: str) -> WorkflowView | None:
        """Full view (agents/phases/activity) for one run, built on demand."""
        try:
            stat = self.store.run_path(run_id).stat()
        except (OSError, ValueError):
            return None
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._detail_cache.get(run_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        record = self.store.load_run(run_id)
        if not record:
            return None
        view = workflow_view(record, jsonl_reader=self._jsonl_reader, detail=True)
        if view.status not in ACTIVE_STATUSES:
            self._detail_cache[run_id] = (signature, view)
        return view

    def save_markdown(self, workflow: WorkflowView) -> Path:
        from ..view.render import render_saved_markdown

        path = self.store.exports_dir / f"{workflow.run_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_saved_markdown(workflow.record), encoding="utf-8")
        return path

    def request_control(self, workflow: WorkflowView, action: str) -> dict[str, Any]:
        return self.control_client.request(
            owner=str(workflow.record.get("controlOwner") or ""),
            run_id=workflow.run_id,
            action=action,
            expected_status=workflow.status,
        )

    def hydrate_agent_activity(
        self,
        workflow: WorkflowView,
        *,
        phase_index: int,
        agent_index: int,
    ) -> WorkflowView:
        if not workflow.phases:
            return workflow
        phase_index = max(0, min(phase_index, len(workflow.phases) - 1))
        phase = workflow.phases[phase_index]
        if not phase.agents:
            return workflow
        agent_index = max(0, min(agent_index, len(phase.agents) - 1))
        agent = phase.agents[agent_index]
        transcript_activity = _read_transcript_activity(agent.transcript_path, self._jsonl_reader)
        if not transcript_activity:
            return workflow
        hydrated = replace(agent, activity=tuple(transcript_activity[-8:]))
        phase_agents = tuple(
            hydrated if index == agent_index else item
            for index, item in enumerate(phase.agents)
        )
        phases = tuple(
            replace(item, agents=phase_agents) if index == phase_index else item
            for index, item in enumerate(workflow.phases)
        )
        agents = tuple(hydrated if item.id == agent.id else item for item in workflow.agents)
        return replace(workflow, phases=phases, agents=agents)


def workflow_view(
    record: dict[str, Any],
    *,
    jsonl_reader: "_JsonlTailReader | None" = None,
    detail: bool = True,
) -> WorkflowView:
    """Build a TUI view of a run.

    With ``detail=False`` only the fields the list/groups need are populated
    (name, status, counts, tokens, duration) — the per-agent ``AgentView``s,
    phases, and journal reads are skipped, which is the bulk of the cost. Full
    detail (agents/phases/activity) is built on demand when a workflow is opened.
    """
    snapshot = record.get("workflow")
    if not isinstance(snapshot, dict):
        snapshot = {}
    meta = snapshot.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    totals = snapshot.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    raw_agents = [agent for agent in _all_agents(snapshot) if isinstance(agent, dict)]

    if detail:
        reader = jsonl_reader or _JsonlTailReader()
        journal = _read_journal(record.get("journalFile"), reader)
        outcomes = _journal_outcomes(record.get("journalFile"), reader)
        agents = tuple(_agent_view(agent, journal, outcomes) for agent in raw_agents)
        phases = _phase_views(snapshot, agents)
    else:
        agents = ()
        phases = ()

    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    return WorkflowView(
        run_id=str(record.get("runId") or ""),
        task_id=str(record.get("taskId") or ""),
        name=str(meta.get("name") or record.get("summary") or source.get("ref") or "dynamic-workflow"),
        description=str(meta.get("description") or record.get("summary") or ""),
        status=str(record.get("status") or "unknown"),
        current_phase=str(snapshot.get("current_phase") or ""),
        phases=phases,
        agents=agents,
        agent_count=len(raw_agents),
        tokens=_as_int(totals.get("tokens")) or sum(_as_int(agent.get("tokens")) for agent in raw_agents),
        tool_calls=_as_int(totals.get("tool_calls")) or sum(_as_int(agent.get("tool_calls")) for agent in raw_agents),
        duration_seconds=_duration_seconds(record, snapshot),
        record=record,
    )


def _agent_view(
    agent: dict[str, Any],
    journal: dict[str, list[str]],
    outcomes: dict[str, str] | None = None,
) -> AgentView:
    agent_id = str(agent.get("id") or "?")
    full_outcome = (outcomes or {}).get(agent_id)
    outcome = full_outcome or str(agent.get("result_preview") or agent.get("error") or "Still running...")
    return AgentView(
        id=agent_id,
        label=str(agent.get("label") or f"agent-{agent_id}"),
        status=str(agent.get("status") or "queued"),
        phase=str(agent.get("phase") or "Agents"),
        prompt=str(agent.get("prompt") or agent.get("prompt_preview") or ""),
        outcome=outcome,
        model=str(agent.get("model") or ""),
        tokens=_as_int(agent.get("tokens")),
        tool_calls=_as_int(agent.get("tool_calls")),
        duration_seconds=_as_duration(agent.get("duration_seconds")),
        transcript_path=str(agent.get("transcript_path") or ""),
        activity=tuple((journal.get(agent_id) or [])[-8:]),
    )


def _phase_views(snapshot: dict[str, Any], agents: tuple[AgentView, ...]) -> tuple[PhaseView, ...]:
    titles: list[str] = []
    for phase in snapshot.get("phases") or []:
        title = phase.get("title") if isinstance(phase, dict) else phase
        clean = str(title or "").strip()
        if clean and clean not in titles:
            titles.append(clean)
    for agent in agents:
        if agent.phase not in titles:
            titles.append(agent.phase)
    if not titles:
        titles.append("Agents")
    return tuple(
        PhaseView(title=title, agents=tuple(agent for agent in agents if agent.phase == title))
        for title in titles
    )


def _all_agents(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    agents = [agent for agent in snapshot.get("agents") or [] if isinstance(agent, dict)]
    for child in snapshot.get("children") or []:
        if isinstance(child, dict):
            agents.extend(_all_agents(child))
    return agents


def _read_journal(raw_path: Any, reader: "_JsonlTailReader") -> dict[str, list[str]]:
    events: dict[str, list[str]] = {}
    for value in reader.read(raw_path):
        agent_id = str(value.get("agentId") or "")
        if not agent_id:
            continue
        event_type = str(value.get("type") or "")
        if event_type == "started":
            text = "Agent started"
        elif event_type == "result":
            text = "Result recorded"
        elif event_type == "error":
            text = f"Error: {_preview(value.get('error'), 90)}"
        elif event_type == "activity" and value.get("activity"):
            text = str(value.get("activity"))
        else:
            continue
        events.setdefault(agent_id, []).append(text)
    return events


def _journal_outcomes(raw_path: Any, reader: "_JsonlTailReader") -> dict[str, str]:
    """Full agent results from the journal's `result` events (the run record only
    keeps a 180-char preview; the complete output is journaled)."""
    outcomes: dict[str, str] = {}
    for value in reader.read(raw_path):
        if str(value.get("type") or "") != "result":
            continue
        agent_id = str(value.get("agentId") or "")
        if not agent_id or value.get("result") is None:
            continue
        outcomes[agent_id] = _result_text(value.get("result"))
    return outcomes


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _read_transcript_activity(raw_path: Any, reader: "_JsonlTailReader") -> list[str]:
    activity: list[str] = []
    for row in reader.read(raw_path):
        message = row.get("message") if isinstance(row.get("message"), dict) else row
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = function.get("name") or call.get("name") or "tool"
                args = function.get("arguments") or call.get("arguments") or call.get("input") or ""
                activity.append(f"{name}({_preview(args, 76)})")
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                activity.append(f"{block.get('name') or 'tool'}({_preview(block.get('input') or '', 76)})")
    return activity


class _JsonlTailReader:
    """Cache stable files and bound I/O for live transcripts rewritten in place."""

    def __init__(self, max_bytes: int = JSONL_TAIL_BYTES, max_entries: int = 256):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._cache: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}

    def read(self, raw_path: Any) -> list[dict[str, Any]]:
        if not raw_path:
            return []
        path = Path(str(raw_path))
        try:
            stat = path.stat()
        except OSError:
            return []
        cached = self._cache.get(path)
        signature = (stat.st_mtime_ns, stat.st_size)
        if cached and cached[:2] == signature:
            return cached[2]
        try:
            with path.open("rb") as handle:
                offset = max(0, stat.st_size - self.max_bytes)
                handle.seek(offset)
                data = handle.read(self.max_bytes)
        except OSError:
            return []
        if offset:
            _, _, data = data.partition(b"\n")
        values: list[dict[str, Any]] = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                values.append(value)
        self._cache[path] = (signature[0], signature[1], values)
        while len(self._cache) > self.max_entries:
            self._cache.pop(next(iter(self._cache)))
        return values


def _duration_seconds(record: dict[str, Any], snapshot: dict[str, Any]) -> float:
    try:
        duration = float(snapshot.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if record.get("status") not in ACTIVE_STATUSES:
        return max(0.0, duration)
    started = _parse_iso(record.get("startedAt") or record.get("createdAt"))
    if started is None:
        return max(0.0, duration)
    return max(duration, (datetime.now(timezone.utc) - started).total_seconds())


def _parse_iso(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _preview(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_duration(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None
