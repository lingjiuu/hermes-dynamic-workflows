"""Text rendering for workflow status snapshots."""

from __future__ import annotations

from typing import Any

from ..core.text import preview


def render_workflow_text(snapshot: dict[str, Any], *, completed: bool = True, max_agents: int = 12) -> str:
    meta = snapshot.get("meta") or {}
    name = meta.get("name") or "dynamic-workflow"
    errors = _all_errors(snapshot)
    totals = _totals(snapshot)
    status = "completed" if completed else "running"

    header = f"- Workflow: {name} ({totals['done']}/{totals['agents']} done)"
    if totals["running"]:
        header += f", {totals['running']} running"
    if errors:
        header += f", {len(errors)} error(s)"

    parts = [f"Workflow {status}", header]
    _render_frame_tree(parts, snapshot, indent="  ", max_agents=max_agents)
    return "\n".join(parts)


def render_agent_overview(runs: list[dict[str, Any]], *, max_agents_per_run: int = 6) -> str:
    if not runs:
        return "No workflow runs found.\n\nRun `hermes-workflows` in a terminal for live monitoring and controls."
    running = sum(1 for run in runs if run.get("status") in {"queued", "running", "paused", "stopping"})
    completed = sum(1 for run in runs if run.get("status") == "completed")
    lines = ["Dynamic workflows", f"{running} running . {completed} completed", ""]
    for run in runs:
        snapshot = run.get("workflow") or {}
        meta = snapshot.get("meta") or {}
        name = meta.get("name") or run.get("source", {}).get("ref") or "workflow"
        totals = _totals(snapshot)
        errors = totals.get("errors") or 0
        if errors:
            status_line = f"{totals['done']}/{totals['agents']} agents done . {errors} err"
        else:
            status_line = f"{totals['done']}/{totals['agents']} agents done"
        if totals["running"]:
            status_line += f" . {totals['running']} running"
        status_line += (
            f" . {_format_tokens(totals['tokens'])} tokens . "
            f"{_format_duration(_duration(run, snapshot))} . {run.get('status')}"
        )
        task_id = str(run.get("taskId") or "")
        task_part = f"Task: {task_id} . " if task_id else ""
        lines.append(f"{status_icon(run.get('status'))} {name}  {run.get('runId')}")
        lines.append(f"  {task_part}{status_line}")
        agents = _all_agents(snapshot)
        if agents:
            for agent in agents[:max_agents_per_run]:
                lines.append(f"  - {render_agent_row(agent)}")
            hidden = len(agents) - max_agents_per_run
            if hidden > 0:
                lines.append(f"  - ... {hidden} more agent(s)")
        else:
            lines.append("  - no agents started")
        lines.append("")
    lines.append("Run `hermes-workflows` in a terminal for live monitoring and controls.")
    return "\n".join(lines).rstrip()


def render_saved_markdown(run: dict[str, Any]) -> str:
    snapshot = run.get("workflow") or {}
    completed = run.get("status") not in {"queued", "running", "paused", "stopping"}
    lines = ["# Workflow Run", "", render_workflow_text(snapshot, completed=completed), ""]
    if run.get("result") is not None:
        lines.extend(["## Result", "", preview(run.get("result"), 4000), ""])
    errors = _all_errors(snapshot)
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {preview(error, 300)}" for error in errors)
        lines.append("")
    return "\n".join(lines)


def render_agent_row(agent: dict[str, Any]) -> str:
    status = status_icon(agent.get("status"))
    label = agent.get("label") or f"agent-{agent.get('id', '?')}"
    parts = [f"#{agent.get('id')} {status} {label}"]
    if agent.get("model"):
        parts.append(str(agent.get("model")))
    if agent.get("tokens"):
        parts.append(f"{_format_tokens(agent.get('tokens'))} tok")
    if agent.get("cache_read_tokens"):
        parts.append(f"{_format_tokens(agent.get('cache_read_tokens'))} cached read")
    if agent.get("cache_write_tokens"):
        parts.append(f"{_format_tokens(agent.get('cache_write_tokens'))} cache write")
    if agent.get("tool_calls"):
        parts.append(f"{agent.get('tool_calls')} tools")
    if agent.get("agent_type"):
        parts.append(f"type:{agent.get('agent_type')}")
    if agent.get("isolation") == "worktree":
        parts.append("worktree")
    structured = agent.get("structured")
    if isinstance(structured, dict):
        structured_status = structured.get("status")
        if structured_status == "failed":
            parts.append("schema failed")
    if agent.get("error"):
        parts.append(preview(agent.get("error"), 120))
    return " . ".join(parts)


def status_icon(status: Any) -> str:
    return {
        "queued": ".",
        "running": "*",
        "stopping": "~",
        "paused": "=",
        "completed": "+",
        "done": "+",
        "error": "!",
        "failed": "!",
        "stopped": "x",
        "skipped": "-",
    }.get(str(status or ""), "?")


def _render_frame_tree(parts: list[str], frame: dict[str, Any], *, indent: str, max_agents: int) -> None:
    phases = _phase_names(frame, recursive=False)
    rendered_ids: set[Any] = set()
    agents = frame.get("agents") or []
    for phase in phases:
        parts.append(f"{indent}[{phase}]")
        for agent in agents[-max_agents:]:
            if agent.get("phase") == phase:
                parts.append(_render_agent(agent, indent + "  "))
                rendered_ids.add(agent.get("id"))
    unphased = [agent for agent in agents[-max_agents:] if agent.get("id") not in rendered_ids]
    if unphased:
        if phases:
            parts.append(f"{indent}[Other]")
        for agent in unphased:
            parts.append(_render_agent(agent, indent + "  "))
    hidden = max(0, len(agents) - max_agents)
    if hidden:
        parts.append(f"{indent}... {hidden} earlier agent(s)")
    for line in (frame.get("logs") or [])[-5:]:
        parts.append(f"{indent}log: {preview(line, 120)}")
    for child in frame.get("children") or []:
        child_meta = child.get("meta") or {}
        child_name = child_meta.get("name") or child.get("source_ref") or "workflow"
        child_totals = _totals(child)
        parts.append(
            f"{indent}> {child_name} "
            f"({child_totals['done']}/{child_totals['agents']} done)"
        )
        _render_frame_tree(parts, child, indent=indent + "  ", max_agents=max_agents)


def _render_agent(agent: dict[str, Any], indent: str) -> str:
    marker = {
        "queued": ".",
        "running": "*",
        "done": "+",
        "error": "!",
        "skipped": "-",
    }.get(agent.get("status"), "?")
    label = agent.get("label") or f"agent-{agent.get('id', '?')}"
    line = f"{indent}#{agent.get('id')} {marker} {label}"
    structured = agent.get("structured")
    if isinstance(structured, dict) and structured.get("status") == "failed":
        line += f" [{structured.get('status')}]"
    if agent.get("error"):
        line += f" - {preview(agent['error'], 100)}"
    return line


def _phase_names(snapshot: dict[str, Any], *, recursive: bool) -> list[str]:
    phases: list[str] = []
    for phase in snapshot.get("phases") or []:
        if isinstance(phase, dict):
            title = str(phase.get("title") or "").strip()
        else:
            title = str(phase or "").strip()
        if title and title not in phases:
            phases.append(title)
    for agent in snapshot.get("agents") or []:
        phase = agent.get("phase")
        if phase and phase not in phases:
            phases.append(str(phase))
    if recursive:
        for child in snapshot.get("children") or []:
            for phase in _phase_names(child, recursive=True):
                if phase not in phases:
                    phases.append(phase)
    return phases


def _all_agents(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    agents = list(snapshot.get("agents") or [])
    for child in snapshot.get("children") or []:
        if isinstance(child, dict):
            agents.extend(_all_agents(child))
    return agents


def _all_errors(snapshot: dict[str, Any]) -> list[str]:
    errors = [str(error) for error in snapshot.get("errors") or []]
    for child in snapshot.get("children") or []:
        if isinstance(child, dict):
            errors.extend(_all_errors(child))
    return errors


def _duration(run: dict[str, Any], snapshot: dict[str, Any]) -> float:
    value = snapshot.get("duration_seconds")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _totals(snapshot: dict[str, Any]) -> dict[str, int]:
    provided = snapshot.get("totals")
    if isinstance(provided, dict):
        return {
            "agents": _as_int(provided.get("agents")),
            "done": _as_int(provided.get("done")),
            "running": _as_int(provided.get("running")),
            "errors": _as_int(provided.get("errors")),
            "tokens": _as_int(provided.get("tokens")),
            "tool_calls": _as_int(provided.get("tool_calls")),
            "cache_read_tokens": _as_int(provided.get("cache_read_tokens")),
            "cache_write_tokens": _as_int(provided.get("cache_write_tokens")),
        }
    agents = _all_agents(snapshot)
    return {
        "agents": len(agents),
        "done": sum(1 for agent in agents if agent.get("status") == "done"),
        "running": sum(1 for agent in agents if agent.get("status") == "running"),
        "errors": len(_all_errors(snapshot)) + sum(1 for agent in agents if agent.get("status") == "error"),
        "tokens": sum(_as_int(agent.get("tokens")) for agent in agents),
        "tool_calls": sum(_as_int(agent.get("tool_calls")) for agent in agents),
        "cache_read_tokens": sum(_as_int(agent.get("cache_read_tokens")) for agent in agents),
        "cache_write_tokens": sum(_as_int(agent.get("cache_write_tokens")) for agent in agents),
    }


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_tokens(value: Any) -> str:
    number = _as_int(value)
    if number >= 1000:
        return f"{number / 1000:.1f}K"
    return str(number)


def _format_duration(seconds: Any) -> str:
    try:
        total = int(float(seconds or 0))
    except (TypeError, ValueError):
        total = 0
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
