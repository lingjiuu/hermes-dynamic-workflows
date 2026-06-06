"""Text rendering for workflow status snapshots."""

from __future__ import annotations

from typing import Any


def preview(value: Any, max_chars: int = 160) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "..."


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


def render_runs_list(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No workflow runs found."
    running = sum(1 for run in runs if run.get("status") in {"queued", "running", "stopping"})
    completed = sum(1 for run in runs if run.get("status") == "completed")
    lines = ["Dynamic workflows", f"{running} running . {completed} completed", ""]
    for index, run in enumerate(runs, start=1):
        snapshot = run.get("workflow") or {}
        meta = snapshot.get("meta") or {}
        name = meta.get("name") or run.get("source", {}).get("ref") or "workflow"
        marker = ">" if index == 1 and run.get("status") in {"queued", "running", "stopping"} else " "
        totals = _totals(snapshot)
        errors = totals.get("errors") or 0
        error_note = f" . {errors} err" if errors else ""
        lines.append(
            f"{marker} {status_icon(run.get('status'))} {name}  "
            f"{totals['agents']} agents . {_format_tokens(totals['tokens'])} tok{error_note} . "
            f"{_format_duration(_duration(run, snapshot))} . {run.get('runId')}"
        )
    lines.extend(
        [
            "",
            "Use: /workflows <runId> . /workflows <runId> phase <name|index> . "
            "/workflows <runId> agent <id|label> . /workflow-stop <runId>",
        ]
    )
    return "\n".join(lines)


def render_run_detail(run: dict[str, Any]) -> str:
    snapshot = run.get("workflow") or {}
    meta = snapshot.get("meta") or {}
    name = meta.get("name") or run.get("source", {}).get("ref") or "workflow"
    description = meta.get("description") or ""
    totals = _totals(snapshot)
    header = (
        f"{totals['done']}/{totals['agents']} agents . "
        f"{_format_tokens(totals['tokens'])} tok"
    )
    if totals.get("cache_read_tokens"):
        header += f" ({_format_tokens(totals['cache_read_tokens'])} cached read)"
    header += (
        f" . {_format_duration(_duration(run, snapshot))} . "
        f"{run.get('status')} . {run.get('runId')}"
    )
    lines = [
        str(name),
        str(description) if description else "",
        header,
        "",
    ]
    _render_frame_detail(lines, snapshot, level=0)
    errors = _all_errors(snapshot)
    if errors:
        lines.append("Errors")
        lines.extend(f"- {preview(error, 180)}" for error in errors)
        lines.append("")
    if run.get("result") is not None:
        lines.append("Result")
        lines.append(preview(run.get("result"), 1000))
    return "\n".join(line for line in lines if line is not None).rstrip()


def render_phase_detail(run: dict[str, Any], selector: str) -> str:
    snapshot = run.get("workflow") or {}
    phases = _phase_names(snapshot, recursive=True)
    phase = _select_phase(phases, selector)
    if phase is None:
        return f"Phase not found: {selector}"
    agents = [agent for agent in _all_agents(snapshot) if agent.get("phase") == phase]
    lines = [f"{phase} . {len(agents)} agents", ""]
    if not agents:
        lines.append("Not started yet")
    else:
        lines.extend(render_agent_row(agent) for agent in agents)
    return "\n".join(lines)


def render_agent_detail(run: dict[str, Any], selector: str) -> str:
    snapshot = run.get("workflow") or {}
    agent = _select_agent(_all_agents(snapshot), selector)
    if agent is None:
        return f"Agent not found: {selector}"
    lines = [
        f"agent #{agent.get('id')} {agent.get('label')}",
        f"Status: {agent.get('status')}",
    ]
    if agent.get("phase"):
        lines.append(f"Phase: {agent.get('phase')}")
    if agent.get("agent_type"):
        lines.append(f"Agent type: {agent.get('agent_type')}")
    if agent.get("isolation"):
        lines.append(f"Isolation: {agent.get('isolation')}")
    if agent.get("workspace"):
        lines.append(f"Workspace: {agent.get('workspace')}")
    if agent.get("model"):
        lines.append(f"Model: {agent.get('model')}")
    if agent.get("hermes_session_id"):
        lines.append(f"Hermes session: {agent.get('hermes_session_id')}")
    if agent.get("transcript_path"):
        lines.append(f"Transcript: {agent.get('transcript_path')}")
    stats = []
    if agent.get("tokens"):
        stats.append(f"{_format_tokens(agent.get('tokens'))} tok")
    if agent.get("cache_read_tokens"):
        stats.append(f"{_format_tokens(agent.get('cache_read_tokens'))} cached read")
    if agent.get("cache_write_tokens"):
        stats.append(f"{_format_tokens(agent.get('cache_write_tokens'))} cache write")
    if agent.get("tool_calls"):
        stats.append(f"{agent.get('tool_calls')} tool calls")
    if int(agent.get("attempts") or 0) > 1:
        stats.append(f"{agent.get('attempts')} attempts")
    if stats:
        lines.append("Stats: " + " . ".join(stats))
    structured = agent.get("structured")
    if isinstance(structured, dict) and structured:
        lines.extend(["", "Structured output"])
        lines.append(f"Status: {structured.get('status') or 'unknown'}")
        if structured.get("mode"):
            lines.append(f"Mode: {structured.get('mode')}")
        if structured.get("attempts") is not None:
            lines.append(f"Attempts: {structured.get('attempts')}")
        if structured.get("error"):
            lines.append(f"Error: {structured.get('error')}")
    lines.extend(["", "Prompt", preview(agent.get("prompt") or agent.get("prompt_preview") or "", 2000)])
    lines.extend(["", "Outcome", agent.get("result_preview") or agent.get("error") or "Still running..."])
    return "\n".join(lines)


def render_saved_markdown(run: dict[str, Any]) -> str:
    return "# Workflow Run\n\n" + render_run_detail(run) + "\n"


def render_agent_row(agent: dict[str, Any]) -> str:
    status = status_icon(agent.get("status"))
    label = agent.get("label") or f"agent-{agent.get('id', '?')}"
    parts = [f"#{agent.get('id')} {status} {label}"]
    if agent.get("model"):
        parts.append(str(agent.get("model")))
    if agent.get("tokens"):
        parts.append(f"{_format_tokens(agent.get('tokens'))} tok")
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
        child_name = child_meta.get("name") or child.get("source_ref") or "subworkflow"
        child_totals = _totals(child)
        parts.append(
            f"{indent}> {child_name} "
            f"({child_totals['done']}/{child_totals['agents']} done)"
        )
        _render_frame_tree(parts, child, indent=indent + "  ", max_agents=max_agents)


def _render_frame_detail(lines: list[str], frame: dict[str, Any], *, level: int) -> None:
    prefix = "  " * level
    if level > 0:
        meta = frame.get("meta") or {}
        name = meta.get("name") or frame.get("source_ref") or "subworkflow"
        totals = _totals(frame)
        lines.append(f"{prefix}> {name} . {totals['done']}/{totals['agents']} agents")
    phases = _phase_names(frame, recursive=False)
    agents = frame.get("agents") or []
    if phases:
        lines.append(f"{prefix}Phases")
        for index, phase in enumerate(phases, start=1):
            phase_agents = [agent for agent in agents if agent.get("phase") == phase]
            phase_done = sum(1 for agent in phase_agents if agent.get("status") == "done")
            suffix = "not started" if not phase_agents else f"{phase_done}/{len(phase_agents)}"
            current = ">" if phase == frame.get("current_phase") else " "
            lines.append(f"{prefix}{current} {index} {phase} {suffix}")
        lines.append("")
    for phase in phases or ["Agents"]:
        phase_agents = agents if phase == "Agents" else [agent for agent in agents if agent.get("phase") == phase]
        if not phase_agents:
            continue
        lines.append(f"{prefix}{phase} . {len(phase_agents)} agents")
        for agent in phase_agents:
            lines.append(prefix + "  " + render_agent_row(agent))
        lines.append("")
    for child in frame.get("children") or []:
        _render_frame_detail(lines, child, level=level + 1)


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


def _select_phase(phases: list[str], selector: str) -> str | None:
    clean = str(selector or "").strip()
    if not clean:
        return None
    if clean.isdigit():
        index = int(clean) - 1
        if 0 <= index < len(phases):
            return phases[index]
    for phase in phases:
        if phase == clean or phase.lower() == clean.lower():
            return phase
    return None


def _select_agent(agents: list[dict[str, Any]], selector: str) -> dict[str, Any] | None:
    clean = str(selector or "").strip()
    if not clean:
        return None
    for agent in agents:
        if str(agent.get("id")) == clean or str(agent.get("label")) == clean:
            return agent
    lowered = clean.lower()
    for agent in agents:
        if str(agent.get("label") or "").lower() == lowered:
            return agent
    return None


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
