"""Slash commands for workflow run inspection and control."""

from __future__ import annotations

from ..engine.manager import get_run_manager


def workflows_command(raw_args: str = "") -> str:
    arg = (raw_args or "").strip()
    manager = get_run_manager()
    if not arg:
        return manager.format_list(limit=12)
    if arg in {"list", "ls"}:
        return manager.format_list(limit=20)
    parts = arg.split()
    run_id = parts[0]
    if len(parts) == 1:
        return manager.format_detail(run_id)
    subcommand = parts[1].lower()
    if subcommand == "phase":
        if len(parts) < 3:
            return "Usage: /workflows <runId> phase <name|index>"
        return manager.format_phase(run_id, " ".join(parts[2:]))
    if subcommand == "agent":
        if len(parts) < 3:
            return "Usage: /workflows <runId> agent <id|label>"
        return manager.format_agent(run_id, " ".join(parts[2:]))
    if subcommand == "save":
        path = " ".join(parts[2:]).strip() or None
        return manager.save_markdown(run_id, path)
    return (
        "Usage: /workflows [list|<runId>|<runId> phase <name|index>|"
        "<runId> agent <id|label>|<runId> save [path]]"
    )


def workflow_stop_command(raw_args: str = "") -> str:
    run_id = (raw_args or "").strip().split()[0] if (raw_args or "").strip() else ""
    if not run_id:
        return "Usage: /workflow-stop <runId>"
    ok = get_run_manager().stop(run_id)
    if ok:
        return f"Stop requested for workflow {run_id}."
    return f"Workflow run not found or already finished: {run_id}"
