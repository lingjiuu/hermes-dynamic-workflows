"""Slash commands for workflow run inspection and control."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from ..engine.manager import get_run_manager
from ..plugin import registrar


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
        return _save_named_workflow(manager, run_id, parts[2:])
    if subcommand == "export":
        path = " ".join(parts[2:]).strip() or None
        return manager.save_markdown(run_id, path)
    return (
        "Usage: /workflows [list|<runId>|<runId> phase <name|index>|"
        "<runId> agent <id|label>|<runId> save <name> [user|project]|"
        "<runId> export [path]]"
    )


def workflow_stop_command(raw_args: str = "") -> str:
    run_id = (raw_args or "").strip().split()[0] if (raw_args or "").strip() else ""
    if not run_id:
        return "Usage: /workflow-stop <runId>"
    ok = get_run_manager().stop(run_id)
    if ok:
        return f"Stop requested for workflow {run_id}."
    return f"Workflow run not found or already finished: {run_id}"


def _save_named_workflow(manager: Any, run_id: str, rest: list[str]) -> str:
    if not rest:
        return "Usage: /workflows <runId> save <name> [user|project]"
    name = rest[0]
    scope = "project"
    for token in rest[1:]:
        flag = token.lower().lstrip("-")
        if flag in {"user", "global"}:
            scope = "user"
        elif flag in {"project", "local"}:
            scope = "project"
    result = manager.save_named_workflow(
        run_id,
        name,
        scope=scope,
        cwd=os.environ.get("TERMINAL_CWD") or os.getcwd(),
    )
    if not result.get("ok"):
        return result.get("message", "Failed to save workflow")

    saved = result["name"]
    registered = registrar.register_command(
        saved,
        make_named_workflow_handler(saved),
        description=f"Run saved dynamic workflow '{saved}'.",
        args_hint="[args]",
    )
    lines = [f"Saved workflow '{saved}' to {result['path']} ({result['scope']} scope)."]
    if registered:
        lines.append(f"Run it now as /{saved} or via the workflow tool (name: {saved}).")
    else:
        lines.append(
            f"Run it via the workflow tool (name: {saved}); "
            f"/{saved} becomes available on the next start."
        )
    return " ".join(lines)


def make_named_workflow_handler(name: str) -> Callable[..., str]:
    """Build a slash-command handler that launches a saved workflow by name."""

    def _handler(raw_args: str = "") -> str:
        return _start_named_workflow(name, raw_args)

    return _handler


def _start_named_workflow(name: str, raw_args: str = "") -> str:
    params: dict[str, Any] = {"name": name}
    parsed = _parse_command_args(raw_args)
    if parsed is not None:
        params["args"] = parsed
    try:
        record = get_run_manager().start_from_params(
            params,
            cwd=os.environ.get("TERMINAL_CWD") or os.getcwd(),
        )
    except Exception as exc:
        return f"Failed to start workflow '{name}': {type(exc).__name__}: {exc}"
    return (
        f"Workflow '{name}' started: {record['runId']}. "
        f"Use /workflows {record['runId']} to inspect progress."
    )


def _parse_command_args(raw_args: str) -> Any:
    """Best-effort parse of slash-command text into the workflow `args` value.

    A bare token list arrives as text; we try JSON first (so callers can pass
    arrays/objects), then fall back to the raw string, then None when empty.
    """
    text = (raw_args or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def discover_named_workflows(cwd: str | None = None) -> list[str]:
    """Return saved workflow names from project, user store, and bundled dirs.

    Project names win over user/bundled names on collision (first seen wins).
    """
    from ..storage.store import _RESERVED_WORKFLOW_NAMES, default_store_root

    dirs: list[Path] = []
    if cwd:
        dirs.append(Path(cwd) / ".hermes" / "workflows")
    dirs.append(default_store_root() / "workflows")
    plugin_root = Path(__file__).resolve().parent.parent
    dirs.append(plugin_root / "workflows")

    names: list[str] = []
    seen: set[str] = set()
    for directory in dirs:
        try:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.py")):
                stem = path.stem
                if not stem or stem.startswith("_") or stem in _RESERVED_WORKFLOW_NAMES:
                    continue
                if stem not in seen:
                    seen.add(stem)
                    names.append(stem)
        except OSError:
            continue
    return names
