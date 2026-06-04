"""Persistent storage for workflow scripts and runs."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..engine.errors import WorkflowParseError

RUN_ID_RE = re.compile(r"^wf_[a-z0-9-]{6,}$")


@dataclass(frozen=True)
class WorkflowSource:
    script: str
    source_type: str
    source_ref: str
    saved_script_path: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return f"wf_{uuid.uuid4().hex[:12]}"


class WorkflowStore:
    def __init__(self, root: Path | None = None):
        self.root = root or default_store_root()
        self.runs_dir = self.root / "runs"
        self.scripts_dir = self.root / "scripts"
        self.workflows_dir = self.root / "workflows"
        self.exports_dir = self.root / "exports"
        for path in (self.root, self.runs_dir, self.scripts_dir, self.workflows_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def run_path(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self.runs_dir / f"{run_id}.json"

    def script_path(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self.scripts_dir / f"{run_id}.py"

    def save_script(self, run_id: str, script: str) -> Path:
        path = self.script_path(run_id)
        path.write_text(script, encoding="utf-8")
        return path

    def save_run(self, record: dict[str, Any]) -> None:
        run_id = str(record.get("runId") or "")
        path = self.run_path(run_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_path(run_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("wf_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = self.load_run(path.stem)
            if data:
                runs.append(data)
            if len(runs) >= limit:
                break
        return runs

    def find_named_workflow(self, name: str, cwd: str | None = None) -> Path | None:
        clean = _safe_workflow_name(name)
        candidates: list[Path] = []
        if cwd:
            candidates.append(Path(cwd) / ".hermes" / "workflows" / f"{clean}.py")
        candidates.append(self.workflows_dir / f"{clean}.py")
        plugin_root = Path(__file__).resolve().parent.parent
        candidates.append(plugin_root / "workflows" / f"{clean}.py")
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None


def resolve_workflow_source(
    params: dict[str, Any],
    *,
    store: WorkflowStore,
    cwd: str | None = None,
) -> WorkflowSource:
    script_path = params.get("scriptPath")
    name = params.get("name")
    script = params.get("script")

    if script_path:
        path = _resolve_script_path(str(script_path), cwd)
        return WorkflowSource(
            script=path.read_text(encoding="utf-8"),
            source_type="scriptPath",
            source_ref=str(path),
        )

    if name:
        path = store.find_named_workflow(str(name), cwd=cwd)
        if path is None:
            raise WorkflowParseError(f"unknown workflow name: {name}")
        return WorkflowSource(
            script=path.read_text(encoding="utf-8"),
            source_type="name",
            source_ref=str(name),
        )

    if isinstance(script, str) and script.strip():
        return WorkflowSource(
            script=script,
            source_type="script",
            source_ref="inline",
        )

    raise WorkflowParseError("provide one of script, scriptPath, or name")


def default_store_root() -> Path:
    override = os.getenv("HERMES_DYNAMIC_WORKFLOWS_HOME")
    if override:
        return Path(override).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "dynamic-workflows"
    except Exception:
        return Path.home() / ".hermes" / "dynamic-workflows"


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"invalid workflow run id: {run_id!r}")


def _safe_workflow_name(name: str) -> str:
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("workflow name must not be empty")
    if "/" in clean or "\\" in clean or ".." in clean:
        raise ValueError(f"invalid workflow name: {name!r}")
    return clean


def _resolve_script_path(raw: str, cwd: str | None) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(cwd or ".").resolve() / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise WorkflowParseError(f"workflow scriptPath does not exist: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise WorkflowParseError(f"workflow scriptPath is too large: {path}")
    return path
