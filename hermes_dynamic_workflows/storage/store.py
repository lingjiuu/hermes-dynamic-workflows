"""Persistent storage for workflow scripts and runs."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from ..core.errors import WorkflowParseError

RUN_ID_RE = re.compile(r"^wf_[a-z0-9-]{6,}$")

# Names a saved workflow must not take, because they would shadow the plugin's
# own slash commands / tool.
_RESERVED_WORKFLOW_NAMES = frozenset({"workflows", "workflow"})


@dataclass(frozen=True)
class WorkflowSource:
    script: str
    source_type: str
    source_ref: str
    saved_script_path: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    raw = uuid.uuid4().hex
    return f"wf_{raw[:8]}-{raw[8:11]}"


def new_task_id() -> str:
    return f"wg{_base36(uuid.uuid4().int)[:7]}"


class WorkflowStore:
    def __init__(self, root: Path | None = None):
        self.layout_root = Path(root).expanduser() if root is not None else default_layout_root()
        self.root = Path(root).expanduser() if root is not None else default_store_root()
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

    def session_dir(self, cwd: str | None, session_id: str) -> Path:
        return self.layout_root / "projects" / sanitize_path(cwd or "") / session_id

    def workflow_scripts_dir(self, cwd: str | None, session_id: str) -> Path:
        return self.session_dir(cwd, session_id) / "workflows" / "scripts"

    def workflow_script_path(
        self,
        cwd: str | None,
        session_id: str,
        run_id: str,
        name: str,
    ) -> Path:
        _validate_run_id(run_id)
        stem = slugify_name(name) or "dynamic-workflow"
        return self.workflow_scripts_dir(cwd, session_id) / f"{stem}-{run_id}.py"

    def transcript_dir(self, cwd: str | None, session_id: str, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self.session_dir(cwd, session_id) / "subagents" / "workflows" / run_id

    def task_output_path(self, cwd: str | None, session_id: str, task_id: str) -> Path:
        return (
            Path(os.getenv("HERMES_DYNAMIC_WORKFLOWS_TMPDIR") or gettempdir())
            / f"hermes-{_uid()}"
            / sanitize_path(cwd or "")
            / session_id
            / "tasks"
            / f"{sanitize_filename(task_id)}.output"
        )

    def save_script(self, run_id: str, script: str) -> Path:
        path = self.script_path(run_id)
        path.write_text(script, encoding="utf-8")
        return path

    def save_workflow_script(
        self,
        *,
        cwd: str | None,
        session_id: str,
        run_id: str,
        name: str,
        script: str,
    ) -> Path:
        path = self.workflow_script_path(cwd, session_id, run_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
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

    def list_run_paths(self, limit: int | None = 20) -> list[Path]:
        paths = sorted(
            self.runs_dir.glob("wf_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if limit is None:
            return paths
        return paths[: max(0, limit)]

    def list_runs(self, limit: int = 20, *, session_id: str | None = None) -> list[dict[str, Any]]:
        wanted_session = str(session_id or "").strip()
        runs: list[dict[str, Any]] = []
        for path in self.list_run_paths(limit=None):
            data = self.load_run(path.stem)
            if data and (not wanted_session or str(data.get("workflowSessionId") or "").strip() == wanted_session):
                runs.append(data)
            if len(runs) >= limit:
                break
        return runs

    def find_run_by_task_id(self, task_id: str) -> dict[str, Any] | None:
        wanted = str(task_id)
        for path in sorted(self.runs_dir.glob("wf_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = self.load_run(path.stem)
            if data and str(data.get("taskId") or "") == wanted:
                return data
        return None

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
            saved_script_path=str(path),
        )

    if name:
        path = store.find_named_workflow(str(name), cwd=cwd)
        if path is None:
            raise WorkflowParseError(f"unknown workflow name: {name}")
        return WorkflowSource(
            script=path.read_text(encoding="utf-8"),
            source_type="name",
            source_ref=str(name),
            saved_script_path=str(path),
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
    hermes_home = os.getenv("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser() / "dynamic-workflows"
    try:
        from ..host import session as host_session

        return Path(host_session.hermes_home()) / "dynamic-workflows"
    except Exception:
        return Path.home() / ".hermes" / "dynamic-workflows"


def default_layout_root() -> Path:
    override = os.getenv("HERMES_DYNAMIC_WORKFLOWS_HOME")
    if override:
        return Path(override).expanduser()
    hermes_home = os.getenv("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser()
    try:
        from ..host import session as host_session

        return Path(host_session.hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def sanitize_path(value: str) -> str:
    """Claude-Code-style project directory key: non-alnum characters become '-'."""

    raw = str(value or "").strip() or "unknown-cwd"
    sanitized = re.sub(r"[^a-zA-Z0-9]", "-", raw)
    if len(sanitized) <= 180:
        return sanitized
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:10]
    return f"{sanitized[:180]}-{suffix}"


def slugify_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug[:80].strip("-")


def sanitize_filename(value: str) -> str:
    raw = str(value or "").strip() or "workflow"
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "-", raw).strip(".-")
    return clean[:160] or "workflow"


def _uid() -> str:
    try:
        return str(os.getuid())
    except Exception:
        return "user"


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    chars: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        chars.append(alphabet[rem])
    return "".join(reversed(chars)) or "0"


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
