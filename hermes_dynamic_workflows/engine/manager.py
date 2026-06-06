"""Background workflow run manager."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import ResumeCache
from .config import PluginConfig, load_config
from .errors import WorkflowLaunchDenied, WorkflowRuntimeError, WorkflowToolUseError
from .sandbox import extract_meta, parse_script
from .token_budget import parse_token_budget
from ..storage.store import (
    WorkflowStore,
    new_run_id,
    new_task_id,
    resolve_workflow_source,
    sanitize_filename,
    utc_now_iso,
)
from ..ui.display import (
    render_agent_detail,
    render_phase_detail,
    render_run_detail,
    render_runs_list,
    render_saved_markdown,
    render_workflow_text,
)
from .runtime import WorkflowOptions, run_workflow


@dataclass
class ManagedRun:
    run_id: str
    stop_event: threading.Event
    record: dict[str, Any]
    thread: threading.Thread | None = None
    child_runner: Any = None
    transcript_exporters: dict[str, "LiveTranscriptExporter"] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


class LiveTranscriptExporter:
    """Keep a child agent transcript file refreshed from Hermes SessionDB."""

    def __init__(
        self,
        *,
        session_id: str,
        transcript_path: Path,
        meta_path: Path,
        metadata: dict[str, Any],
        interval_seconds: float = 0.5,
    ) -> None:
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.meta_path = meta_path
        self.interval_seconds = interval_seconds
        self._metadata = dict(metadata)
        self._metadata_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"workflow-transcript-{sanitize_filename(session_id)[:32]}",
            daemon=True,
        )

    def start(self) -> None:
        self.flush()
        self._thread.start()

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        with self._metadata_lock:
            self._metadata.update(metadata)

    def stop(self, *, final: bool = True) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        if final:
            self.flush()

    def flush(self) -> None:
        messages = _load_session_messages(self.session_id)
        with self._metadata_lock:
            metadata = dict(self._metadata)
        _write_agent_transcript_files(
            self.transcript_path,
            self.meta_path,
            metadata=metadata,
            messages=messages,
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.flush()
            except Exception:
                pass
        try:
            self.flush()
        except Exception:
            pass


class WorkflowRunManager:
    def __init__(self, store: WorkflowStore | None = None, config: PluginConfig | None = None):
        self.store = store or WorkflowStore()
        self.config = config or load_config()
        self._runs: dict[str, ManagedRun] = {}
        self._lock = threading.RLock()

    def start_from_params(
        self,
        params: dict[str, Any],
        *,
        cwd: str | None = None,
        plugin_context: Any = None,
        tool_use_id: str | None = None,
        host_session_id: str | None = None,
        user_task: str | None = None,
    ) -> dict[str, Any]:
        config = self.config
        cwd_value = cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
        source = resolve_workflow_source(params, store=self.store, cwd=cwd)
        meta = extract_meta(parse_script(source.script, config))
        resume_from = str(params.get("resumeFromRunId") or "").strip() or None
        active_resume = self._active_resume_run(resume_from) if resume_from else None
        if active_resume:
            active_task_id = str(active_resume.get("taskId") or "")
            raise WorkflowToolUseError(
                f"Workflow {resume_from} is still running (task {active_task_id}). "
                f'Stop it first with task_stop({{"task_id":"{active_task_id}"}}) '
                "before resuming."
            )
        approved, reason = _approve_launch(meta, config, plugin_context)
        if not approved:
            raise WorkflowLaunchDenied(
                f'Workflow "{meta.get("name") or "workflow"}" was not launched: {reason}. '
                "Do not retry; tell the user it needs their approval."
            )
        run_id = new_run_id()
        task_id = new_task_id()
        workflow_session_id = _resolve_workflow_session_id(
            plugin_context,
            host_session_id=host_session_id,
        )
        saved_path = self._script_path_for_source(
            source,
            run_id=run_id,
            session_id=workflow_session_id,
            cwd=cwd_value,
            meta=meta,
        )
        transcript_dir = self.store.transcript_dir(cwd_value, workflow_session_id, run_id)
        journal_path = transcript_dir / "journal.jsonl"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        journal_path.touch(exist_ok=True)
        previous = self.store.load_run(resume_from) if resume_from else None
        resume_cache = ResumeCache.from_run(previous)
        args = params["args"] if "args" in params else None
        token_budget = parse_token_budget(user_task)
        # Captured in the launching (parent) context, which carries the gateway
        # session vars when the run is started from a gateway session.
        session_context = _capture_gateway_session_context()

        stop_event = threading.Event()
        record = {
            "runId": run_id,
            "taskId": task_id,
            "toolUseId": tool_use_id,
            "status": "queued",
            "createdAt": utc_now_iso(),
            "startedAt": None,
            "finishedAt": None,
            "cwd": cwd_value,
            "workflowSessionId": workflow_session_id,
            "scriptPath": str(saved_path),
            "transcriptDir": str(transcript_dir),
            "journalFile": str(journal_path),
            "summary": meta.get("description") or meta.get("name") or "workflow",
            "source": {
                "type": source.source_type,
                "ref": source.source_ref,
            },
            "resumeFromRunId": resume_from,
            "args": args,
            "tokenBudget": token_budget,
            "result": None,
            "error": None,
            "display": "",
            "workflow": None,
            "agentCache": {},
            "outputFile": None,
            "transcriptFiles": [],
            "transcriptMetaFiles": [],
        }
        managed = ManagedRun(run_id=run_id, stop_event=stop_event, record=record)

        with self._lock:
            self._runs[run_id] = managed
        self.store.save_run(record)

        thread = threading.Thread(
            target=self._run_thread,
            args=(managed, source.script, args, config, resume_cache, cwd, plugin_context, token_budget, session_context),
            name=f"workflow-{run_id}",
            daemon=True,
        )
        managed.thread = thread
        thread.start()
        return self._public_record(record)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            managed = self._runs.get(run_id)
        if managed:
            with managed.lock:
                return self._public_record(dict(managed.record))
        record = self.store.load_run(run_id)
        return self._public_record(record) if record else None

    def get_by_task_id(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            managed_runs = list(self._runs.values())
        for managed in managed_runs:
            with managed.lock:
                if str(managed.record.get("taskId") or "") == str(task_id):
                    return self._public_record(dict(managed.record))
        record = self.store.find_run_by_task_id(str(task_id))
        return self._public_record(record) if record else None

    def _active_resume_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            managed = self._runs.get(run_id)
        if not managed:
            return None
        with managed.lock:
            if managed.record.get("status") in {"queued", "running", "stopping"}:
                return self._public_record(dict(managed.record))
        return None

    def stop_task(self, task_id: str) -> dict[str, Any] | None:
        """Stop an active background workflow by Claude-style task id.

        This intentionally only checks live managed runs. Historical runs in
        state are not stoppable tasks, so stopping a completed or already
        stopped task should look like Claude Code's "No task found" result.
        """
        wanted = str(task_id or "")
        if not wanted:
            return None
        with self._lock:
            managed_runs = list(self._runs.values())
        for managed in managed_runs:
            child_runner = None
            with managed.lock:
                record = managed.record
                if str(record.get("taskId") or "") != wanted:
                    continue
                if record.get("status") not in {"queued", "running"}:
                    return None
                managed.stop_event.set()
                child_runner = managed.child_runner
                record["status"] = "stopping"
                self.store.save_run(record)
                summary = str(record.get("summary") or record.get("runId") or wanted)
                result = {
                    "message": f"Successfully stopped task: {wanted} ({summary})",
                    "task_id": wanted,
                    "task_type": "local_workflow",
                }
            if child_runner is not None and hasattr(child_runner, "interrupt_all"):
                try:
                    child_runner.interrupt_all()
                except Exception:
                    pass
            return result
        return None

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return [self._public_record(run) for run in self.store.list_runs(limit=limit)]

    def stop(self, run_id: str) -> bool:
        with self._lock:
            managed = self._runs.get(run_id)
        if not managed:
            record = self.store.load_run(run_id)
            if not record or record.get("status") not in {"queued", "running"}:
                return False
            record["status"] = "stopped"
            record["finishedAt"] = utc_now_iso()
            self.store.save_run(record)
            return True

        managed.stop_event.set()
        child_runner = managed.child_runner
        if child_runner is not None and hasattr(child_runner, "interrupt_all"):
            try:
                child_runner.interrupt_all()
            except Exception:
                pass
        with managed.lock:
            if managed.record.get("status") in {"queued", "running"}:
                managed.record["status"] = "stopping"
                self.store.save_run(managed.record)
        return True

    def format_list(self, limit: int = 10) -> str:
        runs = self.list(limit=limit)
        return render_runs_list(runs)

    def format_detail(self, run_id: str) -> str:
        run = self.get(run_id)
        if not run:
            return f"Workflow run not found: {run_id}"
        return render_run_detail(run)

    def format_phase(self, run_id: str, selector: str) -> str:
        run = self.get(run_id)
        if not run:
            return f"Workflow run not found: {run_id}"
        return render_phase_detail(run, selector)

    def format_agent(self, run_id: str, selector: str) -> str:
        run = self.get(run_id)
        if not run:
            return f"Workflow run not found: {run_id}"
        return render_agent_detail(run, selector)

    def _script_path_for_source(
        self,
        source,
        *,
        run_id: str,
        session_id: str,
        cwd: str,
        meta: dict[str, Any],
    ) -> Path:
        if source.source_type == "script":
            return self.store.save_workflow_script(
                cwd=cwd,
                session_id=session_id,
                run_id=run_id,
                name=str(meta.get("name") or "dynamic-workflow"),
                script=source.script,
            )
        if source.saved_script_path:
            return Path(source.saved_script_path)
        return self.store.save_workflow_script(
            cwd=cwd,
            session_id=session_id,
            run_id=run_id,
            name=str(meta.get("name") or "dynamic-workflow"),
            script=source.script,
        )

    def save_markdown(self, run_id: str, path: str | None = None) -> str:
        run = self.get(run_id)
        if not run:
            return f"Workflow run not found: {run_id}"
        if path:
            target = Path(path).expanduser()
            if not target.is_absolute():
                target = Path(run.get("cwd") or os.getcwd()) / target
        else:
            target = self.store.exports_dir / f"{run_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_saved_markdown(run), encoding="utf-8")
        return f"Saved workflow {run_id} to {target}"

    def save_named_workflow(
        self,
        run_id: str,
        name: str,
        *,
        scope: str = "project",
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Save a run's script as a reusable named workflow.

        Writes to ``<cwd>/.hermes/workflows/<name>.py`` (project scope) or the
        user store's ``workflows/<name>.py`` (user scope). Either location is
        resolvable later by passing ``name`` to the workflow tool, and the
        caller can register a ``/<name>`` slash command for it.
        """
        from ..storage.store import _RESERVED_WORKFLOW_NAMES, _safe_workflow_name

        run = self.get(run_id)
        if not run:
            return {"ok": False, "message": f"Workflow run not found: {run_id}"}
        script = self._load_run_script(run, run_id)
        if not script:
            return {"ok": False, "message": f"No saved script found for run {run_id}"}
        try:
            safe = _safe_workflow_name(name)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}
        if safe in _RESERVED_WORKFLOW_NAMES:
            return {"ok": False, "message": f"'{safe}' is reserved; choose another name"}

        if scope == "user":
            target = self.store.workflows_dir / f"{safe}.py"
        else:
            base = Path(cwd or run.get("cwd") or os.getcwd()).expanduser()
            target = base / ".hermes" / "workflows" / f"{safe}.py"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(script, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "message": f"Could not write {target}: {exc}"}
        return {"ok": True, "name": safe, "path": str(target), "scope": scope}

    def _load_run_script(self, run: dict[str, Any], run_id: str) -> str | None:
        candidates: list[Path] = []
        script_path = run.get("scriptPath")
        if script_path:
            candidates.append(Path(script_path))
        try:
            candidates.append(self.store.script_path(run_id))
        except Exception:
            pass
        for candidate in candidates:
            try:
                if candidate and candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            except OSError:
                continue
        return None

    def wait(self, run_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        with self._lock:
            managed = self._runs.get(run_id)
        if managed and managed.thread:
            managed.thread.join(timeout=timeout)
        return self.get(run_id)

    def _run_thread(
        self,
        managed: ManagedRun,
        script: str,
        args: Any,
        config: PluginConfig,
        resume_cache: ResumeCache,
        cwd: str | None,
        plugin_context: Any,
        token_budget: int | None = None,
        session_context: dict[str, str] | None = None,
    ) -> None:
        try:
            from ..agents.runner import HermesChildAgentRunner

            managed.child_runner = HermesChildAgentRunner(config, session_context=session_context)
            self._update(managed, status="running", startedAt=utc_now_iso())
            result = run_workflow(
                script,
                WorkflowOptions(
                    args=args,
                    cwd=cwd or os.environ.get("TERMINAL_CWD") or os.getcwd(),
                    config=config,
                    child_runner=managed.child_runner,
                    stop_event=managed.stop_event,
                    resume_cache=resume_cache,
                    on_update=lambda state: self._update_state(managed, state),
                    on_journal=lambda event: self._append_journal_event(managed, event),
                    plugin_context=plugin_context,
                    token_budget_total=token_budget,
                    source_ref=str(managed.record.get("scriptPath") or ""),
                ),
            )
            snapshot = result.state.snapshot()
            self._sync_live_child_transcripts(managed, snapshot)
            if managed.stop_event.is_set():
                status = "stopped"
            else:
                status = _derive_run_status(snapshot)
            self._update(
                managed,
                status=status,
                finishedAt=utc_now_iso(),
                result=result.value,
                workflow=snapshot,
                display=render_workflow_text(snapshot, completed=True),
                agentCache=resume_cache.current,
            )
        except BaseException as exc:
            # BaseException so a WorkflowHalt (stop / deadline / hard limit),
            # which derives from BaseException, is recorded as the run's final
            # status instead of dying as an unhandled thread exception.
            status = "stopped" if managed.stop_event.is_set() else "error"
            self._update(
                managed,
                status=status,
                finishedAt=utc_now_iso(),
                error=f"{type(exc).__name__}: {exc}",
                agentCache=resume_cache.current,
            )
        finally:
            self._stop_live_transcript_exporters(managed)
            self._finalize_completion_artifacts(managed)
            _notify_completion(plugin_context, managed.record, config)

    def _finalize_completion_artifacts(self, managed: ManagedRun) -> None:
        with managed.lock:
            record = managed.record
            try:
                _write_output_file(record, self.store)
            except Exception:
                pass
            try:
                _export_child_transcripts(record, self.store)
            except Exception as exc:
                record["transcriptExportError"] = f"{type(exc).__name__}: {exc}"
            self.store.save_run(record)

    def _update_state(self, managed: ManagedRun, state) -> None:
        snapshot = state.snapshot()
        self._sync_live_child_transcripts(managed, snapshot)
        self._update(
            managed,
            workflow=snapshot,
            display=render_workflow_text(snapshot, completed=False),
        )

    def _sync_live_child_transcripts(self, managed: ManagedRun, snapshot: dict[str, Any]) -> None:
        transcript_dir_raw = managed.record.get("transcriptDir")
        if not transcript_dir_raw:
            return
        transcript_dir = Path(str(transcript_dir_raw))
        transcript_dir.mkdir(parents=True, exist_ok=True)
        with managed.lock:
            transcript_files = managed.record.setdefault("transcriptFiles", [])
            meta_files = managed.record.setdefault("transcriptMetaFiles", [])
            for agent in _iter_agent_snapshots(snapshot):
                session_id = _agent_session_id(agent)
                if not session_id:
                    continue
                path = _agent_transcript_path(transcript_dir, session_id)
                meta_path = _agent_meta_path(path)
                metadata = _agent_transcript_metadata(managed.record, agent, session_id)
                agent["transcript_path"] = str(path)
                agent["transcript_meta_path"] = str(meta_path)
                _append_unique(transcript_files, str(path))
                _append_unique(meta_files, str(meta_path))
                exporter = managed.transcript_exporters.get(session_id)
                if exporter is None:
                    exporter = LiveTranscriptExporter(
                        session_id=session_id,
                        transcript_path=path,
                        meta_path=meta_path,
                        metadata=metadata,
                    )
                    managed.transcript_exporters[session_id] = exporter
                    try:
                        exporter.start()
                    except Exception as exc:
                        managed.record["transcriptExportError"] = f"{type(exc).__name__}: {exc}"
                else:
                    exporter.update_metadata(metadata)

    def _stop_live_transcript_exporters(self, managed: ManagedRun) -> None:
        with managed.lock:
            exporters = list(managed.transcript_exporters.values())
        for exporter in exporters:
            try:
                exporter.stop(final=True)
            except Exception as exc:
                with managed.lock:
                    managed.record["transcriptExportError"] = f"{type(exc).__name__}: {exc}"

    def _update(self, managed: ManagedRun, **fields: Any) -> None:
        with managed.lock:
            managed.record.update(fields)
            self.store.save_run(managed.record)

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    def _append_journal_event(self, managed: ManagedRun, event: dict[str, Any]) -> None:
        with managed.lock:
            path_raw = str(managed.record.get("journalFile") or "")
            if not path_raw:
                return
            path = Path(path_raw)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except Exception as exc:
                managed.record["journalError"] = f"{type(exc).__name__}: {exc}"


def _content_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _completion_output_text(record: dict[str, Any]) -> str:
    if record.get("result") is not None:
        return _content_from_value(record.get("result"))
    if record.get("error"):
        return str(record.get("error") or "")
    return ""


def _write_output_file(record: dict[str, Any], store: WorkflowStore) -> None:
    text = _completion_output_text(record)
    if not text:
        return
    task_id = str(record.get("taskId") or record.get("runId") or "")
    session_id = str(record.get("workflowSessionId") or "")
    cwd = str(record.get("cwd") or "")
    if not task_id or not session_id:
        return
    path = store.task_output_path(cwd, session_id, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    record["outputFile"] = str(path)


def _export_child_transcripts(record: dict[str, Any], store: WorkflowStore) -> None:
    snapshot = record.get("workflow")
    if not isinstance(snapshot, dict):
        return
    transcript_dir_raw = record.get("transcriptDir")
    if not transcript_dir_raw:
        return
    transcript_dir = Path(str(transcript_dir_raw))
    transcript_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    meta_files: list[str] = []
    for agent in _iter_agent_snapshots(snapshot):
        session_id = _agent_session_id(agent)
        if not session_id:
            continue
        path = _agent_transcript_path(transcript_dir, session_id)
        meta_path = _agent_meta_path(path)
        _write_agent_transcript(path, record=record, agent=agent, session_id=session_id)
        agent["transcript_path"] = str(path)
        agent["transcript_meta_path"] = str(meta_path)
        files.append(str(path))
        meta_files.append(str(meta_path))
    if files:
        record["transcriptFiles"] = files
        record["transcriptMetaFiles"] = meta_files
        record["transcriptsExportedAt"] = utc_now_iso()


def _write_agent_transcript(
    path: Path,
    *,
    record: dict[str, Any],
    agent: dict[str, Any],
    session_id: str,
) -> None:
    messages = _load_session_messages(session_id)
    _write_agent_transcript_files(
        path,
        _agent_meta_path(path),
        metadata=_agent_transcript_metadata(record, agent, session_id),
        messages=messages,
    )


def _write_agent_transcript_files(
    path: Path,
    meta_path: Path,
    *,
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(meta_path, metadata)
    lines = [
        json.dumps({"type": "message", "message": message}, ensure_ascii=False, default=str)
        for message in messages
    ]
    _write_text_atomic(path, "".join(f"{line}\n" for line in lines))


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _agent_session_id(agent: dict[str, Any]) -> str:
    return str(
        agent.get("hermes_session_id")
        or agent.get("session_id")
        or agent.get("task_id")
        or ""
    )


def _agent_transcript_path(transcript_dir: Path, session_id: str) -> Path:
    return transcript_dir / f"agent-{sanitize_filename(session_id)}.jsonl"


def _agent_meta_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".meta.json")


def _agent_transcript_metadata(
    record: dict[str, Any],
    agent: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    return {
        "run_id": record.get("runId"),
        "workflow_task_id": record.get("taskId"),
        "tool_use_id": record.get("toolUseId"),
        "workflow_session_id": record.get("workflowSessionId"),
        "agent_id": agent.get("id"),
        "agent_label": agent.get("label"),
        "agent_status": agent.get("status"),
        "agent_type": agent.get("agent_type"),
        "phase": agent.get("phase"),
        "session_id": session_id,
        "runner": agent.get("runner"),
        "model": agent.get("model"),
        "workspace": agent.get("workspace"),
        "isolation": agent.get("isolation"),
        "prompt": agent.get("prompt"),
        "prompt_preview": agent.get("prompt_preview"),
        "tokens": agent.get("tokens"),
        "tool_calls": agent.get("tool_calls"),
        "updated_at": utc_now_iso(),
    }


def _append_unique(items: Any, value: str) -> None:
    if not isinstance(items, list):
        return
    if value not in items:
        items.append(value)


def _load_session_messages(session_id: str) -> list[dict[str, Any]]:
    try:
        from hermes_state import SessionDB

        return SessionDB().get_messages(session_id, include_inactive=True)
    except Exception:
        return []


def _iter_agent_snapshots(snapshot: dict[str, Any]):
    for agent in snapshot.get("agents") or []:
        if isinstance(agent, dict):
            yield agent
    for child in snapshot.get("children") or []:
        if isinstance(child, dict):
            yield from _iter_agent_snapshots(child)


def _resolve_workflow_session_id(plugin_context: Any, *, host_session_id: str | None = None) -> str:
    if host_session_id:
        return str(host_session_id)
    for attr in ("session_id", "sessionId"):
        value = getattr(plugin_context, attr, None) if plugin_context is not None else None
        if value:
            return str(value)
    for method_name in ("get_session_id", "current_session_id"):
        method = getattr(plugin_context, method_name, None) if plugin_context is not None else None
        if callable(method):
            try:
                value = method()
            except Exception:
                value = None
            if value:
                return str(value)
    cli_ref = _plugin_context_cli_ref(plugin_context)
    for value in (
        getattr(getattr(cli_ref, "agent", None), "session_id", None),
        getattr(cli_ref, "session_id", None),
    ):
        if value:
            return str(value)
    for name in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
        env_value = _get_hermes_session_env(name)
        if env_value:
            return env_value
    raise WorkflowRuntimeError(
        "Hermes did not provide a session id for workflow layout. "
        "Expected task_id/session_id kwargs, plugin_context CLI session, "
        "or gateway session context."
    )


def _plugin_context_cli_ref(plugin_context: Any) -> Any:
    manager = getattr(plugin_context, "_manager", None) if plugin_context is not None else None
    if manager is not None:
        return getattr(manager, "_cli_ref", None)
    return None


def _get_hermes_session_env(name: str) -> str:
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env(name, "") or "").strip()
    except Exception:
        return os.getenv(name, "").strip()


def _approve_launch(meta: dict[str, Any], config: PluginConfig, plugin_context: Any) -> tuple[bool, str]:
    """Gate a top-level workflow launch when ``require_launch_approval`` is on.

    Runs in the launching (parent) foreground turn, so the session context is
    native — no cross-thread propagation needed. Returns ``(approved, reason)``.
    Channels: gateway -> approve/deny buttons (blocks until tapped); CLI ->
    synchronous confirm; no interactive channel (headless) -> deny.
    """
    if not config.require_launch_approval:
        return True, ""

    name = str(meta.get("name") or "workflow")
    desc = str(meta.get("description") or "")
    label = f"workflow-launch:{name}"
    human = f'Launch dynamic workflow "{name}"' + (f" - {desc}" if desc else "")

    try:
        from tools import approval as _approval
    except Exception:
        return False, "launch approval required but Hermes' approval engine is unavailable"

    # Gateway: reuse the session-keyed approve/deny flow (blocks until resolved).
    try:
        if _approval._is_gateway_approval_context():
            session_key = _approval.get_current_session_key()
            notify_cb = _approval._gateway_notify_cbs.get(session_key)
            if notify_cb is None:
                return False, "launch approval required but no gateway approval channel is registered"
            decision = _approval._await_gateway_decision(
                session_key,
                notify_cb,
                {
                    "command": label,
                    "pattern_key": "workflow_launch",
                    "pattern_keys": ["workflow_launch"],
                    "description": human,
                },
                surface="gateway",
            )
            ok = bool(decision.get("resolved")) and decision.get("choice") not in (None, "deny")
            return (True, "") if ok else (False, "workflow launch was denied or timed out")
    except Exception as exc:
        return False, f"launch approval failed: {type(exc).__name__}: {exc}"

    # CLI interactive: synchronous confirm via the established callback pattern.
    if (os.environ.get("HERMES_INTERACTIVE") or "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from tools.terminal_tool import _get_approval_callback

            cb = _get_approval_callback()
        except Exception:
            cb = None
        try:
            choice = _approval.prompt_dangerous_approval(label, human, approval_callback=cb)
        except Exception as exc:
            return False, f"launch approval prompt failed: {type(exc).__name__}: {exc}"
        return (True, "") if (choice and choice != "deny") else (False, "workflow launch was denied")

    return False, (
        "launch approval required but no interactive channel "
        "(set require_launch_approval=false / HERMES_DYNAMIC_WORKFLOWS_REQUIRE_LAUNCH_APPROVAL=0 "
        "for unattended/headless use)"
    )


def _capture_gateway_session_context() -> dict[str, str] | None:
    """Capture the launching gateway session vars (parent context only).

    Returns None outside a gateway session. Used so a child worker thread can
    re-apply them and route a flagged command to the originating user for
    mid-run approval (child_approval_policy="ask").
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return None
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if not platform:
        return None  # not a gateway session
    keys = {
        "platform": "HERMES_SESSION_PLATFORM",
        "chat_id": "HERMES_SESSION_CHAT_ID",
        "chat_name": "HERMES_SESSION_CHAT_NAME",
        "thread_id": "HERMES_SESSION_THREAD_ID",
        "user_id": "HERMES_SESSION_USER_ID",
        "user_name": "HERMES_SESSION_USER_NAME",
        "session_key": "HERMES_SESSION_KEY",
    }
    return {field: get_session_env(env, "") for field, env in keys.items()}


def _notify_completion(plugin_context: Any, record: dict[str, Any], config: PluginConfig) -> None:
    """On terminal state, inject a Claude-Code-style <task-notification> into the
    conversation so the model can deliver the result without the user polling
    /workflows. Best-effort and CLI-only: ctx.inject_message returns False in
    gateway mode; any failure is swallowed so it never affects the run.
    """
    if not config.notify_on_complete or plugin_context is None:
        return
    inject = getattr(plugin_context, "inject_message", None)
    if not callable(inject):
        return
    try:
        inject(_render_task_notification(record, config.notify_result_preview_chars))
    except Exception:
        pass


def _render_task_notification(record: dict[str, Any], preview_chars: int) -> str:
    """Mirror Claude Code's LocalAgentTask task-notification block, adapted to a
    workflow run (tool_uses -> agents, plus errors)."""
    run_id = record.get("runId") or ""
    task_id = record.get("taskId") or run_id
    status = str(record.get("status") or "completed")
    snapshot = record.get("workflow") or {}
    meta = snapshot.get("meta") or {}
    name = meta.get("name") or "workflow"
    totals = snapshot.get("totals") or {}

    if record.get("error"):
        summary = f'Dynamic workflow "{name}" {status}: {record["error"]}'
    elif status == "completed":
        summary = f'Dynamic workflow "{name}" completed'
    elif status == "failed":
        summary = f'Dynamic workflow "{name}" failed: all agents errored'
    elif status == "stopped":
        summary = f'Dynamic workflow "{name}" was stopped'
    else:
        summary = f'Dynamic workflow "{name}" {status}'

    result_text = _completion_output_text(record)
    truncated = len(result_text) > preview_chars > 0
    if truncated:
        remaining = len(result_text) - preview_chars
        output_file = str(record.get("outputFile") or "")
        suffix = f"\n... (truncated {remaining} chars"
        if output_file:
            suffix += f", full result in {output_file}"
        suffix += ")"
        result_text = result_text[:preview_chars] + suffix

    agents = int(totals.get("agents") or 0)
    tokens = int(totals.get("tokens") or 0)
    tool_uses = int(totals.get("tool_calls") or 0)
    duration_ms = int(float(snapshot.get("duration_seconds") or 0) * 1000)

    lines = [
        "<task-notification>",
        f"<task-id>{task_id}</task-id>",
    ]
    tool_use_id = str(record.get("toolUseId") or "")
    if tool_use_id:
        lines.append(f"<tool-use-id>{tool_use_id}</tool-use-id>")
    output_file = str(record.get("outputFile") or "")
    if output_file:
        lines.append(f"<output-file>{output_file}</output-file>")
    lines.extend(
        [
            f"<status>{status}</status>",
            f"<summary>{summary}</summary>",
        ]
    )
    if result_text:
        lines.append(f"<result>{result_text}</result>")
    lines.append(
        f"<usage><agent_count>{agents}</agent_count>"
        f"<subagent_tokens>{tokens}</subagent_tokens>"
        f"<tool_uses>{tool_uses}</tool_uses>"
        f"<duration_ms>{duration_ms}</duration_ms></usage>"
    )
    lines.append("</task-notification>")
    return "\n".join(lines)


def _derive_run_status(snapshot: dict[str, Any]) -> str:
    """A run that finished but where every agent errored is 'failed', not
    'completed'. Partial failures stay 'completed' (surfaced via the error
    count in the display)."""
    totals = snapshot.get("totals") or {}
    agents = int(totals.get("agents") or 0)
    done = int(totals.get("done") or 0)
    if agents > 0 and done == 0:
        return "failed"
    return "completed"


_MANAGER: WorkflowRunManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_run_manager() -> WorkflowRunManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = WorkflowRunManager()
        return _MANAGER
