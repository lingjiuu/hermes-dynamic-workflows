"""Background workflow run manager."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine.cache import ResumeCache
from ..core.config import PluginConfig, load_config
from ..engine.context import PauseGate
from ..core.errors import WorkflowLaunchDenied, WorkflowRuntimeError, WorkflowToolUseError
from ..engine.sandbox import extract_meta, parse_script
from ..core.token_budget import parse_token_budget
from ..storage.store import (
    WorkflowStore,
    new_run_id,
    new_task_id,
    resolve_workflow_source,
    sanitize_filename,
    utc_now_iso,
)
from ..storage.control import ControlListener, new_control_owner
from ..view.render import (
    render_agent_overview,
    render_saved_markdown,
    render_workflow_text,
)
from ..engine.runtime import WorkflowOptions, run_workflow
from .transcripts import (
    LiveTranscriptExporter,
    _export_child_transcripts,
    _agent_session_id,
    _is_active_agent_snapshot,
    _agent_transcript_path,
    _agent_meta_path,
    _agent_transcript_metadata,
    _append_unique,
    _iter_agent_snapshots,
)


@dataclass
class ManagedRun:
    run_id: str
    stop_event: threading.Event
    pause_gate: PauseGate
    record: dict[str, Any]
    thread: threading.Thread | None = None
    child_runner: Any = None
    plugin_context: Any = None
    session_context: dict[str, str] | None = None
    approval_callback: Any = field(default=None, repr=False)
    parent_runtime: dict[str, Any] | None = field(default=None, repr=False)
    transcript_exporter: "LiveTranscriptExporter | None" = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class WorkflowRunManager:
    def __init__(
        self,
        store: WorkflowStore | None = None,
        config: PluginConfig | None = None,
        *,
        enable_control: bool = False,
    ):
        self.store = store or WorkflowStore()
        self.config = config or load_config()
        self.control_owner = new_control_owner()
        self._runs: dict[str, ManagedRun] = {}
        self._lock = threading.RLock()
        self._control_listener: ControlListener | None = None
        if enable_control:
            self.start_control_listener()

    def start_control_listener(self) -> bool:
        with self._lock:
            if self._control_listener is not None:
                return True
            listener = ControlListener(
                store=self.store,
                owner=self.control_owner,
                handler=self._handle_control_request,
            )
            self._control_listener = listener
        try:
            listener.start()
        except Exception:
            with self._lock:
                if self._control_listener is listener:
                    self._control_listener = None
            return False
        return True

    def stop_control_listener(self) -> None:
        with self._lock:
            listener = self._control_listener
            self._control_listener = None
        if listener is not None:
            listener.stop()

    def start_from_params(
        self,
        params: dict[str, Any],
        *,
        cwd: str | None = None,
        plugin_context: Any = None,
        parent_agent: Any = None,
        host_session_id: str | None = None,
        user_task: str | None = None,
        launch_approved: bool = False,
        restart_from_run_id: str | None = None,
        token_budget_total_override: int | None = None,
        session_context_override: dict[str, str] | None = None,
        approval_callback_override: Any = None,
        parent_runtime_override: dict[str, Any] | None = None,
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
        approved, reason = (True, "") if launch_approved else _approve_launch(meta, config, plugin_context)
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
        token_budget = (
            token_budget_total_override
            if token_budget_total_override is not None
            else parse_token_budget(user_task)
        )
        # Captured in the launching (parent) context, which carries the gateway
        # session vars when the run is started from a gateway session.
        session_context = (
            session_context_override
            if session_context_override is not None
            else _capture_gateway_session_context()
        )
        approval_callback = (
            approval_callback_override
            if approval_callback_override is not None
            else _capture_cli_approval_callback()
        )
        parent_runtime = (
            dict(parent_runtime_override)
            if parent_runtime_override is not None
            else _capture_parent_runtime(parent_agent, plugin_context=plugin_context)
        )

        stop_event = threading.Event()
        pause_gate = PauseGate()
        record = {
            "runId": run_id,
            "taskId": task_id,
            "status": "queued",
            "createdAt": utc_now_iso(),
            "startedAt": None,
            "finishedAt": None,
            "cwd": cwd_value,
            "workflowSessionId": workflow_session_id,
            "controlOwner": self.control_owner if self._control_listener is not None else None,
            "scriptPath": str(saved_path),
            "transcriptDir": str(transcript_dir),
            "journalFile": str(journal_path),
            "summary": meta.get("description") or meta.get("name") or "workflow",
            "source": {
                "type": source.source_type,
                "ref": source.source_ref,
            },
            "resumeFromRunId": resume_from,
            "restartedFromRunId": restart_from_run_id,
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
        managed = ManagedRun(
            run_id=run_id,
            stop_event=stop_event,
            pause_gate=pause_gate,
            record=record,
            plugin_context=plugin_context,
            session_context=session_context,
            approval_callback=approval_callback,
            parent_runtime=parent_runtime,
        )

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
            if managed.record.get("status") in {"queued", "running", "paused", "stopping"}:
                return self._public_record(dict(managed.record))
        return None

    def stop_task(self, task_id: str) -> dict[str, Any] | None:
        """Stop an active background workflow by its task id.

        This intentionally only checks live managed runs. Historical runs in
        state are not stoppable tasks, so stopping a completed or already
        stopped task should return a "No task found" result.
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
                if record.get("status") not in {"queued", "running", "paused"}:
                    return None
                managed.stop_event.set()
                managed.pause_gate.resume()
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

    def skip_agent(self, task_id: str, child_task_id: str) -> bool:
        """Skip one active child agent without stopping its workflow run."""
        wanted = str(task_id or "")
        child_wanted = str(child_task_id or "")
        if not wanted or not child_wanted:
            return False
        with self._lock:
            managed_runs = list(self._runs.values())
        for managed in managed_runs:
            with managed.lock:
                if str(managed.record.get("taskId") or "") != wanted:
                    continue
                if managed.record.get("status") not in {"queued", "running", "paused"}:
                    return False
                runner = managed.child_runner
            if runner is None or not hasattr(runner, "skip_child"):
                return False
            return bool(runner.skip_child(child_wanted))
        return False

    def list(self, limit: int = 20, *, session_id: str | None = None) -> list[dict[str, Any]]:
        return [
            self._public_record(run)
            for run in self.store.list_runs(limit=limit, session_id=session_id)
        ]

    def stop(self, run_id: str) -> bool:
        with self._lock:
            managed = self._runs.get(run_id)
        if not managed:
            record = self.store.load_run(run_id)
            if not record or record.get("status") not in {"queued", "running", "paused"}:
                return False
            record["status"] = "stopped"
            record["finishedAt"] = utc_now_iso()
            self.store.save_run(record)
            return True

        with managed.lock:
            if managed.record.get("status") not in {"queued", "running", "paused"}:
                return False
            managed.stop_event.set()
            managed.pause_gate.resume()
            child_runner = managed.child_runner
            managed.record["status"] = "stopping"
            self.store.save_run(managed.record)
        if child_runner is not None and hasattr(child_runner, "interrupt_all"):
            try:
                child_runner.interrupt_all()
            except Exception:
                pass
        return True

    def pause(self, run_id: str) -> bool:
        with self._lock:
            managed = self._runs.get(run_id)
        if managed is None:
            return False
        with managed.lock:
            if managed.record.get("status") not in {"queued", "running"}:
                return False
            managed.pause_gate.pause()
            managed.record["status"] = "paused"
            managed.record["pausedAt"] = utc_now_iso()
            self.store.save_run(managed.record)
        return True

    def resume(self, run_id: str) -> bool:
        with self._lock:
            managed = self._runs.get(run_id)
        if managed is None:
            return False
        with managed.lock:
            if managed.record.get("status") != "paused":
                return False
            managed.pause_gate.resume()
            managed.record["status"] = "running"
            managed.record["resumedAt"] = utc_now_iso()
            self.store.save_run(managed.record)
        return True

    def restart(self, run_id: str) -> dict[str, Any] | None:
        record = self.get(run_id)
        if record is None:
            return None
        with self._lock:
            managed = self._runs.get(run_id)
        if managed is not None and record.get("status") in {"queued", "running", "paused", "stopping"}:
            self.stop(run_id)
            final = self.wait(run_id, timeout=5)
            if final and final.get("status") not in {"stopped", "completed", "failed", "error"}:
                raise WorkflowRuntimeError(f"workflow {run_id} did not stop before restart")

        script_path = str(record.get("scriptPath") or "")
        if not script_path or not Path(script_path).is_file():
            raise WorkflowRuntimeError(f"workflow script is unavailable for restart: {script_path}")
        params: dict[str, Any] = {"scriptPath": script_path}
        if "args" in record:
            params["args"] = record.get("args")
        return self.start_from_params(
            params,
            cwd=str(record.get("cwd") or os.getcwd()),
            plugin_context=managed.plugin_context if managed is not None else None,
            host_session_id=str(record.get("workflowSessionId") or "") or None,
            launch_approved=True,
            restart_from_run_id=run_id,
            token_budget_total_override=record.get("tokenBudget"),
            session_context_override=managed.session_context if managed is not None else None,
            approval_callback_override=managed.approval_callback if managed is not None else None,
            parent_runtime_override=managed.parent_runtime if managed is not None else None,
        )

    def _handle_control_request(self, request: dict[str, Any]) -> dict[str, Any]:
        run_id = str(request.get("runId") or "")
        action = str(request.get("action") or "")
        record = self.get(run_id)
        if record is None:
            return {"ok": False, "action": action, "runId": run_id, "message": f"Workflow run not found: {run_id}"}
        if str(record.get("controlOwner") or "") != self.control_owner:
            return {"ok": False, "action": action, "runId": run_id, "message": "Workflow is owned by another Hermes process."}
        expected = str(request.get("expectedStatus") or "")
        if expected and str(record.get("status") or "") != expected:
            return {
                "ok": False,
                "action": action,
                "runId": run_id,
                "status": record.get("status"),
                "message": f"Workflow status changed from {expected} to {record.get('status')}; retry the action.",
            }
        if action == "stop":
            ok = self.stop(run_id)
            message = f"Stop requested for {run_id}." if ok else f"Workflow {run_id} is not stoppable."
            return {"ok": ok, "action": action, "runId": run_id, "status": "stopping" if ok else record.get("status"), "message": message}
        if action == "pause":
            ok = self.pause(run_id)
            message = f"Paused {run_id}; running agents may finish." if ok else f"Workflow {run_id} is not pausable."
            return {"ok": ok, "action": action, "runId": run_id, "status": "paused" if ok else record.get("status"), "message": message}
        if action == "resume":
            ok = self.resume(run_id)
            message = f"Resumed {run_id}." if ok else f"Workflow {run_id} is not paused."
            return {"ok": ok, "action": action, "runId": run_id, "status": "running" if ok else record.get("status"), "message": message}
        if action == "restart":
            restarted = self.restart(run_id)
            if restarted is None:
                return {"ok": False, "action": action, "runId": run_id, "message": f"Workflow run not found: {run_id}"}
            new_run_id = str(restarted.get("runId") or "")
            return {
                "ok": True,
                "action": action,
                "runId": run_id,
                "newRunId": new_run_id,
                "status": restarted.get("status"),
                "message": f"Restarted {run_id} as {new_run_id}.",
            }
        return {"ok": False, "action": action, "runId": run_id, "message": f"Unsupported control action: {action}"}

    def format_agent_overview(self, limit: int = 12, *, session_id: str | None = None) -> str:
        runs = self.list(limit=limit, session_id=session_id)
        return render_agent_overview(runs)

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
            from ..child.runner import HermesChildAgentRunner

            runner_kwargs = {
                "session_context": session_context,
                "approval_session_key": _workflow_approval_session_key(managed, session_context),
                "parent_runtime": managed.parent_runtime,
            }
            if managed.approval_callback is not None:
                runner_kwargs["approval_callback"] = managed.approval_callback
            managed.child_runner = HermesChildAgentRunner(config, **runner_kwargs)
            self._update(
                managed,
                status="paused" if managed.pause_gate.is_paused else "running",
                startedAt=utc_now_iso(),
            )
            result = run_workflow(
                script,
                WorkflowOptions(
                    args=args,
                    cwd=cwd or os.environ.get("TERMINAL_CWD") or os.getcwd(),
                    config=config,
                    child_runner=managed.child_runner,
                    stop_event=managed.stop_event,
                    pause_gate=managed.pause_gate,
                    resume_cache=resume_cache,
                    on_update=lambda state: self._update_state(managed, state),
                    on_journal=lambda event: self._append_journal_event(managed, event),
                    plugin_context=plugin_context,
                    token_budget_total=token_budget,
                    source_ref=str(managed.record.get("scriptPath") or ""),
                    store=self.store,
                ),
            )
            snapshot = result.state.snapshot()
            self._sync_live_child_transcripts(managed, snapshot)
            if managed.stop_event.is_set():
                status = "stopped"
            else:
                status = "completed"
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
            status = "stopped" if managed.stop_event.is_set() else "failed"
            self._update(
                managed,
                status=status,
                finishedAt=utc_now_iso(),
                error=_runtime_error_text(exc),
                agentCache=resume_cache.current,
            )
        finally:
            self._stop_live_transcript_exporter(managed)
            self._finalize_completion_artifacts(managed)
            _notify_completion(plugin_context, managed.record, config, managed.session_context)

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
        targets: list[dict[str, Any]] = []
        start_exporter = False
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
                targets.append(
                    {
                        "session_id": session_id,
                        "transcript_path": path,
                        "meta_path": meta_path,
                        "metadata": metadata,
                        "active": _is_active_agent_snapshot(agent),
                    }
                )
            if targets and managed.transcript_exporter is None:
                managed.transcript_exporter = LiveTranscriptExporter(run_id=managed.run_id)
                start_exporter = True
            exporter = managed.transcript_exporter
        if exporter is None:
            return
        if start_exporter:
            try:
                exporter.start()
            except Exception as exc:
                with managed.lock:
                    managed.record["transcriptExportError"] = f"{type(exc).__name__}: {exc}"
        for target in targets:
            try:
                exporter.upsert(**target)
            except Exception as exc:
                with managed.lock:
                    managed.record["transcriptExportError"] = f"{type(exc).__name__}: {exc}"

    def _stop_live_transcript_exporter(self, managed: ManagedRun) -> None:
        with managed.lock:
            exporter = managed.transcript_exporter
        if exporter is None:
            return
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


def _runtime_error_text(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    frames = traceback.format_tb(exc.__traceback__, limit=8)
    if frames:
        return message + "\n" + "".join(frames).rstrip()
    return message


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


def _capture_parent_runtime(parent_agent: Any, *, plugin_context: Any = None) -> dict[str, Any] | None:
    """Snapshot the launching agent runtime for child-model inheritance.

    The snapshot stays on ManagedRun only. It must never be added to the
    persisted run record because it can contain credentials and live pools.
    """
    agent = parent_agent
    if agent is None:
        cli_ref = _plugin_context_cli_ref(plugin_context)
        agent = getattr(cli_ref, "agent", None) if cli_ref is not None else None
    if agent is None:
        agent = _gateway_running_agent()
    if agent is None:
        return None

    model = str(getattr(agent, "model", "") or "").strip()
    if not model:
        return None

    runtime: dict[str, Any] = {"model": model}
    for key in (
        "provider",
        "base_url",
        "api_key",
        "api_mode",
        "acp_command",
        "reasoning_config",
        "service_tier",
        "max_tokens",
    ):
        value = getattr(agent, key, None)
        if value is not None:
            runtime[key] = value
    if not runtime.get("api_key"):
        client_kwargs = getattr(agent, "_client_kwargs", None)
        if isinstance(client_kwargs, dict) and client_kwargs.get("api_key"):
            runtime["api_key"] = client_kwargs["api_key"]

    acp_args = getattr(agent, "acp_args", None)
    if acp_args:
        runtime["acp_args"] = list(acp_args)

    credential_pool = getattr(agent, "_credential_pool", None)
    if credential_pool is not None:
        runtime["credential_pool"] = credential_pool

    fallback_chain = getattr(agent, "_fallback_chain", None)
    if fallback_chain:
        runtime["fallback_model"] = list(fallback_chain)
    else:
        fallback_model = getattr(agent, "_fallback_model", None)
        if fallback_model:
            runtime["fallback_model"] = fallback_model

    request_overrides = getattr(agent, "request_overrides", None)
    if isinstance(request_overrides, dict) and request_overrides:
        runtime["request_overrides"] = dict(request_overrides)
    return runtime


def _gateway_running_agent() -> Any:
    """Return the active or cached agent for the current gateway session."""
    session_key = _get_hermes_session_env("HERMES_SESSION_KEY")
    if not session_key:
        return None
    try:
        from ..host import gateway as host_gateway

        runner = host_gateway.gateway_runner_ref()
        if runner is None:
            return None
        running = getattr(runner, "_running_agents", None)
        if isinstance(running, dict):
            agent = running.get(session_key)
            if getattr(agent, "model", None):
                return agent
        cache = getattr(runner, "_agent_cache", None)
        cached = cache.get(session_key) if isinstance(cache, dict) else None
        if isinstance(cached, tuple):
            cached = cached[0] if cached else None
        return cached if getattr(cached, "model", None) else None
    except Exception:
        return None


def _get_hermes_session_env(name: str) -> str:
    try:
        from ..host import gateway as host_gateway

        return str(host_gateway.raw_session_env(name, "") or "").strip()
    except Exception:
        return os.getenv(name, "").strip()


def _capture_cli_approval_callback() -> Any:
    """Capture the live CLI approval UI for background workflow children."""
    if (os.environ.get("HERMES_INTERACTIVE") or "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    try:
        from tools.terminal_tool import _get_approval_callback

        callback = _get_approval_callback()
        return callback if callable(callback) else None
    except Exception:
        return None


def _workflow_approval_session_key(
    managed: ManagedRun,
    session_context: dict[str, str] | None,
) -> str:
    return str(
        (session_context or {}).get("session_key")
        or managed.record.get("workflowSessionId")
        or managed.run_id
    )


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
            decision = _await_gateway_launch_decision(
                _approval,
                session_key,
                notify_cb,
                {
                    "command": label,
                    "pattern_key": "workflow_launch",
                    "pattern_keys": ["workflow_launch"],
                    "description": human,
                },
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


def _await_gateway_launch_decision(_approval: Any, session_key: str, notify_cb: Any, data: dict[str, Any]) -> dict[str, Any]:
    legacy_wait = getattr(_approval, "_await_gateway_decision", None)
    if callable(legacy_wait):
        return legacy_wait(session_key, notify_cb, data, surface="gateway")

    entry_cls = getattr(_approval, "_ApprovalEntry", None)
    lock = getattr(_approval, "_lock", None)
    queues = getattr(_approval, "_gateway_queues", None)
    if entry_cls is None or lock is None or not isinstance(queues, dict):
        raise RuntimeError("Hermes gateway approval queue API is unavailable")

    entry = entry_cls(data)
    with lock:
        queues.setdefault(session_key, []).append(entry)

    fire_hook = getattr(_approval, "_fire_approval_hook", None)
    if callable(fire_hook):
        fire_hook(
            "pre_approval_request",
            command=data.get("command", ""),
            description=data.get("description", ""),
            pattern_key=data.get("pattern_key", ""),
            pattern_keys=list(data.get("pattern_keys") or []),
            session_key=session_key,
            surface="gateway",
        )

    try:
        notify_cb(data)
    except Exception:
        with lock:
            queue = queues.get(session_key, [])
            if entry in queue:
                queue.remove(entry)
            if not queue:
                queues.pop(session_key, None)
        raise

    timeout = 300
    get_config = getattr(_approval, "_get_approval_config", None)
    if callable(get_config):
        try:
            timeout = int((get_config() or {}).get("gateway_timeout", timeout))
        except (TypeError, ValueError):
            timeout = 300

    touch_activity_if_due = None
    now = time.monotonic()
    activity_state: dict[str, Any] = {"start": now, "last_touch": now}
    try:
        from tools.environments.base import touch_activity_if_due as _touch_activity_if_due

        touch_activity_if_due = _touch_activity_if_due
    except Exception:
        pass

    resolved = False
    deadline = time.monotonic() + max(0, timeout)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if entry.event.wait(timeout=min(1.0, remaining)):
            resolved = True
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for workflow launch approval")

    with lock:
        queue = queues.get(session_key, [])
        if entry in queue:
            queue.remove(entry)
        if not queue:
            queues.pop(session_key, None)

    choice = entry.result
    if callable(fire_hook):
        fire_hook(
            "post_approval_response",
            command=data.get("command", ""),
            description=data.get("description", ""),
            pattern_key=data.get("pattern_key", ""),
            pattern_keys=list(data.get("pattern_keys") or []),
            session_key=session_key,
            surface="gateway",
            choice=(choice if resolved and choice else "timeout"),
        )

    return {"resolved": resolved and choice is not None, "choice": choice}


def _capture_gateway_session_context() -> dict[str, str] | None:
    """Capture the launching gateway session vars (parent context only).

    Returns None outside a gateway session. Used so a child worker thread can
    re-apply them and route a flagged command to the originating user for
    mid-run approval (child_approval_policy="ask").
    """
    from ..host import gateway as host_gateway

    try:
        platform = host_gateway.raw_session_env("HERMES_SESSION_PLATFORM", "")
    except Exception:
        return None
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
        "message_id": "HERMES_SESSION_MESSAGE_ID",
    }
    return {field: host_gateway.raw_session_env(env, "") for field, env in keys.items()}


def _notify_completion(
    plugin_context: Any,
    record: dict[str, Any],
    config: PluginConfig,
    session_context: dict[str, str] | None = None,
) -> None:
    """On terminal state, inject a <task-notification> into the
    conversation so the model can deliver the result without the user polling
    /workflows. In gateway mode, where CLI injection is unavailable after the
    parent turn returns, send a concise completion message to the origin chat.
    Best effort: any failure is swallowed so it never affects the run.
    """
    if not config.notify_on_complete:
        return
    notification = _render_task_notification(record, config.notify_result_preview_chars)
    injected = False
    inject = getattr(plugin_context, "inject_message", None) if plugin_context is not None else None
    try:
        if callable(inject):
            injected = bool(inject(notification))
    except Exception:
        pass
    if injected:
        return
    _send_gateway_completion_notification(record, config, session_context)


def _send_gateway_completion_notification(
    record: dict[str, Any],
    config: PluginConfig,
    session_context: dict[str, str] | None,
) -> None:
    context = dict(session_context or {})
    platform = str(context.get("platform") or "").strip().lower()
    chat_id = str(context.get("chat_id") or "").strip()
    if not platform or not chat_id:
        return

    try:
        from agent.async_utils import safe_schedule_threadsafe
        from ..host import gateway as host_gateway

        runner = host_gateway.gateway_runner_ref()
        if runner is None:
            return
        adapter_key, adapter = _gateway_adapter_for_platform(runner, platform)
        if adapter is None:
            return

        source = _gateway_source_for_context(runner, context)
        if source is not None:
            chat_id = str(getattr(source, "chat_id", chat_id) or chat_id)
            metadata = _gateway_thread_metadata(runner, source=source, adapter=adapter)
        else:
            metadata = _gateway_thread_metadata(runner, context=context, adapter_key=adapter_key, adapter=adapter)
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)
        metadata["notify"] = True

        loop = getattr(runner, "_gateway_loop", None)
        if loop is None:
            return
        future = safe_schedule_threadsafe(
            adapter.send(chat_id, _render_gateway_completion_message(record, config), metadata=metadata),
            loop,
        )
        if future is not None:
            future.result(timeout=15)
    except Exception:
        pass


def _gateway_adapter_for_platform(runner: Any, platform: str) -> tuple[Any, Any]:
    adapters = getattr(runner, "adapters", None)
    if not isinstance(adapters, dict):
        return None, None
    for key, adapter in adapters.items():
        value = str(getattr(key, "value", key) or "").lower()
        if value == platform:
            return key, adapter
    return None, None


def _gateway_source_for_context(runner: Any, context: dict[str, str]) -> Any:
    session_key = str(context.get("session_key") or "").strip()
    sources = getattr(runner, "_session_sources", None)
    if session_key and hasattr(sources, "get"):
        try:
            source = sources.get(session_key)
        except Exception:
            source = None
        if source is not None:
            return source
    return None


def _gateway_thread_metadata(
    runner: Any,
    *,
    source: Any | None = None,
    context: dict[str, str] | None = None,
    adapter_key: Any = None,
    adapter: Any = None,
) -> dict[str, Any] | None:
    if source is not None:
        method = getattr(runner, "_thread_metadata_for_source", None)
        if callable(method):
            try:
                return method(source, getattr(source, "message_id", None))
            except Exception:
                pass
    context = context or {}
    thread_id = str(context.get("thread_id") or "").strip()
    if not thread_id:
        return None
    method = getattr(runner, "_thread_metadata_for_target", None)
    if callable(method):
        try:
            return method(
                adapter_key,
                str(context.get("chat_id") or ""),
                thread_id,
                chat_type="dm",
                reply_to_message_id=str(context.get("message_id") or "") or None,
                adapter=adapter,
            )
        except Exception:
            pass
    return {"thread_id": thread_id}


def _render_gateway_completion_message(record: dict[str, Any], config: PluginConfig) -> str:
    status = str(record.get("status") or "completed")
    task_id = str(record.get("taskId") or record.get("runId") or "")
    summary = str(record.get("summary") or "Dynamic workflow")
    icon = "✅" if status == "completed" else "⏹" if status == "stopped" else "❌"
    lines = [
        f"{icon} Workflow {status}: {summary}",
        f"Task: {task_id}",
    ]
    if record.get("error"):
        lines.append(f"Error: {str(record.get('error') or '').strip()}")
    else:
        result_text = _completion_output_text(record).strip()
        if result_text:
            preview_chars = config.notify_result_preview_chars
            if preview_chars > 0 and len(result_text) > preview_chars:
                remaining = len(result_text) - preview_chars
                result_text = result_text[:preview_chars] + f"\n... (truncated {remaining} chars)"
            lines.append(f"Result:\n{result_text}")
    output_file = str(record.get("outputFile") or "")
    if output_file:
        lines.append(f"Output: {output_file}")
    return "\n".join(lines)


def _render_task_notification(record: dict[str, Any], preview_chars: int) -> str:
    """Build a task-notification block adapted to a workflow run (tool_uses
    mapped to agents, plus errors)."""
    run_id = record.get("runId") or ""
    task_id = record.get("taskId") or run_id
    status = str(record.get("status") or "completed")
    snapshot = record.get("workflow") or {}
    meta = snapshot.get("meta") or {}
    name = meta.get("name") or "workflow"
    totals = snapshot.get("totals") or {}

    if record.get("error"):
        if status == "failed":
            summary = f'Dynamic workflow "{name}" failed: {record["error"]}'
        else:
            summary = f'Dynamic workflow "{name}" {status}: {record["error"]}'
    elif status == "completed":
        summary = f'Dynamic workflow "{name}" completed'
    elif status == "stopped":
        summary = f'Dynamic workflow "{name}" was stopped'
    else:
        summary = f'Dynamic workflow "{name}" {status}'

    include_result = not record.get("error")
    result_text = _completion_output_text(record) if include_result else ""
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
    recovery = str(record.get("transcriptDir") or "")
    if record.get("error") and recovery:
        lines.append(f"<recovery>Agent transcripts: {recovery}</recovery>")
    lines.append(
        f"<usage><agent_count>{agents}</agent_count>"
        f"<subagent_tokens>{tokens}</subagent_tokens>"
        f"<tool_uses>{tool_uses}</tool_uses>"
        f"<duration_ms>{duration_ms}</duration_ms></usage>"
    )
    lines.append("</task-notification>")
    return "\n".join(lines)

_MANAGER: WorkflowRunManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_run_manager() -> WorkflowRunManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = WorkflowRunManager(enable_control=True)
        return _MANAGER
