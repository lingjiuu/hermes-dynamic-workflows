"""Background workflow run manager."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import ResumeCache
from ..core.config import PluginConfig, load_config
from .context import PauseGate
from ..core.errors import WorkflowLaunchDenied, WorkflowRuntimeError, WorkflowToolUseError
from .sandbox import extract_meta, parse_script
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
from .runtime import WorkflowOptions, run_workflow


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
    parent_runtime: dict[str, Any] | None = field(default=None, repr=False)
    transcript_exporter: "LiveTranscriptExporter | None" = None
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class LiveTranscriptTarget:
    session_id: str
    transcript_path: Path
    meta_path: Path
    metadata: dict[str, Any]
    active: bool = True
    transcript_signature: str | None = None
    metadata_signature: str | None = None
    export_mode: str = "pending"
    fallback_reason: str | None = None
    lineage_session_ids: tuple[str, ...] = ()
    session_states: dict[str, tuple[int, int | None, int | None, int, int]] = field(
        default_factory=dict
    )
    last_message_id: int = 0


@dataclass(frozen=True)
class TranscriptRead:
    action: str
    messages: list[dict[str, Any]]
    export_mode: str
    fallback_reason: str | None = None
    lineage_session_ids: tuple[str, ...] = ()


class SessionTranscriptReader:
    """Read child transcripts incrementally, with a public-API full fallback."""

    _REQUIRED_MESSAGE_COLUMNS = {
        "id",
        "session_id",
        "role",
        "content",
        "tool_calls",
    }
    _REQUIRED_SESSION_COLUMNS = {
        "id",
        "parent_session_id",
        "started_at",
        "ended_at",
        "end_reason",
    }

    def __init__(self, db: Any = None) -> None:
        self._db = db
        self._incremental_reason: str | None = None
        self._has_active_column = False
        if self._db is None:
            try:
                from hermes_state import SessionDB

                self._db = SessionDB()
            except Exception as exc:
                self._incremental_reason = f"SessionDB unavailable: {type(exc).__name__}: {exc}"
        if self._db is not None and self._incremental_reason is None:
            self._incremental_reason = self._probe_incremental_capability()

    def close(self) -> None:
        close = getattr(self._db, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def read(self, target: LiveTranscriptTarget, *, force_rebuild: bool = False) -> TranscriptRead:
        if target.export_mode == "full_fallback" or self._incremental_reason:
            reason = target.fallback_reason or self._incremental_reason or "incremental mode disabled"
            return self._read_full_fallback(target, reason)
        try:
            return self._read_incremental(target, force_rebuild=force_rebuild)
        except Exception as exc:
            reason = f"incremental read failed: {type(exc).__name__}: {exc}"
            target.export_mode = "full_fallback"
            target.fallback_reason = reason
            return self._read_full_fallback(target, reason)

    def _probe_incremental_capability(self) -> str | None:
        conn = getattr(self._db, "_conn", None)
        lock = getattr(self._db, "_lock", None)
        if conn is None or lock is None:
            return "SessionDB private connection is unavailable"
        try:
            with lock:
                message_columns = {
                    str(row["name"] if hasattr(row, "keys") else row[1])
                    for row in conn.execute("PRAGMA table_info(messages)").fetchall()
                }
                self._has_active_column = "active" in message_columns
                session_columns = {
                    str(row["name"] if hasattr(row, "keys") else row[1])
                    for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
                }
        except Exception as exc:
            return f"schema probe failed: {type(exc).__name__}: {exc}"
        missing_messages = sorted(self._REQUIRED_MESSAGE_COLUMNS - message_columns)
        missing_sessions = sorted(self._REQUIRED_SESSION_COLUMNS - session_columns)
        if missing_messages or missing_sessions:
            return (
                "unsupported SessionDB schema"
                f" (messages missing={missing_messages}, sessions missing={missing_sessions})"
            )
        return None

    def _read_incremental(
        self,
        target: LiveTranscriptTarget,
        *,
        force_rebuild: bool,
    ) -> TranscriptRead:
        conn = self._db._conn
        lock = self._db._lock
        with lock:
            conn.execute("BEGIN")
            try:
                lineage = self._lineage(conn, target.session_id)
                states = self._session_states(
                    conn,
                    lineage,
                    has_active=self._has_active_column,
                )
                rebuild = (
                    force_rebuild
                    or not target.lineage_session_ids
                    or lineage != target.lineage_session_ids
                )
                if rebuild:
                    rows = self._message_rows(conn, lineage)
                else:
                    rows = self._message_rows(conn, lineage, after_id=target.last_message_id)
                    rebuild = not self._is_append_only(
                        target.session_states,
                        states,
                        rows,
                        has_active=self._has_active_column,
                    )
                    if rebuild:
                        rows = self._message_rows(conn, lineage)
                conn.commit()
            except BaseException:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise

        messages = [self._decode_message(row) for row in rows]
        target.export_mode = "incremental"
        target.fallback_reason = None
        target.lineage_session_ids = lineage
        target.session_states = states
        target.last_message_id = max(
            (state[2] or 0 for state in states.values()),
            default=0,
        )
        return TranscriptRead(
            action="rebuild" if rebuild else ("append" if messages else "unchanged"),
            messages=messages,
            export_mode="incremental",
            lineage_session_ids=lineage,
        )

    def _read_full_fallback(self, target: LiveTranscriptTarget, reason: str) -> TranscriptRead:
        if self._db is not None and callable(getattr(self._db, "get_messages", None)):
            try:
                messages = self._db.get_messages(target.session_id, include_inactive=True)
            except TypeError:
                messages = self._db.get_messages(target.session_id)
        else:
            messages = _load_session_messages(target.session_id)
        signature = _json_signature(messages)
        action = "unchanged" if signature == target.transcript_signature else "rebuild"
        target.transcript_signature = signature
        target.export_mode = "full_fallback"
        target.fallback_reason = reason
        target.lineage_session_ids = (target.session_id,)
        return TranscriptRead(
            action=action,
            messages=messages,
            export_mode="full_fallback",
            fallback_reason=reason,
            lineage_session_ids=target.lineage_session_ids,
        )

    @staticmethod
    def _lineage(conn: Any, root_session_id: str) -> tuple[str, ...]:
        rows = conn.execute(
            """
            WITH RECURSIVE lineage(id) AS (
                SELECT id FROM sessions WHERE id = ?
                UNION
                SELECT child.id
                FROM lineage
                JOIN sessions parent ON parent.id = lineage.id
                JOIN sessions child ON child.parent_session_id = lineage.id
                WHERE parent.end_reason = 'compression'
                  AND parent.ended_at IS NOT NULL
                  AND child.started_at >= parent.ended_at
            )
            SELECT lineage.id
            FROM lineage
            JOIN sessions ON sessions.id = lineage.id
            ORDER BY sessions.started_at, sessions.id
            """,
            (root_session_id,),
        ).fetchall()
        ids = tuple(str(row["id"] if hasattr(row, "keys") else row[0]) for row in rows)
        return ids or (root_session_id,)

    @staticmethod
    def _session_states(
        conn: Any,
        lineage: tuple[str, ...],
        *,
        has_active: bool,
    ) -> dict[str, tuple[int, int | None, int | None, int, int]]:
        placeholders = ",".join("?" for _ in lineage)
        if has_active:
            active_state_sql = (
                "SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS active_count, "
                "SUM(CASE WHEN active = 1 THEN id ELSE 0 END) AS active_id_sum "
            )
        else:
            active_state_sql = "COUNT(*) AS active_count, COALESCE(SUM(id), 0) AS active_id_sum "
        rows = conn.execute(
            "SELECT session_id, COUNT(*) AS row_count, MIN(id) AS min_id, "
            "MAX(id) AS max_id, "
            f"{active_state_sql}"
            f"FROM messages WHERE session_id IN ({placeholders}) GROUP BY session_id",
            lineage,
        ).fetchall()
        states = {
            str(row["session_id"]): (
                int(row["row_count"] or 0),
                int(row["min_id"]) if row["min_id"] is not None else None,
                int(row["max_id"]) if row["max_id"] is not None else None,
                int(row["active_count"] or 0),
                int(row["active_id_sum"] or 0),
            )
            for row in rows
        }
        for session_id in lineage:
            states.setdefault(session_id, (0, None, None, 0, 0))
        return states

    @staticmethod
    def _message_rows(
        conn: Any,
        lineage: tuple[str, ...],
        *,
        after_id: int | None = None,
    ) -> list[Any]:
        placeholders = ",".join("?" for _ in lineage)
        params: tuple[Any, ...] = tuple(lineage)
        after_clause = ""
        if after_id is not None:
            after_clause = " AND id > ?"
            params += (after_id,)
        return conn.execute(
            f"SELECT * FROM messages WHERE session_id IN ({placeholders})"
            f"{after_clause} ORDER BY id",
            params,
        ).fetchall()

    @staticmethod
    def _is_append_only(
        previous: dict[str, tuple[int, int | None, int | None, int, int]],
        current: dict[str, tuple[int, int | None, int | None, int, int]],
        new_rows: list[Any],
        *,
        has_active: bool,
    ) -> bool:
        if set(previous) != set(current):
            return False
        by_session: dict[str, list[Any]] = {}
        for row in new_rows:
            by_session.setdefault(str(row["session_id"]), []).append(row)
        for session_id, old in previous.items():
            rows = by_session.get(session_id, [])
            if has_active:
                new_active_ids = [
                    int(row["id"]) for row in rows if int(row["active"] or 0) == 1
                ]
            else:
                new_active_ids = [int(row["id"]) for row in rows]
            expected = (
                old[0] + len(rows),
                old[1] if old[0] else (int(rows[0]["id"]) if rows else None),
                int(rows[-1]["id"]) if rows else old[2],
                old[3] + len(new_active_ids),
                old[4] + sum(new_active_ids),
            )
            if current[session_id] != expected:
                return False
        return True

    def _decode_message(self, row: Any) -> dict[str, Any]:
        message = dict(row)
        decode_content = getattr(self._db, "_decode_content", None)
        if callable(decode_content):
            message["content"] = decode_content(message.get("content"))
        if message.get("tool_calls"):
            try:
                message["tool_calls"] = json.loads(message["tool_calls"])
            except (TypeError, json.JSONDecodeError):
                message["tool_calls"] = []
        return message


class LiveTranscriptExporter:
    """Refresh all live child transcripts for one workflow from SessionDB."""

    def __init__(
        self,
        *,
        run_id: str,
        interval_seconds: float = 0.5,
        reader: SessionTranscriptReader | None = None,
    ) -> None:
        self.run_id = run_id
        self.interval_seconds = interval_seconds
        self._reader = reader or SessionTranscriptReader()
        self._targets: dict[str, LiveTranscriptTarget] = {}
        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"workflow-transcripts-{sanitize_filename(run_id)[:32]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def upsert(
        self,
        *,
        session_id: str,
        transcript_path: Path,
        meta_path: Path,
        metadata: dict[str, Any],
        active: bool,
    ) -> None:
        with self._lock:
            target = self._targets.get(session_id)
            was_active = target.active if target is not None else None
            if target is None:
                target = LiveTranscriptTarget(
                    session_id=session_id,
                    transcript_path=transcript_path,
                    meta_path=meta_path,
                    metadata=dict(metadata),
                    active=active,
                )
                self._targets[session_id] = target
            else:
                target.transcript_path = transcript_path
                target.meta_path = meta_path
                target.metadata = dict(metadata)
                target.active = active
        # Write a newly discovered child immediately. When it becomes terminal,
        # do one last refresh before removing it from periodic polling.
        if was_active is None or (was_active and not active):
            self.flush(session_ids=[session_id])

    def stop(self, *, final: bool = True) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        if final:
            self.flush(force_rebuild=True)
            self._reader.close()

    def flush(
        self,
        *,
        session_ids: list[str] | None = None,
        active_only: bool = False,
        force_rebuild: bool = False,
    ) -> None:
        with self._lock:
            if session_ids is None:
                targets = list(self._targets.values())
            else:
                targets = [
                    self._targets[session_id]
                    for session_id in session_ids
                    if session_id in self._targets
                ]
            if active_only:
                targets = [target for target in targets if target.active]
        target_ids = [target.session_id for target in targets]
        if not target_ids:
            return
        first_error: Exception | None = None
        # One reader and one writer at a time per workflow. This avoids N
        # concurrent SQLite connections and atomic-temp-file races.
        with self._flush_lock:
            for session_id in target_ids:
                try:
                    self._flush_target(session_id, force_rebuild=force_rebuild)
                except Exception as exc:
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def _flush_target(self, session_id: str, *, force_rebuild: bool = False) -> None:
        with self._lock:
            target = self._targets.get(session_id)
            if target is None:
                return
            transcript_path = target.transcript_path
            meta_path = target.meta_path
            metadata = dict(target.metadata)
            previous_metadata_signature = target.metadata_signature

        read = self._reader.read(target, force_rebuild=force_rebuild)
        metadata["transcript_export_mode"] = read.export_mode
        metadata["transcript_lineage_session_ids"] = list(read.lineage_session_ids)
        if read.fallback_reason:
            metadata["transcript_export_fallback_reason"] = read.fallback_reason
        metadata_signature = _json_signature(
            {key: value for key, value in metadata.items() if key != "updated_at"}
        )
        if read.action == "rebuild":
            _write_agent_transcript_files(
                transcript_path,
                meta_path,
                metadata=metadata,
                messages=read.messages,
            )
        elif read.action == "append":
            _append_agent_transcript_messages(transcript_path, read.messages)
            if metadata_signature != previous_metadata_signature:
                _write_json_atomic(meta_path, metadata)
        elif metadata_signature != previous_metadata_signature:
            _write_json_atomic(meta_path, metadata)
        with self._lock:
            target = self._targets.get(session_id)
            if target is not None:
                target.metadata_signature = metadata_signature

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.flush(active_only=True)
            except Exception:
                pass


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

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return [self._public_record(run) for run in self.store.list_runs(limit=limit)]

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

    def format_agent_overview(self, limit: int = 12) -> str:
        runs = self.list(limit=limit)
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

            managed.child_runner = HermesChildAgentRunner(
                config,
                session_context=session_context,
                parent_runtime=managed.parent_runtime,
            )
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
        # The live exporter has already performed a final validated rebuild. Keep
        # that file because it can include compression-lineage sessions; only use
        # the legacy full export path when live export never produced artifacts.
        if not path.is_file() or not meta_path.is_file():
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


def _append_agent_transcript_messages(path: Path, messages: list[dict[str, Any]]) -> None:
    if not messages:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"type": "message", "message": message}, ensure_ascii=False, default=str)
        for message in messages
    ]
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o666)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"short transcript append: wrote {written} of {len(payload)} bytes")
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _json_signature(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _agent_session_id(agent: dict[str, Any]) -> str:
    return str(
        agent.get("hermes_session_id")
        or agent.get("session_id")
        or agent.get("task_id")
        or ""
    )


def _is_active_agent_snapshot(agent: dict[str, Any]) -> bool:
    return str(agent.get("status") or "") in {"queued", "running", "retrying"}


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
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
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
