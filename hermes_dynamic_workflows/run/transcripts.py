"""Live child-agent transcript reading and export.

Reads child sessions from Hermes' SessionDB (incrementally when the schema
allows, with a public-API full-read fallback) and writes per-agent JSONL
transcripts + meta sidecars under a run's transcript directory. Self-contained:
the run manager calls into here, never the reverse.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..storage.store import sanitize_filename, utc_now_iso


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
                from ..host import session as host_session

                self._db = host_session.create_session_db()
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
        from ..host import session as host_session

        return host_session.create_session_db().get_messages(session_id, include_inactive=True)
    except Exception:
        return []


def _iter_agent_snapshots(snapshot: dict[str, Any]):
    for agent in snapshot.get("agents") or []:
        if isinstance(agent, dict):
            yield agent
    for child in snapshot.get("children") or []:
        if isinstance(child, dict):
            yield from _iter_agent_snapshots(child)
