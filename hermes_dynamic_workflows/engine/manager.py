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
from ..storage.store import WorkflowStore, new_run_id, resolve_workflow_source, utc_now_iso
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
    lock: threading.RLock = field(default_factory=threading.RLock)


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
    ) -> dict[str, Any]:
        config = self.config
        source = resolve_workflow_source(params, store=self.store, cwd=cwd)
        run_id = new_run_id()
        saved_path = self.store.save_script(run_id, source.script)
        resume_from = str(params.get("resumeFromRunId") or "").strip() or None
        previous = self.store.load_run(resume_from) if resume_from else None
        resume_cache = ResumeCache.from_run(previous)
        args = params["args"] if "args" in params else None

        stop_event = threading.Event()
        record = {
            "runId": run_id,
            "status": "queued",
            "createdAt": utc_now_iso(),
            "startedAt": None,
            "finishedAt": None,
            "cwd": cwd or os.environ.get("TERMINAL_CWD") or os.getcwd(),
            "scriptPath": str(saved_path),
            "source": {
                "type": source.source_type,
                "ref": source.source_ref,
            },
            "resumeFromRunId": resume_from,
            "args": args,
            "result": None,
            "error": None,
            "display": "",
            "workflow": None,
            "agentCache": {},
        }
        managed = ManagedRun(run_id=run_id, stop_event=stop_event, record=record)

        with self._lock:
            self._runs[run_id] = managed
        self.store.save_run(record)

        thread = threading.Thread(
            target=self._run_thread,
            args=(managed, source.script, args, config, resume_cache, cwd, plugin_context),
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
    ) -> None:
        try:
            from ..agents.runner import HermesChildAgentRunner

            managed.child_runner = HermesChildAgentRunner(config)
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
                    plugin_context=plugin_context,
                ),
            )
            if managed.stop_event.is_set():
                status = "stopped"
            else:
                status = "completed"
            snapshot = result.state.snapshot()
            self._update(
                managed,
                status=status,
                finishedAt=utc_now_iso(),
                result=result.value,
                workflow=snapshot,
                display=render_workflow_text(snapshot, completed=True),
                agentCache=resume_cache.current,
            )
        except Exception as exc:
            status = "stopped" if managed.stop_event.is_set() else "error"
            self._update(
                managed,
                status=status,
                finishedAt=utc_now_iso(),
                error=f"{type(exc).__name__}: {exc}",
                agentCache=resume_cache.current,
            )

    def _update_state(self, managed: ManagedRun, state) -> None:
        snapshot = state.snapshot()
        self._update(
            managed,
            workflow=snapshot,
            display=render_workflow_text(snapshot, completed=False),
        )

    def _update(self, managed: ManagedRun, **fields: Any) -> None:
        with managed.lock:
            managed.record.update(fields)
            self.store.save_run(managed.record)

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)


def _content_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


_MANAGER: WorkflowRunManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_run_manager() -> WorkflowRunManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = WorkflowRunManager()
        return _MANAGER
