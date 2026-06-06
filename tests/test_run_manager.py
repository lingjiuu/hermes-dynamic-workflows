from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import WorkflowRuntimeError
from hermes_dynamic_workflows.engine import manager as manager_module
from hermes_dynamic_workflows.engine.manager import (
    LiveTranscriptExporter,
    SessionTranscriptReader,
    WorkflowRunManager,
    _capture_parent_runtime,
    _gateway_running_agent,
)
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner
from hermes_dynamic_workflows.storage.store import WorkflowStore


class RecordingRunner(ChildAgentRunner):
    """Thread-safe runner that records each call's label and returns a stable
    per-label result, so a resume that reuses cached results makes no new
    run() calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.labels: list[str] = []

    def run(self, request: ChildAgentRequest):
        with self._lock:
            self.labels.append(request.label)
        return f"result:{request.label}"


class BudgetRunner(ChildAgentRunner):
    def __init__(self, tokens: int):
        self.tokens = tokens

    def run(self, request: ChildAgentRequest):
        return ChildAgentResult(content=request.label, metadata={"tokens": self.tokens})


class RecordingCtx:
    """Fake PluginContext capturing inject_message calls (CLI notification)."""

    def __init__(self, fail: bool = False):
        self.messages: list[str] = []
        self.fail = fail
        self.session_id = "test-session"

    def inject_message(self, content: str, role: str = "user") -> bool:
        if self.fail:
            raise RuntimeError("inject failed")
        self.messages.append(content)
        return True


class CliRefCtx:
    def __init__(self, *, cli_session_id: str, agent_session_id: str | None = None):
        agent = type("Agent", (), {"session_id": agent_session_id})() if agent_session_id else None
        cli = type("Cli", (), {"session_id": cli_session_id, "agent": agent})()
        self._manager = type("PluginManager", (), {"_cli_ref": cli})()


class ParentRuntimeTests(unittest.TestCase):
    def test_cli_plugin_context_supplies_current_agent_runtime(self):
        parent = SimpleNamespace(
            model="cli-session-model",
            provider="cli-provider",
            base_url="https://cli.example/v1",
            api_key="cli-secret",
            api_mode="chat_completions",
        )
        ctx = SimpleNamespace(
            _manager=SimpleNamespace(
                _cli_ref=SimpleNamespace(agent=parent),
            ),
        )

        runtime = _capture_parent_runtime(None, plugin_context=ctx)

        self.assertEqual(runtime["model"], "cli-session-model")
        self.assertEqual(runtime["api_key"], "cli-secret")

    def test_gateway_running_agent_is_used_when_tool_dispatch_has_no_parent_agent(self):
        parent = SimpleNamespace(
            model="gateway-session-model",
            provider="gateway-provider",
            base_url="https://gateway.example/v1",
            api_key="gateway-secret",
            api_mode="chat_completions",
        )

        with patch(
            "hermes_dynamic_workflows.engine.manager._gateway_running_agent",
            return_value=parent,
        ):
            runtime = _capture_parent_runtime(None)

        self.assertEqual(runtime["model"], "gateway-session-model")
        self.assertEqual(runtime["provider"], "gateway-provider")
        self.assertEqual(runtime["api_key"], "gateway-secret")

    def test_gateway_session_agent_falls_back_to_cached_agent(self):
        cached = SimpleNamespace(model="cached-session-model")
        runner = SimpleNamespace(
            _running_agents={},
            _agent_cache={"gateway-key": (cached, "signature")},
        )
        gateway_run = ModuleType("gateway.run")
        gateway_run._gateway_runner_ref = lambda: runner
        gateway_pkg = ModuleType("gateway")
        gateway_pkg.__path__ = []
        gateway_pkg.run = gateway_run

        with (
            patch(
                "hermes_dynamic_workflows.engine.manager._get_hermes_session_env",
                return_value="gateway-key",
            ),
            patch.dict(
                sys.modules,
                {"gateway": gateway_pkg, "gateway.run": gateway_run},
            ),
        ):
            self.assertIs(_gateway_running_agent(), cached)


class FailingRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        raise RuntimeError("always fails")


class HalfFailingRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        if request.label == "a":
            raise RuntimeError("boom")
        return f"ok:{request.label}"


class CountingRunner(ChildAgentRunner):
    calls = 0

    def run(self, request: ChildAgentRequest):
        type(self).calls += 1
        return f"{type(self).calls}:{request.label}"


class TranscriptRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        return ChildAgentResult(
            content=f"done:{request.label}",
            metadata={
                "task_id": f"child-session-{request.id}",
                "session_id": f"child-session-{request.id}",
                "hermes_session_id": f"child-session-{request.id}",
                "tokens": 9,
                "tool_calls": 2,
            },
        )


class LiveTranscriptRunner(ChildAgentRunner):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, request: ChildAgentRequest):
        metadata = {
            "runner": "standalone",
            "task_id": "live-child-session",
            "session_id": "live-child-session",
            "hermes_session_id": "live-child-session",
            "workspace": request.cwd,
            "isolation": request.isolation or "shared",
            "model": "test-model",
            "tokens": 0,
            "tool_calls": 0,
        }
        if request.on_start is not None:
            request.on_start(metadata)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test runner was not released")
        return ChildAgentResult(
            content=f"done:{request.label}",
            metadata={**metadata, "tokens": 11, "tool_calls": 3},
        )


class IncrementalTestDB:
    """Small SessionDB-compatible SQLite store for transcript exporter tests."""

    def __init__(self):
        self._lock = threading.RLock()
        self.get_messages_calls = 0
        self._session_clock = 0
        self._conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                active INTEGER NOT NULL DEFAULT 1
            );
            """
        )

    def close(self):
        with self._lock:
            self._conn.close()

    def create_session(self, session_id: str, parent_session_id: str | None = None):
        with self._lock:
            self._session_clock += 1
            self._conn.execute(
                "INSERT INTO sessions (id, parent_session_id, started_at) VALUES (?, ?, ?)",
                (session_id, parent_session_id, self._session_clock),
            )

    def end_session(self, session_id: str, reason: str):
        with self._lock:
            self._session_clock += 1
            self._conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                (self._session_clock, reason, session_id),
            )

    def append_message(self, session_id: str, role: str, content: str, *, active: bool = True):
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, active) VALUES (?, ?, ?, ?)",
                (session_id, role, content, 1 if active else 0),
            )
            return int(cursor.lastrowid)

    def replace_messages(self, session_id: str, messages: list[dict]):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                for message in messages:
                    self._conn.execute(
                        "INSERT INTO messages (session_id, role, content, active) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            message["role"],
                            message["content"],
                            1 if message.get("active", True) else 0,
                        ),
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def set_active(self, message_id: int, active: bool):
        with self._lock:
            self._conn.execute(
                "UPDATE messages SET active = ? WHERE id = ?",
                (1 if active else 0, message_id),
            )

    def get_messages(self, session_id: str, include_inactive: bool = False):
        self.get_messages_calls += 1
        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ?"
                f"{active_clause} ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _decode_content(content):
        return content


class RecordingSessionTranscriptReader(SessionTranscriptReader):
    def __init__(self, db):
        super().__init__(db=db)
        self.reads: list[str] = []
        self._reads_lock = threading.Lock()

    def read(self, target, *, force_rebuild=False):
        with self._reads_lock:
            self.reads.append(target.session_id)
        return super().read(target, force_rebuild=force_rebuild)

    def clear_reads(self):
        with self._reads_lock:
            self.reads.clear()


class PublicOnlyTranscriptDB:
    """Fallback fixture without SessionDB private connection/schema access."""

    def __init__(self):
        self.messages: dict[str, list[dict]] = {}
        self.get_messages_calls = 0

    def get_messages(self, session_id: str, include_inactive: bool = False):
        self.get_messages_calls += 1
        return list(self.messages.get(session_id, []))

    def close(self):
        return None


class SkipAwareRunner(ChildAgentRunner):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.skipped: list[str] = []

    def run(self, request: ChildAgentRequest):
        if request.on_start is not None:
            request.on_start({"task_id": "child-to-skip", "session_id": "child-to-skip"})
        self.started.set()
        self.release.wait(timeout=5)
        return "released"

    def skip_child(self, task_id: str) -> bool:
        self.skipped.append(task_id)
        self.release.set()
        return True


class MetadataRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        return ChildAgentResult(
            content="metadata-result",
            metadata={
                "runner": "standalone",
                "workspace": request.cwd,
                "agent_type": request.agent_type,
                "isolation": request.isolation or "shared",
                "model": "test-model",
                "tokens": 1234,
                "cache_read_tokens": 2048,
                "cache_write_tokens": 512,
                "tool_calls": 5,
            },
        )


class DictRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        return {
            "items": [
                {
                    "title": request.label,
                    "summary": "structured result",
                    "source": "unit-test",
                }
            ]
        }


class RunManagerTests(unittest.TestCase):
    def setUp(self):
        CountingRunner.calls = 0
        self._env_patcher = patch.dict(os.environ, {"HERMES_SESSION_ID": "unit-session"}, clear=False)
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()

    def test_task_output_path_uses_platform_tempdir_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp) / "store")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_DYNAMIC_WORKFLOWS_TMPDIR", None)
                default_path = store.task_output_path(
                    "C:\\Users\\me\\project",
                    "session-1",
                    "task:1",
                )
            default_path.relative_to(Path(tempfile.gettempdir()))
            self.assertEqual(default_path.name, "task-1.output")

            with tempfile.TemporaryDirectory() as output_tmp:
                with patch.dict(os.environ, {"HERMES_DYNAMIC_WORKFLOWS_TMPDIR": output_tmp}):
                    override_path = store.task_output_path("/repo", "session-2", "task-2")
            override_path.relative_to(Path(output_tmp))

    def test_manager_routes_internal_single_agent_skip(self):
        script = """
meta = {"name": "skip-one", "description": "Test workflow"}

return await agent("wait", {"label": "worker"})
"""
        runner = SkipAwareRunner()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=runner,
            ):
                record = manager.start_from_params({"script": script}, cwd=tmp)
                self.assertTrue(runner.started.wait(timeout=2))
                self.assertTrue(manager.pause(record["runId"]))
                self.assertTrue(manager.skip_agent(record["taskId"], "child-to-skip"))
                self.assertTrue(manager.resume(record["runId"]))
                manager.wait(record["runId"], timeout=2)

        self.assertEqual(runner.skipped, ["child-to-skip"])

    def test_script_path_run(self):
        script = """
meta = {"name": "from-path", "description": "Test workflow"}

return await agent("work", {"label": "path-agent"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_path = root / "workflow.py"
            script_path.write_text(script, encoding="utf-8")
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=CountingRunner()):
                record = manager.start_from_params({"scriptPath": str(script_path)}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "1:path-agent")
        self.assertEqual(final["source"]["type"], "scriptPath")
        self.assertEqual(final["scriptPath"], str(script_path.resolve()))

    def test_inline_script_saved_under_session_workflow_scripts(self):
        script = """
meta = {"name": "Inline Save", "description": "Test workflow"}

return "ok"
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            record = manager.start_from_params({"script": script}, cwd=str(root), plugin_context=RecordingCtx())
            manager.wait(record["runId"], timeout=2)

            script_path = Path(record["scriptPath"])
            self.assertEqual(script_path.name, f"inline-save-{record['runId']}.py")
            self.assertIn("projects", script_path.parts)
            self.assertIn("workflows", script_path.parts)
            self.assertIn("scripts", script_path.parts)
            self.assertTrue(script_path.read_text(encoding="utf-8").strip().startswith("meta ="))

    def test_cli_ref_session_id_is_used_for_workflow_layout(self):
        script = """
meta = {"name": "cli session", "description": "Test workflow"}

return "ok"
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = CliRefCtx(cli_session_id="cli-session", agent_session_id="agent-session")
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            record = manager.start_from_params({"script": script}, cwd=str(root), plugin_context=ctx)
            manager.wait(record["runId"], timeout=2)

            self.assertEqual(record["workflowSessionId"], "agent-session")
            self.assertIn("agent-session", Path(record["scriptPath"]).parts)
            self.assertIn("agent-session", Path(record["transcriptDir"]).parts)

    def test_missing_host_session_id_fails_instead_of_synthesizing(self):
        script = """
meta = {"name": "no session", "description": "Test workflow"}

return "ok"
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(WorkflowRuntimeError):
                    manager.start_from_params({"script": script}, cwd=tmp)

    def test_resume_reuses_unchanged_prefix(self):
        script = """
meta = {"name": "resume", "description": "Test workflow"}

return [
    await agent("a", {"label": "a"}),
    await agent("b", {"label": "b"}),
]
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=CountingRunner()):
                first = manager.start_from_params({"script": script}, cwd=tmp)
                first_final = manager.wait(first["runId"], timeout=2)
                second = manager.start_from_params(
                    {"script": script, "resumeFromRunId": first["runId"]},
                    cwd=tmp,
                )
                second_final = manager.wait(second["runId"], timeout=2)

        self.assertEqual(first_final["result"], ["1:a", "2:b"])
        self.assertEqual(second_final["result"], ["1:a", "2:b"])
        self.assertEqual(CountingRunner.calls, 2)

    def test_formats_agent_overview(self):
        script = """
meta = {"name": "inspectable", "description": "Test workflow", "phases": ["Search"]}

phase("Search")
return await agent("inspect metadata", {"label": "meta-agent", "agentType": "researcher"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_types = root / ".hermes" / "dynamic-workflows" / "agents"
            agent_types.mkdir(parents=True)
            (agent_types / "researcher.md").write_text(
                "Inspect carefully and return raw findings.",
                encoding="utf-8",
            )
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=MetadataRunner()):
                record = manager.start_from_params({"script": script}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)
                overview = manager.format_agent_overview()

        self.assertIn("inspectable", overview)
        self.assertIn(final["runId"], overview)
        self.assertIn("meta-agent", overview)
        self.assertIn("test-model", overview)
        self.assertIn("1.2K tok", overview)
        self.assertIn("2.0K cached read", overview)

    def test_resume_reuses_parallel_results(self):
        # Regression for the content-addressed resume cache: under the old
        # sequence-keyed cache, parallel()'s non-deterministic reserve order
        # broke resume after the first parallel block. Fingerprint keying makes
        # resume order-independent, so the second run reuses all three results
        # and issues no new child runs.
        script = """
meta = {"name": "parallel-resume", "description": "Test workflow"}

return await parallel([
    lambda: agent("alpha", {"label": "a"}),
    lambda: agent("beta", {"label": "b"}),
    lambda: agent("gamma", {"label": "c"}),
])
"""
        runner = RecordingRunner()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)), config=PluginConfig(concurrency=3, require_launch_approval=False)
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=runner,
            ):
                first = manager.start_from_params({"script": script}, cwd=tmp)
                first_final = manager.wait(first["runId"], timeout=3)
                self.assertEqual(len(runner.labels), 3)
                second = manager.start_from_params(
                    {"script": script, "resumeFromRunId": first["runId"]}, cwd=tmp
                )
                second_final = manager.wait(second["runId"], timeout=3)

        self.assertEqual(first_final["status"], "completed")
        self.assertEqual(second_final["result"], first_final["result"])
        # No new child runs on resume — all three came from the cache.
        self.assertEqual(len(runner.labels), 3)

    def test_internal_token_budget_gates_run(self):
        script = """
meta = {"name": "budget-param", "description": "Test workflow"}

await agent("a", {"label": "a"})
return await agent("b", {"label": "b"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=BudgetRunner(tokens=20_000),
            ):
                record = manager.start_from_params(
                    {"script": script, "token_budget": 1},
                    cwd=tmp,
                    user_task="+10k run a workflow",
                )
                final = manager.wait(record["runId"], timeout=2)

        self.assertEqual(record["tokenBudget"], 10_000)
        # First agent spends 20k > 10k, so the second agent's reservation trips the
        # hard ceiling and the run fails.
        self.assertEqual(final["status"], "failed")
        self.assertIn("budget", (final["error"] or "").lower())

    def test_all_agents_failed_inside_parallel_still_completes_with_none_results(self):
        script = """
meta = {"name": "all-fail", "description": "Test workflow"}

return await parallel([
    lambda: agent("a", {"label": "a"}),
    lambda: agent("b", {"label": "b"}),
])
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)), config=PluginConfig(concurrency=2, require_launch_approval=False)
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=FailingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp)
                final = manager.wait(rec["runId"], timeout=3)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], [None, None])

    def test_partial_failure_stays_completed(self):
        script = """
meta = {"name": "partial", "description": "Test workflow"}

return await parallel([
    lambda: agent("a", {"label": "a"}),
    lambda: agent("b", {"label": "b"}),
])
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)), config=PluginConfig(concurrency=2, require_launch_approval=False)
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=HalfFailingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp)
                final = manager.wait(rec["runId"], timeout=3)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], [None, "ok:b"])

    def test_completion_injects_task_notification(self):
        script = """
meta = {"name": "notify-me", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        ctx = RecordingCtx()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=CountingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp, plugin_context=ctx)
                final = manager.wait(rec["runId"], timeout=2)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(len(ctx.messages), 1)
        msg = ctx.messages[0]
        self.assertIn("<task-notification>", msg)
        self.assertIn(f"<task-id>{rec['taskId']}</task-id>", msg)
        self.assertIn("<output-file>", msg)
        self.assertIn("<status>completed</status>", msg)
        self.assertIn('Dynamic workflow "notify-me" completed', msg)
        self.assertIn("<agent_count>1</agent_count>", msg)
        self.assertIn("<subagent_tokens>", msg)
        self.assertIn("<tool_uses>", msg)
        self.assertIn("<duration_ms>", msg)
        self.assertNotIn("<errors>", msg)
        self.assertTrue(msg.strip().endswith("</task-notification>"))
        self.assertTrue(Path(final["outputFile"]).is_file())

    def test_runtime_failure_injects_failed_task_notification_without_result(self):
        script = """
meta = {"name": "runtime-boom", "description": "Runtime boom"}

raise Exception("boom")
"""
        ctx = RecordingCtx()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=CountingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp, plugin_context=ctx)
                final = manager.wait(rec["runId"], timeout=2)

        self.assertEqual(final["status"], "failed")
        self.assertIn("Exception: boom", final["error"])
        self.assertEqual(len(ctx.messages), 1)
        msg = ctx.messages[0]
        self.assertIn("<task-notification>", msg)
        self.assertIn(f"<task-id>{rec['taskId']}</task-id>", msg)
        self.assertIn("<output-file>", msg)
        self.assertIn("<status>failed</status>", msg)
        self.assertIn('Dynamic workflow "runtime-boom" failed: Exception: boom', msg)
        self.assertIn(f"<recovery>Agent transcripts: {final['transcriptDir']}</recovery>", msg)
        self.assertIn("<agent_count>0</agent_count>", msg)
        self.assertNotIn("<result>", msg)
        self.assertTrue(Path(final["outputFile"]).is_file())
        self.assertIn("Exception: boom", Path(final["outputFile"]).read_text(encoding="utf-8"))

    def test_child_transcript_files_are_written_while_running(self):
        script = """
meta = {"name": "live-transcripts", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        runner = LiveTranscriptRunner()
        fake_messages = [{"role": "user", "content": "running message"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(
                store=WorkflowStore(root / "store"),
                config=PluginConfig(require_launch_approval=False),
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=runner,
            ), patch(
                "hermes_dynamic_workflows.engine.manager._load_session_messages",
                side_effect=lambda session_id: list(fake_messages),
            ):
                rec = manager.start_from_params({"script": script}, cwd=str(root), plugin_context=RecordingCtx())
                self.assertTrue(runner.started.wait(timeout=2))

                running = manager.get(rec["runId"])
                self.assertEqual(running["status"], "running")
                transcript_path = Path(running["transcriptFiles"][0])
                meta_path = Path(running["transcriptMetaFiles"][0])
                self.assertEqual(transcript_path.name, "agent-live-child-session.jsonl")
                self.assertEqual(meta_path.name, "agent-live-child-session.meta.json")
                self.assertTrue(transcript_path.is_file())
                self.assertTrue(meta_path.is_file())
                self.assertIn("running message", transcript_path.read_text(encoding="utf-8"))
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.assertEqual(meta["session_id"], "live-child-session")
                self.assertEqual(meta["agent_label"], "worker")

                fake_messages.append({"role": "assistant", "content": "final message"})
                runner.release.set()
                final = manager.wait(rec["runId"], timeout=2)

            final_path = Path(final["transcriptFiles"][0])
            self.assertIn("final message", final_path.read_text(encoding="utf-8"))
            self.assertEqual(final["workflow"]["agents"][0]["transcript_path"], str(final_path))
            self.assertEqual(final["workflow"]["agents"][0]["transcript_meta_path"], str(meta_path))

    def test_live_transcript_exporter_batches_active_agents_and_skips_unchanged_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = IncrementalTestDB()
            for session_id in ("child-a", "child-b"):
                db.create_session(session_id)
                db.append_message(session_id, "user", session_id)
            reader = RecordingSessionTranscriptReader(db)
            exporter = LiveTranscriptExporter(
                run_id="wf_batch-test",
                interval_seconds=60,
                reader=reader,
            )
            with patch(
                "hermes_dynamic_workflows.engine.manager._write_agent_transcript_files",
            ) as write_files:
                exporter.start()
                for session_id in ("child-a", "child-b"):
                    transcript_path = root / f"agent-{session_id}.jsonl"
                    exporter.upsert(
                        session_id=session_id,
                        transcript_path=transcript_path,
                        meta_path=transcript_path.with_suffix(".meta.json"),
                        metadata={"session_id": session_id, "agent_status": "running"},
                        active=True,
                    )

                self.assertEqual(write_files.call_count, 2)
                reader.clear_reads()
                exporter.flush(active_only=True)
                self.assertEqual(write_files.call_count, 2)
                self.assertEqual(reader.reads, ["child-a", "child-b"])

                exporter.upsert(
                    session_id="child-b",
                    transcript_path=root / "agent-child-b.jsonl",
                    meta_path=root / "agent-child-b.meta.json",
                    metadata={"session_id": "child-b", "agent_status": "done"},
                    active=False,
                )
                reader.clear_reads()
                exporter.flush(active_only=True)
                self.assertEqual(reader.reads, ["child-a"])
                self.assertEqual(write_files.call_count, 2)

                exporter.stop(final=True)
                self.assertEqual(write_files.call_count, 4)
                self.assertFalse(exporter._thread.is_alive())

    def test_live_transcript_incremental_append_lineage_and_rebuild_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = IncrementalTestDB()
            db.create_session("root-session")
            first_id = db.append_message("root-session", "user", "first")
            reader = SessionTranscriptReader(db=db)
            exporter = LiveTranscriptExporter(
                run_id="wf_incremental-test",
                interval_seconds=60,
                reader=reader,
            )
            transcript_path = root / "agent-root-session.jsonl"
            meta_path = transcript_path.with_suffix(".meta.json")
            exporter.upsert(
                session_id="root-session",
                transcript_path=transcript_path,
                meta_path=meta_path,
                metadata={"session_id": "root-session", "agent_status": "running"},
                active=True,
            )

            db.append_message("root-session", "assistant", "appended")
            exporter.flush(active_only=True)
            lines = transcript_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("appended", lines[-1])

            db.create_session("delegate-session", parent_session_id="root-session")
            db.append_message("delegate-session", "user", "delegate must stay separate")
            db.end_session("root-session", "compression")
            db.create_session("compressed-session", parent_session_id="root-session")
            db.append_message("compressed-session", "user", "after compression")
            exporter.flush(active_only=True)
            content = transcript_path.read_text(encoding="utf-8")
            self.assertIn("first", content)
            self.assertIn("appended", content)
            self.assertIn("after compression", content)
            self.assertNotIn("delegate must stay separate", content)

            db.set_active(first_id, False)
            exporter.flush(active_only=True)
            rows = [
                json.loads(line)["message"]
                for line in transcript_path.read_text(encoding="utf-8").splitlines()
            ]
            first = next(row for row in rows if row["content"] == "first")
            self.assertEqual(first["active"], 0)

            db.replace_messages(
                "root-session",
                [{"role": "user", "content": "replacement"}],
            )
            exporter.flush(active_only=True)
            content = transcript_path.read_text(encoding="utf-8")
            self.assertNotIn('"content": "first"', content)
            self.assertNotIn('"content": "appended"', content)
            self.assertIn("replacement", content)
            self.assertIn("after compression", content)

            exporter.upsert(
                session_id="root-session",
                transcript_path=transcript_path,
                meta_path=meta_path,
                metadata={"session_id": "root-session", "agent_status": "done"},
                active=False,
            )
            exporter.stop(final=True)

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["transcript_export_mode"], "incremental")
            self.assertEqual(
                set(meta["transcript_lineage_session_ids"]),
                {"root-session", "compressed-session"},
            )
            # The public full-read API is never used on the incremental path.
            self.assertEqual(db.get_messages_calls, 0)

    def test_live_transcript_incremental_capability_failure_uses_full_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = PublicOnlyTranscriptDB()
            db.messages["legacy-session"] = [{"role": "user", "content": "first"}]
            exporter = LiveTranscriptExporter(
                run_id="wf_fallback-test",
                interval_seconds=60,
                reader=SessionTranscriptReader(db=db),
            )
            transcript_path = root / "agent-legacy-session.jsonl"
            meta_path = transcript_path.with_suffix(".meta.json")
            exporter.upsert(
                session_id="legacy-session",
                transcript_path=transcript_path,
                meta_path=meta_path,
                metadata={"session_id": "legacy-session", "agent_status": "running"},
                active=True,
            )

            db.messages["legacy-session"].append({"role": "assistant", "content": "second"})
            exporter.flush(active_only=True)
            before = transcript_path.read_text(encoding="utf-8")
            exporter.flush(active_only=True)
            after = transcript_path.read_text(encoding="utf-8")
            self.assertEqual(before, after)

            exporter.stop(final=True)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["transcript_export_mode"], "full_fallback")
            self.assertIn(
                "private connection is unavailable",
                meta["transcript_export_fallback_reason"],
            )
            self.assertGreaterEqual(db.get_messages_calls, 3)
            self.assertIn("second", after)

    def test_live_transcript_append_uses_single_os_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent-child.jsonl"
            original_write = manager_module.os.write
            write_lengths: list[int] = []

            def record_write(fd, payload):
                write_lengths.append(len(payload))
                return original_write(fd, payload)

            with patch("hermes_dynamic_workflows.engine.manager.os.write", side_effect=record_write):
                manager_module._append_agent_transcript_messages(
                    path,
                    [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "second"},
                    ],
                )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(write_lengths), 1)
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["message"]["content"], "first")
            self.assertEqual(json.loads(lines[1])["message"]["content"], "second")

    def test_live_transcript_without_active_column_still_uses_incremental_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = IncrementalTestDB()
            db.create_session("legacy-schema-session")
            db.append_message("legacy-schema-session", "user", "legacy content")
            db._conn.execute("ALTER TABLE messages DROP COLUMN active")
            exporter = LiveTranscriptExporter(
                run_id="wf-legacy-incremental-test",
                interval_seconds=60,
                reader=SessionTranscriptReader(db=db),
            )
            transcript_path = root / "agent-legacy-schema-session.jsonl"
            meta_path = transcript_path.with_suffix(".meta.json")
            exporter.upsert(
                session_id="legacy-schema-session",
                transcript_path=transcript_path,
                meta_path=meta_path,
                metadata={"session_id": "legacy-schema-session", "agent_status": "running"},
                active=True,
            )
            db._conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                ("legacy-schema-session", "assistant", "incremental legacy append"),
            )
            exporter.flush(active_only=True)
            db._conn.execute("DELETE FROM messages WHERE session_id = ?", ("legacy-schema-session",))
            db._conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                ("legacy-schema-session", "user", "legacy replacement"),
            )
            exporter.flush(active_only=True)
            exporter.stop(final=True)

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["transcript_export_mode"], "incremental")
            self.assertNotIn("transcript_export_fallback_reason", meta)
            content = transcript_path.read_text(encoding="utf-8")
            self.assertNotIn("legacy content", content)
            self.assertNotIn("incremental legacy append", content)
            self.assertIn("legacy replacement", content)
            self.assertEqual(db.get_messages_calls, 0)

    def test_live_transcript_schema_mismatch_uses_full_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = IncrementalTestDB()
            db.create_session("unsupported-schema-session")
            db.append_message("unsupported-schema-session", "user", "fallback content")
            db._conn.execute("ALTER TABLE messages DROP COLUMN tool_calls")
            exporter = LiveTranscriptExporter(
                run_id="wf-schema-fallback-test",
                interval_seconds=60,
                reader=SessionTranscriptReader(db=db),
            )
            transcript_path = root / "agent-unsupported-schema-session.jsonl"
            meta_path = transcript_path.with_suffix(".meta.json")
            exporter.upsert(
                session_id="unsupported-schema-session",
                transcript_path=transcript_path,
                meta_path=meta_path,
                metadata={"session_id": "unsupported-schema-session", "agent_status": "running"},
                active=True,
            )
            exporter.stop(final=True)

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["transcript_export_mode"], "full_fallback")
            self.assertIn("messages missing=['tool_calls']", meta["transcript_export_fallback_reason"])
            self.assertIn("fallback content", transcript_path.read_text(encoding="utf-8"))

    def test_live_transcript_exporter_stress_hundreds_of_concurrent_agents(self):
        agent_count = 300
        update_rounds = 3
        session_ids = [f"stress-child-{index:03d}" for index in range(agent_count)]
        stats_lock = threading.Lock()
        rebuild_writes: Counter[str] = Counter()
        append_writes: Counter[str] = Counter()
        original_write = manager_module._write_agent_transcript_files
        original_append = manager_module._append_agent_transcript_messages

        def write_files(path, meta_path, *, metadata, messages):
            with stats_lock:
                rebuild_writes[str(path)] += 1
            original_write(
                path,
                meta_path,
                metadata=metadata,
                messages=messages,
            )

        def append_messages(path, messages):
            with stats_lock:
                append_writes[str(path)] += len(messages)
            original_append(path, messages)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "wf_stress-test"
            db = IncrementalTestDB()
            for session_id in session_ids:
                db.create_session(session_id)
                db.append_message(session_id, "user", f"start:{session_id}")
            reader = RecordingSessionTranscriptReader(db)
            exporter = LiveTranscriptExporter(
                run_id=run_id,
                interval_seconds=0.005,
                reader=reader,
            )

            def upsert(session_id: str, status: str) -> None:
                transcript_path = root / f"agent-{session_id}.jsonl"
                exporter.upsert(
                    session_id=session_id,
                    transcript_path=transcript_path,
                    meta_path=transcript_path.with_suffix(".meta.json"),
                    metadata={"session_id": session_id, "agent_status": status},
                    active=status == "running",
                )

            with patch(
                "hermes_dynamic_workflows.engine.manager._write_agent_transcript_files",
                side_effect=write_files,
            ), patch(
                "hermes_dynamic_workflows.engine.manager._append_agent_transcript_messages",
                side_effect=append_messages,
            ):
                exporter.start()
                with ThreadPoolExecutor(max_workers=32) as pool:
                    list(pool.map(lambda session_id: upsert(session_id, "running"), session_ids))

                exporter_threads = [
                    thread
                    for thread in threading.enumerate()
                    if thread.name == f"workflow-transcripts-{run_id}"
                ]
                self.assertEqual(len(exporter_threads), 1)

                for round_index in range(update_rounds):
                    for session_id in session_ids:
                        db.append_message(session_id, "assistant", f"round:{round_index}")
                    # Concurrent callers simulate an immediate state update racing
                    # the exporter's own periodic refresh.
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        list(pool.map(lambda _: exporter.flush(active_only=True), range(8)))

                completed = session_ids[: agent_count // 2]
                active = session_ids[agent_count // 2 :]
                with ThreadPoolExecutor(max_workers=32) as pool:
                    list(pool.map(lambda session_id: upsert(session_id, "done"), completed))

                # Stop the periodic loop after exercising its races above, then
                # make the active-set assertion deterministic.
                exporter.stop(final=False)
                reader.clear_reads()
                exporter.flush(active_only=True)
                loaded_after_half_complete = set(reader.reads)
                self.assertTrue(set(active).issubset(loaded_after_half_complete))
                self.assertTrue(set(completed).isdisjoint(loaded_after_half_complete))

                with ThreadPoolExecutor(max_workers=32) as pool:
                    list(pool.map(lambda session_id: upsert(session_id, "done"), active))
                exporter.stop(final=True)

            self.assertFalse(exporter._thread.is_alive())
            self.assertFalse(list(root.glob("*.tmp")))
            self.assertEqual(len(list(root.glob("*.jsonl"))), agent_count)
            self.assertEqual(len(list(root.glob("*.meta.json"))), agent_count)
            # Initial and final validation rebuilds only. Intermediate rounds are
            # true append-only writes despite concurrent flush callers.
            self.assertEqual(set(rebuild_writes.values()), {2})
            self.assertEqual(set(append_writes.values()), {update_rounds})
            for session_id in session_ids:
                transcript_path = root / f"agent-{session_id}.jsonl"
                meta_path = transcript_path.with_suffix(".meta.json")
                self.assertIn(
                    f"round:{update_rounds - 1}",
                    transcript_path.read_text(encoding="utf-8"),
                )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.assertEqual(meta["agent_status"], "done")
                self.assertEqual(meta["transcript_export_mode"], "incremental")

    def test_completion_exports_child_transcripts_after_run(self):
        script = """
meta = {"name": "transcripts", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        fake_messages = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "done"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=TranscriptRunner(),
            ), patch(
                "hermes_dynamic_workflows.engine.manager._load_session_messages",
                return_value=fake_messages,
            ):
                rec = manager.start_from_params({"script": script}, cwd=str(root), plugin_context=RecordingCtx())
                final = manager.wait(rec["runId"], timeout=2)

            transcript_dir = Path(final["transcriptDir"])
            self.assertEqual(transcript_dir.name, final["runId"])
            files = final["transcriptFiles"]
            self.assertEqual(len(files), 1)
            transcript_path = Path(files[0])
            self.assertEqual(transcript_path.parent, transcript_dir)
            self.assertEqual(transcript_path.name, "agent-child-session-1.jsonl")
            meta_files = final["transcriptMetaFiles"]
            self.assertEqual(len(meta_files), 1)
            meta_path = Path(meta_files[0])
            self.assertEqual(meta_path.name, "agent-child-session-1.meta.json")
            content = transcript_path.read_text(encoding="utf-8")
            self.assertIn('"content": "done"', content)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["session_id"], "child-session-1")
            self.assertEqual(meta["agent_label"], "worker")
            agent = final["workflow"]["agents"][0]
            self.assertEqual(agent["transcript_path"], str(transcript_path))
            self.assertEqual(agent["transcript_meta_path"], str(meta_path))

    def test_journal_records_started_and_full_agent_result(self):
        script = """
meta = {"name": "journal", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(
                store=WorkflowStore(root / "store"),
                config=PluginConfig(require_launch_approval=False),
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=DictRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=str(root), plugin_context=RecordingCtx())
                final = manager.wait(rec["runId"], timeout=2)

            journal_path = Path(final["journalFile"])
            self.assertEqual(journal_path.parent, Path(final["transcriptDir"]))
            events = [
                json.loads(line)
                for line in journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([event["type"] for event in events], ["started", "result"])
        self.assertTrue(events[0]["key"].startswith("v2:"))
        self.assertEqual(events[0]["agentId"], "1")
        self.assertEqual(events[1]["agentId"], "1")
        self.assertEqual(
            events[1]["result"],
            {
                "items": [
                    {
                        "title": "worker",
                        "summary": "structured result",
                        "source": "unit-test",
                    }
                ]
            },
        )

    def test_completion_notification_disabled(self):
        script = """
meta = {"name": "quiet", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        ctx = RecordingCtx()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(notify_on_complete=False, require_launch_approval=False),
            )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=CountingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp, plugin_context=ctx)
                manager.wait(rec["runId"], timeout=2)

        self.assertEqual(ctx.messages, [])

    def test_completion_notification_failure_does_not_break_run(self):
        script = """
meta = {"name": "notify-fail", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        ctx = RecordingCtx(fail=True)  # inject_message raises (e.g. gateway/edge)
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=CountingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp, plugin_context=ctx)
                final = manager.wait(rec["runId"], timeout=2)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "1:worker")


if __name__ == "__main__":
    unittest.main()
