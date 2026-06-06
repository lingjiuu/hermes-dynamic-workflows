from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.errors import WorkflowRuntimeError
from hermes_dynamic_workflows.engine.manager import WorkflowRunManager
from hermes_dynamic_workflows.engine.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner
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

    def test_script_path_run(self):
        script = """
meta = {"name": "from-path"}

def workflow():
    return agent("work", {"label": "path-agent"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_path = root / "workflow.py"
            script_path.write_text(script, encoding="utf-8")
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=CountingRunner()):
                record = manager.start_from_params({"scriptPath": str(script_path)}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "1:path-agent")
        self.assertEqual(final["source"]["type"], "scriptPath")
        self.assertEqual(final["scriptPath"], str(script_path.resolve()))

    def test_inline_script_saved_under_session_workflow_scripts(self):
        script = """
meta = {"name": "Inline Save"}

def workflow():
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
meta = {"name": "cli session"}

def workflow():
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
meta = {"name": "no session"}

def workflow():
    return "ok"
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(WorkflowRuntimeError):
                    manager.start_from_params({"script": script}, cwd=tmp)

    def test_resume_reuses_unchanged_prefix(self):
        script = """
meta = {"name": "resume"}

def workflow():
    return [
        agent("a", {"label": "a"}),
        agent("b", {"label": "b"}),
    ]
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=CountingRunner()):
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

    def test_formats_agent_detail_and_saves_markdown(self):
        script = """
meta = {"name": "inspectable", "phases": ["Search"]}

def workflow():
    phase("Search")
    return agent("inspect metadata", {"label": "meta-agent", "agentType": "researcher"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=MetadataRunner()):
                record = manager.start_from_params({"script": script}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)
                detail = manager.format_agent(final["runId"], "1")
                saved = manager.save_markdown(final["runId"])

        self.assertIn("meta-agent", detail)
        self.assertIn("test-model", detail)
        self.assertIn("1.2K tok", detail)
        self.assertIn("2.0K cached read", detail)
        self.assertIn("Saved workflow", saved)

    def test_save_named_workflow_writes_reusable_script(self):
        from hermes_dynamic_workflows.ui.commands import discover_named_workflows

        script = """
meta = {"name": "audit"}

def workflow():
    return agent("audit", {"label": "auditor"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowStore(root / "store")
            manager = WorkflowRunManager(store=store, config=PluginConfig(require_launch_approval=False))
            with patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=CountingRunner()):
                record = manager.start_from_params({"script": script}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)

            project = manager.save_named_workflow(final["runId"], "repo-audit", scope="project", cwd=str(root))
            user = manager.save_named_workflow(final["runId"], "user-audit", scope="user", cwd=str(root))
            reserved = manager.save_named_workflow(final["runId"], "workflows", scope="project", cwd=str(root))

            self.assertTrue(project["ok"])
            self.assertEqual(project["name"], "repo-audit")
            project_path = Path(project["path"])
            self.assertEqual(project_path, root / ".hermes" / "workflows" / "repo-audit.py")
            self.assertIn("def workflow()", project_path.read_text(encoding="utf-8"))

            self.assertTrue(user["ok"])
            self.assertEqual(Path(user["path"]), store.workflows_dir / "user-audit.py")

            self.assertFalse(reserved["ok"])

            discovered = discover_named_workflows(str(root))
            self.assertIn("repo-audit", discovered)

    def test_resume_reuses_parallel_results(self):
        # Regression for the content-addressed resume cache: under the old
        # sequence-keyed cache, parallel()'s non-deterministic reserve order
        # broke resume after the first parallel block. Fingerprint keying makes
        # resume order-independent, so the second run reuses all three results
        # and issues no new child runs.
        script = """
meta = {"name": "parallel-resume"}

def workflow():
    return parallel([
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
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
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
meta = {"name": "budget-param"}

def workflow():
    agent("a", {"label": "a"})
    return agent("b", {"label": "b"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
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
        # hard ceiling and the run errors.
        self.assertEqual(final["status"], "error")
        self.assertIn("budget", (final["error"] or "").lower())

    def test_all_agents_failed_marks_run_failed(self):
        script = """
meta = {"name": "all-fail"}

def workflow():
    return parallel([
        lambda: agent("a", {"label": "a"}),
        lambda: agent("b", {"label": "b"}),
    ])
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)), config=PluginConfig(concurrency=2, require_launch_approval=False)
            )
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
                return_value=FailingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp)
                final = manager.wait(rec["runId"], timeout=3)

        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["result"], [None, None])

    def test_partial_failure_stays_completed(self):
        script = """
meta = {"name": "partial"}

def workflow():
    return parallel([
        lambda: agent("a", {"label": "a"}),
        lambda: agent("b", {"label": "b"}),
    ])
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)), config=PluginConfig(concurrency=2, require_launch_approval=False)
            )
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
                return_value=HalfFailingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp)
                final = manager.wait(rec["runId"], timeout=3)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], [None, "ok:b"])

    def test_completion_injects_task_notification(self):
        script = """
meta = {"name": "notify-me"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        ctx = RecordingCtx()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
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

    def test_child_transcript_files_are_written_while_running(self):
        script = """
meta = {"name": "live-transcripts"}

def workflow():
    return agent("do it", {"label": "worker"})
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
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
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

    def test_completion_exports_child_transcripts_after_run(self):
        script = """
meta = {"name": "transcripts"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        fake_messages = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "done"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
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
meta = {"name": "journal"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowRunManager(
                store=WorkflowStore(root / "store"),
                config=PluginConfig(require_launch_approval=False),
            )
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
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
meta = {"name": "quiet"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        ctx = RecordingCtx()
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(notify_on_complete=False, require_launch_approval=False),
            )
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
                return_value=CountingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp, plugin_context=ctx)
                manager.wait(rec["runId"], timeout=2)

        self.assertEqual(ctx.messages, [])

    def test_completion_notification_failure_does_not_break_run(self):
        script = """
meta = {"name": "notify-fail"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        ctx = RecordingCtx(fail=True)  # inject_message raises (e.g. gateway/edge)
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig(require_launch_approval=False))
            with patch(
                "hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner",
                return_value=CountingRunner(),
            ):
                rec = manager.start_from_params({"script": script}, cwd=tmp, plugin_context=ctx)
                final = manager.wait(rec["runId"], timeout=2)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "1:worker")


if __name__ == "__main__":
    unittest.main()
