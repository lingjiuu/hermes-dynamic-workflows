from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.engine.manager import WorkflowRunManager
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentRunner
from hermes_dynamic_workflows.plugin.task_stop import task_stop
from hermes_dynamic_workflows.plugin.workflow import (
    DYNAMIC_WORKFLOW_SCHEMA,
    get_dynamic_workflow_schema,
    workflow,
)
from hermes_dynamic_workflows.storage.store import WorkflowStore


class FakeRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        return f"done:{request.label}"


class BlockingRunner(ChildAgentRunner):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.interrupted = False

    def run(self, request: ChildAgentRequest):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test runner was not released")
        return f"done:{request.label}"

    def interrupt_all(self):
        self.interrupted = True
        self.release.set()


class ToolTests(unittest.TestCase):
    def test_dynamic_schema_lists_available_workflow_agent_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_dir = root / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "reviewer.md").write_text(
                """---
name: code-reviewer
description: Review code for bugs and regressions.
model: test-model
toolsets: [file, terminal]
---

Review code carefully.
""",
                encoding="utf-8",
            )

            schema = get_dynamic_workflow_schema(cwd=str(root))

        self.assertIn("Available agent types and the tools they have access to:", schema["description"])
        self.assertIn(
            "- code-reviewer: Review code for bugs and regressions. "
            "(Tools: read_file, write_file, patch, search_files, terminal, process)",
            schema["description"],
        )
        self.assertNotIn("- none discovered", schema["description"])

    def test_tool_parses_budget_only_from_user_task(self):
        script = """
meta = {"name": "budget-source", "description": "Test workflow"}

await agent("do it", {"label": "worker"})
return budget.total
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=FakeRunner()),
            ):
                with_budget = workflow(
                    {"script": script, "token_budget": 1},
                    task_id="tool-session",
                    user_task="+500k run a workflow",
                )
                with_budget_id = re.search(
                    r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$",
                    with_budget,
                    re.MULTILINE,
                ).group(1)
                with_budget_final = manager.wait(with_budget_id, timeout=2)

                without_budget = workflow(
                    {"script": script, "token_budget": 1},
                    task_id="tool-session",
                    user_task="run a workflow",
                )
                without_budget_id = re.search(
                    r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$",
                    without_budget,
                    re.MULTILINE,
                ).group(1)
                without_budget_final = manager.wait(without_budget_id, timeout=2)

        self.assertNotIn("token_budget", DYNAMIC_WORKFLOW_SCHEMA["parameters"]["properties"])
        self.assertEqual(with_budget_final["tokenBudget"], 500_000)
        self.assertEqual(with_budget_final["result"], 500_000)
        self.assertIsNone(without_budget_final["tokenBudget"])
        self.assertIsNone(without_budget_final["result"])

    def test_tool_returns_claude_style_launch_text(self):
        script = """
meta = {"name": "tool-test", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=FakeRunner()),
            ):
                result = workflow({"script": script, "args": ["x"]}, task_id="tool-session")
                match = re.search(r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$", result, re.MULTILINE)
                self.assertIsNotNone(match)
                run_id = match.group(1)
                final = manager.wait(run_id, timeout=2)

        self.assertRegex(result, r"Workflow launched in background\. Task ID: wg[a-z0-9]{7}")
        self.assertIn("Summary: Test workflow", result)
        self.assertIn("Transcript dir:", result)
        self.assertIn("tool-session", result)
        self.assertNotIn("(written when the workflow completes)", result)
        self.assertIn("Script file:", result)
        self.assertIn(f"Run ID: {run_id}", result)
        self.assertIn("To resume after editing the script: Workflow({scriptPath:", result)
        self.assertIn(f'resumeFromRunId: "{run_id}"', result)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "done:worker")
        self.assertEqual(final["workflow"]["meta"]["name"], "tool-test")

    def test_tool_captures_parent_runtime_in_memory_without_persisting_credentials(self):
        script = """
meta = {"name": "inherit-runtime", "description": "Test workflow"}

return await agent("do it", {"label": "worker"})
"""
        parent = SimpleNamespace(
            model="session-switched-model",
            provider="custom:session",
            base_url="https://session.example/v1",
            api_key="session-secret",
            api_mode="chat_completions",
            reasoning_config={"effort": "high"},
            service_tier="priority",
            request_overrides={"extra_body": {"routing": "session"}},
            _credential_pool=object(),
            _fallback_chain=[{"provider": "fallback", "model": "fallback-model"}],
        )
        captured = {}

        def runner_factory(config, session_context=None, parent_runtime=None):
            captured["parent_runtime"] = parent_runtime
            return FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            manager = WorkflowRunManager(
                store=store,
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch(
                    "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                    side_effect=runner_factory,
                ),
            ):
                launch = workflow(
                    {"script": script},
                    task_id="tool-session",
                    parent_agent=parent,
                )
                run_id = re.search(
                    r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$",
                    launch,
                    re.MULTILINE,
                ).group(1)
                final = manager.wait(run_id, timeout=2)
                persisted = store.load_run(run_id)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(captured["parent_runtime"]["model"], "session-switched-model")
        self.assertEqual(captured["parent_runtime"]["api_key"], "session-secret")
        self.assertNotIn("parent_runtime", persisted)
        self.assertNotIn("session-secret", json.dumps(persisted))

    def test_task_stop_stops_active_workflow_by_task_id(self):
        script = """
meta = {"name": "stop-test", "description": "Stop me"}

return await agent("wait", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = BlockingRunner()
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.plugin.task_stop.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=runner),
            ):
                launch = workflow({"script": script}, task_id="tool-session")
                run_id = re.search(
                    r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$",
                    launch,
                    re.MULTILINE,
                ).group(1)
                task_id = re.search(r"Task ID: (wg[a-z0-9]{7})", launch).group(1)
                self.assertTrue(runner.started.wait(timeout=2))

                out = json.loads(task_stop({"task_id": task_id}))
                second = json.loads(task_stop({"task_id": task_id}))
                final = manager.wait(run_id, timeout=2)

        self.assertEqual(
            out,
            {
                "message": f"Successfully stopped task: {task_id} (Stop me)",
                "task_id": task_id,
                "task_type": "local_workflow",
            },
        )
        self.assertTrue(runner.interrupted)
        self.assertEqual(
            second,
            {"error": f"No task found with ID: {task_id}"},
        )
        self.assertEqual(final["status"], "stopped")

    def test_task_stop_errors_for_missing_or_unknown_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with patch("hermes_dynamic_workflows.plugin.task_stop.get_run_manager", return_value=manager):
                missing = task_stop({})
                unknown = task_stop({"task_id": "wgunknown"})

        self.assertEqual(
            json.loads(missing),
            {"error": "Missing required parameter: task_id"},
        )
        self.assertEqual(
            json.loads(unknown),
            {"error": "No task found with ID: wgunknown"},
        )

    def test_resume_active_workflow_returns_tool_use_error(self):
        script = """
meta = {"name": "resume-active", "description": "Still running"}

return await agent("wait", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = BlockingRunner()
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner", return_value=runner),
            ):
                launch = workflow({"script": script}, task_id="tool-session")
                run_id = re.search(
                    r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$",
                    launch,
                    re.MULTILINE,
                ).group(1)
                task_id = re.search(r"Task ID: (wg[a-z0-9]{7})", launch).group(1)
                self.assertTrue(runner.started.wait(timeout=2))

                blocked = workflow(
                    {"script": script, "resumeFromRunId": run_id},
                    task_id="tool-session",
                )
                manager.stop_task(task_id)
                final = manager.wait(run_id, timeout=2)
                run_count = len(manager.list(limit=10))

        self.assertEqual(
            json.loads(blocked),
            {
                "error": (
                    f"Workflow {run_id} is still running (task {task_id}). "
                    f'Stop it first with task_stop({{"task_id":"{task_id}"}}) '
                    "before resuming."
                )
            },
        )
        self.assertEqual(final["status"], "stopped")
        self.assertEqual(run_count, 1)

    def test_static_script_error_returns_tool_error_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager):
                result = workflow({"script": "return 1"}, task_id="tool-session")

        self.assertEqual(
            json.loads(result),
            {
                "error": (
                    "Invalid workflow script: `meta = {...}` must be the FIRST "
                    "statement in the script"
                )
            },
        )
        self.assertEqual(manager.list(limit=10), [])

if __name__ == "__main__":
    unittest.main()
