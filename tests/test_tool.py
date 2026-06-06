from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.manager import WorkflowRunManager
from hermes_dynamic_workflows.engine.types import ChildAgentRequest, ChildAgentRunner
from hermes_dynamic_workflows.plugin.task_stop import task_stop
from hermes_dynamic_workflows.plugin.workflow import DYNAMIC_WORKFLOW_SCHEMA, workflow
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
    def test_tool_parses_budget_only_from_user_task(self):
        script = """
meta = {"name": "budget-source"}

def workflow():
    agent("do it", {"label": "worker"})
    return budget.total
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=FakeRunner()),
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
meta = {"name": "tool-test"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=FakeRunner()),
            ):
                result = workflow({"script": script, "args": ["x"]}, task_id="tool-session")
                match = re.search(r"^Run ID: (wf_[a-z0-9]{8}-[a-z0-9]{3})$", result, re.MULTILINE)
                self.assertIsNotNone(match)
                run_id = match.group(1)
                final = manager.wait(run_id, timeout=2)

        self.assertRegex(result, r"Workflow launched in background\. Task ID: wg[a-z0-9]{7}")
        self.assertIn("Summary: tool-test", result)
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

    def test_task_stop_stops_active_workflow_by_task_id(self):
        script = """
meta = {"name": "stop-test", "description": "Stop me"}

def workflow():
    return agent("wait", {"label": "worker"})
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
                patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=runner),
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

def workflow():
    return agent("wait", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = BlockingRunner()
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(require_launch_approval=False),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.workflow.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=runner),
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

if __name__ == "__main__":
    unittest.main()
