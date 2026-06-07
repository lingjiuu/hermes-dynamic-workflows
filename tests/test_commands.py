from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_dynamic_workflows.adapters.commands import workflows_command
from hermes_dynamic_workflows.run.manager import WorkflowRunManager
from hermes_dynamic_workflows.storage.store import WorkflowStore


class CommandTests(unittest.TestCase):
    def test_workflows_command_shows_only_current_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            _save_run(store, "wf_current1-aaa", "sess-current", "current-workflow")
            _save_run(store, "wf_other22-bbb", "sess-other", "other-workflow")
            manager = WorkflowRunManager(store=store)

            with patch("hermes_dynamic_workflows.adapters.commands.get_run_manager", return_value=manager):
                output = workflows_command(plugin_context=SimpleNamespace(session_id="sess-current"))

        self.assertIn("current-workflow", output)
        self.assertNotIn("other-workflow", output)

    def test_workflows_command_uses_cli_ref_agent_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            _save_run(store, "wf_agent11-aaa", "agent-session", "agent-session-workflow")
            _save_run(store, "wf_cli222-bbb", "cli-session", "cli-session-workflow")
            manager = WorkflowRunManager(store=store)
            agent = SimpleNamespace(session_id="agent-session")
            cli_ref = SimpleNamespace(session_id="cli-session", agent=agent)
            ctx = SimpleNamespace(_manager=SimpleNamespace(_cli_ref=cli_ref))

            with patch("hermes_dynamic_workflows.adapters.commands.get_run_manager", return_value=manager):
                output = workflows_command(plugin_context=ctx)

        self.assertIn("agent-session-workflow", output)
        self.assertNotIn("cli-session-workflow", output)


def _save_run(store: WorkflowStore, run_id: str, session_id: str, name: str) -> None:
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    store.save_run(
        {
            "runId": run_id,
            "taskId": f"task-{run_id}",
            "workflowSessionId": session_id,
            "status": "completed",
            "createdAt": started.isoformat(),
            "startedAt": started.isoformat(),
            "cwd": "/tmp/project",
            "workflow": {
                "meta": {"name": name, "description": name},
                "duration_seconds": 10,
                "agents": [],
                "children": [],
                "errors": [],
                "totals": {
                    "agents": 0,
                    "done": 0,
                    "running": 0,
                    "tokens": 0,
                    "tool_calls": 0,
                },
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
