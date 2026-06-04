from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.manager import WorkflowRunManager
from hermes_dynamic_workflows.engine.types import ChildAgentRequest, ChildAgentRunner
from hermes_dynamic_workflows.plugin.tool import workflow
from hermes_dynamic_workflows.storage.store import WorkflowStore


class FakeRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        return f"done:{request.label}"


class ToolTests(unittest.TestCase):
    def test_tool_returns_json_payload(self):
        script = """
meta = {"name": "tool-test"}

def workflow():
    return agent("do it", {"label": "worker"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowRunManager(
                store=WorkflowStore(Path(tmp)),
                config=PluginConfig(),
            )
            with (
                patch("hermes_dynamic_workflows.plugin.tool.get_run_manager", return_value=manager),
                patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=FakeRunner()),
            ):
                payload = json.loads(workflow({"script": script, "args": ["x"]}))
                final = manager.wait(payload["runId"], timeout=2)

        self.assertIn(payload["status"], {"queued", "running", "completed"})
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "done:worker")
        self.assertEqual(final["workflow"]["meta"]["name"], "tool-test")


if __name__ == "__main__":
    unittest.main()
