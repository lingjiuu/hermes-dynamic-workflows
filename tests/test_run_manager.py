from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.manager import WorkflowRunManager
from hermes_dynamic_workflows.engine.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner
from hermes_dynamic_workflows.storage.store import WorkflowStore


class CountingRunner(ChildAgentRunner):
    calls = 0

    def run(self, request: ChildAgentRequest):
        type(self).calls += 1
        return f"{type(self).calls}:{request.label}"


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
                "tool_calls": 5,
            },
        )


class RunManagerTests(unittest.TestCase):
    def setUp(self):
        CountingRunner.calls = 0

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
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig())
            with patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=CountingRunner()):
                record = manager.start_from_params({"scriptPath": str(script_path)}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "1:path-agent")
        self.assertEqual(final["source"]["type"], "scriptPath")

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
            manager = WorkflowRunManager(store=WorkflowStore(Path(tmp)), config=PluginConfig())
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
            manager = WorkflowRunManager(store=WorkflowStore(root / "store"), config=PluginConfig())
            with patch("hermes_dynamic_workflows.agents.runner.HermesChildAgentRunner", return_value=MetadataRunner()):
                record = manager.start_from_params({"script": script}, cwd=str(root))
                final = manager.wait(record["runId"], timeout=2)
                detail = manager.format_agent(final["runId"], "1")
                saved = manager.save_markdown(final["runId"])

        self.assertIn("meta-agent", detail)
        self.assertIn("test-model", detail)
        self.assertIn("1.2K tok", detail)
        self.assertIn("Saved workflow", saved)


if __name__ == "__main__":
    unittest.main()
