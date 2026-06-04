from __future__ import annotations

import json
import unittest

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow
from hermes_dynamic_workflows.engine.structured_tool import (
    clear_expectation,
    pop_result,
    register_expectation,
    submit_structured_output_handler,
)
from hermes_dynamic_workflows.engine.types import (
    ChildAgentRequest,
    ChildAgentResult,
    ChildAgentRunner,
)

_SCHEMA = {"type": "object", "required": ["ok"]}


class CaptureRunner(ChildAgentRunner):
    """Simulates a child that submitted a schema-valid result via the tool."""

    def __init__(self, value):
        self.requests: list[ChildAgentRequest] = []
        self.value = value

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        return ChildAgentResult(
            content="done",
            metadata={
                "structured_captured": True,
                "structured_result": self.value,
                "structured_attempts": 1,
            },
        )


class BrokerTests(unittest.TestCase):
    def test_valid_submit_is_recorded(self):
        register_expectation("t1", _SCHEMA)
        try:
            out = json.loads(
                submit_structured_output_handler({"result": {"ok": True}}, task_id="t1")
            )
            self.assertEqual(out["status"], "accepted")
            captured, value, attempts = pop_result("t1")
            self.assertTrue(captured)
            self.assertEqual(value, {"ok": True})
            self.assertEqual(attempts, 1)
        finally:
            clear_expectation("t1")

    def test_invalid_submit_is_rejected_with_error(self):
        register_expectation("t2", _SCHEMA)
        try:
            out = json.loads(
                submit_structured_output_handler({"result": {"nope": 1}}, task_id="t2")
            )
            self.assertEqual(out["status"], "rejected")
            self.assertIn("error", out)
            captured, _value, attempts = pop_result("t2")
            self.assertFalse(captured)
            self.assertEqual(attempts, 1)
        finally:
            clear_expectation("t2")

    def test_retry_then_accept_counts_attempts(self):
        register_expectation("t3", _SCHEMA)
        try:
            first = json.loads(
                submit_structured_output_handler({"result": {}}, task_id="t3")
            )
            second = json.loads(
                submit_structured_output_handler({"result": {"ok": 1}}, task_id="t3")
            )
            self.assertEqual(first["status"], "rejected")
            self.assertEqual(second["status"], "accepted")
            captured, value, attempts = pop_result("t3")
            self.assertTrue(captured)
            self.assertEqual(value, {"ok": 1})
            self.assertEqual(attempts, 2)
        finally:
            clear_expectation("t3")

    def test_submit_without_expectation_is_rejected(self):
        out = json.loads(submit_structured_output_handler({"result": 1}, task_id="missing"))
        self.assertEqual(out["status"], "rejected")

    def test_missing_result_argument_is_rejected(self):
        out = json.loads(submit_structured_output_handler({}, task_id="t4"))
        self.assertEqual(out["status"], "rejected")


class ToolChannelTests(unittest.TestCase):
    def test_default_mode_uses_tool_channel_and_captured_result(self):
        script = """
meta = {"name": "tool-channel"}

def workflow():
    return agent("return status", {"label": "json", "schema": {"type": "object", "required": ["ok"]}})
"""
        runner = CaptureRunner({"ok": True, "n": 5})
        result = run_workflow(
            script,
            WorkflowOptions(config=PluginConfig(), child_runner=runner),
        )

        self.assertEqual(result.value, {"ok": True, "n": 5})
        request = runner.requests[0]
        self.assertTrue(request.structured_tool)
        self.assertIn("workflow_submit_structured_output", request.prompt)
        agent = result.state.snapshot()["agents"][0]
        self.assertEqual(agent["structured"]["mode"], "tool")
        self.assertEqual(agent["structured"]["status"], "valid")


if __name__ == "__main__":
    unittest.main()
