from __future__ import annotations

import unittest

from hermes_dynamic_workflows.engine.approval_hook import (
    evaluate_command_gate,
    pre_tool_call_handler,
)


def _dangerous(_cmd):
    return (True, "delete in root path", "delete in root path")


def _safe(_cmd):
    return (False, "", "")


def _deny(_cmd, _desc):
    return "deny"


def _approve(_cmd, _desc):
    return "approve"


class CommandGateTests(unittest.TestCase):
    def test_deny_policy_blocks_dangerous(self):
        result = evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(), policy="deny", smart_approve=_deny
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["action"], "block")

    def test_safe_command_allowed(self):
        self.assertIsNone(
            evaluate_command_gate("ls", classify=_safe, allowlist=set(), policy="deny", smart_approve=_deny)
        )

    def test_allowlisted_pattern_allowed_even_under_deny(self):
        self.assertIsNone(
            evaluate_command_gate(
                "rm -rf /tmp/x",
                classify=_dangerous,
                allowlist={"delete in root path"},
                policy="deny",
                smart_approve=_deny,
            )
        )

    def test_approve_policy_allows(self):
        self.assertIsNone(
            evaluate_command_gate("rm -rf /tmp/x", classify=_dangerous, allowlist=set(), policy="approve", smart_approve=_deny)
        )

    def test_smart_approve_allows(self):
        self.assertIsNone(
            evaluate_command_gate("pytest", classify=_dangerous, allowlist=set(), policy="smart", smart_approve=_approve)
        )

    def test_smart_deny_blocks(self):
        result = evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(), policy="smart", smart_approve=_deny
        )
        self.assertEqual(result["action"], "block")

    def test_smart_eval_failure_blocks(self):
        def boom(_c, _d):
            raise RuntimeError("llm down")

        result = evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(), policy="smart", smart_approve=boom
        )
        self.assertEqual(result["action"], "block")


class HandlerFastPathTests(unittest.TestCase):
    def test_ignores_non_workflow_task(self):
        # Non-workflow task_id short-circuits before any classification.
        self.assertIsNone(
            pre_tool_call_handler(tool_name="terminal", args={"command": "rm -rf /x"}, task_id="other-123")
        )

    def test_ignores_non_terminal_tool(self):
        self.assertIsNone(
            pre_tool_call_handler(tool_name="web_search", args={"query": "x"}, task_id="workflow-abc123")
        )

    def test_ignores_missing_command(self):
        self.assertIsNone(
            pre_tool_call_handler(tool_name="terminal", args={}, task_id="workflow-abc123")
        )


if __name__ == "__main__":
    unittest.main()
