from __future__ import annotations

import unittest

from hermes_dynamic_workflows.adapters.hooks import (
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

    def test_ask_defers_when_live_gateway_channel(self):
        # A live gateway channel exists -> defer (None) to Hermes' gateway
        # approve/deny buttons.
        self.assertIsNone(
            evaluate_command_gate(
                "rm -rf /tmp/x", classify=_dangerous, allowlist=set(),
                policy="ask", smart_approve=_deny, has_gateway_channel=True,
            )
        )

    def test_ask_degrades_to_smart_without_channel(self):
        # No reachable human (the common detached-child case): ask degrades to
        # ask_fallback. With smart approving, the command is allowed.
        self.assertIsNone(
            evaluate_command_gate(
                "rm -rf /tmp/x", classify=_dangerous, allowlist=set(),
                policy="ask", smart_approve=_approve,
                has_gateway_channel=False, ask_fallback="smart",
            )
        )

    def test_ask_degrades_to_smart_and_blocks_when_smart_denies(self):
        result = evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(),
            policy="ask", smart_approve=_deny,
            has_gateway_channel=False, ask_fallback="smart",
        )
        self.assertEqual(result["action"], "block")

    def test_ask_degrades_to_deny_when_configured(self):
        result = evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(),
            policy="ask", smart_approve=_approve,
            has_gateway_channel=False, ask_fallback="deny",
        )
        self.assertEqual(result["action"], "block")

    def test_on_allow_fires_with_pattern_key_when_allowed(self):
        seen = []
        result = evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(),
            policy="approve", smart_approve=_deny,
            on_allow=lambda key: seen.append(key),
        )
        self.assertIsNone(result)
        self.assertEqual(seen, ["delete in root path"])

    def test_on_allow_does_not_fire_when_blocked(self):
        seen = []
        evaluate_command_gate(
            "rm -rf /tmp/x", classify=_dangerous, allowlist=set(),
            policy="deny", smart_approve=_deny,
            on_allow=lambda key: seen.append(key),
        )
        self.assertEqual(seen, [])

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


class InheritResolutionTests(unittest.TestCase):
    def _resolve(self, mode):
        import sys, types
        from unittest.mock import patch

        appr = types.ModuleType("tools.approval")
        appr._get_approval_mode = lambda: mode
        pkg = types.ModuleType("tools")
        pkg.approval = appr
        from hermes_dynamic_workflows.adapters.hooks import _resolve_policy

        class _Cfg:
            child_approval_policy = "inherit"

        with patch.dict(sys.modules, {"tools": pkg, "tools.approval": appr}):
            return _resolve_policy(_Cfg())

    def test_inherit_maps_manual_to_ask(self):
        self.assertEqual(self._resolve("manual"), "ask")

    def test_inherit_maps_smart_to_smart(self):
        self.assertEqual(self._resolve("smart"), "smart")

    def test_inherit_maps_off_to_approve(self):
        self.assertEqual(self._resolve("off"), "approve")

    def test_non_inherit_passes_through(self):
        from hermes_dynamic_workflows.adapters.hooks import _resolve_policy

        class _Cfg:
            child_approval_policy = "deny"

        self.assertEqual(_resolve_policy(_Cfg()), "deny")


if __name__ == "__main__":
    unittest.main()
