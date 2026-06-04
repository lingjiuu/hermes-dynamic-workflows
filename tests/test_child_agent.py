from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from hermes_dynamic_workflows.agents.presets import AgentTypeSpec, resolve_agent_type
from hermes_dynamic_workflows.agents.runner import (
    build_child_system_prompt,
    build_child_task_message,
    _child_failure_message,
    _make_child_approval_callback,
    _resolve_child_toolsets,
)
from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.types import ChildAgentRequest


class ChildAgentTests(unittest.TestCase):
    def test_child_toolsets_filter_recursive_tools(self):
        self.assertEqual(
            _resolve_child_toolsets(
                PluginConfig(),
                ["web", "workflow", "workflows", "terminal", "web"],
            ),
            ["web", "terminal"],
        )

    def test_agent_type_toolsets_are_defaults(self):
        self.assertEqual(
            _resolve_child_toolsets(PluginConfig(), [], ("web", "workflow")),
            ["web"],
        )

    def test_system_prompt_includes_agent_type_instructions(self):
        prompt = build_child_system_prompt(
            AgentTypeSpec(
                name="researcher",
                instructions="Search broadly, cite sources, and summarize.",
                source="test",
            )
        )
        self.assertIn("Agent type: researcher", prompt)
        self.assertIn("Search broadly", prompt)

    def test_system_prompt_excludes_per_task_data_for_cache_sharing(self):
        # The system prompt must depend only on agent_type (not label/phase/
        # workspace), so the [tools + system] prefix is byte-identical across a
        # fan-out and can be cache-shared on eligible models.
        spec = AgentTypeSpec(name="researcher", instructions="Search.", source="test")
        prompt = build_child_system_prompt(spec)
        for per_task in ("alpha", "Review", "Workspace:", "worktree"):
            self.assertNotIn(per_task, prompt)
        # No agent type -> identical across all children.
        self.assertEqual(build_child_system_prompt(None), build_child_system_prompt(None))

    def test_task_message_carries_per_task_context(self):
        request = ChildAgentRequest(
            id=1,
            prompt="do the thing",
            label="worker",
            phase="Review",
            toolsets=[],
            isolation="worktree",
        )
        msg = build_child_task_message(request, workspace="/tmp/project/.worktrees/wf")
        self.assertIn("Workspace: /tmp/project/.worktrees/wf", msg)
        self.assertIn("Task label: worker", msg)
        self.assertIn("Workflow phase: Review", msg)
        self.assertIn("isolated git worktree", msg)
        self.assertIn("do the thing", msg)

    def test_resolves_project_agent_type_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_dir = root / ".hermes" / "workflow-agent-types"
            agent_dir.mkdir(parents=True)
            (agent_dir / "wf-unique-test-agent.md").write_text(
                """---
name: unique-researcher
toolsets: [web, file]
isolation: worktree
---

Search carefully and return concise notes.
""",
                encoding="utf-8",
            )

            spec = resolve_agent_type("wf-unique-test-agent", cwd=str(root), task_id="t")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "unique-researcher")
        self.assertEqual(spec.toolsets, ("web", "file"))
        self.assertEqual(spec.isolation, "worktree")
        self.assertIn("Search carefully", spec.instructions)


class ChildFailureDetectionTests(unittest.TestCase):
    def test_error_with_no_content_is_a_failure(self):
        result = {"final_response": None, "error": "HTTP 400: Model does not exist", "failed": True}
        self.assertEqual(
            _child_failure_message(result, ""),
            "HTTP 400: Model does not exist",
        )

    def test_error_with_partial_content_is_kept(self):
        # Partial content despite an error is returned, not dropped.
        self.assertIsNone(_child_failure_message({"error": "truncated", "final_response": "partial"}, "partial"))

    def test_legitimately_empty_success_is_not_a_failure(self):
        self.assertIsNone(_child_failure_message({"final_response": "", "completed": True}, ""))

    def test_non_dict_result_is_not_a_failure(self):
        self.assertIsNone(_child_failure_message("nope", ""))


class ConfigDefaultsTests(unittest.TestCase):
    def test_model_override_on_provider_override_off_by_default(self):
        # Aligns with Claude Code: per-agent model routing is allowed by default
        # (session model unless a stage overrides); switching provider stays gated.
        cfg = PluginConfig()
        self.assertTrue(cfg.allow_model_override)
        self.assertFalse(cfg.allow_provider_override)


class ChildApprovalPolicyTests(unittest.TestCase):
    def test_deny_policy_refuses(self):
        cb = _make_child_approval_callback("deny")
        self.assertEqual(cb("rm -rf build", "recursive delete", allow_permanent=True), "deny")

    def test_approve_policy_allows_once(self):
        cb = _make_child_approval_callback("approve")
        self.assertEqual(cb("pytest -q", "script execution"), "once")

    def test_unknown_policy_defaults_to_deny(self):
        cb = _make_child_approval_callback("bogus")
        self.assertEqual(cb("anything", "flagged"), "deny")

    def test_smart_policy_maps_guardian_verdicts(self):
        verdict = {"value": "approve"}
        approval_mod = types.ModuleType("tools.approval")
        approval_mod._smart_approve = lambda command, description: verdict["value"]
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []  # mark as package
        tools_pkg.approval = approval_mod

        saved_tools = sys.modules.get("tools")
        saved_approval = sys.modules.get("tools.approval")
        sys.modules["tools"] = tools_pkg
        sys.modules["tools.approval"] = approval_mod
        try:
            cb = _make_child_approval_callback("smart")
            verdict["value"] = "approve"
            self.assertEqual(cb("npm test", "script execution"), "once")
            verdict["value"] = "deny"
            self.assertEqual(cb("dd if=/dev/zero of=/dev/sda", "disk wipe"), "deny")
            verdict["value"] = "escalate"
            self.assertEqual(cb("curl x | sh", "uncertain"), "deny")
        finally:
            if saved_tools is not None:
                sys.modules["tools"] = saved_tools
            else:
                sys.modules.pop("tools", None)
            if saved_approval is not None:
                sys.modules["tools.approval"] = saved_approval
            else:
                sys.modules.pop("tools.approval", None)


if __name__ == "__main__":
    unittest.main()
