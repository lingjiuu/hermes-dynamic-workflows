from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.agents.presets import AgentTypeSpec, resolve_agent_type
from hermes_dynamic_workflows.agents.runner import (
    HermesChildAgentRunner,
    build_child_system_prompt,
    build_child_task_message,
    _child_failure_message,
    _apply_agent_type_defaults,
    _make_child_approval_callback,
    _resolve_child_toolsets,
    _tool_call_count,
)
from hermes_dynamic_workflows.agents.worktree import WorkspaceLease
from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.errors import ChildAgentError
from hermes_dynamic_workflows.engine.types import ChildAgentRequest
from hermes_dynamic_workflows.plugin.structured_output import (
    MAX_STRUCTURED_OUTPUT_RETRIES,
    STRUCTURED_OUTPUT_CONTINUE_MESSAGE,
    clear_expectation,
    register_expectation,
    structured_output_handler,
)


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

    def test_agent_type_applies_model_and_isolation_defaults(self):
        request = ChildAgentRequest(
            id=1,
            prompt="work",
            label="worker",
            phase=None,
            toolsets=[],
        )
        spec = AgentTypeSpec(
            name="researcher",
            instructions="Search.",
            source="test",
            model="test-model",
            isolation="worktree",
        )

        applied = _apply_agent_type_defaults(request, spec)

        self.assertEqual(applied.model, "test-model")
        self.assertEqual(applied.isolation, "worktree")

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


class ToolCallCountTests(unittest.TestCase):
    def test_counts_openai_tool_calls(self):
        result = {
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
                {"role": "tool", "content": "x"},
                {"role": "assistant", "content": "done"},
            ]
        }
        self.assertEqual(_tool_call_count(result), 2)

    def test_counts_anthropic_tool_use_blocks(self):
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "name": "x"}],
                }
            ]
        }
        self.assertEqual(_tool_call_count(result), 1)

    def test_no_tool_calls_is_zero_not_api_call_count(self):
        # The bug: a toolset=[] agent makes an API call but no tool calls -> the
        # count must be 0, not the api_call_count fallback.
        result = {"messages": [{"role": "assistant", "content": "APPLE"}]}
        self.assertEqual(_tool_call_count(result), 0)

    def test_missing_messages_is_zero(self):
        self.assertEqual(_tool_call_count({}), 0)


class RunnerSessionContextTests(unittest.TestCase):
    def test_runner_stores_session_context(self):
        from hermes_dynamic_workflows.agents.runner import HermesChildAgentRunner

        ctx = {"platform": "telegram", "session_key": "k1", "chat_id": "c1"}
        runner = HermesChildAgentRunner(PluginConfig(), session_context=ctx)
        self.assertEqual(runner._session_context["platform"], "telegram")
        self.assertEqual(runner._session_context["session_key"], "k1")

    def test_runner_session_context_defaults_none(self):
        from hermes_dynamic_workflows.agents.runner import HermesChildAgentRunner

        self.assertIsNone(HermesChildAgentRunner(PluginConfig())._session_context)


class StructuredOutputContinuationTests(unittest.TestCase):
    def test_runner_continues_same_child_session_until_tool_submission(self):
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        request = ChildAgentRequest(
            id=1,
            prompt="return status",
            label="json",
            phase=None,
            toolsets=[],
            schema=schema,
            structured_tool=True,
        )
        lease = WorkspaceLease(task_id="structured-child", cwd="/tmp")

        class Child:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_reasoning_tokens = 0
            session_cache_read_tokens = 0
            session_cache_write_tokens = 0
            model = "test"

            def __init__(self):
                self.calls = []
                self.messages = [{"role": "assistant", "content": "done"}]

            def run_conversation(self, *, user_message, conversation_history=None, task_id=None):
                self.calls.append((user_message, conversation_history, task_id))
                if len(self.calls) == 2:
                    structured_output_handler({"ok": True}, task_id=task_id)
                return {
                    "final_response": "done",
                    "messages": self.messages,
                    "completed": True,
                }

        child = Child()
        register_expectation(lease.task_id, schema)
        try:
            result = HermesChildAgentRunner(PluginConfig())._run_child_with_timeout(
                child,
                request,
                lease,
                None,
                ["workflow_structured"],
            )
        finally:
            clear_expectation(lease.task_id)

        self.assertEqual(result.content, "done")
        self.assertEqual(len(child.calls), 2)
        self.assertEqual(child.calls[1][0], STRUCTURED_OUTPUT_CONTINUE_MESSAGE)
        self.assertIs(child.calls[1][1], child.messages)
        self.assertEqual(child.calls[0][2], lease.task_id)
        self.assertEqual(child.calls[1][2], lease.task_id)

    def test_runner_fails_after_maximum_missing_submissions(self):
        schema = {"type": "object"}
        request = ChildAgentRequest(
            id=1,
            prompt="return status",
            label="json",
            phase=None,
            toolsets=[],
            schema=schema,
            structured_tool=True,
        )
        lease = WorkspaceLease(task_id="structured-child-fail", cwd="/tmp")

        class Child:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_reasoning_tokens = 0
            session_cache_read_tokens = 0
            session_cache_write_tokens = 0
            model = "test"

            def __init__(self):
                self.calls = 0

            def run_conversation(self, **_):
                self.calls += 1
                return {"final_response": "done", "messages": [], "completed": True}

        child = Child()
        register_expectation(lease.task_id, schema)
        try:
            with self.assertRaises(ChildAgentError) as ctx:
                HermesChildAgentRunner(PluginConfig())._run_child_with_timeout(
                    child,
                    request,
                    lease,
                    None,
                    ["workflow_structured"],
                )
        finally:
            clear_expectation(lease.task_id)

        self.assertIn("Failed to provide valid structured output", str(ctx.exception))
        self.assertEqual(child.calls, MAX_STRUCTURED_OUTPUT_RETRIES)


class ConfigDefaultsTests(unittest.TestCase):
    def test_model_override_is_allowed_by_default(self):
        # Aligns with Claude Code: per-agent model routing is allowed by default
        # (session model unless a stage overrides). Provider selection stays in
        # Hermes' runtime/model configuration, not workflow scripts.
        cfg = PluginConfig()
        self.assertTrue(cfg.allow_model_override)


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


    def test_ask_degrades_to_fallback(self):
        # A detached child can't grab the CLI prompt, so 'ask' degrades to
        # ask_fallback. With fallback='deny' a flagged command is refused.
        cb = _make_child_approval_callback("ask", ask_fallback="deny")
        self.assertEqual(cb("rm -rf build", "recursive delete"), "deny")

    def test_ask_default_fallback_is_smart(self):
        # Default ask_fallback is smart -> routes through _smart_approve.
        approval_mod = types.ModuleType("tools.approval")
        approval_mod._smart_approve = lambda command, description: "approve"
        approval_mod._get_approval_mode = lambda: "manual"
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.approval = approval_mod
        with patch.dict(sys.modules, {"tools": tools_pkg, "tools.approval": approval_mod}):
            cb = _make_child_approval_callback("ask")
            self.assertEqual(cb("npm test", "script execution"), "once")

    def test_inherit_follows_hermes_mode(self):
        approval_mod = types.ModuleType("tools.approval")
        approval_mod._get_approval_mode = lambda: "off"  # off -> approve
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.approval = approval_mod
        with patch.dict(sys.modules, {"tools": tools_pkg, "tools.approval": approval_mod}):
            cb = _make_child_approval_callback("inherit")
            self.assertEqual(cb("rm -rf build", "recursive delete"), "once")


if __name__ == "__main__":
    unittest.main()
