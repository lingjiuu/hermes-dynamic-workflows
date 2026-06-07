from __future__ import annotations

import sys
import os
import io
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.child.presets import AgentTypeSpec, list_agent_types, resolve_agent_type
from hermes_dynamic_workflows.child.runner import (
    HermesChildAgentRunner,
    _WorkflowApprovalCoordinator,
    build_child_system_prompt,
    build_child_task_message,
    _child_failure_message,
    _child_metadata,
    _compact_tool_progress_line,
    _display_width,
    _apply_agent_type_defaults,
    _configure_child_tools,
    _discoverable_child_toolsets,
    _make_child_approval_callback,
    _resolve_child_toolsets,
    _tool_progress_line_width,
    _tool_call_count,
)
from hermes_dynamic_workflows.child.worktree import WorkspaceLease, create_workspace_lease
from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import ChildAgentError
from hermes_dynamic_workflows.core.types import ChildAgentRequest
from hermes_dynamic_workflows.child.structured_output import (
    MAX_STRUCTURED_OUTPUT_RETRIES,
    STRUCTURED_OUTPUT_CONTINUE_MESSAGE,
    clear_expectation,
    register_expectation,
    structured_output_handler,
)


def _tool_definition(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_name(definition: dict) -> str:
    return str((definition.get("function") or {}).get("name") or "")


class ChildAgentTests(unittest.TestCase):
    def test_clean_worktree_is_removed_without_modifying_tracked_gitignore(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            gitignore = repo / ".gitignore"
            gitignore.write_text("existing/\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

            lease = create_workspace_lease(cwd=str(repo), isolation="worktree", label="clean")
            worktree = Path(lease.path or "")
            branch = str(lease.branch or "")
            self.assertTrue(worktree.is_dir())
            self.assertEqual(gitignore.read_text(encoding="utf-8"), "existing/\n")

            lease.cleanup()

            self.assertFalse(worktree.exists())
            branches = subprocess.run(
                ["git", "branch", "--list", branch],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(branches.stdout.strip(), "")

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

    def test_default_child_toolsets_include_read_only_skills_surface(self):
        self.assertEqual(
            _resolve_child_toolsets(PluginConfig(), []),
            ["web", "file", "terminal", "skills"],
        )

    def test_explicit_agent_type_does_not_gain_discoverable_toolsets(self):
        with patch(
            "hermes_dynamic_workflows.child.runner._discoverable_child_toolsets",
            return_value=["mcp-github", "plugin-extra"],
        ):
            self.assertEqual(
                _resolve_child_toolsets(
                    PluginConfig(),
                    [],
                    ("file",),
                    include_discoverable=False,
                ),
                ["file"],
            )

    def test_discoverable_toolsets_exclude_workflow_controls(self):
        class Registry:
            toolsets = {
                "mcp_search": "mcp-search",
                "plugin_extra": "plugin-extra",
                "workflow": "workflow",
                "task_stop": "workflow",
                "structured_output": "workflow_structured",
                "delegate_task": "delegation",
                "built_in_extra": "discord_admin",
                "read_file": "file",
            }

            def get_all_tool_names(self):
                return list(self.toolsets)

            def get_toolset_for_tool(self, name):
                return self.toolsets.get(name)

        registry_mod = types.ModuleType("tools.registry")
        registry_mod.registry = Registry()
        search_mod = types.ModuleType("tools.tool_search")
        search_mod.is_deferrable_tool_name = lambda name: name != "read_file"
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.registry = registry_mod
        tools_pkg.tool_search = search_mod
        plugins_mod = types.ModuleType("hermes_cli.plugins")
        plugins_mod.get_plugin_manager = lambda: types.SimpleNamespace(
            _plugin_tool_names={"plugin_extra", "workflow", "task_stop"}
        )
        hermes_cli_pkg = types.ModuleType("hermes_cli")
        hermes_cli_pkg.__path__ = []
        hermes_cli_pkg.plugins = plugins_mod

        with patch.dict(
            sys.modules,
            {
                "hermes_cli": hermes_cli_pkg,
                "hermes_cli.plugins": plugins_mod,
                "tools": tools_pkg,
                "tools.registry": registry_mod,
                "tools.tool_search": search_mod,
            },
        ):
            discovered = _discoverable_child_toolsets(PluginConfig())

        self.assertEqual(discovered, ["mcp-search", "plugin-extra"])

    def test_child_tool_surface_forces_tool_search_and_keeps_skills_read_only(self):
        definitions = [
            _tool_definition("read_file"),
            _tool_definition("skills_list"),
            _tool_definition("skill_view"),
            _tool_definition("skill_manage"),
            _tool_definition("mcp_search"),
            _tool_definition("structured_output"),
        ]
        captured = {}

        model_tools = types.ModuleType("model_tools")

        def get_tool_definitions(**kwargs):
            captured["get"] = kwargs
            return definitions

        model_tools.get_tool_definitions = get_tool_definitions
        search_mod = types.ModuleType("tools.tool_search")

        class ToolSearchConfig:
            @staticmethod
            def from_raw(raw):
                captured["config"] = raw
                return raw

        def assemble_tool_defs(tool_defs, *, config):
            captured["assembled"] = [_tool_name(item) for item in tool_defs]
            return types.SimpleNamespace(
                tool_defs=[
                    _tool_definition("read_file"),
                    _tool_definition("skills_list"),
                    _tool_definition("skill_view"),
                    _tool_definition("tool_search"),
                    _tool_definition("tool_describe"),
                    _tool_definition("tool_call"),
                ]
            )

        search_mod.ToolSearchConfig = ToolSearchConfig
        search_mod.assemble_tool_defs = assemble_tool_defs
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.tool_search = search_mod

        class Child:
            tools = []
            valid_tool_names = set()
            enabled_toolsets = []

        child = Child()
        with patch.dict(
            sys.modules,
            {
                "model_tools": model_tools,
                "tools": tools_pkg,
                "tools.tool_search": search_mod,
            },
        ):
            _configure_child_tools(
                child,
                toolsets=["file", "skills", "mcp-search", "workflow_structured"],
                blocked_toolsets=PluginConfig().blocked_child_toolsets,
            )

        self.assertNotIn("skill_manage", captured["assembled"])
        self.assertNotIn("structured_output", captured["assembled"])
        self.assertEqual(captured["config"], {"enabled": "on"})
        self.assertEqual(captured["get"]["enabled_toolsets"], [
            "file",
            "skills",
            "mcp-search",
            "workflow_structured",
        ])
        self.assertEqual(
            child.enabled_toolsets,
            ["file", "skills", "mcp-search"],
        )
        self.assertIn("structured_output", child.valid_tool_names)
        self.assertIn("tool_search", child.valid_tool_names)
        self.assertNotIn("skill_manage", child.valid_tool_names)

    def test_child_tool_surface_applies_agent_type_allowed_tools(self):
        definitions = [
            _tool_definition("read_file"),
            _tool_definition("write_file"),
            _tool_definition("patch"),
            _tool_definition("search_files"),
            _tool_definition("terminal"),
            _tool_definition("process"),
        ]

        model_tools = types.ModuleType("model_tools")
        model_tools.get_tool_definitions = lambda **kwargs: definitions
        search_mod = types.ModuleType("tools.tool_search")

        class ToolSearchConfig:
            @staticmethod
            def from_raw(raw):
                return raw

        def assemble_tool_defs(tool_defs, *, config):
            return types.SimpleNamespace(tool_defs=tool_defs)

        search_mod.ToolSearchConfig = ToolSearchConfig
        search_mod.assemble_tool_defs = assemble_tool_defs
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.tool_search = search_mod

        class Child:
            tools = []
            valid_tool_names = set()
            enabled_toolsets = []

        child = Child()
        with patch.dict(
            sys.modules,
            {
                "model_tools": model_tools,
                "tools": tools_pkg,
                "tools.tool_search": search_mod,
            },
        ):
            _configure_child_tools(
                child,
                toolsets=["file", "terminal"],
                blocked_toolsets=PluginConfig().blocked_child_toolsets,
                allowed_tools=("read_file", "search_files", "terminal", "process"),
            )

        self.assertEqual(
            child.valid_tool_names,
            {"read_file", "search_files", "terminal", "process"},
        )
        self.assertNotIn("write_file", child.valid_tool_names)
        self.assertNotIn("patch", child.valid_tool_names)

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

    def test_system_prompt_defines_final_text_as_raw_return_value(self):
        prompt = build_child_system_prompt(None)
        self.assertIn("subagent spawned by a workflow orchestration script", prompt)
        self.assertIn("Use the tools available to complete the task", prompt)
        self.assertIn("returned verbatim as a string to the calling script", prompt)
        self.assertIn("not a message to a human", prompt)

    def test_structured_output_instruction_is_appended_to_system_prompt(self):
        plain = build_child_system_prompt(None)
        structured = build_child_system_prompt(None, structured_output=True)

        self.assertNotIn("structured_output tool", plain)
        self.assertIn("structured_output tool", structured)
        self.assertTrue(structured.startswith(plain))

    def test_internal_skip_interrupts_only_requested_child(self):
        class Child:
            def __init__(self):
                self.interrupted = False

            def interrupt(self):
                self.interrupted = True

        runner = HermesChildAgentRunner(PluginConfig())
        first = Child()
        second = Child()
        runner._active_children = {"first": first, "second": second}

        self.assertTrue(runner.skip_child("first"))
        self.assertTrue(first.interrupted)
        self.assertFalse(second.interrupted)
        self.assertTrue(runner._consume_skipped("first"))
        self.assertFalse(runner._consume_skipped("first"))
        self.assertFalse(runner.skip_child("missing"))

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

    def test_agent_type_inherit_model_means_no_override(self):
        request = ChildAgentRequest(
            id=1,
            prompt="work",
            label="worker",
            phase=None,
            toolsets=[],
        )
        spec = AgentTypeSpec(
            name="planner",
            instructions="Plan.",
            source="test",
            model="inherit",
        )

        applied = _apply_agent_type_defaults(request, spec)

        self.assertIsNone(applied.model)

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

    def test_task_message_excludes_display_only_label_and_phase(self):
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
        self.assertNotIn("Task label: worker", msg)
        self.assertNotIn("Workflow phase: Review", msg)
        self.assertIn("isolated git worktree", msg)
        self.assertIn("do the thing", msg)

    def test_default_child_metadata_uses_workflow_subagent_type(self):
        class Child:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_reasoning_tokens = 0
            session_cache_read_tokens = 0
            session_cache_write_tokens = 0
            model = "test"
            messages = []

        metadata = _child_metadata(
            Child(),
            {},
            WorkspaceLease(task_id="child", cwd="/tmp"),
            None,
            ["file"],
        )

        self.assertEqual(metadata["agent_type"], "workflow-subagent")
        self.assertIsNone(metadata["agent_type_source"])

    def test_bundled_agent_types_inherit_launching_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {"HERMES_DYNAMIC_WORKFLOWS_HOME": str(Path(tmp) / "store")},
            ):
                for name in ("Explore", "Plan", "general-purpose", "verification"):
                    spec = resolve_agent_type(name, cwd=tmp)
                    self.assertIsNotNone(spec)
                    assert spec is not None
                    self.assertEqual(spec.model, "inherit")
                    if name in {"Explore", "Plan"}:
                        self.assertEqual(
                            spec.allowed_tools,
                            ("read_file", "search_files", "terminal", "process"),
                        )
                    if name == "verification":
                        self.assertNotIn("write_file", spec.allowed_tools)
                        self.assertNotIn("patch", spec.allowed_tools)

    def test_resolves_project_agent_type_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_dir = root / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "wf-unique-test-agent.md").write_text(
                """---
name: unique-researcher
description: Search code and summarize findings.
toolsets: [web, file]
allowed_tools: [read_file, search_files]
disallowed_tools: [write_file, patch]
isolation: worktree
---

Search carefully and return concise notes.
""",
                encoding="utf-8",
            )

            spec = resolve_agent_type("wf-unique-test-agent", cwd=str(root))
            named_spec = resolve_agent_type("unique-researcher", cwd=str(root))
            listed = list_agent_types(cwd=str(root))

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "unique-researcher")
        self.assertEqual(spec.description, "Search code and summarize findings.")
        self.assertEqual(spec.toolsets, ("web", "file"))
        self.assertEqual(spec.allowed_tools, ("read_file", "search_files"))
        self.assertEqual(spec.disallowed_tools, ("write_file", "patch"))
        self.assertEqual(spec.isolation, "worktree")
        self.assertIn("Search carefully", spec.instructions)
        self.assertIsNotNone(named_spec)
        assert named_spec is not None
        self.assertEqual(named_spec.name, "unique-researcher")
        self.assertIn("unique-researcher", [item.name for item in listed])

    def test_resolves_user_agent_type_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            agent_dir = store / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "global-reviewer.md").write_text(
                """---
name: global-reviewer
model: test-model
---

Review from the user-wide workflow agent store.
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HERMES_DYNAMIC_WORKFLOWS_HOME": str(store)}):
                spec = resolve_agent_type("global-reviewer", cwd=str(Path(tmp) / "project"))

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "global-reviewer")
        self.assertEqual(spec.model, "test-model")
        self.assertIn("user-wide workflow agent store", spec.instructions)


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

    def test_compact_tool_progress_line_prioritizes_and_truncates_args(self):
        long_command = (
            "curl -sL https://example.test/search?q=anthropic+workflow+approval "
            "| python3 -c \"import sys; print(sys.stdin.read()[:100])\""
        )

        line = _compact_tool_progress_line(
            "search:models",
            "terminal",
            {"unused": "ignored", "command": long_command, "path": "/Users/Apple/code/MyProjects/hermes-dynamic-workflows/hermes_dynamic_workflows/child/runner.py"},
        )

        self.assertTrue(line.startswith("↳ search:models · terminal("))
        self.assertIn('path:"/Users/Apple/code/MyProjects/hermes-dyn...rmes_dynamic_workflows/child/runner.py"', line)
        self.assertIn('command:"curl -sL https://example.test/search?q=...rt sys; print(sys.stdin.read()[:100])\\""', line)
        self.assertNotIn("unused", line)
        self.assertNotIn("\n", line)

    def test_compact_tool_progress_line_can_fit_terminal_width(self):
        line = _compact_tool_progress_line(
            "综合搜索",
            "structured_output",
            {
                "benchmarks": "关键基准测试成绩：" + "很长" * 80,
                "capabilities": "能力：" + "很长" * 80,
                "pricing": "价格：" + "很长" * 80,
            },
            max_width=72,
        )

        self.assertLessEqual(_display_width(line), 72)
        self.assertTrue(line.startswith("↳ 综合搜索 · structured_output("))
        self.assertIn("...", line)
        self.assertNotIn("\n", line)

    def test_compact_tool_progress_line_handles_wide_label_and_emoji(self):
        line = _compact_tool_progress_line(
            "Agent 核心技术突破 🔍",
            "terminal",
            {
                "command": (
                    'curl -sL "https://lite.duckduckgo.com/lite/?q=agent+research" '
                    "| grep result-snippet | head -100"
                )
            },
            max_width=40,
        )

        self.assertLessEqual(_display_width(line), 40)
        self.assertTrue(line.startswith("↳ Agent 核心技术突破 🔍 · terminal("))
        self.assertIn("terminal(c...)", line)
        self.assertNotIn("\n", line)

    def test_compact_tool_progress_line_truncates_label_only_when_prefix_is_too_wide(self):
        line = _compact_tool_progress_line(
            "这是一个非常非常长的工作流子代理名称",
            "structured_output",
            {"name": "result"},
            max_width=40,
        )

        self.assertLessEqual(_display_width(line), 40)
        self.assertTrue(line.startswith("↳ 这是...名称"))
        self.assertIn(" · structured_output(", line)
        self.assertNotIn("\n", line)

    def test_tool_progress_line_width_respects_narrow_terminal(self):
        with patch(
            "hermes_dynamic_workflows.child.runner.shutil.get_terminal_size",
            return_value=os.terminal_size((30, 24)),
        ):
            self.assertEqual(_tool_progress_line_width(), 28)

    def test_child_agent_installs_compact_tool_progress_callback(self):
        seen_kwargs = {}

        class FakeAIAgent:
            def __init__(self, **kwargs):
                seen_kwargs.update(kwargs)
                self.model = kwargs.get("model")

        run_agent_mod = types.ModuleType("run_agent")
        run_agent_mod.AIAgent = FakeAIAgent
        runner = HermesChildAgentRunner(PluginConfig())
        request = ChildAgentRequest(
            id=1,
            prompt="work",
            label="search:models",
            phase=None,
            toolsets=[],
        )
        lease = WorkspaceLease(task_id="workflow-abc123", cwd="/tmp")
        runtime = {"model": "test-model"}

        with patch.dict(sys.modules, {"run_agent": run_agent_mod}):
            child = runner._build_agent(request, runtime, [], lease, None)

        self.assertIsInstance(child, FakeAIAgent)
        self.assertEqual(seen_kwargs["quiet_mode"], True)
        self.assertEqual(seen_kwargs["platform"], "cli")
        self.assertTrue(callable(seen_kwargs["tool_progress_callback"]))
        self.assertTrue(callable(seen_kwargs["thinking_callback"]))
        self.assertIsNone(seen_kwargs["thinking_callback"]("pondering..."))

    def test_tool_progress_callback_prints_one_started_line_on_tty(self):
        class TtyStringIO(io.StringIO):
            def isatty(self):
                return True

        runner = HermesChildAgentRunner(PluginConfig())
        request = ChildAgentRequest(
            id=1,
            prompt="work",
            label="read:source",
            phase=None,
            toolsets=[],
        )
        lease = WorkspaceLease(task_id="workflow-abc123", cwd="/tmp")
        callback = runner._make_tool_progress_callback(request, lease)
        stream = TtyStringIO()

        with patch("sys.stdout", stream):
            callback("tool.started", "read_file", None, {"path": "/tmp/project/src/really_long_file_name.py"})
            callback("tool.completed", "read_file", None, {"path": "/tmp/project/src/really_long_file_name.py"})

        self.assertEqual(
            stream.getvalue(),
            '↳ read:source · read_file(path:"/tmp/project/src/really_long_file_name.py")\n',
        )


class RunnerSessionContextTests(unittest.TestCase):
    def test_runner_stores_session_context(self):
        from hermes_dynamic_workflows.child.runner import HermesChildAgentRunner

        ctx = {"platform": "telegram", "session_key": "k1", "chat_id": "c1"}
        runner = HermesChildAgentRunner(PluginConfig(), session_context=ctx)
        self.assertEqual(runner._session_context["platform"], "telegram")
        self.assertEqual(runner._session_context["session_key"], "k1")

    def test_runner_session_context_defaults_none(self):
        from hermes_dynamic_workflows.child.runner import HermesChildAgentRunner

        self.assertIsNone(HermesChildAgentRunner(PluginConfig())._session_context)

    def test_runner_stores_live_cli_approval_callback(self):
        from hermes_dynamic_workflows.child.runner import HermesChildAgentRunner

        callback = lambda command, description: "once"
        runner = HermesChildAgentRunner(PluginConfig(), approval_callback=callback)
        self.assertIs(runner._approval_callback, callback)

    def test_runner_stores_approval_session_key(self):
        runner = HermesChildAgentRunner(PluginConfig(), approval_session_key="parent-session")
        self.assertEqual(runner._approval_session_key, "parent-session")

    def test_default_and_inherit_models_use_captured_parent_runtime(self):
        runtime = {
            "model": "session-switched-model",
            "provider": "custom:session",
            "base_url": "https://session.example/v1",
            "api_key": "session-secret",
            "api_mode": "chat_completions",
            "request_overrides": {"extra_body": {"routing": "session"}},
        }
        runner = HermesChildAgentRunner(
            PluginConfig(allow_model_override=False),
            parent_runtime=runtime,
        )

        for model in (None, "inherit"):
            request = ChildAgentRequest(
                id=1,
                prompt="work",
                label="worker",
                phase=None,
                toolsets=[],
                model=model,
            )
            self.assertEqual(runner._resolve_runtime(request), runtime)


class StructuredOutputContinuationTests(unittest.TestCase):
    def test_runner_binds_parent_approval_session_key_inside_child_thread(self):
        seen = []
        current = {"value": ""}
        resets = []

        approval_mod = types.ModuleType("tools.approval")

        def set_current_session_key(value):
            previous = current["value"]
            current["value"] = value
            return previous

        def reset_current_session_key(token):
            resets.append(token)
            current["value"] = token

        approval_mod.set_current_session_key = set_current_session_key
        approval_mod.reset_current_session_key = reset_current_session_key
        terminal_mod = types.ModuleType("tools.terminal_tool")
        terminal_mod.set_approval_callback = lambda callback: None
        terminal_mod.register_task_env_overrides = lambda task_id, overrides: None
        terminal_mod.clear_task_env_overrides = lambda task_id: None
        terminal_mod.cleanup_vm = lambda task_id: None
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.approval = approval_mod
        tools_pkg.terminal_tool = terminal_mod

        class Child:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_reasoning_tokens = 0
            session_cache_read_tokens = 0
            session_cache_write_tokens = 0
            model = "test"

            def run_conversation(self, **_):
                seen.append(current["value"])
                return {"final_response": "done", "messages": [], "completed": True}

        request = ChildAgentRequest(
            id=1,
            prompt="work",
            label="worker",
            phase=None,
            toolsets=[],
        )
        lease = WorkspaceLease(task_id="approval-session-child", cwd="/tmp")
        runner = HermesChildAgentRunner(
            PluginConfig(),
            approval_session_key="parent-session",
        )
        with patch.dict(
            sys.modules,
            {
                "tools": tools_pkg,
                "tools.approval": approval_mod,
                "tools.terminal_tool": terminal_mod,
            },
        ):
            result = runner._run_child_with_timeout(child=Child(), request=request, lease=lease, agent_type=None, toolsets=[])

        self.assertEqual(result.content, "done")
        self.assertEqual(seen, ["parent-session"])
        self.assertEqual(resets, [""])

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
    def test_workflow_approval_coordinator_reuses_explicit_session_grant(self):
        calls = []
        events = []

        def interactive(command, description, **kwargs):
            calls.append((command, description, kwargs))
            return "session"

        coordinator = _WorkflowApprovalCoordinator(interactive)
        first = coordinator.callback_for("workflow-1", events.append)
        second = coordinator.callback_for("workflow-2", events.append)

        self.assertEqual(first("curl a | python3", "pipe to interpreter"), "session")
        self.assertEqual(second("curl b | python3", "pipe to interpreter"), "once")
        self.assertEqual(len(calls), 1)
        self.assertIn("reused", [event["status"] for event in events])

    def test_workflow_approval_coordinator_does_not_reuse_allow_once(self):
        calls = []

        def interactive(command, description, **kwargs):
            calls.append(command)
            return "once"

        coordinator = _WorkflowApprovalCoordinator(interactive)
        first = coordinator.callback_for("workflow-1")
        second = coordinator.callback_for("workflow-2")

        self.assertEqual(first("curl a | python3", "pipe to interpreter"), "once")
        self.assertEqual(second("curl b | python3", "pipe to interpreter"), "once")
        self.assertEqual(calls, ["curl a | python3", "curl b | python3"])

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
        # Without a captured live channel, ask degrades to ask_fallback.
        cb = _make_child_approval_callback("ask", ask_fallback="deny")
        self.assertEqual(cb("rm -rf build", "recursive delete"), "deny")

    def test_ask_uses_captured_cli_approval_callback(self):
        calls = []

        def interactive(command, description, **kwargs):
            calls.append((command, description, kwargs))
            return "session"

        cb = _make_child_approval_callback(
            "ask",
            ask_fallback="deny",
            interactive_callback=interactive,
        )
        self.assertEqual(cb("python3 -c \"print('ok')\"", "script execution"), "session")
        self.assertEqual(calls[0][0], "python3 -c \"print('ok')\"")

    def test_ask_auto_allows_obvious_read_only_terminal_command(self):
        calls = []

        def interactive(command, description, **kwargs):
            calls.append(command)
            return "session"

        cb = _make_child_approval_callback(
            "ask",
            ask_fallback="deny",
            interactive_callback=interactive,
        )
        command = (
            "curl -sL https://example.test/feed.xml 2>/dev/null | "
            "python3 -c \"import sys, re; data = sys.stdin.read(); print(re.findall('title', data))\""
        )

        self.assertEqual(cb(command, "pipe to interpreter"), "once")
        self.assertEqual(calls, [])

    def test_ask_does_not_auto_allow_terminal_write(self):
        calls = []

        def interactive(command, description, **kwargs):
            calls.append(command)
            return "deny"

        cb = _make_child_approval_callback(
            "ask",
            ask_fallback="deny",
            interactive_callback=interactive,
        )
        command = "curl -sL https://example.test | python3 -c \"open('x', 'w').write('bad')\""

        self.assertEqual(cb(command, "script execution"), "deny")
        self.assertEqual(calls, [command])

    def test_inherit_manual_uses_captured_cli_approval_callback(self):
        approval_mod = types.ModuleType("tools.approval")
        approval_mod._get_approval_mode = lambda: "manual"
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []
        tools_pkg.approval = approval_mod
        calls = []

        def interactive(command, description):
            calls.append(command)
            return "once"

        with patch.dict(sys.modules, {"tools": tools_pkg, "tools.approval": approval_mod}):
            cb = _make_child_approval_callback("inherit", interactive_callback=interactive)
            self.assertEqual(cb("rm -rf build", "recursive delete"), "once")
        self.assertEqual(calls, ["rm -rf build"])

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
