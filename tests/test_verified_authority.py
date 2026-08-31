from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hermes_dynamic_workflows.child.runner import HermesChildAgentRunner
from hermes_dynamic_workflows.child.verified_authority import (
    VERIFIED_AUTHORITY_PRESET_IDS,
    VerifiedAuthorityPreset,
    VerifiedAuthorityEnforcer,
    lock_verified_authority,
    verified_authority_policy_bytes,
    verified_authority_policy_digest,
)
from hermes_dynamic_workflows.core.errors import ChildAgentError
from hermes_dynamic_workflows.core.types import ChildAgentRequest


class VerifiedAuthorityPolicyTests(unittest.TestCase):
    def test_policy_bytes_and_digest_are_stable_public_contracts(self):
        first = verified_authority_policy_bytes()
        second = verified_authority_policy_bytes()

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertEqual(len(verified_authority_policy_digest()), 64)
        self.assertIn("dws-read-only-v1", VERIFIED_AUTHORITY_PRESET_IDS)
        self.assertIn(b'"maximumScale":64', first)
        self.assertIn(b'"allowStructuredOutput":true', first)
        self.assertIn(b'"structured_output"', first)

    def test_locked_policy_replaces_document_child_authority(self):
        with TemporaryDirectory() as tmp:
            context = lock_verified_authority(
                "dws-read-only-v1",
                enforced_max_scale=2,
                cwd=tmp,
                execution_digest="a" * 64,
            )
            request = ChildAgentRequest(
                id=1,
                prompt="inspect",
                label="worker",
                phase=None,
                toolsets=["dangerous-raw-toolset"],
                model="document-model",
                agent_type="substituted-preset",
                isolation="worktree",
                cwd=tmp,
            )

            constrained = VerifiedAuthorityEnforcer(context).constrain(request)

        self.assertEqual(constrained.toolsets, ["web", "file", "terminal"])
        self.assertIsNone(constrained.model)
        self.assertEqual(constrained.agent_type, "verified-dws-read-only")
        self.assertEqual(constrained.isolation, "shared")
        self.assertEqual(constrained.cwd, str(Path(tmp).resolve()))
        self.assertEqual(
            constrained.resolved.allowed_tools,
            ("web_search", "web_extract", "read_file", "search_files", "terminal", "structured_output"),
        )

    def test_verified_authority_schema_is_allowed_only_when_bound_by_locked_preset(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        with TemporaryDirectory() as tmp:
            context = lock_verified_authority(
                "dws-read-only-v1",
                enforced_max_scale=2,
                cwd=tmp,
                execution_digest="d" * 64,
            )
            request = ChildAgentRequest(
                id=1,
                prompt="inspect",
                label="worker",
                phase=None,
                toolsets=["dangerous-raw-toolset"],
                model="document-model",
                agent_type="substituted-preset",
                isolation="worktree",
                cwd=tmp,
                schema=schema,
                structured_tool=True,
            )

            constrained = VerifiedAuthorityEnforcer(context).constrain(request)

        self.assertEqual(constrained.schema, schema)
        self.assertTrue(constrained.structured_tool)
        self.assertEqual(
            constrained.resolved.allowed_tools,
            ("web_search", "web_extract", "read_file", "search_files", "terminal", "structured_output"),
        )

    def test_request_overrides_child_context_and_scale_widening_fail_closed(self):
        with TemporaryDirectory() as tmp:
            context = lock_verified_authority(
                "dws-read-only-v1",
                enforced_max_scale=1,
                cwd=tmp,
                execution_digest="b" * 64,
            )
            enforcer = VerifiedAuthorityEnforcer(context)
            base = ChildAgentRequest(
                id=1,
                prompt="inspect",
                label="worker",
                phase=None,
                toolsets=[],
                cwd=tmp,
            )

            with self.assertRaises(ChildAgentError):
                enforcer.constrain(replace(base, request_overrides={"temperature": 2}))
            with self.assertRaises(ChildAgentError):
                enforcer.constrain(replace(base, cwd=str(Path(tmp) / "other")))
            enforcer.constrain(base)
            with self.assertRaises(ChildAgentError):
                enforcer.constrain(replace(base, id=2))

            substituted = replace(context, policy_digest="0" * 64)
            with self.assertRaises(ChildAgentError):
                VerifiedAuthorityEnforcer(substituted).constrain(base)

    def test_verified_authority_rejects_schema_when_locked_preset_disallows_it(self):
        schema = {"type": "object"}
        locked_preset = VerifiedAuthorityPreset(
            preset_id="dws-no-structured-v1",
            agent_type="verified-dws-no-structured",
            instructions="Read only.",
            toolsets=("web", "file", "terminal"),
            allowed_tools=("web_search", "read_file"),
            disallowed_tools=(),
            isolation="shared",
            model_policy="inherit",
            allow_request_overrides=False,
            maximum_scale=1,
            allow_structured_output=False,
        )

        with TemporaryDirectory() as tmp:
            base_context = lock_verified_authority(
                "dws-read-only-v1",
                enforced_max_scale=1,
                cwd=tmp,
                execution_digest="e" * 64,
            )
            context = replace(base_context, preset_id=locked_preset.preset_id)
            request = ChildAgentRequest(
                id=1,
                prompt="inspect",
                label="worker",
                phase=None,
                toolsets=[],
                cwd=tmp,
                schema=schema,
                structured_tool=True,
            )

            with (
                patch(
                    "hermes_dynamic_workflows.child.verified_authority.verified_authority_policy_digest",
                    return_value=context.policy_digest,
                ),
                patch(
                    "hermes_dynamic_workflows.child.verified_authority.resolve_verified_authority_preset",
                    return_value=locked_preset,
                ),
                self.assertRaises(ChildAgentError),
            ):
                VerifiedAuthorityEnforcer(context).constrain(request)

    def test_runner_only_keeps_explicitly_allowed_structured_output_for_verified_children(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        with TemporaryDirectory() as tmp:
            context = lock_verified_authority(
                "dws-read-only-v1",
                enforced_max_scale=1,
                cwd=tmp,
                execution_digest="f" * 64,
            )
            runner = HermesChildAgentRunner(
                config=type(
                    "Config",
                    (),
                    {
                        "keep_worktrees": False,
                        "allow_model_override": True,
                        "default_child_toolsets": ["web", "file", "terminal"],
                        "blocked_child_toolsets": [],
                        "child_approval_policy": "approve",
                    },
                )(),
                verified_authority_context=context,
            )
            request = ChildAgentRequest(
                id=7,
                prompt="inspect",
                label="worker",
                phase=None,
                toolsets=[],
                cwd=tmp,
                schema=schema,
                structured_tool=True,
            )
            observed: dict[str, object] = {}

            def fake_build_agent(constrained_request, runtime, toolsets, lease, agent_type):
                observed["request"] = constrained_request
                observed["toolsets"] = list(toolsets)
                observed["agent_type"] = agent_type
                child = type(
                    "Child",
                    (),
                    {
                        "tools": [
                            {"function": {"name": "web_search"}},
                            {"function": {"name": "structured_output"}},
                            {"function": {"name": "write_file"}},
                        ],
                        "valid_tool_names": set(),
                    },
                )()
                return child

            def fake_run_child(child, constrained_request, lease, agent_type, toolsets):
                observed["tool_names"] = {tool["function"]["name"] for tool in child.tools}
                return "ok"

            with (
                patch.object(runner, "_resolve_runtime", return_value={"model": "runtime-model"}),
                patch(
                    "hermes_dynamic_workflows.child.runner.create_workspace_lease",
                    return_value=type(
                        "Lease",
                        (),
                        {"task_id": "lease-1", "cwd": tmp, "path": tmp, "repo_root": None, "isolation": "shared"},
                    )(),
                ),
                patch("hermes_dynamic_workflows.child.runner._prepare_mcp_tool_registry"),
                patch.object(runner, "_build_agent", side_effect=fake_build_agent),
                patch.object(runner, "_run_child_with_timeout", side_effect=fake_run_child),
            ):
                result = runner.run(request)

        self.assertEqual(result, "ok")
        self.assertIn("workflow_structured", observed["toolsets"])
        self.assertIn("structured_output", observed["tool_names"])
        self.assertNotIn("write_file", observed["tool_names"])
        self.assertTrue(observed["tool_names"].issubset(set(observed["request"].resolved.allowed_tools)))

    def test_handwritten_structured_output_behavior_is_unchanged_without_verified_authority(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        runner = HermesChildAgentRunner(
            config=type(
                "Config",
                (),
                {
                    "keep_worktrees": False,
                    "allow_model_override": True,
                    "default_child_toolsets": ["web", "file", "terminal"],
                    "blocked_child_toolsets": [],
                    "child_approval_policy": "approve",
                },
            )(),
        )
        request = ChildAgentRequest(
            id=8,
            prompt="inspect",
            label="worker",
            phase=None,
            toolsets=[],
            cwd="/tmp",
            schema=schema,
            structured_tool=True,
        )
        observed: dict[str, object] = {}

        def fake_build_agent(constrained_request, runtime, toolsets, lease, agent_type):
            observed["toolsets"] = list(toolsets)
            child = type(
                "Child",
                (),
                {
                    "tools": [
                        {"function": {"name": "structured_output"}},
                    ],
                    "valid_tool_names": set(),
                },
            )()
            return child

        def fake_run_child(child, constrained_request, lease, agent_type, toolsets):
            observed["tool_names"] = {tool["function"]["name"] for tool in child.tools}
            return "ok"

        with (
            patch.object(runner, "_resolve_runtime", return_value={"model": "runtime-model"}),
            patch(
                "hermes_dynamic_workflows.child.runner.create_workspace_lease",
                return_value=type(
                    "Lease",
                    (),
                    {"task_id": "lease-2", "cwd": "/tmp", "path": "/tmp", "repo_root": None, "isolation": "shared"},
                )(),
            ),
            patch("hermes_dynamic_workflows.child.runner._prepare_mcp_tool_registry"),
            patch.object(runner, "_build_agent", side_effect=fake_build_agent),
            patch.object(runner, "_run_child_with_timeout", side_effect=fake_run_child),
        ):
            result = runner.run(request)

        self.assertEqual(result, "ok")
        self.assertIn("workflow_structured", observed["toolsets"])
        self.assertIn("structured_output", observed["tool_names"])

    def test_runner_applies_verified_authority_before_child_execution(self):
        with TemporaryDirectory() as tmp:
            context = lock_verified_authority(
                "dws-read-only-v1",
                enforced_max_scale=2,
                cwd=tmp,
                execution_digest="c" * 64,
            )
            runner = HermesChildAgentRunner(
                config=type("Config", (), {
                    "keep_worktrees": False,
                    "allow_model_override": True,
                    "default_child_toolsets": ["web", "file", "terminal"],
                    "blocked_child_toolsets": [],
                    "child_approval_policy": "approve",
                })(),
                verified_authority_context=context,
            )
            request = ChildAgentRequest(
                id=7,
                prompt="inspect",
                label="worker",
                phase=None,
                toolsets=["danger"],
                model="substituted-model",
                agent_type="substituted-agent-type",
                isolation="worktree",
                cwd=tmp,
                request_overrides={"temperature": 2},
            )

            with patch.object(runner, "_resolve_runtime", return_value={"model": "runtime-model"}):
                with self.assertRaises(ChildAgentError):
                    runner.run(request)

            safe_request = replace(request, request_overrides=None)
            observed: dict[str, object] = {}

            def fake_build_agent(constrained_request, runtime, toolsets, lease, agent_type):
                observed["request"] = constrained_request
                observed["toolsets"] = list(toolsets)
                observed["agent_type"] = agent_type
                child = type("Child", (), {"tools": [], "valid_tool_names": set()})()
                return child

            def fake_run_child(child, constrained_request, lease, agent_type, toolsets):
                return "ok"

            with (
                patch.object(runner, "_resolve_runtime", return_value={"model": "runtime-model"}),
                patch("hermes_dynamic_workflows.child.runner.create_workspace_lease", return_value=type(
                    "Lease",
                    (),
                    {"task_id": "lease-1", "cwd": tmp, "path": tmp, "repo_root": None, "isolation": "shared"},
                )()),
                patch("hermes_dynamic_workflows.child.runner._prepare_mcp_tool_registry"),
                patch("hermes_dynamic_workflows.child.runner._configure_child_tools"),
                patch.object(runner, "_build_agent", side_effect=fake_build_agent),
                patch.object(runner, "_run_child_with_timeout", side_effect=fake_run_child),
            ):
                result = runner.run(safe_request)

        self.assertEqual(result, "ok")
        constrained = observed["request"]
        self.assertEqual(constrained.toolsets, ["web", "file", "terminal"])
        self.assertIsNone(constrained.model)
        self.assertEqual(constrained.agent_type, "verified-dws-read-only")
        self.assertEqual(constrained.isolation, "shared")
        self.assertIsNone(constrained.request_overrides)
        self.assertEqual(observed["toolsets"], ["web", "file", "terminal"])


if __name__ == "__main__":
    unittest.main()
