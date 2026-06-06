from __future__ import annotations

import ast
import unittest

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import SandboxViolation, WorkflowParseError
from hermes_dynamic_workflows.engine.sandbox import (
    LOOP_GUARD_NAME,
    extract_meta,
    parse_script,
)


def workflow_script(body: str) -> str:
    return 'meta = {"name": "test", "description": "Test workflow"}\n\n' + body


class SandboxTests(unittest.TestCase):
    def test_blocks_import(self):
        with self.assertRaises(SandboxViolation):
            parse_script(workflow_script("import os\nreturn 1"), PluginConfig())

    def test_blocks_dunder_attribute(self):
        with self.assertRaises(SandboxViolation):
            parse_script(workflow_script("return (1).__class__"), PluginConfig())

    def test_requires_meta_first(self):
        with self.assertRaises(WorkflowParseError) as ctx:
            parse_script("return 1", PluginConfig())
        self.assertEqual(
            str(ctx.exception),
            "Invalid workflow script: `meta = {...}` must be the FIRST statement in the script",
        )

    def test_requires_meta_description(self):
        with self.assertRaises(WorkflowParseError) as missing:
            parse_script('meta = {"name": "missing"}\nreturn 1', PluginConfig())
        self.assertEqual(
            str(missing.exception),
            "Invalid workflow script: meta.description must be a non-empty string",
        )

        with self.assertRaises(WorkflowParseError) as empty:
            parse_script('meta = {"name": "empty", "description": ""}\nreturn 1', PluginConfig())
        self.assertEqual(
            str(empty.exception),
            "Invalid workflow script: meta.description must be a non-empty string",
        )

    def test_requires_meta_name(self):
        with self.assertRaises(WorkflowParseError) as missing:
            parse_script('meta = {"description": "Missing name"}\nreturn 1', PluginConfig())
        self.assertEqual(
            str(missing.exception),
            "Invalid workflow script: meta.name must be a non-empty string",
        )

        with self.assertRaises(WorkflowParseError) as empty:
            parse_script('meta = {"name": "", "description": "Empty name"}\nreturn 1', PluginConfig())
        self.assertEqual(
            str(empty.exception),
            "Invalid workflow script: meta.name must be a non-empty string",
        )

    def test_rejects_dynamic_meta_literals(self):
        with self.assertRaises(WorkflowParseError) as interpolation:
            parse_script('meta = {"name": f"wf-{1}", "description": "Bad"}\nreturn 1', PluginConfig())
        self.assertEqual(
            str(interpolation.exception),
            "Invalid workflow script: meta must be a pure literal: template interpolation not allowed in meta",
        )

        with self.assertRaises(WorkflowParseError) as call:
            parse_script('meta = {"name": str("wf"), "description": "Bad"}\nreturn 1', PluginConfig())
        self.assertEqual(
            str(call.exception),
            "Invalid workflow script: meta must be a pure literal: non-literal node type in meta: Call",
        )

        with self.assertRaises(WorkflowParseError) as spread:
            parse_script(
                'meta = {**{"name": "wf"}, "description": "Bad"}\nreturn 1',
                PluginConfig(),
            )
        self.assertEqual(
            str(spread.exception),
            "Invalid workflow script: meta must be a pure literal: only plain properties allowed in meta",
        )

    def test_parse_error_uses_plain_python_message(self):
        with self.assertRaises(WorkflowParseError) as ctx:
            parse_script(
                'meta = {"name": "syntax", "description": "Syntax"}\nreturn (',
                PluginConfig(),
            )
        self.assertIn("Invalid workflow script: Script parse error:", str(ctx.exception))
        self.assertIn("Workflow scripts must be plain Python.", str(ctx.exception))

    def test_rejects_nondeterministic_calls(self):
        with self.assertRaises(SandboxViolation) as ctx:
            parse_script(workflow_script("return time.time()"), PluginConfig())
        self.assertEqual(
            str(ctx.exception),
            "Workflow scripts must be deterministic: current time and randomness are "
            "unavailable (breaks resume). Stamp results after the workflow returns, "
            "or pass timestamps via args.",
        )

    def test_rejects_old_workflow_entrypoint(self):
        with self.assertRaises(WorkflowParseError):
            parse_script(
                workflow_script("async def workflow():\n    return 1"),
                PluginConfig(),
            )

    def test_allows_workflow_calls(self):
        tree = parse_script(
            """
meta = {"name": "ok", "description": "Test workflow"}

return await agent("hello")
""",
            PluginConfig(),
        )
        self.assertIsNotNone(tree)

    def test_allows_phase_objects(self):
        tree = parse_script(
            """
meta = {"name": "ok", "description": "Test workflow", "phases": [{"title": "Scan", "detail": "inspect", "model": "sonnet"}]}

return await agent("hello")
""",
            PluginConfig(),
        )
        meta = extract_meta(tree)
        self.assertEqual(meta["phases"][0]["title"], "Scan")


class ControlFlowAllowedTests(unittest.TestCase):
    """while/try/raise are pure control flow — now allowed (the docs' loop-
    until-budget / loop-until-dry / catch-gracefully patterns need them)."""

    def test_while_is_allowed(self):
        parse_script(workflow_script("while a > 0:\n    a = a - 1\n"), PluginConfig())

    def test_try_except_exception_is_allowed(self):
        parse_script(
            workflow_script("try:\n    x = 1\nexcept Exception:\n    x = 2\n"),
            PluginConfig(),
        )

    def test_raise_is_allowed(self):
        parse_script(workflow_script("raise Exception('boom')\n"), PluginConfig())


class WildcardExceptForbiddenTests(unittest.TestCase):
    """A WorkflowHalt is BaseException, so `except Exception` can't catch it; we
    additionally forbid the wildcard forms that could."""

    def test_bare_except_rejected(self):
        with self.assertRaises(SandboxViolation):
            parse_script(workflow_script("try:\n    x = 1\nexcept:\n    x = 2\n"), PluginConfig())

    def test_except_base_exception_rejected(self):
        with self.assertRaises(SandboxViolation):
            parse_script(
                workflow_script("try:\n    x = 1\nexcept BaseException:\n    x = 2\n"),
                PluginConfig(),
            )

    def test_except_tuple_with_base_exception_rejected(self):
        with self.assertRaises(SandboxViolation):
            parse_script(
                workflow_script(
                    "try:\n    x = 1\nexcept (ValueError, BaseException):\n    x = 2\n"
                ),
                PluginConfig(),
            )


class LoopInstrumentationTests(unittest.TestCase):
    def test_while_test_is_wrapped_with_guard(self):
        tree = parse_script(workflow_script("while a:\n    a = a - 1\n"), PluginConfig())
        self.assertIn(LOOP_GUARD_NAME, ast.unparse(tree))

    def test_for_loop_body_starts_with_guard(self):
        tree = parse_script(workflow_script("for i in range(3):\n    x = i\n"), PluginConfig())
        for_node = next(node for node in ast.walk(tree) if isinstance(node, ast.For))
        self.assertIn(LOOP_GUARD_NAME, ast.unparse(for_node.body[0]))


if __name__ == "__main__":
    unittest.main()
