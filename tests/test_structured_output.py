from __future__ import annotations

import builtins
import json
import unittest
from unittest.mock import patch

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow
from hermes_dynamic_workflows.engine.structured import StructuredOutputError, validate_json_schema
from hermes_dynamic_workflows.engine.types import (
    ChildAgentRequest,
    ChildAgentResult,
    ChildAgentRunner,
)
from hermes_dynamic_workflows.plugin.structured_output import (
    STRUCTURED_OUTPUT_SUCCESS,
    STRUCTURED_OUTPUT_TOOL_SCHEMA,
    build_tool_schema_instruction,
    clear_expectation,
    peek_result,
    register_expectation,
    specialize_structured_output_tool,
    structured_output_handler,
    structured_output_tool_scope,
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


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
    def test_valid_submit_returns_claude_style_success_and_is_recorded(self):
        register_expectation("t1", _SCHEMA)
        try:
            out = structured_output_handler({"ok": True}, task_id="t1")
            self.assertEqual(out, STRUCTURED_OUTPUT_SUCCESS)
            captured, value, attempts = peek_result("t1")
            self.assertTrue(captured)
            self.assertEqual(value, {"ok": True})
            self.assertEqual(attempts, 1)
        finally:
            clear_expectation("t1")

    def test_invalid_submit_returns_hermes_tool_error(self):
        register_expectation("t2", _SCHEMA)
        try:
            out = structured_output_handler({"nope": 1}, task_id="t2")
            error = json.loads(out)["error"]
            self.assertTrue(error.startswith("Output does not match required schema: "))
            self.assertIn("must have required property 'ok'", error)
            captured, _value, attempts = peek_result("t2")
            self.assertFalse(captured)
            self.assertEqual(attempts, 1)
        finally:
            clear_expectation("t2")

    def test_retry_then_accept_counts_attempts(self):
        register_expectation("t3", _SCHEMA)
        try:
            first = structured_output_handler({}, task_id="t3")
            second = structured_output_handler({"ok": True}, task_id="t3")
            self.assertTrue(
                json.loads(first)["error"].startswith("Output does not match required schema: ")
            )
            self.assertEqual(second, STRUCTURED_OUTPUT_SUCCESS)
            captured, value, attempts = peek_result("t3")
            self.assertTrue(captured)
            self.assertEqual(value, {"ok": True})
            self.assertEqual(attempts, 2)
        finally:
            clear_expectation("t3")

    def test_submit_without_expectation_returns_schema_error_content(self):
        out = structured_output_handler({"ok": True}, task_id="missing")
        self.assertTrue(
            json.loads(out)["error"].startswith("Output does not match required schema: ")
        )

    def test_fifth_invalid_submit_interrupts_and_later_valid_submit_is_rejected(self):
        interrupts = []
        register_expectation("t4", _SCHEMA, lambda: interrupts.append(True))
        try:
            for _ in range(5):
                out = structured_output_handler({}, task_id="t4")
                self.assertTrue(
                    json.loads(out)["error"].startswith("Output does not match required schema: ")
                )
            late = structured_output_handler({"ok": True}, task_id="t4")
            captured, _value, attempts = peek_result("t4")
        finally:
            clear_expectation("t4")

        self.assertEqual(interrupts, [True])
        self.assertIn("maximum structured output attempts exceeded", json.loads(late)["error"])
        self.assertFalse(captured)
        self.assertEqual(attempts, 6)


class DynamicToolSchemaTests(unittest.TestCase):
    def test_tool_description_and_prompt_match_claude_wording_with_lowercase_name(self):
        self.assertEqual(
            STRUCTURED_OUTPUT_TOOL_SCHEMA["description"],
            "Return structured output in the requested format",
        )
        self.assertEqual(
            build_tool_schema_instruction().strip(),
            (
                "Use the structured_output tool to return your final response in the requested "
                "structured format. You MUST call this tool exactly once at the end of your "
                "response to provide the structured output."
            ),
        )

    def test_specializes_child_local_tool_schema_without_mutating_original(self):
        original = [
            {
                "type": "function",
                "function": {
                    "name": "structured_output",
                    "description": "placeholder",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        specialized = specialize_structured_output_tool(original, _SCHEMA)

        self.assertEqual(specialized[0]["function"]["parameters"], _SCHEMA)
        self.assertEqual(original[0]["function"]["parameters"], {"type": "object", "properties": {}})
        self.assertIsNot(specialized[0], original[0])

    def test_fallback_schema_precheck_rejects_complex_keywords_without_jsonschema(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "jsonschema" or name.startswith("jsonschema."):
                raise ImportError(name)
            return original_import(name, *args, **kwargs)

        schema = {
            "type": "object",
            "patternProperties": {"^x-": {"type": "string"}},
        }
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(StructuredOutputError) as ctx:
                validate_json_schema(schema)

        message = str(ctx.exception)
        self.assertIn("jsonschema package is not installed", message)
        self.assertIn("patternProperties", message)
        self.assertIn('Install "jsonschema>=4,<5"', message)

    def test_tool_registry_scope_registers_once_and_removes_after_last_user(self):
        class Entry:
            def __init__(self):
                self.handler = structured_output_handler

        class Registry:
            def __init__(self):
                self.entry = None
                self.register_calls = 0
                self.deregister_calls = 0

            def get_entry(self, _name):
                return self.entry

            def register(self, **_kwargs):
                self.register_calls += 1
                self.entry = Entry()

            def deregister(self, _name):
                self.deregister_calls += 1
                self.entry = None

        registry = Registry()
        with patch(
            "hermes_dynamic_workflows.plugin.structured_output._tool_registry",
            return_value=registry,
        ):
            self.assertIsNone(registry.entry)
            with structured_output_tool_scope():
                self.assertIsNotNone(registry.entry)
                with structured_output_tool_scope():
                    self.assertEqual(registry.register_calls, 1)
                self.assertIsNotNone(registry.entry)
            self.assertIsNone(registry.entry)

        self.assertEqual(registry.register_calls, 1)
        self.assertEqual(registry.deregister_calls, 1)


class ToolChannelTests(unittest.TestCase):
    def test_schema_agent_requires_structured_output_and_returns_captured_result(self):
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
        self.assertIn("structured_output", request.prompt)
        agent = result.state.snapshot()["agents"][0]
        self.assertEqual(agent["structured"]["mode"], "tool")
        self.assertEqual(agent["structured"]["status"], "valid")


if __name__ == "__main__":
    unittest.main()
