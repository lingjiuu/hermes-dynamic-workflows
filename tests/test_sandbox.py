from __future__ import annotations

import unittest

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.errors import SandboxViolation
from hermes_dynamic_workflows.engine.sandbox import extract_meta, parse_script


class SandboxTests(unittest.TestCase):
    def test_blocks_import(self):
        with self.assertRaises(SandboxViolation):
            parse_script("import os\nreturn_value = 1", PluginConfig())

    def test_blocks_dunder_attribute(self):
        with self.assertRaises(SandboxViolation):
            parse_script("return_value = (1).__class__", PluginConfig())

    def test_allows_workflow_calls(self):
        tree = parse_script(
            """
meta = {"name": "ok"}

def workflow():
    return agent("hello")
""",
            PluginConfig(),
        )
        self.assertIsNotNone(tree)

    def test_allows_phase_objects(self):
        tree = parse_script(
            """
meta = {"name": "ok", "phases": [{"title": "Scan", "detail": "inspect", "model": "sonnet"}]}

def workflow():
    return agent("hello")
""",
            PluginConfig(),
        )
        meta = extract_meta(tree)
        self.assertEqual(meta["phases"][0]["title"], "Scan")


if __name__ == "__main__":
    unittest.main()
