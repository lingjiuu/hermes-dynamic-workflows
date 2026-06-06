from __future__ import annotations

import unittest

from hermes_dynamic_workflows.engine.token_budget import parse_token_budget


class TokenBudgetParserTests(unittest.TestCase):
    def test_parses_claude_style_targets(self):
        cases = {
            "+500k run a workflow": 500_000,
            "+500k使用 workflow": 500_000,
            "run a workflow +1.5m!": 1_500_000,
            "Please spend 2M tokens reviewing this": 2_000_000,
            "Please use 1B token on this": 1_000_000_000,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_token_budget(text), expected)

    def test_returns_none_without_supported_directive(self):
        for text in (None, "", "run a workflow", "there are +500k records"):
            with self.subTest(text=text):
                self.assertIsNone(parse_token_budget(text))


if __name__ == "__main__":
    unittest.main()
