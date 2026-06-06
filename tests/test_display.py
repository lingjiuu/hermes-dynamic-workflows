from __future__ import annotations

import unittest

from hermes_dynamic_workflows.ui.display import render_agent_detail, render_workflow_text


class DisplayTests(unittest.TestCase):
    def test_renders_unphased_agents_when_phases_exist(self):
        text = render_workflow_text(
            {
                "meta": {"name": "display"},
                "phases": ["Review"],
                "agents": [
                    {"id": 1, "label": "unphased", "status": "done", "phase": None},
                    {"id": 2, "label": "review", "status": "running", "phase": "Review"},
                ],
            },
            completed=False,
        )

        self.assertIn("[Review]", text)
        self.assertIn("review", text)
        self.assertIn("[Other]", text)
        self.assertIn("unphased", text)

    def test_renders_child_workflow_and_finds_child_agent(self):
        run = {
            "runId": "wf_test123",
            "status": "running",
            "workflow": {
                "meta": {"name": "parent"},
                "phases": [],
                "agents": [],
                "children": [
                    {
                        "meta": {"name": "child"},
                        "phases": [{"title": "Child"}],
                        "agents": [
                            {
                                "id": 2,
                                "label": "child-agent",
                                "status": "done",
                                "phase": "Child",
                                "prompt": "work",
                                "result_preview": "done",
                            }
                        ],
                        "children": [],
                        "errors": [],
                    }
                ],
                "errors": [],
            },
        }

        text = render_workflow_text(run["workflow"], completed=False)
        detail = render_agent_detail(run, "2")

        self.assertIn("> child", text)
        self.assertIn("child-agent", text)
        self.assertIn("agent #2 child-agent", detail)

    def test_renders_structured_output_detail(self):
        run = {
            "runId": "wf_test123",
            "status": "completed",
            "workflow": {
                "meta": {"name": "structured"},
                "phases": [],
                "agents": [
                    {
                        "id": 1,
                        "label": "json",
                        "status": "done",
                        "prompt": "work",
                        "result_preview": "{'ok': True}",
                        "structured": {
                            "status": "valid",
                            "mode": "tool",
                            "attempts": 2,
                        },
                    }
                ],
                "children": [],
                "errors": [],
            },
        }

        render_workflow_text(run["workflow"], completed=True)
        detail = render_agent_detail(run, "json")

        self.assertIn("Structured output", detail)
        self.assertIn("Status: valid", detail)
        self.assertIn("Mode: tool", detail)
        self.assertIn("Attempts: 2", detail)


if __name__ == "__main__":
    unittest.main()
