from __future__ import annotations

import unittest

from hermes_dynamic_workflows.view.render import render_agent_overview, render_workflow_text


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
        overview = render_agent_overview([run])

        self.assertIn("> child", text)
        self.assertIn("child-agent", text)
        self.assertIn("child-agent", overview)

    def test_renders_agent_overview_with_structured_failure(self):
        run = {
            "runId": "wf_test123",
            "status": "completed",
            "taskId": "wgtest01",
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
                            "status": "failed",
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
        overview = render_agent_overview([run])

        self.assertIn("structured", overview)
        self.assertIn("wgtest01", overview)
        self.assertIn("json", overview)
        self.assertIn("schema failed", overview)


if __name__ == "__main__":
    unittest.main()
