from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow
from hermes_dynamic_workflows.engine.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner


class FakeRunner(ChildAgentRunner):
    def __init__(self, responses=None):
        self.requests: list[ChildAgentRequest] = []
        self.responses = list(responses or [])

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return f"{request.label}:{request.prompt}"


class IdRunner(ChildAgentRunner):
    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        return f"{request.id}:{request.label}"


class TokenRunner(ChildAgentRunner):
    def __init__(self, tokens: int):
        self.tokens = tokens

    def run(self, request: ChildAgentRequest):
        return ChildAgentResult(content=request.label, metadata={"tokens": self.tokens})


class ResponseFormatFallbackRunner(ChildAgentRunner):
    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        if request.request_overrides:
            raise RuntimeError("response_format is not supported by this provider")
        return '{"ok": true}'


class RuntimeTests(unittest.TestCase):
    def test_runs_workflow_function(self):
        script = """
meta = {"name": "simple", "phases": ["scan"]}

def workflow():
    phase("scan")
    return agent("inspect repo", {"label": "scan-agent", "toolsets": ["web"]})
"""
        runner = FakeRunner()
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, "scan-agent:inspect repo")
        self.assertEqual(result.agent_count, 1)
        self.assertEqual(runner.requests[0].label, "scan-agent")
        self.assertEqual(runner.requests[0].toolsets, ["web"])
        self.assertEqual(result.state.current_phase, "scan")

    def test_parallel_preserves_order(self):
        script = """
meta = {"name": "parallel"}

def workflow():
    return parallel([
        lambda: agent("a", {"label": "a"}),
        lambda: agent("b", {"label": "b"}),
        lambda: agent("c", {"label": "c"}),
    ])
"""
        runner = FakeRunner()
        result = run_workflow(
            script,
            WorkflowOptions(config=PluginConfig(concurrency=2), child_runner=runner),
        )

        self.assertEqual(result.value, ["a:a", "b:b", "c:c"])
        self.assertEqual({req.label for req in runner.requests}, {"a", "b", "c"})

    def test_structured_output(self):
        script = """
meta = {"name": "structured"}

def workflow():
    return agent(
        "return status",
        {"label": "json", "schema": {"type": "object", "required": ["ok"]}},
    )
"""
        runner = FakeRunner(responses=['{"ok": true}'])
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, {"ok": True})

    def test_structured_output_repairs_invalid_json(self):
        script = """
meta = {"name": "structured-repair"}

def workflow():
    return agent(
        "return status",
        {"label": "json", "schema": {"type": "object", "required": ["ok"]}},
    )
"""
        runner = FakeRunner(responses=["not json", '{"ok": true}'])
        result = run_workflow(
            script,
            WorkflowOptions(
                config=PluginConfig(structured_repair_with_llm=False),
                child_runner=runner,
            ),
        )

        self.assertEqual(result.value, {"ok": True})
        self.assertEqual(len(runner.requests), 2)
        self.assertIn("structured-repair", runner.requests[1].label)
        agent = result.state.snapshot()["agents"][0]
        self.assertEqual(agent["structured"]["status"], "repaired")

    def test_structured_output_falls_back_when_response_format_is_unsupported(self):
        script = """
meta = {"name": "structured-fallback"}

def workflow():
    return agent(
        "return status",
        {"label": "json", "schema": {"type": "object", "required": ["ok"]}},
    )
"""
        runner = ResponseFormatFallbackRunner()
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, {"ok": True})
        self.assertEqual(len(runner.requests), 2)
        self.assertIsNotNone(runner.requests[0].request_overrides)
        self.assertIsNone(runner.requests[1].request_overrides)
        agent = result.state.snapshot()["agents"][0]
        self.assertEqual(agent["structured"]["status"], "valid")
        self.assertEqual(agent["structured"]["mode"], "prompt")
        self.assertIn("response_format_error", agent["structured"])

    def test_requires_agent_call(self):
        script = """
meta = {"name": "empty"}

def workflow():
    return "no agents"
"""
        with self.assertRaises(Exception):
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=FakeRunner()))

    def test_subworkflow_shares_global_agent_sequence_and_snapshot_tree(self):
        parent = """
meta = {"name": "parent", "phases": [{"title": "Root"}]}

def workflow():
    phase("Root")
    first = agent("root", {"label": "root"})
    child = subworkflow({"scriptPath": args["child"]})
    last = agent("after", {"label": "after"})
    return [first, child, last]
"""
        child = """
meta = {"name": "child", "phases": [{"title": "Child"}]}

def workflow():
    phase("Child")
    return agent("child", {"label": "child"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            child_path = Path(tmp) / "child.py"
            child_path.write_text(child, encoding="utf-8")
            runner = IdRunner()
            result = run_workflow(
                parent,
                WorkflowOptions(
                    args={"child": str(child_path)},
                    cwd=tmp,
                    config=PluginConfig(),
                    child_runner=runner,
                ),
            )

        self.assertEqual(result.value, ["1:root", "2:child", "3:after"])
        snapshot = result.state.snapshot()
        self.assertEqual(snapshot["agents"][0]["id"], 1)
        self.assertEqual(snapshot["children"][0]["agents"][0]["id"], 2)
        self.assertEqual(snapshot["agents"][1]["id"], 3)
        self.assertEqual(snapshot["totals"]["agents"], 3)

    def test_budget_is_token_budget(self):
        script = """
meta = {"name": "budget"}

def workflow():
    agent("a", {"label": "a"})
    return {"total": budget.total, "spent": budget.spent(), "remaining": budget.remaining()}
"""
        result = run_workflow(
            script,
            WorkflowOptions(
                config=PluginConfig(token_budget_total=100),
                child_runner=TokenRunner(tokens=40),
            ),
        )

        self.assertEqual(result.value, {"total": 100, "spent": 40, "remaining": 60})

    def test_token_budget_blocks_further_agents(self):
        script = """
meta = {"name": "budget-stop"}

def workflow():
    agent("a", {"label": "a"})
    return agent("b", {"label": "b"})
"""
        with self.assertRaises(Exception):
            run_workflow(
                script,
                WorkflowOptions(
                    config=PluginConfig(token_budget_total=10),
                    child_runner=TokenRunner(tokens=20),
                ),
            )

    def test_subworkflow_nesting_is_one_level(self):
        parent = """
meta = {"name": "parent"}

def workflow():
    return subworkflow({"scriptPath": args["child"]}, args)
"""
        child = """
meta = {"name": "child"}

def workflow():
    return subworkflow({"scriptPath": args["grand"]})
"""
        grand = """
meta = {"name": "grand"}

def workflow():
    return agent("grand")
"""
        with tempfile.TemporaryDirectory() as tmp:
            child_path = Path(tmp) / "child.py"
            grand_path = Path(tmp) / "grand.py"
            child_path.write_text(child, encoding="utf-8")
            grand_path.write_text(grand, encoding="utf-8")
            with self.assertRaises(Exception):
                run_workflow(
                    parent,
                    WorkflowOptions(
                        args={"child": str(child_path), "grand": str(grand_path)},
                        cwd=tmp,
                        config=PluginConfig(),
                        child_runner=FakeRunner(),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
