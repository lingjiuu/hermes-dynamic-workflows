from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from hermes_dynamic_workflows.engine.cache import ResumeCache, agent_fingerprint, is_cache_miss
from hermes_dynamic_workflows.engine.config import PluginConfig
from hermes_dynamic_workflows.engine.errors import WorkflowLimitExceeded
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


class RuntimeTests(unittest.TestCase):
    def test_runs_workflow_function(self):
        script = """
meta = {"name": "simple", "phases": ["scan"]}

def workflow():
    phase("scan")
    return agent("inspect repo", {"label": "scan-agent"})
"""
        runner = FakeRunner()
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, "scan-agent:inspect repo")
        self.assertEqual(result.agent_count, 1)
        self.assertEqual(runner.requests[0].label, "scan-agent")
        self.assertEqual(runner.requests[0].toolsets, [])
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
        runner = FakeRunner(
            responses=[
                ChildAgentResult(
                    content="done",
                    metadata={
                        "structured_captured": True,
                        "structured_result": {"ok": True},
                        "structured_attempts": 1,
                    },
                )
            ]
        )
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, {"ok": True})

    def test_structured_output_does_not_parse_final_message(self):
        script = """
meta = {"name": "structured-no-parse"}

def workflow():
    return agent(
        "return status",
        {"label": "json", "schema": {"type": "object", "required": ["ok"]}},
    )
"""
        runner = FakeRunner(responses=['{"ok": true}'])
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertIsNone(result.value)
        self.assertEqual(len(runner.requests), 1)
        agent = result.state.snapshot()["agents"][0]
        self.assertEqual(agent["status"], "error")
        self.assertIn("did not submit valid structured output", agent["error"])

    def test_invalid_structured_schema_fails_before_child_launch(self):
        script = """
meta = {"name": "invalid-schema"}

def workflow():
    return agent(
        "return status",
        {"label": "json", "schema": {"type": 123}},
    )
"""
        runner = FakeRunner()
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(runner.requests, [])
        self.assertIn("invalid JSON Schema", str(ctx.exception))

    def test_agent_rejects_runtime_policy_options(self):
        script = """
meta = {"name": "unsupported-options"}

def workflow():
    return agent("go", {"label": "r", "toolsets": ["web"], "retries": 2})
"""
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=FakeRunner()))
        self.assertIn("unsupported agent() option", str(ctx.exception))
        self.assertIn("toolsets", str(ctx.exception))
        self.assertIn("retries", str(ctx.exception))

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
                config=PluginConfig(),
                child_runner=TokenRunner(tokens=40),
                token_budget_total=100,
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
        # Budget exhaustion is a hard ceiling: it raises WorkflowLimitExceeded,
        # a WorkflowHalt (BaseException) a script's `except Exception` cannot
        # swallow — so it is NOT an `Exception` subclass.
        with self.assertRaises(WorkflowLimitExceeded):
            run_workflow(
                script,
                WorkflowOptions(
                    config=PluginConfig(),
                    child_runner=TokenRunner(tokens=20),
                    token_budget_total=10,
                ),
            )
        self.assertFalse(issubclass(WorkflowLimitExceeded, Exception))

    def test_meta_token_budget_is_ignored(self):
        script = """
meta = {"name": "budget-meta", "token_budget": 100}

def workflow():
    agent("a", {"label": "a"})
    return {"total": budget.total, "remaining": budget.remaining()}
"""
        result = run_workflow(
            script,
            WorkflowOptions(config=PluginConfig(), child_runner=TokenRunner(tokens=40)),
        )

        self.assertIsNone(result.value["total"])
        self.assertEqual(result.value["remaining"], float("inf"))

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


class ResumeCacheTests(unittest.TestCase):
    def test_content_addressed_fifo_for_duplicate_fingerprints(self):
        fp = agent_fingerprint("same prompt", {"label": "x"})
        run1 = ResumeCache()
        run1.put(fp, "r1")
        run1.put(fp, "r2")

        run2 = ResumeCache(run1.current)
        # Two identical calls each consume one cached result (FIFO), then miss.
        self.assertEqual(run2.get(fp), "r1")
        self.assertEqual(run2.get(fp), "r2")
        self.assertTrue(is_cache_miss(run2.get(fp)))

    def test_ignores_malformed_cache_without_crashing(self):
        fp = agent_fingerprint("p", {"label": "y"})
        # Unexpected shapes (e.g. a crashed/hand-edited run) are ignored -> miss.
        cache = ResumeCache({fp: {"not": "a list"}, "other": 123})
        self.assertTrue(is_cache_miss(cache.get(fp)))


class ControlFlowRuntimeTests(unittest.TestCase):
    def test_while_loop_runs_end_to_end(self):
        script = """
meta = {"name": "while-ok"}

def workflow():
    results = []
    i = 0
    while i < 3:
        results.append(agent("x" + str(i)))
        i = i + 1
    return results
"""
        runner = FakeRunner()
        result = run_workflow(script, WorkflowOptions(child_runner=runner))
        self.assertEqual(len(result.value), 3)
        self.assertEqual([r.prompt for r in runner.requests], ["x0", "x1", "x2"])

    def test_try_except_handles_recoverable_error(self):
        script = """
meta = {"name": "try-ok"}

def workflow():
    try:
        y = 1 / 0
    except Exception:
        y = "caught"
    agent("a")
    return y
"""
        result = run_workflow(script, WorkflowOptions(child_runner=TokenRunner(tokens=1)))
        self.assertEqual(result.value, "caught")

    def test_except_exception_cannot_swallow_budget_halt(self):
        # A while loop that catches Exception around agent() must STILL halt when
        # the token budget is exhausted — the halt is BaseException, not caught.
        script = """
meta = {"name": "no-swallow"}

def workflow():
    out = []
    while True:
        try:
            out.append(agent("x"))
        except Exception:
            out.append("swallowed")
    return out
"""
        with self.assertRaises(WorkflowLimitExceeded):
            run_workflow(
                script,
                WorkflowOptions(
                    config=PluginConfig(),
                    child_runner=TokenRunner(tokens=20),
                    token_budget_total=10,
                ),
            )

    def test_compute_only_loop_is_bounded_by_iteration_cap(self):
        # A pure-compute infinite loop (never calls agent()) is bounded by the
        # injected loop guard's iteration cap — proving the deadline/stop check
        # actually fires inside such a loop.
        script = """
meta = {"name": "spin"}

def workflow():
    agent("a")
    while True:
        pass
    return 1
"""
        with self.assertRaises(WorkflowLimitExceeded):
            run_workflow(
                script,
                WorkflowOptions(
                    config=PluginConfig(max_loop_iterations=100),
                    child_runner=TokenRunner(tokens=1),
                ),
            )


if __name__ == "__main__":
    unittest.main()
