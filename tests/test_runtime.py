from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from hermes_dynamic_workflows.engine.cache import ResumeCache, agent_fingerprint, is_cache_miss
from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import (
    ChildAgentError,
    ChildAgentSkipped,
    WorkflowLimitExceeded,
    WorkflowParseError,
)
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner
from hermes_dynamic_workflows.storage.store import WorkflowStore


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


class FailingRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        raise RuntimeError(f"failed:{request.label}")


class SkippingRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        raise ChildAgentSkipped("skipped by user")


class RuntimeTests(unittest.TestCase):
    def test_runs_strict_async_script_body(self):
        script = """
meta = {"name": "simple", "description": "Test workflow", "phases": ["scan"]}

phase("scan")
return await agent("inspect repo", {"label": "scan-agent"})
"""
        runner = FakeRunner()
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, "scan-agent:inspect repo")
        self.assertEqual(result.agent_count, 1)
        self.assertEqual(runner.requests[0].label, "scan-agent")
        self.assertEqual(runner.requests[0].toolsets, ["web", "file", "terminal", "skills"])
        self.assertEqual(result.state.current_phase, "scan")

    def test_rejects_sync_workflow_function(self):
        script = """
meta = {"name": "sync-is-not-supported", "description": "Test workflow"}

def workflow():
    return "old sync DSL"
"""
        with self.assertRaises(WorkflowParseError) as ctx:
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=FakeRunner()))

        self.assertIn("do not define workflow()", str(ctx.exception))

    def test_top_level_await_script_body(self):
        script = """
meta = {"name": "top-level-await", "description": "Test workflow", "phases": ["scan"]}

phase("scan")
return await agent("inspect repo", {"label": "top-agent"})
"""
        runner = FakeRunner()
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(result.value, "top-agent:inspect repo")
        self.assertEqual(result.agent_count, 1)
        self.assertEqual(runner.requests[0].label, "top-agent")
        self.assertEqual(result.state.current_phase, "scan")

    def test_parallel_preserves_order(self):
        script = """
meta = {"name": "parallel", "description": "Test workflow"}

return await parallel([
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

    def test_parallel_rejects_arrays_over_vm_boundary_before_agent_launch(self):
        script = """
meta = {"name": "too-many-parallel", "description": "Test workflow"}

thunks = [lambda i=i: agent(str(i), {"label": str(i)}) for i in range(4097)]
return await parallel(thunks)
"""
        runner = FakeRunner()
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(runner.requests, [])
        self.assertIn(
            "array length 4097 exceeds the maximum of 4096 supported across the workflow VM boundary",
            str(ctx.exception),
        )

    def test_pipeline_rejects_arrays_over_vm_boundary_before_agent_launch(self):
        script = """
meta = {"name": "too-many-pipeline", "description": "Test workflow"}

items = list(range(4097))
return await pipeline(items, lambda item, original, index: agent(str(item)))
"""
        runner = FakeRunner()
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))

        self.assertEqual(runner.requests, [])
        self.assertIn(
            "array length 4097 exceeds the maximum of 4096 supported across the workflow VM boundary",
            str(ctx.exception),
        )

    def test_structured_output(self):
        script = """
meta = {"name": "structured", "description": "Test workflow"}

return await agent(
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
meta = {"name": "structured-no-parse", "description": "Test workflow"}

return await agent(
    "return status",
    {"label": "json", "schema": {"type": "object", "required": ["ok"]}},
)
"""
        runner = FakeRunner(responses=['{"ok": true}'])
        with self.assertRaises(ChildAgentError):
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=runner))
        self.assertEqual(len(runner.requests), 1)

    def test_invalid_structured_schema_fails_before_child_launch(self):
        script = """
meta = {"name": "invalid-schema", "description": "Test workflow"}

return await agent(
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
meta = {"name": "unsupported-options", "description": "Test workflow"}

return await agent("go", {"label": "r", "toolsets": ["web"], "retries": 2})
"""
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=FakeRunner()))
        self.assertIn("unsupported agent() option", str(ctx.exception))
        self.assertIn("toolsets", str(ctx.exception))
        self.assertIn("retries", str(ctx.exception))

    def test_workflow_may_return_without_agent_call(self):
        script = """
meta = {"name": "empty", "description": "Test workflow"}

return "no agents"
"""
        result = run_workflow(script, WorkflowOptions(config=PluginConfig(), child_runner=FakeRunner()))
        self.assertEqual(result.value, "no agents")
        self.assertEqual(result.agent_count, 0)

    def test_direct_agent_failure_raises(self):
        script = """
meta = {"name": "direct-failure", "description": "Test workflow"}

return await agent("fail", {"label": "direct"})
"""
        with self.assertRaises(ChildAgentError) as ctx:
            run_workflow(script, WorkflowOptions(child_runner=FailingRunner()))
        self.assertIn("failed:direct", str(ctx.exception))

    def test_pipeline_agent_failure_drops_item_and_skips_remaining_stages(self):
        script = """
meta = {"name": "pipeline-failure", "description": "Test workflow"}

return await pipeline(
    ["a", "b"],
    lambda item, original, index: agent(item, {"label": item}),
    lambda prior, original, index: agent("after-" + original, {"label": "after-" + original}),
)
"""

        class HalfFailingRunner(ChildAgentRunner):
            def __init__(self):
                self.labels = []

            def run(self, request):
                self.labels.append(request.label)
                if request.label == "a":
                    raise RuntimeError("no a")
                return request.label

        runner = HalfFailingRunner()
        result = run_workflow(script, WorkflowOptions(child_runner=runner))
        self.assertEqual(result.value, [None, "after-b"])
        self.assertNotIn("after-a", runner.labels)

    def test_intentionally_skipped_agent_returns_none(self):
        script = """
meta = {"name": "skip", "description": "Test workflow"}

return await agent("skip me", {"label": "skipped"})
"""
        result = run_workflow(script, WorkflowOptions(child_runner=SkippingRunner()))
        self.assertIsNone(result.value)
        agent_state = result.state.snapshot()["agents"][0]
        self.assertEqual(agent_state["status"], "skipped")
        self.assertEqual(agent_state["error"], "")

    def test_unknown_agent_type_raises_before_child_launch(self):
        script = """
meta = {"name": "missing-agent-type", "description": "Test workflow"}

return await agent("work", {"agentType": "definitely-missing"})
"""
        runner = FakeRunner()
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(child_runner=runner))
        self.assertIn(
            "agent({agentType}): agent type 'definitely-missing' not found",
            str(ctx.exception),
        )
        self.assertIn("Available agents:", str(ctx.exception))
        self.assertEqual(runner.requests, [])

    def test_agent_type_inherit_model_reaches_child_as_no_override(self):
        script = """
meta = {"name": "inherit-agent-model", "description": "Test workflow"}

return await agent("work", {"agentType": "planner"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "planner.md").write_text(
                "---\nname: planner\nmodel: inherit\n---\n\nPlan carefully.\n",
                encoding="utf-8",
            )
            runner = FakeRunner()
            run_workflow(script, WorkflowOptions(cwd=tmp, child_runner=runner))

        self.assertEqual(len(runner.requests), 1)
        self.assertIsNone(runner.requests[0].model)

    def test_phase_model_applies_to_agent_without_explicit_model(self):
        script = """
meta = {
    "name": "phase-model",
    "description": "Test workflow",
    "phases": [{"title": "Search", "model": "sonnet"}],
}

phase("Search")
return await agent("work")
"""
        runner = FakeRunner()
        run_workflow(script, WorkflowOptions(child_runner=runner))

        self.assertEqual(len(runner.requests), 1)
        self.assertEqual(runner.requests[0].phase, "Search")
        self.assertEqual(runner.requests[0].model, "sonnet")

    def test_agent_phase_option_uses_matching_phase_model(self):
        script = """
meta = {
    "name": "opts-phase-model",
    "description": "Test workflow",
    "phases": [{"title": "Verify", "model": "haiku"}],
}

return await agent("work", {"phase": "Verify"})
"""
        runner = FakeRunner()
        run_workflow(script, WorkflowOptions(child_runner=runner))

        self.assertEqual(len(runner.requests), 1)
        self.assertEqual(runner.requests[0].phase, "Verify")
        self.assertEqual(runner.requests[0].model, "haiku")

    def test_agent_model_overrides_phase_model(self):
        script = """
meta = {
    "name": "explicit-model",
    "description": "Test workflow",
    "phases": [{"title": "Search", "model": "sonnet"}],
}

phase("Search")
return await agent("work", {"model": "opus"})
"""
        runner = FakeRunner()
        run_workflow(script, WorkflowOptions(child_runner=runner))

        self.assertEqual(len(runner.requests), 1)
        self.assertEqual(runner.requests[0].phase, "Search")
        self.assertEqual(runner.requests[0].model, "opus")

    def test_public_isolation_only_accepts_worktree(self):
        script = """
meta = {"name": "strict-isolation", "description": "Test workflow"}

return await agent("work", {"isolation": "shared"})
"""
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(child_runner=FakeRunner()))
        self.assertIn("isolation must be 'worktree'", str(ctx.exception))

    def test_log_requires_string(self):
        script = """
meta = {"name": "strict-log", "description": "Test workflow"}

log({"not": "text"})
return None
"""
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(child_runner=FakeRunner()))
        self.assertIn("log() expects a string", str(ctx.exception))

    def test_removed_script_globals_are_unavailable(self):
        for name, script_line in (
            ("cwd", "return cwd"),
            ("print", 'print("no")'),
            ("set", "return set([1])"),
        ):
            with self.subTest(name=name):
                script = f'''
meta = {{"name": "no-{name}", "description": "Test workflow"}}

{script_line}
'''
                with self.assertRaises(NameError):
                    run_workflow(script, WorkflowOptions(child_runner=FakeRunner()))

    def test_workflow_helper_shares_global_agent_sequence_and_snapshot_tree(self):
        parent = """
meta = {"name": "parent", "description": "Test workflow", "phases": [{"title": "Root"}]}

phase("Root")
first = await agent("root", {"label": "root"})
child = await workflow({"scriptPath": args["child"]})
last = await agent("after", {"label": "after"})
return [first, child, last]
"""
        child = """
meta = {"name": "child", "description": "Test workflow", "phases": [{"title": "Child"}]}

phase("Child")
return await agent("child", {"label": "child"})
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
meta = {"name": "budget", "description": "Test workflow"}

await agent("a", {"label": "a"})
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
meta = {"name": "budget-stop", "description": "Test workflow"}

await agent("a", {"label": "a"})
return await agent("b", {"label": "b"})
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
meta = {"name": "budget-meta", "description": "Test workflow", "token_budget": 100}

await agent("a", {"label": "a"})
return {"total": budget.total, "remaining": budget.remaining()}
"""
        result = run_workflow(
            script,
            WorkflowOptions(config=PluginConfig(), child_runner=TokenRunner(tokens=40)),
        )

        self.assertIsNone(result.value["total"])
        self.assertEqual(result.value["remaining"], float("inf"))

    def test_workflow_helper_nesting_is_one_level(self):
        parent = """
meta = {"name": "parent", "description": "Test workflow"}

return await workflow({"scriptPath": args["child"]}, args)
"""
        child = """
meta = {"name": "child", "description": "Test workflow"}

return await workflow({"scriptPath": args["grand"]})
"""
        grand = """
meta = {"name": "grand", "description": "Test workflow"}

return await agent("grand")
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

    def test_workflow_helper_rejects_inline_script_reference(self):
        script = """
meta = {"name": "strict-nested-ref", "description": "Test workflow"}

return await workflow({"script": "meta = {}"})
"""
        with self.assertRaises(Exception) as ctx:
            run_workflow(script, WorkflowOptions(child_runner=FakeRunner()))
        self.assertIn("workflow() expects a non-empty workflow name or", str(ctx.exception))

    def test_named_nested_workflow_uses_parent_store(self):
        parent = """
meta = {"name": "parent-store", "description": "Test workflow"}

return await workflow("private-child")
"""
        child = """
meta = {"name": "private-child", "description": "Test workflow"}

return await agent("child", {"label": "private-child"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp) / "custom-store")
            (store.workflows_dir / "private-child.py").write_text(child, encoding="utf-8")
            runner = FakeRunner()
            result = run_workflow(
                parent,
                WorkflowOptions(
                    cwd=tmp,
                    child_runner=runner,
                    store=store,
                ),
            )
        self.assertEqual(result.value, "private-child:child")

    def test_unknown_nested_workflow_reports_available_names(self):
        script = """
meta = {"name": "unknown-child", "description": "Test workflow"}

return await workflow("missing-child")
"""
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp) / "custom-store")
            with self.assertRaises(Exception) as ctx:
                run_workflow(
                    script,
                    WorkflowOptions(
                        cwd=tmp,
                        child_runner=FakeRunner(),
                        store=store,
                    ),
                )

        self.assertIn(
            "workflow('missing-child'): no workflow with that name. Available: none",
            str(ctx.exception),
        )

    def test_resume_cache_ignores_label_and_phase(self):
        first_script = """
meta = {"name": "cache-display-one", "description": "Test workflow"}

return await agent("same prompt", {"label": "first", "phase": "One"})
"""
        second_script = """
meta = {"name": "cache-display-two", "description": "Test workflow"}

return await agent("same prompt", {"label": "second", "phase": "Two"})
"""
        first_runner = FakeRunner()
        first_cache = ResumeCache()
        first = run_workflow(
            first_script,
            WorkflowOptions(child_runner=first_runner, resume_cache=first_cache),
        )
        second_runner = FakeRunner()
        second = run_workflow(
            second_script,
            WorkflowOptions(
                child_runner=second_runner,
                resume_cache=ResumeCache(first_cache.current),
            ),
        )
        self.assertEqual(second.value, first.value)
        self.assertEqual(second_runner.requests, [])

    def test_resume_cache_invalidates_when_agent_type_content_changes(self):
        script = """
meta = {"name": "cache-agent-type", "description": "Test workflow"}

return await agent("same prompt", {"agentType": "researcher"})
"""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            agent_file = agent_dir / "researcher.md"
            agent_file.write_text("Version one.", encoding="utf-8")
            first_cache = ResumeCache()
            run_workflow(
                script,
                WorkflowOptions(
                    cwd=tmp,
                    child_runner=FakeRunner(),
                    resume_cache=first_cache,
                ),
            )
            agent_file.write_text("Version two.", encoding="utf-8")
            second_runner = FakeRunner()
            run_workflow(
                script,
                WorkflowOptions(
                    cwd=tmp,
                    child_runner=second_runner,
                    resume_cache=ResumeCache(first_cache.current),
                ),
            )
        self.assertEqual(len(second_runner.requests), 1)

    def test_resume_cache_does_not_cross_workspaces(self):
        script = """
meta = {"name": "cache-workspace", "description": "Test workflow"}

return await agent("same prompt")
"""
        with tempfile.TemporaryDirectory() as first_cwd, tempfile.TemporaryDirectory() as second_cwd:
            first_cache = ResumeCache()
            run_workflow(
                script,
                WorkflowOptions(
                    cwd=first_cwd,
                    child_runner=FakeRunner(),
                    resume_cache=first_cache,
                ),
            )
            second_runner = FakeRunner()
            run_workflow(
                script,
                WorkflowOptions(
                    cwd=second_cwd,
                    child_runner=second_runner,
                    resume_cache=ResumeCache(first_cache.current),
                ),
            )
        self.assertEqual(len(second_runner.requests), 1)


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
meta = {"name": "while-ok", "description": "Test workflow"}

results = []
i = 0
while i < 3:
    results.append(await agent("x" + str(i)))
    i = i + 1
return results
"""
        runner = FakeRunner()
        result = run_workflow(script, WorkflowOptions(child_runner=runner))
        self.assertEqual(len(result.value), 3)
        self.assertEqual([r.prompt for r in runner.requests], ["x0", "x1", "x2"])

    def test_try_except_handles_recoverable_error(self):
        script = """
meta = {"name": "try-ok", "description": "Test workflow"}

try:
    y = 1 / 0
except Exception:
    y = "caught"
await agent("a")
return y
"""
        result = run_workflow(script, WorkflowOptions(child_runner=TokenRunner(tokens=1)))
        self.assertEqual(result.value, "caught")

    def test_except_exception_cannot_swallow_budget_halt(self):
        # A while loop that catches Exception around agent() must STILL halt when
        # the token budget is exhausted — the halt is BaseException, not caught.
        script = """
meta = {"name": "no-swallow", "description": "Test workflow"}

out = []
while True:
    try:
        out.append(await agent("x"))
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
meta = {"name": "spin", "description": "Test workflow"}

await agent("a")
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

    def test_compute_only_for_loop_is_bounded_by_iteration_cap(self):
        script = """
meta = {"name": "for-spin", "description": "Test workflow"}

for i in range(1000000):
    value = i
return value
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
