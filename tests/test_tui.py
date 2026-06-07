from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.storage.store import WorkflowStore
from hermes_dynamic_workflows.tui.app import TuiController
from hermes_dynamic_workflows.tui.model import (
    PhaseView,
    WorkflowRepository,
    _JsonlTailReader,
    group_sessions,
)
from hermes_dynamic_workflows.tui.render import RenderState, _display_width, render_screen


class FakeControlClient:
    def __init__(self):
        self.requests = []

    def request(self, **kwargs):
        self.requests.append(kwargs)
        action = kwargs["action"]
        response = {"ok": True, "message": f"{action} accepted"}
        if action == "restart":
            response["newRunId"] = "wf_fake-completed"
        return response


class TuiTests(unittest.TestCase):
    def test_repository_reads_run_journal_and_live_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _fake_store(Path(tmp))
            repository = WorkflowRepository(store)
            running = repository.detail("wf_fake-running")
            hydrated = repository.hydrate_agent_activity(running, phase_index=0, agent_index=0)

        self.assertEqual(running.name, "dynamic-workflow-research")
        self.assertEqual([phase.title for phase in running.phases], ["Search", "Summarize"])
        self.assertEqual(hydrated.agents[0].activity[-2:], ('WebSearch({"query":"dynamic workflows"})', 'Read({"path":"paper.pdf"})'))
        self.assertEqual(running.agents[1].activity, ("Agent started",))

    def test_repository_reads_live_tool_activity_from_journal_without_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _fake_store(Path(tmp))
            record = store.load_run("wf_fake-running")
            assert record is not None
            with Path(record["journalFile"]).open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "activity",
                            "agentId": "2",
                            "activity": 'terminal({"command":"pwd"})',
                        }
                    )
                    + "\n"
                )
            store.save_run(record)
            running = WorkflowRepository(store).detail("wf_fake-running")

        self.assertIn('terminal({"command":"pwd"})', running.agents[1].activity)

    def test_renders_claude_style_list_workflow_and_agent_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = WorkflowRepository(_fake_store(Path(tmp)))
            summaries = repository.load()  # list view uses lightweight summaries
            full = repository.detail("wf_fake-running")  # detail views use the full view
            full = repository.hydrate_agent_activity(full, phase_index=0, agent_index=0)
            list_text = "\n".join(
                render_screen(
                    summaries,
                    RenderState(expanded=frozenset({"sess-live"})),
                    width=120,
                    height=28,
                )
            )
            workflow_text = "\n".join(
                render_screen([full], RenderState(view="workflow"), width=120, height=28)
            )
            agent_text = "\n".join(
                render_screen([full], RenderState(view="agent"), width=120, height=32)
            )

        self.assertIn("Dynamic workflows", list_text)
        self.assertIn("2 sessions · 1 running", list_text)
        self.assertIn("live-project", list_text)  # session group header (cwd basename)
        self.assertIn("dynamic-workflow-research", list_text)  # workflow row in expanded group
        self.assertIn("Phases", workflow_text)
        self.assertIn("Search · 2 agents", workflow_text)
        self.assertIn("search:claude-articles", workflow_text)
        self.assertIn("Prompt ·", agent_text)
        self.assertIn("Activity · last 2 of 2", agent_text)
        self.assertIn("WebSearch", agent_text)
        self.assertIn("Still running...", agent_text)

    def test_agent_row_and_detail_show_duration_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = WorkflowRepository(_fake_store(Path(tmp))).detail("wf_fake-completed")
        timed = replace(base.agents[0], duration_seconds=76.0)
        wf = replace(base, agents=(timed,), phases=(PhaseView(title="Search", agents=(timed,)),))
        workflow_text = "\n".join(render_screen([wf], RenderState(view="workflow"), width=110, height=24))
        agent_text = "\n".join(render_screen([wf], RenderState(view="agent"), width=110, height=24))
        self.assertIn("· 1m 16s", workflow_text)
        self.assertIn("· 1m 16s", agent_text)

    def test_agent_detail_scrolls_with_indicator(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = WorkflowRepository(_fake_store(Path(tmp))).detail("wf_fake-completed")
        agent = replace(base.agents[0], outcome="\n".join(f"line {i}" for i in range(60)))
        wf = replace(base, agents=(agent,), phases=(PhaseView(title="Search", agents=(agent,)),))
        top = "\n".join(render_screen([wf], RenderState(view="agent"), width=100, height=20))
        self.assertRegex(top, r"1-\d+ of \d+ ↓")
        self.assertIn("line 0", top)
        bottom = "\n".join(render_screen([wf], RenderState(view="agent", detail_scroll=999), width=100, height=20))
        self.assertRegex(bottom, r"of \d+ ↑")
        self.assertIn("line 59", bottom)
        self.assertNotIn("line 0", bottom)

    def test_repository_caches_terminal_runs_between_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _fake_store(Path(tmp))
            calls: list[str] = []
            original = store.load_run
            store.load_run = lambda run_id: (calls.append(run_id), original(run_id))[1]
            repository = WorkflowRepository(store)
            repository.load()
            repository.load()
            # Terminal runs are parsed once and then served from cache; active runs
            # are re-parsed each load so their live duration keeps ticking.
            self.assertEqual(calls.count("wf_fake-completed"), 1)
            self.assertEqual(calls.count("wf_fake-running"), 2)
            record = original("wf_fake-completed")
            assert record is not None
            calls.clear()
            store.save_run(record)
            repository.load()
            self.assertIn("wf_fake-completed", calls)

    def test_load_returns_summaries_and_detail_builds_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = WorkflowRepository(_fake_store(Path(tmp)))
            summaries = repo.load()
            live = next(workflow for workflow in summaries if workflow.run_id == "wf_fake-running")
            # summary: no per-agent detail or phases built...
            self.assertEqual(live.agents, ())
            self.assertEqual(live.phases, ())
            # ...but the counts the list needs are still present
            self.assertEqual(live.agent_count, 2)
            self.assertGreater(live.tokens, 0)
            # detail() builds the full view on demand
            full = repo.detail("wf_fake-running")
            self.assertEqual(len(full.agents), 2)
            self.assertEqual([phase.title for phase in full.phases], ["Search", "Summarize"])
            self.assertEqual(full.agent_count, 2)

    def test_world_version_changes_when_a_run_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _fake_store(Path(tmp))
            repo = WorkflowRepository(store)
            before = repo.world_version()
            time.sleep(0.01)
            record = store.load_run("wf_fake-running")
            assert record is not None
            store.save_run(record)  # tmp+rename bumps the runs-dir mtime
            self.assertNotEqual(repo.world_version(), before)

    def test_right_arrow_on_run_opens_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = TuiController(WorkflowRepository(_fake_store(Path(tmp))))
            controller.refresh()
            controller.handle_key("down")            # session header -> run row
            self.assertEqual(controller._cursor_item()[0], "run")
            controller.handle_key("right")           # → drills into the workflow
            self.assertEqual(controller.state.view, "workflow")

    def test_workflow_phase_arrows_move_even_with_summary_current_run(self):
        # Regression: phase count must come from the detail view, not the summary.
        with tempfile.TemporaryDirectory() as tmp:
            controller = TuiController(WorkflowRepository(_fake_store(Path(tmp))))
            controller.refresh()
            controller.handle_key("down")
            controller.handle_key("enter")           # open workflow, focus = phases
            self.assertEqual(controller.state.phase_index, 0)
            controller.handle_key("down")            # move down the Phases pane
            self.assertEqual(controller.state.phase_index, 1)

    def test_agent_outcome_uses_full_journal_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _fake_store(Path(tmp))
            record = store.load_run("wf_fake-running")
            assert record is not None
            full_result = "FULL OUTCOME LINE\n" * 50
            with Path(record["journalFile"]).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "result", "agentId": "1", "result": full_result}) + "\n")
            store.save_run(record)
            view = WorkflowRepository(store).detail("wf_fake-running")
        agent = next(item for item in view.agents if item.id == "1")
        self.assertEqual(agent.outcome, full_result)  # full journal result, not the 180-char preview

    def test_controller_scroll_keys_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = TuiController(WorkflowRepository(_fake_store(Path(tmp))))
            controller.refresh()
            controller.handle_key("down")   # session header -> its workflow row
            controller.handle_key("enter")  # open workflow (focus: phases)
            controller.handle_key("right")  # focus the agents pane
            controller.handle_key("enter")  # open agent
            self.assertEqual(controller.state.view, "agent")
            controller.handle_key("j")
            controller.handle_key("j")
            self.assertEqual(controller.state.detail_scroll, 2)
            # The selected agent's detail fits this viewport, so frame() clamps to 0.
            controller.frame(120, 40)
            self.assertEqual(controller.state.detail_scroll, 0)
            controller.handle_key("j")
            self.assertEqual(controller.state.detail_scroll, 1)
            controller.handle_key("down")
            self.assertEqual(controller.state.detail_scroll, 0)

    def test_controller_navigation_save_and_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _fake_store(Path(tmp))
            controller = TuiController(WorkflowRepository(store))
            controller.refresh()

            controller.handle_key("down")   # session header -> its workflow row
            controller.handle_key("enter")
            self.assertEqual(controller.state.view, "workflow")
            self.assertEqual(controller.state.focus, "phases")
            controller.handle_key("right")   # step into the agents pane (no drill yet)
            self.assertEqual(controller.state.view, "workflow")
            self.assertEqual(controller.state.focus, "agents")
            controller.handle_key("right")   # now drill into the agent
            self.assertEqual(controller.state.view, "agent")
            controller.handle_key("down")
            self.assertEqual(controller.state.agent_index, 1)
            controller.handle_key("esc")
            self.assertEqual(controller.state.view, "workflow")
            self.assertEqual(controller.state.focus, "agents")
            controller.handle_key("left")    # back to the phases pane
            self.assertEqual(controller.state.focus, "phases")
            controller.handle_key("s")
            self.assertIn("Saved to", controller.state.message)
            self.assertTrue((store.exports_dir / f"{controller.current_run.run_id}.md").is_file())

            record = store.load_run("wf_fake-running")
            assert record is not None
            record["status"] = "completed"
            record["workflow"]["agents"][0]["status"] = "done"
            record["workflow"]["agents"][0]["tokens"] = 44846
            record["workflow"]["totals"] = {
                "agents": 2,
                "done": 1,
                "running": 1,
                "tokens": 56846,
                "tool_calls": 15,
            }
            store.save_run(record)
            controller.refresh()

            self.assertEqual(controller.current_run.run_id, "wf_fake-running")
            self.assertEqual(controller.current_run.status, "completed")
            self.assertEqual(controller.current_run.tokens, 56846)

    def test_rendered_panels_handle_wide_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflows = WorkflowRepository(_fake_store(Path(tmp))).load()
            lines = render_screen(
                workflows,
                RenderState(view="workflow"),
                width=88,
                height=24,
            )

        self.assertTrue(all(_display_width(line) <= 88 for line in lines))

    def test_rendered_views_keep_selected_items_and_footer_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = WorkflowRepository(_fake_store(Path(tmp))).detail("wf_fake-running")
        many_workflows = [
            replace(base, run_id=f"wf-{index}", name=f"run-{index}", description=f"run-{index}")
            for index in range(30)
        ]
        # all 30 share base's session; expand it and put the cursor on run-29 (item 30)
        list_lines = render_screen(
            many_workflows,
            RenderState(expanded=frozenset({base.session_id}), list_cursor=30),
            width=100,
            height=12,
        )

        phases = tuple(
            PhaseView(title=f"phase-{index}", agents=base.agents)
            for index in range(20)
        )
        workflow_with_phases = replace(base, phases=phases)
        workflow_lines = render_screen(
            [workflow_with_phases],
            RenderState(view="workflow", phase_index=19),
            width=100,
            height=14,
        )

        agents = tuple(
            replace(base.agents[0], id=str(index), label=f"agent-{index}")
            for index in range(20)
        )
        workflow_with_agents = replace(
            base,
            agents=agents,
            phases=(PhaseView(title="agents", agents=agents),),
        )
        agent_lines = render_screen(
            [workflow_with_agents],
            RenderState(view="agent", agent_index=19),
            width=100,
            height=14,
        )

        self.assertIn("run-29", "\n".join(list_lines))
        self.assertNotIn("run-0", "\n".join(list_lines))
        self.assertIn("Esc to close", list_lines[-1])
        self.assertIn("phase-19", "\n".join(workflow_lines))
        self.assertNotIn("phase-0 ", "\n".join(workflow_lines))
        self.assertIn("agent-19", "\n".join(agent_lines))
        self.assertNotIn("agent-0 ", "\n".join(agent_lines))

    def test_non_tty_command_prints_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fake_store(Path(tmp))
            env = dict(os.environ)
            env["HERMES_DYNAMIC_WORKFLOWS_HOME"] = tmp
            result = subprocess.run(
                [sys.executable, "-m", "hermes_dynamic_workflows.tui.app"],
                cwd=Path(__file__).resolve().parent.parent,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("Dynamic workflows", result.stdout)
        self.assertIn("dynamic-workflow-research", result.stdout)

    def test_non_tty_command_respects_hermes_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fake_store(Path(tmp) / "dynamic-workflows")
            env = dict(os.environ)
            env.pop("HERMES_DYNAMIC_WORKFLOWS_HOME", None)
            env["HERMES_HOME"] = tmp
            result = subprocess.run(
                [sys.executable, "-m", "hermes_dynamic_workflows.tui.app"],
                cwd=Path(__file__).resolve().parent.parent,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("dynamic-workflow-research", result.stdout)

    def test_jsonl_reader_caches_stable_files_and_reads_bounded_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.jsonl"
            path.write_text(
                "\n".join(json.dumps({"index": index, "text": "x" * 80}) for index in range(20))
                + "\n",
                encoding="utf-8",
            )
            reader = _JsonlTailReader(max_bytes=300)
            first = reader.read(path)
            second = reader.read(path)
            self.assertIs(first, second)
            self.assertLess(len(first), 20)
            self.assertEqual(first[-1]["index"], 19)

            path.write_text(json.dumps({"index": 99}) + "\n", encoding="utf-8")
            third = reader.read(path)

        self.assertIsNot(third, first)
        self.assertEqual(third, [{"index": 99}])

    def test_transcript_activity_is_loaded_only_for_selected_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = WorkflowRepository(_fake_store(Path(tmp)))
            with patch(
                "hermes_dynamic_workflows.tui.model._read_transcript_activity",
                return_value=["Read(file.py)"],
            ) as read_activity:
                repository.load()
                read_activity.assert_not_called()
                running = repository.detail("wf_fake-running")
                hydrated = repository.hydrate_agent_activity(running, phase_index=0, agent_index=0)

        read_activity.assert_called_once()
        self.assertEqual(hydrated.agents[0].activity, ("Read(file.py)",))

    def test_group_sessions_buckets_drops_session_less_and_marks_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflows = WorkflowRepository(_fake_store(Path(tmp))).load()
        groups = group_sessions(workflows)
        self.assertEqual({group.key for group in groups}, {"sess-live", "sess-done"})
        # the session with a running run sorts first and is flagged current
        self.assertEqual(groups[0].key, "sess-live")
        self.assertTrue(groups[0].is_current)
        self.assertFalse(groups[1].is_current)
        self.assertEqual(groups[0].project, "live-project")
        self.assertEqual(groups[0].running, 1)
        # a run without a session id is dropped from the grouped view
        live = next(workflow for workflow in workflows if workflow.session_id == "sess-live")
        orphan = replace(live, record={**live.record, "workflowSessionId": ""})
        self.assertEqual(len(group_sessions(workflows + [orphan])), 2)

    def test_controller_accordion_expand_collapse_and_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = TuiController(WorkflowRepository(_fake_store(Path(tmp))))
            controller.refresh()
            # current (running) session expanded by default, cursor on its header
            self.assertIn("sess-live", controller.state.expanded)
            self.assertEqual(controller._cursor_item()[0], "group")
            controller.handle_key("left")   # collapse current
            self.assertNotIn("sess-live", controller.state.expanded)
            controller.handle_key("right")  # expand again
            self.assertIn("sess-live", controller.state.expanded)
            controller.handle_key("down")   # onto the workflow row
            self.assertEqual(controller._cursor_item()[0], "run")
            controller.handle_key("enter")  # open it
            self.assertEqual(controller.state.view, "workflow")
            self.assertEqual(controller.current_run.session_id, "sess-live")
            controller.handle_key("esc")    # back to the list, cursor still on the run
            controller.handle_key("left")   # collapse from a run -> cursor jumps to header
            self.assertNotIn("sess-live", controller.state.expanded)
            self.assertEqual(controller._cursor_item()[0], "group")

    def test_controller_sends_stop_pause_resume_and_restart_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = FakeControlClient()
            store = _fake_store(Path(tmp))
            controller = TuiController(WorkflowRepository(store, control_client=control))
            controller.refresh()
            controller.handle_key("down")   # select the running workflow inside its session

            controller.handle_key("p")
            controller.handle_key("x")
            controller.handle_key("r")

            record = store.load_run("wf_fake-running")
            assert record is not None
            record["status"] = "paused"
            store.save_run(record)
            controller.refresh()
            controller.handle_key("p")   # cursor still on the (now paused) run -> resume

        self.assertEqual([request["action"] for request in control.requests], ["pause", "stop", "restart", "resume"])
        self.assertEqual(control.requests[0]["owner"], "fake-control-owner")
        self.assertIn("resume accepted", controller.state.message)


def _fake_store(root: Path) -> WorkflowStore:
    store = WorkflowStore(root)
    transcript_dir = root / "projects" / "-fake-project" / "fake-session" / "subagents" / "workflows" / "wf_fake-running"
    transcript_dir.mkdir(parents=True)
    journal = transcript_dir / "journal.jsonl"
    journal.write_text(
        "\n".join(
            [
                json.dumps({"type": "started", "agentId": "1"}),
                json.dumps({"type": "started", "agentId": "2"}),
                json.dumps({"type": "result", "agentId": "3", "result": "summary"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    transcript = transcript_dir / "agent-search-1.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "WebSearch",
                                        "arguments": '{"query":"dynamic workflows"}',
                                    }
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Read",
                                    "input": {"path": "paper.pdf"},
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    running = {
        "runId": "wf_fake-running",
        "taskId": "wgfake01",
        "workflowSessionId": "sess-live",
        "cwd": "/work/live-project",
        "controlOwner": "fake-control-owner",
        "status": "running",
        "createdAt": (datetime.now(timezone.utc) - timedelta(seconds=36)).isoformat(),
        "startedAt": (datetime.now(timezone.utc) - timedelta(seconds=35)).isoformat(),
        "summary": "并行搜集 Claude dynamic workflow 文章和相关学术论文，最后汇总",
        "journalFile": str(journal),
        "transcriptDir": str(transcript_dir),
        "workflow": {
            "meta": {
                "name": "dynamic-workflow-research",
                "description": "并行搜集 Claude dynamic workflow 文章和相关学术论文，最后汇总",
            },
            "phases": [{"title": "Search"}, {"title": "Summarize"}],
            "current_phase": "Search",
            "duration_seconds": 35,
            "agents": [
                {
                    "id": 1,
                    "label": "search:claude-articles",
                    "phase": "Search",
                    "status": "running",
                    "prompt": "搜索 Claude Code dynamic workflow 官方文章并提炼关键内容。",
                    "model": "Sonnet 4.6",
                    "tokens": 12100,
                    "tool_calls": 7,
                    "transcript_path": str(transcript),
                },
                {
                    "id": 2,
                    "label": "search:academic-papers",
                    "phase": "Search",
                    "status": "running",
                    "prompt": "搜索 dynamic workflow 理论相关论文。",
                    "model": "Sonnet 4.6",
                    "tokens": 11000,
                    "tool_calls": 6,
                },
            ],
            "children": [],
            "errors": [],
            "totals": {
                "agents": 2,
                "done": 0,
                "running": 2,
                "tokens": 23100,
                "tool_calls": 13,
            },
        },
    }
    completed = {
        "runId": "wf_fake-completed",
        "taskId": "wgfake02",
        "workflowSessionId": "sess-done",
        "cwd": "/work/done-project",
        "controlOwner": "fake-control-owner",
        "status": "completed",
        "createdAt": "2026-06-05T00:00:00+00:00",
        "startedAt": "2026-06-05T00:00:00+00:00",
        "finishedAt": "2026-06-05T00:02:15+00:00",
        "summary": "完成的研究 workflow",
        "workflow": {
            "meta": {"name": "completed-research", "description": "完成的研究 workflow"},
            "phases": [{"title": "Search"}],
            "duration_seconds": 135,
            "agents": [
                {
                    "id": 3,
                    "label": "synthesis",
                    "phase": "Search",
                    "status": "done",
                    "prompt": "汇总",
                    "result_preview": "已完成汇总。",
                    "tokens": 44846,
                    "tool_calls": 22,
                }
            ],
            "children": [],
            "errors": [],
            "totals": {
                "agents": 1,
                "done": 1,
                "running": 0,
                "tokens": 44846,
                "tool_calls": 22,
            },
        },
    }
    store.save_run(completed)
    store.save_run(running)
    return store


if __name__ == "__main__":
    unittest.main()
