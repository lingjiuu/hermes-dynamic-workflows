from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.engine.context import PauseGate
from hermes_dynamic_workflows.engine.manager import WorkflowRunManager
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentRunner
from hermes_dynamic_workflows.storage.control import ControlClient
from hermes_dynamic_workflows.storage.store import WorkflowStore


SEQUENTIAL_SCRIPT = """
meta = {"name": "controlled", "description": "Exercise workflow controls", "phases": ["Work"]}

phase("Work")
first = await agent("first", {"label": "first"})
second = await agent("second", {"label": "second"})
result = [first, second]
"""


class SequentialRunner(ChildAgentRunner):
    def __init__(self):
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_started = threading.Event()
        self.interrupted = False

    def run(self, request: ChildAgentRequest):
        if request.label == "first":
            self.first_started.set()
            if not self.release_first.wait(timeout=5):
                raise TimeoutError("first agent was not released")
        else:
            self.second_started.set()
        return f"done:{request.label}"

    def interrupt_all(self):
        self.interrupted = True
        self.release_first.set()


class ImmediateRunner(ChildAgentRunner):
    def run(self, request: ChildAgentRequest):
        return f"done:{request.label}"


class ControlTests(unittest.TestCase):
    def test_pause_holds_next_agent_until_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = SequentialRunner()
            store = WorkflowStore(Path(tmp))
            manager = WorkflowRunManager(
                store=store,
                config=PluginConfig(require_launch_approval=False),
                enable_control=True,
            )
            client = ControlClient(store)
            try:
                with patch(
                    "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                    return_value=runner,
                ):
                    record = manager.start_from_params(
                        {"script": SEQUENTIAL_SCRIPT},
                        cwd=tmp,
                        host_session_id="control-session",
                    )
                    self.assertTrue(runner.first_started.wait(timeout=2))
                    paused = client.request(
                        owner=record["controlOwner"],
                        run_id=record["runId"],
                        action="pause",
                        expected_status="running",
                    )
                    self.assertTrue(paused["ok"])
                    self.assertEqual(paused["status"], "paused")

                    runner.release_first.set()
                    self.assertFalse(runner.second_started.wait(timeout=0.3))
                    self.assertEqual(manager.get(record["runId"])["status"], "paused")

                    resumed = client.request(
                        owner=record["controlOwner"],
                        run_id=record["runId"],
                        action="resume",
                        expected_status="paused",
                    )
                    self.assertTrue(resumed["ok"])
                    self.assertTrue(runner.second_started.wait(timeout=2))
                    final = manager.wait(record["runId"], timeout=2)
            finally:
                manager.stop_control_listener()

        self.assertEqual(final["status"], "completed")
        self.assertIsNone(final["result"])

    def test_stop_and_restart_through_control_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            manager = WorkflowRunManager(
                store=store,
                config=PluginConfig(require_launch_approval=False),
                enable_control=True,
            )
            client = ControlClient(store)
            try:
                blocking = SequentialRunner()
                with patch(
                    "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                    return_value=blocking,
                ):
                    active = manager.start_from_params(
                        {"script": SEQUENTIAL_SCRIPT},
                        cwd=tmp,
                        host_session_id="control-session",
                    )
                    self.assertTrue(blocking.first_started.wait(timeout=2))
                    stopped = client.request(
                        owner=active["controlOwner"],
                        run_id=active["runId"],
                        action="stop",
                        expected_status="running",
                    )
                    final = manager.wait(active["runId"], timeout=2)
                self.assertTrue(stopped["ok"])
                self.assertTrue(blocking.interrupted)
                self.assertEqual(final["status"], "stopped")

                with patch(
                    "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                    return_value=ImmediateRunner(),
                ):
                    restarted = client.request(
                        owner=active["controlOwner"],
                        run_id=active["runId"],
                        action="restart",
                        expected_status="stopped",
                    )
                    new_final = manager.wait(restarted["newRunId"], timeout=2)
            finally:
                manager.stop_control_listener()

        self.assertTrue(restarted["ok"])
        self.assertNotEqual(restarted["newRunId"], active["runId"])
        self.assertEqual(new_final["status"], "completed")
        self.assertEqual(new_final["restartedFromRunId"], active["runId"])

    def test_restart_active_workflow_stops_old_run_and_preserves_session_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = SequentialRunner()
            store = WorkflowStore(Path(tmp))
            manager = WorkflowRunManager(
                store=store,
                config=PluginConfig(require_launch_approval=False),
                enable_control=True,
            )
            client = ControlClient(store)
            try:
                with (
                    patch(
                        "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                        return_value=runner,
                    ),
                    patch(
                        "hermes_dynamic_workflows.engine.manager._capture_gateway_session_context",
                        return_value={"platform": "telegram", "session_key": "gateway-session"},
                    ) as capture_context,
                ):
                    active = manager.start_from_params(
                        {"script": SEQUENTIAL_SCRIPT},
                        cwd=tmp,
                        host_session_id="control-session",
                    )
                    self.assertTrue(runner.first_started.wait(timeout=2))
                    restarted = client.request(
                        owner=active["controlOwner"],
                        run_id=active["runId"],
                        action="restart",
                        expected_status="running",
                        wait_seconds=3,
                    )
                    old_final = manager.wait(active["runId"], timeout=2)
                    new_final = manager.wait(restarted["newRunId"], timeout=2)
            finally:
                manager.stop_control_listener()

        self.assertTrue(restarted["ok"])
        self.assertEqual(old_final["status"], "stopped")
        self.assertEqual(new_final["status"], "completed")
        self.assertEqual(capture_context.call_count, 1)
        self.assertEqual(
            manager._runs[restarted["newRunId"]].session_context,
            {"platform": "telegram", "session_key": "gateway-session"},
        )

    def test_control_queue_works_between_two_real_processes(self):
        worker = r"""
import sys
import time
from pathlib import Path
from hermes_dynamic_workflows.storage.control import ControlListener
from hermes_dynamic_workflows.storage.store import WorkflowStore

root = Path(sys.argv[1])
owner = "separate-process-owner"
store = WorkflowStore(root)
listener = ControlListener(
    store=store,
    owner=owner,
    handler=lambda request: {
        "ok": True,
        "action": request["action"],
        "runId": request["runId"],
        "message": "handled in worker process",
    },
)
listener.start()
(root / "ready").write_text(owner, encoding="utf-8")
deadline = time.time() + 10
while time.time() < deadline and not (root / "done").exists():
    time.sleep(0.05)
listener.stop()
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = subprocess.Popen([sys.executable, "-c", worker, tmp])
            try:
                self.assertTrue(_wait_for(root / "ready", timeout=3))
                response = ControlClient(WorkflowStore(root)).request(
                    owner="separate-process-owner",
                    run_id="wf_separate-process",
                    action="pause",
                    wait_seconds=3,
                )
            finally:
                (root / "done").touch()
                process.wait(timeout=3)

        self.assertEqual(process.returncode, 0)
        self.assertTrue(response["ok"])
        self.assertEqual(response["message"], "handled in worker process")

    def test_expected_status_rejects_stale_control_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            manager = WorkflowRunManager(
                store=store,
                config=PluginConfig(require_launch_approval=False),
                enable_control=True,
            )
            try:
                with patch(
                    "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                    return_value=ImmediateRunner(),
                ):
                    record = manager.start_from_params(
                        {"script": SEQUENTIAL_SCRIPT},
                        cwd=tmp,
                        host_session_id="control-session",
                    )
                    final = manager.wait(record["runId"], timeout=2)
                response = ControlClient(store).request(
                    owner=record["controlOwner"],
                    run_id=record["runId"],
                    action="stop",
                    expected_status="running",
                )
            finally:
                manager.stop_control_listener()

        self.assertEqual(final["status"], "completed")
        self.assertFalse(response["ok"])
        self.assertIn("status changed", response["message"])

    def test_completed_workflow_is_not_stoppable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            manager = WorkflowRunManager(
                store=store,
                config=PluginConfig(require_launch_approval=False),
                enable_control=True,
            )
            try:
                with patch(
                    "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                    return_value=ImmediateRunner(),
                ):
                    record = manager.start_from_params(
                        {"script": SEQUENTIAL_SCRIPT},
                        cwd=tmp,
                        host_session_id="control-session",
                    )
                    final = manager.wait(record["runId"], timeout=2)
                response = ControlClient(store).request(
                    owner=record["controlOwner"],
                    run_id=record["runId"],
                    action="stop",
                    expected_status="completed",
                )
            finally:
                manager.stop_control_listener()

        self.assertEqual(final["status"], "completed")
        self.assertFalse(response["ok"])
        self.assertIn("not stoppable", response["message"])

    def test_paused_time_does_not_consume_workflow_deadline(self):
        gate = PauseGate()
        gate.pause()
        result: list[object] = []

        def run() -> None:
            result.append(
                run_workflow(
                    """
meta = {"name": "pause-deadline", "description": "Exercise paused deadline accounting"}
result = await agent("work", {"label": "worker"})
""",
                    WorkflowOptions(
                        config=PluginConfig(workflow_timeout_seconds=0.1),
                        child_runner=ImmediateRunner(),
                        pause_gate=gate,
                    ),
                )
            )

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.3)
        self.assertTrue(thread.is_alive())
        gate.resume()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertIsNone(result[0].value)

    def test_control_listener_failure_does_not_block_workflow_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "hermes_dynamic_workflows.engine.manager.ControlListener.start",
                side_effect=OSError("read-only control directory"),
            ):
                manager = WorkflowRunManager(
                    store=WorkflowStore(Path(tmp)),
                    config=PluginConfig(require_launch_approval=False),
                    enable_control=True,
                )
            with patch(
                "hermes_dynamic_workflows.agent.runner.HermesChildAgentRunner",
                return_value=ImmediateRunner(),
            ):
                record = manager.start_from_params(
                    {"script": SEQUENTIAL_SCRIPT},
                    cwd=tmp,
                    host_session_id="control-session",
                )
                final = manager.wait(record["runId"], timeout=2)

        self.assertIsNone(record["controlOwner"])
        self.assertEqual(final["status"], "completed")


def _wait_for(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


if __name__ == "__main__":
    unittest.main()
