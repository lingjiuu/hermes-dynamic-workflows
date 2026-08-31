from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.run.manager import _approve_launch, _approve_verified_launch

META = {"name": "demo", "description": "a workflow"}


@contextmanager
def fake_approval(
    *,
    gateway=False,
    gateway_choice="once",
    notify_present=True,
    cli_choice="once",
    legacy_wait=False,
    gateway_timeout=1,
    install_touch=False,
):
    """Inject fake tools.approval / tools.terminal_tool so _approve_launch's
    channel logic can be exercised without the real Hermes engine."""
    appr = types.ModuleType("tools.approval")
    appr._is_gateway_approval_context = lambda: gateway
    appr.get_current_session_key = lambda default="default": "sess"
    appr._lock = threading.RLock()
    appr._gateway_queues = {}
    appr._get_approval_config = lambda: {"gateway_timeout": gateway_timeout}
    appr._fire_approval_hook = lambda *a, **k: None

    class ApprovalEntry:
        def __init__(self, data):
            self.event = threading.Event()
            self.data = data
            self.result = None

    appr._ApprovalEntry = ApprovalEntry

    def notify(*a, **k):
        if a:
            appr._last_notified.append(a[0])
        queue = appr._gateway_queues.get("sess", [])
        if queue and gateway_choice != "timeout":
            queue[-1].result = gateway_choice
            queue[-1].event.set()

    appr._gateway_notify_cbs = {"sess": notify} if notify_present else {}
    appr._last_notified = []
    if legacy_wait:
        appr._await_gateway_decision = lambda sk, cb, data, surface=None: {
            "resolved": True,
            "choice": gateway_choice,
        }
    appr.prompt_dangerous_approval = lambda command, description, approval_callback=None: cli_choice

    term = types.ModuleType("tools.terminal_tool")
    term._get_approval_callback = lambda: None

    pkg = types.ModuleType("tools")
    pkg.approval = appr
    pkg.terminal_tool = term
    modules = {"tools": pkg, "tools.approval": appr, "tools.terminal_tool": term}
    if install_touch:
        env_pkg = types.ModuleType("tools.environments")
        base = types.ModuleType("tools.environments.base")
        base.touch_activity_if_due = lambda state, label: (state["start"], state["last_touch"])
        pkg.environments = env_pkg
        env_pkg.base = base
        modules["tools.environments"] = env_pkg
        modules["tools.environments.base"] = base

    with patch.dict(sys.modules, modules):
        yield appr


class LaunchApprovalConfigTests(unittest.TestCase):
    def test_default_is_on(self):
        self.assertTrue(PluginConfig().require_launch_approval)


class LaunchApprovalDecisionTests(unittest.TestCase):
    def test_off_always_approves(self):
        approved, _ = _approve_launch(META, PluginConfig(require_launch_approval=False), None)
        self.assertTrue(approved)

    def test_gateway_approve(self):
        with fake_approval(gateway=True, gateway_choice="once"):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertTrue(approved)

    def test_gateway_deny(self):
        with fake_approval(gateway=True, gateway_choice="deny"):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("denied", reason)

    def test_gateway_legacy_wait_compat(self):
        with fake_approval(gateway=True, gateway_choice="once", legacy_wait=True):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertTrue(approved)

    def test_gateway_no_channel_denies(self):
        with fake_approval(gateway=True, notify_present=False):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("no gateway approval channel", reason)

    def test_gateway_timeout_activity_state_is_initialized(self):
        with fake_approval(gateway=True, gateway_choice="timeout", gateway_timeout=0.01, install_touch=True):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("timed out", reason)

    def test_cli_approve(self):
        with fake_approval(gateway=False, cli_choice="once"), \
                patch.dict(os.environ, {"HERMES_INTERACTIVE": "1"}):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertTrue(approved)

    def test_cli_deny(self):
        with fake_approval(gateway=False, cli_choice="deny"), \
                patch.dict(os.environ, {"HERMES_INTERACTIVE": "1"}):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)

    def test_headless_no_channel_denies(self):
        env = {k: v for k, v in os.environ.items() if k != "HERMES_INTERACTIVE"}
        with fake_approval(gateway=False), patch.dict(os.environ, env, clear=True):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("no interactive channel", reason)

    def test_verified_launch_approval_remains_mandatory_when_config_flag_is_off(self):
        env = {k: v for k, v in os.environ.items() if k != "HERMES_INTERACTIVE"}
        with fake_approval(gateway=False), patch.dict(os.environ, env, clear=True):
            approved, reason = _approve_verified_launch(
                META,
                approval_id="wfa_test-1",
                execution_digest="a" * 64,
                authority_preset_id="dws-read-only-v1",
                enforced_max_scale=4,
            )
        self.assertFalse(approved)
        self.assertIn("approval required", reason)

    def test_verified_gateway_approval_binds_digest_preset_and_scale(self):
        with fake_approval(gateway=True, gateway_choice="once") as approval:
            approved, _ = _approve_verified_launch(
                META,
                approval_id="wfa_test-2",
                execution_digest="b" * 64,
                authority_preset_id="dws-read-only-v1",
                enforced_max_scale=7,
            )

        self.assertTrue(approved)
        queued = approval._gateway_queues.get("sess", [])
        self.assertEqual(queued, [])
        # notify() receives the same data object placed in the queue entry.
        # The fake helper resolves immediately, so inspect the last entry data.
        last = approval._last_notified[-1]
        self.assertEqual(last["approval_id"], "wfa_test-2")
        self.assertEqual(last["execution_digest"], "b" * 64)
        self.assertEqual(last["authority_preset_id"], "dws-read-only-v1")
        self.assertEqual(last["enforced_max_scale"], 7)
        self.assertIn("dws-read-only-v1", last["description"])
        self.assertIn("bbbbbbbb", last["description"])


if __name__ == "__main__":
    unittest.main()
