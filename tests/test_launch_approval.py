from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.engine.manager import _approve_launch

META = {"name": "demo", "description": "a workflow"}


@contextmanager
def fake_approval(*, gateway=False, gateway_choice="once", notify_present=True, cli_choice="once"):
    """Inject fake tools.approval / tools.terminal_tool so _approve_launch's
    channel logic can be exercised without the real Hermes engine."""
    appr = types.ModuleType("tools.approval")
    appr._is_gateway_approval_context = lambda: gateway
    appr.get_current_session_key = lambda default="default": "sess"
    appr._gateway_notify_cbs = {"sess": (lambda *a, **k: None)} if notify_present else {}
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

    with patch.dict(sys.modules, {"tools": pkg, "tools.approval": appr, "tools.terminal_tool": term}):
        yield


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

    def test_gateway_no_channel_denies(self):
        with fake_approval(gateway=True, notify_present=False):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("no gateway approval channel", reason)

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


if __name__ == "__main__":
    unittest.main()
