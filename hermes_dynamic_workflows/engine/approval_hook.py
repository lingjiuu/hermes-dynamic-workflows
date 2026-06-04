"""pre_tool_call hook — make child_approval_policy authoritative everywhere.

A workflow child's terminal command normally goes through Hermes'
``check_dangerous_command``, which consults the plugin's per-thread approval
callback ONLY in the CLI-interactive branch. But workflow children run in
detached background threads that don't carry the session's interactive/gateway
context, so in headless (and contextvar-based gateway) Hermes' fallback would
auto-approve (headless) or orphan (gateway) a flagged command — silently
bypassing ``child_approval_policy`` (e.g. a "deny" default that doesn't deny).

This ``pre_tool_call`` hook closes that gap: it runs before Hermes' context
branching and enforces the policy for workflow-child terminal commands, but
only in NON-CLI contexts. In CLI it defers to the working per-thread callback
(returns None) so the policy isn't evaluated twice (which would, e.g., fire a
second ``_smart_approve`` LLM call).

The hook fires for every tool call in the process; non-workflow / non-terminal
calls return immediately, so the parent session and other agents are untouched.
"""

from __future__ import annotations

from typing import Any, Callable

# Child task ids are minted as ``workflow-<uuid>`` by the runner.
WORKFLOW_CHILD_PREFIX = "workflow-"
TERMINAL_TOOLS = {"terminal"}


def _block(description: str) -> dict[str, str]:
    return {
        "action": "block",
        "message": (
            "BLOCKED by dynamic-workflows child_approval_policy: a background "
            f"workflow child agent may not run this flagged command ({description}). "
            "Find an approach that doesn't require it (or set child_approval_policy "
            "to 'smart'/'approve' in plugin config)."
        ),
    }


def evaluate_command_gate(
    command: str,
    *,
    classify: Callable[[str], tuple],
    allowlist: Any,
    policy: str,
    smart_approve: Callable[[str, str], str],
) -> dict[str, str] | None:
    """Pure policy decision for a workflow-child terminal command in a non-CLI
    context. Returns a block directive, or None to allow.

    Only genuinely dangerous, non-allowlisted commands are gated; the hardline
    floor and the rest of Hermes' engine still run downstream for allowed ones.
    """
    is_dangerous, pattern_key, description = classify(command)
    if not is_dangerous:
        return None
    try:
        if pattern_key in allowlist:
            return None  # user explicitly allowlisted this pattern
    except TypeError:
        pass
    if policy == "approve":
        return None
    if policy == "smart":
        try:
            if smart_approve(command, description) == "approve":
                return None
        except Exception:
            pass  # smart eval failed -> fall through to block (safe)
    return _block(description)


def _is_interactive_cli() -> bool:
    try:
        from utils import env_var_enabled

        return bool(env_var_enabled("HERMES_INTERACTIVE"))
    except Exception:
        # On import failure treat as non-CLI so the policy is still enforced
        # (the safe direction).
        return False


def _policy() -> str:
    try:
        from .config import load_config

        return load_config().child_approval_policy
    except Exception:
        return "deny"


def pre_tool_call_handler(
    tool_name: str | None = None,
    args: Any = None,
    task_id: str = "",
    **_: Any,
) -> dict[str, str] | None:
    # Fast path: only workflow-child terminal commands are gated here.
    if not (isinstance(task_id, str) and task_id.startswith(WORKFLOW_CHILD_PREFIX)):
        return None
    if tool_name not in TERMINAL_TOOLS:
        return None
    command = args.get("command") if isinstance(args, dict) else None
    if not (isinstance(command, str) and command.strip()):
        return None
    # CLI is already handled by the per-thread approval callback; defer to it so
    # the policy isn't evaluated twice.
    if _is_interactive_cli():
        return None
    try:
        from tools.approval import (
            _smart_approve,
            detect_dangerous_command,
            load_permanent_allowlist,
        )
    except Exception:
        return None  # can't classify -> don't risk a false block
    try:
        allowlist = load_permanent_allowlist() or set()
    except Exception:
        allowlist = set()
    return evaluate_command_gate(
        command,
        classify=detect_dangerous_command,
        allowlist=allowlist,
        policy=_policy(),
        smart_approve=_smart_approve,
    )
