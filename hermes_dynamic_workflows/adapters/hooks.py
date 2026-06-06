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
            "to 'smart'/'approve', or ask_fallback to 'smart', in plugin config)."
        ),
    }


def evaluate_command_gate(
    command: str,
    *,
    classify: Callable[[str], tuple],
    allowlist: Any,
    policy: str,
    smart_approve: Callable[[str, str], str],
    has_gateway_channel: bool = False,
    ask_fallback: str = "smart",
    on_allow: Callable[[str], None] | None = None,
) -> dict[str, str] | None:
    """Pure policy decision for a workflow-child terminal command in a non-CLI
    context. Returns a block directive, or None to allow.

    Only genuinely dangerous, non-allowlisted commands are gated; the hardline
    floor and the rest of Hermes' engine still run downstream.

    ``ask`` routes to the user only when a live gateway approval channel exists
    (``has_gateway_channel``) — then it defers to Hermes' gateway approve/deny.
    Otherwise (the common case: a detached workflow child has no reachable human
    in any context, since the gateway notify bridge is torn down when the
    launching turn ends) it degrades to ``ask_fallback`` (smart | deny | approve).

    When a flagged command is allowed by policy, ``on_allow(pattern_key)`` runs
    (the handler wires it to ``approve_session``) so the decision sticks past
    Hermes' own downstream context re-gating — which would otherwise turn a
    detached gateway child's allowed command into an unanswerable "pending".
    """
    is_dangerous, pattern_key, description = classify(command)
    if not is_dangerous:
        return None
    try:
        if pattern_key in allowlist:
            return None  # user explicitly allowlisted this pattern
    except TypeError:
        pass

    if policy == "ask":
        if has_gateway_channel:
            return None  # defer to Hermes' gateway approve/deny (real buttons)
        policy = ask_fallback if ask_fallback in ("smart", "deny", "approve") else "smart"

    allow = False
    if policy == "approve":
        allow = True
    elif policy == "smart":
        try:
            allow = smart_approve(command, description) == "approve"
        except Exception:
            allow = False  # smart eval failed -> block (safe)

    if allow:
        if on_allow is not None:
            try:
                on_allow(pattern_key)
            except Exception:
                pass
        return None
    return _block(description)


def _is_interactive_cli() -> bool:
    try:
        from utils import env_var_enabled

        return bool(env_var_enabled("HERMES_INTERACTIVE"))
    except Exception:
        # On import failure treat as non-CLI so the policy is still enforced
        # (the safe direction).
        return False


def _config() -> Any:
    try:
        from ..core.config import load_config

        return load_config()
    except Exception:
        return None


def _resolve_policy(cfg: Any) -> str:
    """The configured policy, resolving ``inherit`` to Hermes' own
    ``approvals.mode`` (manual->ask, smart->smart, off->approve)."""
    policy = getattr(cfg, "child_approval_policy", "deny") if cfg is not None else "deny"
    if policy != "inherit":
        return policy
    try:
        from tools.approval import _get_approval_mode

        mode = _get_approval_mode()
    except Exception:
        return "deny"  # can't read Hermes config -> safe default
    return {"manual": "ask", "smart": "smart", "off": "approve"}.get(mode, "deny")


def _ask_fallback(cfg: Any) -> str:
    return getattr(cfg, "ask_fallback", "smart") if cfg is not None else "smart"


def _has_gateway_channel() -> bool:
    """True only when a *live* gateway approval channel is reachable for this
    thread's session — i.e. Hermes' check_all_command_guards could actually
    route a prompt to the user. A detached workflow child usually has none (the
    notify bridge is torn down when the launching turn ends), so ``ask`` then
    degrades rather than orphaning the command as an unanswerable 'pending'."""
    try:
        from tools import approval as _approval

        if not _approval._is_gateway_approval_context():
            return False
        session_key = _approval.get_current_session_key()
        with _approval._lock:
            return _approval._gateway_notify_cbs.get(session_key) is not None
    except Exception:
        return False


def _make_on_allow():
    """approve_session() the allowed pattern so the decision sticks past
    Hermes' downstream re-gating (its check_all_command_guards runs again and,
    for a detached gateway child, would otherwise re-flag the command)."""
    try:
        from tools import approval as _approval

        session_key = _approval.get_current_session_key()
    except Exception:
        return None

    def _on_allow(pattern_key: str) -> None:
        from tools.approval import approve_session

        approve_session(session_key, pattern_key)

    return _on_allow


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
    cfg = _config()
    return evaluate_command_gate(
        command,
        classify=detect_dangerous_command,
        allowlist=allowlist,
        policy=_resolve_policy(cfg),
        smart_approve=_smart_approve,
        has_gateway_channel=_has_gateway_channel(),
        ask_fallback=_ask_fallback(cfg),
        on_allow=_make_on_allow(),
    )
