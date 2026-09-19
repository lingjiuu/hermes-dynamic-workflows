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

import json
import re
import shlex
import threading
from typing import Any, Callable

from .._hermes_compat import resolve

# Child task ids are minted as ``workflow-<uuid>`` by the runner.
WORKFLOW_CHILD_PREFIX = "workflow-"
TERMINAL_TOOLS = {"terminal"}
_CHILD_OBSERVERS: dict[str, Callable[[dict[str, Any]], None]] = {}
_CHILD_OBSERVERS_LOCK = threading.RLock()
_READ_ONLY_REDIRECT_RE = re.compile(r'(?:^|\s)(?:[0-9]?>|[0-9]?>>|&>)\s*(["\']?)([^\s"\']+)\1')
_MUTATING_SHELL_RE = re.compile(
    r"\b("
    r"rm|mv|cp|install|chmod|chown|mkdir|touch|tee|truncate|dd|mkfs|"
    r"git\s+(?:reset|clean|checkout|switch|branch|commit|push|pull|merge|rebase)|"
    r"npm\s+(?:install|add|remove|update)|"
    r"pip(?:3)?\s+install|"
    r"python(?:3)?\s+-m\s+pip\s+install"
    r")\b",
    re.IGNORECASE,
)
_PYTHON_WRITE_RE = re.compile(
    r"\b("
    r"open|exec|eval|compile|__import__|subprocess|os|pathlib|shutil|socket|requests|"
    r"write|writelines|remove|unlink|rename|replace|rmdir|mkdir|chmod|chown|system|popen|spawn"
    r")\b",
    re.IGNORECASE,
)
_CURL_WRITE_FLAGS = {
    "-o",
    "--output",
    "-O",
    "--remote-name",
    "--remote-header-name",
    "-T",
    "--upload-file",
    "-d",
    "--data",
    "--data-raw",
    "--data-binary",
    "--data-urlencode",
    "-F",
    "--form",
}
_READ_ONLY_FILTER_COMMANDS = {
    "cat",
    "cut",
    "grep",
    "head",
    "jq",
    "rg",
    "sed",
    "sort",
    "tail",
    "tr",
    "uniq",
    "wc",
}


def register_child_observer(task_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
    if not task_id or not callable(callback):
        return
    with _CHILD_OBSERVERS_LOCK:
        _CHILD_OBSERVERS[task_id] = callback


def unregister_child_observer(task_id: str) -> None:
    with _CHILD_OBSERVERS_LOCK:
        _CHILD_OBSERVERS.pop(task_id, None)


def _notify_child_observer(task_id: str, event: dict[str, Any]) -> None:
    with _CHILD_OBSERVERS_LOCK:
        callback = _CHILD_OBSERVERS.get(task_id)
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        pass


def _tool_activity(tool_name: str, args: Any) -> str:
    try:
        rendered = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(args or "")
    if len(rendered) > 140:
        rendered = rendered[:137] + "..."
    return f"{tool_name}({rendered})"


def is_obviously_read_only_terminal_command(command: str) -> bool:
    """Return True for shell commands that only fetch/read/filter text.

    This is intentionally conservative. It exists to avoid approval storms from
    workflow research children that use shell pipelines for web/file reads,
    while still keeping mutations, installs, writes, and arbitrary scripts on
    the normal approval path.
    """
    raw = str(command or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if any(token in lowered for token in ("<<", "$(", "`")):
        return False
    if _has_unquoted_command_separator(raw):
        return False
    if _MUTATING_SHELL_RE.search(raw):
        return False
    for match in _READ_ONLY_REDIRECT_RE.finditer(raw):
        target = match.group(2)
        if target not in {"/dev/null", "&1", "&2"}:
            return False
    parts = _split_pipeline(raw)
    if not parts:
        return False
    return all(_is_read_only_pipeline_part(part, index) for index, part in enumerate(parts))


def _split_pipeline(command: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    escaped = False
    for char in command:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == "|":
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _has_unquoted_command_separator(command: str) -> bool:
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == ";":
            return True
        if command.startswith("&&", index) or command.startswith("||", index):
            return True
        index += 1
    return False


def _is_read_only_pipeline_part(part: str, index: int) -> bool:
    try:
        tokens = shlex.split(part)
    except ValueError:
        return False
    tokens = _strip_env_prefix(tokens)
    if not tokens:
        return False
    name = tokens[0].rsplit("/", 1)[-1]
    if name in {"curl", "wget"}:
        return _is_read_only_fetch(tokens)
    if name in _READ_ONLY_FILTER_COMMANDS:
        return _is_read_only_filter(name, tokens)
    if name in {"python", "python3"}:
        return index > 0 and _is_read_only_python(tokens)
    return False


def _strip_env_prefix(tokens: list[str]) -> list[str]:
    while tokens and tokens[0] in {"env", "command"}:
        tokens = tokens[1:]
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            tokens = tokens[1:]
    return tokens


def _is_read_only_fetch(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens[1:], start=1):
        if token in _CURL_WRITE_FLAGS:
            return False
        if token in {"-X", "--request"}:
            method = tokens[index + 1].upper() if index + 1 < len(tokens) else ""
            if method and method not in {"GET", "HEAD"}:
                return False
        if token.startswith("-X") and len(token) > 2 and token[2:].upper() not in {"GET", "HEAD"}:
            return False
        if token.startswith("--request=") and token.split("=", 1)[1].upper() not in {"GET", "HEAD"}:
            return False
    return True


def _is_read_only_filter(name: str, tokens: list[str]) -> bool:
    if name == "sed" and any(token.startswith("-i") or token == "--in-place" for token in tokens[1:]):
        return False
    if name == "rg" and any(token in {"--files-with-matches", "--files"} for token in tokens[1:]):
        return True
    return True


def _is_read_only_python(tokens: list[str]) -> bool:
    code = ""
    for index, token in enumerate(tokens[1:], start=1):
        if token == "-c" and index + 1 < len(tokens):
            code = tokens[index + 1]
            break
        if token.startswith("-c") and len(token) > 2:
            code = token[2:]
            break
    if not code:
        return False
    return not _PYTHON_WRITE_RE.search(code)


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

    if is_obviously_read_only_terminal_command(command):
        if on_allow is not None:
            try:
                on_allow(pattern_key)
            except Exception:
                pass
        return None

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
        mode = resolve("_get_approval_mode")()
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

        if not resolve("_is_gateway_approval_context")():
            return False
        session_key = resolve("get_current_session_key")()
        with _approval._lock:
            return _approval._gateway_notify_cbs.get(session_key) is not None
    except Exception:
        return False


def _make_on_allow():
    """approve_session() the allowed pattern so the decision sticks past
    Hermes' downstream re-gating (its check_all_command_guards runs again and,
    for a detached gateway child, would otherwise re-flag the command)."""
    try:
        session_key = resolve("get_current_session_key")()
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
    # Observe every workflow-child tool call before applying the terminal-only
    # command policy below. This provides live TUI progress even when Hermes'
    # SessionDB has not flushed the child transcript yet.
    if not (isinstance(task_id, str) and task_id.startswith(WORKFLOW_CHILD_PREFIX)):
        return None
    clean_tool_name = str(tool_name or "tool")
    _notify_child_observer(
        task_id,
        {
            "type": "tool_call",
            "tool_name": clean_tool_name,
            "args": args if isinstance(args, dict) else {},
            "activity": _tool_activity(clean_tool_name, args),
        },
    )
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
        from tools.approval import detect_dangerous_command, load_permanent_allowlist

        _smart_approve = resolve("_smart_approve")
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
