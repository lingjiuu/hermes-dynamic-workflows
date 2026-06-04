"""Configuration helpers for the dynamic workflow plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def default_concurrency() -> int:
    return min(16, max(1, (os.cpu_count() or 4) - 2))


@dataclass(frozen=True)
class PluginConfig:
    concurrency: int = field(default_factory=default_concurrency)
    max_concurrency: int = 16
    max_agents: int = 1000
    workflow_timeout_seconds: float = 900.0
    child_timeout_seconds: float = 300.0
    script_max_chars: int = 524288
    token_budget_total: int | None = None
    mcp_discovery_wait_seconds: float = 0.75
    default_child_toolsets: tuple[str, ...] = ("web", "file", "terminal")
    blocked_child_toolsets: tuple[str, ...] = (
        "workflow",
        "workflows",
        "delegation",
        "code_execution",
        "memory",
        "messaging",
        "clarify",
    )
    # Per-agent model override (agent(model=...)) is allowed by default, matching
    # Claude Code's per-agent / per-stage model routing; the default is still the
    # session model, override only when a stage wants a different tier. Provider
    # override is a bigger surface (different endpoint/key/data flow), so it stays
    # opt-in.
    allow_model_override: bool = True
    allow_provider_override: bool = False
    keep_worktrees: bool = False
    # What a child agent does when Hermes' approval engine flags a command and
    # no human is present to approve it. The engine itself (hardline blocks,
    # permanent allowlist, yolo, smart mode) still runs upstream regardless;
    # this only decides the otherwise-would-prompt case.
    #   deny    -> refuse flagged commands (safe default)
    #   approve -> allow flagged commands (hardline still blocked upstream)
    #   smart   -> defer to Hermes' _smart_approve auxiliary-LLM guardian
    child_approval_policy: str = "deny"
    # How agent(schema=...) constrains child output:
    #   "auto"/"tool" -> child calls workflow_submit_structured_output, validated
    #       at the tool layer with model retry (Claude-Code-style); falls back to
    #       parsing the final message if the tool is never called.
    #   "response_format" -> provider-native json_schema response_format override.
    #   "prompt" -> instruction only, then parse the final message.
    structured_output_mode: str = "auto"
    structured_retries: int = 1
    structured_repair_with_llm: bool = True
    structured_raw_preview_chars: int = 2000


def _as_int(value: Any, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(value: Any, default: float, *, minimum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_str_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        return default
    cleaned = tuple(item for item in items if item)
    return cleaned or default


def _as_optional_int(value: Any, *, minimum: int = 1, maximum: int | None = None) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_mode(value: Any, default: str, allowed: set[str]) -> str:
    clean = str(value or "").strip().lower()
    return clean if clean in allowed else default


def load_config() -> PluginConfig:
    """Load plugin config from Hermes config.yaml and environment variables."""
    raw: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config as _load_hermes_config

        hermes_cfg = _load_hermes_config() or {}
        entries = ((hermes_cfg.get("plugins") or {}).get("entries") or {})
        entry = entries.get("dynamic-workflows") or entries.get("dynamic_workflows") or {}
        if isinstance(entry, dict):
            raw = entry.get("dynamic_workflows") or entry.get("config") or entry
            if not isinstance(raw, dict):
                raw = {}
    except Exception:
        raw = {}

    default = PluginConfig()
    concurrency = _as_int(
        os.getenv("HERMES_DYNAMIC_WORKFLOWS_CONCURRENCY", raw.get("concurrency")),
        default.concurrency,
        minimum=1,
        maximum=32,
    )
    max_concurrency = _as_int(raw.get("max_concurrency"), default.max_concurrency, minimum=1, maximum=32)
    max_concurrency = _as_int(
        os.getenv("HERMES_DYNAMIC_WORKFLOWS_MAX_CONCURRENCY"),
        max_concurrency,
        minimum=1,
        maximum=32,
    )

    return PluginConfig(
        concurrency=min(concurrency, max_concurrency),
        max_concurrency=max_concurrency,
        max_agents=_as_int(raw.get("max_agents"), default.max_agents, minimum=1, maximum=1000),
        workflow_timeout_seconds=_as_float(
            raw.get("workflow_timeout_seconds"),
            default.workflow_timeout_seconds,
        ),
        child_timeout_seconds=_as_float(
            raw.get("child_timeout_seconds"),
            default.child_timeout_seconds,
        ),
        script_max_chars=_as_int(
            raw.get("script_max_chars"),
            default.script_max_chars,
            minimum=1000,
            maximum=1048576,
        ),
        token_budget_total=_as_optional_int(
            os.getenv("HERMES_DYNAMIC_WORKFLOWS_TOKEN_BUDGET", raw.get("token_budget_total")),
            minimum=1,
        ),
        mcp_discovery_wait_seconds=_as_float(
            raw.get("mcp_discovery_wait_seconds"),
            default.mcp_discovery_wait_seconds,
            minimum=0.0,
        ),
        default_child_toolsets=_as_str_tuple(
            raw.get("default_child_toolsets"),
            default.default_child_toolsets,
        ),
        blocked_child_toolsets=_as_str_tuple(
            raw.get("blocked_child_toolsets"),
            default.blocked_child_toolsets,
        ),
        allow_model_override=_as_bool(
            os.getenv("HERMES_DYNAMIC_WORKFLOWS_ALLOW_MODEL_OVERRIDE", raw.get("allow_model_override")),
            default.allow_model_override,
        ),
        allow_provider_override=_as_bool(
            os.getenv("HERMES_DYNAMIC_WORKFLOWS_ALLOW_PROVIDER_OVERRIDE", raw.get("allow_provider_override")),
            default.allow_provider_override,
        ),
        keep_worktrees=_as_bool(
            os.getenv("HERMES_DYNAMIC_WORKFLOWS_KEEP_WORKTREES", raw.get("keep_worktrees")),
            default.keep_worktrees,
        ),
        child_approval_policy=_as_mode(
            os.getenv(
                "HERMES_DYNAMIC_WORKFLOWS_CHILD_APPROVAL_POLICY",
                raw.get("child_approval_policy"),
            ),
            default.child_approval_policy,
            {"deny", "smart", "approve"},
        ),
        structured_output_mode=_as_mode(
            raw.get("structured_output_mode"),
            default.structured_output_mode,
            {"auto", "tool", "response_format", "prompt"},
        ),
        structured_retries=_as_int(
            raw.get("structured_retries"),
            default.structured_retries,
            minimum=0,
            maximum=1,
        ),
        structured_repair_with_llm=_as_bool(
            raw.get("structured_repair_with_llm"),
            default.structured_repair_with_llm,
        ),
        structured_raw_preview_chars=_as_int(
            raw.get("structured_raw_preview_chars"),
            default.structured_raw_preview_chars,
            minimum=200,
            maximum=20000,
        ),
    )
