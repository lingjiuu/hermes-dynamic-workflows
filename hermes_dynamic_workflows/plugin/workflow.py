"""Tool schema and model-facing guidelines."""

from __future__ import annotations

import copy
import json
import os
import traceback
from typing import Any

from ..agent.presets import list_agent_types
from ..core.errors import SandboxViolation, WorkflowParseError, WorkflowToolUseError
from ..engine.manager import get_run_manager
from .tool_errors import tool_error


def workflow(params: dict[str, Any], *, plugin_context: Any = None, **kwargs: Any) -> str:
    try:
        manager = get_run_manager()
        record = manager.start_from_params(
            params or {},
            cwd=os.environ.get("TERMINAL_CWD") or os.getcwd(),
            plugin_context=plugin_context,
            parent_agent=kwargs.get("parent_agent"),
            host_session_id=_host_session_id_from_kwargs(kwargs),
            user_task=kwargs.get("user_task"),
        )
        return _launch_message(record)
    except (WorkflowParseError, SandboxViolation, WorkflowToolUseError) as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return json.dumps(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "trace": _short_traceback(),
            },
            ensure_ascii=False,
        )


def _launch_message(record: dict[str, Any]) -> str:
    run_id = record.get("runId") or ""
    task_id = record.get("taskId") or run_id
    summary = record.get("summary") or "Dynamic workflow"
    transcript_dir = record.get("transcriptDir") or ""
    script_path = record.get("scriptPath") or ""
    return "\n".join(
        [
            f"Workflow launched in background. Task ID: {task_id}",
            f"Summary: {summary}",
            f"Transcript dir: {transcript_dir}",
            f"Script file: {script_path}",
            f"Run ID: {run_id}",
            (
                "To resume after editing the script: "
                f"Workflow({{scriptPath: {json.dumps(script_path, ensure_ascii=False)}, "
                f"resumeFromRunId: {json.dumps(run_id)}}})"
            ),
            "You will be notified when it completes. Use /workflows to watch live progress.",
        ]
    )


def _short_traceback() -> str:
    lines = traceback.format_exc(limit=4).strip().splitlines()
    return "\n".join(lines[-8:])
def _host_session_id_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    for key in (
        "session_id",
        "sessionId",
        "current_session_id",
        "currentSessionId",
        "task_id",
        "taskId",
    ):
        value = kwargs.get(key)
        if value:
            return str(value)
    return None


_DESCRIPTION = (
    'Execute a workflow script that orchestrates multiple subagents deterministically. Workflows run in the background — this tool returns immediately with a task ID, and a <task-notification> arrives when the workflow completes. Use /workflows to watch live progress.\n'
    '\n'
    "A workflow structures work across many agents — to be comprehensive (decompose and cover in parallel), to be confident (independent perspectives and adversarial checks before committing), or to take on scale one context can't hold (migrations, audits, broad sweeps). The script is where you encode that structure: what fans out, what verifies, what synthesizes.\n"
    '\n'
    'ONLY call this tool when the user has explicitly opted into multi-agent orchestration. Workflows can spawn dozens of agents and consume a large amount of tokens; the user must request that scale, not have it inferred. Explicit opt-in means one of:\n'
    '- The user directly asked you to run a workflow or use multi-agent orchestration in their own words ("use a workflow", "run a workflow", "fan out agents", "orchestrate this with subagents"). The ask must be in the user\'s words — a task that would merely benefit from a workflow does not count.\n'
    '- The user invoked a skill or slash command whose instructions tell you to call the workflow tool.\n'
    '- The user asked you to run a specific named or saved workflow.\n'
    '\n'
    'For any other task — even one that would clearly benefit from parallelism — do NOT call this tool. Use the available tools, or briefly describe what a multi-agent workflow could do and how much it would roughly cost, and ask the user whether to run it. Mention they can ask for one with "use a workflow" in a future message to skip the ask.\n'
    '\n'
    "When you do call it, the right move is often **hybrid**: scout inline first (list the files, find the channels, scope the diff) to discover the work-list, then call Workflow to pipeline over it. You don't need to know the shape before the *task* — only before the *orchestration step*.\n"
    '\n'
    'Common single-phase workflows you can chain across turns:\n'
    '- **Understand** — parallel readers over relevant subsystems → structured map\n'
    '- **Design** — judge panel of N independent approaches → scored synthesis\n'
    '- **Review** — dimensions → find → adversarially verify (example below)\n'
    '- **Research** — multi-modal sweep → deep-read → synthesize\n'
    '- **Migrate** — discover sites → transform each (worktree isolation) → verify\n'
    '\n'
    'For larger work, run several in sequence — read each result before deciding the next phase. You stay in the loop; each workflow is one well-scoped fan-out.\n'
    '\n'
    'Pass the script inline via `script` — do not save it to a file first. Every invocation automatically persists its script to a file under the session directory and returns the path in the tool result. To iterate on a workflow, edit that file and re-invoke the workflow tool with `{scriptPath: "<path>"}` instead of resending the full script.\n'
    '\n'
    'Every script must begin with literal `meta = {...}`:\n'
    '  meta = {\n'
    '      "name": "find-flaky-tests",\n'
    '      "description": "Find flaky tests and propose fixes",\n'
    '      "phases": [\n'
    '          {"title": "Scan", "detail": "grep test logs for retries"},\n'
    '          {"title": "Fix", "detail": "one agent per flaky test"},\n'
    '      ],\n'
    '  }\n'
    '\n'
    '  phase("Scan")\n'
    '  flaky = await agent("grep CI logs for retry markers", {"schema": FLAKY_SCHEMA})\n'
    '  ...\n'
    '\n'
    'The `meta` dict must be a PURE LITERAL — no variables, function calls, dict unpacking, or f-string interpolation. Required fields: `name`, `description`. Optional: `whenToUse` (shown in the workflow list), `phases`. Use the SAME phase titles in meta["phases"] as in phase() calls — titles are matched exactly; a phase() call with no matching meta entry just gets its own progress group. Add `model` to a phase entry when that phase uses a specific model override.\n'
    '\n'
    'Script body hooks:\n'
    '- agent(prompt: str, opts: dict | None = None) -> Awaitable[Any] — spawn a subagent. Use `await agent(...)` for direct calls. Inside parallel()/pipeline() callbacks, return the awaitable; the helper will await it. Supported opts are `label`, `phase`, `schema`, `model`, `isolation`, and `agentType`. Without schema, returns its final text as a string. With schema (a JSON Schema), the subagent is forced to call the `structured_output` tool and agent() returns the validated object — no parsing needed. Returns None if the user skips the agent mid-run (filter with `if x is not None`). opts["label"] overrides the display label. opts["phase"] explicitly assigns this agent to a progress group (use this inside pipeline()/parallel() stages to avoid races on the global phase() state — same phase string → same group box). opts["model"] overrides the model for this agent call. Default to omitting it — the agent inherits the resolved session model, which is almost always correct. Only set it when you\'re highly confident a different tier fits the task; when unsure, omit. opts["isolation"]: "worktree" runs the agent in a fresh git worktree — EXPENSIVE (~200-500ms setup + disk per agent), use ONLY when agents mutate files in parallel and would otherwise conflict; the worktree is auto-removed if unchanged. opts["agentType"] uses a custom subagent type (e.g. "Explore", "code-reviewer") instead of the default workflow subagent — resolved from workflow agent files; composes with schema (the custom agent\'s system prompt gets a structured_output instruction appended).\n'
    '- pipeline(items, stage1, stage2, ...) -> Awaitable[list[Any]] — run each item through all stages independently, NO barrier between stages. Call it with `await`. Item A can be in stage 3 while item B is still in stage 1. This is the DEFAULT for multi-stage work. Wall-clock = slowest single-item chain, not sum-of-slowest-per-stage. Every stage callback receives (prev_result, original_item, index) — use original_item/index in later stages to label work without threading context through stage 1\'s return value. A stage that raises drops that item to `None` and skips its remaining stages.\n'
    '- parallel(thunks: list[Callable[[], Awaitable[Any] | Any]]) -> Awaitable[list[Any]] — run tasks concurrently. Call it with `await`. This is a BARRIER: awaits all thunks before returning. A thunk that raises (or whose agent errors) resolves to `None` in the result list — the call itself never raises for per-item failures, so filter out `None` values before using the results. Use ONLY when you genuinely need all results together.\n'
    '- log(message: str) -> None — emit a progress message to the user (shown as a narrator line above the progress tree)\n'
    '- phase(title: str) -> None — start a new phase; subsequent agent() calls are grouped under this title in the progress display\n'
    '- args: Any — the value passed as Workflow\'s `args` input, verbatim (`None` if not provided). Pass arrays/objects as actual JSON values in the tool call, NOT as a JSON-encoded string — `args: ["a.ts", "b.ts"]`, not `args: "[\\"a.ts\\", ...]"` (a stringified list reaches the script as one string, so list/dict operations you expected may fail). Use this to parameterize named workflows — e.g. pass a research question, target path, or config object directly instead of via a side-channel file.\n'
    '- budget: object with `total`, `spent()`, and `remaining()` — the token target parsed from the user\'s "+500k"-style directive for this workflow run. `budget.total` is `None` if no target was set. `budget.spent()` returns child-agent tokens spent by this workflow run. `budget.remaining()` returns `max(0, total - spent())`, or `math.inf` if no target. The target is a HARD ceiling, not advisory: once `spent()` reaches `total`, further agent() calls raise. Use for dynamic loops: `while budget.total and budget.remaining() > 50_000: ...`, or static scaling: `fleet = int(budget.total / 100_000) if budget.total else 5`.\n'
    '- workflow(name_or_ref: str | {"scriptPath": str}, args: Any = None) -> Awaitable[Any] — run another workflow inline as a sub-step and return whatever it returns. Call it with `await`. Pass a name to invoke a saved workflow (same registry as `{name: "..."}`), or `{"scriptPath": path}` to run a script file you wrote earlier. The child shares this run\'s concurrency cap, agent counter, abort signal, and token budget. The args param becomes the child\'s `args` global. Nesting is one level only: workflow() inside a child raises. Raises on unknown name / unreadable scriptPath / child syntax error; catch Exception to handle gracefully.\n'
    '\n'
    'Subagents are told their final text IS the return value (not a human-facing message), so they return raw data. For structured output, use the schema option — validation happens at the tool-call layer so the model retries on mismatch.\n'
    '\n'
    'Workflow agents can reach all session-connected MCP tools when `tool_search` is enabled — schemas load on demand per agent. Caveat: interactively-authenticated MCP servers may be absent in headless/cron runs.\n'
    '\n'
    'Scripts are restricted plain Python. The script body runs in an async context — use `await` directly. Available globals are intentionally narrow: `agent`, `parallel`, `pipeline`, `phase`, `log`, `args`, `budget`, `workflow`, `json`, `math`, safe built-ins, and common exception types. Imports, filesystem/process/network APIs, dunder traversal, dynamic eval/exec, class definitions, and dynamic call targets are rejected. Current time and randomness APIs are unavailable because they would break resume; pass timestamps in via `args`, stamp results after the workflow returns, and for randomness vary the agent prompt/label by index.\n'
    '\n'
    'DEFAULT TO pipeline(). Only reach for a barrier (parallel between stages) when you genuinely need ALL prior-stage results together.\n'
    '\n'
    'A barrier is correct ONLY when stage N needs cross-item context from all of stage N-1:\n'
    '- Dedup/merge across the full result set before expensive downstream work\n'
    '- Early-exit if the total count is zero ("0 bugs found → skip verification entirely")\n'
    '- Stage N\'s prompt references "the other findings" for comparison\n'
    '\n'
    'A barrier is NOT justified by:\n'
    '- "I need to flatten/map/filter first" — do it inside a pipeline stage\n'
    '- "The stages are conceptually separate" — that\'s what pipeline() models\n'
    '- "It\'s cleaner code" — barrier latency is real\n'
    '\n'
    'Smell test: if you wrote\n'
    '  a = await parallel(...)\n'
    '  b = transform(a)\n'
    '  c = await parallel([...])\n'
    "that middle transform doesn't need the barrier. Rewrite as a pipeline with the transform inside a stage. When in doubt: pipeline.\n"
    '\n'
    "Concurrent agent() calls are capped at min(16, cpu cores - 2) per workflow — excess calls queue and run as slots free up. You can still pass 100 items to parallel()/pipeline() and they all complete; only ~10 run at any moment. Total agent count across a workflow's lifetime is capped at 1000 — a runaway-loop backstop set far above any real workflow.\n"
    '\n'
    'The canonical multi-stage pattern:\n'
    '  meta = {\n'
    '      "name": "review-changes",\n'
    '      "description": "Review changed files across dimensions, verify each finding",\n'
    '      "phases": [{"title": "Review"}, {"title": "Verify"}],\n'
    '  }\n'
    '\n'
    '  DIMENSIONS = [\n'
    '      {"key": "bugs", "prompt": "..."},\n'
    '      {"key": "perf", "prompt": "..."},\n'
    '  ]\n'
    '\n'
    '  async def verify_finding(f):\n'
    '      verdict = await agent(\n'
    '          "Adversarially verify: " + f["title"],\n'
    '          {"label": "verify:" + f["file"], "phase": "Verify", "schema": VERDICT_SCHEMA},\n'
    '      )\n'
    '      merged = dict(f)\n'
    '      merged["verdict"] = verdict\n'
    '      return merged\n'
    '\n'
    '  results = await pipeline(\n'
    '      DIMENSIONS,\n'
    '      lambda d, original, index: agent(\n'
    '          d["prompt"],\n'
    '          {"label": "review:" + d["key"], "phase": "Review", "schema": FINDINGS_SCHEMA},\n'
    '      ),\n'
    '      lambda review, original, index: parallel([\n'
    '          lambda f=f: verify_finding(f)\n'
    '          for f in review["findings"]\n'
    '      ]),\n'
    '  )\n'
    '\n'
    '  confirmed = [\n'
    '      f\n'
    '      for group in results\n'
    '      if group\n'
    '      for f in group\n'
    '      if f and f.get("verdict", {}).get("isReal")\n'
    '  ]\n'
    '  return {"confirmed": confirmed}\n'
    '\n'
    'Quality patterns:\n'
    '- Adversarial verify: spawn N independent skeptics per finding, each prompted to REFUTE\n'
    '- Perspective-diverse verify: give each verifier a distinct lens (correctness, security, perf, repro)\n'
    '- Judge panel: generate N independent attempts, score with parallel judges, synthesize from winner\n'
    '- Loop-until-dry: keep spawning finders until K consecutive rounds return nothing new\n'
    '- Multi-modal sweep: parallel agents each searching a different way\n'
    "- Completeness critic: a final agent that asks what's missing\n"
    '\n'
    'Loop-until-budget pattern:\n'
    '  bugs = []\n'
    '  while budget.total and budget.remaining() > 50_000:\n'
    '      result = await agent("Find bugs in this codebase.", {"schema": BUGS_SCHEMA})\n'
    '      bugs.extend(result["bugs"])\n'
    '      log(f"{len(bugs)} found, {round(budget.remaining() / 1000)}k remaining")\n'
    '\n'
    'Use this tool for multi-step orchestration where control flow should be deterministic (loops, conditionals, fan-out) rather than model-driven.\n'
    '\n'
    '## Resume\n'
    '\n'
    'The tool result includes a runId. To resume after a pause, kill, or script edit, relaunch with Workflow({scriptPath, resumeFromRunId})\n'
    'Resume reuses cached results for completed agent() calls whose prompt and relevant opts are unchanged. It uses content-addressed caching, so unchanged calls can still reuse cached results even when parallel scheduling order changes. Still, when editing a workflow, preserve the earliest stable agent() calls whenever possible: changes near the start usually affect downstream prompts and reduce cache reuse, while changes near the end preserve more cached work.'
)

DYNAMIC_WORKFLOW_SCHEMA = {
    "description": _DESCRIPTION,
    "parameters": {'$schema': 'https://json-schema.org/draft/2020-12/schema',
         'additionalProperties': False,
         'type': 'object',
         'properties': {'script': {'type': 'string',
                                   'maxLength': 524288,
                                   'description': 'Self-contained workflow script. Must begin with literal '
                                                  '`meta = {"name": ..., "description": ..., "phases": ...}` (pure '
                                                  'literal, no computed values) followed by the script '
                                                  'body using agent()/parallel()/pipeline()/phase().'},
                        'scriptPath': {'type': 'string',
                                       'description': 'Path to a workflow script file on disk. Every '
                                                      'Workflow invocation persists its script under the '
                                                      'session directory and returns the path in the tool '
                                                      'result. To iterate, edit that file and re-invoke '
                                                      'the workflow tool with the same `scriptPath` '
                                                      'instead of re-sending the full script. Takes '
                                                      'precedence over `script` and `name`.'},
                        'name': {'type': 'string',
                                 'description': 'Name of a predefined workflow (built-in or from '
                                                '.hermes/workflows/). Resolves to a self-contained '
                                                'script.'},
                        'args': {'description': 'Optional input value exposed to the script as the global '
                                                '`args`, verbatim (`None` if not provided). Pass arrays/objects as actual JSON '
                                                'values, NOT as a JSON-encoded string — a stringified list '
                                                'reaches the script as one string, so list/dict operations '
                                                'you expected may fail. Use for '
                                                'parameterized named workflows (e.g. a research '
                                                'question).'},
                        'resumeFromRunId': {'type': 'string',
                                            'pattern': '^wf_[a-z0-9-]{6,}$',
                                            'description': 'Run ID of a prior Workflow invocation to '
                                                           'resume from. Completed agent() calls with '
                                                           'unchanged prompt and relevant opts return '
                                                           'their cached results instantly. Calls whose '
                                                           'prompt or relevant opts changed run live. '
                                                           'Stop the prior run first (task_stop) '
                                                           'before resuming.'},
                        'description': {'type': 'string',
                                        'description': 'Ignored — set the workflow description in the '
                                                       "script's `meta` block."},
                        'title': {'type': 'string',
                                  'description': "Ignored — set the workflow title in the script's `meta` "
                                                 'block.'}}},
}


def get_dynamic_workflow_schema(*, cwd: str | None = None) -> dict[str, Any]:
    """Return the workflow tool schema with session-local agentType hints."""
    schema = copy.deepcopy(DYNAMIC_WORKFLOW_SCHEMA)
    section = _available_agent_types_section(cwd=cwd)
    if section:
        schema["description"] = f"{schema['description'].rstrip()}\n\n{section}"
    return schema


def _available_agent_types_section(*, cwd: str | None) -> str:
    try:
        specs = list_agent_types(cwd=cwd)
    except Exception:
        specs = []
    lines = ["Available agent types and the tools they have access to:"]
    if not specs:
        lines.append("- none discovered")
    else:
        for spec in specs:
            description = spec.description or "Custom workflow agent."
            tools = _agent_type_tools_label(spec)
            lines.append(f"- {spec.name}: {description} (Tools: {tools})")
    return "\n".join(lines)


_FALLBACK_TOOLSET_TOOLS = {
    "web": ["web_search", "web_extract"],
    "file": ["read_file", "write_file", "patch", "search_files"],
    "terminal": ["terminal", "process"],
    "skills": ["skills_list", "skill_view", "skill_manage"],
    "browser": [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_back",
        "browser_press",
        "browser_get_images",
        "browser_vision",
        "browser_console",
        "browser_cdp",
        "browser_dialog",
        "web_search",
    ],
}


def _agent_type_tools_label(spec: Any) -> str:
    allowed = _clean_tool_names(getattr(spec, "allowed_tools", ()) or ())
    disallowed = set(_clean_tool_names(getattr(spec, "disallowed_tools", ()) or ()))
    if allowed:
        tools = [name for name in allowed if name not in disallowed]
        return ", ".join(tools) if tools else "none"

    toolsets = _clean_tool_names(getattr(spec, "toolsets", ()) or ())
    if not toolsets:
        return "default workflow toolsets"

    tools = _resolve_toolset_tools(toolsets)
    if disallowed:
        tools = [name for name in tools if name not in disallowed]
    return ", ".join(tools) if tools else "none"


def _clean_tool_names(values: Any) -> list[str]:
    names: list[str] = []
    for item in values:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names


def _resolve_toolset_tools(toolsets: list[str]) -> list[str]:
    resolve_toolset = None
    try:
        from toolsets import resolve_toolset as _resolve_toolset

        resolve_toolset = _resolve_toolset
    except Exception:
        pass

    tools: list[str] = []
    for toolset in toolsets:
        names = _FALLBACK_TOOLSET_TOOLS.get(toolset)
        if names is None and resolve_toolset is not None:
            try:
                names = resolve_toolset(toolset)
            except Exception:
                names = None
        for name in names or [toolset]:
            if name not in tools:
                tools.append(name)
    return tools
