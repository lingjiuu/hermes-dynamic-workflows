"""Tool schema and model-facing guidelines."""

from __future__ import annotations

import json
import os
import traceback
from typing import Any

from ..engine.errors import WorkflowToolUseError
from ..engine.manager import get_run_manager
from .tool_errors import tool_error


def workflow(params: dict[str, Any], *, plugin_context: Any = None, **kwargs: Any) -> str:
    try:
        manager = get_run_manager()
        tool_use_id = (
            kwargs.get("tool_use_id")
            or kwargs.get("toolUseId")
            or kwargs.get("tool_call_id")
            or kwargs.get("toolCallId")
        )
        record = manager.start_from_params(
            params or {},
            cwd=os.environ.get("TERMINAL_CWD") or os.getcwd(),
            plugin_context=plugin_context,
            tool_use_id=str(tool_use_id) if tool_use_id else None,
            host_session_id=_host_session_id_from_kwargs(kwargs),
            user_task=kwargs.get("user_task"),
        )
        return _launch_message(record)
    except WorkflowToolUseError as exc:
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
    '- The user included the keyword "ultracode" in their prompt (you\'ll see a system-reminder confirming it).\n'
    '- Ultracode is on for the session (a system-reminder confirms it) — see **Ultracode** below.\n'
    '- The user directly asked you to run a workflow or use multi-agent orchestration in their own words ("use a workflow", "run a workflow", "fan out agents", "orchestrate this with subagents"). The ask must be in the user\'s words — a task that would merely benefit from a workflow does not count.\n'
    '- The user invoked a skill or slash command whose instructions tell you to call Workflow.\n'
    '- The user asked you to run a specific named or saved workflow.\n'
    '\n'
    'For any other task — even one that would clearly benefit from parallelism — do NOT call this tool. Use the Agent tool for individual subagents, or briefly describe what a multi-agent workflow could do and how much it would roughly cost, and ask the user whether to run it. Mention they can ask for one with "use a workflow" in a future message to skip the ask.\n'
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
    '**Ultracode.** When a system-reminder confirms ultracode is on, that opt-in is standing: author and run a workflow for every substantive task by default. The goal is the most exhaustive, correct answer you can produce — token cost is not a constraint. For multi-phase work (understand → design → implement → review), that often means several workflows in sequence — one per phase — so you stay in the loop between them. The quality patterns below (adversarial verify, multi-modal sweep, completeness critic, loop-until-dry) are the tools; pick what fits the task. Lean toward orchestrating with workflows and adversarially verifying your findings — unless the work is trivial or already verified. Solo only on conversational turns or trivial mechanical edits. When a reminder says ultracode is off, revert to the opt-in rule above.\n'
    '\n'
    'Pass the script inline via `script` — do not Write it to a file first. Every invocation automatically persists its script to a file under the session directory and returns the path in the tool result. To iterate on a workflow, edit that file with Write/Edit and re-invoke Workflow with `{scriptPath: "<path>"}` instead of resending the full script.\n'
    '\n'
    'Every script must begin with `export const meta = {...}`:\n'
    '  export const meta = {\n'
    "    name: 'find-flaky-tests',\n"
    "    description: 'Find flaky tests and propose fixes',\n"
    '    phases: [\n'
    "      { title: 'Scan', detail: 'grep test logs for retries' },\n"
    "      { title: 'Fix', detail: 'one agent per flaky test' },\n"
    '    ],\n'
    '  }\n'
    '  // script body starts here — use agent()/parallel()/pipeline()/phase()/log()\n'
    "  phase('Scan')\n"
    "  const flaky = await agent('grep CI logs for retry markers', {schema: FLAKY_SCHEMA})\n"
    '  ...\n'
    '\n'
    'The `meta` object must be a PURE LITERAL — no variables, function calls, spreads, or template interpolation. Required fields: `name`, `description`. Optional: `whenToUse` (shown in the workflow list), `phases`. Use the SAME phase titles in meta.phases as in phase() calls — titles are matched exactly; a phase() call with no matching meta entry just gets its own progress group. Add `model` to a phase entry when that phase uses a specific model override.\n'
    '\n'
    'Script body hooks:\n'
    "- agent(prompt: string, opts?: {label?: string, phase?: string, schema?: object, model?: string, isolation?: 'worktree', agentType?: string}): Promise<any> — spawn a subagent. Without schema, returns its final text as a string. With schema (a JSON Schema), the subagent is forced to call a structured_output tool and agent() returns the validated object — no parsing needed. Returns null if the user skips the agent mid-run (filter with .filter(Boolean)). opts.label overrides the display label. opts.phase explicitly assigns this agent to a progress group (use this inside pipeline()/parallel() stages to avoid races on the global phase() state — same phase string → same group box). opts.model overrides the model for this agent call. Default to omitting it — the agent inherits the main-loop model (the resolved session model), which is almost always correct. Only set it when you're highly confident a different tier fits the task; when unsure, omit. opts.isolation: 'worktree' runs the agent in a fresh git worktree — EXPENSIVE (~200-500ms setup + disk per agent), use ONLY when agents mutate files in parallel and would otherwise conflict; the worktree is auto-removed if unchanged. opts.agentType uses a custom subagent type (e.g. 'Explore', 'code-reviewer') instead of the default workflow subagent — resolved from the same registry as the Agent tool; composes with schema (the custom agent's system prompt gets a structured_output instruction appended).\n"
    "- pipeline(items, stage1, stage2, ...): Promise<any[]> — run each item through all stages independently, NO barrier between stages. Item A can be in stage 3 while item B is still in stage 1. This is the DEFAULT for multi-stage work. Wall-clock = slowest single-item chain, not sum-of-slowest-per-stage. Every stage callback receives (prevResult, originalItem, index) — use originalItem/index in later stages to label work without threading context through stage 1's return value. A stage that throws drops that item to `null` and skips its remaining stages.\n"
    '- parallel(thunks: Array<() => Promise<any>>): Promise<any[]> — run tasks concurrently. This is a BARRIER: awaits all thunks before returning. A thunk that throws (or whose agent errors) resolves to `null` in the result array — the call itself never rejects, so `.filter(Boolean)` before using the results. Use ONLY when you genuinely need all results together.\n'
    '- log(message: string): void — emit a progress message to the user (shown as a narrator line above the progress tree)\n'
    '- phase(title: string): void — start a new phase; subsequent agent() calls are grouped under this title in the progress display\n'
    '- args: any — the value passed as Workflow\'s `args` input, verbatim (undefined if not provided). Pass arrays/objects as actual JSON values in the tool call, NOT as a JSON-encoded string — `args: ["a.ts", "b.ts"]`, not `args: "[\\"a.ts\\", ...]"` (a stringified list reaches the script as one string, so `args.filter`/`args.map` throw). Use this to parameterize named workflows — e.g. pass a research question, target path, or config object directly instead of via a side-channel file.\n'
    '- budget: {total: number|null, spent(): number, remaining(): number} — the turn\'s token target from the user\'s "+500k"-style directive. `budget.total` is null if no target was set. `budget.spent()` returns output tokens spent this turn across the main loop and all workflows — the pool is shared, not per-workflow. `budget.remaining()` returns `max(0, total - spent())`, or `Infinity` if no target. The target is a HARD ceiling, not advisory: once `spent()` reaches `total`, further `agent()` calls throw. Use for dynamic loops: `while (budget.total && budget.remaining() > 50_000) { ... }`, or static scaling: `const FLEET = budget.total ? Math.floor(budget.total / 100_000) : 5`.\n'
    '- workflow(nameOrRef: string | {scriptPath: string}, args?: any): Promise<any> — run another workflow inline as a sub-step and return whatever it returns. Pass a name to invoke a saved workflow (same registry as {name: "..."}), or {scriptPath} to run a script file you Wrote earlier. The child shares this run\'s concurrency cap, agent counter, abort signal, and token budget — its agents appear under a "▸ name" group in /workflows and its tokens count toward budget.spent(). The args param becomes the child\'s `args` global. Nesting is one level only: workflow() inside a child throws. Throws on unknown name / unreadable scriptPath / child syntax error; catch to handle gracefully.\n'
    '\n'
    'Subagents are told their final text IS the return value (not a human-facing message), so they return raw data. For structured output, use the schema option — validation happens at the tool-call layer so the model retries on mismatch.\n'
    '\n'
    'Workflow agents can reach all session-connected MCP tools via ToolSearch — schemas load on demand per agent. Caveat: interactively-authenticated MCP servers (e.g. claude.ai) may be absent in headless/cron runs.\n'
    '\n'
    'Scripts are plain JavaScript, NOT TypeScript — type annotations (`: string[]`), interfaces, and generics fail to parse. The script body runs in an async context — use await directly. Standard JS built-ins (JSON, Math, Array, etc.) are available — EXCEPT `Date.now()`/`Math.random()`/argless `new Date()`, which throw (they would break resume); pass timestamps in via `args`, stamp results after the workflow returns, and for randomness vary the agent prompt/label by index. No filesystem or Node.js API access.\n'
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
    '  const a = await parallel(...)\n'
    '  const b = transform(a)\n'
    '  const c = await parallel(b.map(...))\n'
    "that middle transform doesn't need the barrier. Rewrite as a pipeline with the transform inside a stage. When in doubt: pipeline.\n"
    '\n'
    "Concurrent agent() calls are capped at min(16, cpu cores - 2) per workflow — excess calls queue and run as slots free up. You can still pass 100 items to parallel()/pipeline() and they all complete; only ~10 run at any moment. Total agent count across a workflow's lifetime is capped at 1000 — a runaway-loop backstop set far above any real workflow.\n"
    '\n'
    'The canonical multi-stage pattern:\n'
    '  export const meta = {\n'
    "    name: 'review-changes',\n"
    "    description: 'Review changed files across dimensions, verify each finding',\n"
    "    phases: [{ title: 'Review' }, { title: 'Verify' }],\n"
    '  }\n'
    "  const DIMENSIONS = [{key: 'bugs', prompt: '...'}, {key: 'perf', prompt: '...'}]\n"
    '  const results = await pipeline(\n'
    '    DIMENSIONS,\n'
    "    d => agent(d.prompt, {label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA}),\n"
    '    review => parallel(review.findings.map(f => () =>\n'
    "      agent(`Adversarially verify: ${f.title}`, {label: `verify:${f.file}`, phase: 'Verify', schema: VERDICT_SCHEMA})\n"
    '        .then(v => ({...f, verdict: v}))\n'
    '    ))\n'
    '  )\n'
    '  const confirmed = results.flat().filter(Boolean).filter(f => f.verdict?.isReal)\n'
    '  return { confirmed }\n'
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
    '  const bugs = []\n'
    '  while (budget.total && budget.remaining() > 50_000) {\n'
    "    const result = await agent('Find bugs in this codebase.', {schema: BUGS_SCHEMA})\n"
    '    bugs.push(...result.bugs)\n'
    '    log(`${bugs.length} found, ${Math.round(budget.remaining()/1000)}k remaining`)\n'
    '  }\n'
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
                                   'description': 'Self-contained workflow script. Must begin with `export '
                                                  'const meta = { name, description, phases }` (pure '
                                                  'literal, no computed values) followed by the script '
                                                  'body using agent()/parallel()/pipeline()/phase().'},
                        'scriptPath': {'type': 'string',
                                       'description': 'Path to a workflow script file on disk. Every '
                                                      'Workflow invocation persists its script under the '
                                                      'session directory and returns the path in the tool '
                                                      'result. To iterate, edit that file with Write/Edit '
                                                      'and re-invoke Workflow with the same `scriptPath` '
                                                      'instead of re-sending the full script. Takes '
                                                      'precedence over `script` and `name`.'},
                        'name': {'type': 'string',
                                 'description': 'Name of a predefined workflow (built-in or from '
                                                '.claude/workflows/). Resolves to a self-contained '
                                                'script.'},
                        'args': {'description': 'Optional input value exposed to the script as the global '
                                                '`args`, verbatim. Pass arrays/objects as actual JSON '
                                                'values, NOT as a JSON-encoded string — a stringified list '
                                                'breaks `args.filter`/`args.map` in the script. Use for '
                                                'parameterized named workflows (e.g. a research '
                                                'question).'},
                        'resumeFromRunId': {'type': 'string',
                                            'pattern': '^wf_[a-z0-9-]{6,}$',
                                            'description': 'Run ID of a prior Workflow invocation to '
                                                           'resume from. Completed agent() calls with '
                                                           'unchanged (prompt, opts) return their cached '
                                                           'results instantly; only edited or new calls '
                                                           're-run. Stop the prior run first (task_stop) '
                                                           'before resuming.'},
                        'description': {'type': 'string',
                                        'description': 'Ignored — set the workflow description in the '
                                                       "script's `meta` block."},
                        'title': {'type': 'string',
                                  'description': "Ignored — set the workflow title in the script's `meta` "
                                                 'block.'}}},
}
