"""Tool schema and model-facing guidelines."""

from __future__ import annotations

_DESCRIPTION = """Execute a Python workflow script that orchestrates multiple Hermes child agents deterministically. The tool starts a background run and returns immediately with a runId and scriptPath. Use /workflows to list runs, /workflows <runId> to inspect progress/results, and /workflow-stop <runId> to stop a run.

A workflow structures work across many agents: to be comprehensive (decompose and cover in parallel), to be confident (independent perspectives and adversarial checks before committing), or to take on scale one context cannot hold (audits, broad sweeps, large reviews). The script encodes what fans out, what verifies, and what synthesizes.

ONLY call this tool when the user explicitly opted into multi-agent orchestration. Explicit opt-in means the user directly asked to run a workflow, use multi-agent orchestration, fan out agents, orchestrate with subagents, or invoked a skill/slash command whose instructions call for a workflow. For ordinary tasks that would merely benefit from parallelism, do not call this tool; use ordinary tools/subagents or ask whether the user wants a workflow.

When calling this tool, the right move is often hybrid: scout inline first (list files, inspect the diff, discover the work-list), then call workflow to pipeline over the discovered items. You do not need to know the whole shape before the task, only before the orchestration step.

Pass the script inline via script, or pass scriptPath to rerun a saved script, or pass name to run a predefined workflow from .hermes/workflows/<name>.py or ~/.hermes/dynamic-workflows/workflows/<name>.py. Every inline invocation automatically persists the script under ~/.hermes/dynamic-workflows/scripts/<runId>.py and returns that path. To iterate, edit that file and call workflow with scriptPath instead of resending the full script.

Every Python workflow script should define a literal meta dict near the top:

  meta = {
      "name": "review-changes",
      "description": "Review changed files across dimensions and verify findings",
      "phases": [
          {"title": "Review", "detail": "parallel review dimensions"},
          {"title": "Verify", "detail": "adversarially verify findings"},
          {"title": "Synthesize"},
      ],
  }

Then define an entrypoint:

  def workflow():
      phase("Review")
      results = pipeline(...)
      phase("Synthesize")
      return agent("Synthesize these results: " + json.dumps(results), {"label": "synthesis"})

The meta dict must be a PURE LITERAL. No variables, function calls, spreads, f-strings, or computed values. Required: name. Recommended: description and phases. meta["phases"] may contain strings or {"title", "detail", "model"} objects. Use the same phase titles in phase() calls; titles match exactly.

Available Python globals:

- agent(prompt: str, opts?: dict) -> any: spawn a standalone Hermes AIAgent child. This plugin does not call Hermes' native delegation tool. Without schema, returns final text. With opts["schema"] containing a JSON Schema, the child submits its final answer through a dedicated workflow_submit_structured_output tool that is validated at the tool layer; on a schema mismatch the child receives the error and retries, so agent() returns the parsed/validated object (or None if the child never produced valid output). If the child cannot use the tool, it falls back to parsing the final message. opts.label overrides the display label. opts.phase explicitly assigns the agent to a progress group; use this inside parallel()/pipeline() stages to avoid races on global phase() state. opts.toolsets chooses child toolsets; default is configured by the plugin, normally web/file/terminal. Use ["all"] only when the workflow truly needs broad tool access; blocked child toolsets still stay disabled. Hermes ToolSearch/MCP tools are exposed through Hermes' normal tool-search bridge when available. opts.agentType loads a Hermes skill or workflow agent-type preset for the child. opts.isolation may be "worktree" to run the child in a per-agent git worktree. opts.model and opts.provider are disabled by default unless the plugin config explicitly allows overrides. opts.timeout_seconds overrides child timeout for this call.
- pipeline(items, stage1, stage2, ...) -> list: run each item through all stages independently. There is NO barrier between stages: item A can be in stage 3 while item B is still in stage 1. This is the DEFAULT for multi-stage work. Each stage receives (prev_result, original_item, index). A stage that fails drops that item to None.
- parallel(thunks: list[callable]) -> list: run callables concurrently and wait for all results. This is a BARRIER. Use only when you genuinely need all results together. Pass callables, not direct agent calls: parallel([lambda: agent("A"), lambda: agent("B")]).
- phase(title: str) -> None: start a progress phase. Subsequent agent() calls are grouped under this title unless opts.phase is set.
- log(message) -> None: append a workflow-level progress log.
- args: any: the JSON value passed as this tool's args input, verbatim. Pass arrays/objects as actual JSON values, NOT as JSON-encoded strings. Use args for target files, research questions, or config values.
- budget: object: exposes Claude-style token budget fields: total, spent(), and remaining(). total is None unless configured by HERMES_DYNAMIC_WORKFLOWS_TOKEN_BUDGET or plugin config token_budget_total. spent() counts completed child-agent tokens in this workflow run; remaining() returns Infinity when total is None. Once spent reaches total, further agent() calls fail.
- subworkflow(name_or_ref, args=None) -> any: run another workflow synchronously as a sub-step. Pass a workflow name or {"scriptPath": "..."}. Child workflows share the parent run's agent counter, stop signal, deadline, token budget, resume cache, and global concurrency slots. Nesting is limited to one level.
- cwd: str, json, math: current working directory string and safe standard helpers.

Scripts are Python, not JavaScript and not TypeScript. Use lambda callables for parallel thunks. Do not import modules; json and math are already provided. Do not read files directly, shell out, call open/eval/exec/compile/input, or access private/dunder attributes. Child agents should use Hermes tools for repository access.

DEFAULT TO pipeline(). Only use a barrier with parallel() when stage N needs cross-item context from all of stage N-1: dedup/merge across the full result set, early exit when total count is zero, or prompts comparing all prior findings. A barrier is not justified by conceptual stage boundaries or cleaner code.

Canonical review pattern:

  meta = {"name": "review-changes", "description": "Review and verify", "phases": [{"title": "Review"}, {"title": "Verify"}, {"title": "Synthesize"}]}
  FINDINGS_SCHEMA = {"type": "object", "required": ["findings"]}
  VERDICT_SCHEMA = {"type": "object", "required": ["isReal", "reason"]}
  DIMENSIONS = [
      {"key": "bugs", "prompt": "..."},
      {"key": "security", "prompt": "..."},
  ]
  def workflow():
      results = pipeline(
          DIMENSIONS,
          lambda d, original, i: agent(d["prompt"], {"label": "review:" + d["key"], "phase": "Review", "schema": FINDINGS_SCHEMA}),
          lambda review, original, i: parallel([
              lambda f=f: agent("Adversarially verify: " + json.dumps(f), {"label": "verify", "phase": "Verify", "schema": VERDICT_SCHEMA})
              for f in (review or {}).get("findings", [])
          ]),
      )
      phase("Synthesize")
      return agent("Synthesize confirmed findings: " + json.dumps(results), {"label": "synthesis"})

Quality patterns:
- Adversarial verify: spawn independent skeptics per finding, each prompted to refute.
- Perspective-diverse verify: assign correctness/security/performance/repro lenses.
- Judge panel: generate independent approaches, score with parallel judges, synthesize from the winner.
- Loop-until-dry: keep spawning finders until consecutive rounds return nothing new, bounded by budget.total and budget.remaining() when a token budget is configured.
- Completeness critic: final agent asks what is missing before synthesis.

Concurrent agent() calls share one workflow-wide cap. By default the cap is min(16, cpu cores - 2), with plugin config/env overrides available. Excess agent() calls queue until a slot frees. Total agent count across the workflow run is capped at 1000 by default as a runaway-loop backstop.

Resume: pass resumeFromRunId to reuse cached agent() results from the unchanged prefix of a previous run. Same script + same args should produce cache hits until the first changed agent prompt/options. Nested workflow agent calls participate in the same global cache sequence.

Save for reuse: after a run does what the user wanted, the user can save its script as a reusable named workflow with /workflows <runId> save <name> [user|project]. That writes the script to .hermes/workflows/<name>.py (project) or the user store and registers a /<name> slash command, so the same orchestration reruns via /<name> or the workflow tool's name input. /workflows <runId> export [path] writes a markdown transcript of a run instead.

Current Hermes plugin behavior: child agents are created directly as standalone Hermes AIAgent instances. They do not use Hermes' native delegation path and do not appear in Hermes' native /agents delegation tree. agentType first tries Hermes' own skill loader, then falls back to workflow agent-type files from .hermes/workflow-agent-types, ~/.hermes/dynamic-workflows/agent-types, or bundled agent-types. Worktree isolation creates a per-child git worktree and binds the child's Hermes file/terminal tools to that workspace with a task-specific cwd override.
"""

DYNAMIC_WORKFLOW_SCHEMA = {
    "description": _DESCRIPTION,
    "parameters": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "script": {
                "type": "string",
                "maxLength": 524288,
                "description": (
                    "Self-contained Python workflow script. Define a literal meta dict "
                    "and a def workflow(): entrypoint using agent()/parallel()/pipeline()/phase()."
                ),
            },
            "scriptPath": {
                "type": "string",
                "description": (
                    "Path to a workflow Python script on disk. Takes precedence over script and name. "
                    "Use this to rerun or iterate on a script saved by an earlier invocation."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Name of a predefined workflow. Resolves from .hermes/workflows/<name>.py, "
                    "~/.hermes/dynamic-workflows/workflows/<name>.py, or bundled workflows."
                ),
            },
            "args": {
                "description": (
                    "Optional JSON value exposed to the script as global args, verbatim. "
                    "Pass arrays/objects directly, not as JSON-encoded strings."
                ),
            },
            "resumeFromRunId": {
                "type": "string",
                "pattern": "^wf_[a-z0-9-]{6,}$",
                "description": (
                    "Run ID of a previous workflow invocation. Unchanged-prefix "
                    "agent() calls return cached results; edited/new calls run live."
                ),
            },
            "description": {
                "type": "string",
                "description": "Ignored. Set workflow description in the script's meta dict.",
            },
            "title": {
                "type": "string",
                "description": "Ignored. Set workflow name/title in the script's meta dict.",
            },
        },
    },
}
