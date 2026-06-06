# hermes-dynamic-workflows

Claude-Code-style dynamic workflow orchestration for Hermes.

This plugin adds a `workflow` tool that lets the model run a saved
Python workflow script. The script coordinates many child agents with a small,
deterministic API:

- `await agent(prompt, opts=None)`
- `await parallel([lambda: ...])`
- `await pipeline(items, stage1, stage2, ...)`
- `phase(name)`
- `log(message)`
- `await workflow(name_or_ref, args=None)`
- `args`, `budget`

Runs are started in the background. The tool immediately returns a Task ID, a
`runId`, and the persisted script path. When a run finishes, the plugin injects a
Claude-Code-style `<task-notification>` (status + truncated result + usage) back
into the conversation via `ctx.inject_message`, so the model can report the
outcome without polling. The model can stop a live workflow with
`task_stop({"task_id": "<Task ID>"})`. Completion notification is CLI-only
(no-op in gateway) and can be turned off with `notify_on_complete: false`.
Inspect or stop runs with slash commands:

```text
/workflows
/workflows wf_abc123...
/workflows wf_abc123... phase 1
/workflows wf_abc123... agent 2
/workflows wf_abc123... save <name> [user|project]
/workflows wf_abc123... export [path]
/workflow-stop wf_abc123...
```

Or open the standalone, automatically refreshing terminal panel from another
terminal tab:

```bash
hermes-workflows
```

The panel follows the Claude Code workflow views: run list, phase/agent
progress, and per-agent prompt, recent tool activity, and outcome. It reads the
persisted run snapshots, live `journal.jsonl`, and live child-agent transcript
JSONL files; it does not depend on the final task output file.

The standalone panel also sends controls back to the Hermes process that owns a
run:

- `x`: stop the selected workflow and interrupt its active child agents.
- `p`: cooperatively pause/resume. Active agents may finish while paused, but
  no new agents or later pipeline stages start until resume; paused time does
  not count against the workflow deadline.
- `r`: restart the entire workflow from its saved script and args as a fresh
  run with a new Run ID.
- `s`: save a markdown transcript.

Controls use an owner-scoped, expiring request/response queue under the plugin
store, so a second terminal can control the correct Hermes process without
opening a local network port. Existing runs created before the control-capable
plugin version do not have a live control owner; use `task_stop` or
`/workflow-stop` for those runs.

`save` persists the run's script as a reusable named workflow and registers a
`/<name>` slash command for it (project scope writes `.hermes/workflows/<name>.py`,
`user` scope writes the user store). `export` writes a markdown transcript of the
run instead.

## Package Layout

```text
hermes_dynamic_workflows/
  entry.py    plugin registration: register(ctx) wires the tools, command, and hook
  core/       leaf layer: types, errors, config, JSON-schema validation, text/token helpers
  host/       anti-corruption layer — the ONLY package that imports Hermes internals
  child/      standalone child-agent runner, agent-type presets, worktrees, structured_output tool
  agents/     bundled workflow agent-type definitions (Markdown/YAML/JSON)
  engine/     workflow-script runtime, execution context, agent()/parallel()/pipeline() API, sandbox, resume cache
  run/        background run manager (lifecycle/persistence/notifications) + live transcript export
  storage/    persisted scripts, runs, control queue, named-workflow lookup
  view/       pure text/markdown rendering of run snapshots
  adapters/   Hermes-facing tool handlers (workflow, task_stop), /workflows command, approval hook
  tui/        standalone full-screen workflow monitor (the `hermes-workflows` console script)
```

Dependencies point one way only — a layer may import the ones below it, never above:

```text
adapters  →  run  →  engine  →  child  →  storage  →  core
                 ↘  view (pure rendering)        ↘  host (Hermes port)
```

Two invariants keep the graph acyclic and the Hermes coupling contained:

- **Only `host/` imports Hermes internals** (`run_agent`, `hermes_cli.*`,
  `hermes_state`, `hermes_constants`, `gateway.*`, `tools.*`, `model_tools`), so a
  Hermes rename is a one-file change. `host/` currently owns the session/gateway/
  home surface; the child-construction surface still inside `child/runner.py` and
  `adapters/hooks.py` is migrating in.
- **`core/`, `host/`, and `view/` import nothing upward**, so any layer can depend
  on them without risking an import cycle.

The package root stays intentionally thin, and its public imports are lazy
(`__getattr__`), so importing one helper module does not boot the whole runtime.

## Install

For local development from this checkout:

```bash
hermes plugins install file:///Users/Apple/code/MyProjects/hermes-dynamic-workflows --enable --force
hermes tools enable workflow --platform cli
```

For gateway surfaces, enable the toolset for the target platform and restart
the gateway:

```bash
hermes tools enable workflow --platform telegram
hermes gateway restart
```

After publishing the repository, users should be able to install it like any
other Hermes plugin:

```bash
hermes plugins install owner/hermes-dynamic-workflows --enable
hermes tools enable workflow --platform cli
```

`hermes plugins install` clones the plugin into `~/.hermes/plugins`; it does not
install Python dependencies declared in `pyproject.toml`. For full JSON Schema
support in structured workflow outputs, install `jsonschema` into the same Python
environment that runs Hermes:

```bash
python -m pip install "jsonschema>=4,<5"
```

The plugin has a fallback validator for simple schemas, and will return a clear
error if a complex schema needs the full `jsonschema` package. If the plugin is
installed as a Python package, `jsonschema>=4,<5` is declared as a dependency.

A Python package install also exposes `hermes-workflows` directly. Because
`hermes plugins install` clones a plugin without installing its console scripts,
run the wrapper installer once from a cloned/plugin checkout:

```bash
python scripts/install-hermes-workflows.py
```

This writes a small launcher to `~/.local/bin`; make sure that directory is on
`PATH`. Python package installs bring in `windows-curses` on Windows; cloned
plugin installs on Windows should install it into the Hermes Python environment
if `import curses` is unavailable.

## Tool Inputs

The model can call `workflow` with one of these script sources:

- `script`: inline Python workflow source. This is persisted automatically under
  `~/.hermes/projects/<sanitized-cwd>/<sessionId>/workflows/scripts/<meta-name>-<runId>.py`.
- `scriptPath`: path to a saved workflow script. Relative paths resolve from the
  current working directory. Reruns reuse the same script file instead of
  creating a new run-specific copy.
- `name`: predefined workflow name. Resolves from `.hermes/workflows/<name>.py`,
  then `~/.hermes/dynamic-workflows/workflows/<name>.py`, then bundled workflows.

The tool result reports a transcript directory at
`~/.hermes/projects/<sanitized-cwd>/<sessionId>/subagents/workflows/<runId>`
and creates `journal.jsonl` there when the workflow starts. Child-agent
transcripts are exported there as JSONL files as agent sessions become known,
with a final flush after the workflow finishes.

Other inputs:

- `args`: any JSON value exposed to the script as global `args`.
- `resumeFromRunId`: reuse cached `agent()` results from the unchanged prefix of
  a previous run.
- `description` and `title`: accepted for Claude schema compatibility, ignored
  by the runtime. Put metadata in the script's `meta` dict.

## Workflow Script Shape

```python
meta = {
    "name": "repo-audit",
    "description": "Parallel audit and final synthesis",
    "phases": [
        {"title": "Review", "detail": "parallel review"},
        {"title": "Verify", "detail": "adversarial verification"},
        {"title": "Synthesize"},
    ],
}


phase("Review")

findings = await pipeline(
    args["targets"],
    lambda target, original, i: agent(
        "Review this target for bugs: " + target,
        {"label": "review:" + str(i), "phase": "Review"},
    ),
    lambda review, original, i: agent(
        "Verify this review adversarially: " + json.dumps(review),
        {"label": "verify:" + str(i), "phase": "Verify"},
    ),
)

phase("Synthesize")
return await agent(
    "Synthesize the verified findings:\n" + json.dumps(findings, ensure_ascii=False),
    {"label": "synthesis"},
)
```

The workflow script is already an async body: write top-level `await` and
`return` directly, and do not define a `workflow()` entrypoint. The first
statement must be a pure-literal `meta` dict with required `name` and
`description` fields. `phases` is optional. `meta["phases"]` may contain strings or
`{"title", "detail", "model"}` objects.

## Child Agent Strategy

The plugin creates standalone Hermes `AIAgent` children directly. It does not
call Hermes' native delegation tool and does not maintain a native-delegation
compatibility path. This gives the workflow runtime direct control over child
prompts, `agentType`, worktree isolation, stop handling,
and workflow-specific status snapshots.

Standalone children do not appear in Hermes' native `/agents` delegation tree.
Use `/workflows` for workflow history and inspection.

## API Notes

Use `pipeline()` by default for multi-stage work. It lets each item advance
through all stages independently. Use `parallel()` only when a real barrier is
needed, such as merging all first-stage results before continuing.

`await workflow(name_or_ref, args=None)` runs another workflow inline as a child
frame. Child workflows share the parent run's agent counter, stop signal,
deadline, token budget, resume cache, and global concurrency slots. Nesting is
limited to one level.

`agent(prompt, opts)` supports:

- `label`: display label.
- `phase`: explicit progress group.
- `agentType`: load child instructions from a workflow agent file.
- `isolation`: set to `worktree` to run the child in a per-agent git worktree.
- `schema`: JSON Schema. The child must submit its final answer through the
  child-only `structured_output` tool. The tool's parameters are replaced with
  the requested schema for that child. Schema mismatches are returned to the
  child so it can correct and retry; a child that tries to finish without a
  valid submission is continued in the same session. `agent()` returns the
  validated object without parsing the child's final message.
- `model`: routes this agent to a specific model (like Claude Code's per-agent
  model option). Allowed by default; the default is still the session model, so
  set it only when a stage wants a different tier.

Runtime policy is intentionally not part of the public `agent()` API. Tool access
comes from `default_child_toolsets`, `blocked_child_toolsets`, and `agentType`
presets. Provider selection comes from Hermes' model/provider configuration.
Child timeout comes from `child_timeout_seconds`. Whole-child retry policy should
be expressed as explicit workflow control flow when a workflow truly needs it.

`agentType` resolution order:

1. Project files: `.hermes/dynamic-workflows/agents/<name>.md|yaml|json`.
2. User files: `~/.hermes/dynamic-workflows/agents/<name>.md|yaml|json`.
3. Bundled plugin files: `hermes_dynamic_workflows/agents/<name>.md|yaml|json`.

Markdown workflow agent files may use YAML frontmatter:

```markdown
---
name: researcher
model: inherit
toolsets: [web, file]
isolation: worktree
---

You are a focused research child agent...
```

Omitting `model` or setting `model: inherit` uses the launching main agent's
active runtime when available, including an in-session `/model` switch. Direct
or headless launches with no parent agent fall back to the configured Hermes
runtime. The inherited runtime is held in memory only; credentials are never
written to workflow run records, journals, or transcripts. Set a concrete
model only when the agent should intentionally route elsewhere.

`budget` is populated from a Claude-style token target in the current user
message, such as `+500k`, `spend 2M tokens`, or `use 1B tokens`:

- `budget.total`: parsed user token target, or `None` when the current user
  message contains no supported target.
- `budget.spent()`: completed child-agent tokens (input+output+reasoning) in the
  current workflow run.
- `budget.remaining()`: remaining tokens, or `math.inf` when no total is set.

The tool input, workflow `meta`, plugin config, and environment cannot set
`budget.total`. Once `spent()` reaches `total`, further `agent()` calls fail.
The runtime also keeps a separate 1000-agent runaway backstop by default.

The scope is intentionally per-run (this workflow), not the shared per-turn pool
Claude Code uses — the right unit for a standalone tool that bounds one
orchestration.

## Prompt Caching

Child agents are standalone Hermes `AIAgent`s, so they inherit Hermes' prompt
caching automatically (`agent/prompt_caching.py`): for cache-eligible models
(Claude-family, DashScope Qwen, …) the runtime injects `cache_control`
breakpoints, so each child reuses its system+tools+history prefix across its own
tool-calling turns. The plugin does not disable it.

**Cross-child prefix sharing.** Hermes' `system_and_3` strategy puts the prefix
cache breakpoint at the end of the system prompt, which caches `[tools +
system]`. So the child system prompt is kept **byte-identical across the
fan-out** — it carries only the stable scaffolding (base instructions +
agent-type instructions), while per-task context (workspace, label, phase,
worktree note) goes in the child's first user message
(`build_child_task_message`). Children sharing a toolset and agent-type then
share the cached `[tools + system]` prefix within the cache TTL. Measured on
DashScope Qwen: two children with different labels and different tasks but the
same `file` toolset — the second read 2513 of 2607 prompt tokens (~96%) from
cache.

Per-agent `cache_read`/`cache_write` counts surface in `/workflows <runId> agent
<id>`, and the run header shows total cached-read tokens. The savings are
provider-dependent (zero on non-eligible models, e.g. custom `chat_completions`
endpoints).

## Config

Optional `config.yaml` block:

```yaml
plugins:
  entries:
    dynamic-workflows:
      dynamic_workflows:
        # Defaults to min(16, cpu cores - 2) when omitted.
        concurrency: 8
        max_concurrency: 16
        max_agents: 1000
        mcp_discovery_wait_seconds: 0.75
        workflow_timeout_seconds: 900
        child_timeout_seconds: 300
        default_child_toolsets: [web, file, terminal, skills]
        keep_worktrees: false
        require_launch_approval: true   # ask before a top-level launch (default on)
        allow_model_override: true   # per-agent model routing (default on)
        # inherit -> follow Hermes' approvals.mode (default); smart -> _smart_approve
        # guardian (recommended unattended); deny -> refuse flagged; approve -> allow
        # (hardline still blocked); ask -> prompt if a human is reachable, else fall back
        child_approval_policy: inherit
        ask_fallback: smart   # what 'ask' degrades to when no human is reachable
```

Environment overrides:

```bash
HERMES_DYNAMIC_WORKFLOWS_CONCURRENCY=4
HERMES_DYNAMIC_WORKFLOWS_MAX_CONCURRENCY=16
HERMES_DYNAMIC_WORKFLOWS_KEEP_WORKTREES=0
HERMES_DYNAMIC_WORKFLOWS_REQUIRE_LAUNCH_APPROVAL=0  # disable launch gate (automation)
HERMES_DYNAMIC_WORKFLOWS_ALLOW_MODEL_OVERRIDE=0
HERMES_DYNAMIC_WORKFLOWS_CHILD_APPROVAL_POLICY=inherit  # inherit|smart|deny|approve|ask
HERMES_DYNAMIC_WORKFLOWS_ASK_FALLBACK=smart  # what 'ask' degrades to (smart|deny|approve)
```

## Workflow Launch Approval

A single workflow can spawn many child agents and spend real tokens, so — like
Claude Code, which shows a permission card before every workflow — a top-level
launch is gated by `require_launch_approval` (**on by default**). Before the run
starts, the launching session is asked to approve:

- **CLI**: a synchronous confirm prompt.
- **gateway** (Telegram/Discord/…): approve/deny buttons; the launch blocks
  until the user responds (reuses the same approval flow as `ask`).
- **headless / unattended** (no approval channel): the launch is **refused** —
  set `require_launch_approval: false` (or
  `HERMES_DYNAMIC_WORKFLOWS_REQUIRE_LAUNCH_APPROVAL=0`) for automation/cron.

Denied or timed-out launches do not start (the tool returns a "not launched"
message; the model should tell the user, not retry). Only **top-level** launches
are gated — nested `workflow()` calls run under an already-approved parent. The gate runs
in the launching turn (foreground), so it needs no cross-thread propagation.

## Child Agent Approvals

Child agents run their tool calls through Hermes' own approval engine
(`tools/approval.py`): dangerous-command detection, the hardline floor
(`rm -rf /`, `mkfs`, fork bombs, …), the permanent allowlist, yolo, and
gateway async approval all apply exactly as they do for any other agent. The
plugin does not reimplement any of that.

The only workflow-specific decision is what a child does when a *flagged*
command would otherwise prompt a human who is not present (the run is in the
background). That is governed by `child_approval_policy`:

- `inherit` (default): follow Hermes' own `approvals.mode` — `manual` → `ask`,
  `smart` → `smart`, `off` → `approve`. One place to set your approval posture
  (Hermes config) and workflow children follow it.
- `smart`: defer to Hermes' `_smart_approve` auxiliary-LLM guardian — it lets
  through flagged-but-safe commands (e.g. `rm -rf ./build`) and refuses genuinely
  destructive ones. Costs one auxiliary-LLM call **per flagged command only**
  (safe commands never reach the LLM). `escalate` (uncertain) resolves to deny
  since no human is present. **Recommended for unattended/gateway runs** — it
  lets children actually work without a human in the loop.
- `deny`: refuse flagged commands. Safest; may block legitimate-but-flagged
  commands such as some test/build invocations.
- `approve`: allow flagged commands. The hardline floor is still enforced
  upstream by Hermes, so unrecoverable commands remain blocked.
- `ask`: ask the user for mid-run approval *if a live approval channel exists*.
  In a gateway session with the launching turn still open, the flagged command is
  routed to the originating user as approve/deny buttons and the child blocks
  until they respond. **But a workflow runs detached: by the time a child hits a
  flagged command, the launching turn (and the gateway notify bridge) is usually
  gone** — so `ask` then degrades to `ask_fallback` (`smart` by default; may be
  `deny` or `approve`) rather than orphaning the command as an unanswerable
  "pending". In CLI a background child can't grab the synchronous prompt either,
  so `ask` degrades there too. Net: `ask` ≈ "ask a human when one is actually
  reachable, otherwise fall back" — for reliable unattended behavior prefer
  `smart` directly.

When a flagged command is *allowed* by policy, the hook also `approve_session()`s
its pattern so the decision sticks past Hermes' own downstream re-gating (which,
for a detached gateway child, would otherwise re-flag the command and turn it
into an unanswerable "pending").

The policy is authoritative in every context. Under CLI it is enforced by the
per-thread approval callback. Workflow children run in detached background
threads that don't carry the session's interactive/gateway context, so under
headless (and contextvar-based gateway) Hermes' own approval would otherwise
auto-approve or orphan a flagged command — a `pre_tool_call` hook closes that
gap by applying the policy before Hermes' context-dependent branching (deferring
to the CLI callback when interactive, to avoid double-evaluating). The hardline
floor and the permanent allowlist still apply.

This is intentionally a plugin-owned key, independent of Hermes'
`delegation.subagent_auto_approve`, so a workflow's wide fan-out is controlled
separately from native delegation.

## Current Limits

- Workflow scripts are Python, not JavaScript.
- Scripts are AST-validated and run with restricted globals. The validator gates
  **capability** (no import / file / shell / network / dunder traversal — all
  world-access goes through child agents and Hermes' approval engine), not
  **control flow**: `if`/`for`/`while`/`try` are allowed, so loop-until-budget
  and loop-until-dry work. A bare `except:` / `except BaseException` is rejected,
  and run-level halts (user stop, the workflow deadline, and the token/agent/
  loop-iteration limits) derive from `BaseException` so `except Exception` cannot
  swallow them — the run stays cancellable and bounded. This is a guardrail, not
  a perfect security sandbox (true isolation would be a subprocess + RPC, a
  documented future step).
- Worktree isolation is workspace isolation, not a security sandbox. It prevents
  parallel children from editing the same checkout by default, but tool safety
  still depends on Hermes' terminal/file-tool policies and approval settings.
- Child agents cannot recursively use workflow, delegation, memory, messaging,
  clarify, or code-execution toolsets by default.
- The default workflow subagent receives read-only skill access
  (`skills_list`/`skill_view`, never `skill_manage`) and uses Hermes Tool Search
  for installed MCP/plugin tools outside the blocked toolsets. `workflow` and
  `task_stop` remain main-agent-only.
