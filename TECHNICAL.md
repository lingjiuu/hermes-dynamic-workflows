# Hermes Dynamic Workflows — Technical Documentation

The model writes a sandboxed Python script on the fly, the background runtime executes
it, and it orchestrates large numbers of independent Hermes subagents with
`agent()/parallel()/pipeline()`. This document explains the implementation point by
point; all result strings are taken verbatim from the source.

## Core Execution Path

The main agent calls the `workflow` tool → `WorkflowRunManager.start_from_params`:

1. Resolve the source (one of `script` / `scriptPath` / `name`).
2. `parse_script` + `extract_meta`: AST validation, extracting the first-statement literal `meta`.
3. Top-level launch approval (`require_launch_approval`, on by default).
4. Persist the script, create the run record, start a background daemon thread, and **the tool returns synchronously** with the Task ID / Run ID / script path.

Background thread `run_workflow`: wraps the script body after `meta` into a private
async entry point, injects the sandboxed globals, and runs it with `exec`. An
`await agent(...)` in the script → `WorkflowAPI` → a concurrency slot →
`HermesChildAgentRunner` spawns an independent `AIAgent` subagent and returns its text
(or the schema-validated object).

On every state change: a snapshot is written to the run record + `journal.jsonl`, and
subagent transcripts are exported in real time. On terminal state: write the output
file, do a final transcript flush, and inject a `<task-notification>` into the main
conversation (CLI only).

On-disk locations (`<cwd>` is the sanitized working directory):

```
~/.hermes/projects/<cwd>/<sessionId>/workflows/scripts/<name>-<runId>.py   # persisted script
~/.hermes/projects/<cwd>/<sessionId>/subagents/workflows/<runId>/          # transcript directory
    journal.jsonl                                                         # run event stream
    agent-<sessionId>.jsonl  +  .meta.json                                # one per subagent
```

## Python Script API

The script body is itself async: write top-level `await` / `return` directly. **The
first statement must be a pure literal `meta = {...}`** (`name` and `description`
required; `whenToUse` and `phases` optional).

| Global | Signature | Description |
|---|---|---|
| `agent` | `await agent(prompt, opts=None)` | Spawns a subagent. Without a schema it returns text; with `schema` it returns the validated object. `opts`: `label` `phase` `schema` `model` `isolation` `agentType`. Returns `None` if skipped by the user. |
| `pipeline` | `await pipeline(items, stage1, …)` | Each item flows through the stages independently, **no barrier**. Stage callbacks receive `(prev, original, index)`; if a stage throws → that item becomes `None`. The default for multi-stage work. |
| `parallel` | `await parallel(thunks)` | Runs concurrently, **with a barrier**: returns only once all complete. A single failure → `None` in the results (the whole call does not throw). |
| `phase` | `phase(title)` | Starts a progress group. |
| `log` | `log(message)` | Sends a line of progress to the user. |
| `workflow` | `await workflow(name_or_ref, args=None)` | Runs another workflow inline, sharing concurrency/counts/stop/budget; one level of nesting only. |
| `args` | — | The tool input `args` verbatim; `None` if not passed. |
| `budget` | `budget.total` / `spent()` / `remaining()` | Taken from a `+500k`-style target in the user's message. `total` is a hard cap — once reached, `agent()` throws; when unset, `remaining()` is `math.inf`. |

Also available: `json`, `math`, safe builtins, and common exception types. **Forbidden**
(rejected by the sandbox): imports, file/process/network access, dunder traversal,
`eval/exec`, class definitions, dynamic call targets, and time/randomness APIs (they
break resume).

## Tools

The plugin registers two main-agent tools with Hermes (`workflow`, `task_stop`) and one
subagent-only tool (`structured_output`, registered temporarily only while a
schema-bearing subagent is alive).

### workflow

Executes an orchestration script in the background; returns synchronously and injects a
`<task-notification>` on completion.

Tool schema (the description is very long and elided here with `…`; the parameters are
shown in full below):

```json
{
  "description": "Execute a workflow script that orchestrates multiple subagents …",
  "parameters": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "script":          { "type": "string", "maxLength": 524288, "description": "Self-contained workflow script. Must begin with literal `meta = {…}` …" },
      "scriptPath":      { "type": "string", "description": "Path to a workflow script on disk. Takes precedence over `script` and `name`." },
      "name":            { "type": "string", "description": "Name of a predefined workflow (built-in or from .hermes/workflows/)." },
      "args":            { "description": "Value exposed to the script as global `args`, verbatim. Pass real JSON, not a JSON-encoded string." },
      "resumeFromRunId": { "type": "string", "pattern": "^wf_[a-z0-9-]{6,}$", "description": "Run ID to resume from; unchanged agent() calls return cached results." },
      "description":     { "type": "string", "description": "Ignored — set it in the script's meta block." },
      "title":           { "type": "string", "description": "Ignored — set it in the script's meta block." }
    }
  }
}
```

At launch, the list of currently available agentTypes is also appended to the end of the
description.

**Tool call results.** Launch is synchronous — the results below are returned before the
background thread starts, at which point there is **no** notification yet (the run has
not begun).

On successful launch:

```
Workflow launched in background. Workflow Task ID: <taskId>
Summary: <meta.description or meta.name>
Transcript dir: <…/subagents/workflows/<runId>>
Script file: <…/scripts/<name>-<runId>.py>
Run ID: <runId>
To resume after editing the script: Workflow({scriptPath: "<path>", resumeFromRunId: "<runId>"})
You will be notified when it completes. Use /workflows to watch live progress.
```

Validation/parse/input errors are uniformly wrapped as `{"error":"<msg>"}` (where
`<msg>` is one of the following):

```
# missing source
provide one of script, scriptPath, or name

# meta contract (the first statement must be a pure literal meta dict)
Invalid workflow script: `meta = {...}` must be the FIRST statement in the script
Invalid workflow script: meta must be a pure literal
Invalid workflow script: meta must be a pure literal: only plain properties allowed in meta
Invalid workflow script: meta must be a pure literal: template interpolation not allowed in meta
Invalid workflow script: meta must be a pure literal: non-literal node type in meta: <NodeType>
Invalid workflow script: meta.name must be a non-empty string
Invalid workflow script: meta.description must be a non-empty string
Invalid workflow script: meta keys must be strings
Invalid workflow script: forbidden meta key: <key>
Invalid workflow script: meta.<name|description|whenToUse> must be a string
Invalid workflow script: meta.phases must be a list
Invalid workflow script: meta.phases object entries require a title string
Invalid workflow script: meta.phases.<detail|model> must be a string
Invalid workflow script: meta.phases entries must be strings or objects

# parsing / size
Invalid workflow script: Script parse error: <syntax msg> at line <l>, column <c>. Workflow scripts must be plain Python.
Invalid workflow script: workflow script is too large (<n> chars; max <max>)
do not define workflow(); the workflow script body is already async

# sandbox (capability overreach)
forbidden Python syntax: <NodeType>
forbidden name: <name>
forbidden attribute access: <attr>
forbidden method call: <attr>
dynamic call targets are not allowed
workflow script is too complex (>2500 AST nodes)
string literal is too large
integer literal is too large
bare 'except:' is not allowed; catch Exception or a specific type
'except BaseException' is not allowed; catch Exception instead
Workflow scripts must be deterministic: current time and randomness are unavailable (breaks resume). Stamp results after the workflow returns, or pass timestamps via args.

# resume target still running
Workflow <runId> is still running (task <taskId>). Stop it first with task_stop({"task_id":"<taskId>"}) before resuming.
```

A failed launch approval also goes through `tool_error` (clean, no trace):

```json
{"error":"Workflow \"<name>\" was not launched: <reason>. Do not retry; tell the user it needs their approval."}
```

`<reason>` comes from the approval step; common values: `workflow launch was denied`,
`workflow launch was denied or timed out`, `launch approval required but no interactive
channel (…)`, `launch approval required but Hermes' approval engine is unavailable`.

Only other **unexpected** exceptions (genuine internal errors) return diagnostics that
include a trace — `trace` is the last 8 lines of `traceback.format_exc` (file paths,
line numbers, offending code), visible to the model alongside the tool result to aid
reporting: `{"error":"<Type>: <msg>","trace":"<last 8 lines of traceback>"}`.

**Completion notification.** Once the run reaches a terminal state (CLI only), this is
injected:

```
<task-notification>
<task-id><taskId></task-id>
<output-file><path></output-file>        # when an output file exists
<status><completed|failed|stopped|…></status>
<summary>Dynamic workflow "<name>" <completed | was stopped | failed: <error> | <status>: <error>></summary>
<result><the result; truncated past notify_result_preview_chars with a "full result in <file>" note></result>   # when there is no error
<recovery>Agent transcripts: <transcriptDir></recovery>      # when there is an error
<usage><agent_count>N</agent_count><subagent_tokens>T</subagent_tokens><tool_uses>U</tool_uses><duration_ms>D</duration_ms></usage>
</task-notification>
```

### task_stop

Stops a background run by its Task ID (only affects live runs; completed/historical runs
are treated as not found).

Tool schema:

```json
{
  "description": "- Stop a running workflow by its Task ID …",
  "parameters": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": false,
    "properties": { "task_id": { "type": "string", "description": "The ID of the workflow task to stop" } },
    "required": ["task_id"]
  }
}
```

Tool call results:

```jsonc
// success (compact JSON)
{"message":"Successfully stopped task: <taskId> (<summary>)","task_id":"<taskId>","task_type":"local_workflow"}

// missing parameter
{"error":"Missing required parameter: task_id"}

// not found / not in a live state (not queued|running|paused)
{"error":"No task found with ID: <taskId>"}
```

### structured_output (subagent-only)

Registered temporarily only while a schema-bearing subagent is alive; this tool's
parameters are replaced with the schema requested in `agent(…, {"schema"})`, and the
subagent must call it to submit its final result. `agent()` returns the validated object
— no need to parse the subagent's text. At most `MAX_STRUCTURED_OUTPUT_RETRIES = 5`
attempts.

If a subagent tries to finish without submitting, a continue instruction is appended and
it is kept in the same session:

```
You MUST call the structured_output tool to complete this request. Call this tool now.
```

Tool call results:

```jsonc
// success (plain text, not JSON)
Structured output provided successfully

// validation failure: errors are the individual items joined by ", "
{"error":"Output does not match required schema: <errors>"}
//   each item looks like: /path/to/field: must have required property 'x'
//             /path: must be <type>      root: must NOT have additional properties
//             /path: must be equal to one of the allowed values     /path: must match pattern "…"

// no expectation registered (should not occur in theory)
{"error":"Output does not match required schema: root: no structured-output expectation is registered for this task"}

// maximum attempts exceeded
{"error":"Output does not match required schema: root: maximum structured output attempts exceeded (5)"}
```

> Validation prefers `jsonschema` (Draft 2020-12); when it's not installed, it falls back
> to a built-in lightweight validator (covering common keywords like
> object/array/string/number, required, enum, additionalProperties).

## Prompt Cache

Subagents are independent `AIAgent`s and inherit Hermes's prompt caching: for cacheable
models (the Claude family, DashScope Qwen, …) a `cache_control` breakpoint is injected,
and each subagent reuses the `[tools + system]` prefix across its own multi-turn tool
calls.

**Sharing the prefix across subagents**: Hermes's `system_and_3` strategy places the
breakpoint at the end of the system prompt. To make this work, **the subagent system
prompt stays byte-for-byte identical across the entire fan-out** — it contains only
stable scaffolding (base instructions + agentType instructions), while per-task context
(workspace, label, phase, worktree hints) goes into the subagent's first user message
(`build_child_task_message`). Subagents with the same toolset + same agentType therefore
share the cache prefix. The savings depend on the provider (0 for non-cacheable models).

## Concurrency and Limits

- **Concurrency slots**: one semaphore per run, capped at `concurrency` (default
  `min(16, cpu-2)`, and ≤ `max_concurrency`=16). `parallel()/pipeline()` can submit any
  number of items, but only about slot-many run at once and the rest queue.
- **Agent cap** `max_agents` (default 1000): a runaway fallback gate, far above any real
  workflow.
- **Loop gate**: each `while/for` iteration injects `__wf_tick__()`, which checks for
  stop / deadline / the loop cap `max_loop_iterations` (default 1e7). This lets the
  deadline fire even inside a pure compute loop.
- **Deadline** `workflow_timeout_seconds` (default 900s, paused time not counted).
- **Subagent timeout** `child_timeout_seconds` (default 300s): a single timeout raises
  `WorkflowTimeout` back into the script (catchable with `try/except`).

Run-level hard stops (user stop, deadline, budget/agent/loop caps) derive from
`BaseException`, so the script's `except Exception` cannot swallow them; the sandbox also
forbids `except:` / `except BaseException`.

## Permission Governance

Three layers, all reusing Hermes's own approval engine rather than rebuilding one:

1. **Launch approval** (`require_launch_approval`, on by default): before a top-level
   launch — the CLI confirms synchronously; the gateway sends approve/deny buttons and
   blocks; unattended (no channel) means denial. A nested `workflow()` inherits the
   already-approved parent run and is not approved separately.
2. **Subagent command approval** (`child_approval_policy`): subagent tool calls go
   through the Hermes approval engine as usual (dangerous-command detection, the hardline
   floor, the permanent allowlist, yolo, gateway async approval). This key only decides
   what happens when a flagged command would normally prompt a human but no one is
   present (a background run): `inherit` (follow Hermes `approvals.mode`, default) /
   `smart` (an assisting LLM guard, recommended for unattended runs) / `deny` / `approve`
   / `ask` (ask a human if a channel exists, otherwise fall back to `ask_fallback`).
3. **The pre_tool_call hook**: background subagents run in threads detached from the
   session context, where Hermes's own approval would, lacking context, either wrongly
   wave commands through or hang. The hook applies the policy above before Hermes's
   context branch; when it lets something through it also `approve_session()`s that
   pattern, so downstream re-gating doesn't turn the command into a pending with no one
   to answer it. The hardline floor and the permanent allowlist always remain in effect.

## Transcript (Rebuilding the Execution Trace from state.db)

Subagents are independent `AIAgent`s, and their messages land in Hermes's `SessionDB`
(SQLite). So that users / the main agent can see each subagent's full execution trace,
the runtime exports these messages as `agent-<sessionId>.jsonl` (+ a `.meta.json`
sidecar of the same name).

- **Incremental read** (`SessionTranscriptReader`): reads the `messages` / `sessions`
  tables directly. It first uses a recursive CTE to resolve the **compaction lineage**
  (when a subagent is context-compacted it spawns a new session, chained into one
  lineage), then uses `(row count, min/max id, active count and sum of ids)` to decide
  whether it's append-only — if so it only appends new messages, otherwise it rebuilds
  wholesale.
- **Full fallback**: on schema mismatch / unavailable private connection /
  incremental-read exception, it falls back to the public API
  `get_messages(include_inactive=True)` and uses a content signature to detect changes.
- **Live export** (`LiveTranscriptExporter`): refreshes active subagents every 0.5s, with
  a single reader + single writer running serially (avoiding contention between multiple
  SQLite connections and the atomic temp file); flushes once more when a subagent reaches
  a terminal state; and does one final, validated rebuild when the run ends.

`/workflows` and the dashboard use this to show each subagent's prompt, recent tool
activity, and output, **without depending** on the final output file.

## Sandbox and Determinism

After AST validation the script is `exec`'d with restricted globals; what's gated is
**capability**, not **control flow**: `if/for/while/try` are allowed (needed for
loop-until-budget / loop-until-dry), but imports, file/process/network access, dunder
traversal, `eval/exec/compile/open/getattr…`, class definitions, and dynamic call
targets are all rejected; time/randomness APIs are forbidden (they break resume). This is
a guardrail, not a perfect sandbox — true isolation needs subprocesses + RPC (a future
step).

## Resume / Content-Addressed Cache

`resumeFromRunId` reuses the **unchanged** `agent()` results from the previous run.
Fingerprints are content-addressed (prompt + relevant opts), so even if the concurrent
scheduling order changes, unchanged calls still hit. When editing the script, try to keep
the early, stable `agent()` calls: an early change flows downstream into later prompts and
reduces reuse, while a late change preserves more of the cache.

## Token Budget

`budget.total` is parsed from a target in the current user message (`+500k`,
`spend 2M tokens`, `use 1B tokens`, …); `None` if not stated. `spent()` is the token
count (input + output + reasoning) of this run's completed subagents. Once `total` is
reached, `agent()` raises `WorkflowLimitExceeded` (a run-level hard stop). The scope is
**a single run**, not Claude Code's per-turn shared pool — the boundary a standalone tool
ought to have. Tool inputs / `meta` / config / environment cannot set `total`.

## agentType / worktree / Named Workflows

- **agentType**: `agent(agentType="…")` loads subagent instructions from a workflow agent
  file. Resolution order: project
  `.hermes/dynamic-workflows/agents/<name>.{md,yaml,json}` → user
  `~/.hermes/dynamic-workflows/agents/<name>.…` → the plugin's built-in
  `agents/<name>.md`. Markdown supports YAML frontmatter (`model` / `toolsets` /
  `isolation`, …). Built-in: `explore`, `general-purpose`, `plan`, `verification`.
- **worktree**: `agent(isolation="worktree")` runs each subagent in its own git worktree,
  preventing conflicts from concurrent edits to the same checkout. This is workspace
  isolation, not a security sandbox; the worktree is deleted after use by default
  (`keep_worktrees` off).

## Control (Pause / Resume / Stop / Restart)

The standalone dashboard `hermes-workflows` (in a separate terminal) sends control back
to the Hermes process that owns the run via an **owner-scoped, expiring request/response
queue** (kept under the plugin store, opening no local port):

- `x` stops the run and interrupts its active subagents.
- `p` cooperative pause/resume: while paused, no new subagents or subsequent pipeline
  stages start (those already running can finish), and paused time does not count toward
  the deadline.
- `r` restarts the whole thing as a brand-new run with a new Run ID, using the saved
  script and args.
- `s` saves a markdown transcript.

## Configuration

No separate config file: the plugin reads the
`plugins.entries.dynamic-workflows.dynamic_workflows:` section from Hermes's
`config.yaml`, and supports `HERMES_DYNAMIC_WORKFLOWS_*` environment variable overrides.
For the keys, defaults, and meanings, see the Configuration section of the README.