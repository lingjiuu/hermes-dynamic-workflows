# Hermes Dynamic Workflows

> **Claude-Code-style dynamic workflows for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

English | [简体中文](./README.zh-CN.md) | 日本語

You can now use **Dynamic Workflows** in Hermes: have the model write a sandboxed Python
script on the fly, execute it in the background runtime, and orchestrate large numbers
of independent subagents with `agent()/parallel()/pipeline()` — ideal for codebase
audits, large-scale migrations, and cross-validated research. Inspired by
[Dynamic Workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code).

https://github.com/user-attachments/assets/06ef3d0d-4d89-48c4-9851-e1cae690e9b0

## Quick Start

Install and enable in one line:

```bash
hermes plugins install lingjiuu/hermes-dynamic-workflows --enable
```

> Gateway users: run `hermes gateway restart` after installing.

Once it's installed, just tell Hermes "run a workflow that …" and you're set.

### Live Dashboard (optional, requires a separate step)

`hermes plugins install` only clones the plugin — it does not install its console
scripts, so the dashboard command has to be installed once separately:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/plugins/dynamic-workflows/scripts/install-hermes-workflows.py"
# Installs to ~/.local/bin
```

Then, in **a separate terminal**, run `hermes-workflows` to open the interactive
dashboard, where you can watch the run list, per-phase/per-agent progress, and each
subagent's prompt and output in real time.

## Configuration (optional)

The plugin reads the following section from Hermes's `~/.hermes/config.yaml` (every key
can also be overridden via a `HERMES_DYNAMIC_WORKFLOWS_*` environment variable):

```yaml
plugins:
  entries:
    dynamic-workflows:
      dynamic_workflows:
        concurrency: 8                # Max concurrent agents (default: min(16, cpu-2))
        max_concurrency: 16           # Hard cap on concurrency
        max_agents: 1000              # Max total agents per run (runaway guard)
        workflow_timeout_seconds: 900 # Wall-clock timeout for the whole run (excludes paused time)
        child_timeout_seconds: 300    # Timeout for a single child agent
        blocked_child_toolsets: [workflow, delegation, code_execution, memory, messaging, clarify]
                                      # Toolsets child agents are forbidden to use
        default_child_toolsets: [web, file, terminal, skills]
                                      # Default toolsets for child agents (used when no agentType is given)
        keep_worktrees: false         # Whether to keep each agent's git worktree (auto-cleaned by default)
        allow_model_override: true    # Whether agent(model=...) may override the model
        require_launch_approval: true # Require confirmation before a top-level workflow launches (denied if nobody is online)
        child_approval_policy: inherit # Child agent approval policy: inherit|smart|deny|approve|ask
        ask_fallback: smart           # Fallback when "ask" has no one to reach: smart|deny|approve
        notify_on_complete: true      # Notify the originating CLI or gateway session on completion
        notify_result_preview_chars: 2000  # Truncation length (chars) for the result preview in notifications
```

## Script API

A workflow script is just a piece of async Python whose first statement is a literal
`meta`; after that you orchestrate child agents using the sandboxed globals:

```python
meta = {
    "name": "repo-audit",
    "description": "Parallel review, then adversarial verify",
    "phases": [{"title": "Review"}, {"title": "Verify"}],
}

# Each target flows through review → verify independently
# (pipeline has no barrier: A can be at verify while B is still at review)
findings = await pipeline(
    args["targets"],
    lambda t, _o, i: agent(f"Review for bugs: {t}", {"label": f"review:{i}", "phase": "Review"}),
    lambda r, _o, i: agent(f"Verify adversarially: {json.dumps(r)}", {"label": f"verify:{i}", "phase": "Verify"}),
)
return await agent("Synthesize the verified findings:\n" + json.dumps(findings))
```

- `agent(prompt, opts)` spawns a child agent; `opts` may include `schema` (enforce
  structured output), `model`, `agentType`, and `isolation="worktree"`.
- `pipeline` (default, no barrier) / `parallel` (with barrier) handle concurrency;
  `phase`/`log` report progress; `workflow()` runs a named workflow inline; `args` /
  `budget` access the input arguments and the token budget.

### Agent Type

Specify a child agent's type via `agentType` in the script; if omitted, it defaults to
`general-purpose` (full toolset):

| Type | Toolset | Description |
|------|---------|-------------|
| `general-purpose` | `*` (all safe tools) | Default; good for searching code, researching complex problems, and multi-step tasks |
| `explore` | Read-only (read_file, search_files, terminal) | Fast codebase exploration; good for locating files and searching keywords |
| `plan` | Read-only (read_file, search_files, terminal) | Software architecture design; outputs a step-by-step implementation plan |
| `verification` | web + file + terminal + browser | Verifies implementation correctness; runs build/test/lint to emit PASS/FAIL |

Agent types are resolved from three locations in priority order (on a name collision,
earlier locations override later ones):

1. `<project>/.hermes/dynamic-workflows/agents/*.md`  — project level, applies only to the current project
2. `~/.hermes/dynamic-workflows/agents/*.md`          — user level, applies globally
3. `<plugin>/hermes_dynamic_workflows/agents/*.md`    — built-in defaults (general-purpose/explore/plan/verification)

To add a custom type, create a new `.md` file under directory 1 or 2 in the following format:

```markdown
---
name: my-agent
description: "A short description of what this agent is for; the model uses it to automatically pick the right agent."
model: inherit
toolsets: [web, file, terminal]
---

Write the agent's system prompt here to guide its behavior, style, and constraints.
```

`name` and `description` are required; `model` defaults to `inherit` (inherits the
current session's model); `toolsets` defaults to the global `default_child_toolsets`;
optional fields also include `allowed_tools`, `disallowed_tools`, and `isolation`.

At runtime the plugin persists the script and the full execution trace (transcript) of
every child agent, and injects a `<task-notification>` into the conversation on
completion — no polling required. Use `/workflows` to view history and details.

## Deep Dive

For implementation details (core execution path, tools and full call results, prompt
cache, concurrency and limits, permission governance, rebuilding transcripts from
`state.db`, sandboxing, resume…), see [TECHNICAL.md](./TECHNICAL.md).

## License

[MIT](./LICENSE)
