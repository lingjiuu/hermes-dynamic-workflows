"""Hermes plugin entrypoint for dynamic workflows."""

from __future__ import annotations

import os

from hermes_dynamic_workflows.engine.approval_hook import pre_tool_call_handler
from hermes_dynamic_workflows.plugin import registrar
from hermes_dynamic_workflows.plugin.task_stop import TASK_STOP_SCHEMA, task_stop
from hermes_dynamic_workflows.plugin.workflow import DYNAMIC_WORKFLOW_SCHEMA, workflow
from hermes_dynamic_workflows.ui.commands import (
    discover_named_workflows,
    make_named_workflow_handler,
    workflow_stop_command,
    workflows_command,
)


def register(ctx) -> None:
    """Register the workflow tool and commands with Hermes."""
    registrar.set_plugin_context(ctx)

    def _workflow_handler(params, **kwargs):
        return workflow(params, plugin_context=ctx, **kwargs)

    ctx.register_tool(
        name="workflow",
        toolset="workflow",
        schema=DYNAMIC_WORKFLOW_SCHEMA,
        handler=_workflow_handler,
        description=(
            "Run deterministic Python workflow scripts that orchestrate "
            "multiple Hermes child agents with agent(), parallel(), and pipeline()."
        ),
    )

    def _task_stop_handler(params, **kwargs):
        return task_stop(params, **kwargs)

    ctx.register_tool(
        name="task_stop",
        toolset="workflow",
        schema=TASK_STOP_SCHEMA,
        handler=_task_stop_handler,
        description="Stop a running background task by ID.",
    )
    # Make child_approval_policy authoritative for workflow-child terminal
    # commands even in non-CLI contexts (where Hermes would otherwise
    # auto-approve/orphan). In CLI this defers to the per-thread callback.
    ctx.register_hook("pre_tool_call", pre_tool_call_handler)
    ctx.register_command(
        name="workflows",
        handler=workflows_command,
        description="List dynamic workflow runs or show one run by ID.",
        args_hint="[runId]",
    )
    ctx.register_command(
        name="workflow-stop",
        handler=workflow_stop_command,
        description="Stop a running dynamic workflow.",
        args_hint="<runId>",
    )
    _register_saved_workflow_commands(ctx)


def _register_saved_workflow_commands(ctx) -> None:
    """Expose saved workflows as slash commands."""
    cwd = os.environ.get("TERMINAL_CWD") or os.getcwd()
    try:
        names = discover_named_workflows(cwd)
    except Exception:
        names = []
    for name in names:
        try:
            ctx.register_command(
                name=name,
                handler=make_named_workflow_handler(name),
                description=f"Run saved dynamic workflow '{name}'.",
                args_hint="[args]",
            )
        except Exception:
            continue
