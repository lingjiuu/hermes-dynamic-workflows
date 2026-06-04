"""Hermes plugin entrypoint for dynamic workflows."""

from __future__ import annotations

import os

from hermes_dynamic_workflows.engine.approval_hook import pre_tool_call_handler
from hermes_dynamic_workflows.engine.structured_tool import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    STRUCTURED_OUTPUT_TOOL_SCHEMA,
    STRUCTURED_OUTPUT_TOOLSET,
    submit_structured_output_handler,
)
from hermes_dynamic_workflows.plugin import registrar
from hermes_dynamic_workflows.plugin.schema import DYNAMIC_WORKFLOW_SCHEMA
from hermes_dynamic_workflows.plugin.tool import workflow
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
    # Child-agent-only tool: a schema'd agent() call submits its final answer
    # through this tool, validated at the tool layer with model retry. Lives in
    # its own toolset so normal sessions never see it; workflow children opt in.
    ctx.register_tool(
        name=STRUCTURED_OUTPUT_TOOL_NAME,
        toolset=STRUCTURED_OUTPUT_TOOLSET,
        schema=STRUCTURED_OUTPUT_TOOL_SCHEMA,
        handler=submit_structured_output_handler,
        description="Submit a dynamic-workflow child agent's final schema-validated answer.",
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
    """Expose each saved workflow as a ``/<name>`` slash command, like Claude
    Code surfaces saved workflows in ``/`` autocomplete."""
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
