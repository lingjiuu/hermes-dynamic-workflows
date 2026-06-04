"""Hermes plugin entrypoint for dynamic workflows."""

from __future__ import annotations

from .hermes_dynamic_workflows.plugin.schema import DYNAMIC_WORKFLOW_SCHEMA
from .hermes_dynamic_workflows.plugin.tool import workflow
from .hermes_dynamic_workflows.ui.commands import workflow_stop_command, workflows_command


def register(ctx) -> None:
    """Register the workflow tool with Hermes."""
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
