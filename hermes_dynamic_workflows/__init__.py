"""Dynamic workflow runtime for Hermes plugins."""

from __future__ import annotations

__all__ = [
    "WorkflowOptions",
    "WorkflowResult",
    "run_workflow",
    "workflow",
]


def __getattr__(name: str):
    if name in {"WorkflowOptions", "WorkflowResult", "run_workflow"}:
        from .engine import runtime

        return getattr(runtime, name)
    if name == "workflow":
        from .adapters.workflow import workflow

        return workflow
    raise AttributeError(name)
