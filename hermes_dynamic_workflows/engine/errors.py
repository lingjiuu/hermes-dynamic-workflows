"""Workflow-specific exceptions."""

from __future__ import annotations


class DynamicWorkflowError(Exception):
    """Base class for plugin errors."""


class WorkflowParseError(DynamicWorkflowError):
    """Raised when a workflow script cannot be parsed."""


class SandboxViolation(DynamicWorkflowError):
    """Raised when a workflow script uses forbidden Python syntax."""


class WorkflowRuntimeError(DynamicWorkflowError):
    """Raised when workflow execution fails."""


class WorkflowTimeout(WorkflowRuntimeError):
    """Raised when a workflow or child agent exceeds its timeout."""


class WorkflowStopped(WorkflowRuntimeError):
    """Raised when a workflow run is stopped by the user."""


class ChildAgentError(WorkflowRuntimeError):
    """Raised when a child agent fails."""
