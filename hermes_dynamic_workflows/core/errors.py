"""Workflow-specific exceptions.

Two families, split on purpose:

* ``DynamicWorkflowError`` (→ ``Exception``): recoverable. A workflow script may
  ``try/except Exception`` around these (e.g. a child agent that failed, a
  nested workflow error, bad indexing of results) and handle them gracefully.
* ``WorkflowHalt`` (→ ``BaseException``): run-level halts — user stop, the
  workflow deadline, and hard limits (token budget / agent cap / loop cap).
  These derive from ``BaseException`` so a script's ``except Exception`` cannot
  swallow them; the run stays cancellable and bounded no matter what the script
  catches. The sandbox additionally forbids bare ``except:`` / ``except
  BaseException`` so they cannot be caught by a wildcard either.
"""

from __future__ import annotations


class DynamicWorkflowError(Exception):
    """Base class for recoverable plugin errors (catchable by scripts)."""


class WorkflowParseError(DynamicWorkflowError):
    """Raised when a workflow script cannot be parsed."""


class SandboxViolation(DynamicWorkflowError):
    """Raised when a workflow script uses forbidden Python syntax."""


class WorkflowRuntimeError(DynamicWorkflowError):
    """Raised when workflow execution fails (API misuse, bad return, etc.)."""


class ChildAgentError(WorkflowRuntimeError):
    """Raised when a child agent fails."""


class ChildAgentSkipped(DynamicWorkflowError):
    """Raised internally when one child agent is intentionally skipped."""


class WorkflowLaunchDenied(DynamicWorkflowError):
    """Raised when a top-level launch is not approved by the user (or no
    approval channel is available). The caller should tell the user, not retry."""


class WorkflowToolUseError(DynamicWorkflowError):
    """Raised when the workflow tool should return a tool_use_error result."""


class WorkflowTimeout(ChildAgentError):
    """A single child agent exceeded its own timeout.

    Recoverable and per-agent: ``agent()`` records it and raises it to the
    workflow script. This is NOT the whole-run deadline — that is
    ``WorkflowDeadlineExceeded`` below.
    """


class WorkflowHalt(BaseException):
    """Base for run-level halts that a script must never be able to catch.

    Derives from ``BaseException`` (not ``Exception``) so ``except Exception``
    in a workflow script cannot swallow it.
    """


class WorkflowStopped(WorkflowHalt):
    """Raised when a workflow run is stopped by the user."""


class WorkflowDeadlineExceeded(WorkflowHalt):
    """Raised when the whole workflow exceeds its wall-clock deadline."""


class WorkflowLimitExceeded(WorkflowHalt):
    """Raised when a hard run limit is hit: token budget, agent cap, loop cap."""
