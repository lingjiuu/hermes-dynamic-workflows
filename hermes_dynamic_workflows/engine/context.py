"""Shared execution graph for a workflow run."""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Iterator

from .cache import ResumeCache
from ..core.config import PluginConfig
from ..core.errors import (
    WorkflowDeadlineExceeded,
    WorkflowLimitExceeded,
    WorkflowStopped,
)
from ..core.types import ChildAgentRunner, WorkflowFrame, WorkflowState, normalize_phase_specs


class PauseGate:
    """Cooperative run gate that excludes paused time from the run deadline."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._paused_at: float | None = None
        self._paused_seconds = 0.0

    @property
    def is_paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def paused_seconds(self) -> float:
        with self._condition:
            current = monotonic() - self._paused_at if self._paused and self._paused_at is not None else 0.0
            return self._paused_seconds + current

    def pause(self) -> bool:
        with self._condition:
            if self._paused:
                return False
            self._paused = True
            self._paused_at = monotonic()
            self._condition.notify_all()
            return True

    def resume(self) -> bool:
        with self._condition:
            if not self._paused:
                return False
            if self._paused_at is not None:
                self._paused_seconds += max(0.0, monotonic() - self._paused_at)
            self._paused = False
            self._paused_at = None
            self._condition.notify_all()
            return True

    def wait(self, stop_event: threading.Event) -> None:
        with self._condition:
            while self._paused and not stop_event.is_set():
                self._condition.wait(timeout=0.1)


@dataclass
class WorkflowExecutionContext:
    config: PluginConfig
    runner: ChildAgentRunner
    stop_event: threading.Event
    pause_gate: PauseGate
    resume_cache: ResumeCache
    deadline: float
    root: WorkflowFrame
    on_update: Callable[[WorkflowState], None] | None = None
    on_journal: Callable[[dict[str, Any]], None] | None = None
    plugin_context: Any = None
    token_budget_total: int | None = None
    store: Any = None
    state: WorkflowState = field(init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _agent_slots: threading.BoundedSemaphore = field(init=False)
    _agent_counter: int = 0
    _frame_counter: int = 0
    _agent_count: int = 0
    _spent_tokens: int = 0
    _loop_ticks: int = 0

    def __post_init__(self) -> None:
        self.state = WorkflowState(self.root)
        concurrency = min(self.config.concurrency, self.config.max_concurrency)
        self._agent_slots = threading.BoundedSemaphore(max(1, concurrency))

    @property
    def spent_tokens(self) -> int:
        with self._lock:
            return self._spent_tokens

    @property
    def agent_count(self) -> int:
        with self._lock:
            return self._agent_count

    @property
    def remaining_tokens(self) -> float:
        total = self.token_budget_total
        if total is None:
            return math.inf
        return max(0, total - self.spent_tokens)

    def reserve_agent(self) -> int:
        self.check_runtime()
        with self._lock:
            if self._agent_count >= self.config.max_agents:
                raise WorkflowLimitExceeded(
                    f"workflow agent count exceeded ({self.config.max_agents})"
                )
            if self.token_budget_total is not None and self._spent_tokens >= self.token_budget_total:
                raise WorkflowLimitExceeded(
                    f"workflow token budget exceeded ({self.token_budget_total})"
                )
            self._agent_counter += 1
            self._agent_count += 1
            return self._agent_counter

    def record_tokens(self, tokens: int) -> None:
        if tokens <= 0:
            return
        with self._lock:
            self._spent_tokens += int(tokens)

    @contextmanager
    def agent_slot(self) -> Iterator[None]:
        self.check_runtime()
        self._agent_slots.acquire()
        try:
            self.check_runtime()
            yield
        finally:
            self._agent_slots.release()

    def create_child_frame(
        self,
        *,
        parent: WorkflowFrame,
        meta: dict[str, Any],
        args: Any,
        cwd: str,
        source_ref: str | None = None,
    ) -> WorkflowFrame:
        with self._lock:
            self._frame_counter += 1
            frame = WorkflowFrame(
                id=f"frame-{self._frame_counter}",
                meta=meta,
                args=args,
                cwd=cwd,
                phases=normalize_phase_specs(meta.get("phases")),
                parent_id=parent.id,
                source_ref=source_ref,
            )
            parent.children.append(frame)
            self.notify()
            return frame

    def check_runtime(self) -> None:
        if self.stop_event.is_set():
            raise WorkflowStopped("workflow was stopped")
        self.pause_gate.wait(self.stop_event)
        if self.stop_event.is_set():
            raise WorkflowStopped("workflow was stopped")
        if monotonic() - self.pause_gate.paused_seconds > self.deadline:
            raise WorkflowDeadlineExceeded(
                f"workflow timed out after {self.config.workflow_timeout_seconds:.0f}s"
            )

    def tick_loop(self) -> bool:
        """Cooperative guard injected into every ``while`` loop test by the
        sandbox. Makes the wall-clock deadline and user-stop actually fire
        inside a pure-compute loop (one that never calls agent()), and caps
        total loop iterations as a runaway backstop. Returns True so the
        original loop test still controls the loop.

        Lock-free on purpose: the deadline/stop reads are already thread-safe,
        and the iteration counter is only an approximate backstop, so a racy
        increment under concurrent loops is acceptable and keeps tight loops
        cheap.
        """
        self.check_runtime()
        self._loop_ticks += 1
        if self._loop_ticks > self.config.max_loop_iterations:
            raise WorkflowLimitExceeded(
                f"workflow loop iteration cap exceeded ({self.config.max_loop_iterations})"
            )
        return True

    def notify(self) -> None:
        if self.on_update is None:
            return
        try:
            self.on_update(self.state)
        except Exception:
            pass

    def journal(self, event: dict[str, Any]) -> None:
        if self.on_journal is None:
            return
        try:
            self.on_journal(event)
        except Exception:
            pass
