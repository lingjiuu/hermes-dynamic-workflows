"""One-command full-screen workflow monitor."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from typing import Any

from .model import (
    SessionGroup,
    WorkflowRepository,
    WorkflowView,
    group_sessions,
    list_items,
)
from .render import (
    Line,
    RenderState,
    agent_detail_scroll_max,
    display_width,
    render_styled,
    text_of,
)


# Active runs poll fast (live timers/activity); when everything is terminal the
# loop just stat()s the runs dir and only reloads when it actually changed.
REFRESH_ACTIVE = 0.5
REFRESH_IDLE = 1.0


class TuiController:
    def __init__(self, repository: WorkflowRepository | None = None):
        self.repository = repository or WorkflowRepository()
        self.workflows: list[WorkflowView] = []
        self.groups: list[SessionGroup] = []
        self.has_active = False
        self.state = RenderState()
        self.should_exit = False
        self._expanded_initialized = False

    def refresh(self) -> None:
        selected_run_id = self.current_run.run_id if self.current_run else ""
        self.workflows = self.repository.load()
        self.groups = group_sessions(self.workflows)
        self.has_active = any(group.running for group in self.groups)
        if not self._expanded_initialized:
            current = next((group.key for group in self.groups if group.is_current), "")
            self.state = replace(self.state, expanded=frozenset({current}) if current else frozenset())
            self._expanded_initialized = True
        run_index = self.state.run_index
        if selected_run_id:
            for index, workflow in enumerate(self.workflows):
                if workflow.run_id == selected_run_id:
                    run_index = index
                    break
        self.state = replace(
            self.state,
            run_index=_clamp(run_index, len(self.workflows)),
            list_cursor=_clamp(self.state.list_cursor, len(self._items())),
        )
        self._normalize_nested_selection()

    @property
    def current_run(self) -> WorkflowView | None:
        if not self.workflows:
            return None
        return self.workflows[_clamp(self.state.run_index, len(self.workflows))]

    def _items(self) -> list[tuple[str, int, int]]:
        return list_items(self.groups, self.state.expanded)

    def _cursor_item(self) -> tuple[str, int, int] | None:
        items = self._items()
        if not items:
            return None
        return items[_clamp(self.state.list_cursor, len(items))]

    def _active_workflow(self) -> WorkflowView | None:
        """The workflow the current action targets (cursor's run in list view)."""
        if self.state.view != "list":
            return self.current_run
        item = self._cursor_item()
        if item and item[0] == "run":
            return self.groups[item[1]].workflows[item[2]]
        return None

    def handle_key(self, key: str) -> None:
        if key in {"q", "Q"}:
            self.should_exit = True
            return
        if key == "up":
            self._move(-1)
        elif key == "down":
            self._move(1)
        elif key == "right":
            self._expand() if self.state.view == "list" else self._enter()
        elif key == "left":
            self._collapse() if self.state.view == "list" else self._back()
        elif key == "enter":
            self._enter()
        elif key in {"esc", "backspace"}:
            self._back()
        elif key in {"s", "S"}:
            self._save()
        elif key in {"x", "X"}:
            self._control("stop")
        elif key in {"p", "P"}:
            target = self._active_workflow()
            self._control("resume" if target and target.status == "paused" else "pause")
        elif key in {"r", "R"}:
            self._control("restart")

    def _detail_view(self) -> WorkflowView | None:
        """Full (agents/phases) view for the open run; built on demand, cached."""
        run = self.current_run
        if run is None:
            return None
        return self.repository.detail(run.run_id) or run

    def frame(self, width: int, height: int) -> list[Line]:
        if self.state.view in ("workflow", "agent") and self.current_run:
            full = self._detail_view()
            if self.state.view == "agent":
                full = self.repository.hydrate_agent_activity(
                    full,
                    phase_index=self.state.phase_index,
                    agent_index=self.state.agent_index,
                )
                max_scroll = agent_detail_scroll_max(full, self.state, width, height)
                if self.state.detail_scroll > max_scroll:
                    self.state = replace(self.state, detail_scroll=max_scroll)
            workflows = list(self.workflows)
            workflows[_clamp(self.state.run_index, len(workflows))] = full
            return render_styled(workflows, self.state, width=width, height=height, groups=self.groups)
        return render_styled(self.workflows, self.state, width=width, height=height, groups=self.groups)

    def _scroll_detail(self, delta: int) -> None:
        self.state = replace(
            self.state,
            detail_scroll=max(0, self.state.detail_scroll + delta),
            message="",
        )

    def _move(self, delta: int) -> None:
        if self.state.view == "list":
            self.state = replace(
                self.state,
                list_cursor=_clamp(self.state.list_cursor + delta, len(self._items())),
                message="",
            )
        elif self.state.view == "workflow":
            if self.state.focus == "agents":
                self.state = replace(
                    self.state,
                    agent_index=_clamp(self.state.agent_index + delta, len(self._current_phase_agents())),
                    message="",
                )
            else:
                detail = self._detail_view()
                count = len(detail.phases) if detail else 0
                self.state = replace(
                    self.state,
                    phase_index=_clamp(self.state.phase_index + delta, count),
                    agent_index=0,
                    message="",
                )
        elif self.state.view == "agent":
            if self.state.focus == "detail":
                self._scroll_detail(delta)
            else:
                self.state = replace(
                    self.state,
                    agent_index=_clamp(self.state.agent_index + delta, len(self._current_phase_agents())),
                    detail_scroll=0,
                    prompt_expanded=False,
                    message="",
                )

    def _expand(self) -> None:
        item = self._cursor_item()
        if item is None:
            return
        kind, group_index, run_index = item
        if kind == "run":
            self._open_run(group_index, run_index)  # → on a run drills into it
            return
        group = self.groups[group_index]
        if group.key not in self.state.expanded:
            self.state = replace(self.state, expanded=self.state.expanded | {group.key}, message="")
        elif group.workflows:
            # already open — step into its first run
            self.state = replace(self.state, list_cursor=self.state.list_cursor + 1, message="")

    def _collapse(self) -> None:
        item = self._cursor_item()
        if item is None:
            return
        kind, group_index, _ = item
        key = self.groups[group_index].key
        if kind == "run":
            header = next(i for i, (k, g, _) in enumerate(self._items()) if k == "group" and g == group_index)
            self.state = replace(self.state, list_cursor=header, expanded=self.state.expanded - {key}, message="")
        elif key in self.state.expanded:
            self.state = replace(self.state, expanded=self.state.expanded - {key}, message="")

    def _enter(self) -> None:
        if self.state.view == "list":
            item = self._cursor_item()
            if item is None:
                return
            kind, group_index, run_index = item
            if kind == "group":
                key = self.groups[group_index].key
                expanded = self.state.expanded ^ {key}
                self.state = replace(self.state, expanded=expanded, message="")
                return
            self._open_run(group_index, run_index)
        elif self.state.view == "workflow":
            if self.state.focus == "agents":
                # drill into the agent view
                self.state = replace(self.state, view="agent", focus="agents", detail_scroll=0, message="")
                return
            # left pane: step into the right (agents) pane
            agents = self._current_phase_agents()
            if not agents:
                self.state = replace(self.state, message="This phase has no agents yet.")
                return
            self.state = replace(self.state, focus="agents", agent_index=0, message="")
        elif self.state.view == "agent":
            if self.state.focus == "agents":
                self.state = replace(self.state, focus="detail", prompt_expanded=False, message="")
            else:
                # focus on detail: Enter toggles prompt expand/collapse
                self.state = replace(self.state, prompt_expanded=not self.state.prompt_expanded, message="")

    def _back(self) -> None:
        if self.state.view == "agent":
            if self.state.focus == "detail":
                # back to the left agent list
                self.state = replace(self.state, focus="agents", prompt_expanded=False, message="")
            else:
                self.state = replace(self.state, view="workflow", focus="agents", detail_scroll=0, prompt_expanded=False, message="")
        elif self.state.view == "workflow":
            if self.state.focus == "agents":
                self.state = replace(self.state, focus="phases", message="")
            else:
                self.state = replace(self.state, view="list", message="")
        else:
            self.should_exit = True

    def _open_run(self, group_index: int, run_index: int) -> None:
        workflow = self.groups[group_index].workflows[run_index]
        flat = self._flat_index(workflow.run_id)
        if flat is None:
            return
        full = self.repository.detail(workflow.run_id) or workflow
        self.state = replace(
            self.state,
            view="workflow",
            run_index=flat,
            phase_index=_active_phase_index(full),
            agent_index=0,
            focus="phases",
            detail_scroll=0,
            message="",
        )

    def _flat_index(self, run_id: str) -> int | None:
        for index, workflow in enumerate(self.workflows):
            if workflow.run_id == run_id:
                return index
        return None

    def _save(self) -> None:
        workflow = self._active_workflow()
        if not workflow:
            self.state = replace(self.state, message="Select a workflow first.")
            return
        try:
            path = self.repository.save_markdown(workflow)
            self.state = replace(self.state, message=f"Saved to {path}")
        except OSError as exc:
            self.state = replace(self.state, message=f"Save failed: {exc}")

    def _control(self, action: str) -> None:
        workflow = self._active_workflow()
        if not workflow:
            self.state = replace(self.state, message="Select a workflow first.")
            return
        response = self.repository.request_control(workflow, action)
        self.state = replace(self.state, message=str(response.get("message") or "Control request sent."))
        self.refresh()
        new_run_id = str(response.get("newRunId") or "")
        if new_run_id:
            flat = self._flat_index(new_run_id)
            if flat is not None:
                self.state = replace(self.state, run_index=flat, phase_index=0, agent_index=0)

    def _current_phase_agents(self):
        workflow = self._detail_view()
        if not workflow:
            return ()
        if not workflow.phases:
            return workflow.agents
        phase = workflow.phases[_clamp(self.state.phase_index, len(workflow.phases))]
        return phase.agents

    def _normalize_nested_selection(self) -> None:
        workflow = self._detail_view() if self.state.view in ("workflow", "agent") else None
        phase_count = len(workflow.phases) if workflow else 0
        phase_index = _clamp(self.state.phase_index, phase_count)
        agent_count = len(workflow.phases[phase_index].agents) if workflow and workflow.phases else 0
        self.state = replace(
            self.state,
            phase_index=phase_index,
            agent_index=_clamp(self.state.agent_index, agent_count),
        )


def main() -> int:
    controller = TuiController()
    controller.refresh()
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in controller.frame(width=120, height=max(12, len(controller.workflows) * 2 + 8)):
            print(text_of(line).rstrip())
        return 0
    try:
        import curses
    except ImportError:
        print("hermes-workflows needs terminal curses support.", file=sys.stderr)
        return 1
    try:
        curses.wrapper(lambda screen: _run_curses(screen, controller, curses))
    except KeyboardInterrupt:
        return 0
    return 0


def _run_curses(screen: Any, controller: TuiController, curses: Any) -> None:
    _configure_curses(screen, curses)
    last_poll = 0.0
    last_version = controller.repository.world_version()
    last_frame: list[Line] | None = None
    last_size: tuple[int, int] | None = None
    dirty = True
    while not controller.should_exit:
        now = time.monotonic()
        interval = REFRESH_ACTIVE if controller.has_active else REFRESH_IDLE
        if now - last_poll >= interval:
            last_poll = now
            if controller.has_active:
                # live runs: reload every tick so timers/activity advance
                controller.refresh()
                last_version = controller.repository.world_version()
                dirty = True
            else:
                # idle: O(1) dir-mtime check; only reload when something changed
                version = controller.repository.world_version()
                if version != last_version:
                    last_version = version
                    controller.refresh()
                    dirty = True
        height, width = screen.getmaxyx()
        size = (height, width)
        if size != last_size:
            last_size = size
            dirty = True
        if dirty:
            frame = controller.frame(width, height)
            if frame != last_frame:
                _draw(screen, frame, curses)
                last_frame = frame
            dirty = False
        key = screen.getch()
        if key != -1:
            controller.handle_key(_key_name(key, curses))
            dirty = True


def _configure_curses(screen: Any, curses: Any) -> None:
    screen.keypad(True)
    screen.timeout(100)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    _set_alternate_scroll(False)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)


def _set_alternate_scroll(enabled: bool) -> None:
    """Toggle the terminal's alternate-scroll mode (DECSET 1007).

    On the alternate screen many terminals translate the mouse wheel into ↑/↓
    key presses; disabling it makes the wheel inert so the list only moves with
    the arrow keys.
    """
    try:
        sys.stdout.write("\033[?1007h" if enabled else "\033[?1007l")
        sys.stdout.flush()
    except Exception:
        pass


def _style_attrs(curses: Any) -> dict[str, int]:
    italic = getattr(curses, "A_ITALIC", 0)
    cyan, green, yellow = curses.color_pair(1), curses.color_pair(2), curses.color_pair(3)
    return {
        "": 0,
        "title": cyan | curses.A_BOLD,
        "rule": cyan,
        "divider": 0,  # normal weight — brighter than the dim body text
        "dim": curses.A_DIM,
        "footer": curses.A_DIM | italic,
        "ok": green,
        "warn": yellow,
        "error": yellow,
        "running": cyan,
        "stopped": curses.A_DIM,
        "sel": curses.A_BOLD,
    }


def _draw(screen: Any, lines: list[Line], curses: Any) -> None:
    # erase() (not clear()) lets curses compute a minimal screen diff on refresh,
    # which avoids the full-repaint flicker clear() forces every frame.
    screen.erase()
    height, width = screen.getmaxyx()
    attrs = _style_attrs(curses)
    for row, line in enumerate(lines[:height]):
        # Only the very bottom row must avoid the last cell (writing it scrolls
        # curses and raises); every other row may use the full width.
        limit = width - 1 if row == height - 1 else width
        col = 0
        for span in line:
            if col >= limit:
                break
            try:
                screen.addnstr(row, col, span.text, max(0, limit - col), attrs.get(span.style, 0))
            except curses.error:
                pass
            col += display_width(span.text)
    screen.refresh()


def _key_name(key: int, curses: Any) -> str:
    mapping = {
        curses.KEY_UP: "up",
        curses.KEY_DOWN: "down",
        curses.KEY_LEFT: "left",
        curses.KEY_RIGHT: "right",
        curses.KEY_ENTER: "enter",
        curses.KEY_BACKSPACE: "backspace",
        10: "enter",
        13: "enter",
        27: "esc",
        127: "backspace",
    }
    if key in mapping:
        return mapping[key]
    try:
        return chr(key)
    except (ValueError, OverflowError):
        return ""


def _clamp(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))


def _active_phase_index(workflow: WorkflowView | None) -> int:
    if not workflow or not workflow.current_phase:
        return 0
    for index, phase in enumerate(workflow.phases):
        if phase.title == workflow.current_phase:
            return index
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
