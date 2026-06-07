"""Styled rendering for the full-screen workflow TUI.

Each rendered line is a list of ``Span`` (text + style key). The curses layer
maps style keys to attributes/colors, so a single row can mix colors — e.g. a
green status icon, a neutral name, and dim metrics — matching the Claude Code
``/workflows`` panel. ``render_screen`` flattens spans to plain text for the
non-interactive snapshot and the tests.
"""

from __future__ import annotations

import textwrap
import unicodedata
from dataclasses import dataclass

from .model import AgentView, PhaseView, SessionGroup, WorkflowView, group_sessions, list_items


_DETAIL_HEADER_LINES = 4
_LIST_FOOTER = "↑/↓ to select · → in · ← out · Enter to open · x to stop · s to save · Esc to close"
_WORKFLOW_FOOTER = "↑↓ select · → in · ← back · x stop · p pause · r restart · s save"
_AGENT_FOOTER = "↑↓ agent · j/k scroll · x stop · p pause · r restart · s save · esc back"


@dataclass(frozen=True)
class Span:
    text: str
    style: str = ""


Line = list[Span]


@dataclass(frozen=True)
class RenderState:
    view: str = "list"
    run_index: int = 0
    phase_index: int = 0
    agent_index: int = 0
    detail_scroll: int = 0
    list_cursor: int = 0
    expanded: frozenset[str] = frozenset()
    focus: str = "phases"  # workflow view: "phases" (left) or "agents" (right)
    message: str = ""


def render_styled(
    workflows: list[WorkflowView],
    state: RenderState,
    *,
    width: int,
    height: int,
    groups: "list[SessionGroup] | None" = None,
) -> list[Line]:
    width = max(40, width)
    height = max(12, height)
    if state.view == "workflow" and workflows:
        lines = _render_workflow(workflows[_clamp(state.run_index, len(workflows))], state, width, height)
    elif state.view == "agent" and workflows:
        lines = _render_agent(workflows[_clamp(state.run_index, len(workflows))], state, width, height)
    else:
        lines = _render_list(workflows, state, width, height, groups)
    return _fit(lines, width, height)


def render_screen(
    workflows: list[WorkflowView],
    state: RenderState,
    *,
    width: int,
    height: int,
    groups: "list[SessionGroup] | None" = None,
) -> list[str]:
    return [text_of(line) for line in render_styled(workflows, state, width=width, height=height, groups=groups)]


def text_of(line: Line) -> str:
    return "".join(span.text for span in line)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def _render_list(
    workflows: list[WorkflowView],
    state: RenderState,
    width: int,
    height: int,
    groups: "list[SessionGroup] | None" = None,
) -> list[Line]:
    if groups is None:
        groups = group_sessions(workflows)
    running = sum(group.running for group in groups)
    lines: list[Line] = [
        _rule(width),
        [_sp("  "), _sp("Dynamic workflows", "title")],
        [_sp(f"  {len(groups)} sessions · {running} running", "dim")],
        [],
    ]
    if not groups:
        lines.append([_sp("  No workflow runs found.", "dim")])
        lines.append([])
        lines.append([_sp("  Start a workflow in Hermes, then this panel refreshes automatically.", "dim")])
        return _pad_to_footer(lines, _LIST_FOOTER, state.message, width, height)

    items = list_items(groups, state.expanded)
    cursor = _clamp(state.list_cursor, len(items))
    rendered: list[Line] = []
    for index, (kind, group_index, run_index) in enumerate(items):
        selected = index == cursor
        group = groups[group_index]
        if kind == "group":
            rendered.append(_group_header(group, group.key in state.expanded, selected, width))
        else:
            rendered.append(_session_run_row(group.workflows[run_index], selected, width))
    body_height = max(1, height - len(lines) - 2)
    lines.extend(_window_lines(rendered, cursor, body_height))
    return _pad_to_footer(lines, _LIST_FOOTER, state.message, width, height)


def _group_header(group: SessionGroup, expanded: bool, selected: bool, width: int) -> Line:
    caret = "▾ " if expanded else "▸ "
    line: Line = [
        _sp("  "),
        _sp(caret, "sel" if selected else "dim"),
        _sp(group.project, "sel" if selected else ""),
        _sp(f" · {group.ago}", "dim"),
        _sp(f"  {_session_counts(group)}", "dim"),
    ]
    if group.is_current:
        line.append(_sp("  ● current", "running"))
    return _crop_line(line, width)


def _session_counts(group: SessionGroup) -> str:
    parts = []
    if group.running:
        parts.append(f"{group.running} running")
    if group.done:
        parts.append(f"{group.done} done")
    if group.failed:
        parts.append(f"{group.failed} failed")
    return " · ".join(parts) or "no runs"


def _session_run_row(workflow: WorkflowView, selected: bool, width: int) -> Line:
    metrics = (
        f"  {workflow.agent_count} agents · {_tokens(workflow.tokens)} tok · "
        f"{_duration(workflow.duration_seconds)}"
    )
    return _crop_line(
        [
            _sp("      "),
            _sp("❯ ", "sel") if selected else _sp("  "),
            _sp(_status_icon(workflow.status), _status_style(workflow.status)),
            _sp(" "),
            _sp(workflow.name, "sel" if selected else ""),
            _sp(metrics, "dim"),
        ],
        width,
    )


def _render_workflow(workflow: WorkflowView, state: RenderState, width: int, height: int) -> list[Line]:
    phase_index = _clamp(state.phase_index, len(workflow.phases))
    phase = workflow.phases[phase_index] if workflow.phases else PhaseView("Agents", ())
    progress = f"{workflow.done}/{workflow.agent_count} agents · {_duration(workflow.duration_seconds)}"
    header = _detail_header(workflow.name, workflow.description, progress, width)
    left_width = max(18, min(24, width // 5))
    right_width = max(20, width - left_width - 5)
    body_height = max(5, height - _DETAIL_HEADER_LINES - 2)

    phases_focused = state.focus != "agents"
    left: list[Line] = []
    for index, item in enumerate(workflow.phases):
        selected = index == phase_index
        active = selected and phases_focused
        done = bool(item.agents) and item.done == len(item.agents)
        suffix = "not started" if not item.agents else f"{item.done}/{len(item.agents)}"
        left.append(
            [
                _sp("❯ ", "sel") if selected else _sp("  "),
                _sp("✓" if done else str(index + 1), "ok" if done else "dim"),
                _sp(" " + item.title, "sel" if active else ""),
                _sp(" " + suffix, "dim"),
            ]
        )
    left = _window_lines(left, phase_index, body_height)

    agent_index = _clamp(state.agent_index, len(phase.agents))
    right: list[Line] = []
    if not phase.agents:
        right.append([_sp("No agents in this phase yet.", "dim")])
    right.extend(
        _agent_row(agent, right_width, selected=(not phases_focused and index == agent_index))
        for index, agent in enumerate(phase.agents)
    )

    panel = _two_columns(
        "Phases", left, f"{phase.title} · {len(phase.agents)} agents", right,
        left_width=left_width, right_width=right_width, height=body_height,
    )
    return header + panel + [_footer(_WORKFLOW_FOOTER, state.message, width)]


def _render_agent(workflow: WorkflowView, state: RenderState, width: int, height: int) -> list[Line]:
    phase, agents, agent = _selected_agent(workflow, state)
    agent_index = _clamp(state.agent_index, len(agents))
    progress = f"{workflow.done}/{workflow.agent_count} agents · {_duration(workflow.duration_seconds)}"
    header = _detail_header(workflow.name, workflow.description, progress, width)
    left_width, right_width, body_height = _agent_geometry(width, height)

    left: list[Line] = [
        [
            _sp("❯ ", "sel") if index == agent_index else _sp("  "),
            _sp(_status_icon(item.status), _status_style(item.status)),
            _sp(" " + item.label, "sel" if index == agent_index else ""),
        ]
        for index, item in enumerate(agents)
    ]
    left = _window_lines(left, agent_index, body_height)
    right = _scroll_view(_agent_detail(agent, right_width), state.detail_scroll, body_height, right_width)

    panel = _two_columns(
        f"{phase.title} · {len(agents)} agents", left,
        agent.label if agent else "Agent", right,
        left_width=left_width, right_width=right_width, height=body_height,
    )
    return header + panel + [_footer(_AGENT_FOOTER, state.message, width)]


def _detail_header(name: str, description: str, progress: str, width: int) -> list[Line]:
    return [
        [],
        [_sp("  "), _sp(name, "title")],
        _left_right([_sp("  "), _sp(description, "dim")], progress + "  ", "dim", width),
        _rule(width),
    ]


# --------------------------------------------------------------------------- #
# Agent detail (right column of the agent view) + scrolling
# --------------------------------------------------------------------------- #
def _selected_agent(workflow: WorkflowView, state: RenderState):
    phase_index = _clamp(state.phase_index, len(workflow.phases))
    phase = workflow.phases[phase_index] if workflow.phases else PhaseView("Agents", workflow.agents)
    agents = list(phase.agents)
    agent = agents[_clamp(state.agent_index, len(agents))] if agents else None
    return phase, agents, agent


def _agent_geometry(width: int, height: int) -> tuple[int, int, int]:
    left_width = max(22, min(28, width // 4))
    right_width = max(20, width - left_width - 5)
    body_height = max(5, height - _DETAIL_HEADER_LINES - 2)
    return left_width, right_width, body_height


def _agent_detail(agent: AgentView | None, width: int) -> list[Line]:
    if agent is None:
        return [[_sp("No agents in this phase.", "dim")]]
    inner = max(8, width - 2)
    metrics = f"{_tokens(agent.tokens)} tok · {agent.tool_calls} tool calls"
    if agent.duration_seconds is not None:
        metrics += f" · {_duration(agent.duration_seconds)}"
    lines: list[Line] = [
        [_sp(_status_label(agent.status), _status_style(agent.status))]
        + ([_sp(f" · {agent.model}", "dim")] if agent.model else []),
        [_sp(metrics, "dim")],
        [],
        [_sp(f"Prompt · {max(1, len(agent.prompt.splitlines()))} lines", "dim")],
    ]
    preview, hidden = _clip(_wrapped_block(agent.prompt, inner), 6)
    lines.extend([_sp("  " + item)] for item in preview)
    if hidden:
        lines.append([_sp(f"  … {hidden} more line{'s' if hidden != 1 else ''}", "dim")])
    lines.append([])
    lines.append([_sp(f"Activity · last {min(6, len(agent.activity))} of {len(agent.activity)}", "dim")])
    if agent.activity:
        lines.extend([_sp("  " + _crop(item, inner))] for item in agent.activity[-6:])
    else:
        lines.append([_sp("  No tool activity yet", "dim")])
    lines.append([])
    lines.append([_sp("Outcome", "dim")])
    lines.extend([_sp("  " + item)] for item in _wrapped_block(agent.outcome, inner))
    return lines


def agent_detail_scroll_max(workflow: WorkflowView, state: RenderState, width: int, height: int) -> int:
    """Largest valid `detail_scroll` for the agent view at this size."""
    width = max(40, width)
    height = max(12, height)
    _phase, _agents, agent = _selected_agent(workflow, state)
    _left_width, right_width, body_height = _agent_geometry(width, height)
    return _max_scroll(len(_agent_detail(agent, right_width)), body_height)


def _max_scroll(total: int, height: int) -> int:
    if total <= height:
        return 0
    return max(0, total - max(1, height - 1))


def _scroll_view(lines: list[Line], offset: int, height: int, width: int) -> list[Line]:
    total = len(lines)
    if total <= height:
        return lines
    view_height = max(1, height - 1)
    offset = max(0, min(offset, total - view_height))
    window = lines[offset : offset + view_height]
    window = window + [[] for _ in range(max(0, view_height - len(window)))]
    end = offset + min(view_height, total - offset)
    arrow = "↕" if offset > 0 and end < total else ("↓" if end < total else "↑")
    window.append(_rjust([_sp(f"{offset + 1}-{end} of {total} {arrow}", "dim")], width))
    return window


def _agent_row(agent: AgentView, width: int, *, selected: bool = False) -> Line:
    left: Line = [
        _sp("❯ ", "sel") if selected else _sp("  "),
        _sp(_status_icon(agent.status), _status_style(agent.status)),
        _sp(" " + agent.label, "sel" if selected else ""),
    ]
    metrics = f"{_tokens(agent.tokens)} tok · {agent.tool_calls} tools"
    if agent.duration_seconds is not None:
        metrics += f" · {_duration(agent.duration_seconds)}"
    right: Line = []
    if agent.model:
        right.append(_sp(agent.model + "  ", "dim"))
    right.append(_sp(metrics, "dim"))
    gap = max(1, width - _line_width(left) - _line_width(right))
    return _crop_line(left + [_sp(" " * gap)] + right, width)


# --------------------------------------------------------------------------- #
# Two-column panel
# --------------------------------------------------------------------------- #
def _two_columns(
    left_title: str,
    left_lines: list[Line],
    right_title: str,
    right_lines: list[Line],
    *,
    left_width: int,
    right_width: int,
    height: int,
) -> list[Line]:
    divider = _sp(" │ ", "divider")
    heading = (
        [_sp("  ")]
        + _pad_line(_crop_line([_sp(left_title, "dim")], left_width), left_width)
        + [divider]
        + _crop_line([_sp(right_title, "dim")], right_width)
    )
    out: list[Line] = [heading]
    for index in range(height):
        left = left_lines[index] if index < len(left_lines) else []
        right = right_lines[index] if index < len(right_lines) else []
        out.append(
            [_sp("  ")]
            + _pad_line(_crop_line(left, left_width), left_width)
            + [divider]
            + _pad_line(_crop_line(right, right_width), right_width)
        )
    return out


# --------------------------------------------------------------------------- #
# Line helpers
# --------------------------------------------------------------------------- #
def _sp(text: str, style: str = "") -> Span:
    return Span(text, style)


def _rule(width: int) -> Line:
    return [_sp("  " + "─" * max(0, width - 4), "rule")]


def _footer(default: str, message: str, width: int) -> Line:
    return _crop_line([_sp("  " + (message or default), "footer")], width)


def _pad_to_footer(lines: list[Line], default: str, message: str, width: int, height: int) -> list[Line]:
    lines.extend([] for _ in range(max(0, height - len(lines) - 2)))
    lines.append([])
    lines.append(_footer(default, message, width))
    return lines


def _left_right(left: Line, right_text: str, right_style: str, width: int) -> Line:
    right_width = _display_width(right_text)
    left = _crop_line(left, max(1, width - right_width))
    gap = max(0, width - right_width - _line_width(left))
    return left + [_sp(" " * gap)] + [_sp(right_text, right_style)]


def _rjust(line: Line, width: int) -> Line:
    used = _line_width(line)
    if used >= width:
        return _crop_line(line, width)
    return [_sp(" " * (width - used))] + line


def _line_width(line: Line) -> int:
    return sum(_display_width(span.text) for span in line)


def _pad_line(line: Line, width: int) -> Line:
    used = _line_width(line)
    if used < width:
        return line + [_sp(" " * (width - used))]
    return line


def _crop_line(line: Line, width: int) -> Line:
    if _line_width(line) <= width:
        return line
    budget = max(0, width - 1)
    out: Line = []
    used = 0
    for span in line:
        span_width = _display_width(span.text)
        if used + span_width <= budget:
            out.append(span)
            used += span_width
            continue
        piece = _crop_cells(span.text, budget - used)
        if piece:
            out.append(Span(piece, span.style))
        break
    out.append(Span("…", out[-1].style if out else ""))
    return out


def _fit(lines: list[Line], width: int, height: int) -> list[Line]:
    fitted = [_crop_line(line, width) for line in lines[:height]]
    return fitted + [[] for _ in range(max(0, height - len(fitted)))]


def _window_lines(lines: list[Line], selected_index: int, height: int) -> list[Line]:
    if len(lines) <= height:
        return lines
    selected_index = _clamp(selected_index, len(lines))
    start = max(0, min(selected_index - height // 2, len(lines) - height))
    return lines[start : start + height]


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def _wrapped_block(text: str, width: int) -> list[str]:
    """Wrap text to `width`, preserving blank lines and leading indentation so
    structured output (JSON, code) stays readable when scrolled."""
    width = max(8, width)
    out: list[str] = []
    for raw in str(text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            out.append("")
            continue
        indent = raw[: len(raw) - len(raw.lstrip())][: max(0, width - 4)]
        out.extend(
            textwrap.wrap(
                stripped,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [indent]
        )
    return out or [""]


def _clip(lines: list[str], max_lines: int) -> tuple[list[str], int]:
    if len(lines) <= max_lines:
        return lines, 0
    return lines[:max_lines], len(lines) - max_lines


def _crop(text: str, width: int) -> str:
    text = str(text or "")
    if _display_width(text) <= width:
        return text
    if width <= 3:
        return _crop_cells(text, width)
    return _crop_cells(text, width - 3) + "..."


def _crop_cells(text: str, width: int) -> str:
    cells = 0
    chars: list[str] = []
    for char in text:
        char_width = _char_width(char)
        if cells + char_width > width:
            break
        chars.append(char)
        cells += char_width
    return "".join(chars)


def display_width(text: str) -> int:
    return _display_width(text)


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in str(text or ""))


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _clamp(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))


def _status_icon(status: str) -> str:
    return {
        "queued": "○",
        "running": "◌",
        "stopping": "◌",
        "paused": "Ⅱ",
        "completed": "✓",
        "done": "✓",
        "failed": "!",
        "error": "!",
        "stopped": "x",
    }.get(status, "?")


def _status_style(status: str) -> str:
    return {
        "queued": "dim",
        "running": "running",
        "stopping": "running",
        "paused": "warn",
        "completed": "ok",
        "done": "ok",
        "failed": "error",
        "error": "error",
        "stopped": "dim",
    }.get(status, "dim")


def _status_label(status: str) -> str:
    return {
        "queued": "Queued",
        "running": "Running",
        "stopping": "Stopping",
        "paused": "Paused",
        "completed": "Completed",
        "done": "Completed",
        "failed": "Failed",
        "error": "Error",
        "stopped": "Stopped",
    }.get(status, status.title() or "Unknown")


def _tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{_trim(value / 1_000_000)}m"
    if value >= 1000:
        return f"{_trim(value / 1000)}k"
    return str(value)


def _trim(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
