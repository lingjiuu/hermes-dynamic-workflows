"""Agent type resolution for workflow child agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.store import default_store_root


@dataclass(frozen=True)
class AgentTypeSpec:
    name: str
    instructions: str
    source: str
    toolsets: tuple[str, ...] = ()
    model: str | None = None
    provider: str | None = None
    isolation: str | None = None


def resolve_agent_type(name: str | None, *, cwd: str | None = None, task_id: str | None = None) -> AgentTypeSpec | None:
    clean = str(name or "").strip()
    if not clean:
        return None

    skill_spec = _resolve_hermes_skill(clean, task_id=task_id)
    if skill_spec is not None:
        return skill_spec

    path = _find_agent_type_file(clean, cwd=cwd)
    if path is None:
        return None
    return _load_agent_type_file(clean, path)


def _resolve_hermes_skill(name: str, *, task_id: str | None) -> AgentTypeSpec | None:
    """Load agentType through Hermes' own skill resolver when possible."""
    try:
        from agent.skill_commands import build_preloaded_skills_prompt

        prompt, loaded_names, missing = build_preloaded_skills_prompt([name], task_id=task_id)
    except Exception:
        return None
    if not prompt.strip() or not loaded_names or missing:
        return None
    return AgentTypeSpec(
        name=loaded_names[0],
        instructions=prompt.strip(),
        source=f"hermes-skill:{name}",
    )


def _find_agent_type_file(name: str, *, cwd: str | None) -> Path | None:
    rel = _safe_agent_type_relative_path(name)
    bases: list[Path] = []
    if cwd:
        bases.append(Path(cwd).expanduser() / ".hermes" / "workflow-agent-types")
    bases.append(default_store_root() / "agent-types")
    plugin_root = Path(__file__).resolve().parent.parent
    bases.append(plugin_root / "agent-types")

    suffix = rel.suffix.lower()
    rels = [rel] if suffix else [rel.with_suffix(ext) for ext in (".md", ".yaml", ".yml", ".json")]
    for base in bases:
        for candidate_rel in rels:
            candidate = (base / candidate_rel).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
    return None


def _safe_agent_type_relative_path(name: str) -> Path:
    raw = str(name or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("agentType must not be empty")
    parts = [part for part in raw.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"invalid agentType: {name!r}")
    if any(part.startswith(".") for part in parts):
        raise ValueError(f"invalid agentType: {name!r}")
    return Path(*parts)


def _load_agent_type_file(name: str, path: Path) -> AgentTypeSpec:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_structured_agent_type(name, path, _read_json(path))
    if suffix in {".yaml", ".yml"}:
        return _load_structured_agent_type(name, path, _read_yaml(path))
    return _load_markdown_agent_type(name, path)


def _load_markdown_agent_type(name: str, path: Path) -> AgentTypeSpec:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(text)
    body = body.strip() or text.strip()
    return AgentTypeSpec(
        name=str(frontmatter.get("name") or Path(name).stem),
        instructions=body,
        source=str(path),
        toolsets=_as_tuple(frontmatter.get("toolsets") or frontmatter.get("tools")),
        model=_as_optional_str(frontmatter.get("model")),
        provider=_as_optional_str(frontmatter.get("provider")),
        isolation=_as_optional_str(frontmatter.get("isolation")),
    )


def _load_structured_agent_type(name: str, path: Path, data: Any) -> AgentTypeSpec:
    if not isinstance(data, dict):
        raise ValueError(f"agentType file must contain an object: {path}")
    instructions = (
        data.get("instructions")
        or data.get("system_prompt")
        or data.get("prompt")
        or data.get("content")
        or ""
    )
    instructions = str(instructions).strip()
    if not instructions:
        raise ValueError(f"agentType file is missing instructions: {path}")
    return AgentTypeSpec(
        name=str(data.get("name") or Path(name).stem),
        instructions=instructions,
        source=str(path),
        toolsets=_as_tuple(data.get("toolsets") or data.get("tools")),
        model=_as_optional_str(data.get("model")),
        provider=_as_optional_str(data.get("provider")),
        isolation=_as_optional_str(data.get("isolation")),
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    try:
        from agent.skill_utils import parse_frontmatter

        frontmatter, body = parse_frontmatter(text)
        return (frontmatter or {}, body or "")
    except Exception:
        pass
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    try:
        data = _read_yaml_text(raw)
    except Exception:
        data = {}
    return (data if isinstance(data, dict) else {}, body)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Any:
    return _read_yaml_text(path.read_text(encoding="utf-8"))


def _read_yaml_text(text: str) -> Any:
    try:
        import yaml
    except Exception as exc:
        return _read_simple_yaml_text(text)
    return yaml.safe_load(text) or {}


def _read_simple_yaml_text(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
        else:
            data[key] = value.strip("'\"")
    return data


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _as_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    return clean or None
