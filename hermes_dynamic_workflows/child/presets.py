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
    description: str = ""
    toolsets: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    model: str | None = None
    isolation: str | None = None


def resolve_agent_type(name: str | None, *, cwd: str | None = None) -> AgentTypeSpec | None:
    clean = str(name or "").strip()
    if not clean:
        return None

    path = _find_agent_type_file(clean, cwd=cwd)
    if path is not None:
        return _load_agent_type_file(clean, path)

    folded = clean.casefold()
    for spec in list_agent_types(cwd=cwd):
        if spec.name.casefold() == folded:
            return spec
    return None


def list_agent_types(*, cwd: str | None = None) -> list[AgentTypeSpec]:
    """Return active workflow agent types in resolution precedence order."""
    active: dict[str, AgentTypeSpec] = {}
    for base in _agent_type_bases(cwd):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".yaml", ".yml", ".json"}:
                continue
            try:
                rel = path.resolve().relative_to(base.resolve())
            except ValueError:
                continue
            try:
                spec = _load_agent_type_file(str(rel.with_suffix("")), path)
            except Exception:
                continue
            active.setdefault(spec.name, spec)
    return list(active.values())


def _find_agent_type_file(name: str, *, cwd: str | None) -> Path | None:
    rel = _safe_agent_type_relative_path(name)
    suffix = rel.suffix.lower()
    rels = [rel] if suffix else [rel.with_suffix(ext) for ext in (".md", ".yaml", ".yml", ".json")]
    for base in _agent_type_bases(cwd):
        for candidate_rel in rels:
            candidate = (base / candidate_rel).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
    return None


def _agent_type_bases(cwd: str | None) -> list[Path]:
    bases: list[Path] = []
    if cwd:
        bases.append(Path(cwd).expanduser() / ".hermes" / "dynamic-workflows" / "agents")
    bases.append(default_store_root() / "agents")
    plugin_root = Path(__file__).resolve().parent.parent
    bases.append(plugin_root / "agents")
    return bases


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
        description=_description_from(frontmatter),
        toolsets=_as_tuple(frontmatter.get("toolsets") or frontmatter.get("tools")),
        allowed_tools=_as_tuple(frontmatter.get("allowed_tools")),
        disallowed_tools=_as_tuple(frontmatter.get("disallowed_tools")),
        model=_as_optional_str(frontmatter.get("model")),
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
        description=_description_from(data),
        toolsets=_as_tuple(data.get("toolsets") or data.get("tools")),
        allowed_tools=_as_tuple(data.get("allowed_tools")),
        disallowed_tools=_as_tuple(data.get("disallowed_tools")),
        model=_as_optional_str(data.get("model")),
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


def _description_from(data: dict[str, Any]) -> str:
    description = str(data.get("description") or data.get("whenToUse") or "").strip()
    return description.replace("\\n", "\n").replace('\\"', '"')


def _as_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip()
    return clean or None
