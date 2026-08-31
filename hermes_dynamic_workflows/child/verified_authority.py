"""Runtime-owned authority policy for verified DWS child launches."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..core.errors import ChildAgentError
from ..core.types import ChildAgentRequest, ResolvedAgentSpec

VERIFIED_AUTHORITY_POLICY_VERSION = "hermes-dws-authority/v1"


@dataclass(frozen=True)
class VerifiedAuthorityPreset:
    preset_id: str
    agent_type: str
    instructions: str
    toolsets: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...]
    isolation: str
    model_policy: str
    allow_request_overrides: bool
    maximum_scale: int

    def policy_record(self) -> dict[str, object]:
        return {
            "agentType": self.agent_type,
            "allowRequestOverrides": self.allow_request_overrides,
            "allowedTools": list(self.allowed_tools),
            "disallowedTools": list(self.disallowed_tools),
            "instructions": self.instructions,
            "isolation": self.isolation,
            "maximumScale": self.maximum_scale,
            "modelPolicy": self.model_policy,
            "toolsets": list(self.toolsets),
        }


_READ_ONLY = VerifiedAuthorityPreset(
    preset_id="dws-read-only-v1",
    agent_type="verified-dws-read-only",
    instructions=(
        "Execute the compiled workflow task without widening its authority. "
        "Read and analyze only; do not mutate files or external state."
    ),
    toolsets=("web", "file", "terminal"),
    allowed_tools=("web_search", "web_extract", "read_file", "search_files", "terminal"),
    disallowed_tools=("write_file", "patch", "process", "skill_manage"),
    isolation="shared",
    model_policy="inherit",
    allow_request_overrides=False,
    maximum_scale=64,
)

VERIFIED_AUTHORITY_PRESETS: Mapping[str, VerifiedAuthorityPreset] = MappingProxyType(
    {_READ_ONLY.preset_id: _READ_ONLY}
)
VERIFIED_AUTHORITY_PRESET_IDS = tuple(sorted(VERIFIED_AUTHORITY_PRESETS))


def resolve_verified_authority_preset(preset_id: str) -> VerifiedAuthorityPreset:
    clean = str(preset_id or "").strip()
    try:
        return VERIFIED_AUTHORITY_PRESETS[clean]
    except KeyError as exc:
        available = ", ".join(VERIFIED_AUTHORITY_PRESET_IDS)
        raise ValueError(f"unknown verified authority preset {preset_id!r}; available: {available}") from exc


def verified_authority_policy_bytes() -> bytes:
    policy = {
        "presets": {
            preset_id: VERIFIED_AUTHORITY_PRESETS[preset_id].policy_record()
            for preset_id in VERIFIED_AUTHORITY_PRESET_IDS
        },
        "version": VERIFIED_AUTHORITY_POLICY_VERSION,
    }
    return (
        json.dumps(policy, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def verified_authority_policy_digest() -> str:
    return hashlib.sha256(verified_authority_policy_bytes()).hexdigest()


@dataclass(frozen=True)
class VerifiedAuthorityContext:
    preset_id: str
    enforced_max_scale: int
    cwd: str
    execution_digest: str
    policy_digest: str


def lock_verified_authority(
    preset_id: str,
    *,
    enforced_max_scale: int,
    cwd: str,
    execution_digest: str,
) -> VerifiedAuthorityContext:
    preset = resolve_verified_authority_preset(preset_id)
    if isinstance(enforced_max_scale, bool) or not isinstance(enforced_max_scale, int):
        raise ValueError("enforced maximum scale must be an integer")
    if enforced_max_scale < 1 or enforced_max_scale > preset.maximum_scale:
        raise ValueError(
            f"enforced maximum scale must be between 1 and {preset.maximum_scale} for {preset.preset_id}"
        )
    clean_digest = str(execution_digest or "").strip().lower()
    if len(clean_digest) != 64 or any(ch not in "0123456789abcdef" for ch in clean_digest):
        raise ValueError("execution digest must be a SHA-256 hex digest")
    return VerifiedAuthorityContext(
        preset_id=preset.preset_id,
        enforced_max_scale=enforced_max_scale,
        cwd=str(Path(cwd).expanduser().resolve()),
        execution_digest=clean_digest,
        policy_digest=verified_authority_policy_digest(),
    )


class VerifiedAuthorityEnforcer:
    """Constrain every child request to one immutable verified authority lock."""

    def __init__(self, context: VerifiedAuthorityContext):
        self.context = context
        self._launch_count = 0
        self._lock = threading.Lock()

    @property
    def launch_count(self) -> int:
        with self._lock:
            return self._launch_count

    def constrain(self, request: ChildAgentRequest) -> ChildAgentRequest:
        preset = self._validated_preset()
        if request.request_overrides:
            raise ChildAgentError("verified workflow children forbid provider request overrides")
        request_cwd = str(Path(request.cwd or "").expanduser().resolve())
        if request_cwd != self.context.cwd:
            raise ChildAgentError(
                "verified workflow child cwd does not match the prepared authority context"
            )
        with self._lock:
            if self._launch_count >= self.context.enforced_max_scale:
                raise ChildAgentError(
                    "verified workflow enforced maximum scale "
                    f"of {self.context.enforced_max_scale} was exceeded"
                )
            self._launch_count += 1

        from .presets import verified_agent_type_spec

        agent_type = verified_agent_type_spec(preset)
        resolved = ResolvedAgentSpec(
            requested_agent_type=agent_type.name,
            agent_type_spec=agent_type,
            model=None,
            isolation=preset.isolation,
            toolsets=preset.toolsets,
            allowed_tools=preset.allowed_tools,
            disallowed_tools=preset.disallowed_tools,
            system_prompt_hash=self.context.policy_digest,
            workspace=self.context.cwd,
        )
        return replace(
            request,
            toolsets=list(preset.toolsets),
            model=None,
            agent_type=agent_type.name,
            isolation=preset.isolation,
            cwd=self.context.cwd,
            request_overrides=None,
            resolved=resolved,
        )

    def _validated_preset(self) -> VerifiedAuthorityPreset:
        if self.context.policy_digest != verified_authority_policy_digest():
            raise ChildAgentError("verified authority policy digest does not match the runtime table")
        try:
            preset = resolve_verified_authority_preset(self.context.preset_id)
        except ValueError as exc:
            raise ChildAgentError(str(exc)) from exc
        if self.context.enforced_max_scale > preset.maximum_scale:
            raise ChildAgentError("verified authority scale exceeds the runtime preset maximum")
        return preset
