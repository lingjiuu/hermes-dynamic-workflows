"""Immutable artifacts and provenance for verified workflow launches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

_EXECUTION_IDENTITY_DOMAIN = b"hermes-dws-execution-identity-v1\0"
_SCRIPT_DOMAIN = b"hermes-dws-immutable-script-v1\0"
VERIFIED_PROVENANCE_VERSION = "hermes-dws-verified-run/v1"
_BOUND_FIELDS = (
    "runId",
    "taskId",
    "approvalId",
    "scriptDigest",
    "executionDigest",
    "executionIdentity",
    "authorityPresetId",
    "authorityPolicyDigest",
    "enforcedMaxScale",
    "cwd",
    "args",
)
_REQUIRED_COMPLETION_FIELDS = ("completedAt", "compilerReceipt")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 JSON representation used by verified records."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def execution_identity_digest(envelope: dict[str, Any]) -> str:
    """Hash a versioned execution-identity envelope with domain separation."""

    if not isinstance(envelope, dict):
        raise TypeError("execution identity must be an object")
    version = envelope.get("version")
    if isinstance(version, str):
        valid_version = bool(version.strip())
    else:
        valid_version = isinstance(version, int) and not isinstance(version, bool) and version > 0
    if not valid_version:
        raise ValueError("execution identity must include a non-empty version")
    return hashlib.sha256(_EXECUTION_IDENTITY_DOMAIN + canonical_json_bytes(envelope)).hexdigest()


def immutable_script_digest(script_bytes: bytes) -> str:
    if not isinstance(script_bytes, bytes):
        raise TypeError("verified workflow script must be immutable bytes")
    return hashlib.sha256(_SCRIPT_DOMAIN + script_bytes).hexdigest()


def make_pending_provenance(
    *,
    run_id: str,
    task_id: str,
    approval_id: str,
    script_bytes: bytes,
    execution_identity: dict[str, Any],
    authority_preset_id: str,
    authority_policy_digest: str,
    enforced_max_scale: int,
    cwd: str,
    args: Any = None,
) -> dict[str, Any]:
    if not authority_preset_id:
        raise ValueError("verified authority preset ID must not be empty")
    if isinstance(enforced_max_scale, bool) or not isinstance(enforced_max_scale, int) or enforced_max_scale < 1:
        raise ValueError("enforced maximum scale must be a positive integer")
    identity = _json_copy(execution_identity)
    arguments = _json_copy(args)
    return {
        "version": VERIFIED_PROVENANCE_VERSION,
        "state": "pending",
        "runId": str(run_id),
        "taskId": str(task_id),
        "approvalId": str(approval_id),
        "scriptDigest": immutable_script_digest(script_bytes),
        "executionDigest": execution_identity_digest(identity),
        "executionIdentity": identity,
        "authorityPresetId": str(authority_preset_id),
        "authorityPolicyDigest": str(authority_policy_digest),
        "enforcedMaxScale": enforced_max_scale,
        "cwd": str(cwd),
        "args": arguments,
    }


@dataclass(frozen=True)
class PreparedRun:
    """Opaque durable handle returned by the verified prepare phase."""

    run_id: str
    task_id: str
    approval_id: str
    script_path: str
    script_digest: str
    execution_digest: str
    authority_preset_id: str
    enforced_max_scale: int
    _pending_bytes: bytes = field(repr=False)
    _nonce: str = field(repr=False)

    @classmethod
    def from_pending(cls, pending: dict[str, Any], *, script_path: str, nonce: str) -> "PreparedRun":
        if pending.get("state") != "pending" or pending.get("version") != VERIFIED_PROVENANCE_VERSION:
            raise ValueError("prepared run requires pending verified provenance")
        encoded = canonical_json_bytes(pending)
        return cls(
            run_id=str(pending["runId"]),
            task_id=str(pending["taskId"]),
            approval_id=str(pending["approvalId"]),
            script_path=str(script_path),
            script_digest=str(pending["scriptDigest"]),
            execution_digest=str(pending["executionDigest"]),
            authority_preset_id=str(pending["authorityPresetId"]),
            enforced_max_scale=int(pending["enforcedMaxScale"]),
            _pending_bytes=encoded,
            _nonce=str(nonce),
        )

    @property
    def pending_provenance(self) -> dict[str, Any]:
        return json.loads(self._pending_bytes.decode("utf-8"))

    def completed_provenance(self, additional: dict[str, Any] | None = None) -> dict[str, Any]:
        return build_completed_provenance(self, additional)


def build_completed_provenance(
    prepared: PreparedRun,
    additional: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = prepared.pending_provenance
    completed["state"] = "completed"
    if additional is not None:
        if not isinstance(additional, dict):
            raise TypeError("completed provenance additions must be an object")
        forbidden = {"version", "state", *_BOUND_FIELDS}.intersection(additional)
        if forbidden:
            raise ValueError(
                "completed provenance additions cannot replace prepared fields: "
                + ", ".join(sorted(forbidden))
            )
        completed.update(_json_copy(additional))
    return completed


def validate_completed_provenance(prepared: PreparedRun, completed: dict[str, Any]) -> None:
    if not isinstance(completed, dict):
        raise TypeError("completed provenance must be an object")
    if completed.get("version") != VERIFIED_PROVENANCE_VERSION:
        raise ValueError("completed provenance version does not match the prepared run")
    if completed.get("state") != "completed":
        raise ValueError("completed provenance must have state='completed'")
    pending = prepared.pending_provenance
    for field_name in _BOUND_FIELDS:
        if field_name not in completed:
            raise ValueError(f"completed provenance is missing prepared field: {field_name}")
        if canonical_json_bytes(completed[field_name]) != canonical_json_bytes(pending[field_name]):
            raise ValueError(f"completed provenance does not bind prepared field: {field_name}")
    for field_name in _REQUIRED_COMPLETION_FIELDS:
        value = completed.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"completed provenance is missing required completion field: {field_name}")
    canonical_json_bytes(completed)


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))
