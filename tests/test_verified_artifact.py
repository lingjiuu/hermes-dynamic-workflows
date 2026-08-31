from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_dynamic_workflows.run.verified_artifact import (
    PreparedRun,
    build_completed_provenance,
    execution_identity_digest,
    immutable_script_digest,
    make_pending_provenance,
    validate_completed_provenance,
)
from hermes_dynamic_workflows.storage.store import WorkflowStore


class VerifiedArtifactTests(unittest.TestCase):
    def test_execution_identity_digest_is_canonical_and_version_bound(self):
        left = {"version": "dws-execution-identity/v1", "target": "hermes", "nested": {"b": 2, "a": 1}}
        right = {"nested": {"a": 1, "b": 2}, "target": "hermes", "version": "dws-execution-identity/v1"}

        self.assertEqual(execution_identity_digest(left), execution_identity_digest(right))
        self.assertNotEqual(
            execution_identity_digest(left),
            execution_identity_digest({**left, "version": "dws-execution-identity/v2"}),
        )

    def test_completed_provenance_binds_every_prepared_value(self):
        identity = {"version": "dws-execution-identity/v1", "target": "hermes"}
        pending = make_pending_provenance(
            run_id="wf_abcdef-123",
            task_id="wg1234567",
            approval_id="wfa_approval-1",
            script_bytes=b"return 1\n",
            execution_identity=identity,
            authority_preset_id="dws-read-only-v1",
            authority_policy_digest="f" * 64,
            enforced_max_scale=4,
            cwd="/trusted/project",
            args={"question": "why"},
        )
        prepared = PreparedRun.from_pending(pending, script_path="/immutable/wf.py", nonce="secret")
        completed = build_completed_provenance(
            prepared,
            {
                "completedAt": "2026-08-31T12:00:00+00:00",
                "compilerReceipt": "ok",
            },
        )

        validate_completed_provenance(prepared, completed)
        for field in (
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
        ):
            changed = dict(completed)
            changed[field] = "substituted"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_completed_provenance(prepared, changed)

        for field in ("completedAt", "compilerReceipt"):
            missing = dict(completed)
            missing.pop(field, None)
            with self.subTest(required_completion_field=field), self.assertRaises(ValueError):
                validate_completed_provenance(prepared, missing)

    def test_store_persists_verified_script_bytes_and_verifies_digest(self):
        with TemporaryDirectory() as tmp:
            store = WorkflowStore(Path(tmp))
            script_bytes = b"meta = {\"name\": \"x\", \"description\": \"y\"}\nreturn 1\n"
            digest = immutable_script_digest(script_bytes)

            path = store.save_verified_script("wf_abcdef12-345", script_bytes, expected_digest=digest)

            self.assertEqual(path.read_bytes(), script_bytes)
            self.assertEqual(store.load_verified_script("wf_abcdef12-345", expected_digest=digest), script_bytes)

            path.write_bytes(script_bytes + b"# tampered\n")
            with self.assertRaises(ValueError):
                store.load_verified_script("wf_abcdef12-345", expected_digest=digest)


if __name__ == "__main__":
    unittest.main()
