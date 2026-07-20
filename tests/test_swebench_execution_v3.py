from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from anchor_mvp.tooling.swebench_execution_v3 import (
    DISTILLATION_VALIDATION_STATE_SCHEMA,
    ExecutionContractError,
    SealedValidationResult,
    _wsl_behavior_probes,
    approved_builder_policy,
    build_execution_attestation,
    distillation_execution_evidence_hashes,
    distillation_lineage_sha256,
    distillation_validation_state_sha256,
    evaluate_v3_gold_gate,
    resolve_official_instance_image_key,
    sha256_file,
    sign_distillation_execution_receipt,
    verify_distillation_execution_receipt,
    verify_execution_attestation,
)
from anchor_mvp.tooling.tool_contract import v3_contract_descriptor


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "tooling" / "swebench_execution_v3.lock.json"
WRAPPER = ROOT / "scripts" / "tooling" / "anchor_validate.py"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_distillation_lineage_hash_is_fixed_order_and_revision_bound() -> None:
    records = {
        stage: {"revision": 1, "artifact_sha256": _digest(stage)}
        for stage in (
            "security",
            "domain_review",
            "domain_builder",
            "tool_policy",
            "planner",
        )
    }
    bindings = {
        "checkpoint_id": _digest("checkpoint"),
        "config_sha256": _digest("config"),
        "execution_lock_sha256": _digest("lock"),
        "task_id_sha256": _digest("task"),
    }
    first = distillation_lineage_sha256(
        **bindings,
        stage_records=records,
    )
    reordered = distillation_lineage_sha256(
        **bindings,
        stage_records=dict(reversed(list(records.items()))),
    )
    assert first == reordered
    changed = {
        **records,
        "domain_builder": {
            "revision": 2,
            "artifact_sha256": records["domain_builder"]["artifact_sha256"],
        },
    }
    assert first != distillation_lineage_sha256(
        **bindings,
        stage_records=changed,
    )


def test_distillation_validation_state_binds_terminal_output_and_patch() -> None:
    command = "anchor-validate compile"
    command_sha = _digest(command)
    output_sha = _digest("validator-output")
    patch_sha = _digest("final-patch")
    validator_sha = _digest("validator-version")
    changed_files = [{"path": "src/a.js", "sha256": _digest("file-state")}]
    changed_paths = ["src/a.js"]
    validator_result = {
        "schema_version": "anchor.train-sandbox-validation.v1",
        "validator_version": "1.0.1",
        "mode": "compile",
        "success": True,
        "not_official_swebench_pass": True,
        "validation_level": "syntax",
        "changed_paths": changed_paths,
        "changed_paths_sha256": _digest(
            json.dumps(
                changed_paths,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "final_state_sha256": _digest("final-state"),
        "validators": ["node-check"],
    }
    builder = {
        "tool_calls": [
            {
                "sequence": 1,
                "tool": "bash",
                "command": command,
                "command_sha256": command_sha,
                "invocation_sha256": _digest("invocation"),
                "execution_scope": "isolated-instance-container",
            }
        ],
        "tool_results": [
            {
                "sequence": 1,
                "tool": "bash",
                "status": "completed",
                "exit_code": 0,
                "command_sha256": command_sha,
                "invocation_sha256": _digest("invocation"),
                "output_sha256": output_sha,
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            }
        ],
        "validation_state": {
            "schema_version": DISTILLATION_VALIDATION_STATE_SCHEMA,
            "final_patch_sha256": patch_sha,
            "changed_files": changed_files,
            "changed_files_sha256": _digest(
                json.dumps(
                    changed_files,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "terminal_validation_output_sha256": output_sha,
            "terminal_command_sha256": command_sha,
            "validator_version_sha256": validator_sha,
            "validator_result": validator_result,
        },
    }
    digest = distillation_validation_state_sha256(
        builder,
        final_patch_sha256=patch_sha,
        validator_version_sha256=validator_sha,
    )
    assert len(digest) == 64 and int(digest, 16) >= 0
    with pytest.raises(
        ExecutionContractError,
        match="distillation_validation_state_unbound",
    ):
        distillation_validation_state_sha256(
            builder,
            final_patch_sha256=_digest("later-patch"),
            validator_version_sha256=validator_sha,
        )


def test_distillation_receipt_binds_real_validation_without_official_pass() -> None:
    command = "anchor-validate test"
    command_sha = _digest(command)
    invocation_sha = _digest("invocation")
    calls = [
        {
            "sequence": 1,
            "tool": "bash",
            "command": command,
            "command_sha256": command_sha,
            "invocation_sha256": invocation_sha,
            "execution_scope": "isolated-instance-container",
        }
    ]
    results = [
        {
            "sequence": 1,
            "tool": "bash",
            "status": "completed",
            "exit_code": 0,
            "command_sha256": command_sha,
            "invocation_sha256": invocation_sha,
            "output_sha256": _digest("validation-output"),
            "visible_to_model": True,
            "provenance": "public-repo-validation-model-visible",
            "execution_scope": "isolated-instance-container",
        }
    ]
    hashes = distillation_execution_evidence_hashes(calls, results)
    assert hashes["qualifying_validation_count"] == 1
    validation_state = {"schema_version": "test.validation-state.v1"}
    bindings = {
        "checkpoint_id": _digest("checkpoint"),
        "config_sha256": _digest("config"),
        "execution_lock_sha256": _digest("lock"),
        "source_bank_manifest_sha256": _digest("bank-manifest"),
        "candidate_task_artifact_sha256": _digest("task-artifact"),
        "candidate_work_order_artifacts_sha256": _digest("order-artifacts"),
        "task_id_sha256": _digest("task"),
        "instance_id_sha256": _digest("instance"),
        "repo_sha256": _digest("repo"),
        "base_commit": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "image_id_sha256": _digest("image-id"),
        "final_patch_sha256": _digest("patch"),
        "tool_transcript_sha256": hashes["tool_transcript_sha256"],
        "validation_evidence_sha256": hashes["validation_evidence_sha256"],
        "validation_state_sha256": _digest(
            json.dumps(
                validation_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "validator_version_sha256": _digest("validator-version"),
        "lineage_sha256": _digest("lineage"),
    }
    receipt = sign_distillation_execution_receipt(
        bindings=bindings,
        validation_state=validation_state,
        receipt_id=_digest("receipt"),
        issued_at="2026-07-18T00:00:00Z",
        trusted_receipt_key=_RECEIPT_KEY,
    )
    assert receipt["status"] == "SELF_VERIFIED"
    assert receipt["not_official_swebench_pass"] is True
    assert receipt["cleanup_success"] is True
    assert verify_distillation_execution_receipt(
        receipt,
        trusted_receipt_key=_RECEIPT_KEY,
        expected_bindings=bindings,
    )
    assert not verify_distillation_execution_receipt(
        {**receipt, "cleanup_success": False},
        trusted_receipt_key=_RECEIPT_KEY,
        expected_bindings=bindings,
    )


def test_distillation_evidence_rejects_empty_validation_output() -> None:
    command = "anchor-validate test"
    command_sha = _digest(command)
    invocation_sha = _digest("invocation")
    calls = [
        {
            "sequence": 1,
            "tool": "bash",
            "command": command,
            "command_sha256": command_sha,
            "invocation_sha256": invocation_sha,
            "execution_scope": "isolated-instance-container",
        }
    ]
    results = [
        {
            "sequence": 1,
            "tool": "bash",
            "status": "completed",
            "exit_code": 0,
            "command_sha256": command_sha,
            "invocation_sha256": invocation_sha,
            "output_sha256": None,
            "visible_to_model": True,
            "provenance": "public-repo-validation-model-visible",
            "execution_scope": "isolated-instance-container",
        }
    ]
    with pytest.raises(
        ExecutionContractError,
        match="distillation_validation_evidence_invalid",
    ):
        distillation_execution_evidence_hashes(calls, results)


def test_distillation_evidence_does_not_promote_lint_to_self_verified() -> None:
    command = "anchor-validate lint"
    command_sha = _digest(command)
    invocation_sha = _digest("lint-invocation")
    calls = [
        {
            "sequence": 1,
            "tool": "bash",
            "command": command,
            "command_sha256": command_sha,
            "invocation_sha256": invocation_sha,
            "execution_scope": "isolated-instance-container",
        }
    ]
    results = [
        {
            "sequence": 1,
            "tool": "bash",
            "status": "completed",
            "exit_code": 0,
            "command_sha256": command_sha,
            "invocation_sha256": invocation_sha,
            "output_sha256": _digest("lint-output"),
            "visible_to_model": True,
            "provenance": "public-repo-validation-model-visible",
            "execution_scope": "isolated-instance-container",
        }
    ]
    with pytest.raises(
        ExecutionContractError,
        match="distillation_validation_evidence_missing",
    ):
        distillation_execution_evidence_hashes(calls, results)


def test_distillation_evidence_rejects_validate_then_edit() -> None:
    command = "anchor-validate compile"
    command_sha = _digest(command)
    calls = [
        {
            "sequence": 1,
            "tool": "bash",
            "command": command,
            "command_sha256": command_sha,
            "invocation_sha256": _digest("validate"),
            "execution_scope": "isolated-instance-container",
        },
        {
            "sequence": 2,
            "tool": "edit",
            "command": None,
            "command_sha256": None,
            "invocation_sha256": _digest("edit"),
            "execution_scope": "isolated-instance-container",
        },
    ]
    results = [
        {
            "sequence": 1,
            "tool": "bash",
            "status": "completed",
            "exit_code": 0,
            "command_sha256": command_sha,
            "invocation_sha256": _digest("validate"),
            "output_sha256": _digest("validation-output"),
            "visible_to_model": True,
            "provenance": "public-repo-validation-model-visible",
            "execution_scope": "isolated-instance-container",
        },
        {
            "sequence": 2,
            "tool": "edit",
            "status": "completed",
            "exit_code": 0,
            "command_sha256": None,
            "invocation_sha256": _digest("edit"),
            "output_sha256": _digest("edit-output"),
            "visible_to_model": True,
            "provenance": "public-repo-validation-model-visible",
            "execution_scope": "isolated-instance-container",
        },
    ]
    with pytest.raises(
        ExecutionContractError,
        match="distillation_validation_not_terminal",
    ):
        distillation_execution_evidence_hashes(calls, results)


class _FakeTestSpec:
    def __init__(self, image_key: str) -> None:
        self.instance_image_key = image_key


class _FakeOfficialModule:
    TestSpec = _FakeTestSpec

    @staticmethod
    def make_test_spec(
        instance: dict[str, object],
        *,
        namespace: str,
        base_image_tag: str,
        env_image_tag: str,
        instance_image_tag: str,
        arch: str,
    ) -> _FakeTestSpec:
        del base_image_tag, env_image_tag
        local = f"sweb.eval.{arch}.{str(instance['instance_id']).lower()}:{instance_image_tag}"
        return _FakeTestSpec(f"{namespace}/{local}".replace("__", "_1776_"))


@pytest.mark.parametrize(
    ("instance_id", "expected"),
    [
        (
            "django__django-12345",
            "swebench/sweb.eval.x86_64.django_1776_django-12345:latest",
        ),
        (
            "pandas-dev__pandas-7",
            "swebench/sweb.eval.x86_64.pandas-dev_1776_pandas-7:latest",
        ),
    ],
)
def test_image_key_is_taken_from_official_testspec_and_matches_locked_rule(
    instance_id: str, expected: str
) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert (
        resolve_official_instance_image_key(
            {"instance_id": instance_id}, lock, _FakeOfficialModule
        )
        == expected
    )


def test_image_key_rejects_official_api_mismatch() -> None:
    class WrongModule(_FakeOfficialModule):
        @staticmethod
        def make_test_spec(*args: object, **kwargs: object) -> _FakeTestSpec:
            del args, kwargs
            return _FakeTestSpec("attacker/other:latest")

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    with pytest.raises(ExecutionContractError, match="image_rule_mismatch"):
        resolve_official_instance_image_key(
            {"instance_id": "django__django-1"}, lock, WrongModule
        )


def test_builder_policy_allows_only_exact_approved_container_shell_commands() -> None:
    planner = {
        "tool_proposals": [
            {"proposal_id": "read-1", "tool": "read", "input": {"path": "a.py"}},
            {"proposal_id": "edit-1", "tool": "edit", "input": {"path": "a.py"}},
            {"proposal_id": "grep-1", "tool": "grep", "input": {"pattern": "x"}},
            {
                "proposal_id": "test-1",
                "tool": "bash",
                "input": {"command": "anchor-validate test"},
            },
        ]
    }
    decisions = {
        "decisions": [
            {"proposal_id": "read-1", "decision": "APPROVE"},
            {"proposal_id": "edit-1", "decision": "APPROVE"},
            {"proposal_id": "grep-1", "decision": "DENY"},
            {"proposal_id": "test-1", "decision": "APPROVE"},
        ]
    }
    policy = approved_builder_policy(planner, decisions)
    assert policy.allowed_tools == ("bash", "edit", "read")
    assert policy.allowed_commands == ("anchor-validate test",)
    assert policy.opencode_permissions()["bash"] == {
        "*": "deny",
        "anchor-validate test": "allow",
    }

    planner["tool_proposals"].append(
        {
            "proposal_id": "unsafe-1",
            "tool": "bash",
            "input": {"command": "python -m pytest; curl https://example.invalid"},
        }
    )
    decisions["decisions"].append({"proposal_id": "unsafe-1", "decision": "APPROVE"})
    with pytest.raises(ExecutionContractError, match="bash_command_invalid"):
        approved_builder_policy(planner, decisions)


def test_builder_policy_rejects_arbitrary_approved_bash_command() -> None:
    planner = {
        "tool_proposals": [
            {"proposal_id": "edit-1", "tool": "edit", "input": {"path": "a.py"}},
            {
                "proposal_id": "bash-1",
                "tool": "bash",
                "input": {"command": "python -m pytest -q tests/test_a.py"},
            },
        ]
    }
    decisions = {
        "decisions": [
            {"proposal_id": "edit-1", "decision": "APPROVE"},
            {"proposal_id": "bash-1", "decision": "APPROVE"},
        ]
    }
    with pytest.raises(ExecutionContractError, match="bash_command_invalid"):
        approved_builder_policy(planner, decisions)


def test_builder_policy_requires_an_approved_write_capability() -> None:
    planner = {
        "tool_proposals": [
            {"proposal_id": "read-1", "tool": "read", "input": {"path": "a.py"}},
            {
                "proposal_id": "test-1",
                "tool": "bash",
                "input": {"command": "anchor-validate test"},
            },
        ]
    }
    decisions = {
        "decisions": [
            {"proposal_id": "read-1", "decision": "APPROVE"},
            {"proposal_id": "test-1", "decision": "APPROVE"},
        ]
    }
    with pytest.raises(ExecutionContractError, match="coverage_invalid"):
        approved_builder_policy(planner, decisions)


def test_builder_policy_canonicalizes_opencode_edit_permission_family() -> None:
    planner = {
        "tool_proposals": [
            {"proposal_id": "write-1", "tool": "write", "input": {"path": "a.py"}},
            {
                "proposal_id": "test-1",
                "tool": "bash",
                "input": {"command": "anchor-validate test"},
            },
        ]
    }
    decisions = {
        "decisions": [
            {"proposal_id": "write-1", "decision": "APPROVE"},
            {"proposal_id": "test-1", "decision": "APPROVE"},
        ]
    }

    policy = approved_builder_policy(planner, decisions)

    assert policy.allowed_tools == ("bash", "edit")
    assert policy.is_tool_allowed("edit")
    assert policy.is_tool_allowed("write")
    assert policy.is_tool_allowed("apply_patch")


def test_builder_policy_rejects_ambiguous_approved_edit_permission_family() -> None:
    planner = {
        "tool_proposals": [
            {"proposal_id": "edit-1", "tool": "edit", "input": {"path": "a.py"}},
            {"proposal_id": "write-1", "tool": "write", "input": {"path": "b.py"}},
        ]
    }
    decisions = {
        "decisions": [
            {"proposal_id": "edit-1", "decision": "APPROVE"},
            {"proposal_id": "write-1", "decision": "APPROVE"},
        ]
    }

    with pytest.raises(
        ExecutionContractError,
        match="builder_policy_proposal_binding_ambiguous",
    ):
        approved_builder_policy(planner, decisions)


def test_v3_descriptor_separates_visible_iteration_from_hidden_official_eval() -> None:
    descriptor = v3_contract_descriptor()
    assert "bash" in descriptor["model_tools"]
    assert descriptor["model_bash_policy"] == {
        "allow_host_shell": False,
        "command_authorization": (
            "planner_proposal_intersect_tool_policy_approve_exact_string"
        ),
        "execution_scope": "isolated-instance-container",
        "workdir": "/testbed",
        "network": "none-with-supervisor-unix-socket-loopback-bridge",
        "route_scope": "fixed-target-ccswitch-only",
        "public_egress_blocking": "enforced-and-behavior-probed",
        "result_visible_to_model": True,
    }
    assert descriptor["public_repo_validation"]["result_visible_to_model"] is True
    assert descriptor["hidden_official_eval"] == {
        "provenance": "official-swebench-harness-system-private",
        "phase": "post-agent",
        "fresh_container": True,
        "network": "none",
        "result_visible_to_model": False,
    }


def test_wsl_behavior_probe_is_local_only_digest_pinned_and_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "opencode"
    binary.write_bytes(b"bound-binary")
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        observed["args"] = list(args)
        observed["script"] = kwargs.get("input")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "canonical_worktree=1\n"
                "post_agent_hidden_eval=1\n"
                "representative_runtime=1\n"
                "supervisor_unix_route_reachable=1\n"
                "model_network_none=1\n"
                "public_egress_blocked=1\n"
            ).encode("utf-8"),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _wsl_behavior_probes(
        runtime={
            "wsl_distro": "Ubuntu-22.04",
            "native_probe_root": "/var/lib/anchor/swebench-v3",
        },
        image_key="swebench/sweb.eval.x86_64.django_1776_django-1:latest",
        image_digest="sha256:" + "a" * 64,
        linux_binary=binary,
    )
    assert all(result.values())
    args = observed["args"]
    assert args[-4:] == [
        "swebench/sweb.eval.x86_64.django_1776_django-1:latest",
        "sha256:" + "a" * 64,
        "/var/lib/anchor/swebench-v3",
        str(binary),
    ]
    script = bytes(observed["script"]).decode("utf-8")
    assert script.count("--pull=never") == 3
    assert script.count("--network none") == 3
    assert "slirp4netns" not in script
    assert "ccswitch.sock" in script
    assert "169.254.169.254" in script
    assert "huggingface.co" in script
    assert 'IMAGE_REF="${IMAGE_KEY%:*}@${IMAGE_DIGEST}"' in script
    assert "dst=/testbed,rw" in script
    assert "OLD_MODEL_ID" in script
    assert "result_visible_to_model" not in script


def test_wsl_behavior_probe_without_image_is_content_free_fail_closed(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-opencode"
    assert _wsl_behavior_probes(
        runtime={
            "wsl_distro": "Ubuntu-22.04",
            "native_probe_root": "/var/lib/anchor/swebench-v3",
        },
        image_key="",
        image_digest="",
        linux_binary=missing,
    ) == {
        "canonical_worktree": False,
        "post_agent_hidden_eval": False,
        "representative_runtime": False,
        "supervisor_unix_route_reachable": False,
        "model_network_none": False,
        "public_egress_blocked": False,
    }


def test_sealed_validator_result_has_no_raw_output_or_test_identity() -> None:
    empty = _digest("")
    result = SealedValidationResult(
        status="PASS",
        exit_code=0,
        duration_ms=1.25,
        stdout_sha256=empty,
        stderr_sha256=empty,
        report_hash=_digest("private-report"),
    )
    assert set(result.to_dict()) == {
        "status",
        "exit_code",
        "duration_ms",
        "stdout_sha256",
        "stderr_sha256",
        "report_hash",
    }
    with pytest.raises(ExecutionContractError, match="result_shape_invalid"):
        SealedValidationResult.from_mapping({**result.to_dict(), "stdout": "secret"})


def test_wrapper_rejects_arbitrary_command_with_content_free_shape() -> None:
    safe_environment = {
        key: value
        for key in ("SystemRoot", "SYSTEMROOT", "WINDIR")
        if (value := os.environ.get(key))
    }
    completed = subprocess.run(
        [sys.executable, str(WRAPPER), "python", "-c", "print('x')"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        env=safe_environment,
    )
    assert completed.returncode == 64
    assert completed.stderr == ""
    assert set(json.loads(completed.stdout)) == {
        "status",
        "exit_code",
        "duration_ms",
        "stdout_sha256",
        "stderr_sha256",
        "report_hash",
    }


_RECEIPT_KEY = b"system-owned-test-receipt-key-32b"
_PATCH_BYTES = b"diff --git a/a.py b/a.py"


def _gold_evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    bindings = {
        "checkpoint_id": _digest("checkpoint"),
        "task_id_sha256": _digest("task"),
        "revision": 1,
        "instance_id_sha256": _digest("instance"),
        "image_digest": "sha256:" + "a" * 64,
        "base_commit": "b" * 40,
        "patch_sha256": hashlib.sha256(_PATCH_BYTES).hexdigest(),
        "lock_sha256": _digest("lock"),
    }
    official = {
        **bindings,
        "receipt_schema": "anchor.official-eval-receipt.v1",
        "receipt_id": _digest("receipt"),
        "key_id": "test-supervisor-key",
        "issued_by": "anchor-official-eval-supervisor",
        "provenance": "official-swebench-harness-system-private",
        "system_private": True,
        "visible_to_model": False,
        "status": "PASS",
        "exit_code": 0,
        "duration_ms": 10.0,
        "stdout_sha256": _digest("stdout"),
        "stderr_sha256": _digest("stderr"),
        "report_hash": _digest("report"),
    }
    official["receipt_hmac_sha256"] = hmac.new(
        _RECEIPT_KEY,
        json.dumps(
            official,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    edit_input = {"path": "a.py"}
    edit_actual_input_sha256 = _digest("edit-actual-input")
    edit_invocation = _digest(
        json.dumps(
            {
                "input": edit_input,
                "actual_input_sha256": edit_actual_input_sha256,
                "planner_proposal_id": "edit-1",
                "tool": "edit",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    command = "anchor-validate test"
    bash_input = {"command": command}
    bash_actual_input_sha256 = _digest("bash-actual-input")
    bash_invocation = _digest(
        json.dumps(
            {
                "input": bash_input,
                "actual_input_sha256": bash_actual_input_sha256,
                "planner_proposal_id": "validate-1",
                "tool": "bash",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    evidence: dict[str, object] = {
        "workspace_diff": _PATCH_BYTES.decode("utf-8"),
        "tool_calls": [
            {
                "sequence": 1,
                "tool": "edit",
                "input": edit_input,
                "input_provenance": "planner-approved-authorization-scope",
                "actual_input_sha256": edit_actual_input_sha256,
                "invocation_sha256": edit_invocation,
                "planner_proposal_id": "edit-1",
                "tool_policy_decision": "APPROVE",
                "execution_scope": "isolated-instance-container",
            },
            {
                "sequence": 2,
                "tool": "bash",
                "input": bash_input,
                "input_provenance": "planner-approved-authorization-scope",
                "actual_input_sha256": bash_actual_input_sha256,
                "command": command,
                "command_sha256": _digest(command),
                "invocation_sha256": bash_invocation,
                "planner_proposal_id": "validate-1",
                "tool_policy_decision": "APPROVE",
                "execution_scope": "isolated-instance-container",
            },
        ],
        "tool_results": [
            {
                "sequence": 1,
                "tool": "edit",
                "status": "completed",
                "exit_code": 0,
                "actual_input_sha256": edit_actual_input_sha256,
                "invocation_sha256": edit_invocation,
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            },
            {
                "sequence": 2,
                "tool": "bash",
                "status": "completed",
                "exit_code": 0,
                "actual_input_sha256": bash_actual_input_sha256,
                "command_sha256": _digest(command),
                "invocation_sha256": bash_invocation,
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            },
        ],
        "rejected_events": 0,
        "error_codes": [],
        "review_decision": "PASS",
        "security_decision": "PASS",
    }
    return evidence, bindings, official


def _write_private_receipt(
    tmp_path: Path, receipt: dict[str, object], patch: bytes = _PATCH_BYTES
) -> tuple[Path, Path, Path]:
    private_root = tmp_path / "system-private"
    private_root.mkdir()
    receipt_path = private_root / "official-receipt.json"
    patch_path = private_root / "final.patch"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    patch_path.write_bytes(patch)
    return private_root, receipt_path, patch_path


def test_gold_gate_requires_bound_system_private_official_eval(tmp_path: Path) -> None:
    evidence, bindings, receipt = _gold_evidence()
    private_root, receipt_path, patch_path = _write_private_receipt(tmp_path, receipt)
    assert evaluate_v3_gold_gate(
        evidence,
        expected_bindings=bindings,
        trusted_receipt_key=_RECEIPT_KEY,
        system_private_root=private_root,
        official_receipt_path=receipt_path,
        final_patch_path=patch_path,
    ) == (True, ())

    tampered = {
        **receipt,
        "visible_to_model": True,
    }
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    accepted, reasons = evaluate_v3_gold_gate(
        evidence,
        expected_bindings=bindings,
        trusted_receipt_key=_RECEIPT_KEY,
        system_private_root=private_root,
        official_receipt_path=receipt_path,
        final_patch_path=patch_path,
    )
    assert accepted is False
    assert "gold_official_eval_receipt_authentication_failed" in reasons
    assert "gold_official_eval_receipt_missing_or_unbound" in reasons


def test_gold_gate_rejects_unbound_public_validation_result(tmp_path: Path) -> None:
    evidence, bindings, receipt = _gold_evidence()
    private_root, receipt_path, patch_path = _write_private_receipt(tmp_path, receipt)
    evidence["tool_results"] = [
        dict(item, command_sha256=_digest("different"))
        if item.get("sequence") == 2
        else item
        for item in evidence["tool_results"]
    ]
    accepted, reasons = evaluate_v3_gold_gate(
        evidence,
        expected_bindings=bindings,
        trusted_receipt_key=_RECEIPT_KEY,
        system_private_root=private_root,
        official_receipt_path=receipt_path,
        final_patch_path=patch_path,
    )
    assert accepted is False
    assert "gold_public_validation_trace_missing_or_unbound" in reasons


def test_gold_gate_recomputes_final_patch_bytes(tmp_path: Path) -> None:
    evidence, bindings, receipt = _gold_evidence()
    private_root, receipt_path, patch_path = _write_private_receipt(
        tmp_path, receipt, patch=b"different-patch-bytes"
    )
    accepted, reasons = evaluate_v3_gold_gate(
        evidence,
        expected_bindings=bindings,
        trusted_receipt_key=_RECEIPT_KEY,
        system_private_root=private_root,
        official_receipt_path=receipt_path,
        final_patch_path=patch_path,
    )
    assert accepted is False
    assert "gold_patch_bytes_binding_failed" in reasons


def test_gold_gate_rejects_caller_supplied_receipt_even_if_private_is_valid(
    tmp_path: Path,
) -> None:
    evidence, bindings, receipt = _gold_evidence()
    evidence["official_eval_receipt"] = receipt
    private_root, receipt_path, patch_path = _write_private_receipt(tmp_path, receipt)
    accepted, reasons = evaluate_v3_gold_gate(
        evidence,
        expected_bindings=bindings,
        trusted_receipt_key=_RECEIPT_KEY,
        system_private_root=private_root,
        official_receipt_path=receipt_path,
        final_patch_path=patch_path,
    )
    assert accepted is False
    assert "gold_caller_supplied_official_receipt_forbidden" in reasons


def test_real_local_dry_probe_is_false_with_exact_remaining_gates() -> None:
    report = build_execution_attestation(ROOT, LOCK)
    assert report["ready"] is False
    assert report["bindings"]["dataset"]["present_and_bound"] is True
    assert report["bindings"]["canonical_worktree"]["probe_passed"] is False
    assert report["bindings"]["canonical_worktree"]["native_wsl_root"].startswith("/")
    assert report["bindings"]["validator"]["self_test"] is True
    assert report["bindings"]["validator"]["rejects_arbitrary_commands"] is True
    remaining = set(report["remaining_gates"])
    assert {
        "instance_image_probe_missing",
        "opencode_testbed_workdir_contract_missing",
        "opencode_testbed_workdir_support_missing",
        "canonical_testbed_worktree_probe_missing",
        "representative_instance_runtime_smoke_missing",
        "supervisor_unix_route_probe_missing",
        "model_network_none_probe_missing",
        "model_public_egress_block_probe_missing",
        "patched_anchor_sandbox_probe_attestation_missing_or_invalid",
        "official_testspec_final_diff_eval_attestation_missing_or_invalid",
    }.issubset(remaining)
    harness = report["bindings"]["official_harness"]
    checkout = ROOT / "artifacts/tooling/swebench-harness"
    if checkout.is_dir():
        assert harness["clean_checkout"] is True
        assert harness["import_ok"] is True
        assert "official_harness_checkout_missing" not in remaining
        assert "official_harness_install_missing" not in remaining
    else:
        assert {
            "official_harness_checkout_missing",
            "official_harness_install_missing",
            "official_testspec_behavior_probe_missing",
        }.issubset(remaining)
    cache = report["bindings"]["on_demand_image_cache"]
    assert cache["offline_integrity_probe"] is True
    assert cache["ledger_state"] in {"empty", "bound"}
    assert cache["pull_during_model_or_eval"] is False
    representative = report["bindings"]["representative_live_probe"]
    assert representative["present"] is False
    assert representative["automatic_execution"] is False
    assert representative["provider_requests_during_inspection"] == 0
    assert (
        "--confirm-representative-live" in representative["required_generation_command"]
    )
    assert (
        "official_full_bank_image_acquisition_probe_not_implemented"
        not in report["remaining_gates"]
    )
    independent = report["bindings"]["isolation_probes"]["independent_execution_probes"]
    assert independent == {
        "patched_anchor_sandbox_end_to_end": False,
        "official_testspec_on_final_diff": False,
        "manifest_self_claim_sufficient": False,
    }


def test_schema_claim_cannot_replace_recomputed_attestation(tmp_path: Path) -> None:
    fake = tmp_path / "attestation.json"
    fake.write_text(
        json.dumps(
            {
                "schema_version": "anchor.multilang-execution-attestation.v1",
                "tool_contract_version": "anchor.execution-tool-contract.v3",
                "ready": True,
                "remaining_gates": [],
            }
        ),
        encoding="utf-8",
    )
    result = verify_execution_attestation(
        ROOT,
        fake,
        LOCK,
        expected_lock_sha256=sha256_file(LOCK),
    )
    assert result["ready"] is False
    assert result["reason_code"] == "multilang_execution_attestation_stale"
