"""Fail-closed SWE-bench v3 execution contract.

This module deliberately separates collection from Gold admission.  Failed,
blocked, and partial attempts may still be retained by the raw/checkpoint
ledger; :func:`evaluate_v3_gold_gate` is for the later export/partition step.

The official SWE-bench harness and instance images are optional local inputs.
Their absence is reported as a remaining gate and can never be converted into
``ready=true`` by a schema-shaped attestation file.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, redirect_stdout
import hashlib
import hmac
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterator, Mapping, Protocol, Sequence

from .policy import ToolPolicy
from .tool_contract import v3_contract_descriptor


EXECUTION_LOCK_SCHEMA = "anchor.swebench-execution-lock.v1"
EXECUTION_ATTESTATION_SCHEMA = "anchor.multilang-execution-attestation.v1"
EXECUTION_TOOL_CONTRACT_V3 = "anchor.execution-tool-contract.v3"
REPRESENTATIVE_PROBE_ATTESTATION_SCHEMA = (
    "anchor.swebench-representative-probe-attestation.v1"
)
SEALED_VALIDATOR_REQUEST_SCHEMA = "anchor.sealed-validator-request.v1"
SEALED_VALIDATOR_RESULT_SCHEMA = "anchor.sealed-validator-result.v1"

OFFICIAL_HARNESS_REPOSITORY = "https://github.com/SWE-bench/SWE-bench.git"
OFFICIAL_HARNESS_REVISION = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
OFFICIAL_HARNESS_VERSION = "4.1.0"
OFFICIAL_HARNESS_PYTHON_REQUIRES = ">=3.10"
OFFICIAL_TEST_SPEC_MODULE = "swebench.harness.test_spec.test_spec"
OFFICIAL_TEST_SPEC_FACTORY = "make_test_spec"
OFFICIAL_TEST_SPEC_CLASS = "TestSpec"
OFFICIAL_REPO_DIRECTORY = "/testbed"

VALIDATOR_ACTIONS = ("compile", "test", "lint")
# These are supervisor commands, not OpenCode/model commands.  Hidden official
# evaluation status is itself an oracle channel and must never enter a prompt,
# model-visible tool result, session export, or domain-review input.
SEALED_VALIDATOR_COMMANDS = tuple(
    f"anchor-validate {action}" for action in VALIDATOR_ACTIONS
)
DISTILLATION_VALIDATOR_ACTIONS = ("compile", "test")
DISTILLATION_SEALED_VALIDATOR_COMMANDS = tuple(
    f"anchor-validate {action}" for action in DISTILLATION_VALIDATOR_ACTIONS
)
GLOBAL_BUILDER_TOOLS = frozenset(
    {"read", "bash", "edit", "apply_patch", "write", "grep", "glob", "list"}
)
BUILDER_WRITE_TOOLS = frozenset({"edit", "apply_patch", "write"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_SOCKET_PATH = re.compile(r"^/run/anchor-validator/[A-Za-z0-9_.-]+\.sock$")
_IMAGE_KEY = re.compile(r"^[a-z0-9][a-z0-9._/:\-]{1,511}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NATIVE_WSL_ROOT = re.compile(r"^/(?:var/lib|home)/[A-Za-z0-9_./-]+$")

OFFICIAL_EVAL_RECEIPT_SCHEMA = "anchor.official-eval-receipt.v1"
OFFICIAL_EVAL_RECEIPT_ISSUER = "anchor-official-eval-supervisor"
OFFICIAL_EVAL_RECEIPT_PROVENANCE = "official-swebench-harness-system-private"
DISTILLATION_EXECUTION_RECEIPT_SCHEMA = (
    "anchor.swebench-distillation-execution-receipt.v1"
)
DISTILLATION_EXECUTION_STATUS = "SELF_VERIFIED"
DISTILLATION_EXECUTION_EVIDENCE_TIER = "real_sandbox_self_verified"
DISTILLATION_LINEAGE_SCHEMA = "anchor.swebench-distillation-lineage.v1"
DISTILLATION_VALIDATION_STATE_SCHEMA = (
    "anchor.swebench-distillation-validation-state.v1"
)
DISTILLATION_VALIDATOR_RESULT_SCHEMA = "anchor.train-sandbox-validation.v1"
DISTILLATION_VALIDATOR_VERSION = "1.0.1"
DISTILLATION_VALIDATOR_FAMILY = "python-bookworm+node22-bookworm+anchor-validator"
DISTILLATION_RECEIPT_KEY_PATH = "/var/lib/anchor/keys/distillation-execution-hmac-v1"
DISTILLATION_LINEAGE_STAGES = (
    "planner",
    "tool_policy",
    "domain_builder",
    "domain_review",
    "security",
)
DISTILLATION_VALIDATED_CODE_SUFFIXES = frozenset({".js", ".mjs", ".cjs", ".py"})
DISTILLATION_VALIDATED_NON_CODE_SUFFIXES = frozenset(
    {
        "",
        ".css",
        ".csv",
        ".html",
        ".json",
        ".lock",
        ".md",
        ".rst",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
DISTILLATION_EXECUTION_BINDING_KEYS = frozenset(
    {
        "checkpoint_id",
        "config_sha256",
        "execution_lock_sha256",
        "source_bank_manifest_sha256",
        "candidate_task_artifact_sha256",
        "candidate_work_order_artifacts_sha256",
        "task_id_sha256",
        "instance_id_sha256",
        "repo_sha256",
        "base_commit",
        "image_digest",
        "image_id_sha256",
        "final_patch_sha256",
        "tool_transcript_sha256",
        "validation_evidence_sha256",
        "validation_state_sha256",
        "validator_version_sha256",
        "lineage_sha256",
    }
)
DISTILLATION_EXECUTION_UNSIGNED_FIELDS = frozenset(
    set(DISTILLATION_EXECUTION_BINDING_KEYS)
    | {
        "receipt_schema",
        "receipt_id",
        "status",
        "evidence_tier",
        "not_official_swebench_pass",
        "cleanup_success",
        "issued_at",
        "validation_state",
    }
)
OFFICIAL_EVAL_BINDING_KEYS = frozenset(
    {
        "checkpoint_id",
        "task_id_sha256",
        "revision",
        "instance_id_sha256",
        "image_digest",
        "base_commit",
        "patch_sha256",
        "lock_sha256",
    }
)
OFFICIAL_EVAL_RECEIPT_UNSIGNED_FIELDS = frozenset(
    set(OFFICIAL_EVAL_BINDING_KEYS)
    | {
        "receipt_schema",
        "receipt_id",
        "key_id",
        "issued_by",
        "provenance",
        "system_private",
        "visible_to_model",
        "status",
        "exit_code",
        "duration_ms",
        "stdout_sha256",
        "stderr_sha256",
        "report_hash",
    }
)


class ExecutionContractError(ValueError):
    """Stable, content-free v3 contract error."""


@contextmanager
def official_harness_import_scope() -> Iterator[None]:
    """Import pure SWE-bench TestSpec code on a Windows control plane.

    SWE-bench 4.1.0 imports ``prepare_images`` from its package initializer and
    that module imports POSIX-only ``resource`` unconditionally.  Anchor never
    executes image preparation on Windows; acquisition and official evaluation
    remain inside WSL/Podman.  Provide a deliberately unusable module only for
    the duration of the package import so the pure TestSpec/grading code can be
    loaded without modifying the exact clean upstream checkout.
    """

    if os.name != "nt" or "resource" in sys.modules:
        yield
        return
    shim = ModuleType("resource")
    shim.RLIMIT_NOFILE = -1  # type: ignore[attr-defined]

    def _unsupported(*_args: object, **_kwargs: object) -> None:
        raise OSError("resource_is_posix_only")

    shim.setrlimit = _unsupported  # type: ignore[attr-defined]
    sys.modules["resource"] = shim
    try:
        yield
    finally:
        if sys.modules.get("resource") is shim:
            del sys.modules["resource"]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def candidate_artifact_set_sha256(
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    """Hash one exact, sorted candidate-shard inventory.

    Both the live producer and the later Gold consumer use this helper so a
    task whose five work orders ever cross shard boundaries cannot acquire two
    subtly different provenance digests.  Only project-relative POSIX paths
    and exact file SHA-256 values are accepted.
    """

    normalized: list[dict[str, str]] = []
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ExecutionContractError("candidate_artifact_binding_invalid")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ExecutionContractError("candidate_artifact_binding_invalid")
        path = PurePosixPath(relative)
        if (
            not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or not _SHA256.fullmatch(digest)
        ):
            raise ExecutionContractError("candidate_artifact_binding_invalid")
        normalized.append({"path": relative, "sha256": digest})
    normalized.sort(key=lambda item: item["path"])
    if not normalized or len({item["path"] for item in normalized}) != len(normalized):
        raise ExecutionContractError("candidate_artifact_binding_invalid")
    return _sha256_bytes(_canonical(normalized).encode("utf-8"))


def distillation_lineage_sha256(
    *,
    checkpoint_id: str,
    config_sha256: str,
    execution_lock_sha256: str,
    task_id_sha256: str,
    stage_records: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash the exact five-stage artifact lineage used by one final receipt.

    The fixed stage order prevents producers and exporters from assigning
    different meanings to an otherwise identical mapping.  Callers pass only
    checkpoint event metadata; model/sample bodies never enter this helper.
    """

    bindings = (
        checkpoint_id,
        config_sha256,
        execution_lock_sha256,
        task_id_sha256,
    )
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value) for value in bindings
    ):
        raise ExecutionContractError("distillation_lineage_binding_invalid")
    if set(stage_records) != set(DISTILLATION_LINEAGE_STAGES):
        raise ExecutionContractError("distillation_lineage_stages_invalid")
    stages: list[dict[str, Any]] = []
    for stage in DISTILLATION_LINEAGE_STAGES:
        record = stage_records[stage]
        if not isinstance(record, Mapping) or set(record) != {
            "revision",
            "artifact_sha256",
        }:
            raise ExecutionContractError("distillation_lineage_record_invalid")
        revision = record["revision"]
        artifact_sha256 = record["artifact_sha256"]
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or not isinstance(artifact_sha256, str)
            or not _SHA256.fullmatch(artifact_sha256)
        ):
            raise ExecutionContractError("distillation_lineage_record_invalid")
        stages.append(
            {
                "stage": stage,
                "revision": revision,
                "artifact_sha256": artifact_sha256,
            }
        )
    payload = {
        "schema_version": DISTILLATION_LINEAGE_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config_sha256,
        "execution_lock_sha256": execution_lock_sha256,
        "task_id_sha256": task_id_sha256,
        "stages": stages,
    }
    return _sha256_bytes(_canonical(payload).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_official_eval_receipt(
    *,
    bindings: Mapping[str, Any],
    receipt_id: str,
    key_id: str,
    status: str,
    exit_code: int,
    duration_ms: float,
    stdout_sha256: str,
    stderr_sha256: str,
    report_hash: str,
    trusted_receipt_key: bytes,
) -> dict[str, Any]:
    """Create a supervisor-authenticated, system-private evaluation receipt."""

    if set(bindings) != set(OFFICIAL_EVAL_BINDING_KEYS):
        raise ExecutionContractError("official_eval_receipt_bindings_invalid")
    if not isinstance(trusted_receipt_key, bytes) or len(trusted_receipt_key) < 32:
        raise ExecutionContractError("official_eval_receipt_key_invalid")
    unsigned: dict[str, Any] = {
        **dict(bindings),
        "receipt_schema": OFFICIAL_EVAL_RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "key_id": key_id,
        "issued_by": OFFICIAL_EVAL_RECEIPT_ISSUER,
        "provenance": OFFICIAL_EVAL_RECEIPT_PROVENANCE,
        "system_private": True,
        "visible_to_model": False,
        "status": status,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "report_hash": report_hash,
    }
    if (
        set(unsigned) != set(OFFICIAL_EVAL_RECEIPT_UNSIGNED_FIELDS)
        or not _SHA256.fullmatch(receipt_id)
        or not _SHA256.fullmatch(str(bindings["checkpoint_id"]))
        or not _SHA256.fullmatch(str(bindings["task_id_sha256"]))
        or isinstance(bindings["revision"], bool)
        or not isinstance(bindings["revision"], int)
        or bindings["revision"] < 1
        or not key_id
        or status not in {"PASS", "FAIL"}
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or isinstance(duration_ms, bool)
        or not isinstance(duration_ms, (int, float))
        or not math.isfinite(float(duration_ms))
        or float(duration_ms) < 0
        or not isinstance(bindings["image_digest"], str)
        or not _IMAGE_DIGEST.fullmatch(bindings["image_digest"])
        or not isinstance(bindings["base_commit"], str)
        or not _COMMIT.fullmatch(bindings["base_commit"])
        or any(
            not _SHA256.fullmatch(str(unsigned[name]))
            for name in (
                "instance_id_sha256",
                "patch_sha256",
                "lock_sha256",
                "stdout_sha256",
                "stderr_sha256",
                "report_hash",
            )
        )
    ):
        raise ExecutionContractError("official_eval_receipt_fields_invalid")
    signature = hmac.new(
        trusted_receipt_key,
        _canonical(unsigned).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {**unsigned, "receipt_hmac_sha256": signature}


def verify_official_eval_receipt(
    receipt: Mapping[str, Any],
    *,
    trusted_receipt_key: bytes,
    expected_bindings: Mapping[str, Any] | None = None,
    require_pass: bool,
) -> bool:
    """Authenticate a receipt and optionally require a resolved evaluation."""

    if not isinstance(trusted_receipt_key, bytes) or len(trusted_receipt_key) < 32:
        return False
    if set(receipt) != set(OFFICIAL_EVAL_RECEIPT_UNSIGNED_FIELDS) | {
        "receipt_hmac_sha256"
    }:
        return False
    unsigned = {
        name: receipt.get(name)
        for name in sorted(OFFICIAL_EVAL_RECEIPT_UNSIGNED_FIELDS)
    }
    claimed_hmac = receipt.get("receipt_hmac_sha256")
    expected_hmac = hmac.new(
        trusted_receipt_key,
        _canonical(unsigned).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not (
        isinstance(claimed_hmac, str)
        and _SHA256.fullmatch(claimed_hmac)
        and hmac.compare_digest(claimed_hmac, expected_hmac)
    ):
        return False
    if expected_bindings is not None and (
        set(expected_bindings) != set(OFFICIAL_EVAL_BINDING_KEYS)
        or any(
            receipt.get(name) != expected_bindings[name]
            for name in OFFICIAL_EVAL_BINDING_KEYS
        )
    ):
        return False
    allowed_statuses = {"PASS"} if require_pass else {"PASS", "FAIL"}
    return bool(
        receipt.get("receipt_schema") == OFFICIAL_EVAL_RECEIPT_SCHEMA
        and _SHA256.fullmatch(str(receipt.get("receipt_id", "")))
        and isinstance(receipt.get("key_id"), str)
        and bool(receipt.get("key_id"))
        and receipt.get("issued_by") == OFFICIAL_EVAL_RECEIPT_ISSUER
        and receipt.get("provenance") == OFFICIAL_EVAL_RECEIPT_PROVENANCE
        and receipt.get("system_private") is True
        and receipt.get("visible_to_model") is False
        and receipt.get("status") in allowed_statuses
        and _SHA256.fullmatch(str(receipt.get("checkpoint_id", "")))
        and _SHA256.fullmatch(str(receipt.get("task_id_sha256", "")))
        and isinstance(receipt.get("revision"), int)
        and not isinstance(receipt.get("revision"), bool)
        and int(receipt.get("revision", 0)) >= 1
        and isinstance(receipt.get("exit_code"), int)
        and not isinstance(receipt.get("exit_code"), bool)
        and receipt.get("exit_code") == 0
        and isinstance(receipt.get("duration_ms"), (int, float))
        and not isinstance(receipt.get("duration_ms"), bool)
        and math.isfinite(float(receipt.get("duration_ms", -1)))
        and float(receipt.get("duration_ms", -1)) >= 0
        and _IMAGE_DIGEST.fullmatch(str(receipt.get("image_digest", "")))
        and _COMMIT.fullmatch(str(receipt.get("base_commit", "")))
        and all(
            _SHA256.fullmatch(str(receipt.get(name, "")))
            for name in (
                "instance_id_sha256",
                "patch_sha256",
                "lock_sha256",
                "stdout_sha256",
                "stderr_sha256",
                "report_hash",
            )
        )
    )


def distillation_execution_evidence_hashes(
    tool_calls: Sequence[Mapping[str, Any]],
    tool_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a complete ordered tool transcript and real public validation.

    This never promotes public validation to an official SWE-bench verdict.  A
    qualifying result must correlate with one exact ``anchor-validate`` call,
    have a zero exit code, and carry a non-empty output digest from the isolated
    sandbox.
    """

    if not tool_calls or len(tool_calls) != len(tool_results):
        raise ExecutionContractError("distillation_tool_transcript_invalid")
    calls = sorted(
        (dict(item) for item in tool_calls), key=lambda item: item.get("sequence", 0)
    )
    results = sorted(
        (dict(item) for item in tool_results), key=lambda item: item.get("sequence", 0)
    )
    call_sequences = [item.get("sequence") for item in calls]
    result_sequences = [item.get("sequence") for item in results]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in call_sequences
        )
        or len(set(call_sequences)) != len(call_sequences)
        or call_sequences != result_sequences
    ):
        raise ExecutionContractError("distillation_tool_transcript_invalid")
    qualifying: list[dict[str, Any]] = []
    for call, result in zip(calls, results):
        if (
            call.get("tool") != result.get("tool")
            or call.get("invocation_sha256") != result.get("invocation_sha256")
            or not _SHA256.fullmatch(str(call.get("invocation_sha256", "")))
            or call.get("execution_scope") != "isolated-instance-container"
            or result.get("execution_scope") != "isolated-instance-container"
        ):
            raise ExecutionContractError("distillation_tool_transcript_unbound")
        command = call.get("command")
        if command not in DISTILLATION_SEALED_VALIDATOR_COMMANDS:
            continue
        command_sha256 = _sha256_bytes(str(command).encode("utf-8"))
        if (
            call.get("tool") != "bash"
            or call.get("command_sha256") != command_sha256
            or result.get("command_sha256") != command_sha256
            or result.get("status") != "completed"
            or result.get("exit_code") != 0
            or not _SHA256.fullmatch(str(result.get("output_sha256", "")))
            or result.get("provenance") != "public-repo-validation-model-visible"
            or result.get("visible_to_model") is not True
        ):
            raise ExecutionContractError("distillation_validation_evidence_invalid")
        qualifying.append(
            {
                "sequence": call["sequence"],
                "command": command,
                "command_sha256": command_sha256,
                "invocation_sha256": call["invocation_sha256"],
                "output_sha256": result["output_sha256"],
                "exit_code": 0,
            }
        )
    if not qualifying:
        raise ExecutionContractError("distillation_validation_evidence_missing")
    if qualifying[-1]["sequence"] != call_sequences[-1]:
        raise ExecutionContractError("distillation_validation_not_terminal")
    transcript = {"tool_calls": calls, "tool_results": results}
    return {
        "tool_transcript_sha256": _sha256_bytes(_canonical(transcript).encode("utf-8")),
        "validation_evidence_sha256": _sha256_bytes(
            _canonical(qualifying).encode("utf-8")
        ),
        "qualifying_validation_count": len(qualifying),
    }


def distillation_tool_evidence(
    builder_output: Mapping[str, Any],
) -> tuple[str, str]:
    """Return shared producer/consumer hashes for one builder artifact."""

    calls = builder_output.get("tool_calls")
    results = builder_output.get("tool_results")
    if not isinstance(calls, list) or not isinstance(results, list):
        raise ExecutionContractError("distillation_tool_transcript_invalid")
    if not all(isinstance(item, Mapping) for item in calls + results):
        raise ExecutionContractError("distillation_tool_transcript_invalid")
    evidence = distillation_execution_evidence_hashes(calls, results)
    return (
        str(evidence["tool_transcript_sha256"]),
        str(evidence["validation_evidence_sha256"]),
    )


def distillation_validation_state_sha256(
    builder_output: Mapping[str, Any],
    *,
    final_patch_sha256: str,
    validator_version_sha256: str,
) -> str:
    """Bind the audited terminal validator result to the captured final patch."""

    if not _SHA256.fullmatch(final_patch_sha256) or not _SHA256.fullmatch(
        validator_version_sha256
    ):
        raise ExecutionContractError("distillation_validation_state_binding_invalid")
    calls = builder_output.get("tool_calls")
    results = builder_output.get("tool_results")
    if not isinstance(calls, list) or not isinstance(results, list):
        raise ExecutionContractError("distillation_tool_transcript_invalid")
    if not all(isinstance(item, Mapping) for item in calls + results):
        raise ExecutionContractError("distillation_tool_transcript_invalid")
    # This both validates every correlation and proves the qualifying validator
    # is the final tool result, after every possible write/edit/apply_patch.
    distillation_execution_evidence_hashes(calls, results)
    terminal_call = max(calls, key=lambda item: item.get("sequence", 0))
    terminal_result = max(results, key=lambda item: item.get("sequence", 0))
    command_sha256 = terminal_call.get("command_sha256")
    output_sha256 = terminal_result.get("output_sha256")
    if (
        terminal_call.get("command") not in DISTILLATION_SEALED_VALIDATOR_COMMANDS
        or terminal_call.get("sequence") != terminal_result.get("sequence")
        or not _SHA256.fullmatch(str(command_sha256 or ""))
        or not _SHA256.fullmatch(str(output_sha256 or ""))
    ):
        raise ExecutionContractError("distillation_validation_state_binding_invalid")
    state = builder_output.get("validation_state")
    fields = {
        "schema_version",
        "final_patch_sha256",
        "changed_files",
        "changed_files_sha256",
        "terminal_validation_output_sha256",
        "terminal_command_sha256",
        "validator_version_sha256",
        "validator_result",
    }
    if not isinstance(state, Mapping) or set(state) != fields:
        raise ExecutionContractError("distillation_validation_state_invalid")
    changed_files = state.get("changed_files")
    if not isinstance(changed_files, list) or not changed_files:
        raise ExecutionContractError("distillation_validation_state_invalid")
    normalized: list[dict[str, str]] = []
    validated_code_count = 0
    for item in changed_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ExecutionContractError("distillation_validation_state_invalid")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ExecutionContractError("distillation_validation_state_invalid")
        path = PurePosixPath(relative)
        if (
            not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix.casefold()
            not in (
                DISTILLATION_VALIDATED_CODE_SUFFIXES
                | DISTILLATION_VALIDATED_NON_CODE_SUFFIXES
            )
            or not _SHA256.fullmatch(digest)
        ):
            raise ExecutionContractError("distillation_validation_state_invalid")
        normalized.append({"path": relative, "sha256": digest})
        if path.suffix.casefold() in DISTILLATION_VALIDATED_CODE_SUFFIXES:
            validated_code_count += 1
    if normalized != sorted(normalized, key=lambda item: item["path"]) or len(
        {item["path"] for item in normalized}
    ) != len(normalized):
        raise ExecutionContractError("distillation_validation_state_invalid")
    if validated_code_count < 1:
        raise ExecutionContractError("distillation_validation_state_invalid")
    changed_files_sha256 = _sha256_bytes(_canonical(normalized).encode("utf-8"))
    validator_result = state.get("validator_result")
    result_fields = {
        "schema_version",
        "validator_version",
        "mode",
        "success",
        "not_official_swebench_pass",
        "validation_level",
        "changed_paths",
        "changed_paths_sha256",
        "final_state_sha256",
        "validators",
    }
    expected_mode = str(terminal_call.get("command", "")).rsplit(" ", 1)[-1]
    changed_paths = (
        validator_result.get("changed_paths")
        if isinstance(validator_result, Mapping)
        else None
    )
    validators = (
        validator_result.get("validators")
        if isinstance(validator_result, Mapping)
        else None
    )
    if (
        not isinstance(validator_result, Mapping)
        or set(validator_result) != result_fields
        or validator_result.get("schema_version")
        != DISTILLATION_VALIDATOR_RESULT_SCHEMA
        or validator_result.get("validator_version") != DISTILLATION_VALIDATOR_VERSION
        or validator_result.get("mode") != expected_mode
        or validator_result.get("success") is not True
        or validator_result.get("not_official_swebench_pass") is not True
        or validator_result.get("validation_level")
        != ("native_test" if expected_mode == "test" else "syntax")
        or changed_paths != [item["path"] for item in normalized]
        or not isinstance(changed_paths, list)
        or changed_paths != sorted(set(changed_paths))
        or validator_result.get("changed_paths_sha256")
        != _sha256_bytes(_canonical(changed_paths).encode("utf-8"))
        or not _SHA256.fullmatch(str(validator_result.get("final_state_sha256", "")))
        or not isinstance(validators, list)
        or not validators
        or validators != sorted(set(validators))
        or not all(isinstance(item, str) and item for item in validators)
    ):
        raise ExecutionContractError("distillation_validation_state_invalid")
    if (
        state.get("schema_version") != DISTILLATION_VALIDATION_STATE_SCHEMA
        or state.get("final_patch_sha256") != final_patch_sha256
        or state.get("changed_files_sha256") != changed_files_sha256
        or state.get("terminal_validation_output_sha256") != output_sha256
        or state.get("terminal_command_sha256") != command_sha256
        or state.get("validator_version_sha256") != validator_version_sha256
    ):
        raise ExecutionContractError("distillation_validation_state_unbound")
    return _sha256_bytes(_canonical(dict(state)).encode("utf-8"))


def sign_distillation_execution_receipt(
    *,
    bindings: Mapping[str, Any],
    validation_state: Mapping[str, Any],
    receipt_id: str,
    issued_at: str,
    trusted_receipt_key: bytes,
) -> dict[str, Any]:
    """Sign self-verified train execution evidence after successful cleanup."""

    if set(bindings) != set(DISTILLATION_EXECUTION_BINDING_KEYS):
        raise ExecutionContractError("distillation_execution_bindings_invalid")
    if not isinstance(trusted_receipt_key, bytes) or len(trusted_receipt_key) < 32:
        raise ExecutionContractError("distillation_execution_key_invalid")
    unsigned: dict[str, Any] = {
        **dict(bindings),
        "receipt_schema": DISTILLATION_EXECUTION_RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "status": DISTILLATION_EXECUTION_STATUS,
        "evidence_tier": DISTILLATION_EXECUTION_EVIDENCE_TIER,
        "not_official_swebench_pass": True,
        "cleanup_success": True,
        "issued_at": issued_at,
        "validation_state": dict(validation_state),
    }
    digest_fields = DISTILLATION_EXECUTION_BINDING_KEYS - {
        "base_commit",
        "image_digest",
    }
    if (
        set(unsigned) != set(DISTILLATION_EXECUTION_UNSIGNED_FIELDS)
        or not _SHA256.fullmatch(receipt_id)
        or not isinstance(issued_at, str)
        or not issued_at.endswith("Z")
        or len(issued_at) < 20
        or not _COMMIT.fullmatch(str(bindings.get("base_commit", "")))
        or not _IMAGE_DIGEST.fullmatch(str(bindings.get("image_digest", "")))
        or any(
            not _SHA256.fullmatch(str(bindings.get(name, ""))) for name in digest_fields
        )
        or _sha256_bytes(_canonical(dict(validation_state)).encode("utf-8"))
        != bindings.get("validation_state_sha256")
    ):
        raise ExecutionContractError("distillation_execution_fields_invalid")
    signature = hmac.new(
        trusted_receipt_key,
        _canonical(unsigned).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {**unsigned, "receipt_hmac_sha256": signature}


def verify_distillation_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    trusted_receipt_key: bytes,
    expected_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Authenticate a self-verified train receipt without claiming benchmark PASS."""

    if not isinstance(trusted_receipt_key, bytes) or len(trusted_receipt_key) < 32:
        return False
    if set(receipt) != set(DISTILLATION_EXECUTION_UNSIGNED_FIELDS) | {
        "receipt_hmac_sha256"
    }:
        return False
    unsigned = {
        name: receipt.get(name)
        for name in sorted(DISTILLATION_EXECUTION_UNSIGNED_FIELDS)
    }
    claimed = receipt.get("receipt_hmac_sha256")
    expected = hmac.new(
        trusted_receipt_key,
        _canonical(unsigned).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not (
        isinstance(claimed, str)
        and _SHA256.fullmatch(claimed)
        and hmac.compare_digest(claimed, expected)
    ):
        return False
    if expected_bindings is not None and (
        set(expected_bindings) != set(DISTILLATION_EXECUTION_BINDING_KEYS)
        or any(
            receipt.get(name) != expected_bindings[name]
            for name in DISTILLATION_EXECUTION_BINDING_KEYS
        )
    ):
        return False
    digest_fields = DISTILLATION_EXECUTION_BINDING_KEYS - {
        "base_commit",
        "image_digest",
    }
    validation_state = receipt.get("validation_state")
    return bool(
        receipt.get("receipt_schema") == DISTILLATION_EXECUTION_RECEIPT_SCHEMA
        and receipt.get("status") == DISTILLATION_EXECUTION_STATUS
        and receipt.get("evidence_tier") == DISTILLATION_EXECUTION_EVIDENCE_TIER
        and receipt.get("not_official_swebench_pass") is True
        and receipt.get("cleanup_success") is True
        and _SHA256.fullmatch(str(receipt.get("receipt_id", "")))
        and isinstance(receipt.get("issued_at"), str)
        and str(receipt.get("issued_at")).endswith("Z")
        and _COMMIT.fullmatch(str(receipt.get("base_commit", "")))
        and _IMAGE_DIGEST.fullmatch(str(receipt.get("image_digest", "")))
        and all(_SHA256.fullmatch(str(receipt.get(name, ""))) for name in digest_fields)
        and isinstance(validation_state, Mapping)
        and _sha256_bytes(_canonical(dict(validation_state)).encode("utf-8"))
        == receipt.get("validation_state_sha256")
    )


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionContractError(code)
    return value


def _exact(value: Mapping[str, Any], fields: set[str], code: str) -> None:
    if set(value) != fields:
        raise ExecutionContractError(code)


def _project_path(root: Path, value: object, code: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExecutionContractError(code)
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ExecutionContractError(code) from exc
    return candidate


@dataclass(frozen=True)
class SealedValidationResult:
    status: str
    exit_code: int
    duration_ms: float
    stdout_sha256: str
    stderr_sha256: str
    report_hash: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SealedValidationResult":
        _exact(
            value,
            {
                "status",
                "exit_code",
                "duration_ms",
                "stdout_sha256",
                "stderr_sha256",
                "report_hash",
            },
            "sealed_validator_result_shape_invalid",
        )
        status = value.get("status")
        exit_code = value.get("exit_code")
        duration_ms = value.get("duration_ms")
        stdout_sha256 = value.get("stdout_sha256")
        stderr_sha256 = value.get("stderr_sha256")
        report_hash = value.get("report_hash")
        if (
            status not in {"PASS", "FAIL", "TIMEOUT", "OOM", "ERROR"}
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or duration_ms < 0
            or not all(
                isinstance(item, str) and _SHA256.fullmatch(item)
                for item in (stdout_sha256, stderr_sha256, report_hash)
            )
        ):
            raise ExecutionContractError("sealed_validator_result_invalid")
        return cls(
            status=str(status),
            exit_code=exit_code,
            duration_ms=float(duration_ms),
            stdout_sha256=str(stdout_sha256),
            stderr_sha256=str(stderr_sha256),
            report_hash=str(report_hash),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 3),
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "report_hash": self.report_hash,
        }


def _content_free_result(
    status: str, exit_code: int, code: str
) -> SealedValidationResult:
    digest = _sha256_bytes(code.encode("ascii"))
    return SealedValidationResult(
        status=status,
        exit_code=exit_code,
        duration_ms=0.0,
        stdout_sha256=_sha256_bytes(b""),
        stderr_sha256=_sha256_bytes(b""),
        report_hash=digest,
    )


class ValidatorTransport(Protocol):
    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class UnixSocketValidatorTransport:
    """One-shot local transport; the endpoint is fixed outside the workspace."""

    def __init__(self, socket_path: str, *, timeout_seconds: float = 900.0) -> None:
        if not _SOCKET_PATH.fullmatch(socket_path):
            raise ExecutionContractError("sealed_validator_socket_invalid")
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        encoded = (_canonical(dict(payload)) + "\n").encode("utf-8")
        if len(encoded) > 8192:
            raise ExecutionContractError("sealed_validator_request_too_large")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout_seconds)
            client.connect(self.socket_path)
            client.sendall(encoded)
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                size += len(chunk)
                if size > 65536:
                    raise ExecutionContractError("sealed_validator_response_too_large")
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        try:
            value = json.loads(b"".join(chunks).split(b"\n", 1)[0])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ExecutionContractError("sealed_validator_response_invalid") from exc
        return _mapping(value, "sealed_validator_response_invalid")


def invoke_sealed_validator(
    action: str,
    *,
    instance_id: str,
    token: str,
    transport: ValidatorTransport,
) -> SealedValidationResult:
    if action not in VALIDATOR_ACTIONS:
        raise ExecutionContractError("sealed_validator_action_invalid")
    if not _INSTANCE_ID.fullmatch(instance_id):
        raise ExecutionContractError("sealed_validator_instance_id_invalid")
    if not _TOKEN.fullmatch(token):
        raise ExecutionContractError("sealed_validator_token_invalid")
    response = transport.request(
        {
            "schema_version": SEALED_VALIDATOR_REQUEST_SCHEMA,
            "action": action,
            "instance_id": instance_id,
            "token": token,
        }
    )
    result = SealedValidationResult.from_mapping(response)
    return result


def validator_cli(argv: Sequence[str] | None = None) -> int:
    """CLI exposed as exact ``anchor-validate compile|test|lint`` commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--self-test"]:
        print(_canonical(_content_free_result("PASS", 0, "self_test").to_dict()))
        return 0
    if len(arguments) != 1 or arguments[0] not in VALIDATOR_ACTIONS:
        print(_canonical(_content_free_result("ERROR", 64, "invalid_action").to_dict()))
        return 64
    action = arguments[0]
    instance_id = os.environ.get("ANCHOR_VALIDATOR_INSTANCE_ID", "")
    token = os.environ.get("ANCHOR_VALIDATOR_TOKEN", "")
    socket_path = os.environ.get("ANCHOR_VALIDATOR_SOCKET", "")
    try:
        result = invoke_sealed_validator(
            action,
            instance_id=instance_id,
            token=token,
            transport=UnixSocketValidatorTransport(socket_path),
        )
    except (ExecutionContractError, OSError, TimeoutError):
        result = _content_free_result("ERROR", 78, "validator_unavailable")
    print(_canonical(result.to_dict()))
    return (
        0
        if result.status == "PASS" and result.exit_code == 0
        else result.exit_code or 1
    )


def approved_builder_policy(
    planner_output: Mapping[str, Any],
    tool_policy_output: Mapping[str, Any],
    *,
    timeout_seconds: float = 900.0,
) -> ToolPolicy:
    """Return global allowlist intersect planner proposals intersect approvals."""

    proposals = planner_output.get("tool_proposals")
    decisions = tool_policy_output.get("decisions")
    if (
        not isinstance(proposals, list)
        or not proposals
        or not isinstance(decisions, list)
    ):
        raise ExecutionContractError("builder_policy_input_invalid")
    decision_map: dict[str, str] = {}
    for raw in decisions:
        item = _mapping(raw, "builder_policy_decision_invalid")
        proposal_id = item.get("proposal_id")
        decision = item.get("decision")
        if (
            not isinstance(proposal_id, str)
            or not proposal_id
            or proposal_id != proposal_id.strip()
            or proposal_id in decision_map
            or decision not in {"APPROVE", "DENY"}
        ):
            raise ExecutionContractError("builder_policy_decision_invalid")
        decision_map[proposal_id] = str(decision)

    proposal_ids: set[str] = set()
    approved_bindings: set[tuple[str, str | None]] = set()
    allowed_tools: set[str] = set()
    allowed_commands: set[str] = set()
    for raw in proposals:
        proposal = _mapping(raw, "builder_policy_proposal_invalid")
        proposal_id = proposal.get("proposal_id")
        tool = proposal.get("tool")
        tool_input = proposal.get("input")
        if (
            not isinstance(proposal_id, str)
            or not proposal_id
            or proposal_id != proposal_id.strip()
            or proposal_id in proposal_ids
            or not isinstance(tool, str)
            or not isinstance(tool_input, Mapping)
        ):
            raise ExecutionContractError("builder_policy_proposal_invalid")
        proposal_ids.add(proposal_id)
        if decision_map.get(proposal_id) != "APPROVE":
            continue
        if tool not in GLOBAL_BUILDER_TOOLS:
            raise ExecutionContractError("builder_policy_tool_outside_global_allowlist")
        canonical_tool = ToolPolicy.normalize_tool(tool)
        binding = (canonical_tool, None)
        if tool == "bash":
            if set(tool_input) != {"command"}:
                raise ExecutionContractError("builder_policy_bash_input_invalid")
            command = tool_input.get("command")
            if not isinstance(command, str):
                raise ExecutionContractError("builder_policy_bash_command_invalid")
            normalized = ToolPolicy.normalize_command(command)
            if (
                not normalized
                or normalized != command
                or command not in SEALED_VALIDATOR_COMMANDS
            ):
                raise ExecutionContractError("builder_policy_bash_command_invalid")
            binding = (canonical_tool, command)
            allowed_commands.add(command)
        if binding in approved_bindings:
            raise ExecutionContractError("builder_policy_proposal_binding_ambiguous")
        approved_bindings.add(binding)
        allowed_tools.add(canonical_tool)
    if (
        set(decision_map) != proposal_ids
        or not allowed_tools
        or not allowed_tools.intersection(BUILDER_WRITE_TOOLS)
    ):
        raise ExecutionContractError("builder_policy_coverage_invalid")
    return ToolPolicy(
        allowed_tools=tuple(sorted(allowed_tools)),
        allowed_commands=tuple(sorted(allowed_commands)),
        timeout_seconds=timeout_seconds,
    )


def evaluate_v3_gold_gate(
    evidence: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
    trusted_receipt_key: bytes,
    system_private_root: Path,
    official_receipt_path: Path,
    final_patch_path: Path,
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate export eligibility using a system-authenticated eval receipt.

    The receipt and finalized patch are read from a supervisor-owned private
    directory, never from the caller evidence. ``trusted_receipt_key`` is also
    supervisor-owned and never enters a model transcript.
    """

    reasons: list[str] = []
    workspace_diff = evidence.get("workspace_diff")
    if not isinstance(workspace_diff, str) or not workspace_diff.strip():
        reasons.append("gold_diff_missing")
    elif _sha256_bytes(workspace_diff.encode("utf-8")) != expected_bindings.get(
        "patch_sha256"
    ):
        # The builder contract stores an exact UTF-8 git binary diff.  Newline
        # normalization or substituting another task's diff changes the hash.
        reasons.append("gold_evidence_diff_binding_failed")
    calls = evidence.get("tool_calls")
    results = evidence.get("tool_results")
    if not isinstance(calls, list) or not isinstance(results, list):
        reasons.append("gold_tool_trace_missing")
        calls = []
        results = []
    call_sequences = [
        item.get("sequence") for item in calls if isinstance(item, Mapping)
    ]
    result_sequences = [
        item.get("sequence") for item in results if isinstance(item, Mapping)
    ]
    if (
        any(
            isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1
            for sequence in call_sequences + result_sequences
        )
        or len(set(call_sequences)) != len(call_sequences)
        or len(set(result_sequences)) != len(result_sequences)
        or set(call_sequences) != set(result_sequences)
    ):
        reasons.append("gold_tool_trace_sequence_invalid")
    calls_by_sequence = {
        item.get("sequence"): item for item in calls if isinstance(item, Mapping)
    }
    results_by_sequence = {
        item.get("sequence"): item for item in results if isinstance(item, Mapping)
    }
    trace_binding_invalid = False
    for sequence, call in calls_by_sequence.items():
        result = results_by_sequence.get(sequence)
        if (
            not isinstance(result, Mapping)
            or not isinstance(call.get("invocation_sha256"), str)
            or _SHA256.fullmatch(str(call.get("invocation_sha256"))) is None
            or result.get("invocation_sha256") != call.get("invocation_sha256")
            or result.get("tool") != call.get("tool")
            or not isinstance(call.get("actual_input_sha256"), str)
            or _SHA256.fullmatch(str(call.get("actual_input_sha256"))) is None
            or result.get("actual_input_sha256") != call.get("actual_input_sha256")
            or call.get("input_provenance") != "planner-approved-authorization-scope"
            or not isinstance(call.get("planner_proposal_id"), str)
            or not call.get("planner_proposal_id")
            or call.get("tool_policy_decision") != "APPROVE"
            or call.get("execution_scope") != "isolated-instance-container"
            or result.get("execution_scope") != "isolated-instance-container"
            or result.get("visible_to_model") is not True
        ):
            trace_binding_invalid = True
            break
    if trace_binding_invalid:
        reasons.append("gold_tool_trace_binding_invalid")
    edit_sequences = {
        item.get("sequence")
        for item in calls
        if isinstance(item, Mapping)
        and item.get("tool") in {"edit", "apply_patch", "write"}
    }
    completed_sequences = {
        item.get("sequence")
        for item in results
        if isinstance(item, Mapping)
        and isinstance(item.get("sequence"), int)
        and not isinstance(item.get("sequence"), bool)
        and item.get("status") == "completed"
        and item.get("exit_code") in {None, 0}
    }
    if not edit_sequences.intersection(completed_sequences):
        reasons.append("gold_edit_not_observed")
    public_validation_calls: dict[object, str] = {}
    for item in calls:
        if not isinstance(item, Mapping) or item.get("tool") != "bash":
            continue
        tool_input = item.get("input")
        command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
        if (
            isinstance(command, str)
            and ToolPolicy.normalize_command(command) == command
            and command in SEALED_VALIDATOR_COMMANDS
            and isinstance(item.get("planner_proposal_id"), str)
            and bool(item.get("planner_proposal_id"))
            and item.get("tool_policy_decision") == "APPROVE"
            and item.get("execution_scope") == "isolated-instance-container"
        ):
            public_validation_calls[item.get("sequence")] = _sha256_bytes(
                command.encode("utf-8")
            )
    public_validation_results = {
        item.get("sequence"): item.get("command_sha256")
        for item in results
        if isinstance(item, Mapping)
        and item.get("status") == "completed"
        and item.get("exit_code") == 0
        and item.get("visible_to_model") is True
        and item.get("provenance") == "public-repo-validation-model-visible"
        and item.get("execution_scope") == "isolated-instance-container"
        and isinstance(item.get("command_sha256"), str)
    }
    if not any(
        public_validation_results.get(sequence) == command_sha256
        for sequence, command_sha256 in public_validation_calls.items()
    ):
        reasons.append("gold_public_validation_trace_missing_or_unbound")
    required_binding_keys = set(OFFICIAL_EVAL_BINDING_KEYS)
    if set(expected_bindings) != required_binding_keys:
        raise ExecutionContractError("gold_expected_bindings_invalid")
    if not isinstance(trusted_receipt_key, bytes) or len(trusted_receipt_key) < 32:
        raise ExecutionContractError("gold_trusted_receipt_context_invalid")
    private_root = system_private_root.resolve()
    try:
        receipt_file = official_receipt_path.resolve(strict=True)
        patch_file = final_patch_path.resolve(strict=True)
        receipt_file.relative_to(private_root)
        patch_file.relative_to(private_root)
        if (
            official_receipt_path.is_symlink()
            or final_patch_path.is_symlink()
            or not receipt_file.is_file()
            or not patch_file.is_file()
        ):
            raise OSError("private_file_invalid")
        receipt_raw = json.loads(receipt_file.read_text(encoding="utf-8"))
        official = _mapping(receipt_raw, "gold_official_eval_receipt_invalid")
        final_patch_bytes = patch_file.read_bytes()
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ExecutionContractError,
    ):
        official = None
        final_patch_bytes = b""
        reasons.append("gold_system_private_artifact_unreadable")
    if "official_eval_receipt" in evidence or "official_eval_proof" in evidence:
        reasons.append("gold_caller_supplied_official_receipt_forbidden")
    patch_bytes_bound = bool(final_patch_bytes) and (
        _sha256_bytes(final_patch_bytes) == expected_bindings["patch_sha256"]
    )
    if not patch_bytes_bound:
        reasons.append("gold_patch_bytes_binding_failed")
    official_ok = False
    receipt_authenticated = False
    if isinstance(official, Mapping):
        unsigned_fields = required_binding_keys | {
            "receipt_schema",
            "receipt_id",
            "key_id",
            "issued_by",
            "provenance",
            "system_private",
            "visible_to_model",
            "status",
            "exit_code",
            "duration_ms",
            "stdout_sha256",
            "stderr_sha256",
            "report_hash",
        }
        unsigned = {key: official.get(key) for key in sorted(unsigned_fields)}
        claimed_hmac = official.get("receipt_hmac_sha256")
        expected_hmac = hmac.new(
            trusted_receipt_key,
            _canonical(unsigned).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        receipt_authenticated = (
            isinstance(claimed_hmac, str)
            and _SHA256.fullmatch(claimed_hmac) is not None
            and hmac.compare_digest(claimed_hmac, expected_hmac)
        )
        official_ok = (
            set(official) == unsigned_fields | {"receipt_hmac_sha256"}
            and official.get("receipt_schema") == "anchor.official-eval-receipt.v1"
            and _SHA256.fullmatch(str(official.get("receipt_id", "")))
            and isinstance(official.get("key_id"), str)
            and bool(official.get("key_id"))
            and official.get("issued_by") == "anchor-official-eval-supervisor"
            and receipt_authenticated
            and patch_bytes_bound
            and official.get("provenance") == "official-swebench-harness-system-private"
            and official.get("system_private") is True
            and official.get("visible_to_model") is False
            and official.get("status") == "PASS"
            and official.get("exit_code") == 0
            and all(
                official.get(key) == expected_bindings[key]
                for key in required_binding_keys
            )
            and _SHA256.fullmatch(str(official.get("checkpoint_id", "")))
            and _SHA256.fullmatch(str(official.get("task_id_sha256", "")))
            and isinstance(official.get("revision"), int)
            and not isinstance(official.get("revision"), bool)
            and int(official.get("revision", 0)) >= 1
            and _IMAGE_DIGEST.fullmatch(str(official.get("image_digest", "")))
            and _COMMIT.fullmatch(str(official.get("base_commit", "")))
            and all(
                _SHA256.fullmatch(str(official.get(key, "")))
                for key in (
                    "instance_id_sha256",
                    "patch_sha256",
                    "lock_sha256",
                    "stdout_sha256",
                    "stderr_sha256",
                    "report_hash",
                )
            )
            and isinstance(official.get("duration_ms"), (int, float))
            and not isinstance(official.get("duration_ms"), bool)
            and float(official.get("duration_ms", -1)) >= 0
        )
    if isinstance(official, Mapping) and not receipt_authenticated:
        reasons.append("gold_official_eval_receipt_authentication_failed")
    if not official_ok:
        reasons.append("gold_official_eval_receipt_missing_or_unbound")
    if evidence.get("rejected_events") not in {0, None}:
        reasons.append("gold_rejected_event_present")
    error_codes = evidence.get("error_codes")
    if error_codes not in (None, (), []):
        reasons.append("gold_error_present")
    if evidence.get("review_decision") != "PASS":
        reasons.append("gold_review_not_passed")
    if evidence.get("security_decision") != "PASS":
        reasons.append("gold_security_not_passed")
    if evidence.get("oracle_assisted") is True and not (
        evidence.get("source_split") == "train"
        and evidence.get("oracle_feedback_count") in {0, 1}
        and evidence.get("adaptive_revision_after_oracle") is False
        and evidence.get("not_for_pass_at_1_claim") is True
    ):
        reasons.append("gold_oracle_assisted_provenance_invalid")
    return not reasons, tuple(reasons)


def resolve_official_instance_image_key(
    instance: Mapping[str, Any], lock: Mapping[str, Any], module: object
) -> str:
    """Use the official TestSpec property, then assert its locked tag invariant."""

    harness = _mapping(lock.get("official_harness"), "execution_lock_harness_invalid")
    image = _mapping(lock.get("image_policy"), "execution_lock_image_policy_invalid")
    factory = getattr(module, str(harness["factory"]), None)
    spec_class = getattr(module, str(harness["class"]), None)
    if not callable(factory) or not isinstance(spec_class, type):
        raise ExecutionContractError("official_testspec_api_missing")
    instance_id = instance.get("instance_id")
    if not isinstance(instance_id, str) or not _INSTANCE_ID.fullmatch(instance_id):
        raise ExecutionContractError("official_testspec_instance_id_invalid")
    spec = factory(
        dict(instance),
        namespace=image["namespace"],
        base_image_tag=image["base_image_tag"],
        env_image_tag=image["env_image_tag"],
        instance_image_tag=image["instance_image_tag"],
        arch=image["arch"],
    )
    if not isinstance(spec, spec_class):
        raise ExecutionContractError("official_testspec_factory_type_mismatch")
    key = getattr(spec, "instance_image_key", None)
    local_key = (
        f"sweb.eval.{image['arch']}.{instance_id.lower()}:{image['instance_image_tag']}"
    )
    expected = f"{image['namespace']}/{local_key}".replace(
        "__", str(image["remote_double_underscore_escape"])
    )
    if key != expected or not isinstance(key, str) or not _IMAGE_KEY.fullmatch(key):
        raise ExecutionContractError("official_testspec_image_rule_mismatch")
    return key


def load_execution_lock(
    root: Path, path: Path, *, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ExecutionContractError("execution_lock_missing")
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ExecutionContractError("execution_lock_sha256_mismatch")
    value = _mapping(
        json.loads(path.read_text(encoding="utf-8")), "execution_lock_invalid"
    )
    _exact(
        value,
        {
            "schema_version",
            "dataset",
            "official_harness",
            "image_policy",
            "opencode",
            "ccswitch",
            "validator",
            "sealed_probe_instance",
            "runtime",
        },
        "execution_lock_shape_invalid",
    )
    if value.get("schema_version") != EXECUTION_LOCK_SCHEMA:
        raise ExecutionContractError("execution_lock_schema_mismatch")
    dataset = _mapping(value.get("dataset"), "execution_lock_dataset_invalid")
    harness = _mapping(value.get("official_harness"), "execution_lock_harness_invalid")
    validator = _mapping(value.get("validator"), "execution_lock_validator_invalid")
    image = _mapping(value.get("image_policy"), "execution_lock_image_policy_invalid")
    runtime = _mapping(value.get("runtime"), "execution_lock_runtime_invalid")
    opencode = _mapping(value.get("opencode"), "execution_lock_opencode_invalid")
    ccswitch = _mapping(value.get("ccswitch"), "execution_lock_ccswitch_invalid")
    _exact(
        dataset,
        {
            "dataset_id",
            "revision",
            "split",
            "parquet",
            "parquet_sha256",
            "parquet_bytes",
            "row_count",
        },
        "execution_lock_dataset_shape_invalid",
    )
    _exact(
        harness,
        {
            "repository",
            "revision",
            "version",
            "python_requires",
            "checkout",
            "module",
            "factory",
            "class",
            "repo_directory",
        },
        "execution_lock_harness_shape_invalid",
    )
    _exact(
        image,
        {
            "arch",
            "namespace",
            "base_image_tag",
            "env_image_tag",
            "instance_image_tag",
            "remote_double_underscore_escape",
        },
        "execution_lock_image_policy_shape_invalid",
    )
    _exact(
        opencode,
        {
            "bundle_manifest",
            "patch_manifest",
            "required_tool_contract_version",
        },
        "execution_lock_opencode_shape_invalid",
    )
    _exact(
        ccswitch,
        {"route_manifest"},
        "execution_lock_ccswitch_shape_invalid",
    )
    _exact(
        validator,
        {
            "wrapper",
            "wrapper_sha256",
            "adapter",
            "adapter_sha256",
            "runtime_adapter",
            "runtime_adapter_sha256",
            "coordinator",
            "coordinator_sha256",
            "route_diagnostics",
            "route_diagnostics_sha256",
            "representative_probe_runner",
            "representative_probe_runner_sha256",
            "representative_probe_builder",
            "representative_probe_builder_sha256",
            "tool_contract",
            "tool_contract_sha256",
            "policy",
            "policy_sha256",
            "execution_config",
            "execution_config_sha256",
            "execution_models",
            "execution_models_sha256",
            "execution_runner",
            "execution_runner_sha256",
            "execution_trace",
            "execution_trace_sha256",
            "session_export",
            "session_export_sha256",
            "allowed_actions",
            "result_schema",
            "distillation_validator",
            "distillation_validator_sha256",
            "distillation_validator_version",
            "distillation_validator_family",
            "distillation_containerfile",
            "distillation_containerfile_sha256",
            "distillation_image_reference",
            "distillation_image_id_sha256",
            "distillation_result_schema",
            "distillation_allowed_actions",
        },
        "execution_lock_validator_shape_invalid",
    )
    _exact(
        runtime,
        {
            "wsl_distro",
            "podman_mode",
            "native_probe_root",
            "receipt_key_path",
            "distillation_receipt_key_path",
            "representative_probe_attestation",
            "canonical_repo_directory",
            "model_workdir",
            "single_rw_mount",
            "validator_network",
            "image_pull_policy",
            "model_shell_scope",
            "model_network_mode",
            "model_route_scope",
            "model_public_egress_blocking",
            "hidden_eval_phase",
            "hidden_eval_network",
        },
        "execution_lock_runtime_shape_invalid",
    )
    if (
        dataset.get("dataset_id") != "SWE-bench/SWE-bench"
        or dataset.get("revision") != "7074ef12ea2a6f70a228943c1336553333c22786"
        or dataset.get("split") != "train"
        or dataset.get("parquet_sha256")
        != "0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69"
        or dataset.get("row_count") != 19008
        or dataset.get("parquet_bytes") != 106492326
        or harness.get("repository") != OFFICIAL_HARNESS_REPOSITORY
        or harness.get("revision") != OFFICIAL_HARNESS_REVISION
        or harness.get("version") != OFFICIAL_HARNESS_VERSION
        or harness.get("python_requires") != OFFICIAL_HARNESS_PYTHON_REQUIRES
        or harness.get("module") != OFFICIAL_TEST_SPEC_MODULE
        or harness.get("factory") != OFFICIAL_TEST_SPEC_FACTORY
        or harness.get("class") != OFFICIAL_TEST_SPEC_CLASS
        or harness.get("repo_directory") != OFFICIAL_REPO_DIRECTORY
        or validator.get("allowed_actions") != list(VALIDATOR_ACTIONS)
        or validator.get("result_schema") != SEALED_VALIDATOR_RESULT_SCHEMA
        or validator.get("distillation_validator_version")
        != DISTILLATION_VALIDATOR_VERSION
        or validator.get("distillation_validator_family")
        != DISTILLATION_VALIDATOR_FAMILY
        or validator.get("distillation_result_schema")
        != DISTILLATION_VALIDATOR_RESULT_SCHEMA
        or validator.get("distillation_allowed_actions")
        != list(DISTILLATION_VALIDATOR_ACTIONS)
        or not isinstance(validator.get("distillation_image_reference"), str)
        or not re.fullmatch(
            r"localhost/anchor-train-sandbox@sha256:[0-9a-f]{64}",
            str(validator.get("distillation_image_reference")),
        )
        or not _SHA256.fullmatch(str(validator.get("distillation_image_id_sha256", "")))
        or opencode.get("required_tool_contract_version") != EXECUTION_TOOL_CONTRACT_V3
        or image
        != {
            "arch": "x86_64",
            "namespace": "swebench",
            "base_image_tag": "latest",
            "env_image_tag": "latest",
            "instance_image_tag": "latest",
            "remote_double_underscore_escape": "_1776_",
        }
        or runtime.get("wsl_distro") != "Ubuntu-22.04"
        or runtime.get("podman_mode") != "wsl-rootful"
        or runtime.get("receipt_key_path")
        != "/var/lib/anchor/keys/official-eval-hmac-v1"
        or runtime.get("distillation_receipt_key_path") != DISTILLATION_RECEIPT_KEY_PATH
        or runtime.get("canonical_repo_directory") != OFFICIAL_REPO_DIRECTORY
        or runtime.get("model_workdir") != OFFICIAL_REPO_DIRECTORY
        or runtime.get("single_rw_mount") is not True
        or runtime.get("validator_network") != "none"
        or runtime.get("image_pull_policy") != "never"
        or runtime.get("model_shell_scope") != "isolated-instance-container"
        or runtime.get("model_network_mode") != "none"
        or runtime.get("model_route_scope")
        != "supervisor-fixed-target-unix-socket-loopback-bridge"
        or runtime.get("model_public_egress_blocking") != "enforced-and-behavior-probed"
        or runtime.get("hidden_eval_phase") != "post-agent-fresh-container"
        or runtime.get("hidden_eval_network") != "none"
        or not isinstance(runtime.get("native_probe_root"), str)
        or not _NATIVE_WSL_ROOT.fullmatch(str(runtime.get("native_probe_root")))
        or ".." in str(runtime.get("native_probe_root")).split("/")
    ):
        raise ExecutionContractError("execution_lock_binding_invalid")
    for section, key in (
        (dataset, "parquet"),
        (harness, "checkout"),
        (opencode, "bundle_manifest"),
        (opencode, "patch_manifest"),
        (ccswitch, "route_manifest"),
        (validator, "wrapper"),
        (validator, "adapter"),
        (validator, "runtime_adapter"),
        (validator, "coordinator"),
        (validator, "route_diagnostics"),
        (validator, "representative_probe_runner"),
        (validator, "representative_probe_builder"),
        (validator, "tool_contract"),
        (validator, "policy"),
        (validator, "execution_config"),
        (validator, "execution_models"),
        (validator, "execution_runner"),
        (validator, "execution_trace"),
        (validator, "session_export"),
        (validator, "distillation_validator"),
        (validator, "distillation_containerfile"),
    ):
        _project_path(root, section.get(key), "execution_lock_path_escape")
    _project_path(
        root, value.get("sealed_probe_instance"), "execution_lock_path_escape"
    )
    _project_path(
        root,
        runtime.get("representative_probe_attestation"),
        "execution_lock_path_escape",
    )
    for key in (
        "wrapper_sha256",
        "adapter_sha256",
        "runtime_adapter_sha256",
        "coordinator_sha256",
        "route_diagnostics_sha256",
        "representative_probe_runner_sha256",
        "representative_probe_builder_sha256",
        "tool_contract_sha256",
        "policy_sha256",
        "execution_config_sha256",
        "execution_models_sha256",
        "execution_runner_sha256",
        "execution_trace_sha256",
        "session_export_sha256",
        "distillation_validator_sha256",
        "distillation_containerfile_sha256",
    ):
        digest = validator.get(key)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ExecutionContractError("execution_lock_code_sha256_invalid")
    return value


def _git_output(checkout: Path, args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _wrapper_probe(wrapper: Path) -> tuple[bool, bool]:
    if not wrapper.is_file():
        return False, False

    def run(argument: str) -> tuple[int, Mapping[str, Any] | None]:
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                code = validator_cli([argument])
            parsed = json.loads(output.getvalue())
            return code, _mapping(parsed, "wrapper_probe_invalid")
        except (json.JSONDecodeError, ExecutionContractError):
            return -1, None

    self_code, self_value = run("--self-test")
    reject_code, reject_value = run("shell")
    self_ok = False
    reject_ok = False
    try:
        if self_value is not None:
            self_result = SealedValidationResult.from_mapping(self_value)
            self_ok = self_code == 0 and self_result.status == "PASS"
        if reject_value is not None:
            reject_result = SealedValidationResult.from_mapping(reject_value)
            reject_ok = reject_code == 64 and reject_result.status == "ERROR"
    except ExecutionContractError:
        pass
    return self_ok, reject_ok


_WSL_BEHAVIOR_PROBE = r"""
set -u
IMAGE_KEY=$1
IMAGE_DIGEST=$2
PROBE_ROOT=$3
WINDOWS_BINARY=$4
IMAGE_REF="${IMAGE_KEY%:*}@${IMAGE_DIGEST}"
WORK=""
ROUTE_PID=""
SOURCE_ID=""
MODEL_ID=""
VALIDATOR_ID=""
canonical=0
post_agent=0
runtime=0
route=0
network_none=0
egress=0

cleanup() {
    [ -z "$ROUTE_PID" ] || kill "$ROUTE_PID" >/dev/null 2>&1 || true
    [ -z "$VALIDATOR_ID" ] || podman rm -f "$VALIDATOR_ID" >/dev/null 2>&1 || true
    [ -z "$MODEL_ID" ] || podman rm -f "$MODEL_ID" >/dev/null 2>&1 || true
    [ -z "$SOURCE_ID" ] || podman rm -f "$SOURCE_ID" >/dev/null 2>&1 || true
    [ -z "$WORK" ] || rm -rf "$WORK"
}
trap cleanup EXIT HUP INT TERM

if install -d -m 700 "$PROBE_ROOT" >/dev/null 2>&1; then
    WORK=$(mktemp -d "$PROBE_ROOT/probe.XXXXXX" 2>/dev/null || true)
fi
if [ -n "$WORK" ] && [ "$(stat -f -c %T "$WORK" 2>/dev/null || true)" = "ext2/ext3" ]; then
    ROUTE_DIR="$WORK/route"
    ROUTE_SOCKET="$ROUTE_DIR/ccswitch.sock"
    install -d -m 700 "$ROUTE_DIR"
    python3 -c 'import os,socket,sys; p=sys.argv[1]; s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(p); os.chmod(p,0o600); s.listen(8); exec("while True:\n c,_=s.accept()\n data=b\"\"\n while b\"\\r\\n\\r\\n\" not in data and len(data)<65536:\n  x=c.recv(4096)\n  if not x: break\n  data+=x\n c.sendall(b\"HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\nContent-Length: 2\\r\\nConnection: close\\r\\n\\r\\n{}\")\n c.close()")' "$ROUTE_SOCKET" >/dev/null 2>&1 &
    ROUTE_PID=$!
    attempt=0
    while [ ! -S "$ROUTE_SOCKET" ] && [ "$attempt" -lt 50 ]; do
        sleep 0.1
        attempt=$((attempt + 1))
    done
    mkdir -p "$WORK/testbed"
    if [ -S "$ROUTE_SOCKET" ]; then
        SOURCE_ID=$(podman create --pull=never --network none "$IMAGE_REF" sleep 300 2>/dev/null || true)
    fi
    if [ -n "$SOURCE_ID" ] && podman cp "${SOURCE_ID}:/testbed/." "$WORK/testbed" >/dev/null 2>&1; then
        podman rm -f "$SOURCE_ID" >/dev/null 2>&1 || true
        SOURCE_ID=""
        BINARY=$(wslpath -u "$WINDOWS_BINARY" 2>/dev/null || true)
        MODEL_ID=$(podman create --pull=never --network none --workdir /testbed \
            --read-only --cap-drop all --security-opt no-new-privileges \
            --tmpfs /tmp:rw,nosuid,size=64m \
            --mount "type=bind,src=$WORK/testbed,dst=/testbed,rw" \
            --mount "type=bind,src=$ROUTE_DIR,dst=/run/anchor-route,ro" \
            --mount "type=bind,src=$BINARY,dst=/anchor/bin/opencode,ro" \
            "$IMAGE_REF" sleep 300 2>/dev/null || true)
        if [ -n "$MODEL_ID" ]; then
            MOUNTS=$(podman inspect --format '{{range .Mounts}}{{.Source}}|{{.Destination}}|{{.RW}}{{"\n"}}{{end}}' "$MODEL_ID" 2>/dev/null || true)
            MODE=$(podman inspect --format '{{.HostConfig.NetworkMode}}' "$MODEL_ID" 2>/dev/null || true)
            RW_COUNT=$(printf '%s\n' "$MOUNTS" | awk -F'|' '$3=="true" {n++} END {print n+0}')
            if [ "$RW_COUNT" = "1" ] && \
                printf '%s\n' "$MOUNTS" | grep -F "$WORK/testbed|/testbed|true" >/dev/null && \
                printf '%s\n' "$MOUNTS" | grep -F "$ROUTE_DIR|/run/anchor-route|false" >/dev/null && \
                [ "$MODE" = "none" ] && \
                podman start "$MODEL_ID" >/dev/null 2>&1 && \
                podman exec --workdir /testbed "$MODEL_ID" sh -c \
                    'test "$PWD" = /testbed && test -d . && test -r . && test -w . && test "$(stat -f -c %T .)" = "ext2/ext3"' \
                    >/dev/null 2>&1; then
                canonical=1
                network_none=1
                if podman exec -d "$MODEL_ID" python3 -c \
                    'import socket,threading; p="/run/anchor-route/ccswitch.sock"; pump=lambda a,b: [b.sendall(x) for x in iter(lambda:a.recv(65536),b"")]; s=socket.socket(); s.bind(("127.0.0.1",18080)); s.listen(8); exec("while True:\n c,_=s.accept()\n u=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); u.connect(p)\n threading.Thread(target=pump,args=(c,u),daemon=True).start()\n threading.Thread(target=pump,args=(u,c),daemon=True).start()")' \
                    >/dev/null 2>&1; then
                    sleep 0.2
                fi
                if [ -n "$BINARY" ] && [ -f "$BINARY" ] && \
                    podman exec --workdir /testbed "$MODEL_ID" timeout 30 \
                        /anchor/bin/opencode --version >/dev/null 2>&1 && \
                    podman exec "$MODEL_ID" python3 -c \
                        'import urllib.request; [urllib.request.urlopen("http://127.0.0.1:18080"+p,timeout=5).read() for p in ("/anchor/health","/v1/models","/v1/responses")]' \
                        >/dev/null 2>&1; then
                    route=1
                fi
                if podman exec "$MODEL_ID" python3 -c \
                    'import socket; targets=(("1.1.1.1",443),("169.254.169.254",80),("10.0.2.2",80)); exec("for h,p in targets:\n try: socket.create_connection((h,p),timeout=1); raise SystemExit(20)\n except OSError: pass\nfor h in (\"github.com\",\"huggingface.co\"):\n try: socket.getaddrinfo(h,443); raise SystemExit(21)\n except OSError: pass")' \
                    >/dev/null 2>&1; then
                    egress=1
                fi
                if [ "$route" = "1" ] && [ "$egress" = "1" ] && \
                    podman exec --workdir /testbed "$MODEL_ID" sh -c \
                        'printf anchor-v3 > .anchor-v3-probe && sync' >/dev/null 2>&1; then
                    runtime=1
                fi
            fi
            podman rm -f "$MODEL_ID" >/dev/null 2>&1 || true
            OLD_MODEL_ID=$MODEL_ID
            MODEL_ID=""
            VALIDATOR_ID=$(podman create --pull=never --network none --workdir /testbed \
                --mount "type=bind,src=$WORK/testbed,dst=/testbed,rw" \
                "$IMAGE_REF" sleep 300 2>/dev/null || true)
            if [ -n "$VALIDATOR_ID" ] && [ "$VALIDATOR_ID" != "$OLD_MODEL_ID" ]; then
                VMOUNTS=$(podman inspect --format '{{range .Mounts}}{{.Source}}|{{.Destination}}|{{.RW}}{{"\n"}}{{end}}' "$VALIDATOR_ID" 2>/dev/null || true)
                VMODE=$(podman inspect --format '{{.HostConfig.NetworkMode}}' "$VALIDATOR_ID" 2>/dev/null || true)
                if [ "$VMOUNTS" = "$WORK/testbed|/testbed|true" ] && \
                    [ "$VMODE" = "none" ] && \
                    podman start "$VALIDATOR_ID" >/dev/null 2>&1 && \
                    podman exec --workdir /testbed "$VALIDATOR_ID" sh -c \
                        'test -f .anchor-v3-probe && test ! -e /tmp/anchor-opencode && test "$(stat -f -c %T .)" = "ext2/ext3"' \
                        >/dev/null 2>&1; then
                    post_agent=1
                fi
            fi
        fi
    fi
fi
printf 'canonical_worktree=%s\n' "$canonical"
printf 'post_agent_hidden_eval=%s\n' "$post_agent"
printf 'representative_runtime=%s\n' "$runtime"
printf 'supervisor_unix_route_reachable=%s\n' "$route"
printf 'model_network_none=%s\n' "$network_none"
printf 'public_egress_blocked=%s\n' "$egress"
"""


def _wsl_behavior_probes(
    *,
    runtime: Mapping[str, Any],
    image_key: str,
    image_digest: str,
    linux_binary: Path,
) -> dict[str, bool]:
    """Run content-free behavior probes without ever pulling an image.

    The script creates one ext4-backed testbed, gives the model container one
    writable bind mount at ``/testbed``, then removes it before creating the
    hidden-evaluator container.  ``--pull=never`` and the inspected digest pin
    are repeated on every container creation.
    """

    result = {
        "canonical_worktree": False,
        "post_agent_hidden_eval": False,
        "representative_runtime": False,
        "supervisor_unix_route_reachable": False,
        "model_network_none": False,
        "public_egress_blocked": False,
    }
    if (
        not _IMAGE_KEY.fullmatch(image_key)
        or not _IMAGE_DIGEST.fullmatch(image_digest)
        or not linux_binary.is_file()
    ):
        return result
    safe_environment = {
        key: value
        for key in ("SystemRoot", "SYSTEMROOT", "WINDIR", "PATH")
        if (value := os.environ.get(key))
    }
    try:
        completed = subprocess.run(
            [
                "wsl.exe",
                "--distribution",
                str(runtime["wsl_distro"]),
                "--user",
                "root",
                "--exec",
                "sh",
                "-s",
                "--",
                image_key,
                image_digest,
                str(runtime["native_probe_root"]),
                str(linux_binary),
            ],
            input=_WSL_BEHAVIOR_PROBE.replace("\r\n", "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=False,
            env=safe_environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if completed.returncode != 0:
        return result
    observed: dict[str, str] = {}
    stdout = completed.stdout.decode("utf-8", errors="replace")
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in result and value in {"0", "1"}:
            observed[key] = value
    if set(observed) != set(result):
        return result
    return {key: observed[key] == "1" for key in result}


def _wsl_receipt_key_probe(
    runtime: Mapping[str, Any],
    *,
    path_field: str = "receipt_key_path",
) -> bool:
    """Check only key metadata; key bytes never cross the WSL boundary."""

    try:
        completed = subprocess.run(
            [
                "wsl.exe",
                "--distribution",
                str(runtime["wsl_distro"]),
                "--user",
                "root",
                "--exec",
                "stat",
                "-c",
                "%a:%u:%g:%F:%s",
                str(runtime[path_field]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, KeyError):
        return False
    if completed.returncode != 0:
        return False
    fields = completed.stdout.decode("ascii", errors="replace").strip().split(":")
    return bool(
        len(fields) == 5
        and fields[:4] == ["600", "0", "0", "regular file"]
        and fields[4].isdigit()
        and int(fields[4]) >= 32
    )


def _wsl_verify_representative_private_bindings(
    runtime: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    lock_sha256: str,
    representative: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Verify the private image-ledger entry and receipt without exporting the key."""

    payload = {
        "lock_sha256": lock_sha256,
        "dataset_revision": dataset.get("revision"),
        "native_root": runtime.get("native_probe_root"),
        "receipt_key_path": runtime.get("receipt_key_path"),
        "representative": dict(representative),
        "receipt": dict(receipt),
    }
    script = r"""
import hashlib,hmac,json,pathlib,re,stat,subprocess,sys
value=json.load(sys.stdin); rep=value['representative']; receipt=value['receipt']
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(v): return hashlib.sha256(v).hexdigest()
hex64=re.compile(r'^[0-9a-f]{64}$'); digest=re.compile(r'^sha256:[0-9a-f]{64}$')
root=pathlib.Path(value['native_root']); key_path=pathlib.Path(value['receipt_key_path'])
image_ok=False; receipt_ok=False
try:
 ledger_path=root/'image-cache'/'ledger.json'; s=ledger_path.lstat()
 assert stat.S_ISREG(s.st_mode) and s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o600
 ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
 unsigned={k:ledger[k] for k in ('schema_version','execution_lock_sha256','dataset_revision','entry_count','entries')}
 assert ledger.get('schema_version')=='anchor.swebench-image-cache-ledger.v1'
 assert ledger.get('execution_lock_sha256')==value['lock_sha256']
 assert ledger.get('dataset_revision')==value['dataset_revision']
 assert ledger.get('content_sha256')==sha(canonical(unsigned).encode())
 assert ledger.get('entry_count')==len(ledger.get('entries',{}))
 entry=ledger['entries'][rep['task_id_sha256']]
 fields={'execution_lock_sha256','dataset_revision','task_id_sha256','instance_id_sha256','base_commit','image_key','image_digest','image_ref','recipe_sha256','acquisition_mode','binding_sha256'}
 assert set(entry)==fields
 assert entry['execution_lock_sha256']==value['lock_sha256']
 assert entry['dataset_revision']==value['dataset_revision']
 assert entry['task_id_sha256']==rep['task_id_sha256']
 assert entry['instance_id_sha256']==rep['instance_id_sha256']
 assert entry['base_commit']==rep['base_commit']
 assert sha(entry['image_key'].encode())==rep['image_key_sha256']
 assert entry['image_digest']==rep['image_digest']
 assert entry['binding_sha256']==rep['image_cache_binding_sha256']
 assert entry['binding_sha256']==sha(canonical({k:entry[k] for k in fields-{'binding_sha256'}}).encode())
 assert entry['acquisition_mode'] in {'pull','official-recipe-build'}
 def state(reference):
  p=subprocess.run(['podman','image','inspect','--format','{{.Digest}}|{{.Id}}',reference],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=60,check=False)
  if p.returncode: return None
  parts=p.stdout.decode('utf-8','replace').strip().split('|',1)
  if len(parts)!=2: return None
  return parts[0] if digest.fullmatch(parts[0]) else parts[1] if digest.fullmatch(parts[1]) else None
 assert state(entry['image_ref'])==entry['image_digest']
 assert state(entry['image_key'])==entry['image_digest']
 image_ok=True
except Exception: image_ok=False
try:
 s=key_path.lstat(); assert stat.S_ISREG(s.st_mode) and s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o600
 key=key_path.read_bytes(); assert len(key)>=32
 unsigned_fields={'checkpoint_id','task_id_sha256','revision','instance_id_sha256','image_digest','base_commit','patch_sha256','lock_sha256','receipt_schema','receipt_id','key_id','issued_by','provenance','system_private','visible_to_model','status','exit_code','duration_ms','stdout_sha256','stderr_sha256','report_hash'}
 assert set(receipt)==unsigned_fields|{'receipt_hmac_sha256'}
 unsigned={name:receipt[name] for name in sorted(unsigned_fields)}
 expected=hmac.new(key,canonical(unsigned).encode(),hashlib.sha256).hexdigest()
 assert hmac.compare_digest(receipt['receipt_hmac_sha256'],expected)
 assert receipt['key_id']==sha(key)[:16]
 assert receipt['receipt_schema']=='anchor.official-eval-receipt.v1'
 assert receipt['issued_by']=='anchor-official-eval-supervisor'
 assert receipt['provenance']=='official-swebench-harness-system-private'
 assert receipt['system_private'] is True and receipt['visible_to_model'] is False
 assert receipt['status'] in {'PASS','FAIL'} and receipt['exit_code']==0
 for name in ('checkpoint_id','task_id_sha256','instance_id_sha256','image_digest','base_commit','patch_sha256','lock_sha256'):
  assert receipt[name]==rep[name]
 assert receipt['revision']==rep['revision']
 receipt_ok=True
except Exception: receipt_ok=False
print(canonical({'image_binding_verified':image_ok,'receipt_authenticated':receipt_ok}))
"""
    try:
        completed = subprocess.run(
            [
                "wsl.exe",
                "--distribution",
                str(runtime["wsl_distro"]),
                "--user",
                "root",
                "--exec",
                "python3",
                "-c",
                script,
            ],
            input=(_canonical(payload) + "\n").encode("utf-8"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=180,
            check=False,
        )
        result = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (
        OSError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
    ):
        return False, False
    if completed.returncode != 0 or not isinstance(result, Mapping):
        return False, False
    return (
        result.get("image_binding_verified") is True,
        result.get("receipt_authenticated") is True,
    )


def _representative_probe_attestation(
    root: Path,
    lock: Mapping[str, Any],
    *,
    lock_sha256: str,
) -> tuple[bool, bool, dict[str, Any]]:
    """Validate an explicit, content-hash-bound representative live probe."""

    runtime = _mapping(lock["runtime"], "execution_lock_runtime_invalid")
    dataset = _mapping(lock["dataset"], "execution_lock_dataset_invalid")
    opencode = _mapping(lock["opencode"], "execution_lock_opencode_invalid")
    ccswitch = _mapping(lock["ccswitch"], "execution_lock_ccswitch_invalid")
    path = _project_path(
        root,
        runtime["representative_probe_attestation"],
        "execution_lock_path_escape",
    )
    status: dict[str, Any] = {
        "path": str(runtime["representative_probe_attestation"]),
        "present": path.is_file(),
        "content_sha256": None,
        "artifact_bindings_valid": False,
        "image_binding_verified": False,
        "official_receipt_authenticated": False,
        "content_free": True,
    }
    if not path.is_file() or path.is_symlink():
        return False, False, status
    try:
        value = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            "representative_probe_attestation_invalid",
        )
        _exact(
            value,
            {
                "schema_version",
                "execution_lock_sha256",
                "opencode",
                "ccswitch",
                "representative",
                "content_free",
                "content_sha256",
            },
            "representative_probe_attestation_shape_invalid",
        )
        opencode_binding = _mapping(
            value["opencode"], "representative_probe_opencode_invalid"
        )
        ccswitch_binding = _mapping(
            value["ccswitch"], "representative_probe_ccswitch_invalid"
        )
        representative = _mapping(
            value["representative"], "representative_probe_binding_invalid"
        )
        _exact(
            opencode_binding,
            {
                "baseline_commit",
                "patch_manifest",
                "patch_manifest_sha256",
                "bundle_manifest",
                "bundle_manifest_sha256",
                "linux_binary",
                "linux_binary_sha256",
            },
            "representative_probe_opencode_shape_invalid",
        )
        _exact(
            ccswitch_binding,
            {"route_manifest", "route_manifest_sha256"},
            "representative_probe_ccswitch_shape_invalid",
        )
        _exact(
            representative,
            {
                "checkpoint_id",
                "task_id_sha256",
                "revision",
                "instance_id_sha256",
                "image_key_sha256",
                "image_digest",
                "image_cache_binding_sha256",
                "base_commit",
                "final_patch",
                "final_patch_sha256",
                "official_receipt",
                "official_receipt_sha256",
            },
            "representative_probe_binding_shape_invalid",
        )
        unsigned = {name: value[name] for name in value if name != "content_sha256"}
        if (
            value.get("schema_version") != REPRESENTATIVE_PROBE_ATTESTATION_SCHEMA
            or value.get("execution_lock_sha256") != lock_sha256
            or value.get("content_free") is not True
            or value.get("content_sha256")
            != _sha256_bytes(_canonical(unsigned).encode("utf-8"))
        ):
            raise ExecutionContractError("representative_probe_content_binding_invalid")
        patch_manifest = _project_path(
            root, opencode_binding["patch_manifest"], "execution_lock_path_escape"
        )
        bundle_manifest = _project_path(
            root, opencode_binding["bundle_manifest"], "execution_lock_path_escape"
        )
        linux_binary = _project_path(
            root, opencode_binding["linux_binary"], "execution_lock_path_escape"
        )
        route_manifest = _project_path(
            root, ccswitch_binding["route_manifest"], "execution_lock_path_escape"
        )
        final_patch = _project_path(
            root, representative["final_patch"], "execution_lock_path_escape"
        )
        receipt_path = _project_path(
            root, representative["official_receipt"], "execution_lock_path_escape"
        )
        files = (
            (patch_manifest, opencode_binding["patch_manifest_sha256"]),
            (bundle_manifest, opencode_binding["bundle_manifest_sha256"]),
            (linux_binary, opencode_binding["linux_binary_sha256"]),
            (route_manifest, ccswitch_binding["route_manifest_sha256"]),
            (final_patch, representative["final_patch_sha256"]),
            (receipt_path, representative["official_receipt_sha256"]),
        )
        if any(
            path_value.is_symlink()
            or not path_value.is_file()
            or not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
            or sha256_file(path_value) != expected
            for path_value, expected in files
        ):
            raise ExecutionContractError("representative_probe_artifact_hash_invalid")
        if final_patch.stat().st_size < 1:
            raise ExecutionContractError("representative_probe_final_patch_empty")
        if (
            opencode_binding["patch_manifest"] != opencode["patch_manifest"]
            or opencode_binding["bundle_manifest"] != opencode["bundle_manifest"]
            or ccswitch_binding["route_manifest"] != ccswitch["route_manifest"]
        ):
            raise ExecutionContractError("representative_probe_locked_path_mismatch")
        patch_value = _mapping(
            json.loads(patch_manifest.read_text(encoding="utf-8")),
            "representative_probe_patch_manifest_invalid",
        )
        bundle_value = _mapping(
            json.loads(bundle_manifest.read_text(encoding="utf-8")),
            "representative_probe_bundle_manifest_invalid",
        )
        route_value = _mapping(
            json.loads(route_manifest.read_text(encoding="utf-8")),
            "representative_probe_route_manifest_invalid",
        )
        source = _mapping(
            bundle_value.get("source"), "representative_probe_bundle_manifest_invalid"
        )
        platforms = _mapping(
            bundle_value.get("platforms"),
            "representative_probe_bundle_manifest_invalid",
        )
        linux_entry = _mapping(
            platforms.get("linux-x64"), "representative_probe_bundle_manifest_invalid"
        )
        binary_entry = _mapping(
            linux_entry.get("binary"), "representative_probe_bundle_manifest_invalid"
        )
        expected_linux = (
            bundle_manifest.parent / str(binary_entry.get("path", ""))
        ).resolve()
        expected_linux.relative_to(bundle_manifest.parent.resolve())
        if (
            patch_value.get("baseline_commit") != opencode_binding["baseline_commit"]
            or patch_value.get("tool_contract_version") != EXECUTION_TOOL_CONTRACT_V3
            or source.get("baseline_commit") != opencode_binding["baseline_commit"]
            or source.get("tool_contract_version") != EXECUTION_TOOL_CONTRACT_V3
            or source.get("patch_source_manifest_sha256")
            != opencode_binding["patch_manifest_sha256"]
            or expected_linux != linux_binary.resolve()
            or binary_entry.get("sha256") != opencode_binding["linux_binary_sha256"]
            or route_value.get("schema_version") != "anchor.ccswitch-route-manifest.v1"
            or route_value.get("ready") is not True
        ):
            raise ExecutionContractError(
                "representative_probe_artifact_lineage_invalid"
            )
        receipt = _mapping(
            json.loads(receipt_path.read_text(encoding="utf-8")),
            "representative_probe_receipt_invalid",
        )
        if (
            any(
                not isinstance(representative.get(name), str)
                or not _SHA256.fullmatch(str(representative.get(name)))
                for name in (
                    "checkpoint_id",
                    "task_id_sha256",
                    "instance_id_sha256",
                    "image_key_sha256",
                    "image_cache_binding_sha256",
                    "final_patch_sha256",
                    "official_receipt_sha256",
                )
            )
            or not isinstance(representative.get("revision"), int)
            or isinstance(representative.get("revision"), bool)
            or int(representative.get("revision", 0)) < 1
            or not isinstance(representative.get("image_digest"), str)
            or not _IMAGE_DIGEST.fullmatch(str(representative.get("image_digest")))
            or not isinstance(representative.get("base_commit"), str)
            or not _COMMIT.fullmatch(str(representative.get("base_commit")))
            or representative["final_patch_sha256"] != sha256_file(final_patch)
        ):
            raise ExecutionContractError("representative_probe_binding_invalid")
        image_ok, receipt_ok = _wsl_verify_representative_private_bindings(
            runtime,
            dataset,
            lock_sha256=lock_sha256,
            representative=representative,
            receipt=receipt,
        )
        status.update(
            {
                "content_sha256": str(value["content_sha256"]),
                "artifact_bindings_valid": True,
                "image_binding_verified": image_ok,
                "official_receipt_authenticated": receipt_ok,
            }
        )
        return bool(image_ok and receipt_ok), bool(receipt_ok), status
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ExecutionContractError,
        KeyError,
    ):
        return False, False, status


def _on_demand_image_cache_probe(
    runtime: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    lock_sha256: str,
) -> tuple[bool, str, str | None, int]:
    """Offline-only integrity probe for the supervisor-private image ledger.

    An absent ledger is a valid empty cache: the first explicitly confirmed
    live task performs trusted on-demand acquisition.  Existing ledgers must be
    content-hash bound to this exact execution lock and dataset revision.  This
    probe invokes no pull/build command and never enumerates task identities.
    """

    native_root = str(runtime.get("native_probe_root", "")).rstrip("/")
    ledger_path = f"{native_root}/image-cache/ledger.json"
    script = r"""
import hashlib,json,pathlib,re,stat,subprocess,sys
path=pathlib.Path(sys.argv[1]); lock=sys.argv[2]; revision=sys.argv[3]
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(v): return hashlib.sha256(v).hexdigest()
podman=subprocess.run(['podman','--version'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30,check=False)
if podman.returncode: raise SystemExit(3)
for directory in (path.parent.parent,path.parent):
 if directory.exists():
  s=directory.lstat()
  if (not stat.S_ISDIR(s.st_mode) or s.st_uid!=0 or
   stat.S_IMODE(s.st_mode)!=0o700): raise SystemExit(3)
if not path.exists():
 print(canonical({'state':'empty','content_sha256':None,'entry_count':0})); raise SystemExit(0)
state=path.lstat()
if (not stat.S_ISREG(state.st_mode) or state.st_uid!=0 or
 stat.S_IMODE(state.st_mode)!=0o600): raise SystemExit(3)
try: value=json.loads(path.read_text(encoding='utf-8'))
except Exception: raise SystemExit(3)
if set(value)!={'schema_version','execution_lock_sha256','dataset_revision','entry_count','entries','content_sha256'}: raise SystemExit(3)
entries=value.get('entries'); unsigned={k:value[k] for k in ('schema_version','execution_lock_sha256','dataset_revision','entry_count','entries')}
if (value.get('schema_version')!='anchor.swebench-image-cache-ledger.v1' or
 value.get('execution_lock_sha256')!=lock or value.get('dataset_revision')!=revision or
 not isinstance(entries,dict) or not isinstance(value.get('entry_count'),int) or
 isinstance(value.get('entry_count'),bool) or value.get('entry_count')!=len(entries) or
 value.get('content_sha256')!=sha(canonical(unsigned).encode())): raise SystemExit(3)
binding_fields={'execution_lock_sha256','dataset_revision','task_id_sha256','instance_id_sha256','base_commit','image_key','image_digest','image_ref','recipe_sha256','acquisition_mode','binding_sha256'}
hex64=re.compile(r'^[0-9a-f]{64}$'); digest=re.compile(r'^sha256:[0-9a-f]{64}$')
for key,entry in entries.items():
 if not hex64.fullmatch(str(key)) or not isinstance(entry,dict) or set(entry)!=binding_fields or entry.get('task_id_sha256')!=key: raise SystemExit(3)
 expected=sha(canonical({k:entry[k] for k in binding_fields-{'binding_sha256'}}).encode())
 if (entry.get('binding_sha256')!=expected or not digest.fullmatch(str(entry.get('image_digest',''))) or
  not hex64.fullmatch(str(entry.get('instance_id_sha256',''))) or
  not hex64.fullmatch(str(entry.get('recipe_sha256',''))) or
  entry.get('acquisition_mode') not in {'pull','official-recipe-build'}): raise SystemExit(3)
print(canonical({'state':'bound','content_sha256':value['content_sha256'],'entry_count':len(entries)}))
"""
    try:
        completed = subprocess.run(
            [
                "wsl.exe",
                "--distribution",
                str(runtime["wsl_distro"]),
                "--user",
                "root",
                "--exec",
                "python3",
                "-c",
                script,
                ledger_path,
                lock_sha256,
                str(dataset["revision"]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        value = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (
        OSError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
    ):
        return False, "unavailable", None, 0
    state = value.get("state") if isinstance(value, Mapping) else None
    digest = value.get("content_sha256") if isinstance(value, Mapping) else None
    count = value.get("entry_count") if isinstance(value, Mapping) else None
    valid = bool(
        completed.returncode == 0
        and state in {"empty", "bound"}
        and (digest is None or (isinstance(digest, str) and _SHA256.fullmatch(digest)))
        and isinstance(count, int)
        and not isinstance(count, bool)
        and 0 <= count <= int(dataset.get("row_count", 0))
        and ((state == "empty" and digest is None and count == 0) or state == "bound")
    )
    return (
        valid,
        str(state or "invalid"),
        str(digest) if digest else None,
        int(count or 0),
    )


def inspect_execution_environment(
    root: Path, lock: Mapping[str, Any], *, lock_sha256: str
) -> dict[str, Any]:
    """Build a deterministic, content-free local attestation snapshot."""

    remaining: list[str] = []
    dataset = _mapping(lock["dataset"], "execution_lock_dataset_invalid")
    harness = _mapping(lock["official_harness"], "execution_lock_harness_invalid")
    opencode = _mapping(lock["opencode"], "execution_lock_opencode_invalid")
    validator = _mapping(lock["validator"], "execution_lock_validator_invalid")
    runtime = _mapping(lock["runtime"], "execution_lock_runtime_invalid")

    receipt_key_ready = _wsl_receipt_key_probe(runtime)
    if not receipt_key_ready:
        remaining.append("official_eval_receipt_key_missing_or_invalid")
    distillation_receipt_key_ready = _wsl_receipt_key_probe(
        runtime,
        path_field="distillation_receipt_key_path",
    )
    if not distillation_receipt_key_ready:
        remaining.append("distillation_receipt_key_missing_or_invalid")
    (
        on_demand_image_cache_ready,
        image_cache_ledger_state,
        image_cache_ledger_sha256,
        image_cache_ledger_entries,
    ) = _on_demand_image_cache_probe(
        runtime,
        dataset,
        lock_sha256=lock_sha256,
    )
    if not on_demand_image_cache_ready:
        remaining.append(
            "official_image_acquisition_supervisor_unavailable_or_ledger_invalid"
        )

    parquet = _project_path(root, dataset["parquet"], "execution_lock_path_escape")
    dataset_ok = parquet.is_file() and sha256_file(parquet) == dataset["parquet_sha256"]
    if not dataset_ok:
        remaining.append("source_train_parquet_binding_failed")

    wrapper = _project_path(root, validator["wrapper"], "execution_lock_path_escape")
    adapter = _project_path(root, validator["adapter"], "execution_lock_path_escape")
    runtime_adapter = _project_path(
        root, validator["runtime_adapter"], "execution_lock_path_escape"
    )
    coordinator = _project_path(
        root, validator["coordinator"], "execution_lock_path_escape"
    )
    route_diagnostics = _project_path(
        root, validator["route_diagnostics"], "execution_lock_path_escape"
    )
    representative_probe_runner = _project_path(
        root,
        validator["representative_probe_runner"],
        "execution_lock_path_escape",
    )
    representative_probe_builder = _project_path(
        root,
        validator["representative_probe_builder"],
        "execution_lock_path_escape",
    )
    tool_contract = _project_path(
        root, validator["tool_contract"], "execution_lock_path_escape"
    )
    policy_module = _project_path(
        root, validator["policy"], "execution_lock_path_escape"
    )
    execution_config = _project_path(
        root, validator["execution_config"], "execution_lock_path_escape"
    )
    execution_models = _project_path(
        root, validator["execution_models"], "execution_lock_path_escape"
    )
    execution_runner = _project_path(
        root, validator["execution_runner"], "execution_lock_path_escape"
    )
    execution_trace = _project_path(
        root, validator["execution_trace"], "execution_lock_path_escape"
    )
    session_export = _project_path(
        root, validator["session_export"], "execution_lock_path_escape"
    )
    distillation_validator = _project_path(
        root,
        validator["distillation_validator"],
        "execution_lock_path_escape",
    )
    distillation_containerfile = _project_path(
        root,
        validator["distillation_containerfile"],
        "execution_lock_path_escape",
    )
    wrapper_hash = sha256_file(wrapper) if wrapper.is_file() else None
    adapter_hash = sha256_file(adapter) if adapter.is_file() else None
    runtime_adapter_hash = (
        sha256_file(runtime_adapter) if runtime_adapter.is_file() else None
    )
    coordinator_hash = sha256_file(coordinator) if coordinator.is_file() else None
    route_diagnostics_hash = (
        sha256_file(route_diagnostics) if route_diagnostics.is_file() else None
    )
    representative_probe_runner_hash = (
        sha256_file(representative_probe_runner)
        if representative_probe_runner.is_file()
        else None
    )
    representative_probe_builder_hash = (
        sha256_file(representative_probe_builder)
        if representative_probe_builder.is_file()
        else None
    )
    tool_contract_hash = sha256_file(tool_contract) if tool_contract.is_file() else None
    policy_hash = sha256_file(policy_module) if policy_module.is_file() else None
    execution_config_hash = (
        sha256_file(execution_config) if execution_config.is_file() else None
    )
    execution_models_hash = (
        sha256_file(execution_models) if execution_models.is_file() else None
    )
    execution_runner_hash = (
        sha256_file(execution_runner) if execution_runner.is_file() else None
    )
    execution_trace_hash = (
        sha256_file(execution_trace) if execution_trace.is_file() else None
    )
    session_export_hash = (
        sha256_file(session_export) if session_export.is_file() else None
    )
    distillation_validator_hash = (
        sha256_file(distillation_validator)
        if distillation_validator.is_file()
        else None
    )
    distillation_containerfile_hash = (
        sha256_file(distillation_containerfile)
        if distillation_containerfile.is_file()
        else None
    )
    code_bound = (
        wrapper_hash == validator["wrapper_sha256"]
        and adapter_hash == validator["adapter_sha256"]
        and runtime_adapter_hash == validator["runtime_adapter_sha256"]
        and coordinator_hash == validator["coordinator_sha256"]
        and route_diagnostics_hash == validator["route_diagnostics_sha256"]
        and representative_probe_runner_hash
        == validator["representative_probe_runner_sha256"]
        and representative_probe_builder_hash
        == validator["representative_probe_builder_sha256"]
        and tool_contract_hash == validator["tool_contract_sha256"]
        and policy_hash == validator["policy_sha256"]
        and execution_config_hash == validator["execution_config_sha256"]
        and execution_models_hash == validator["execution_models_sha256"]
        and execution_runner_hash == validator["execution_runner_sha256"]
        and execution_trace_hash == validator["execution_trace_sha256"]
        and session_export_hash == validator["session_export_sha256"]
        and distillation_validator_hash == validator["distillation_validator_sha256"]
        and distillation_containerfile_hash
        == validator["distillation_containerfile_sha256"]
    )
    if not code_bound:
        remaining.append("sealed_validator_code_binding_failed")
    wrapper_self_test, wrapper_rejects_arbitrary = _wrapper_probe(wrapper)
    if not wrapper_self_test:
        remaining.append("sealed_validator_self_test_failed")
    if not wrapper_rejects_arbitrary:
        remaining.append("sealed_validator_arbitrary_command_rejection_failed")

    bundle_manifest = _project_path(
        root, opencode["bundle_manifest"], "execution_lock_path_escape"
    )
    bundle_hash = sha256_file(bundle_manifest) if bundle_manifest.is_file() else None
    bundle_contract: str | None = None
    model_isolation_contract = False
    testbed_workdir_contract = False
    windows_hash: str | None = None
    linux_hash: str | None = None
    linux_binary_path: Path | None = None
    if bundle_manifest.is_file():
        try:
            bundle = _mapping(
                json.loads(bundle_manifest.read_text(encoding="utf-8")),
                "bundle_invalid",
            )
            source = _mapping(bundle.get("source"), "bundle_invalid")
            bundle_contract = str(source.get("tool_contract_version", ""))
            model_isolation_contract = (
                source.get("tool_contract") == v3_contract_descriptor()
            )
            testbed_workdir_contract = source.get("swebench_runtime") == {
                "allowed_workdirs": ["/workspace", "/testbed"],
                "canonical_workdir": "/testbed",
                "single_rw_worktree": True,
                "state_outside_worktree": True,
            }
            platforms = _mapping(bundle.get("platforms"), "bundle_invalid")
            for platform, target in (
                ("windows-x64", "windows"),
                ("linux-x64", "linux"),
            ):
                entry = _mapping(platforms.get(platform), "bundle_invalid")
                binary = _mapping(entry.get("binary"), "bundle_invalid")
                binary_path = (
                    bundle_manifest.parent / str(binary.get("path", ""))
                ).resolve()
                try:
                    binary_path.relative_to(bundle_manifest.parent.resolve())
                except ValueError as exc:
                    raise ExecutionContractError("bundle_binary_path_escape") from exc
                actual = sha256_file(binary_path) if binary_path.is_file() else None
                if actual != binary.get("sha256"):
                    remaining.append(f"opencode_{target}_binary_binding_failed")
                if target == "windows":
                    windows_hash = actual
                else:
                    linux_hash = actual
                    linux_binary_path = binary_path
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ExecutionContractError,
        ):
            remaining.append("opencode_bundle_invalid")
    else:
        remaining.append("opencode_bundle_missing")
    if bundle_contract != EXECUTION_TOOL_CONTRACT_V3:
        remaining.append("opencode_tool_contract_v3_missing")
    if not model_isolation_contract:
        remaining.append("post_agent_validator_isolation_contract_missing")
    if not testbed_workdir_contract:
        remaining.append("opencode_testbed_workdir_contract_missing")

    checkout = _project_path(root, harness["checkout"], "execution_lock_path_escape")
    observed_revision = None
    clean_checkout = False
    import_ok = False
    testspec_probe = False
    image_probe = False
    image_digest: str | None = None
    image_key: str | None = None
    image_key_sha256 = None
    if not checkout.is_dir():
        remaining.extend(
            [
                "official_harness_checkout_missing",
                "official_harness_install_missing",
                "official_testspec_behavior_probe_missing",
                "instance_image_probe_missing",
            ]
        )
    else:
        observed_revision = _git_output(checkout, ["rev-parse", "HEAD"])
        status = _git_output(
            checkout, ["status", "--porcelain", "--untracked-files=all"]
        )
        clean_checkout = observed_revision == harness["revision"] and status == ""
        if not clean_checkout:
            remaining.append("official_harness_checkout_binding_failed")
        if clean_checkout:
            inserted = str(checkout)
            sys.path.insert(0, inserted)
            try:
                with official_harness_import_scope():
                    package = importlib.import_module("swebench")
                    package_origin = Path(
                        str(getattr(package, "__file__", ""))
                    ).resolve()
                    package_origin.relative_to(checkout.resolve())
                    if getattr(package, "__version__", None) != harness["version"]:
                        raise ExecutionContractError(
                            "official_harness_version_mismatch"
                        )
                    module = importlib.import_module(str(harness["module"]))
                    origin = Path(str(getattr(module, "__file__", ""))).resolve()
                    origin.relative_to(checkout.resolve())
                import_ok = True
                probe_path = _project_path(
                    root, lock["sealed_probe_instance"], "execution_lock_path_escape"
                )
                if not probe_path.is_file():
                    remaining.append("sealed_probe_instance_missing")
                    remaining.extend(
                        [
                            "official_testspec_behavior_probe_missing",
                            "instance_image_probe_missing",
                        ]
                    )
                else:
                    # Private probe input is consumed only inside this trusted process;
                    # neither its fields nor its instance id are retained in the attestation.
                    private_instance = _mapping(
                        json.loads(probe_path.read_text(encoding="utf-8")),
                        "sealed_probe_instance_invalid",
                    )
                    image_key = resolve_official_instance_image_key(
                        private_instance, lock, module
                    )
                    testspec_probe = True
                    image_key_sha256 = _sha256_bytes(image_key.encode("utf-8"))
                    try:
                        image_check = subprocess.run(
                            [
                                "wsl.exe",
                                "--distribution",
                                str(runtime["wsl_distro"]),
                                "--user",
                                "root",
                                "--exec",
                                "podman",
                                "image",
                                "inspect",
                                "--format",
                                "{{.Digest}}",
                                image_key,
                            ],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=30,
                            check=False,
                        )
                        candidate_digest = image_check.stdout.strip()
                        image_probe = image_check.returncode == 0 and bool(
                            _IMAGE_DIGEST.fullmatch(candidate_digest)
                        )
                        image_digest = candidate_digest if image_probe else None
                    except (OSError, subprocess.TimeoutExpired):
                        image_probe = False
                    if not image_probe:
                        remaining.append("instance_image_missing")
            except (
                ImportError,
                AttributeError,
                KeyError,
                ValueError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ExecutionContractError,
            ):
                remaining.extend(
                    [
                        "official_testspec_behavior_probe_failed",
                        "instance_image_probe_missing",
                    ]
                )
            finally:
                if sys.path and sys.path[0] == inserted:
                    sys.path.pop(0)
        if not import_ok:
            remaining.append("official_harness_install_missing")

    # Contract declarations are necessary but never replace behavior.  Run the
    # representative probe only when the exact official image is already local
    # and the patched Linux OpenCode binary is bound.  The probe itself repeats
    # both the digest pin and --pull=never, so this inspection can never fetch an
    # image as a side effect.
    behavior = {
        "canonical_worktree": False,
        "post_agent_hidden_eval": False,
        "representative_runtime": False,
        "supervisor_unix_route_reachable": False,
        "model_network_none": False,
        "public_egress_blocked": False,
    }
    if (
        image_probe
        and isinstance(image_key_sha256, str)
        and isinstance(image_digest, str)
        and isinstance(image_key, str)
        and model_isolation_contract
        and testbed_workdir_contract
        and linux_binary_path is not None
        and linux_hash is not None
    ):
        behavior = _wsl_behavior_probes(
            runtime=runtime,
            image_key=image_key,
            image_digest=image_digest,
            linux_binary=linux_binary_path,
        )
    canonical_worktree_probe = behavior["canonical_worktree"]
    post_agent_isolation_probe = behavior["post_agent_hidden_eval"]
    representative_runtime_smoke = behavior["representative_runtime"]
    route_reachability_probe = behavior["supervisor_unix_route_reachable"]
    model_network_none_probe = behavior["model_network_none"]
    public_egress_blocked_probe = behavior["public_egress_blocked"]
    # Declarations and primitive smoke tests never prove the patched runtime or
    # hidden evaluator.  Those gates are satisfied only by a separate explicit
    # representative probe attestation whose artifacts, private image-ledger
    # binding, and supervisor-HMAC receipt are all revalidated here.  No probe
    # is executed and no image is acquired by this static inspection.
    (
        patched_anchor_sandbox_probe,
        official_final_diff_eval_probe,
        representative_probe,
    ) = _representative_probe_attestation(root, lock, lock_sha256=lock_sha256)
    if not canonical_worktree_probe:
        remaining.append("canonical_testbed_worktree_probe_missing")
    if not testbed_workdir_contract:
        remaining.append("opencode_testbed_workdir_support_missing")
    if not post_agent_isolation_probe:
        remaining.append("post_agent_validator_isolation_probe_missing")
    if not representative_runtime_smoke:
        remaining.append("representative_instance_runtime_smoke_missing")
    if not route_reachability_probe:
        remaining.append("supervisor_unix_route_probe_missing")
    if not model_network_none_probe:
        remaining.append("model_network_none_probe_missing")
    if not public_egress_blocked_probe:
        remaining.append("model_public_egress_block_probe_missing")
    if not patched_anchor_sandbox_probe:
        remaining.append("patched_anchor_sandbox_probe_attestation_missing_or_invalid")
    if not official_final_diff_eval_probe:
        remaining.append(
            "official_testspec_final_diff_eval_attestation_missing_or_invalid"
        )
    if not image_probe:
        remaining.append("representative_instance_smoke_missing")
    remaining = sorted(set(remaining))
    return {
        "schema_version": EXECUTION_ATTESTATION_SCHEMA,
        "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3,
        "lock_sha256": lock_sha256,
        "ready": not remaining,
        "bindings": {
            "dataset": {
                "revision": dataset["revision"],
                "parquet_sha256": dataset["parquet_sha256"],
                "present_and_bound": dataset_ok,
            },
            "official_harness": {
                "repository": harness["repository"],
                "required_revision": harness["revision"],
                "required_version": harness["version"],
                "observed_revision": observed_revision,
                "clean_checkout": clean_checkout,
                "import_ok": import_ok,
                "testspec_probe": testspec_probe,
            },
            "opencode": {
                "bundle_manifest_sha256": bundle_hash,
                "tool_contract_version": bundle_contract,
                "windows_binary_sha256": windows_hash,
                "linux_binary_sha256": linux_hash,
                "model_isolation_contract": model_isolation_contract,
                "testbed_workdir_contract": testbed_workdir_contract,
            },
            "validator": {
                "wrapper_sha256": wrapper_hash,
                "adapter_sha256": adapter_hash,
                "runtime_adapter_sha256": runtime_adapter_hash,
                "coordinator_sha256": coordinator_hash,
                "route_diagnostics_sha256": route_diagnostics_hash,
                "representative_probe_runner_sha256": (
                    representative_probe_runner_hash
                ),
                "representative_probe_builder_sha256": (
                    representative_probe_builder_hash
                ),
                "tool_contract_sha256": tool_contract_hash,
                "policy_sha256": policy_hash,
                "execution_config_sha256": execution_config_hash,
                "execution_models_sha256": execution_models_hash,
                "execution_runner_sha256": execution_runner_hash,
                "execution_trace_sha256": execution_trace_hash,
                "session_export_sha256": session_export_hash,
                "distillation_validator_sha256": distillation_validator_hash,
                "distillation_validator_version": validator[
                    "distillation_validator_version"
                ],
                "distillation_validator_family": validator[
                    "distillation_validator_family"
                ],
                "distillation_containerfile_sha256": (distillation_containerfile_hash),
                "distillation_image_reference": validator[
                    "distillation_image_reference"
                ],
                "distillation_image_id_sha256": validator[
                    "distillation_image_id_sha256"
                ],
                "distillation_result_schema": validator["distillation_result_schema"],
                "distillation_allowed_actions": validator[
                    "distillation_allowed_actions"
                ],
                "code_bound": code_bound,
                "self_test": wrapper_self_test,
                "rejects_arbitrary_commands": wrapper_rejects_arbitrary,
                "allowed_actions": list(VALIDATOR_ACTIONS),
                "result_schema": SEALED_VALIDATOR_RESULT_SCHEMA,
            },
            "supervisor_private_state": {
                "receipt_key_path": runtime["receipt_key_path"],
                "receipt_key_metadata_valid": receipt_key_ready,
                "receipt_key_bytes_exposed": False,
                "distillation_receipt_key_path": runtime[
                    "distillation_receipt_key_path"
                ],
                "distillation_receipt_key_metadata_valid": (
                    distillation_receipt_key_ready
                ),
            },
            "on_demand_image_cache": {
                "acquisition_scope": "trusted-wsl-supervisor-only",
                "network_scope": "supervisor-pull-or-official-recipe-build-only",
                "native_ledger_path": (
                    str(runtime["native_probe_root"]).rstrip("/")
                    + "/image-cache/ledger.json"
                ),
                "ledger_state": image_cache_ledger_state,
                "ledger_content_sha256": image_cache_ledger_sha256,
                "ledger_entries": image_cache_ledger_entries,
                "maximum_entries": dataset["row_count"],
                "offline_integrity_probe": on_demand_image_cache_ready,
                "task_binding": "lock+dataset+task+instance+base+recipe+digest",
                "resume_reconciliation": "immutable-ref-and-mutable-label-before-use",
                "pull_during_model_or_eval": False,
            },
            "representative_live_probe": {
                **representative_probe,
                "required_generation_command": (
                    "py scripts/tooling/run_swebench_v3_representative_probe.py "
                    "--control-run-id <operator-id> "
                    "--confirm-representative-live --confirm-supervisor-network"
                ),
                "automatic_execution": False,
                "provider_requests_during_inspection": 0,
            },
            "instance_image": {
                "resolver": "official TestSpec.instance_image_key",
                "arbitrary_image_input_allowed": False,
                "image_key_sha256": image_key_sha256,
                "image_digest": image_digest,
                "present": image_probe,
            },
            "canonical_worktree": {
                "container_repo_directory": OFFICIAL_REPO_DIRECTORY,
                "single_host_worktree": True,
                "copy_source": "instance-container:/testbed",
                "model_mount": "/testbed",
                "validator_phase": "post-agent-fresh-container",
                "validator_result_visible_to_model": False,
                "native_wsl_root": runtime["native_probe_root"],
                "wsl_distro": runtime["wsl_distro"],
                "filesystem_required": "ext4",
                "probe_passed": canonical_worktree_probe,
            },
            "isolation_probes": {
                "public_model_iteration": {
                    "representative_runtime": representative_runtime_smoke,
                    "network_mode": "none",
                    "model_network_none": model_network_none_probe,
                    "supervisor_unix_route_reachable": route_reachability_probe,
                    "route_scope": "fixed-target-ccswitch-only",
                    "public_egress_blocked": public_egress_blocked_probe,
                    "real_results_visible_to_model": True,
                    "host_shell_allowed": False,
                    "attested_scope": "wsl-network-none-unix-relay-topology",
                    "real_ccswitch_probe_phase": "live-backend-after-route-start",
                    "known_limitation": None,
                },
                "hidden_official_eval": {
                    "post_agent_fresh_container": post_agent_isolation_probe,
                    "network_none": True,
                    "result_visible_to_model": False,
                    "provenance": "official-swebench-harness-system-private",
                },
                "independent_execution_probes": {
                    "patched_anchor_sandbox_end_to_end": patched_anchor_sandbox_probe,
                    "official_testspec_on_final_diff": official_final_diff_eval_probe,
                    "manifest_self_claim_sufficient": False,
                },
            },
        },
        "remaining_gates": remaining,
        "content_free": True,
        "oracle_material_retained": False,
    }


def build_execution_attestation(
    root: Path, lock_path: Path, *, expected_lock_sha256: str | None = None
) -> dict[str, Any]:
    lock = load_execution_lock(root, lock_path, expected_sha256=expected_lock_sha256)
    return inspect_execution_environment(root, lock, lock_sha256=sha256_file(lock_path))


def verify_execution_attestation(
    root: Path,
    attestation_path: Path,
    lock_path: Path,
    *,
    expected_lock_sha256: str,
) -> dict[str, Any]:
    """Re-probe local state and require exact attestation equality."""

    try:
        current = build_execution_attestation(
            root, lock_path, expected_lock_sha256=expected_lock_sha256
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ExecutionContractError,
    ) as exc:
        return {
            "ready": False,
            "reason_code": str(exc) or "execution_contract_probe_failed",
            "remaining_gates": [str(exc) or "execution_contract_probe_failed"],
        }
    if not attestation_path.is_file():
        return {
            "ready": False,
            "reason_code": "multilang_execution_attestation_missing",
            "remaining_gates": current["remaining_gates"],
            "current_probe": current,
        }
    try:
        claimed = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "ready": False,
            "reason_code": "multilang_execution_attestation_invalid",
            "remaining_gates": current["remaining_gates"],
            "current_probe": current,
        }
    if claimed != current:
        return {
            "ready": False,
            "reason_code": "multilang_execution_attestation_stale",
            "remaining_gates": current["remaining_gates"],
            "current_probe": current,
        }
    if current["ready"] is not True:
        return {
            "ready": False,
            "reason_code": "multilang_execution_attestation_incomplete",
            "remaining_gates": current["remaining_gates"],
            "current_probe": current,
        }
    return {
        "ready": True,
        "reason_code": "execution_contract_ready",
        "remaining_gates": [],
        "current_probe": current,
    }
