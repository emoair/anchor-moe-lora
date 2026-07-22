"""Fail-closed consumer for the Qwen training-prerequisite status contract.

The producer's v1 status artifact is intentionally a negative, content-free
snapshot.  Authenticating it proves why training is blocked; it cannot grant
training authority.  This module never follows the protected dataset paths
listed in that snapshot and never loads a tokenizer, model, or accelerator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import yaml


CONFIG_VERSION = "anchor.qwen-train-prerequisite-consumer-config.v1"
CONFIG_SHA256 = "4fdc8173baaa9f14d93a288b18f38691be62bb1fb8e646c579a06d9c78bc1a8a"
STATUS_VERSION = "anchor.qwen-train-prerequisite-status.v1"
TOKENIZER_BINDING_VERSION = "anchor.scaffold-tokenizer-binding-manifest.v1"
TOY_ATTESTATION_VERSION = "anchor.qwen-toy-source-disjoint-attestation.v1"
FORMAL_RELEASE_LOCK_VERSION = "anchor.generic-train-release-lock.v2"

PRODUCER_COMMIT = "a8efe5f55b72960b49bcb1ae3753b633afd14959"
STATUS_SCHEMA_SHA256 = (
    "e8d09abc26effcedc642125b4d84185f0e5072a23f5611f068274bd963c4f577"
)
STATUS_MANIFEST_SHA256 = (
    "70c8f0a866c5fb41c4c3726638b55a66efab77f8b2ee31c27ad31ab55def67da"
)
STATUS_SIDECAR_SHA256 = (
    "706c6bd6bfd0389bffe72ef9f8b34c51ef40885c78d65c9e9ef77ec5d304b948"
)
TOKENIZER_BINDING_SCHEMA_SHA256 = (
    "5b2e7c2e8e6efc1c9b7251fde853631e65806aca0364d9bb092ee9a07d135b25"
)
TOY_ATTESTATION_SCHEMA_SHA256 = (
    "7cdc714308b238db86303c61103d0b3e544cd0123bcdca625d6a9717ef5029ea"
)
FORMAL_RELEASE_LOCK_SCHEMA_SHA256 = (
    "119c55279c48246d45808849b03b9b6873570bcb82103da129ea64812fd3b5aa"
)

ORDERED_CONFIG_FIELDS = (
    "prerequisite_status_schema_sha256",
    "prerequisite_status_manifest_sha256",
    "tokenizer_binding_schema_sha256",
    "tokenizer_binding_manifest_sha256",
    "toy_attestation_schema_sha256",
    "toy_attestation_sha256",
    "formal_release_lock_schema_sha256",
    "formal_release_lock_sha256",
)
ORDERED_CLI_FIELDS = (
    "--prerequisite-status",
    "--prerequisite-status-sha256",
    "--tokenizer-binding",
    "--tokenizer-binding-sha256",
    "--toy-attestation",
    "--toy-attestation-sha256",
    "--formal-release-lock-schema-sha256",
    "--formal-release-lock",
    "--formal-release-lock-sha256",
)
ORDERED_RELEASE_FIELDS = (
    "prerequisite_status_schema_sha256",
    "prerequisite_status_manifest_sha256",
    "natural_language_scaffold_manifest_sha256",
    "long_context_token_inventory_manifest_sha256",
    "tokenizer_binding_schema_sha256",
    "tokenizer_binding_manifest_sha256",
    "trainable_base_snapshot_manifest_sha256",
    "tokenizer_base_compatibility_attestation_sha256",
    "formal_snapshot_manifest_sha256",
    "final_projector_manifest_sha256",
    "source_disjoint_manifest_sha256",
    "generic_execution_contract_sha256",
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CONTENT_KEYS = frozenset(
    {
        "answer",
        "body",
        "content",
        "preview",
        "prompt",
        "target",
        "token_ids",
        "token_indices",
    }
)
_FORMAL_ARTIFACT_KEYS = (
    "snapshot",
    "final_projector",
    "generic_execution",
    "source_disjoint",
    "release_lock",
)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


class QwenPrerequisiteConsumerError(RuntimeError):
    """Raised when a prerequisite artifact cannot be authenticated."""


def _fail(code: str) -> None:
    raise QwenPrerequisiteConsumerError(code)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class _BytesSnapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, code: str) -> None:
        current = _read_bytes_snapshot(self.path, code)
        if current.identity != self.identity or current.sha256 != self.sha256:
            _fail(code)


def _read_bytes_snapshot(path: Path, code: str) -> _BytesSnapshot:
    try:
        if not path.is_file() or path.is_symlink():
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise QwenPrerequisiteConsumerError(code) from exc
    before_id = _stat_identity(before)
    after_id = _stat_identity(after)
    if (
        before_id != after_id
        or after_id != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        identity=after_id,
    )


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    return stat.S_ISLNK(value.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _assert_physical_ancestry(path: Path, root: Path, code: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(code)
    current = root
    if _is_reparse_or_symlink(current):
        _fail(code)
    for part in relative.parts:
        current = current / part
        if current.exists() and _is_reparse_or_symlink(current):
            _fail(code)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _decode_json(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenPrerequisiteConsumerError(code) from exc
    return _mapping(value, code)


def _decode_yaml(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise QwenPrerequisiteConsumerError(code) from exc
    return _mapping(value, code)


def _safe_repository_path(value: object, code: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(code)
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        _fail(code)
    lexical = _REPOSITORY_ROOT / relative
    _assert_physical_ancestry(lexical, _REPOSITORY_ROOT, code)
    candidate = lexical.resolve(strict=False)
    try:
        candidate.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        _fail(code)
    return candidate


def _validate_json_schema(
    *,
    schema_snapshot: _BytesSnapshot,
    expected_schema_sha256: str,
    instance: Mapping[str, Any],
    code: str,
) -> None:
    if schema_snapshot.sha256 != expected_schema_sha256:
        _fail(f"{code}_schema_sha256_mismatch")
    schema = _decode_json(schema_snapshot, f"{code}_schema_invalid")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
    except ImportError as exc:
        raise QwenPrerequisiteConsumerError(
            f"{code}_jsonschema_dependency_unavailable"
        ) from exc
    except Exception as exc:
        raise QwenPrerequisiteConsumerError(f"{code}_schema_validation_failed") from exc


def _validate_schema_document(
    snapshot: _BytesSnapshot, expected_sha256: str, code: str
) -> None:
    if snapshot.sha256 != expected_sha256:
        _fail(f"{code}_sha256_mismatch")
    schema = _decode_json(snapshot, f"{code}_json_invalid")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except ImportError as exc:
        raise QwenPrerequisiteConsumerError(
            f"{code}_jsonschema_dependency_unavailable"
        ) from exc
    except Exception as exc:
        raise QwenPrerequisiteConsumerError(f"{code}_invalid") from exc


def _assert_no_content_fields(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        forbidden = {str(key) for key in value if str(key) in _CONTENT_KEYS}
        if forbidden:
            _fail(code)
        for item in value.values():
            _assert_no_content_fields(item, code)
    elif isinstance(value, list):
        for item in value:
            _assert_no_content_fields(item, code)


def _validate_sidecar(
    *, manifest: _BytesSnapshot, sidecar: _BytesSnapshot, expected_sidecar_sha256: str
) -> None:
    expected = f"{manifest.sha256}  manifest.json\n".encode("ascii")
    if sidecar.sha256 != expected_sidecar_sha256:
        _fail("prerequisite_status_sidecar_physical_sha256_mismatch")
    if sidecar.data != expected:
        _fail("prerequisite_status_sidecar_invalid")


def _validate_blocked_status_semantics(status: Mapping[str, Any]) -> None:
    if status.get("schema_version") != STATUS_VERSION:
        _fail("prerequisite_status_version_mismatch")
    if status.get("status") != "blocked":
        _fail("prerequisite_status_must_remain_blocked")
    if status.get("claim_scope") != (
        "content_free_prerequisite_status_only_no_training_authority"
    ):
        _fail("prerequisite_status_claim_scope_invalid")

    freeze = _mapping(
        status.get("consumer_freeze_requirements"),
        "consumer_freeze_requirements_invalid",
    )
    if tuple(freeze.get("ordered_config_fields", ())) != ORDERED_CONFIG_FIELDS:
        _fail("consumer_config_field_order_drift")
    if tuple(freeze.get("ordered_cli_fields", ())) != ORDERED_CLI_FIELDS:
        _fail("consumer_cli_field_order_drift")
    if tuple(freeze.get("ordered_release_fields", ())) != ORDERED_RELEASE_FIELDS:
        _fail("consumer_release_field_order_drift")
    if freeze.get("unknown_or_missing_result") != "fail_closed":
        _fail("consumer_unknown_result_not_fail_closed")
    if freeze.get("status") != "consumer_freeze_required":
        _fail("consumer_freeze_status_drift")

    formal = _mapping(status.get("formal_artifacts"), "formal_artifacts_invalid")
    if formal.get("matching_frozen_artifact_count") != 0:
        _fail("formal_artifact_count_must_remain_zero")
    for name in _FORMAL_ARTIFACT_KEYS:
        item = _mapping(formal.get(name), f"formal_artifact_{name}_invalid")
        if item.get("status") != "unavailable":
            _fail(f"formal_artifact_{name}_unexpected_status")
        for field in ("artifact_exists", "sidecar_exists", "training_eligible"):
            if item.get(field) is not False:
                _fail(f"formal_artifact_{name}_{field}_must_be_false")

    raw = _mapping(status.get("raw_gold_observation"), "raw_gold_observation_invalid")
    for field in ("coverage_complete", "lineage_complete", "training_ready"):
        if raw.get(field) is not False:
            _fail(f"raw_gold_{field}_must_be_false")
    strict = _mapping(raw.get("strict_complete_chains"), "strict_chain_status_invalid")
    if strict.get("threshold_met") is not False:
        _fail("strict_chain_threshold_must_be_false")

    tokenizer = _mapping(status.get("tokenizer_binding"), "tokenizer_binding_invalid")
    if (
        tokenizer.get("status")
        != "tokenizer_source_candidate_authenticated_binding_pending"
    ):
        _fail("tokenizer_binding_status_drift")
    if tokenizer.get("token_indices_emitted") is not False:
        _fail("token_indices_must_not_be_emitted")
    if tokenizer.get("training_eligible") is not False:
        _fail("tokenizer_binding_must_not_be_training_eligible")

    toy = _mapping(status.get("toy_attestation"), "toy_attestation_invalid")
    if toy.get("status") != "contract_ready_consumer_attestation_pending":
        _fail("toy_attestation_status_drift")
    if (
        toy.get("source_disjoint_claim_status")
        != "unverified_pending_authenticated_attestation"
    ):
        _fail("toy_attestation_claim_must_remain_unverified")
    if toy.get("attestation_artifact_status") != "unavailable":
        _fail("toy_attestation_artifact_must_remain_unavailable")
    for field in ("formal_training_data", "formal_training_authorized"):
        if toy.get(field) is not False:
            _fail(f"toy_attestation_{field}_must_be_false")

    safety = _mapping(status.get("safety"), "prerequisite_safety_invalid")
    false_fields = (
        "training_authorized",
        "canonical_gold_written",
        "heldout_written",
        "heldout_content_read",
        "heldout_content_emitted",
    )
    zero_fields = (
        "provider_requests",
        "network_requests",
        "model_loads",
        "gpu_requests",
        "full_bank_projection_runs",
    )
    if any(safety.get(field) is not False for field in false_fields):
        _fail("prerequisite_safety_boolean_drift")
    if any(safety.get(field) != 0 for field in zero_fields):
        _fail("prerequisite_safety_counter_drift")


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact_fields(
        config,
        {
            "schema_version",
            "claim_scope",
            "paths",
            "producer_contract",
            "bindings",
            "physical_sidecars",
            "artifact_paths",
            "policy",
        },
        "consumer_config_shape_invalid",
    )
    if config.get("schema_version") != CONFIG_VERSION:
        _fail("consumer_config_version_mismatch")
    if config.get("claim_scope") != (
        "content_free_prerequisite_gate_only_no_training_authority"
    ):
        _fail("consumer_config_claim_scope_drift")
    paths = _mapping(config.get("paths"), "consumer_paths_invalid")
    _exact_fields(
        paths,
        {
            "repository_root",
            "prerequisite_status",
            "prerequisite_status_schema",
            "tokenizer_binding_schema",
            "toy_attestation_schema",
            "formal_release_lock_schema",
        },
        "consumer_paths_shape_invalid",
    )
    if paths.get("repository_root") != "../..":
        _fail("consumer_repository_root_drift")
    if dict(paths) != {
        "repository_root": "../..",
        "prerequisite_status": (
            "fixtures/research/qwen_train_prerequisite_status/manifest.json"
        ),
        "prerequisite_status_schema": (
            "configs/research/qwen_train_prerequisite_status.schema.json"
        ),
        "tokenizer_binding_schema": (
            "configs/research/scaffold_tokenizer_binding_manifest.schema.json"
        ),
        "toy_attestation_schema": (
            "configs/research/qwen_toy_source_disjoint_attestation.schema.json"
        ),
        "formal_release_lock_schema": (
            "configs/research/generic_train_release_lock.schema.json"
        ),
    }:
        _fail("consumer_canonical_paths_drift")

    producer = _mapping(config.get("producer_contract"), "producer_contract_invalid")
    if producer != {
        "commit": PRODUCER_COMMIT,
        "prerequisite_status_schema_version": STATUS_VERSION,
        "tokenizer_binding_schema_version": TOKENIZER_BINDING_VERSION,
        "toy_attestation_schema_version": TOY_ATTESTATION_VERSION,
        "formal_release_lock_schema_version": FORMAL_RELEASE_LOCK_VERSION,
    }:
        _fail("producer_contract_identity_drift")

    bindings = _mapping(config.get("bindings"), "consumer_bindings_invalid")
    if tuple(bindings) != ORDERED_CONFIG_FIELDS:
        _fail("consumer_binding_order_drift")
    expected = {
        "prerequisite_status_schema_sha256": STATUS_SCHEMA_SHA256,
        "prerequisite_status_manifest_sha256": STATUS_MANIFEST_SHA256,
        "tokenizer_binding_schema_sha256": TOKENIZER_BINDING_SCHEMA_SHA256,
        "tokenizer_binding_manifest_sha256": None,
        "toy_attestation_schema_sha256": TOY_ATTESTATION_SCHEMA_SHA256,
        "toy_attestation_sha256": None,
        "formal_release_lock_schema_sha256": FORMAL_RELEASE_LOCK_SCHEMA_SHA256,
        "formal_release_lock_sha256": None,
    }
    if dict(bindings) != expected:
        _fail("consumer_binding_identity_drift")

    sidecars = _mapping(config.get("physical_sidecars"), "physical_sidecars_invalid")
    if dict(sidecars) != {"prerequisite_status_sha256": STATUS_SIDECAR_SHA256}:
        _fail("prerequisite_status_sidecar_binding_drift")
    artifacts = _mapping(config.get("artifact_paths"), "artifact_paths_invalid")
    if dict(artifacts) != {
        "tokenizer_binding": None,
        "toy_attestation": None,
        "formal_release_lock": None,
    }:
        _fail("consumer_artifact_paths_must_remain_unavailable")
    policy = _mapping(config.get("policy"), "consumer_policy_invalid")
    if dict(policy) != {
        "unknown_or_missing_result": "fail_closed",
        "require_mandatory_sha256_sidecars": True,
        "training_authorized": False,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "provider_requests": 0,
    }:
        _fail("consumer_policy_drift")


def evaluate_prerequisites(config_path: str | Path) -> dict[str, Any]:
    """Authenticate the frozen blocked snapshot and return a negative decision."""

    requested_config = Path(config_path)
    if ".." in requested_config.parts:
        _fail("consumer_config_path_invalid")
    config_file = requested_config
    if not requested_config.is_absolute():
        config_file = _REPOSITORY_ROOT / requested_config
    _assert_physical_ancestry(
        config_file, _REPOSITORY_ROOT, "consumer_config_path_invalid"
    )
    resolved_config = config_file.resolve(strict=False)
    try:
        resolved_config.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        _fail("consumer_config_path_invalid")
    config_file = resolved_config
    config_snapshot = _read_bytes_snapshot(config_file, "consumer_config_unreadable")
    if config_snapshot.sha256 != CONFIG_SHA256:
        _fail("consumer_config_sha256_mismatch")
    config = _decode_yaml(config_snapshot, "consumer_config_invalid")
    _validate_config(config)
    paths = _mapping(config["paths"], "consumer_paths_invalid")

    status_path = _safe_repository_path(
        paths["prerequisite_status"], "status_path_invalid"
    )
    status_schema_path = _safe_repository_path(
        paths["prerequisite_status_schema"], "status_schema_path_invalid"
    )
    binding_schema_path = _safe_repository_path(
        paths["tokenizer_binding_schema"], "binding_schema_path_invalid"
    )
    toy_schema_path = _safe_repository_path(
        paths["toy_attestation_schema"], "toy_schema_path_invalid"
    )
    release_schema_path = _safe_repository_path(
        paths["formal_release_lock_schema"], "release_schema_path_invalid"
    )

    status_snapshot = _read_bytes_snapshot(status_path, "status_manifest_unreadable")
    sidecar_snapshot = _read_bytes_snapshot(
        status_path.with_name("manifest.json.sha256"), "status_sidecar_unreadable"
    )
    status_schema_snapshot = _read_bytes_snapshot(
        status_schema_path, "status_schema_unreadable"
    )
    binding_schema_snapshot = _read_bytes_snapshot(
        binding_schema_path, "binding_schema_unreadable"
    )
    toy_schema_snapshot = _read_bytes_snapshot(toy_schema_path, "toy_schema_unreadable")
    release_schema_snapshot = _read_bytes_snapshot(
        release_schema_path, "release_schema_unreadable"
    )

    if status_snapshot.sha256 != STATUS_MANIFEST_SHA256:
        _fail("prerequisite_status_manifest_sha256_mismatch")
    _validate_sidecar(
        manifest=status_snapshot,
        sidecar=sidecar_snapshot,
        expected_sidecar_sha256=STATUS_SIDECAR_SHA256,
    )
    status = _decode_json(status_snapshot, "prerequisite_status_json_invalid")
    _assert_no_content_fields(status, "prerequisite_status_contains_content_field")
    _validate_json_schema(
        schema_snapshot=status_schema_snapshot,
        expected_schema_sha256=STATUS_SCHEMA_SHA256,
        instance=status,
        code="prerequisite_status",
    )
    _validate_schema_document(
        binding_schema_snapshot,
        TOKENIZER_BINDING_SCHEMA_SHA256,
        "tokenizer_binding_schema",
    )
    _validate_schema_document(
        toy_schema_snapshot,
        TOY_ATTESTATION_SCHEMA_SHA256,
        "toy_attestation_schema",
    )
    _validate_schema_document(
        release_schema_snapshot,
        FORMAL_RELEASE_LOCK_SCHEMA_SHA256,
        "formal_release_lock_schema",
    )
    _validate_blocked_status_semantics(status)

    for snapshot, code in (
        (config_snapshot, "consumer_config_changed_during_validation"),
        (status_snapshot, "status_manifest_changed_during_validation"),
        (sidecar_snapshot, "status_sidecar_changed_during_validation"),
        (status_schema_snapshot, "status_schema_changed_during_validation"),
        (binding_schema_snapshot, "binding_schema_changed_during_validation"),
        (toy_schema_snapshot, "toy_schema_changed_during_validation"),
        (release_schema_snapshot, "release_schema_changed_during_validation"),
    ):
        snapshot.assert_unchanged(code)

    return {
        "schema_version": "anchor.qwen-train-prerequisite-decision.v1",
        "status": "blocked",
        "training_authorized": False,
        "formal_training_authorized": False,
        "reason": "authenticated_prerequisite_status_reports_missing_formal_artifacts",
        "producer_commit": PRODUCER_COMMIT,
        "bindings": {
            "prerequisite_status_schema_sha256": STATUS_SCHEMA_SHA256,
            "prerequisite_status_manifest_sha256": STATUS_MANIFEST_SHA256,
            "tokenizer_binding_schema_sha256": TOKENIZER_BINDING_SCHEMA_SHA256,
            "tokenizer_binding_manifest_sha256": None,
            "toy_attestation_schema_sha256": TOY_ATTESTATION_SCHEMA_SHA256,
            "toy_attestation_sha256": None,
            "formal_release_lock_schema_sha256": FORMAL_RELEASE_LOCK_SCHEMA_SHA256,
            "formal_release_lock_sha256": None,
        },
        "missing_artifacts": [
            "tokenizer_binding_manifest",
            "toy_source_disjoint_attestation",
            "formal_release_lock",
            "formal_snapshot",
            "final_projector",
            "generic_execution_contract",
            "source_disjoint_manifest",
        ],
        "audit": {
            "protected_dataset_files_read": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate the frozen Qwen prerequisite status (no training)"
    )
    parser.add_argument("--config", required=True)
    # Keep this order identical to producer consumer_freeze_requirements.
    parser.add_argument("--prerequisite-status")
    parser.add_argument("--prerequisite-status-sha256")
    parser.add_argument("--tokenizer-binding")
    parser.add_argument("--tokenizer-binding-sha256")
    parser.add_argument("--toy-attestation")
    parser.add_argument("--toy-attestation-sha256")
    parser.add_argument("--formal-release-lock-schema-sha256")
    parser.add_argument("--formal-release-lock")
    parser.add_argument("--formal-release-lock-sha256")
    return parser


def _validate_cli_overrides(namespace: argparse.Namespace) -> None:
    status_pair = (
        namespace.prerequisite_status,
        namespace.prerequisite_status_sha256,
    )
    if any(value is not None for value in status_pair):
        if status_pair != (
            "fixtures/research/qwen_train_prerequisite_status/manifest.json",
            STATUS_MANIFEST_SHA256,
        ):
            _fail("blocked_v1_status_rejects_unfrozen_cli_overrides")
    if namespace.formal_release_lock_schema_sha256 not in (
        None,
        FORMAL_RELEASE_LOCK_SCHEMA_SHA256,
    ):
        _fail("blocked_v1_status_rejects_unfrozen_cli_overrides")
    unavailable_values: Sequence[object] = (
        namespace.tokenizer_binding,
        namespace.tokenizer_binding_sha256,
        namespace.toy_attestation,
        namespace.toy_attestation_sha256,
        namespace.formal_release_lock,
        namespace.formal_release_lock_sha256,
    )
    if any(value is not None for value in unavailable_values):
        _fail("blocked_v1_status_rejects_unfrozen_cli_overrides")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli_overrides(args)
    decision = evaluate_prerequisites(args.config)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
