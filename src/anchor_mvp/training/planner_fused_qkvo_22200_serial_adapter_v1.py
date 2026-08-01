"""Strict model-free serial gate for the planner-fused 22,200 architecture.

The legacy 3,520 multi-arm runner remains untouched.  This additive adapter
admits only the current planner lineage:

    3,200 source admission -> rank-64 Q+K+V+O planner train/gate/merge/fuse
    -> immutable 19,000 question+route commit -> six Q+O projections/leaves
    -> six independent packages -> fused-base runtime eligibility.

It authenticates metadata only.  It never opens dataset bodies, starts a
provider, loads a model, or allocates a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


ROOT: Final = Path(__file__).resolve().parents[3]
CONFIG_PATH: Final = (
    ROOT / "configs/training/planner_fused_qkvo_22200_serial_adapter_v1.json"
)
CONFIG_SCHEMA_PATH: Final = (
    ROOT / "configs/training/planner_fused_qkvo_22200_serial_adapter_v1.schema.json"
)
SOURCE_SCHEMA_PATH: Final = (
    ROOT / "configs/training/planner_fused_qkvo_22200_source_admission_v1.schema.json"
)
LINEAGE_SCHEMA_PATH: Final = (
    ROOT / "configs/training/planner_fused_qkvo_22200_lineage_v1.schema.json"
)
ROUTE_SCHEMA_PATH: Final = (
    ROOT / "configs/training/planner_fused_qkvo_22200_route_commit_v1.schema.json"
)
PROJECTION_SCHEMA_PATH: Final = (
    ROOT
    / "configs/training/planner_fused_qkvo_22200_six_expert_projection_v1.schema.json"
)
PACKAGE_SCHEMA_PATH: Final = (
    ROOT / "configs/training/planner_fused_qkvo_22200_expert_packages_v1.schema.json"
)

CONFIG_VERSION: Final = "anchor.planner-fused-qkvo-22200-serial-adapter-config.v1"
SOURCE_VERSION: Final = "anchor.planner-fused-qkvo-22200-source-admission.v1"
LINEAGE_VERSION: Final = "anchor.planner-fused-qkvo-22200-lineage.v1"
ROUTE_VERSION: Final = "anchor.planner-fused-qkvo-22200-route-commit.v1"
PROJECTION_VERSION: Final = "anchor.planner-fused-qkvo-22200-six-expert-projection.v1"
PACKAGE_VERSION: Final = "anchor.planner-fused-qkvo-22200-expert-packages.v1"
DECISION_VERSION: Final = "anchor.planner-fused-qkvo-22200-serial-adapter-decision.v1"

STAGE_ORDER: Final = (
    "source_admission",
    "planner_qkvo_train",
    "planner_independent_gate_eval",
    "planner_merge_hf_base_token_equivalent_overlay",
    "planner_compile_fused_q8",
    "immutable_question_route_commit",
    "six_expert_data_projection",
    "six_expert_q_plus_o_serial_train",
    "six_expert_independent_eval_q8_package",
    "fused_base_route_head_final_runtime",
)
OBSOLETE_PLANNER_AUTHORITIES: Final = frozenset(
    {
        "planner_q_only",
        "planner_q_plus_o",
        "planner_q_plus_micro_o",
        "original_base",
        "v14",
        "quarantine",
    }
)
PLANNER_BOOTSTRAP_COUNT: Final = 3_200
LEGACY_CODING_COUNT: Final = 19_000
EXPERT_COUNT: Final = 6
_REPARSE_POINT: Final = 0x400
SIDECAR_VERSION: Final = "anchor.planner-fused-qkvo-22200-physical-sidecar.v1"


class PlannerFusedSerialError(ValueError):
    """A physical metadata chain does not satisfy the current architecture."""


@dataclass(frozen=True)
class Document:
    path: Path
    raw: bytes
    value: Mapping[str, Any]

    @property
    def identity(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "bytes": len(self.raw),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }

    def artifact_identity_under(self, trusted_root: str | Path) -> dict[str, object]:
        """Return the sidecar-bound identity after a lexical trust-root check."""

        return _physical_artifact_identity(
            self.path, self.raw, trusted_root=trusted_root
        )


def _fail(code: str) -> None:
    raise PlannerFusedSerialError(code)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, code: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _sha(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code)
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _is_reparse(value: os.stat_result) -> bool:
    return bool(int(getattr(value, "st_file_attributes", 0)) & _REPARSE_POINT)


def _reject_reparse_chain(*, trusted_root: Path, candidate: Path) -> None:
    """Require a lexical descendant of a non-reparse trusted root."""

    root = trusted_root.absolute()
    target = candidate.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise PlannerFusedSerialError(
            "authenticated_path_outside_trusted_root"
        ) from error
    try:
        root_stat = root.lstat()
    except FileNotFoundError as error:
        raise PlannerFusedSerialError("trusted_root_missing") from error
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or _is_reparse(root_stat)
    ):
        _fail("trusted_root_not_regular_directory")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            observed = current.lstat()
        except FileNotFoundError as error:
            raise PlannerFusedSerialError("authenticated_path_missing") from error
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            _fail("authenticated_path_reparse_forbidden")


def read_regular_utf8_lf(
    path: str | Path, *, trusted_root: str | Path | None = None
) -> bytes:
    """Read exact regular-file bytes without following a reparse/symlink."""

    candidate = Path(path)
    if trusted_root is not None:
        _reject_reparse_chain(trusted_root=Path(trusted_root), candidate=candidate)
    try:
        lexical_before = candidate.lstat()
    except FileNotFoundError as error:
        raise PlannerFusedSerialError("authenticated_file_missing") from error
    if (
        not stat.S_ISREG(lexical_before.st_mode)
        or stat.S_ISLNK(lexical_before.st_mode)
        or _is_reparse(lexical_before)
    ):
        _fail("authenticated_file_not_regular")
    with candidate.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        if _stat_identity(opened_before) != _stat_identity(lexical_before):
            _fail("authenticated_file_open_identity_drift")
        raw = handle.read()
        opened_after = os.fstat(handle.fileno())
    try:
        lexical_after = candidate.lstat()
    except FileNotFoundError as error:
        raise PlannerFusedSerialError("authenticated_file_final_missing") from error
    if not (
        _stat_identity(lexical_before)
        == _stat_identity(opened_before)
        == _stat_identity(opened_after)
        == _stat_identity(lexical_after)
    ):
        _fail("authenticated_file_identity_drift")
    if len(raw) != lexical_after.st_size:
        _fail("authenticated_file_size_drift")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        _fail("authenticated_file_utf8_lf_required")
    try:
        raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise PlannerFusedSerialError("authenticated_file_utf8_required") from error
    return raw


def _document(
    path: str | Path,
    *,
    label: str,
    trusted_root: str | Path | None = None,
) -> Document:
    candidate = Path(path)
    raw = read_regular_utf8_lf(candidate, trusted_root=trusted_root)
    try:
        parsed = json.loads(raw.decode("utf-8", "strict"))
    except json.JSONDecodeError as error:
        raise PlannerFusedSerialError(f"invalid_json:{label}") from error
    return Document(candidate, raw, _mapping(parsed, f"object_required:{label}"))


def _physical_artifact_identity(
    path: Path, raw: bytes, *, trusted_root: str | Path | None = None
) -> dict[str, object]:
    """Return a file identity only when its derived sidecar closes the bytes."""

    sidecar_path = path.with_name(f"{path.name}.sha256")
    sidecar_raw = read_regular_utf8_lf(sidecar_path, trusted_root=trusted_root)
    try:
        sidecar = _mapping(
            json.loads(sidecar_raw.decode("utf-8", "strict")),
            "sidecar_object_required",
        )
    except json.JSONDecodeError as error:
        raise PlannerFusedSerialError("sidecar_invalid_json") from error
    expected = {
        "schema_version": SIDECAR_VERSION,
        "artifact_path": path.as_posix(),
        "artifact_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if sidecar != expected:
        _fail("artifact_sidecar_binding_mismatch")
    return {
        "path": path.as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sidecar_path": sidecar_path.as_posix(),
        "sidecar_bytes": len(sidecar_raw),
        "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
    }


def _artifact_from_identity(
    value: object, *, label: str, trusted_root: str | Path
) -> Document:
    """Safe-read a declared artifact and require its exact physical identity."""

    declared = _mapping(value, f"artifact_identity_required:{label}")
    expected_keys = {
        "path",
        "bytes",
        "sha256",
        "sidecar_path",
        "sidecar_bytes",
        "sidecar_sha256",
    }
    if set(declared) != expected_keys:
        _fail(f"artifact_identity_keys_invalid:{label}")
    candidate = Path(str(declared["path"]))
    document = _document(candidate, label=label, trusted_root=trusted_root)
    actual = _physical_artifact_identity(
        document.path, document.raw, trusted_root=trusted_root
    )
    if declared != actual:
        _fail(f"physical_artifact_identity_mismatch:{label}")
    return document


def _schema(path: Path, *, label: str) -> Mapping[str, Any]:
    schema = _document(path, label=label).value
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise PlannerFusedSerialError(f"invalid_schema:{label}") from error
    return schema


def _validate(document: Document, schema_path: Path, *, label: str) -> None:
    try:
        Draft202012Validator(_schema(schema_path, label=f"{label}_schema")).validate(
            document.value
        )
    except ValidationError as error:
        raise PlannerFusedSerialError(f"schema_validation_failed:{label}") from error


def _require_identity(
    document: Document,
    declared: object,
    *,
    label: str,
    trusted_root: str | Path,
) -> None:
    if _mapping(declared, f"identity_required:{label}") != _physical_artifact_identity(
        document.path, document.raw, trusted_root=trusted_root
    ):
        _fail(f"physical_identity_mismatch:{label}")


def _verify_metadata_manifest(
    value: object,
    *,
    expected_records: int,
    expected_semantic_root: object,
    label: str,
    trusted_root: str | Path,
) -> Document:
    manifest = _artifact_from_identity(value, label=label, trusted_root=trusted_root)
    if (
        manifest.value.get("records") != expected_records
        or manifest.value.get("semantic_root_sha256") != expected_semantic_root
    ):
        _fail(f"source_manifest_semantics_mismatch:{label}")
    return manifest


def _closed_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        _fail(f"closed_keys_invalid:{label}")


def _stage_artifact(
    value: object,
    *,
    label: str,
    schema_version: str,
    stage: str,
    expected_keys: set[str],
    trusted_root: str | Path,
) -> Document:
    document = _artifact_from_identity(value, label=label, trusted_root=trusted_root)
    _closed_keys(document.value, expected_keys, label=label)
    if (
        document.value.get("schema_version") != schema_version
        or document.value.get("stage") != stage
    ):
        _fail(f"stage_artifact_semantics_mismatch:{label}")
    return document


def load_config(path: str | Path = CONFIG_PATH) -> Document:
    document = _document(path, label="config")
    _validate(document, CONFIG_SCHEMA_PATH, label="config")
    if document.value.get("schema_version") != CONFIG_VERSION:
        _fail("config_schema_version_mismatch")
    if (
        tuple(
            _sequence(document.value.get("serial_stage_order"), "stage_order_invalid")
        )
        != STAGE_ORDER
    ):
        _fail("stage_order_invalid")
    return document


def load_source_admission(path: str | Path) -> Document:
    document = _document(path, label="source_admission")
    trusted_root = document.path.parent
    document.artifact_identity_under(trusted_root)
    _validate(document, SOURCE_SCHEMA_PATH, label="source_admission")
    if document.value.get("schema_version") != SOURCE_VERSION:
        _fail("source_schema_version_mismatch")
    planner = _mapping(document.value.get("planner_bootstrap"), "planner_asset_invalid")
    legacy = _mapping(document.value.get("legacy_coding"), "legacy_asset_invalid")
    if planner.get("records") != PLANNER_BOOTSTRAP_COUNT:
        _fail("planner_bootstrap_count_invalid")
    if legacy.get("records") != LEGACY_CODING_COUNT:
        _fail("legacy_coding_count_invalid")
    _verify_metadata_manifest(
        planner.get("manifest_identity"),
        expected_records=PLANNER_BOOTSTRAP_COUNT,
        expected_semantic_root=planner.get("semantic_root_sha256"),
        label="planner_bootstrap_manifest",
        trusted_root=trusted_root,
    )
    _verify_metadata_manifest(
        legacy.get("manifest_identity"),
        expected_records=LEGACY_CODING_COUNT,
        expected_semantic_root=legacy.get("semantic_root_sha256"),
        label="legacy_coding_manifest",
        trusted_root=trusted_root,
    )
    return document


def load_planner_lineage(path: str | Path, *, source_admission: Document) -> Document:
    trusted_root = source_admission.path.parent
    document = _document(path, label="planner_lineage", trusted_root=trusted_root)
    document.artifact_identity_under(trusted_root)
    _validate(document, LINEAGE_SCHEMA_PATH, label="planner_lineage")
    if document.value.get("schema_version") != LINEAGE_VERSION:
        _fail("planner_lineage_schema_version_mismatch")
    _require_identity(
        source_admission,
        document.value.get("source_admission_identity"),
        label="source_admission",
        trusted_root=trusted_root,
    )
    original_base = _stage_artifact(
        document.value.get("original_hf_base_identity"),
        label="original_hf_base",
        schema_version="anchor.planner-fused-qkvo-22200-original-hf-base.v1",
        stage="original_hf_base",
        expected_keys={
            "schema_version",
            "stage",
            "original_hf_base_sha256",
            "tokenizer_template_sha256",
        },
        trusted_root=trusted_root,
    )
    if original_base.value.get("original_hf_base_sha256") != document.value.get(
        "original_hf_base_sha256"
    ):
        _fail("original_hf_base_lineage_mismatch")
    adapter_artifact = _stage_artifact(
        document.value.get("planner_adapter_artifact_identity"),
        label="planner_qkvo_adapter_artifact",
        schema_version="anchor.planner-fused-qkvo-22200-planner-qkvo-adapter-artifact.v1",
        stage="planner_qkvo_adapter_artifact",
        expected_keys={
            "schema_version",
            "stage",
            "original_hf_base_identity",
            "planner_arm",
            "rank",
            "target_modules",
            "learning_rate_ratio",
            "adapter_sha256",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        original_base,
        adapter_artifact.value.get("original_hf_base_identity"),
        label="planner_adapter_original_hf_base",
        trusted_root=trusted_root,
    )
    if (
        adapter_artifact.value.get("planner_arm") != "planner_qkvo_rank64"
        or adapter_artifact.value.get("rank") != 64
        or adapter_artifact.value.get("target_modules")
        != ["q_proj", "k_proj", "v_proj", "o_proj"]
        or adapter_artifact.value.get("learning_rate_ratio")
        != {"q": 1.0, "k": 0.2, "v": 0.1, "o": 0.01}
    ):
        _fail("planner_adapter_artifact_semantics_invalid")
    train = _stage_artifact(
        document.value.get("planner_qkvo_train_receipt_identity"),
        label="planner_qkvo_train",
        schema_version="anchor.planner-fused-qkvo-22200-planner-train-receipt.v1",
        stage="planner_qkvo_train",
        expected_keys={
            "schema_version",
            "stage",
            "source_admission_identity",
            "original_hf_base_identity",
            "planner_adapter_artifact_identity",
            "planner_arm",
            "rank",
            "target_modules",
            "learning_rate_ratio",
            "completed",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        source_admission,
        train.value.get("source_admission_identity"),
        label="planner_qkvo_train_source_admission",
        trusted_root=trusted_root,
    )
    _require_identity(
        original_base,
        train.value.get("original_hf_base_identity"),
        label="planner_qkvo_train_original_hf_base",
        trusted_root=trusted_root,
    )
    _require_identity(
        adapter_artifact,
        train.value.get("planner_adapter_artifact_identity"),
        label="planner_qkvo_train_adapter_artifact",
        trusted_root=trusted_root,
    )
    if (
        train.value.get("planner_arm") != "planner_qkvo_rank64"
        or train.value.get("rank") != 64
        or train.value.get("target_modules") != ["q_proj", "k_proj", "v_proj", "o_proj"]
        or train.value.get("learning_rate_ratio")
        != {"q": 1.0, "k": 0.2, "v": 0.1, "o": 0.01}
        or train.value.get("completed") is not True
    ):
        _fail("planner_qkvo_train_receipt_semantics_invalid")
    gate = _stage_artifact(
        document.value.get("planner_independent_gate_receipt_identity"),
        label="planner_independent_gate",
        schema_version="anchor.planner-fused-qkvo-22200-planner-independent-gate.v1",
        stage="planner_independent_gate_eval",
        expected_keys={
            "schema_version",
            "stage",
            "planner_qkvo_train_receipt_identity",
            "evaluator_identity_sha256",
            "producer_identity_sha256",
            "evaluation_summary_sha256",
            "independent_gate_eval_passed",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        train,
        gate.value.get("planner_qkvo_train_receipt_identity"),
        label="planner_gate_train_receipt",
        trusted_root=trusted_root,
    )
    if gate.value.get("independent_gate_eval_passed") is not True or _sha(
        gate.value.get("evaluator_identity_sha256"), "planner_gate_evaluator"
    ) == _sha(gate.value.get("producer_identity_sha256"), "planner_gate_producer"):
        _fail("planner_independent_gate_failed")
    merge = _stage_artifact(
        document.value.get("merged_hf_base_identity"),
        label="planner_merge",
        schema_version="anchor.planner-fused-qkvo-22200-planner-merge.v1",
        stage="planner_merge_hf_base_token_equivalent_overlay",
        expected_keys={
            "schema_version",
            "stage",
            "original_hf_base_identity",
            "planner_adapter_artifact_identity",
            "planner_independent_gate_receipt_identity",
            "merged_into_original_hf_base",
            "merged_hf_base_sha256",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        gate,
        merge.value.get("planner_independent_gate_receipt_identity"),
        label="planner_merge_gate_receipt",
        trusted_root=trusted_root,
    )
    _require_identity(
        original_base,
        merge.value.get("original_hf_base_identity"),
        label="planner_merge_original_hf_base",
        trusted_root=trusted_root,
    )
    _require_identity(
        adapter_artifact,
        merge.value.get("planner_adapter_artifact_identity"),
        label="planner_merge_adapter_artifact",
        trusted_root=trusted_root,
    )
    if merge.value.get("merged_into_original_hf_base") is not True:
        _fail("planner_merge_not_completed")
    overlay = _stage_artifact(
        document.value.get("token_equivalence_proof_identity"),
        label="planner_token_equivalence_proof",
        schema_version="anchor.planner-fused-qkvo-22200-token-equivalence-proof.v1",
        stage="planner_token_equivalence_proof",
        expected_keys={
            "schema_version",
            "stage",
            "merged_hf_base_identity",
            "token_equivalent_overlay_applied",
            "input_tokenization_digest",
            "output_tokenization_digest",
            "token_equivalence_passed",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        merge,
        overlay.value.get("merged_hf_base_identity"),
        label="planner_token_equivalence_merge",
        trusted_root=trusted_root,
    )
    if (
        overlay.value.get("token_equivalent_overlay_applied") is not True
        or overlay.value.get("token_equivalence_passed") is not True
        or _sha(
            overlay.value.get("input_tokenization_digest"),
            "token_equivalence_input_digest",
        )
        != _sha(
            overlay.value.get("output_tokenization_digest"),
            "token_equivalence_output_digest",
        )
    ):
        _fail("token_equivalent_overlay_not_applied")
    fused = _stage_artifact(
        document.value.get("fused_q8_artifact_identity"),
        label="planner_fused_q8",
        schema_version="anchor.planner-fused-qkvo-22200-fused-q8-artifact.v1",
        stage="planner_compile_fused_q8",
        expected_keys={
            "schema_version",
            "stage",
            "token_equivalence_proof_identity",
            "fused_q8_compiled",
            "fused_base_kind",
            "fused_base_sha256",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        overlay,
        fused.value.get("token_equivalence_proof_identity"),
        label="planner_fused_token_equivalence_proof",
        trusted_root=trusted_root,
    )
    if (
        fused.value.get("fused_q8_compiled") is not True
        or fused.value.get("fused_base_kind") != "planner_fused_q8"
        or fused.value.get("fused_base_sha256")
        != document.value.get("fused_base_sha256")
    ):
        _fail("planner_fused_q8_semantics_invalid")
    if document.value.get("planner_arm") in OBSOLETE_PLANNER_AUTHORITIES:
        _fail("obsolete_planner_authority_rejected")
    return document


def load_route_commit(
    path: str | Path, *, source_admission: Document, planner_lineage: Document
) -> Document:
    trusted_root = source_admission.path.parent
    document = _document(path, label="route_commit", trusted_root=trusted_root)
    document.artifact_identity_under(trusted_root)
    _validate(document, ROUTE_SCHEMA_PATH, label="route_commit")
    if document.value.get("schema_version") != ROUTE_VERSION:
        _fail("route_schema_version_mismatch")
    _require_identity(
        planner_lineage,
        document.value.get("planner_lineage_identity"),
        label="planner_lineage",
        trusted_root=trusted_root,
    )
    route_inventory = _stage_artifact(
        document.value.get("route_inventory_identity"),
        label="immutable_route_inventory",
        schema_version="anchor.planner-fused-qkvo-22200-immutable-route-inventory.v1",
        stage="immutable_question_route_commit",
        expected_keys={
            "schema_version",
            "stage",
            "planner_lineage_identity",
            "fused_base_sha256",
            "legacy_coding_semantic_root_sha256",
            "question_commit_count",
            "route_commit_count",
            "route_commit_root_sha256",
        },
        trusted_root=trusted_root,
    )
    _require_identity(
        planner_lineage,
        route_inventory.value.get("planner_lineage_identity"),
        label="route_inventory_planner_lineage",
        trusted_root=trusted_root,
    )
    legacy = _mapping(
        source_admission.value.get("legacy_coding"), "legacy_asset_invalid"
    )
    if (
        document.value.get("fused_base_sha256")
        != planner_lineage.value.get("fused_base_sha256")
        or document.value.get("legacy_coding_semantic_root_sha256")
        != legacy.get("semantic_root_sha256")
        or document.value.get("question_commit_count") != LEGACY_CODING_COUNT
        or document.value.get("route_commit_count") != LEGACY_CODING_COUNT
        or route_inventory.value.get("fused_base_sha256")
        != document.value.get("fused_base_sha256")
        or route_inventory.value.get("legacy_coding_semantic_root_sha256")
        != document.value.get("legacy_coding_semantic_root_sha256")
        or route_inventory.value.get("question_commit_count")
        != document.value.get("question_commit_count")
        or route_inventory.value.get("route_commit_count")
        != document.value.get("route_commit_count")
        or route_inventory.value.get("route_commit_root_sha256")
        != document.value.get("route_commit_root_sha256")
    ):
        _fail("route_commit_lineage_mismatch")
    return document


def load_projection(path: str | Path, *, route_commit: Document) -> Document:
    trusted_root = route_commit.path.parent
    document = _document(path, label="six_expert_projection", trusted_root=trusted_root)
    document.artifact_identity_under(trusted_root)
    _validate(document, PROJECTION_SCHEMA_PATH, label="six_expert_projection")
    if document.value.get("schema_version") != PROJECTION_VERSION:
        _fail("projection_schema_version_mismatch")
    _require_identity(
        route_commit,
        document.value.get("route_commit_identity"),
        label="route_commit",
        trusted_root=trusted_root,
    )
    if document.value.get("route_commit_root_sha256") != route_commit.value.get(
        "route_commit_root_sha256"
    ) or document.value.get("fused_base_sha256") != route_commit.value.get(
        "fused_base_sha256"
    ):
        _fail("projection_lineage_mismatch")
    projections = [
        _mapping(item, "projection_invalid")
        for item in _sequence(
            document.value.get("projections"), "projection_list_invalid"
        )
    ]
    expert_ids = [str(item.get("expert_id")) for item in projections]
    if len(projections) != EXPERT_COUNT or len(set(expert_ids)) != EXPERT_COUNT:
        _fail("six_distinct_expert_projections_required")
    if sum(int(item.get("records", 0)) for item in projections) != LEGACY_CODING_COUNT:
        _fail("projection_record_count_closure_mismatch")
    for projection in projections:
        expert_id = str(projection.get("expert_id"))
        inventory = _stage_artifact(
            projection.get("projection_inventory_identity"),
            label=f"expert_projection_inventory:{expert_id}",
            schema_version=(
                "anchor.planner-fused-qkvo-22200-expert-projection-inventory.v1"
            ),
            stage="six_expert_data_projection",
            expected_keys={
                "schema_version",
                "stage",
                "route_commit_identity",
                "fused_base_sha256",
                "expert_id",
                "records",
                "projection_root_sha256",
                "adapter",
                "o_learning_rate_ratio_to_q",
            },
            trusted_root=trusted_root,
        )
        _require_identity(
            route_commit,
            inventory.value.get("route_commit_identity"),
            label=f"expert_projection_route_commit:{expert_id}",
            trusted_root=trusted_root,
        )
        for field in (
            "fused_base_sha256",
            "expert_id",
            "records",
            "projection_root_sha256",
            "adapter",
            "o_learning_rate_ratio_to_q",
        ):
            expected = (
                document.value.get(field)
                if field == "fused_base_sha256"
                else projection.get(field)
            )
            if inventory.value.get(field) != expected:
                _fail(f"expert_projection_inventory_mismatch:{expert_id}:{field}")
    return document


def load_packages(path: str | Path, *, projection: Document) -> Document:
    trusted_root = projection.path.parent
    document = _document(path, label="expert_packages", trusted_root=trusted_root)
    document.artifact_identity_under(trusted_root)
    _validate(document, PACKAGE_SCHEMA_PATH, label="expert_packages")
    if document.value.get("schema_version") != PACKAGE_VERSION:
        _fail("package_schema_version_mismatch")
    _require_identity(
        projection,
        document.value.get("projection_identity"),
        label="expert_projection",
        trusted_root=trusted_root,
    )
    if document.value.get("fused_base_sha256") != projection.value.get(
        "fused_base_sha256"
    ):
        _fail("package_fused_base_mismatch")
    projections = [
        _mapping(item, "projection_invalid")
        for item in _sequence(
            projection.value.get("projections"), "projection_list_invalid"
        )
    ]
    expected_ids = {str(item["expert_id"]) for item in projections}
    packages = [
        _mapping(item, "package_invalid")
        for item in _sequence(document.value.get("packages"), "package_list_invalid")
    ]
    package_ids = {str(item.get("expert_id")) for item in packages}
    indexes = {item.get("serial_index") for item in packages}
    if package_ids != expected_ids or indexes != set(range(1, EXPERT_COUNT + 1)):
        _fail("expert_package_serial_assignment_mismatch")
    previous_q8: Document | None = None
    for package in sorted(packages, key=lambda item: int(item["serial_index"])):
        expert_id = str(package["expert_id"])
        train = _stage_artifact(
            package.get("train_receipt_identity"),
            label=f"expert_train:{expert_id}",
            schema_version="anchor.planner-fused-qkvo-22200-expert-q-plus-o-train-receipt.v1",
            stage="six_expert_q_plus_o_serial_train",
            expected_keys={
                "schema_version",
                "stage",
                "projection_identity",
                "expert_id",
                "adapter",
                "o_learning_rate_ratio_to_q",
                "previous_q8_package_identity",
                "completed",
            },
            trusted_root=trusted_root,
        )
        _require_identity(
            projection,
            train.value.get("projection_identity"),
            label=f"expert_train_projection:{expert_id}",
            trusted_root=trusted_root,
        )
        if (
            train.value.get("expert_id") != expert_id
            or train.value.get("adapter") != "q_plus_o_lora"
            or train.value.get("o_learning_rate_ratio_to_q") != 0.1
            or train.value.get("completed") is not True
        ):
            _fail(f"expert_train_semantics_invalid:{expert_id}")
        expected_previous: object = (
            None
            if previous_q8 is None
            else previous_q8.artifact_identity_under(trusted_root)
        )
        if train.value.get("previous_q8_package_identity") != expected_previous:
            _fail(f"expert_train_serial_predecessor_mismatch:{expert_id}")
        if package.get("previous_q8_package_identity") != expected_previous:
            _fail(f"expert_package_serial_predecessor_mismatch:{expert_id}")
        evaluation = _stage_artifact(
            package.get("independent_eval_receipt_identity"),
            label=f"expert_eval:{expert_id}",
            schema_version="anchor.planner-fused-qkvo-22200-expert-independent-eval.v1",
            stage="six_expert_independent_eval_q8_package",
            expected_keys={
                "schema_version",
                "stage",
                "train_receipt_identity",
                "expert_id",
                "evaluator_identity_sha256",
                "producer_identity_sha256",
                "evaluation_summary_sha256",
                "independent_eval_passed",
            },
            trusted_root=trusted_root,
        )
        _require_identity(
            train,
            evaluation.value.get("train_receipt_identity"),
            label=f"expert_eval_train:{expert_id}",
            trusted_root=trusted_root,
        )
        if (
            evaluation.value.get("expert_id") != expert_id
            or evaluation.value.get("independent_eval_passed") is not True
            or _sha(
                evaluation.value.get("evaluator_identity_sha256"),
                f"expert_eval_evaluator:{expert_id}",
            )
            == _sha(
                evaluation.value.get("producer_identity_sha256"),
                f"expert_eval_producer:{expert_id}",
            )
        ):
            _fail(f"expert_eval_semantics_invalid:{expert_id}")
        q8_package = _stage_artifact(
            package.get("q8_package_identity"),
            label=f"expert_q8_package:{expert_id}",
            schema_version="anchor.planner-fused-qkvo-22200-expert-q8-package.v1",
            stage="six_expert_independent_eval_q8_package",
            expected_keys={
                "schema_version",
                "stage",
                "independent_eval_receipt_identity",
                "expert_id",
                "q8_package_sha256",
            },
            trusted_root=trusted_root,
        )
        _require_identity(
            evaluation,
            q8_package.value.get("independent_eval_receipt_identity"),
            label=f"expert_q8_eval:{expert_id}",
            trusted_root=trusted_root,
        )
        if q8_package.value.get("expert_id") != expert_id:
            _fail(f"expert_q8_package_semantics_invalid:{expert_id}")
        previous_q8 = q8_package
    return document


def build_decision(
    *,
    config_path: str | Path = CONFIG_PATH,
    source_admission_path: str | Path | None = None,
    planner_lineage_path: str | Path | None = None,
    route_commit_path: str | Path | None = None,
    projection_path: str | Path | None = None,
    package_path: str | Path | None = None,
) -> dict[str, object]:
    """Emit an immutable metadata decision; this function is never live."""

    config = load_config(config_path)
    common: dict[str, object] = {
        "schema_version": DECISION_VERSION,
        "config_identity": config.identity,
        "serial_stage_order": list(STAGE_ORDER),
        "planner_primary_adapter": "qkvo_lora_rank64",
        "planner_authorizing_arm": "planner_qkvo_rank64",
        "expert_adapter": "q_plus_o_lora",
        "expert_o_learning_rate_ratio_to_q": 0.1,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_runs": 0,
        "training_authorized": False,
        "formal_training_authorized": False,
        "legacy_runtime_modified": False,
    }
    if source_admission_path is None:
        return {
            **common,
            "status": "blocked",
            "next_required_stage": "source_admission",
            "blockers": ["source_admission_required"],
            "expert_leaf_train_permitted": False,
        }
    source = load_source_admission(source_admission_path)
    trusted_root = source.path.parent
    common["source_admission_identity"] = source.artifact_identity_under(trusted_root)
    if planner_lineage_path is None:
        return {
            **common,
            "status": "blocked",
            "next_required_stage": "planner_qkvo_train_gate_merge_fuse",
            "blockers": ["planner_fused_qkvo_lineage_required"],
            "expert_leaf_train_permitted": False,
        }
    lineage = load_planner_lineage(planner_lineage_path, source_admission=source)
    common["planner_lineage_identity"] = lineage.artifact_identity_under(trusted_root)
    if route_commit_path is None:
        return {
            **common,
            "status": "blocked",
            "next_required_stage": "immutable_question_route_commit",
            "blockers": ["fused_base_route_commit_required"],
            "expert_leaf_train_permitted": False,
        }
    route = load_route_commit(
        route_commit_path, source_admission=source, planner_lineage=lineage
    )
    common["route_commit_identity"] = route.artifact_identity_under(trusted_root)
    if projection_path is None:
        return {
            **common,
            "status": "blocked",
            "next_required_stage": "six_expert_data_projection",
            "blockers": ["six_expert_projection_required"],
            "expert_leaf_train_permitted": False,
        }
    projection = load_projection(projection_path, route_commit=route)
    common["six_expert_projection_identity"] = projection.artifact_identity_under(
        trusted_root
    )
    if package_path is None:
        return {
            **common,
            "status": "model_free_six_expert_leaf_plan",
            "next_required_stage": "external_serial_leaf_executor",
            "blockers": [
                "six_expert_packages_required",
                "external_leaf_executor_not_integrated",
            ],
            "expert_leaf_train_permitted": True,
            "serial_concurrency": 1,
            "fused_base_route_head_required": True,
            "full_generation_kv_shared": False,
        }
    packages = load_packages(package_path, projection=projection)
    return {
        **common,
        "status": "model_free_fused_runtime_eligible",
        "next_required_stage": "external_fused_base_runtime",
        "blockers": ["external_runtime_not_integrated"],
        "expert_package_identity": packages.artifact_identity_under(trusted_root),
        "expert_leaf_train_permitted": True,
        "serial_concurrency": 1,
        "fused_base_route_head_required": True,
        "full_generation_kv_shared": False,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--source-admission")
    parser.add_argument("--planner-lineage")
    parser.add_argument("--route-commit")
    parser.add_argument("--projection")
    parser.add_argument("--packages")
    args = parser.parse_args(argv)
    try:
        decision = build_decision(
            config_path=args.config,
            source_admission_path=args.source_admission,
            planner_lineage_path=args.planner_lineage,
            route_commit_path=args.route_commit,
            projection_path=args.projection,
            package_path=args.packages,
        )
    except PlannerFusedSerialError as error:
        decision = {
            "schema_version": DECISION_VERSION,
            "status": "blocked",
            "blockers": [str(error)],
            "provider_requests": 0,
            "model_loads": 0,
            "gpu_runs": 0,
            "training_authorized": False,
            "formal_training_authorized": False,
        }
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["status"].startswith("model_free_") else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
