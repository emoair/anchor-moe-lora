"""One-way release-DAG validation for planner-first 22,200 outputs.

The release chain is intentionally content-addressed in one direction:

    Teacher FINAL -> Producer attestation -> Consumer acceptance
    -> Training-input authority

Each later document may pin an earlier document's physical bytes.  An earlier
document must never carry the later document's physical SHA, which would make
the two artifacts mutually unbuildable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from .planner_first_coding_specialist_22200_v1 import (
    CampaignValidationError,
    canonical_json,
    read_regular_utf8_file,
)


TEACHER_FINAL_SCHEMA_VERSION = "anchor.planner-first-22200-teacher-final.v1"
PRODUCER_ATTESTATION_SCHEMA_VERSION = (
    "anchor.planner-first-22200-producer-attestation.v1"
)
CONSUMER_ACCEPTANCE_SCHEMA_VERSION = "anchor.planner-first-22200-consumer-acceptance.v1"
TRAINING_AUTHORITY_SCHEMA_VERSION = (
    "anchor.planner-first-22200-training-input-authority.v1"
)
HASH_DAG_ORDER = (
    TEACHER_FINAL_SCHEMA_VERSION,
    PRODUCER_ATTESTATION_SCHEMA_VERSION,
    CONSUMER_ACCEPTANCE_SCHEMA_VERSION,
    TRAINING_AUTHORITY_SCHEMA_VERSION,
)


class ReleaseValidationError(CampaignValidationError):
    """A release handoff has an invalid stage, identity, or hash edge."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(f"invalid_json:{label}") from error
    if not isinstance(value, Mapping):
        raise ReleaseValidationError(f"object_required:{label}")
    return {str(key): item for key, item in value.items()}


def _identity(path: Path) -> dict[str, object]:
    raw = read_regular_utf8_file(path)
    return {
        "bytes": len(raw),
        "path": path.as_posix(),
        "sha256": sha256_bytes(raw),
    }


def _load_schema(path: Path) -> dict[str, Any]:
    schema = _mapping(read_regular_utf8_file(path), label="schema")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise ReleaseValidationError(f"invalid_schema:{path.name}") from error
    return schema


def validate_document(path: str | Path, schema_path: str | Path) -> dict[str, Any]:
    document_path = Path(path)
    document = _mapping(read_regular_utf8_file(document_path), label=document_path.name)
    schema = _load_schema(Path(schema_path))
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as error:
        raise ReleaseValidationError(
            f"schema_validation_failed:{document_path.name}"
        ) from error
    return document


def _require_identity(
    actual: Mapping[str, object], declared: object, *, label: str
) -> None:
    if not isinstance(declared, Mapping):
        raise ReleaseValidationError(f"identity_object_required:{label}")
    normalized = {str(key): item for key, item in declared.items()}
    if normalized != actual:
        raise ReleaseValidationError(f"physical_identity_mismatch:{label}")


def _forbid_later_physical_sha(document: Mapping[str, Any], *, stage: str) -> None:
    encoded = canonical_json(document).decode("utf-8")
    forbidden_by_stage = {
        "teacher_final": (
            "producer_attestation_identity",
            "consumer_acceptance_identity",
            "training_input_authority_identity",
        ),
        "producer_attestation": (
            "consumer_acceptance_identity",
            "training_input_authority_identity",
        ),
        "consumer_acceptance": ("training_input_authority_identity",),
    }
    for forbidden in forbidden_by_stage.get(stage, ()):
        if forbidden in encoded:
            raise ReleaseValidationError(
                f"later_physical_sha_backlink_forbidden:{stage}"
            )


def validate_release_chain(
    *,
    teacher_final_path: str | Path,
    teacher_final_schema_path: str | Path,
    producer_attestation_path: str | Path,
    producer_attestation_schema_path: str | Path,
    consumer_acceptance_path: str | Path,
    consumer_acceptance_schema_path: str | Path,
    training_authority_path: str | Path,
    training_authority_schema_path: str | Path,
) -> dict[str, object]:
    """Validate the full one-way chain without interpreting training bodies."""

    teacher_path = Path(teacher_final_path)
    attestation_path = Path(producer_attestation_path)
    acceptance_path = Path(consumer_acceptance_path)
    authority_path = Path(training_authority_path)
    teacher = validate_document(teacher_path, teacher_final_schema_path)
    attestation = validate_document(attestation_path, producer_attestation_schema_path)
    acceptance = validate_document(acceptance_path, consumer_acceptance_schema_path)
    authority = validate_document(authority_path, training_authority_schema_path)
    _forbid_later_physical_sha(teacher, stage="teacher_final")
    _forbid_later_physical_sha(attestation, stage="producer_attestation")
    _forbid_later_physical_sha(acceptance, stage="consumer_acceptance")
    teacher_identity = _identity(teacher_path)
    attestation_identity = _identity(attestation_path)
    acceptance_identity = _identity(acceptance_path)
    _require_identity(
        teacher_identity,
        attestation.get("teacher_final_identity"),
        label="teacher_final",
    )
    _require_identity(
        teacher_identity,
        acceptance.get("teacher_final_identity"),
        label="teacher_final",
    )
    _require_identity(
        attestation_identity,
        acceptance.get("producer_attestation_identity"),
        label="producer_attestation",
    )
    _require_identity(
        acceptance_identity,
        authority.get("consumer_acceptance_identity"),
        label="consumer_acceptance",
    )
    semantic_root = teacher.get("semantic_root_sha256")
    if not isinstance(semantic_root, str):
        raise ReleaseValidationError("teacher_final_semantic_root_missing")
    inventories = teacher.get("inventories")
    if not isinstance(inventories, Mapping):
        raise ReleaseValidationError("teacher_final_inventories_missing")
    try:
        terminal_count = sum(
            int(inventories[name]["count"])
            for name in ("accepted", "rejected", "quarantine")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseValidationError(
            "teacher_final_inventory_counts_invalid"
        ) from error
    if (
        teacher.get("requested_object_count") != 22_200
        or teacher.get("terminal_object_count") != 22_200
        or terminal_count != 22_200
    ):
        raise ReleaseValidationError("teacher_final_inventory_closure_mismatch")
    if acceptance.get("allowed_assets") != teacher.get("ordered_assets"):
        raise ReleaseValidationError("consumer_allowed_assets_teacher_final_mismatch")
    for label, document in (
        ("producer_attestation", attestation),
        ("consumer_acceptance", acceptance),
        ("training_input_authority", authority),
    ):
        if document.get("teacher_final_semantic_root_sha256") != semantic_root:
            raise ReleaseValidationError(f"semantic_root_lineage_mismatch:{label}")
    if (
        acceptance.get("final") is not True
        or acceptance.get("training_authorized") is not True
    ):
        raise ReleaseValidationError("consumer_acceptance_must_authorize_training")
    if acceptance.get("consumed") is not False:
        raise ReleaseValidationError("consumer_acceptance_must_be_unconsumed")
    if (
        authority.get("training_authorized") is not True
        or authority.get("consumed") is not False
    ):
        raise ReleaseValidationError("training_authority_not_ready")
    return {
        "hash_dag_order": list(HASH_DAG_ORDER),
        "teacher_final_identity": teacher_identity,
        "producer_attestation_identity": attestation_identity,
        "consumer_acceptance_identity": acceptance_identity,
        "training_input_authority_identity": _identity(authority_path),
        "semantic_root_sha256": semantic_root,
        "training_authorized": True,
        "consumed": False,
    }
