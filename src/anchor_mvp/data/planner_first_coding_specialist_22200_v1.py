"""Planner-first migration contract for the additive 22,200-object campaign.

This module deliberately contains no provider, model, or GPU entrypoint.  It
only authenticates body-free source manifests and enforces the serial handoff
that must precede a later teacher collection:

    source registration -> planner training -> planner freeze -> route commit
    -> expert projection -> final release

The legacy SWE-bench bank is represented as a candidate source, never as
already-authorized training data.  The 3,200 planner-bootstrap objects are a
separate required source; they cannot be substituted with an old 320/1,000
fixture or invented from counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping


CAMPAIGN_SCHEMA_VERSION = "anchor.planner-first-coding-specialist-22200.v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "anchor.planner-first-source-manifest.v1"
SOURCE_EVIDENCE_SCHEMA_VERSION = "anchor.planner-first-source-evidence.v1"
DECISION_SCHEMA_VERSION = "anchor.planner-first-coding-specialist-decision.v1"
HASH_HEX_LENGTH = 64
LEGACY_CANDIDATE_COUNT = 19_008
LEGACY_SELECTED_COUNT = 19_000
PLANNER_BOOTSTRAP_COUNT = 3_200
TOTAL_OBJECT_COUNT = LEGACY_SELECTED_COUNT + PLANNER_BOOTSTRAP_COUNT
SERIAL_STAGES = (
    "source_registration",
    "planner_train",
    "planner_freeze",
    "planner_question_commit",
    "expert_dataset_projection",
    "expert_leaf_release",
)
HASH_DAG = (
    "teacher_final",
    "producer_attestation",
    "consumer_acceptance",
    "training_input_authority",
)
FORBIDDEN_BODY_FIELDS = frozenset(
    {
        "answer",
        "body",
        "completion",
        "future",
        "gold",
        "heldout",
        "messages",
        "prompt",
        "raw_token_ids",
        "target",
    }
)
_REPARSE_POINT = 0x0400


class CampaignValidationError(ValueError):
    """Raised when a campaign claim cannot be authenticated fail-closed."""


def canonical_json(value: object) -> bytes:
    """Return the single byte representation used for content-addressed roots."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_reparse(stat_result: os.stat_result) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT)


def read_regular_utf8_file(path: str | Path) -> bytes:
    """Read one regular file without silently following a reparse/symlink swap."""

    candidate = Path(path)
    before = os.lstat(candidate)
    if stat.S_ISLNK(before.st_mode) or _is_reparse(before):
        raise CampaignValidationError(f"reparse_or_symlink_not_allowed:{candidate}")
    if not stat.S_ISREG(before.st_mode):
        raise CampaignValidationError(f"regular_file_required:{candidate}")
    with candidate.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise CampaignValidationError(f"open_identity_drift:{candidate}")
        data = handle.read()
        after_open = os.fstat(handle.fileno())
    after = os.lstat(candidate)
    if stat.S_ISLNK(after.st_mode) or _is_reparse(after):
        raise CampaignValidationError(f"reparse_or_symlink_not_allowed:{candidate}")
    if (after_open.st_dev, after_open.st_ino, after_open.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ) or (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise CampaignValidationError(f"path_identity_drift:{candidate}")
    try:
        data.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise CampaignValidationError(f"utf8_required:{candidate}") from error
    return data


def _mapping_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignValidationError(f"invalid_json:{label}") from error
    if not isinstance(value, Mapping):
        raise CampaignValidationError(f"object_required:{label}")
    return {str(key): item for key, item in value.items()}


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HASH_HEX_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_hash(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not _is_hash(value):
        raise CampaignValidationError(f"sha256_required:{key}")
    return str(value)


def _flat_relative_file(base: Path, value: object, *, label: str) -> Path:
    """Resolve a same-directory evidence path without following path tricks."""

    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        raise CampaignValidationError(f"flat_relative_evidence_path_required:{label}")
    parsed = PurePosixPath(value)
    if parsed.name != value or value in {".", ".."} or ":" in value:
        raise CampaignValidationError(f"flat_relative_evidence_path_required:{label}")
    parent_stat = os.lstat(base)
    if not stat.S_ISDIR(parent_stat.st_mode) or _is_reparse(parent_stat):
        raise CampaignValidationError(f"evidence_directory_identity_invalid:{label}")
    return base / value


def _read_physical_identity(
    base: Path, value: object, *, label: str
) -> tuple[dict[str, object], bytes]:
    if not isinstance(value, Mapping):
        raise CampaignValidationError(f"physical_identity_object_required:{label}")
    identity = {str(key): item for key, item in value.items()}
    if set(identity) != {"bytes", "path", "sha256"}:
        raise CampaignValidationError(
            f"physical_identity_closed_shape_required:{label}"
        )
    expected_bytes = identity.get("bytes")
    expected_sha = identity.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not _is_hash(expected_sha)
    ):
        raise CampaignValidationError(f"physical_identity_values_invalid:{label}")
    path = _flat_relative_file(base, identity.get("path"), label=label)
    raw = read_regular_utf8_file(path)
    if len(raw) != expected_bytes or sha256_bytes(raw) != expected_sha:
        raise CampaignValidationError(f"physical_identity_drift:{label}")
    return {
        "bytes": expected_bytes,
        "path": str(identity["path"]),
        "sha256": str(expected_sha),
    }, raw


def _verify_evidence_pair(
    base: Path,
    value: object,
    *,
    label: str,
    evidence_kind: str,
    subject_semantic_root_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CampaignValidationError(f"evidence_pair_object_required:{label}")
    pair = {str(key): item for key, item in value.items()}
    if set(pair) != {"artifact", "sidecar"}:
        raise CampaignValidationError(f"evidence_pair_closed_shape_required:{label}")
    artifact, artifact_raw = _read_physical_identity(
        base, pair["artifact"], label=f"{label}.artifact"
    )
    sidecar, sidecar_raw = _read_physical_identity(
        base, pair["sidecar"], label=f"{label}.sidecar"
    )
    sidecar_value = _mapping_from_bytes(sidecar_raw, label=f"{label}.sidecar")
    if sidecar_value != {"artifact_sha256": artifact["sha256"]}:
        raise CampaignValidationError(f"evidence_sidecar_binding_mismatch:{label}")
    artifact_value = _mapping_from_bytes(artifact_raw, label=f"{label}.artifact")
    if artifact_value != {
        "evidence_kind": evidence_kind,
        "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "subject_semantic_root_sha256": subject_semantic_root_sha256,
    }:
        raise CampaignValidationError(f"evidence_subject_binding_mismatch:{label}")
    return {"artifact": artifact, "sidecar": sidecar}


def _reject_body_fields(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(FORBIDDEN_BODY_FIELDS.intersection(map(str, value)))
        if forbidden:
            raise CampaignValidationError(
                f"body_field_forbidden:{label}:{','.join(forbidden)}"
            )
        for key, item in value.items():
            _reject_body_fields(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_body_fields(item, label=f"{label}[{index}]")


@dataclass(frozen=True)
class SourceManifest:
    source_kind: str
    object_count: int
    training_authorized: bool
    semantic_root_sha256: str
    manifest_sha256: str
    item_ids: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    evidence: dict[str, object]


@dataclass(frozen=True)
class PlannerEvidence:
    training_receipt_sha256: str
    freeze_receipt_sha256: str
    planner_adapter_sha256: str
    primary_arm: str
    frozen: bool


@dataclass(frozen=True)
class LegacySelection:
    """A deterministic 19,000-object selection from the public 19,008 pool."""

    selected_source: SourceManifest
    excluded_object_ids: tuple[str, ...]
    selection_manifest_sha256: str


def load_source_manifest(
    path: str | Path, *, expected_kind: str, expected_count: int
) -> SourceManifest:
    """Authenticate a body-free source inventory without loading training bodies."""

    raw = read_regular_utf8_file(path)
    manifest = _mapping_from_bytes(raw, label="source_manifest")
    _reject_body_fields(manifest, label="source_manifest")
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise CampaignValidationError("source_manifest_schema_version_mismatch")
    if manifest.get("source_kind") != expected_kind:
        raise CampaignValidationError("source_kind_mismatch")
    if manifest.get("content_access") != "metadata_only":
        raise CampaignValidationError("source_manifest_must_be_metadata_only")
    if manifest.get("object_count") != expected_count:
        raise CampaignValidationError("source_object_count_mismatch")
    _require_hash(manifest, "semantic_root_sha256")
    evidence_value = manifest.get("evidence")
    if not isinstance(evidence_value, Mapping):
        raise CampaignValidationError("source_evidence_object_required")
    evidence = {str(key): item for key, item in evidence_value.items()}
    required_evidence = (
        "real_tool_trajectory_inventory",
        "source_terms_acceptance",
        "translation_evidence",
    )
    if set(evidence) != set(required_evidence):
        raise CampaignValidationError("source_evidence_closed_shape_required")
    rows = manifest.get("objects")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise CampaignValidationError("source_object_inventory_mismatch")
    item_ids: list[str] = []
    canonical_rows: list[dict[str, str]] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise CampaignValidationError(f"source_row_object_required:{index}")
        row = {str(key): item for key, item in raw_row.items()}
        if set(row) != {
            "language",
            "source_object_id_sha256",
            "source_payload_sha256",
            "split",
        }:
            raise CampaignValidationError(f"source_row_closed_shape_required:{index}")
        if row.get("language") not in {"en", "zh-CN"} or row.get("split") not in {
            "train",
            "eval_proxy",
        }:
            raise CampaignValidationError(f"source_row_stratum_invalid:{index}")
        source_id = row.get("source_object_id_sha256")
        payload = row.get("source_payload_sha256")
        if not _is_hash(source_id) or not _is_hash(payload):
            raise CampaignValidationError(f"source_row_hash_required:{index}")
        item_ids.append(str(source_id))
        canonical_rows.append({key: str(row[key]) for key in sorted(row)})
    if len(set(item_ids)) != len(item_ids):
        raise CampaignValidationError("source_object_ids_must_be_unique")
    root = sha256_bytes(canonical_json(canonical_rows))
    if root != manifest["semantic_root_sha256"]:
        raise CampaignValidationError("source_semantic_root_mismatch")
    evidence_identities = {
        key: _verify_evidence_pair(
            Path(path).parent,
            evidence[key],
            label=f"source_evidence.{key}",
            evidence_kind=key,
            subject_semantic_root_sha256=root,
        )
        for key in required_evidence
    }
    return SourceManifest(
        source_kind=expected_kind,
        object_count=expected_count,
        training_authorized=manifest.get("training_authorized") is True,
        semantic_root_sha256=root,
        manifest_sha256=sha256_bytes(raw),
        item_ids=tuple(item_ids),
        rows=tuple(canonical_rows),
        evidence=evidence_identities,
    )


def load_legacy19000_selection_manifest(
    path: str | Path,
    *,
    candidate_pool: SourceManifest,
    expected_candidate_count: int = LEGACY_CANDIDATE_COUNT,
    expected_selected_count: int = LEGACY_SELECTED_COUNT,
) -> LegacySelection:
    """Verify a deterministic, complete 19,000-of-19,008 source selection.

    Lexicographic source-ID ordering is a population-selection rule only.  It
    does not claim task uniqueness, quality, or language equivalence.
    """

    if candidate_pool.source_kind != "legacy_coding_candidate_pool":
        raise CampaignValidationError("legacy_candidate_pool_kind_mismatch")
    if candidate_pool.object_count != expected_candidate_count:
        raise CampaignValidationError("legacy_candidate_pool_count_mismatch")
    raw = read_regular_utf8_file(path)
    manifest = _mapping_from_bytes(raw, label="legacy_selection_manifest")
    _reject_body_fields(manifest, label="legacy_selection_manifest")
    expected_excluded_count = expected_candidate_count - expected_selected_count
    if (
        manifest.get("schema_version")
        != "anchor.planner-first-legacy19000-selection.v1"
    ):
        raise CampaignValidationError("legacy_selection_schema_version_mismatch")
    if manifest.get("candidate_pool_manifest_sha256") != candidate_pool.manifest_sha256:
        raise CampaignValidationError("legacy_selection_candidate_manifest_mismatch")
    if (
        manifest.get("candidate_pool_semantic_root_sha256")
        != candidate_pool.semantic_root_sha256
    ):
        raise CampaignValidationError("legacy_selection_candidate_root_mismatch")
    if (
        manifest.get("candidate_pool_count") != expected_candidate_count
        or manifest.get("selected_object_count") != expected_selected_count
        or manifest.get("excluded_object_count") != expected_excluded_count
    ):
        raise CampaignValidationError("legacy_selection_count_mismatch")
    if (
        manifest.get("selection_rule")
        != "lexicographic_source_object_id_sha256_take_first"
    ):
        raise CampaignValidationError("legacy_selection_rule_mismatch")
    selection_evidence = manifest.get("selection_evidence")
    if not isinstance(selection_evidence, Mapping):
        raise CampaignValidationError("legacy_selection_evidence_object_required")
    selection_evidence = {str(key): item for key, item in selection_evidence.items()}
    required_selection_evidence = (
        "operation_authorization",
        "source_terms_acceptance",
    )
    if set(selection_evidence) != set(required_selection_evidence):
        raise CampaignValidationError("legacy_selection_evidence_closed_shape_required")
    if manifest.get("training_authorized") is not True:
        raise CampaignValidationError("legacy_selection_not_training_authorized")
    selected = manifest.get("selected_object_ids")
    excluded = manifest.get("excluded_object_ids")
    if not isinstance(selected, list) or not isinstance(excluded, list):
        raise CampaignValidationError("legacy_selection_id_arrays_required")
    selected_ids = tuple(str(item) for item in selected)
    excluded_ids = tuple(str(item) for item in excluded)
    if (
        len(selected_ids) != expected_selected_count
        or len(excluded_ids) != expected_excluded_count
        or any(not _is_hash(item) for item in (*selected_ids, *excluded_ids))
        or len(set((*selected_ids, *excluded_ids))) != expected_candidate_count
    ):
        raise CampaignValidationError("legacy_selection_id_inventory_invalid")
    ordered_pool = tuple(sorted(candidate_pool.item_ids))
    if (
        tuple(selected_ids) != ordered_pool[:expected_selected_count]
        or tuple(excluded_ids) != ordered_pool[expected_selected_count:]
    ):
        raise CampaignValidationError(
            "legacy_selection_not_deterministic_pool_partition"
        )
    if manifest.get("selected_root_sha256") != sha256_bytes(
        canonical_json(list(selected_ids))
    ):
        raise CampaignValidationError("legacy_selection_selected_root_mismatch")
    if manifest.get("excluded_root_sha256") != sha256_bytes(
        canonical_json(list(excluded_ids))
    ):
        raise CampaignValidationError("legacy_selection_excluded_root_mismatch")
    selected_root = sha256_bytes(canonical_json(list(selected_ids)))
    selection_evidence_subjects = {
        "operation_authorization": selected_root,
        "source_terms_acceptance": candidate_pool.semantic_root_sha256,
    }
    for key in required_selection_evidence:
        _verify_evidence_pair(
            Path(path).parent,
            selection_evidence[key],
            label=f"selection_evidence.{key}",
            evidence_kind=key,
            subject_semantic_root_sha256=selection_evidence_subjects[key],
        )
    selected_set = set(selected_ids)
    selected_rows = tuple(
        sorted(
            (
                row
                for row in candidate_pool.rows
                if row["source_object_id_sha256"] in selected_set
            ),
            key=lambda row: row["source_object_id_sha256"],
        )
    )
    if len(selected_rows) != expected_selected_count:
        raise CampaignValidationError("legacy_selection_selected_row_closure_mismatch")
    semantic_root = sha256_bytes(canonical_json(list(selected_rows)))
    return LegacySelection(
        selected_source=SourceManifest(
            source_kind="legacy_coding_19000",
            object_count=expected_selected_count,
            training_authorized=True,
            semantic_root_sha256=semantic_root,
            manifest_sha256=sha256_bytes(raw),
            item_ids=selected_ids,
            rows=selected_rows,
            evidence=candidate_pool.evidence,
        ),
        excluded_object_ids=excluded_ids,
        selection_manifest_sha256=sha256_bytes(raw),
    )


def _validate_hash_dag(contract: Mapping[str, Any]) -> None:
    dag = contract.get("release_hash_dag")
    if not isinstance(dag, Mapping) or tuple(dag.get("order", ())) != HASH_DAG:
        raise CampaignValidationError("release_hash_dag_order_mismatch")
    if dag.get("physical_sha_backlinks_forbidden") is not True:
        raise CampaignValidationError(
            "release_hash_dag_must_forbid_physical_sha_cycles"
        )
    if dag.get("teacher_final_may_contain_later_attestation_sha") is not False:
        raise CampaignValidationError("teacher_final_attestation_backlink_forbidden")


def load_campaign_contract(path: str | Path) -> dict[str, Any]:
    """Load the closed, non-live campaign plan and enforce its count semantics."""

    raw = read_regular_utf8_file(path)
    contract = _mapping_from_bytes(raw, label="campaign_contract")
    _reject_body_fields(contract, label="campaign_contract")
    if contract.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignValidationError("campaign_schema_version_mismatch")
    if (
        contract.get("strictly_serial") is not True
        or contract.get("max_concurrency") != 1
    ):
        raise CampaignValidationError("strict_serial_single_concurrency_required")
    if tuple(contract.get("stage_order", ())) != SERIAL_STAGES:
        raise CampaignValidationError("stage_order_mismatch")
    if contract.get("problem_object_count") != TOTAL_OBJECT_COUNT:
        raise CampaignValidationError("problem_object_total_mismatch")
    if contract.get("legacy_coding_object_count") != LEGACY_SELECTED_COUNT:
        raise CampaignValidationError("legacy_object_count_mismatch")
    if contract.get("planner_bootstrap_object_count") != PLANNER_BOOTSTRAP_COUNT:
        raise CampaignValidationError("planner_bootstrap_object_count_mismatch")
    if contract.get("legacy_candidate_pool_count") != LEGACY_CANDIDATE_COUNT:
        raise CampaignValidationError("legacy_candidate_pool_count_mismatch")
    if contract.get("legacy_source_is_training_data") is not False:
        raise CampaignValidationError(
            "legacy_source_must_not_be_claimed_as_training_data"
        )
    source_requirements = contract.get("source_authority_requirements")
    if not isinstance(source_requirements, Mapping) or source_requirements != {
        "legacy_selection_manifest_required": True,
        "planner_bootstrap_manifest_required": True,
        "real_tool_trajectory_required": True,
        "source_terms_acceptance_required": True,
        "translation_evidence_required": True,
    }:
        raise CampaignValidationError("source_authority_requirements_mismatch")
    if contract.get("provider_launch_authorized") is not False:
        raise CampaignValidationError("provider_launch_must_remain_false_until_release")
    if contract.get("training_authorized") is not False:
        raise CampaignValidationError(
            "training_authorized_must_remain_false_until_acceptance"
        )
    _validate_hash_dag(contract)
    return contract


def validate_planner_evidence(evidence: PlannerEvidence) -> None:
    if evidence.primary_arm != "planner_q_only" or not evidence.frozen:
        raise CampaignValidationError("frozen_planner_q_only_required")
    for value in (
        evidence.training_receipt_sha256,
        evidence.freeze_receipt_sha256,
        evidence.planner_adapter_sha256,
    ):
        if not _is_hash(value):
            raise CampaignValidationError("planner_evidence_sha256_required")


def expert_projection_decision(
    contract_path: str | Path,
    *,
    legacy_source: SourceManifest | None = None,
    legacy_selection: LegacySelection | None = None,
    planner_bootstrap_source: SourceManifest | None = None,
    planner_evidence: PlannerEvidence | None = None,
) -> dict[str, Any]:
    """Return a body-free serial decision; this function cannot launch a teacher."""

    contract = load_campaign_contract(contract_path)
    blockers: list[str] = []
    sources_ready = True
    if legacy_selection is None:
        blockers.append("legacy19000_deterministic_selection_manifest_missing")
        sources_ready = False
    else:
        if (
            legacy_source is not None
            and legacy_source.semantic_root_sha256
            != legacy_selection.selected_source.semantic_root_sha256
        ):
            raise CampaignValidationError(
                "legacy_source_and_selection_lineage_mismatch"
            )
        legacy_source = legacy_selection.selected_source
    if legacy_source is not None and (
        legacy_source.object_count != LEGACY_SELECTED_COUNT
        or not legacy_source.training_authorized
    ):
        blockers.append("legacy19000_source_not_training_authorized")
        sources_ready = False
    if planner_bootstrap_source is None:
        blockers.append("planner_bootstrap3200_source_manifest_missing")
        sources_ready = False
    elif (
        planner_bootstrap_source.object_count != PLANNER_BOOTSTRAP_COUNT
        or not planner_bootstrap_source.training_authorized
    ):
        blockers.append("planner_bootstrap3200_source_not_training_authorized")
        sources_ready = False
    if sources_ready and planner_evidence is None:
        blockers.append("planner_training_and_freeze_evidence_missing")
    if sources_ready and planner_evidence is not None:
        try:
            validate_planner_evidence(planner_evidence)
        except CampaignValidationError as error:
            blockers.append(str(error))
    next_stage = "source_registration"
    if sources_ready and planner_evidence is None:
        next_stage = "planner_train"
    elif sources_ready and not blockers:
        next_stage = "planner_question_commit"
    status = "ready_for_planner_question_commit" if not blockers else "blocked"
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "campaign_schema_version": contract["schema_version"],
        "status": status,
        "next_required_stage": next_stage,
        "blockers": blockers,
        "problem_object_count": TOTAL_OBJECT_COUNT,
        "legacy_coding_object_count": LEGACY_SELECTED_COUNT,
        "planner_bootstrap_object_count": PLANNER_BOOTSTRAP_COUNT,
        "expert_projection_permitted": status == "ready_for_planner_question_commit",
        "provider_launch_permitted": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "body_reads": 0,
        "raw_token_ids_read": 0,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_runs": 0,
    }


def validate_decision(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise CampaignValidationError("decision_schema_version_mismatch")
    if value.get("provider_launch_permitted") is not False:
        raise CampaignValidationError("decision_must_not_authorize_provider_launch")
    if value.get("training_authorized") is not False:
        raise CampaignValidationError("decision_must_not_authorize_training")
    if value.get("formal_training_authorized") is not False:
        raise CampaignValidationError("decision_must_not_authorize_formal_training")
