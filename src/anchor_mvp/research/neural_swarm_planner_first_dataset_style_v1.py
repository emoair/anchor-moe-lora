"""Body-free dataset-style companion for the Planner-first curriculum.

This module refines the already frozen Planner-first control plane without
changing the existing training runtime.  It models one semantic question as:

* exactly one primary style expert;
* an optional tool sidecar;
* an optional review sidecar; and
* an identity-preservation constraint that is not another output expert.

Every physical projection row still belongs to exactly one leaf expert, so the
companion is compatible with the parent contract's ``one_expert_per_question``
rule.  A bundle-level validator proves that the rows for a semantic question
are exactly the selected experts rather than every available candidate.

The module is model-free.  It never reads question/answer bodies, raw token
IDs, heldout bodies, credentials, or provider data, and it grants no training,
GPU, formal, or release authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence


CONFIG_VERSION: Final = "anchor.neural-swarm-planner-first-dataset-style.v1"
RECORD_VERSION: Final = "anchor.neural-swarm-planner-first-question-record.v2"
MANIFEST_VERSION: Final = "anchor.neural-swarm-planner-first-question-manifest.v2"

EXPERT_ASSETS: Final = (
    "humor",
    "serious",
    "angry_style",
    "tool_call",
    "review_audit",
)
STYLE_EXPERTS: Final = ("humor", "serious", "angry_style")
CAPABILITY_EXPERTS: Final = ("tool_call", "review_audit")
VARIANTS: Final = ("base", "with_review", "with_tool", "with_tool_review")
SPLITS: Final = ("train", "calibration", "eval_proxy")
LANGUAGES: Final = ("en", "zh-CN")
IDENTITY_CLASSES: Final = ("air", "false_google", "false_openai")
HARD_NEGATIVE_KINDS: Final = (
    "wrong_primary_style",
    "missing_required_tool",
    "unnecessary_tool",
    "missing_required_review",
    "unnecessary_review",
    "identity_bit_dropped",
    "identity_false_attribution",
)
CANDIDATE_EXPERT_ASSETS: Final = EXPERT_ASSETS
FALSE_CLAIMS: Final = (
    "existing_training_runtime_modified",
    "records_materialized",
    "planner_question_generation_executed",
    "planner_first_leaf_launcher_integrated",
    "expert_training_authorized",
    "gpu_authorized",
    "training_authorized",
    "formal",
    "release_authorized",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
REPARSE_POINT = 0x400

_VARIANT_FLAGS: Final = {
    "base": (False, False),
    "with_review": (False, True),
    "with_tool": (True, False),
    "with_tool_review": (True, True),
}
_NEGATIVE_RULES: Final = {
    "wrong_primary_style": (
        "style_expert",
        "route_primary_style_mismatch",
    ),
    "missing_required_tool": (
        "tool_required",
        "route_required_tool_missing",
    ),
    "unnecessary_tool": (
        "tool_required",
        "route_unnecessary_tool_selected",
    ),
    "missing_required_review": (
        "review_required",
        "route_required_review_missing",
    ),
    "unnecessary_review": (
        "review_required",
        "route_unnecessary_review_selected",
    ),
    "identity_bit_dropped": (
        "identity_required",
        "route_identity_requirement_dropped",
    ),
    "identity_false_attribution": (
        "identity_provenance",
        "route_identity_invariant_mismatch",
    ),
}


class PlannerFirstDatasetStyleError(ValueError):
    """Raised when a dataset-style identity or invariant fails."""


def _fail(code: str) -> None:
    raise PlannerFirstDatasetStyleError(code)


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _closed_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 representation used by all identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _is_reparse(value: os.stat_result) -> bool:
    return bool(int(getattr(value, "st_file_attributes", 0)) & REPARSE_POINT)


def _reject_reparse_chain(repo_root: Path, path: Path) -> None:
    current = repo_root
    for part in path.relative_to(repo_root).parts:
        current = current / part
        try:
            identity = current.lstat()
        except FileNotFoundError:
            _fail("authenticated_path_missing")
        if stat.S_ISLNK(identity.st_mode) or _is_reparse(identity):
            _fail("authenticated_path_reparse_forbidden")


def _repo_file(repo_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail("repo_path_invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail("repo_path_invalid")
    root = repo_root.resolve(strict=True)
    path = root.joinpath(*pure.parts)
    _reject_reparse_chain(root, path)
    try:
        lexical = path.lstat()
    except FileNotFoundError:
        _fail("repo_file_missing")
    if not stat.S_ISREG(lexical.st_mode):
        _fail("repo_file_not_regular")
    return path


def _stable_read(path: Path) -> bytes:
    try:
        lexical_before = path.lstat()
    except FileNotFoundError:
        _fail("authenticated_file_missing")
    if (
        not stat.S_ISREG(lexical_before.st_mode)
        or stat.S_ISLNK(lexical_before.st_mode)
        or _is_reparse(lexical_before)
    ):
        _fail("authenticated_file_not_regular")
    with path.open("rb") as handle:
        handle_before = os.fstat(handle.fileno())
        if _identity(handle_before) != _identity(lexical_before):
            _fail("authenticated_file_open_identity_drift")
        payload = handle.read()
        handle_after = os.fstat(handle.fileno())
    try:
        lexical_after = path.lstat()
    except FileNotFoundError:
        _fail("authenticated_file_final_identity_missing")
    if not (
        _identity(lexical_before)
        == _identity(handle_before)
        == _identity(handle_after)
        == _identity(lexical_after)
    ):
        _fail("authenticated_file_identity_drift")
    if len(payload) != lexical_before.st_size:
        _fail("authenticated_file_size_drift")
    return payload


def _load_json_file(path: Path) -> Mapping[str, Any]:
    payload = _stable_read(path)
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError:
        _fail("json_utf8_invalid")
    if "\r" in decoded:
        _fail("json_lf_required")
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError:
        _fail("json_parse_failed")
    return _mapping(value, "json_root_invalid")


def _verify_file_pin(repo_root: Path, value: Any, code: str) -> None:
    pin = _mapping(value, f"{code}_invalid")
    _closed_keys(pin, {"path", "sha256"}, f"{code}_keys_invalid")
    expected_sha = _sha(pin.get("sha256"), f"{code}_sha256_invalid")
    path = _repo_file(repo_root, str(pin.get("path")))
    payload = _stable_read(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        _fail(f"{code}_identity_drift")


def validate_config(config: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Validate the additive contract and its frozen parent file pins."""

    expected = {
        "$schema",
        "schema_version",
        "status",
        "claim_scope",
        "parent_contract",
        "read_only_training_evidence",
        "dataset_semantics",
        "partition_policy",
        "hard_negative_policy",
        "claims",
    }
    _closed_keys(config, expected, "dataset_style_config_keys_invalid")
    if (
        config.get("$schema")
        != "./neural_swarm_planner_first_dataset_style_v1.schema.json"
        or config.get("schema_version") != CONFIG_VERSION
        or config.get("status") != "additive_model_free_dataset_style_contract_only"
        or config.get("claim_scope")
        != "planner_first_question_and_expert_projection_metadata_only"
    ):
        _fail("dataset_style_config_semantics_invalid")

    parent = _mapping(config.get("parent_contract"), "parent_contract_invalid")
    _closed_keys(
        parent,
        {
            "git_commit",
            "git_tree",
            "contract",
            "contract_schema",
            "implementation",
        },
        "parent_contract_keys_invalid",
    )
    if (
        not isinstance(parent.get("git_commit"), str)
        or GIT_OBJECT_RE.fullmatch(str(parent.get("git_commit"))) is None
        or not isinstance(parent.get("git_tree"), str)
        or GIT_OBJECT_RE.fullmatch(str(parent.get("git_tree"))) is None
    ):
        _fail("parent_git_identity_invalid")
    _verify_file_pin(repo_root, parent.get("contract"), "parent_contract")
    _verify_file_pin(repo_root, parent.get("contract_schema"), "parent_contract_schema")
    _verify_file_pin(repo_root, parent.get("implementation"), "parent_implementation")

    evidence = _mapping(
        config.get("read_only_training_evidence"),
        "read_only_training_evidence_invalid",
    )
    _closed_keys(
        evidence,
        {
            "source",
            "latest_training_receipt_sha256",
            "latest_failed_attempt_receipt_sha256",
            "latest_dataset_records",
            "historical_invalidated_record_ids",
            "historical_invalidated_normalized_prompts",
            "dataset_gate_passed",
            "training_stage_passed",
            "latest_attempt_consumed",
            "latest_attempt_failure_class",
            "holdout_body_read",
            "training_worktree_modified",
        },
        "read_only_training_evidence_keys_invalid",
    )
    _sha(
        evidence.get("latest_training_receipt_sha256"),
        "training_receipt_identity_invalid",
    )
    _sha(
        evidence.get("latest_failed_attempt_receipt_sha256"),
        "failed_attempt_receipt_identity_invalid",
    )
    if (
        evidence.get("source") != "codex_training_thread_body_free_receipts_only"
        or evidence.get("dataset_gate_passed") is not True
        or evidence.get("training_stage_passed") is not True
        or evidence.get("latest_attempt_consumed") is not False
        or evidence.get("latest_attempt_failure_class")
        != "execution_provenance_not_dataset_quality"
        or evidence.get("holdout_body_read") is not False
        or evidence.get("training_worktree_modified") is not False
    ):
        _fail("read_only_training_evidence_semantics_invalid")
    for key in (
        "latest_dataset_records",
        "historical_invalidated_record_ids",
        "historical_invalidated_normalized_prompts",
    ):
        number = evidence.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            _fail(f"{key}_invalid")

    semantics = _mapping(config.get("dataset_semantics"), "dataset_semantics_invalid")
    _closed_keys(
        semantics,
        {
            "expert_assets",
            "style_experts",
            "capability_experts",
            "route_variants",
            "identity_provenance_classes",
            "candidate_inventory_is_activation_set",
            "one_training_asset_per_record",
            "bundle_projection_must_equal_selected_experts",
            "route_plan_is_bundle_level",
            "split_before_projection",
            "planner_freeze_before_question_materialization",
            "planner_reentry_after_projection",
            "tool_and_review_are_capability_branches_not_style_labels",
            "identity_is_semantic_constraint_not_sixth_output_expert",
        },
        "dataset_semantics_keys_invalid",
    )
    if (
        tuple(semantics.get("expert_assets", ())) != EXPERT_ASSETS
        or tuple(semantics.get("style_experts", ())) != STYLE_EXPERTS
        or tuple(semantics.get("capability_experts", ())) != CAPABILITY_EXPERTS
        or tuple(semantics.get("route_variants", ())) != VARIANTS
        or tuple(semantics.get("identity_provenance_classes", ())) != IDENTITY_CLASSES
    ):
        _fail("dataset_semantics_inventory_mismatch")
    required_true = (
        "one_training_asset_per_record",
        "bundle_projection_must_equal_selected_experts",
        "route_plan_is_bundle_level",
        "split_before_projection",
        "planner_freeze_before_question_materialization",
        "tool_and_review_are_capability_branches_not_style_labels",
        "identity_is_semantic_constraint_not_sixth_output_expert",
    )
    if any(semantics.get(key) is not True for key in required_true):
        _fail("dataset_semantics_required_true_missing")
    if (
        semantics.get("candidate_inventory_is_activation_set") is not False
        or semantics.get("planner_reentry_after_projection") is not False
    ):
        _fail("dataset_semantics_forbidden_claim")

    partition = _mapping(config.get("partition_policy"), "partition_policy_invalid")
    _closed_keys(
        partition,
        {
            "splits",
            "optimization_splits",
            "calibration_splits",
            "evaluation_splits",
            "split_group_keys",
            "near_duplicate_algorithm",
            "near_duplicate_threshold",
            "historical_invalidated_inventory_required",
            "heldout_rows_permitted",
            "heldout_inventory_only",
            "holdout_used_for_style_tuning",
            "holdout_reused_after_evaluation",
        },
        "partition_policy_keys_invalid",
    )
    if (
        tuple(partition.get("splits", ())) != SPLITS
        or tuple(partition.get("optimization_splits", ())) != ("train",)
        or tuple(partition.get("calibration_splits", ())) != ("calibration",)
        or tuple(partition.get("evaluation_splits", ())) != ("eval_proxy",)
        or tuple(partition.get("split_group_keys", ()))
        != (
            "task_bundle_sha256",
            "source_group_sha256",
            "leakage_family_sha256",
            "prompt_digest_sha256",
        )
        or partition.get("near_duplicate_algorithm") != "nfkc_casefold_word_qgram5_dice"
        or partition.get("near_duplicate_threshold") != 0.9
        or partition.get("historical_invalidated_inventory_required") is not True
        or partition.get("heldout_rows_permitted") is not False
        or partition.get("heldout_inventory_only") is not True
        or partition.get("holdout_used_for_style_tuning") is not False
        or partition.get("holdout_reused_after_evaluation") is not False
    ):
        _fail("partition_policy_semantics_invalid")

    negatives = _mapping(
        config.get("hard_negative_policy"),
        "hard_negative_policy_invalid",
    )
    _closed_keys(
        negatives,
        {
            "kinds",
            "serious_identity_priority_variants",
            "with_tool_identity_positive_control_required",
            "plain_identity_fact_must_survive_style",
            "all_required_kinds_must_be_covered",
        },
        "hard_negative_policy_keys_invalid",
    )
    if (
        tuple(negatives.get("kinds", ())) != HARD_NEGATIVE_KINDS
        or tuple(negatives.get("serious_identity_priority_variants", ()))
        != ("base", "with_review", "with_tool_review")
        or negatives.get("with_tool_identity_positive_control_required") is not True
        or negatives.get("plain_identity_fact_must_survive_style") is not True
        or negatives.get("all_required_kinds_must_be_covered") is not True
    ):
        _fail("hard_negative_policy_semantics_invalid")

    claims = _mapping(config.get("claims"), "dataset_style_claims_invalid")
    _closed_keys(claims, set(FALSE_CLAIMS), "dataset_style_claims_keys_invalid")
    if any(claims.get(key) is not False for key in FALSE_CLAIMS):
        _fail("dataset_style_claim_scope_invalid")

    return {
        "status": "dataset_style_contract_valid_execution_blocked",
        "config_sha256": canonical_sha256(config),
        "parent_contract_sha256": parent["contract"]["sha256"],
        "existing_training_runtime_modified": False,
        "training_authorized": False,
        "formal": False,
    }


def _selected_assets(
    style_expert: str,
    *,
    tool_required: bool,
    review_required: bool,
) -> tuple[str, ...]:
    selected = [style_expert]
    if tool_required:
        selected.append("tool_call")
    if review_required:
        selected.append("review_audit")
    return tuple(selected)


def _build_route_plan(
    *,
    style_expert: str,
    tool_required: bool,
    review_required: bool,
    identity_required: bool,
    identity_provenance_class: str | None,
    identity_invariant_sha256: str | None,
) -> dict[str, Any]:
    if style_expert not in STYLE_EXPERTS:
        _fail("route_style_expert_invalid")
    if not all(
        isinstance(value, bool)
        for value in (tool_required, review_required, identity_required)
    ):
        _fail("route_boolean_invalid")
    if identity_required:
        if identity_provenance_class not in IDENTITY_CLASSES:
            _fail("identity_provenance_class_required")
        _sha(identity_invariant_sha256, "identity_invariant_sha256_required")
    elif identity_provenance_class is not None or identity_invariant_sha256 is not None:
        _fail("identity_metadata_forbidden_without_requirement")
    return {
        "candidate_expert_assets": list(CANDIDATE_EXPERT_ASSETS),
        "selected_expert_assets": list(
            _selected_assets(
                style_expert,
                tool_required=tool_required,
                review_required=review_required,
            )
        ),
        "style_expert": style_expert,
        "tool_required": tool_required,
        "review_required": review_required,
        "identity_required": identity_required,
        "identity_provenance_class": identity_provenance_class,
        "identity_invariant_sha256": identity_invariant_sha256,
        "planner_exit_after_commit": True,
    }


def _normalize_sha_list(value: Sequence[str], code: str) -> tuple[str, ...]:
    normalized = tuple(_sha(item, code) for item in _sequence(value, code))
    if len(set(normalized)) != len(normalized):
        _fail(f"{code}_duplicate")
    return normalized


def _build_context_segments(
    *,
    shared_prefix_segment_ids: Sequence[str],
    salient_segment_ids: Sequence[str],
    distractor_segment_ids: Sequence[str],
    route_relevant_evidence_refs: Sequence[str],
) -> dict[str, Any]:
    shared = _normalize_sha_list(
        shared_prefix_segment_ids,
        "shared_prefix_segment_id_invalid",
    )
    salient = _normalize_sha_list(salient_segment_ids, "salient_segment_id_invalid")
    distractor = _normalize_sha_list(
        distractor_segment_ids,
        "distractor_segment_id_invalid",
    )
    evidence = _normalize_sha_list(
        route_relevant_evidence_refs,
        "route_relevant_evidence_ref_invalid",
    )
    if not salient or not evidence:
        _fail("salient_and_evidence_required")
    if set(salient) & set(distractor):
        _fail("salient_distractor_overlap")
    if not set(evidence).issubset(set(salient)):
        _fail("route_evidence_not_salient")
    allowed = tuple(item for item in shared if item in set(salient))
    return {
        "shared_prefix_segment_ids": list(shared),
        "salient_segment_ids": list(salient),
        "distractor_segment_ids": list(distractor),
        "allowed_context_intersection": list(allowed),
        "route_relevant_evidence_refs": list(evidence),
        "future_segment_count": 0,
        "target_segment_count": 0,
    }


def _projection_kind(training_expert_asset: str, style_expert: str) -> str:
    if training_expert_asset == style_expert:
        return "primary_style"
    if training_expert_asset == "tool_call":
        return "tool_sidecar"
    if training_expert_asset == "review_audit":
        return "review_sidecar"
    _fail("projection_asset_not_selected")


def _route_commit(
    semantic_question_id_sha256: str,
    route_plan: Mapping[str, Any],
    planner_adapter_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "domain": "anchor.neural-swarm-planner-route-commit.v2",
            "semantic_question_id_sha256": semantic_question_id_sha256,
            "route_plan": route_plan,
            "planner_adapter_sha256": planner_adapter_sha256,
        }
    )


def build_projection_record(
    *,
    ordinal: int,
    split: str,
    language: str,
    task_bundle_sha256: str,
    source_record_id_sha256: str,
    source_group_sha256: str,
    leakage_family_sha256: str,
    prompt_digest_sha256: str,
    planner_freeze_receipt_sha256: str,
    planner_adapter_sha256: str,
    planner_question_generation_receipt_sha256: str,
    training_expert_asset: str,
    style_expert: str,
    variant: str,
    identity_required: bool,
    identity_provenance_class: str | None,
    identity_invariant_sha256: str | None,
    question_semantic_sha256: str,
    question_payload_sha256: str,
    shared_prefix_segment_ids: Sequence[str],
    salient_segment_ids: Sequence[str],
    distractor_segment_ids: Sequence[str],
    route_relevant_evidence_refs: Sequence[str],
    legacy_question_id_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one body-free, single-expert projection row."""

    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        _fail("record_ordinal_invalid")
    if split not in SPLITS:
        _fail("record_split_invalid")
    if language not in LANGUAGES:
        _fail("record_language_invalid")
    if variant not in VARIANTS:
        _fail("record_variant_invalid")
    if training_expert_asset not in EXPERT_ASSETS:
        _fail("training_expert_asset_invalid")
    for value, code in (
        (task_bundle_sha256, "task_bundle_sha256_invalid"),
        (source_record_id_sha256, "source_record_id_sha256_invalid"),
        (source_group_sha256, "source_group_sha256_invalid"),
        (leakage_family_sha256, "leakage_family_sha256_invalid"),
        (prompt_digest_sha256, "prompt_digest_sha256_invalid"),
        (planner_freeze_receipt_sha256, "planner_freeze_receipt_sha256_invalid"),
        (planner_adapter_sha256, "planner_adapter_sha256_invalid"),
        (
            planner_question_generation_receipt_sha256,
            "planner_question_generation_receipt_sha256_invalid",
        ),
        (question_semantic_sha256, "question_semantic_sha256_invalid"),
        (question_payload_sha256, "question_payload_sha256_invalid"),
    ):
        _sha(value, code)
    if legacy_question_id_sha256 is not None:
        _sha(legacy_question_id_sha256, "legacy_question_id_sha256_invalid")

    tool_required, review_required = _VARIANT_FLAGS[variant]
    route_plan = _build_route_plan(
        style_expert=style_expert,
        tool_required=tool_required,
        review_required=review_required,
        identity_required=identity_required,
        identity_provenance_class=identity_provenance_class,
        identity_invariant_sha256=identity_invariant_sha256,
    )
    if training_expert_asset not in route_plan["selected_expert_assets"]:
        _fail("training_expert_not_selected")
    projection_kind = _projection_kind(training_expert_asset, style_expert)
    context_segments = _build_context_segments(
        shared_prefix_segment_ids=shared_prefix_segment_ids,
        salient_segment_ids=salient_segment_ids,
        distractor_segment_ids=distractor_segment_ids,
        route_relevant_evidence_refs=route_relevant_evidence_refs,
    )
    planner_lineage = {
        "planner_freeze_receipt_sha256": planner_freeze_receipt_sha256,
        "planner_adapter_sha256": planner_adapter_sha256,
        "planner_question_generation_receipt_sha256": (
            planner_question_generation_receipt_sha256
        ),
    }
    semantic_preimage = {
        "domain": "anchor.neural-swarm-planner-semantic-question.v2",
        "split": split,
        "language": language,
        "task_bundle_sha256": task_bundle_sha256,
        "source_group_sha256": source_group_sha256,
        "leakage_family_sha256": leakage_family_sha256,
        "prompt_digest_sha256": prompt_digest_sha256,
        "planner_lineage": planner_lineage,
        "route_plan": route_plan,
        "variant": variant,
        "question_semantic_sha256": question_semantic_sha256,
        "question_payload_sha256": question_payload_sha256,
    }
    semantic_question_id = canonical_sha256(semantic_preimage)
    route_commit = _route_commit(
        semantic_question_id,
        route_plan,
        planner_adapter_sha256,
    )
    record_preimage = {
        "domain": "anchor.neural-swarm-planner-expert-projection.v2",
        "semantic_question_id_sha256": semantic_question_id,
        "legacy_question_id_sha256": legacy_question_id_sha256,
        "ordinal": ordinal,
        "source_record_id_sha256": source_record_id_sha256,
        "training_expert_asset": training_expert_asset,
        "projection_kind": projection_kind,
        "context_segments": context_segments,
        "route_commit_sha256": route_commit,
    }
    return {
        "schema_version": RECORD_VERSION,
        "record_id_sha256": canonical_sha256(record_preimage),
        "semantic_question_id_sha256": semantic_question_id,
        "legacy_question_id_sha256": legacy_question_id_sha256,
        "ordinal": ordinal,
        "split": split,
        "language": language,
        "task_bundle_sha256": task_bundle_sha256,
        "source_record_id_sha256": source_record_id_sha256,
        "source_group_sha256": source_group_sha256,
        "leakage_family_sha256": leakage_family_sha256,
        "prompt_digest_sha256": prompt_digest_sha256,
        "planner_lineage": planner_lineage,
        "training_expert_asset": training_expert_asset,
        "projection_kind": projection_kind,
        "route_plan": route_plan,
        "variant": variant,
        "identity_provenance_class": identity_provenance_class,
        "identity_invariant_sha256": identity_invariant_sha256,
        "question_semantic_sha256": question_semantic_sha256,
        "question_payload_sha256": question_payload_sha256,
        "context_segments": context_segments,
        "optimization_eligible": split == "train",
        "calibration_only": split == "calibration",
        "eval_proxy_only": split == "eval_proxy",
        "route_commit_sha256": route_commit,
        "claims": {
            "body_free_manifest_row": True,
            "question_payload_embedded": False,
            "target_visible": False,
            "future_visible": False,
            "raw_token_ids_stored": False,
        },
    }


def _rebuild_projection_record(row: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "record_id_sha256",
        "semantic_question_id_sha256",
        "legacy_question_id_sha256",
        "ordinal",
        "split",
        "language",
        "task_bundle_sha256",
        "source_record_id_sha256",
        "source_group_sha256",
        "leakage_family_sha256",
        "prompt_digest_sha256",
        "planner_lineage",
        "training_expert_asset",
        "projection_kind",
        "route_plan",
        "variant",
        "identity_provenance_class",
        "identity_invariant_sha256",
        "question_semantic_sha256",
        "question_payload_sha256",
        "context_segments",
        "optimization_eligible",
        "calibration_only",
        "eval_proxy_only",
        "route_commit_sha256",
        "claims",
    }
    _closed_keys(row, expected, "projection_record_keys_invalid")
    if row.get("schema_version") != RECORD_VERSION:
        _fail("projection_record_schema_version_invalid")
    lineage = _mapping(row.get("planner_lineage"), "planner_lineage_invalid")
    _closed_keys(
        lineage,
        {
            "planner_freeze_receipt_sha256",
            "planner_adapter_sha256",
            "planner_question_generation_receipt_sha256",
        },
        "planner_lineage_keys_invalid",
    )
    route_plan = _mapping(row.get("route_plan"), "route_plan_invalid")
    context = _mapping(row.get("context_segments"), "context_segments_invalid")
    return build_projection_record(
        ordinal=row.get("ordinal"),  # type: ignore[arg-type]
        split=str(row.get("split")),
        language=str(row.get("language")),
        task_bundle_sha256=str(row.get("task_bundle_sha256")),
        source_record_id_sha256=str(row.get("source_record_id_sha256")),
        source_group_sha256=str(row.get("source_group_sha256")),
        leakage_family_sha256=str(row.get("leakage_family_sha256")),
        prompt_digest_sha256=str(row.get("prompt_digest_sha256")),
        planner_freeze_receipt_sha256=str(lineage.get("planner_freeze_receipt_sha256")),
        planner_adapter_sha256=str(lineage.get("planner_adapter_sha256")),
        planner_question_generation_receipt_sha256=str(
            lineage.get("planner_question_generation_receipt_sha256")
        ),
        training_expert_asset=str(row.get("training_expert_asset")),
        style_expert=str(route_plan.get("style_expert")),
        variant=str(row.get("variant")),
        identity_required=route_plan.get("identity_required"),  # type: ignore[arg-type]
        identity_provenance_class=row.get("identity_provenance_class"),  # type: ignore[arg-type]
        identity_invariant_sha256=row.get("identity_invariant_sha256"),  # type: ignore[arg-type]
        question_semantic_sha256=str(row.get("question_semantic_sha256")),
        question_payload_sha256=str(row.get("question_payload_sha256")),
        shared_prefix_segment_ids=_sequence(
            context.get("shared_prefix_segment_ids"),
            "shared_prefix_segment_ids_invalid",
        ),
        salient_segment_ids=_sequence(
            context.get("salient_segment_ids"),
            "salient_segment_ids_invalid",
        ),
        distractor_segment_ids=_sequence(
            context.get("distractor_segment_ids"),
            "distractor_segment_ids_invalid",
        ),
        route_relevant_evidence_refs=_sequence(
            context.get("route_relevant_evidence_refs"),
            "route_relevant_evidence_refs_invalid",
        ),
        legacy_question_id_sha256=row.get("legacy_question_id_sha256"),  # type: ignore[arg-type]
    )


def build_route_hard_negative(
    positive_record: Mapping[str, Any],
    *,
    negative_kind: str,
) -> dict[str, Any]:
    """Build a single-variable body-free counterfactual route."""

    if negative_kind not in HARD_NEGATIVE_KINDS:
        _fail("hard_negative_kind_invalid")
    positive = _rebuild_projection_record(positive_record)
    positive_plan = dict(positive["route_plan"])
    negative_plan = dict(positive_plan)
    differs_only_in, failure_code = _NEGATIVE_RULES[negative_kind]

    if negative_kind == "wrong_primary_style":
        current = str(positive_plan["style_expert"])
        negative_plan["style_expert"] = STYLE_EXPERTS[
            (STYLE_EXPERTS.index(current) + 1) % len(STYLE_EXPERTS)
        ]
    elif negative_kind == "missing_required_tool":
        if positive_plan["tool_required"] is not True:
            _fail("hard_negative_requires_tool_positive")
        negative_plan["tool_required"] = False
    elif negative_kind == "unnecessary_tool":
        if positive_plan["tool_required"] is not False:
            _fail("hard_negative_requires_tool_absent")
        negative_plan["tool_required"] = True
    elif negative_kind == "missing_required_review":
        if positive_plan["review_required"] is not True:
            _fail("hard_negative_requires_review_positive")
        negative_plan["review_required"] = False
    elif negative_kind == "unnecessary_review":
        if positive_plan["review_required"] is not False:
            _fail("hard_negative_requires_review_absent")
        negative_plan["review_required"] = True
    elif negative_kind == "identity_bit_dropped":
        if positive_plan["identity_required"] is not True:
            _fail("hard_negative_requires_identity_positive")
        negative_plan["identity_required"] = False
        negative_plan["identity_provenance_class"] = None
        negative_plan["identity_invariant_sha256"] = None
    else:
        if positive_plan["identity_required"] is not True:
            _fail("hard_negative_requires_identity_positive")
        current_class = str(positive_plan["identity_provenance_class"])
        negative_class = {
            "air": "false_google",
            "false_google": "false_openai",
            "false_openai": "false_google",
        }[current_class]
        negative_plan["identity_provenance_class"] = negative_class
        negative_plan["identity_invariant_sha256"] = canonical_sha256(
            {
                "domain": "anchor.neural-swarm-identity-negative-invariant.v1",
                "semantic_question_id_sha256": positive["semantic_question_id_sha256"],
                "identity_provenance_class": negative_class,
            }
        )

    negative_plan["selected_expert_assets"] = list(
        _selected_assets(
            str(negative_plan["style_expert"]),
            tool_required=bool(negative_plan["tool_required"]),
            review_required=bool(negative_plan["review_required"]),
        )
    )
    negative_plan = _build_route_plan(
        style_expert=str(negative_plan["style_expert"]),
        tool_required=bool(negative_plan["tool_required"]),
        review_required=bool(negative_plan["review_required"]),
        identity_required=bool(negative_plan["identity_required"]),
        identity_provenance_class=negative_plan["identity_provenance_class"],
        identity_invariant_sha256=negative_plan["identity_invariant_sha256"],
    )
    negative_route_commit = _route_commit(
        str(positive["semantic_question_id_sha256"]),
        negative_plan,
        str(positive["planner_lineage"]["planner_adapter_sha256"]),
    )
    preimage = {
        "domain": "anchor.neural-swarm-planner-route-hard-negative.v1",
        "semantic_question_id_sha256": positive["semantic_question_id_sha256"],
        "positive_route_commit_sha256": positive["route_commit_sha256"],
        "negative_route_commit_sha256": negative_route_commit,
        "language": positive["language"],
        "split": positive["split"],
        "source_group_sha256": positive["source_group_sha256"],
        "question_payload_sha256": positive["question_payload_sha256"],
        "negative_kind": negative_kind,
        "differs_only_in": differs_only_in,
        "expected_failure_code": failure_code,
        "negative_route_plan": negative_plan,
        "derived_from_holdout": False,
    }
    return {
        "negative_id_sha256": canonical_sha256(preimage),
        **{key: value for key, value in preimage.items() if key != "domain"},
    }


def _count_map(values: Sequence[str], ordered_keys: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in ordered_keys}


def _manifest_counts(
    records: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "splits": _count_map([str(row["split"]) for row in records], SPLITS),
        "languages": _count_map(
            [str(row["language"]) for row in records],
            LANGUAGES,
        ),
        "expert_assets": _count_map(
            [str(row["training_expert_asset"]) for row in records],
            EXPERT_ASSETS,
        ),
        "variants": _count_map(
            [str(row["variant"]) for row in records],
            VARIANTS,
        ),
        "identity_required": _count_map(
            [
                str(bool(row["route_plan"]["identity_required"])).lower()
                for row in records
            ],
            ("false", "true"),
        ),
        "hard_negative_kinds": _count_map(
            [str(row["negative_kind"]) for row in negatives],
            HARD_NEGATIVE_KINDS,
        ),
    }


def build_manifest(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    records: Sequence[Mapping[str, Any]],
    route_hard_negatives: Sequence[Mapping[str, Any]],
    historical_invalidated_inventory_sha256: str,
    heldout_inventory_sha256: str,
    max_observed_dice: float,
) -> dict[str, Any]:
    """Build a candidate metadata manifest; validation remains a separate step."""

    config_result = validate_config(config, repo_root=repo_root)
    invalidated = _sha(
        historical_invalidated_inventory_sha256,
        "historical_invalidated_inventory_sha256_invalid",
    )
    heldout = _sha(heldout_inventory_sha256, "heldout_inventory_sha256_invalid")
    if (
        not isinstance(max_observed_dice, (int, float))
        or isinstance(max_observed_dice, bool)
        or not 0 <= float(max_observed_dice) < 0.9
    ):
        _fail("max_observed_dice_invalid")
    normalized_records = [dict(row) for row in records]
    normalized_negatives = [dict(row) for row in route_hard_negatives]
    first = _mapping(normalized_records[0], "manifest_records_empty")
    lineage = _mapping(first.get("planner_lineage"), "planner_lineage_invalid")
    ordered_ids = [str(row.get("record_id_sha256")) for row in normalized_records]
    bundles = {
        str(row.get("semantic_question_id_sha256")) for row in normalized_records
    }
    return {
        "schema_version": MANIFEST_VERSION,
        "status": "metadata_style_validated_execution_blocked",
        "dataset_style_config_sha256": config_result["config_sha256"],
        "planner_freeze_receipt_sha256": lineage["planner_freeze_receipt_sha256"],
        "planner_adapter_sha256": lineage["planner_adapter_sha256"],
        "planner_question_generation_receipt_sha256": lineage[
            "planner_question_generation_receipt_sha256"
        ],
        "historical_invalidated_inventory_sha256": invalidated,
        "heldout_inventory_sha256": heldout,
        "record_count": len(normalized_records),
        "bundle_count": len(bundles),
        "hard_negative_count": len(normalized_negatives),
        "ordered_record_ids_sha256": canonical_sha256(ordered_ids),
        "record_inventory_sha256": canonical_sha256(normalized_records),
        "counts": _manifest_counts(normalized_records, normalized_negatives),
        "isolation_proof": {
            "algorithm": "nfkc_casefold_word_qgram5_dice",
            "threshold": 0.9,
            "record_id_overlap_with_invalidated": 0,
            "normalized_prompt_overlap_with_invalidated": 0,
            "near_duplicate_violation_count": 0,
            "max_observed_dice": float(max_observed_dice),
            "cross_split_task_bundle_overlap": 0,
            "cross_split_source_group_overlap": 0,
            "cross_split_leakage_family_overlap": 0,
            "cross_split_prompt_digest_overlap": 0,
            "holdout_body_read": False,
            "holdout_used_for_style_tuning": False,
        },
        "records": normalized_records,
        "route_hard_negatives": normalized_negatives,
        "claims": {
            "training_payloads_materialized": False,
            "payload_bodies_embedded": False,
            "training_authorized": False,
            "formal": False,
            "release_authorized": False,
        },
    }


def _validate_route_plan_mapping(value: Any) -> dict[str, Any]:
    plan = _mapping(value, "route_plan_invalid")
    _closed_keys(
        plan,
        {
            "candidate_expert_assets",
            "selected_expert_assets",
            "style_expert",
            "tool_required",
            "review_required",
            "identity_required",
            "identity_provenance_class",
            "identity_invariant_sha256",
            "planner_exit_after_commit",
        },
        "route_plan_keys_invalid",
    )
    rebuilt = _build_route_plan(
        style_expert=str(plan.get("style_expert")),
        tool_required=plan.get("tool_required"),  # type: ignore[arg-type]
        review_required=plan.get("review_required"),  # type: ignore[arg-type]
        identity_required=plan.get("identity_required"),  # type: ignore[arg-type]
        identity_provenance_class=plan.get("identity_provenance_class"),  # type: ignore[arg-type]
        identity_invariant_sha256=plan.get("identity_invariant_sha256"),  # type: ignore[arg-type]
    )
    if dict(plan) != rebuilt:
        _fail("route_plan_identity_mismatch")
    return rebuilt


def _plan_diff_fields(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> set[str]:
    fields: set[str] = set()
    if positive["style_expert"] != negative["style_expert"]:
        fields.add("style_expert")
    if positive["tool_required"] != negative["tool_required"]:
        fields.add("tool_required")
    if positive["review_required"] != negative["review_required"]:
        fields.add("review_required")
    if positive["identity_required"] != negative["identity_required"]:
        fields.add("identity_required")
    elif (
        positive["identity_provenance_class"] != negative["identity_provenance_class"]
        or positive["identity_invariant_sha256"]
        != negative["identity_invariant_sha256"]
    ):
        fields.add("identity_provenance")
    return fields


def _validate_hard_negative(
    value: Any,
    *,
    positive_by_semantic: Mapping[str, Mapping[str, Any]],
    all_positive_route_commits: set[str],
) -> dict[str, Any]:
    negative = _mapping(value, "route_hard_negative_invalid")
    expected_keys = {
        "negative_id_sha256",
        "semantic_question_id_sha256",
        "positive_route_commit_sha256",
        "negative_route_commit_sha256",
        "language",
        "split",
        "source_group_sha256",
        "question_payload_sha256",
        "negative_kind",
        "differs_only_in",
        "expected_failure_code",
        "negative_route_plan",
        "derived_from_holdout",
    }
    _closed_keys(negative, expected_keys, "route_hard_negative_keys_invalid")
    kind = str(negative.get("negative_kind"))
    if kind not in HARD_NEGATIVE_KINDS:
        _fail("route_hard_negative_kind_invalid")
    differs_only_in, failure_code = _NEGATIVE_RULES[kind]
    if (
        negative.get("differs_only_in") != differs_only_in
        or negative.get("expected_failure_code") != failure_code
        or negative.get("derived_from_holdout") is not False
    ):
        _fail("route_hard_negative_semantics_invalid")
    semantic_id = _sha(
        negative.get("semantic_question_id_sha256"),
        "route_hard_negative_semantic_id_invalid",
    )
    positive = positive_by_semantic.get(semantic_id)
    if positive is None:
        _fail("route_hard_negative_positive_missing")
    for key in (
        "language",
        "split",
        "source_group_sha256",
        "question_payload_sha256",
    ):
        if negative.get(key) != positive.get(key):
            _fail(f"route_hard_negative_{key}_mismatch")
    if negative.get("positive_route_commit_sha256") != positive.get(
        "route_commit_sha256"
    ):
        _fail("route_hard_negative_positive_commit_mismatch")
    negative_plan = _validate_route_plan_mapping(negative.get("negative_route_plan"))
    positive_plan = _mapping(positive.get("route_plan"), "positive_route_plan_invalid")
    if _plan_diff_fields(positive_plan, negative_plan) != {differs_only_in}:
        _fail("route_hard_negative_not_single_variable")

    if kind == "missing_required_tool" and not (
        positive_plan["tool_required"] is True
        and negative_plan["tool_required"] is False
    ):
        _fail("route_hard_negative_direction_invalid")
    if kind == "unnecessary_tool" and not (
        positive_plan["tool_required"] is False
        and negative_plan["tool_required"] is True
    ):
        _fail("route_hard_negative_direction_invalid")
    if kind == "missing_required_review" and not (
        positive_plan["review_required"] is True
        and negative_plan["review_required"] is False
    ):
        _fail("route_hard_negative_direction_invalid")
    if kind == "unnecessary_review" and not (
        positive_plan["review_required"] is False
        and negative_plan["review_required"] is True
    ):
        _fail("route_hard_negative_direction_invalid")
    if kind == "identity_bit_dropped" and not (
        positive_plan["identity_required"] is True
        and negative_plan["identity_required"] is False
    ):
        _fail("route_hard_negative_direction_invalid")
    if kind == "identity_false_attribution" and not (
        positive_plan["identity_required"] is True
        and negative_plan["identity_required"] is True
        and positive_plan["identity_provenance_class"]
        != negative_plan["identity_provenance_class"]
    ):
        _fail("route_hard_negative_direction_invalid")

    negative_commit = _route_commit(
        semantic_id,
        negative_plan,
        str(positive["planner_lineage"]["planner_adapter_sha256"]),
    )
    if (
        negative.get("negative_route_commit_sha256") != negative_commit
        or negative_commit in all_positive_route_commits
    ):
        _fail("route_hard_negative_commit_invalid")
    preimage = {
        "domain": "anchor.neural-swarm-planner-route-hard-negative.v1",
        **{
            key: value for key, value in negative.items() if key != "negative_id_sha256"
        },
    }
    if negative.get("negative_id_sha256") != canonical_sha256(preimage):
        _fail("route_hard_negative_id_mismatch")
    return dict(negative)


def _validate_isolation_proof(value: Any) -> dict[str, Any]:
    proof = _mapping(value, "isolation_proof_invalid")
    expected = {
        "algorithm",
        "threshold",
        "record_id_overlap_with_invalidated",
        "normalized_prompt_overlap_with_invalidated",
        "near_duplicate_violation_count",
        "max_observed_dice",
        "cross_split_task_bundle_overlap",
        "cross_split_source_group_overlap",
        "cross_split_leakage_family_overlap",
        "cross_split_prompt_digest_overlap",
        "holdout_body_read",
        "holdout_used_for_style_tuning",
    }
    _closed_keys(proof, expected, "isolation_proof_keys_invalid")
    if (
        proof.get("algorithm") != "nfkc_casefold_word_qgram5_dice"
        or proof.get("threshold") != 0.9
        or any(
            proof.get(key) != 0
            for key in (
                "record_id_overlap_with_invalidated",
                "normalized_prompt_overlap_with_invalidated",
                "near_duplicate_violation_count",
                "cross_split_task_bundle_overlap",
                "cross_split_source_group_overlap",
                "cross_split_leakage_family_overlap",
                "cross_split_prompt_digest_overlap",
            )
        )
        or proof.get("holdout_body_read") is not False
        or proof.get("holdout_used_for_style_tuning") is not False
    ):
        _fail("isolation_proof_semantics_invalid")
    maximum = proof.get("max_observed_dice")
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not 0 <= float(maximum) < 0.9
    ):
        _fail("isolation_proof_max_dice_invalid")
    return dict(proof)


def validate_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate all projection, route, split, and body-free invariants."""

    config_result = validate_config(config, repo_root=repo_root)
    expected_keys = {
        "schema_version",
        "status",
        "dataset_style_config_sha256",
        "planner_freeze_receipt_sha256",
        "planner_adapter_sha256",
        "planner_question_generation_receipt_sha256",
        "historical_invalidated_inventory_sha256",
        "heldout_inventory_sha256",
        "record_count",
        "bundle_count",
        "hard_negative_count",
        "ordered_record_ids_sha256",
        "record_inventory_sha256",
        "counts",
        "isolation_proof",
        "records",
        "route_hard_negatives",
        "claims",
    }
    _closed_keys(manifest, expected_keys, "question_manifest_keys_invalid")
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("status") != "metadata_style_validated_execution_blocked"
        or manifest.get("dataset_style_config_sha256") != config_result["config_sha256"]
    ):
        _fail("question_manifest_semantics_invalid")
    for key in (
        "planner_freeze_receipt_sha256",
        "planner_adapter_sha256",
        "planner_question_generation_receipt_sha256",
        "historical_invalidated_inventory_sha256",
        "heldout_inventory_sha256",
        "ordered_record_ids_sha256",
        "record_inventory_sha256",
    ):
        _sha(manifest.get(key), f"{key}_invalid")

    raw_records = _sequence(
        manifest.get("records"), "question_manifest_records_invalid"
    )
    if not raw_records:
        _fail("question_manifest_records_empty")
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    source_record_ids: set[str] = set()
    semantic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_keys: dict[str, dict[str, str]] = {
        key: {}
        for key in (
            "task_bundle_sha256",
            "source_group_sha256",
            "leakage_family_sha256",
            "prompt_digest_sha256",
        )
    }
    for ordinal, raw_record in enumerate(raw_records):
        row = _mapping(raw_record, "projection_record_invalid")
        rebuilt = _rebuild_projection_record(row)
        if dict(row) != rebuilt:
            _fail("projection_record_identity_mismatch")
        if rebuilt["ordinal"] != ordinal:
            _fail("projection_record_order_invalid")
        record_id = rebuilt["record_id_sha256"]
        if record_id in record_ids:
            _fail("projection_record_id_duplicate")
        record_ids.add(record_id)
        source_record_id = rebuilt["source_record_id_sha256"]
        if source_record_id in source_record_ids:
            _fail("projection_source_record_id_duplicate")
        source_record_ids.add(source_record_id)
        semantic_groups[rebuilt["semantic_question_id_sha256"]].append(rebuilt)
        split_name = rebuilt["split"]
        for key, observed in split_keys.items():
            identity = rebuilt[key]
            previous = observed.setdefault(identity, split_name)
            if previous != split_name:
                _fail(f"cross_split_{key}_overlap")
        records.append(rebuilt)

    for semantic_id, group in semantic_groups.items():
        first = group[0]
        stable_keys = (
            "semantic_question_id_sha256",
            "split",
            "language",
            "task_bundle_sha256",
            "source_group_sha256",
            "leakage_family_sha256",
            "prompt_digest_sha256",
            "planner_lineage",
            "route_plan",
            "variant",
            "identity_provenance_class",
            "identity_invariant_sha256",
            "question_semantic_sha256",
            "question_payload_sha256",
            "route_commit_sha256",
        )
        for row in group[1:]:
            if any(row[key] != first[key] for key in stable_keys):
                _fail("semantic_bundle_projection_drift")
        actual_assets = [row["training_expert_asset"] for row in group]
        if len(actual_assets) != len(set(actual_assets)):
            _fail("semantic_bundle_expert_duplicate")
        expected_assets = list(first["route_plan"]["selected_expert_assets"])
        if actual_assets != expected_assets:
            _fail("semantic_bundle_projection_not_exact_selected_set")
        if first["semantic_question_id_sha256"] != semantic_id:
            _fail("semantic_bundle_identity_mismatch")

    if manifest.get("record_count") != len(records):
        _fail("question_manifest_record_count_mismatch")
    if manifest.get("bundle_count") != len(semantic_groups):
        _fail("question_manifest_bundle_count_mismatch")
    ordered_ids = [row["record_id_sha256"] for row in records]
    if manifest.get("ordered_record_ids_sha256") != canonical_sha256(ordered_ids):
        _fail("question_manifest_ordered_ids_mismatch")
    if manifest.get("record_inventory_sha256") != canonical_sha256(records):
        _fail("question_manifest_record_inventory_mismatch")

    common_lineage = records[0]["planner_lineage"]
    for key in (
        "planner_freeze_receipt_sha256",
        "planner_adapter_sha256",
        "planner_question_generation_receipt_sha256",
    ):
        if manifest.get(key) != common_lineage[key]:
            _fail(f"question_manifest_{key}_binding_mismatch")
    if any(row["planner_lineage"] != common_lineage for row in records):
        _fail("question_manifest_planner_lineage_drift")

    positive_by_semantic = {key: rows[0] for key, rows in semantic_groups.items()}
    positive_commits = {row["route_commit_sha256"] for row in records}
    raw_negatives = _sequence(
        manifest.get("route_hard_negatives"),
        "route_hard_negatives_invalid",
    )
    negatives: list[dict[str, Any]] = []
    negative_ids: set[str] = set()
    for raw_negative in raw_negatives:
        negative = _validate_hard_negative(
            raw_negative,
            positive_by_semantic=positive_by_semantic,
            all_positive_route_commits=positive_commits,
        )
        negative_id = negative["negative_id_sha256"]
        if negative_id in negative_ids:
            _fail("route_hard_negative_id_duplicate")
        negative_ids.add(negative_id)
        negatives.append(negative)
    if manifest.get("hard_negative_count") != len(negatives):
        _fail("hard_negative_count_mismatch")

    expected_counts = _manifest_counts(records, negatives)
    if manifest.get("counts") != expected_counts:
        _fail("question_manifest_counts_mismatch")
    if any(expected_counts["splits"][key] == 0 for key in SPLITS):
        _fail("question_manifest_split_coverage_missing")
    if any(expected_counts["languages"][key] == 0 for key in LANGUAGES):
        _fail("question_manifest_language_coverage_missing")
    if any(expected_counts["expert_assets"][key] == 0 for key in EXPERT_ASSETS):
        _fail("question_manifest_expert_coverage_missing")
    if any(expected_counts["variants"][key] == 0 for key in VARIANTS):
        _fail("question_manifest_variant_coverage_missing")
    if any(
        expected_counts["hard_negative_kinds"][key] == 0 for key in HARD_NEGATIVE_KINDS
    ):
        _fail("question_manifest_hard_negative_coverage_missing")

    semantic_rows = list(positive_by_semantic.values())
    for variant in ("base", "with_review", "with_tool_review"):
        if not any(
            row["route_plan"]["style_expert"] == "serious"
            and row["route_plan"]["identity_required"] is True
            and row["variant"] == variant
            for row in semantic_rows
        ):
            _fail(f"serious_identity_{variant}_coverage_missing")
    if not any(
        row["route_plan"]["style_expert"] == "serious"
        and row["route_plan"]["identity_required"] is True
        and row["variant"] == "with_tool"
        for row in semantic_rows
    ):
        _fail("serious_identity_with_tool_positive_control_missing")

    _validate_isolation_proof(manifest.get("isolation_proof"))
    claims = _mapping(manifest.get("claims"), "question_manifest_claims_invalid")
    _closed_keys(
        claims,
        {
            "training_payloads_materialized",
            "payload_bodies_embedded",
            "training_authorized",
            "formal",
            "release_authorized",
        },
        "question_manifest_claims_keys_invalid",
    )
    if any(value is not False for value in claims.values()):
        _fail("question_manifest_claim_scope_invalid")

    return {
        "schema_version": "anchor.neural-swarm-planner-first-dataset-style-decision.v1",
        "status": "dataset_style_ready_runtime_integration_blocked",
        "dataset_style_config_sha256": config_result["config_sha256"],
        "question_manifest_sha256": canonical_sha256(manifest),
        "record_inventory_sha256": manifest["record_inventory_sha256"],
        "record_count": len(records),
        "bundle_count": len(semantic_groups),
        "hard_negative_count": len(negatives),
        "legacy_one_expert_per_question_preserved": True,
        "candidate_inventory_interpreted_as_activation_set": False,
        "planner_first_leaf_launcher_integrated": False,
        "existing_training_runtime_modified": False,
        "training_authorized": False,
        "formal": False,
        "release_authorized": False,
    }


def load_config(path: Path, *, repo_root: Path) -> Mapping[str, Any]:
    relative = path.relative_to(repo_root).as_posix()
    return _load_json_file(_repo_file(repo_root, relative))


def load_manifest(path: Path, *, repo_root: Path) -> Mapping[str, Any]:
    relative = path.relative_to(repo_root).as_posix()
    return _load_json_file(_repo_file(repo_root, relative))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo_root = args.repo_root.resolve(strict=True)
        config = load_config(args.config, repo_root=repo_root)
        manifest = load_manifest(args.manifest, repo_root=repo_root)
        result = validate_manifest(config, manifest, repo_root=repo_root)
    except (OSError, ValueError, PlannerFirstDatasetStyleError) as exc:
        code = str(exc)
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": code,
                    "training_authorized": False,
                    "formal": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
