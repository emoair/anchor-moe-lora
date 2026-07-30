"""Model-free Planner-first curriculum gate for the neural-swarm training line.

The existing GPU training runtime is deliberately treated as an immutable leaf.
This module only authenticates the dependency chain that must exist before a
leaf expert is eligible to enter that runtime:

planner training -> frozen planner -> planner questions -> expert projections
-> per-expert leaf preflight tickets.

It does not load a model, allocate CUDA memory, call a provider, or grant GPU,
training, formal, or release authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

import yaml


SCHEMA_VERSION: Final = "anchor.neural-swarm-planner-first-curriculum.v1"
RUNTIME_BUNDLE_VERSION: Final = "anchor.neural-swarm-planner-first-runtime-bundle.v1"
DECISION_VERSION: Final = "anchor.neural-swarm-planner-first-decision.v1"

STAGE_ORDER: Final = (
    "planner_train",
    "planner_freeze",
    "planner_question_commit",
    "expert_dataset_projection",
    "expert_leaf_release",
)
EXPERT_ASSETS: Final = (
    "humor",
    "serious",
    "angry_style",
    "tool_call",
    "review_audit",
)
EXPERT_LEAF_ARMS: Final = (
    "humor_q_only",
    "serious_q_only",
    "angry_style_q_only",
    "tool_q_only",
    "review_audit_q_only",
)
PLANNER_DIAGNOSTIC_ARMS: Final = (
    "planner_o_only",
    "planner_q_plus_micro_o",
)
OBSERVED_LEGACY_ORDER: Final = (
    "humor_q_only",
    "serious_q_only",
    "angry_style_q_only",
    "review_audit_q_only",
    "tool_q_only",
    "tool_q_plus_micro_o",
    "planner_q_only",
    "planner_o_only",
    "planner_q_plus_micro_o",
)
HIERARCHICAL_SERIAL_ORDER: Final = (
    "train_and_freeze_general_planner",
    "train_and_freeze_each_specialist_planner",
    "align_selected_chain_activations",
    "train_each_governed_output_expert_on_frozen_upstream_commits",
)
FALSE_CLAIMS: Final = (
    "existing_training_runtime_modified",
    "planner_first_leaf_launcher_integrated",
    "question_generation_executed",
    "expert_training_executed",
    "gpu_authorized",
    "training_authorized",
    "formal",
    "release_authorized",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPARSE_POINT = 0x400


class PlannerFirstCurriculumError(ValueError):
    """Raised when a Planner-first dependency or physical identity fails."""


def _fail(code: str) -> None:
    raise PlannerFirstCurriculumError(code)


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _closed_keys(
    value: Mapping[str, Any],
    expected: set[str],
    code: str,
) -> None:
    if set(value) != expected:
        _fail(code)


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical compact JSON bytes without a trailing newline."""

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
        except FileNotFoundError as exc:
            raise PlannerFirstCurriculumError("authenticated_path_missing") from exc
        if stat.S_ISLNK(identity.st_mode) or _is_reparse(identity):
            _fail("authenticated_path_reparse_forbidden")


def _repo_file(repo_root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or relative.startswith("/")
    ):
        _fail("repo_relative_path_invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("repo_relative_path_invalid")
    candidate = repo_root.joinpath(*pure.parts)
    _reject_reparse_chain(repo_root, candidate)
    return candidate


def _stable_read(path: Path) -> bytes:
    try:
        lexical_before = path.lstat()
    except FileNotFoundError as exc:
        raise PlannerFirstCurriculumError("authenticated_file_missing") from exc
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
        raw = handle.read()
        handle_after = os.fstat(handle.fileno())
    try:
        lexical_after = path.lstat()
    except FileNotFoundError as exc:
        raise PlannerFirstCurriculumError(
            "authenticated_file_final_identity_missing"
        ) from exc
    if not (
        _identity(lexical_before)
        == _identity(handle_before)
        == _identity(handle_after)
        == _identity(lexical_after)
    ):
        _fail("authenticated_file_identity_drift")
    if len(raw) != lexical_after.st_size:
        _fail("authenticated_file_size_drift")
    return raw


def _load_json_bytes(
    raw: bytes,
    code: str,
    *,
    require_lf: bool = True,
) -> Mapping[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf") or (require_lf and b"\r\n" in raw):
        _fail(f"{code}_encoding_invalid")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlannerFirstCurriculumError(f"{code}_parse_failed") from exc
    return _mapping(value, f"{code}_root_invalid")


def _verify_pin(
    repo_root: Path,
    value: Mapping[str, Any],
    *,
    code: str,
    allow_legacy_crlf_equivalence: bool = False,
) -> tuple[Path, bytes]:
    _closed_keys(value, {"path", "sha256"}, f"{code}_keys_invalid")
    expected = _sha(value.get("sha256"), f"{code}_sha256_invalid")
    path = _repo_file(repo_root, str(value.get("path")))
    raw = _stable_read(path)
    if hashlib.sha256(raw).hexdigest() != expected:
        if not allow_legacy_crlf_equivalence or b"\r\n" not in raw:
            _fail(f"{code}_identity_drift")
        canonical = raw.replace(b"\r\n", b"\n")
        if b"\r" in canonical or hashlib.sha256(canonical).hexdigest() != expected:
            _fail(f"{code}_identity_drift")
        raw = canonical
    return path, raw


def validate_contract(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate the Planner-first contract and all frozen leaf-runtime pins."""

    if contract.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version_mismatch")
    if contract.get("status") != "additive_model_free_preflight_only":
        _fail("status_mismatch")

    leaf = _mapping(
        contract.get("frozen_leaf_runtime"),
        "frozen_leaf_runtime_invalid",
    )
    if leaf.get("unchanged") is not True:
        _fail("existing_training_runtime_must_remain_unchanged")
    _, runtime_config_raw = _verify_pin(
        repo_root,
        _mapping(leaf.get("runtime_config"), "runtime_config_pin_invalid"),
        code="runtime_config",
    )
    _verify_pin(
        repo_root,
        _mapping(
            leaf.get("runtime_implementation"),
            "runtime_implementation_pin_invalid",
        ),
        code="runtime_implementation",
    )
    _verify_pin(
        repo_root,
        _mapping(
            leaf.get("runtime_contract_implementation"),
            "runtime_contract_implementation_pin_invalid",
        ),
        code="runtime_contract_implementation",
    )
    try:
        runtime_config = yaml.safe_load(runtime_config_raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PlannerFirstCurriculumError("runtime_config_parse_failed") from exc
    runtime_config = _mapping(runtime_config, "runtime_config_root_invalid")
    arms = _mapping(runtime_config.get("arms"), "runtime_arms_invalid")
    observed_order = tuple(_sequence(arms.get("ordered"), "runtime_arm_order_invalid"))
    if observed_order != OBSERVED_LEGACY_ORDER:
        _fail("frozen_leaf_runtime_arm_order_drift")
    if tuple(leaf.get("observed_arm_order", ())) != OBSERVED_LEGACY_ORDER:
        _fail("contract_observed_arm_order_mismatch")
    if leaf.get("observed_planner_first") is not False:
        _fail("legacy_runtime_must_not_be_misreported_planner_first")
    if leaf.get("legacy_direct_invocation_unchanged") is not True:
        _fail("legacy_direct_invocation_must_remain_unchanged")
    if leaf.get("existing_runtime_enforces_release_ticket") is not False:
        _fail("existing_runtime_ticket_enforcement_overclaim")
    if leaf.get("planner_first_flow_requires_release_ticket") is not True:
        _fail("planner_first_flow_release_ticket_required")

    hierarchy = _mapping(
        contract.get("hierarchical_dependency"),
        "hierarchical_dependency_invalid",
    )
    if (
        hierarchy.get("checkout_eol_compatibility")
        != "canonical_lf_with_legacy_crlf_worktree_equivalence"
    ):
        _fail("hierarchical_checkout_eol_policy_mismatch")
    _, hierarchy_raw = _verify_pin(
        repo_root,
        _mapping(hierarchy.get("contract"), "hierarchical_contract_pin_invalid"),
        code="hierarchical_contract",
        allow_legacy_crlf_equivalence=True,
    )
    _verify_pin(
        repo_root,
        _mapping(
            hierarchy.get("implementation"),
            "hierarchical_implementation_pin_invalid",
        ),
        code="hierarchical_implementation",
        allow_legacy_crlf_equivalence=True,
    )
    hierarchy_value = _load_json_bytes(
        hierarchy_raw,
        "hierarchical_contract",
        require_lf=False,
    )
    serial = _mapping(
        hierarchy_value.get("serial_alignment"),
        "hierarchical_serial_alignment_invalid",
    )
    if tuple(serial.get("order", ())) != HIERARCHICAL_SERIAL_ORDER:
        _fail("hierarchical_serial_order_drift")

    graph = _mapping(contract.get("stage_graph"), "stage_graph_invalid")
    if tuple(graph.get("ordered", ())) != STAGE_ORDER:
        _fail("planner_first_stage_order_mismatch")
    if graph.get("planner_primary_arm") != "planner_q_only":
        _fail("planner_primary_arm_mismatch")
    if tuple(graph.get("planner_diagnostic_arms", ())) != PLANNER_DIAGNOSTIC_ARMS:
        _fail("planner_diagnostic_arms_mismatch")
    if graph.get("diagnostic_arms_can_authorize_downstream") is not False:
        _fail("planner_diagnostic_arm_authority_forbidden")
    if tuple(graph.get("expert_assets", ())) != EXPERT_ASSETS:
        _fail("expert_asset_inventory_mismatch")
    if tuple(graph.get("expert_leaf_arms", ())) != EXPERT_LEAF_ARMS:
        _fail("expert_leaf_arm_inventory_mismatch")
    if graph.get("strictly_serial") is not True or graph.get("max_concurrency") != 1:
        _fail("leaf_execution_must_remain_strictly_serial")
    for key in (
        "planner_must_be_frozen_before_questions",
        "question_manifest_required_before_projection",
        "all_expert_projections_required_before_release",
    ):
        if graph.get(key) is not True:
            _fail(f"stage_graph_{key}_required")
    for key in (
        "planner_reentry_after_expert_start",
        "expert_to_planner_feedback_loop",
    ):
        if graph.get(key) is not False:
            _fail(f"stage_graph_{key}_forbidden")

    questions = _mapping(
        contract.get("planner_question_contract"),
        "planner_question_contract_invalid",
    )
    if questions.get("generator") != "frozen_planner_adapter_only":
        _fail("question_generator_must_be_frozen_planner")
    if questions.get("manifest_body_free") is not True:
        _fail("question_manifest_must_be_body_free")
    if (
        questions.get("question_payload_storage")
        != "separate_immutable_payload_referenced_by_sha256"
    ):
        _fail("question_payload_storage_mismatch")
    if questions.get("one_expert_per_question") is not True:
        _fail("one_expert_per_question_required")
    if questions.get("minimum_questions_per_expert") != 1:
        _fail("minimum_questions_per_expert_mismatch")
    if questions.get("split_before_projection") is not True:
        _fail("split_before_projection_required")

    release = _mapping(contract.get("release_gate"), "release_gate_invalid")
    if tuple(release.get("required_chain", ())) != STAGE_ORDER[:-1]:
        _fail("release_required_chain_mismatch")
    if release.get("output") != "expert_leaf_preflight_tickets_only":
        _fail("release_output_scope_mismatch")
    if release.get("tickets_authorize_gpu") is not False:
        _fail("leaf_ticket_gpu_authority_forbidden")
    if release.get("tickets_modify_existing_runner") is not False:
        _fail("leaf_ticket_runtime_mutation_forbidden")

    claims = _mapping(contract.get("claims"), "claims_invalid")
    for key in FALSE_CLAIMS:
        if claims.get(key) is not False:
            _fail(f"claim_{key}_must_be_false")

    return {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(contract),
        "status": "passed",
        "legacy_runtime_planner_first": False,
        "planner_first_additive_gate_required": True,
        "existing_training_runtime_modified": False,
        "training_authorized": False,
    }


def build_question_row(
    *,
    planner_freeze_receipt_sha256: str,
    planner_adapter_sha256: str,
    ordinal: int,
    expert_asset: str,
    language: str,
    source_record_id_sha256: str,
    question_semantic_sha256: str,
    question_payload_sha256: str,
) -> dict[str, Any]:
    """Build one authenticated body-free Planner question row."""

    _sha(
        planner_freeze_receipt_sha256,
        "question_planner_freeze_receipt_sha256_invalid",
    )
    _sha(planner_adapter_sha256, "question_planner_adapter_sha256_invalid")
    _sha(source_record_id_sha256, "question_source_record_sha256_invalid")
    _sha(question_semantic_sha256, "question_semantic_sha256_invalid")
    _sha(question_payload_sha256, "question_payload_sha256_invalid")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        _fail("question_ordinal_invalid")
    if expert_asset not in EXPERT_ASSETS:
        _fail("question_expert_asset_invalid")
    if language not in {"en", "zh-CN"}:
        _fail("question_language_invalid")
    preimage = {
        "domain": "anchor.neural-swarm-planner-question.v1",
        "planner_freeze_receipt_sha256": planner_freeze_receipt_sha256,
        "planner_adapter_sha256": planner_adapter_sha256,
        "ordinal": ordinal,
        "expert_asset": expert_asset,
        "language": language,
        "split": "train",
        "source_record_id_sha256": source_record_id_sha256,
        "question_semantic_sha256": question_semantic_sha256,
        "question_payload_sha256": question_payload_sha256,
    }
    question_id = canonical_sha256(preimage)
    route_commit = canonical_sha256(
        {
            "domain": "anchor.neural-swarm-planner-route-commit.v1",
            "question_id_sha256": question_id,
            "expert_asset": expert_asset,
            "planner_adapter_sha256": planner_adapter_sha256,
        }
    )
    return {
        "question_id_sha256": question_id,
        "ordinal": ordinal,
        "expert_asset": expert_asset,
        "language": language,
        "split": "train",
        "source_record_id_sha256": source_record_id_sha256,
        "question_semantic_sha256": question_semantic_sha256,
        "question_payload_sha256": question_payload_sha256,
        "route_commit_sha256": route_commit,
    }


def _validate_training_receipt(value: Any) -> tuple[Mapping[str, Any], str]:
    receipt = _mapping(value, "planner_training_receipt_invalid")
    expected_keys = {
        "schema_version",
        "stage",
        "status",
        "arm",
        "source_asset",
        "run_id_sha256",
        "base_model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "training_record_inventory_sha256",
        "adapter_sha256",
        "completed_steps",
        "training_authorized",
        "formal",
    }
    _closed_keys(receipt, expected_keys, "planner_training_receipt_keys_invalid")
    if (
        receipt.get("schema_version")
        != "anchor.neural-swarm-planner-training-receipt.v1"
        or receipt.get("stage") != "planner_train"
        or receipt.get("status") != "passed"
        or receipt.get("arm") != "planner_q_only"
        or receipt.get("source_asset") != "planner_router"
    ):
        _fail("planner_training_receipt_semantics_invalid")
    for key in (
        "run_id_sha256",
        "base_model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "training_record_inventory_sha256",
        "adapter_sha256",
    ):
        _sha(receipt.get(key), f"planner_training_{key}_invalid")
    completed = receipt.get("completed_steps")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 1:
        _fail("planner_training_completed_steps_invalid")
    if (
        receipt.get("training_authorized") is not False
        or receipt.get("formal") is not False
    ):
        _fail("planner_training_claim_scope_invalid")
    return receipt, canonical_sha256(receipt)


def _validate_freeze_receipt(
    value: Any,
    *,
    training_receipt: Mapping[str, Any],
    training_receipt_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    receipt = _mapping(value, "planner_freeze_receipt_invalid")
    expected_keys = {
        "schema_version",
        "stage",
        "status",
        "planner_training_receipt_sha256",
        "planner_adapter_sha256",
        "frozen_parameter_inventory_sha256",
        "adapter_frozen",
        "mutation_after_freeze_allowed",
    }
    _closed_keys(receipt, expected_keys, "planner_freeze_receipt_keys_invalid")
    if (
        receipt.get("schema_version") != "anchor.neural-swarm-planner-freeze-receipt.v1"
        or receipt.get("stage") != "planner_freeze"
        or receipt.get("status") != "passed"
    ):
        _fail("planner_freeze_receipt_semantics_invalid")
    if receipt.get("planner_training_receipt_sha256") != training_receipt_sha256:
        _fail("planner_freeze_training_receipt_binding_mismatch")
    if receipt.get("planner_adapter_sha256") != training_receipt.get("adapter_sha256"):
        _fail("planner_freeze_adapter_binding_mismatch")
    _sha(
        receipt.get("frozen_parameter_inventory_sha256"),
        "planner_frozen_parameter_inventory_sha256_invalid",
    )
    if receipt.get("adapter_frozen") is not True:
        _fail("planner_adapter_must_be_frozen")
    if receipt.get("mutation_after_freeze_allowed") is not False:
        _fail("planner_mutation_after_freeze_forbidden")
    return receipt, canonical_sha256(receipt)


def _validate_question_manifest(
    value: Any,
    *,
    freeze_receipt: Mapping[str, Any],
    freeze_receipt_sha256: str,
    minimum_per_expert: int,
) -> tuple[Mapping[str, Any], str, dict[str, tuple[dict[str, Any], ...]]]:
    manifest = _mapping(value, "planner_question_manifest_invalid")
    expected_keys = {
        "schema_version",
        "stage",
        "status",
        "planner_freeze_receipt_sha256",
        "planner_adapter_sha256",
        "question_count",
        "ordered_question_ids_sha256",
        "rows",
    }
    _closed_keys(manifest, expected_keys, "planner_question_manifest_keys_invalid")
    if (
        manifest.get("schema_version")
        != "anchor.neural-swarm-planner-question-manifest.v1"
        or manifest.get("stage") != "planner_question_commit"
        or manifest.get("status") != "passed"
    ):
        _fail("planner_question_manifest_semantics_invalid")
    if manifest.get("planner_freeze_receipt_sha256") != freeze_receipt_sha256:
        _fail("question_manifest_freeze_binding_mismatch")
    planner_adapter_sha256 = str(freeze_receipt.get("planner_adapter_sha256"))
    if manifest.get("planner_adapter_sha256") != planner_adapter_sha256:
        _fail("question_manifest_adapter_binding_mismatch")

    raw_rows = _sequence(manifest.get("rows"), "question_manifest_rows_invalid")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for expected_ordinal, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, "question_row_invalid")
        expected_keys = {
            "question_id_sha256",
            "ordinal",
            "expert_asset",
            "language",
            "split",
            "source_record_id_sha256",
            "question_semantic_sha256",
            "question_payload_sha256",
            "route_commit_sha256",
        }
        _closed_keys(row, expected_keys, "question_row_keys_invalid")
        if row.get("ordinal") != expected_ordinal:
            _fail("question_row_order_invalid")
        rebuilt = build_question_row(
            planner_freeze_receipt_sha256=freeze_receipt_sha256,
            planner_adapter_sha256=planner_adapter_sha256,
            ordinal=expected_ordinal,
            expert_asset=str(row.get("expert_asset")),
            language=str(row.get("language")),
            source_record_id_sha256=str(row.get("source_record_id_sha256")),
            question_semantic_sha256=str(row.get("question_semantic_sha256")),
            question_payload_sha256=str(row.get("question_payload_sha256")),
        )
        if dict(row) != rebuilt:
            _fail("question_row_identity_mismatch")
        question_id = rebuilt["question_id_sha256"]
        if question_id in seen:
            _fail("question_id_duplicate")
        seen.add(question_id)
        counts[rebuilt["expert_asset"]] += 1
        rows.append(rebuilt)

    if manifest.get("question_count") != len(rows):
        _fail("question_manifest_count_mismatch")
    ordered_ids = [row["question_id_sha256"] for row in rows]
    if manifest.get("ordered_question_ids_sha256") != canonical_sha256(ordered_ids):
        _fail("question_manifest_ordered_ids_mismatch")
    for expert in EXPERT_ASSETS:
        if counts[expert] < minimum_per_expert:
            _fail(f"question_manifest_{expert}_coverage_missing")

    grouped = {
        expert: tuple(row for row in rows if row["expert_asset"] == expert)
        for expert in EXPERT_ASSETS
    }
    normalized = dict(manifest)
    normalized["rows"] = rows
    return normalized, canonical_sha256(normalized), grouped


def _validate_question_generation_receipt(
    value: Any,
    *,
    freeze_receipt: Mapping[str, Any],
    freeze_receipt_sha256: str,
    question_manifest_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    receipt = _mapping(value, "planner_question_generation_receipt_invalid")
    expected_keys = {
        "schema_version",
        "stage",
        "status",
        "generation_mode",
        "generation_run_id_sha256",
        "generator_implementation_sha256",
        "planner_freeze_receipt_sha256",
        "planner_adapter_sha256",
        "question_manifest_sha256",
        "model_request_count",
        "provider_request_count",
        "raw_hidden_reasoning_persisted",
        "training_authorized",
        "formal",
    }
    _closed_keys(
        receipt,
        expected_keys,
        "planner_question_generation_receipt_keys_invalid",
    )
    if (
        receipt.get("schema_version")
        != "anchor.neural-swarm-planner-question-generation-receipt.v1"
        or receipt.get("stage") != "planner_question_commit"
        or receipt.get("status") != "passed"
        or receipt.get("generation_mode") != "frozen_planner_inference"
    ):
        _fail("planner_question_generation_receipt_semantics_invalid")
    for key in (
        "generation_run_id_sha256",
        "generator_implementation_sha256",
    ):
        _sha(receipt.get(key), f"planner_question_generation_{key}_invalid")
    if receipt.get("planner_freeze_receipt_sha256") != freeze_receipt_sha256:
        _fail("question_generation_freeze_binding_mismatch")
    if receipt.get("planner_adapter_sha256") != freeze_receipt.get(
        "planner_adapter_sha256"
    ):
        _fail("question_generation_adapter_binding_mismatch")
    if receipt.get("question_manifest_sha256") != question_manifest_sha256:
        _fail("question_generation_manifest_binding_mismatch")
    model_requests = receipt.get("model_request_count")
    provider_requests = receipt.get("provider_request_count")
    if (
        not isinstance(model_requests, int)
        or isinstance(model_requests, bool)
        or model_requests < 1
        or not isinstance(provider_requests, int)
        or isinstance(provider_requests, bool)
        or provider_requests < 0
        or provider_requests > model_requests
    ):
        _fail("question_generation_request_counts_invalid")
    if receipt.get("raw_hidden_reasoning_persisted") is not False:
        _fail("question_generation_hidden_reasoning_forbidden")
    if (
        receipt.get("training_authorized") is not False
        or receipt.get("formal") is not False
    ):
        _fail("question_generation_claim_scope_invalid")
    return receipt, canonical_sha256(receipt)


def build_runtime_decision(
    contract: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Authenticate the full dependency chain and emit leaf preflight tickets."""

    contract_result = validate_contract(contract, repo_root=repo_root)
    _closed_keys(
        bundle,
        {
            "schema_version",
            "planner_training_receipt",
            "planner_freeze_receipt",
            "planner_question_manifest",
            "planner_question_generation_receipt",
        },
        "runtime_bundle_keys_invalid",
    )
    if bundle.get("schema_version") != RUNTIME_BUNDLE_VERSION:
        _fail("runtime_bundle_schema_version_mismatch")
    training, training_sha = _validate_training_receipt(
        bundle.get("planner_training_receipt")
    )
    freeze, freeze_sha = _validate_freeze_receipt(
        bundle.get("planner_freeze_receipt"),
        training_receipt=training,
        training_receipt_sha256=training_sha,
    )
    question_contract = _mapping(
        contract.get("planner_question_contract"),
        "planner_question_contract_invalid",
    )
    manifest, manifest_sha, grouped = _validate_question_manifest(
        bundle.get("planner_question_manifest"),
        freeze_receipt=freeze,
        freeze_receipt_sha256=freeze_sha,
        minimum_per_expert=int(question_contract["minimum_questions_per_expert"]),
    )
    _, generation_sha = _validate_question_generation_receipt(
        bundle.get("planner_question_generation_receipt"),
        freeze_receipt=freeze,
        freeze_receipt_sha256=freeze_sha,
        question_manifest_sha256=manifest_sha,
    )

    tickets: list[dict[str, Any]] = []
    for execution_order, (expert, arm) in enumerate(
        zip(EXPERT_ASSETS, EXPERT_LEAF_ARMS, strict=True)
    ):
        rows = grouped[expert]
        ordered_ids = [row["question_id_sha256"] for row in rows]
        dataset_inventory_sha256 = canonical_sha256(list(rows))
        ticket_preimage = {
            "domain": "anchor.neural-swarm-expert-leaf-ticket.v1",
            "execution_order": execution_order,
            "expert_asset": expert,
            "leaf_runner_arm": arm,
            "planner_training_receipt_sha256": training_sha,
            "planner_freeze_receipt_sha256": freeze_sha,
            "planner_question_manifest_sha256": manifest_sha,
            "planner_question_generation_receipt_sha256": generation_sha,
            "planner_adapter_sha256": freeze["planner_adapter_sha256"],
            "dataset_inventory_sha256": dataset_inventory_sha256,
            "ordered_question_ids_sha256": canonical_sha256(ordered_ids),
            "question_count": len(rows),
        }
        tickets.append(
            {
                **ticket_preimage,
                "ticket_id_sha256": canonical_sha256(ticket_preimage),
                "leaf_preflight_eligible": True,
                "gpu_authorized": False,
            }
        )

    return {
        "schema_version": DECISION_VERSION,
        "status": "expert_leaf_preflight_ready",
        "stage": "expert_leaf_release",
        "contract_sha256": contract_result["contract_sha256"],
        "planner_training_receipt_sha256": training_sha,
        "planner_freeze_receipt_sha256": freeze_sha,
        "planner_question_manifest_sha256": manifest_sha,
        "planner_question_generation_receipt_sha256": generation_sha,
        "planner_adapter_sha256": freeze["planner_adapter_sha256"],
        "expert_execution_policy": {
            "strictly_serial": True,
            "max_concurrency": 1,
            "planner_reentry_after_expert_start": False,
        },
        "expert_tickets": tickets,
        "existing_training_runtime_modified": False,
        "planner_first_leaf_launcher_integrated": False,
        "expert_training_execution_blocked": True,
        "gpu_authorized": False,
        "training_authorized": False,
        "formal": False,
    }


def load_contract(path: Path, *, repo_root: Path) -> Mapping[str, Any]:
    """Load the contract through a no-reparse stable read."""

    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise PlannerFirstCurriculumError("contract_outside_repo") from exc
    return _load_json_bytes(
        _stable_read(_repo_file(repo_root, relative)),
        "contract",
    )


def load_runtime_bundle(path: Path, *, repo_root: Path) -> Mapping[str, Any]:
    """Load a runtime bundle through a no-reparse stable read."""

    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise PlannerFirstCurriculumError("runtime_bundle_outside_repo") from exc
    return _load_json_bytes(
        _stable_read(_repo_file(repo_root, relative)),
        "runtime_bundle",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("--runtime-bundle", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo_root = args.repo_root.absolute()
    contract = load_contract(args.contract.absolute(), repo_root=repo_root)
    if args.runtime_bundle is None:
        result = {
            "schema_version": DECISION_VERSION,
            "status": "blocked_planner_training_receipt_required",
            "blockers": ["planner_training_receipt_required"],
            "existing_training_runtime_modified": False,
            "gpu_authorized": False,
            "training_authorized": False,
            "formal": False,
        }
        exit_code = 2
    else:
        bundle = load_runtime_bundle(
            args.runtime_bundle.absolute(),
            repo_root=repo_root,
        )
        result = build_runtime_decision(contract, bundle, repo_root=repo_root)
        exit_code = 0
    print(canonical_json_bytes(result).decode("utf-8"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
