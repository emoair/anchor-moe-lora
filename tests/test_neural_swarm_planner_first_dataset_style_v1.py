from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from anchor_mvp.research.neural_swarm_planner_first_dataset_style_v1 import (
    HARD_NEGATIVE_KINDS,
    PlannerFirstDatasetStyleError,
    build_manifest,
    build_projection_record,
    build_route_hard_negative,
    canonical_sha256,
    validate_config,
    validate_manifest,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/research/neural_swarm_planner_first_dataset_style_v1.json"
CONFIG_SCHEMA = (
    REPO / "configs/research/neural_swarm_planner_first_dataset_style_v1.schema.json"
)
RECORD_SCHEMA = (
    REPO / "configs/research/neural_swarm_planner_first_question_record_v2.schema.json"
)
MANIFEST_SCHEMA = (
    REPO
    / "configs/research/neural_swarm_planner_first_question_manifest_v2.schema.json"
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _bundle_rows(
    rows: list[dict],
    *,
    name: str,
    split: str,
    language: str,
    style_expert: str,
    variant: str,
    identity_required: bool,
    source_group_seed: str | None = None,
) -> list[dict]:
    tool_required = variant in {"with_tool", "with_tool_review"}
    review_required = variant in {"with_review", "with_tool_review"}
    assets = [style_expert]
    if tool_required:
        assets.append("tool_call")
    if review_required:
        assets.append("review_audit")
    identity_class = "air" if identity_required else None
    identity_invariant = _sha(f"{name}:identity") if identity_required else None
    payload = _sha(f"{name}:payload")
    semantic = _sha(f"{name}:semantic")
    for asset in assets:
        salient = _sha(f"{name}:{asset}:salient")
        rows.append(
            build_projection_record(
                ordinal=len(rows),
                split=split,
                language=language,
                task_bundle_sha256=_sha(f"{name}:bundle"),
                source_record_id_sha256=_sha(f"{name}:{asset}:source"),
                source_group_sha256=_sha(f"{source_group_seed or name}:source-group"),
                leakage_family_sha256=_sha(f"{name}:leakage-family"),
                prompt_digest_sha256=_sha(f"{name}:prompt"),
                planner_freeze_receipt_sha256=_sha("planner-freeze"),
                planner_adapter_sha256=_sha("planner-adapter"),
                planner_question_generation_receipt_sha256=_sha("question-generation"),
                training_expert_asset=asset,
                style_expert=style_expert,
                variant=variant,
                identity_required=identity_required,
                identity_provenance_class=identity_class,
                identity_invariant_sha256=identity_invariant,
                question_semantic_sha256=semantic,
                question_payload_sha256=payload,
                shared_prefix_segment_ids=[
                    _sha(f"{name}:shared"),
                    salient,
                ],
                salient_segment_ids=[salient],
                distractor_segment_ids=[_sha(f"{name}:{asset}:distractor")],
                route_relevant_evidence_refs=[salient],
            )
        )
    return rows[-len(assets) :]


def _records_and_negatives() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    base_identity = _bundle_rows(
        rows,
        name="serious-identity-base",
        split="train",
        language="en",
        style_expert="serious",
        variant="base",
        identity_required=True,
    )
    review_identity = _bundle_rows(
        rows,
        name="serious-identity-review",
        split="train",
        language="zh-CN",
        style_expert="serious",
        variant="with_review",
        identity_required=True,
    )
    _bundle_rows(
        rows,
        name="serious-identity-tool-review",
        split="eval_proxy",
        language="zh-CN",
        style_expert="serious",
        variant="with_tool_review",
        identity_required=True,
    )
    humor_base = _bundle_rows(
        rows,
        name="humor-base",
        split="train",
        language="en",
        style_expert="humor",
        variant="base",
        identity_required=False,
    )
    _bundle_rows(
        rows,
        name="angry-base",
        split="calibration",
        language="zh-CN",
        style_expert="angry_style",
        variant="base",
        identity_required=False,
    )
    tool_identity = _bundle_rows(
        rows,
        name="serious-identity-tool",
        split="calibration",
        language="en",
        style_expert="serious",
        variant="with_tool",
        identity_required=True,
    )
    negatives = [
        build_route_hard_negative(
            base_identity[0],
            negative_kind="wrong_primary_style",
        ),
        build_route_hard_negative(
            tool_identity[0],
            negative_kind="missing_required_tool",
        ),
        build_route_hard_negative(
            humor_base[0],
            negative_kind="unnecessary_tool",
        ),
        build_route_hard_negative(
            review_identity[0],
            negative_kind="missing_required_review",
        ),
        build_route_hard_negative(
            humor_base[0],
            negative_kind="unnecessary_review",
        ),
        build_route_hard_negative(
            base_identity[0],
            negative_kind="identity_bit_dropped",
        ),
        build_route_hard_negative(
            base_identity[0],
            negative_kind="identity_false_attribution",
        ),
    ]
    return rows, negatives


def _manifest() -> dict:
    records, negatives = _records_and_negatives()
    return build_manifest(
        _config(),
        repo_root=REPO,
        records=records,
        route_hard_negatives=negatives,
        historical_invalidated_inventory_sha256=_sha("invalidated-inventory"),
        heldout_inventory_sha256=_sha("heldout-inventory"),
        max_observed_dice=0.83,
    )


def test_schemas_and_config_are_closed_and_valid() -> None:
    config = _config()
    config_schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    record_schema = json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
    manifest_schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator.check_schema(config_schema)
    jsonschema.Draft202012Validator.check_schema(record_schema)
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    jsonschema.validate(config, config_schema)
    records, _ = _records_and_negatives()
    for row in records:
        jsonschema.validate(row, record_schema)
    registry = Registry().with_resource(
        record_schema["$id"],
        Resource.from_contents(record_schema),
    )
    manifest_validator = jsonschema.Draft202012Validator(
        manifest_schema,
        registry=registry,
    )
    manifest_validator.validate(_manifest())
    unknown_count = _manifest()
    unknown_count["counts"]["splits"]["unexpected"] = 0
    with pytest.raises(jsonschema.ValidationError):
        manifest_validator.validate(unknown_count)
    assert config_schema["additionalProperties"] is False
    assert record_schema["additionalProperties"] is False
    assert manifest_schema["additionalProperties"] is False


def test_config_authenticates_parent_and_is_non_authorizing() -> None:
    decision = validate_config(_config(), repo_root=REPO)
    assert decision["status"] == "dataset_style_contract_valid_execution_blocked"
    assert decision["existing_training_runtime_modified"] is False
    assert decision["training_authorized"] is False
    assert decision["formal"] is False


def test_valid_manifest_preserves_one_row_per_expert_and_bundle_route() -> None:
    manifest = _manifest()
    decision = validate_manifest(_config(), manifest, repo_root=REPO)
    assert decision["status"] == "dataset_style_ready_runtime_integration_blocked"
    assert decision["legacy_one_expert_per_question_preserved"] is True
    assert decision["candidate_inventory_interpreted_as_activation_set"] is False
    assert decision["record_count"] == 10
    assert decision["bundle_count"] == 6
    assert decision["hard_negative_count"] == len(HARD_NEGATIVE_KINDS)
    assert decision["training_authorized"] is False


def test_candidate_inventory_does_not_activate_every_expert() -> None:
    manifest = _manifest()
    base = manifest["records"][0]
    assert base["route_plan"]["candidate_expert_assets"] == [
        "humor",
        "serious",
        "angry_style",
        "tool_call",
        "review_audit",
    ]
    assert base["route_plan"]["selected_expert_assets"] == ["serious"]
    assert base["training_expert_asset"] == "serious"


def test_missing_selected_sidecar_fails_bundle_closure() -> None:
    manifest = _manifest()
    semantic = next(
        row["semantic_question_id_sha256"]
        for row in manifest["records"]
        if row["variant"] == "with_tool"
    )
    manifest["records"] = [
        row
        for row in manifest["records"]
        if not (
            row["semantic_question_id_sha256"] == semantic
            and row["training_expert_asset"] == "tool_call"
        )
    ]
    manifest["record_count"] -= 1
    manifest["ordered_record_ids_sha256"] = canonical_sha256(
        [row["record_id_sha256"] for row in manifest["records"]]
    )
    manifest["record_inventory_sha256"] = canonical_sha256(manifest["records"])
    with pytest.raises(
        PlannerFirstDatasetStyleError,
        match="semantic_bundle_projection_not_exact_selected_set",
    ):
        validate_manifest(_config(), manifest, repo_root=REPO)


def test_cross_split_source_group_is_rejected() -> None:
    records, negatives = _records_and_negatives()
    _bundle_rows(
        records,
        name="cross-split-collision",
        split="calibration",
        language="en",
        style_expert="humor",
        variant="base",
        identity_required=False,
        source_group_seed="serious-identity-base",
    )
    manifest = build_manifest(
        _config(),
        repo_root=REPO,
        records=records,
        route_hard_negatives=negatives,
        historical_invalidated_inventory_sha256=_sha("invalidated-inventory"),
        heldout_inventory_sha256=_sha("heldout-inventory"),
        max_observed_dice=0.83,
    )
    with pytest.raises(
        PlannerFirstDatasetStyleError,
        match="cross_split_source_group_sha256_overlap",
    ):
        validate_manifest(_config(), manifest, repo_root=REPO)


def test_hard_negative_must_change_exactly_one_route_variable() -> None:
    manifest = _manifest()
    negative = manifest["route_hard_negatives"][0]
    negative["negative_route_plan"]["tool_required"] = True
    negative["negative_route_plan"]["selected_expert_assets"].append("tool_call")
    with pytest.raises(PlannerFirstDatasetStyleError):
        validate_manifest(_config(), manifest, repo_root=REPO)


def test_identity_metadata_is_required_and_is_not_an_expert_asset() -> None:
    with pytest.raises(
        PlannerFirstDatasetStyleError,
        match="identity_provenance_class_required",
    ):
        build_projection_record(
            ordinal=0,
            split="train",
            language="en",
            task_bundle_sha256=_sha("bundle"),
            source_record_id_sha256=_sha("source"),
            source_group_sha256=_sha("source-group"),
            leakage_family_sha256=_sha("leakage"),
            prompt_digest_sha256=_sha("prompt"),
            planner_freeze_receipt_sha256=_sha("freeze"),
            planner_adapter_sha256=_sha("adapter"),
            planner_question_generation_receipt_sha256=_sha("generation"),
            training_expert_asset="serious",
            style_expert="serious",
            variant="base",
            identity_required=True,
            identity_provenance_class=None,
            identity_invariant_sha256=None,
            question_semantic_sha256=_sha("semantic"),
            question_payload_sha256=_sha("payload"),
            shared_prefix_segment_ids=[_sha("salient")],
            salient_segment_ids=[_sha("salient")],
            distractor_segment_ids=[],
            route_relevant_evidence_refs=[_sha("salient")],
        )
    assert "identity" not in _config()["dataset_semantics"]["expert_assets"]


def test_target_future_and_raw_token_claims_fail_closed() -> None:
    manifest = _manifest()
    manifest["records"][0]["claims"]["target_visible"] = True
    with pytest.raises(
        PlannerFirstDatasetStyleError,
        match="projection_record_identity_mismatch",
    ):
        validate_manifest(_config(), manifest, repo_root=REPO)


def test_config_parent_pin_drift_fails() -> None:
    config = _config()
    config["parent_contract"]["implementation"]["sha256"] = _sha("wrong")
    with pytest.raises(
        PlannerFirstDatasetStyleError,
        match="parent_implementation_identity_drift",
    ):
        validate_config(config, repo_root=REPO)


def test_holdout_is_inventory_only_not_a_record_split() -> None:
    config = _config()
    assert config["partition_policy"]["heldout_rows_permitted"] is False
    assert config["partition_policy"]["heldout_inventory_only"] is True
    assert "holdout" not in config["partition_policy"]["splits"]
    assert _manifest()["isolation_proof"]["holdout_body_read"] is False
