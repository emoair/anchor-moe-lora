"""Authenticate and consume producer-native hierarchical TaskBoard KV plans.

The SWE-bench TaskBoard projector owns the canonical segment plan.  This
module never derives a second plan and never serializes source block text.  It
validates the embedded plan against the real causal training view and emits a
small, content-free index that a future KV producer can bind to physical model
and tokenizer identities.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from anchor_mvp.research.query_specialization import (
    ROLES,
    QuerySpecializationError,
    QueryTrainingRecord,
    TaskBoardSidecar,
    build_training_view,
    load_taskboard_sidecar_dataset,
)


PLAN_SCHEMA_VERSION = "anchor.hierarchical-task-kv-segment-plan.v1"
SIDECAR_SCHEMA_VERSION = "anchor.swebench-taskboard-sidecar.v2"
PROJECTOR_MANIFEST_SCHEMA_VERSION = (
    "anchor.swebench-taskboard-projector-manifest.v2"
)
PROJECTOR_CONFIG_SCHEMA_VERSION = "anchor.swebench-taskboard-projector-config.v2"
PROJECTOR_VERSION = "anchor.swebench-taskboard-projector.v2"
CONSUMER_CONFIG_SCHEMA_VERSION = (
    "anchor.hierarchical-task-kv-mvp-consumer-config.v2"
)
INDEX_SCHEMA_VERSION = "anchor.hierarchical-task-kv-consumer-index.v1"
EXECUTION_MODE = "decoupled_frozen_prefix_producer_required"
CLAIM_SCOPE = "authenticated_native_plan_index_only"
DEFAULT_CONSUMER_MAX_SHARD_BYTES = 48_000_000
FROZEN_PRODUCER_CONTRACT_SHA256 = {
    "producer_config": (
        "b36945a2693183f0b213da403afcf8bb5611f46298bb849434e7b7d5854ba943"
    ),
    "manifest_schema": (
        "2cd9dc98d2b2865ed0586abfe291e3f6d161686597fcd2a7884c5762d2195347"
    ),
    "sidecar_schema": (
        "c1863bfab69ce2f2388ee37fadae951b14f3d5120706bab032cab3f9aab6bdc5"
    ),
    "segment_plan_schema": (
        "80f760497e0d21f7d4d532db758362a800e845e6919b18b23958caabc7f155bf"
    ),
}
FROZEN_CONSUMER_CONFIG_SHA256 = (
    "f695e02cd2da8ca9c8d40fc99a0c33a4803b23dbc5e7f4cf296d40156315252d"
)

CACHE_SCOPES = (
    "task_shared_prefix",
    "downstream_task_shared_immutable",
    "expert_private_delta",
)
CACHE_IDENTITY_FIELDS = (
    "model_architecture_sha256",
    "tokenizer_sha256",
    "token_order_sha256",
    "position_ids_sha256",
    "rope_config_sha256",
    "kv_producing_weights_sha256",
    "prefix_lineage_sha256",
)
STAGE_TO_EXPERT = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "domain_builder": "frontend_gen",
    "domain_review": "frontend_review",
    "security": "security_gate",
}
FIXED_PARTITIONS = (
    ("train/clean.jsonl", "train", "clean"),
    ("train/noisy.jsonl", "train", "noisy"),
    ("calibration/clean.jsonl", "calibration", "clean"),
)
NON_CLAIMS = (
    "physical_kv_tensors_materialized",
    "full_generation_kv_shared",
    "token_level_moe",
    "naive_in_stack_q_lora_exact_reuse",
    "cache_reuse_before_identity_binding",
    "latency_or_memory_speedup",
)

PLAN_FIELDS = {
    "schema_version",
    "architecture",
    "execution_mode",
    "materialization",
    "full_generation_kv_shared_claimed",
    "token_level_moe_claimed",
    "split_before_augmentation",
    "augmentation_applied_after_split",
    "bindings",
    "shared_prefix_policy",
    "target_delta_policy",
    "cache_compatibility",
    "segments",
}
BINDING_FIELDS = {
    "task_bundle_sha256",
    "task_id",
    "base_task_board_sha256",
    "projector_version",
    "config_sha256",
    "sidecar_schema_sha256",
    "segment_plan_schema_sha256",
    "source_gold_sha256",
    "source_gold_file_sha256",
    "source_snapshot_sha256",
    "source_snapshot_manifest_sha256",
    "split",
    "stage",
    "expert",
    "variant",
}
SEGMENT_FIELDS = {
    "segment_id",
    "content_sha256",
    "source_block_id",
    "serialization_order",
    "causal_order",
    "producer_role",
    "cache_scope",
    "visibility",
    "dependencies",
    "commit_state",
    "parent_segment_id",
    "parent_lineage_sha256",
    "prefix_lineage_sha256",
}
SHARED_PREFIX_POLICY = {
    "membership_rule": "strict_all_five_role_visibility_intersection",
    "ordered_prefix_chain": True,
    "independent_segment_concatenation_allowed": False,
    "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
    "shared_then_mask_allowed": False,
    "forbidden_current_future_preinsert_allowed": False,
}
TARGET_DELTA_POLICY = {
    "initial_cache_scope": "expert_private_delta",
    "promotion_requires": "explicit_committed_and_causally_visible_downstream",
    "promoted_cache_scope": "downstream_task_shared_immutable",
    "current_target_segment_emitted": False,
}
CACHE_COMPATIBILITY = {
    "status": "identity_unbound",
    "cache_reuse_allowed": False,
    "required_exact_match_fields": list(CACHE_IDENTITY_FIELDS),
    "mismatch_result": "cache_incompatible",
    "unknown_result": "cache_incompatible",
    "q_specialization_alone_sufficient_for_exact_reuse": False,
    "naive_in_stack_q_lora_exact_reuse_allowed": False,
}
INDEX_FIELDS = {
    "schema_version",
    "claim_scope",
    "source_plan_location",
    "source_sidecar_record_id",
    "source_gold_record_id",
    "source_gold_sha256",
    "source_gold_file_sha256",
    "source_snapshot_sha256",
    "source_snapshot_manifest_sha256",
    "task_bundle_sha256",
    "base_task_board_sha256",
    "projector_version",
    "projector_config_sha256",
    "sidecar_schema_sha256",
    "segment_plan_schema_version",
    "segment_plan_schema_sha256",
    "canonical_segment_plan_sha256",
    "split",
    "stage",
    "expert",
    "variant",
    "task_id",
    "segment_count",
    "segment_ids_sha256",
    "terminal_prefix_lineage_sha256",
    "segments_by_cache_scope",
    "cache_compatibility_status",
    "cache_reuse_allowed",
}

_SHA256_CHARS = frozenset("0123456789abcdef")


class TaskBoardKVSegmentError(ValueError):
    """Raised when producer-native Task-KV metadata fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise TaskBoardKVSegmentError(f"{path} must be a lowercase SHA-256")
    return value


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskBoardKVSegmentError(f"{path} must be non-empty text")
    return value.strip()


def _require_exact_mapping(
    value: Any, fields: set[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskBoardKVSegmentError(f"{path} must be an object")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise TaskBoardKVSegmentError(
            f"{path} fields changed (missing={missing}, unknown={unknown})"
        )
    return value


def _require_unique_text_list(value: Any, path: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise TaskBoardKVSegmentError(
            f"{path} must be a list of unique non-empty strings"
        )
    return list(value)


def _reject_content_fields(value: Any, path: str = "segment_plan") -> None:
    if isinstance(value, Mapping):
        if "content" in value:
            raise TaskBoardKVSegmentError(
                f"{path} must contain metadata only, never source body text"
            )
        for key, item in value.items():
            _reject_content_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_content_fields(item, f"{path}[{index}]")


def _segment_id(
    *,
    task_bundle_sha256: str,
    source_block_id: str,
    content_sha256: str,
    producer_role: str,
    cache_scope: str,
) -> str:
    return "task-kv-segment-v1:" + _canonical_sha256(
        {
            "task_bundle_sha256": task_bundle_sha256,
            "source_block_id": source_block_id,
            "content_sha256": content_sha256,
            "producer_role": producer_role,
            "cache_scope": cache_scope,
        }
    )


def _binding_from_outer(
    outer: TaskBoardSidecar, record: QueryTrainingRecord
) -> dict[str, str]:
    return {
        "task_bundle_sha256": outer.task_bundle_sha256,
        "task_id": record.task_id,
        "base_task_board_sha256": outer.base_task_board_sha256,
        "projector_version": outer.projector_version,
        "config_sha256": outer.config_sha256,
        "sidecar_schema_sha256": outer.sidecar_schema_sha256,
        "segment_plan_schema_sha256": outer.segment_plan_schema_sha256,
        "source_gold_sha256": outer.source_gold_sha256,
        "source_gold_file_sha256": outer.source_gold_file_sha256,
        "source_snapshot_sha256": outer.source_snapshot_sha256,
        "source_snapshot_manifest_sha256": outer.source_snapshot_manifest_sha256,
        "split": outer.split,
        "stage": outer.stage,
        "expert": outer.expert,
        "variant": outer.variant,
    }


def validate_plan_mapping(
    value: Any,
    *,
    outer: TaskBoardSidecar | None = None,
    record: QueryTrainingRecord | None = None,
) -> str:
    """Validate one canonical producer-native plan and return its SHA-256.

    When ``record`` is supplied, validation uses the real
    :func:`build_training_view` materializer and recomputes every segment from
    the source board.  No wrapper or complete task board is ever stringified
    into a training prompt by this consumer.
    """

    plan = _require_exact_mapping(value, PLAN_FIELDS, "segment_plan")
    _reject_content_fields(plan)
    fixed = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "architecture": "hierarchical_task_kv",
        "execution_mode": EXECUTION_MODE,
        "materialization": "metadata_only_no_tensor_or_kv",
        "full_generation_kv_shared_claimed": False,
        "token_level_moe_claimed": False,
        "split_before_augmentation": True,
        "augmentation_applied_after_split": True,
    }
    if any(plan.get(key) != expected for key, expected in fixed.items()):
        raise TaskBoardKVSegmentError("segment_plan fixed execution contract changed")

    bindings = _require_exact_mapping(
        plan.get("bindings"), BINDING_FIELDS, "segment_plan.bindings"
    )
    for key in (
        "task_bundle_sha256",
        "base_task_board_sha256",
        "config_sha256",
        "sidecar_schema_sha256",
        "segment_plan_schema_sha256",
        "source_gold_sha256",
        "source_gold_file_sha256",
        "source_snapshot_sha256",
        "source_snapshot_manifest_sha256",
    ):
        _require_sha256(bindings.get(key), f"segment_plan.bindings.{key}")
    for key in ("task_id", "projector_version", "split", "stage", "expert", "variant"):
        _require_text(bindings.get(key), f"segment_plan.bindings.{key}")
    if bindings.get("projector_version") != PROJECTOR_VERSION:
        raise TaskBoardKVSegmentError("segment_plan projector version changed")
    if bindings.get("stage") not in STAGE_TO_EXPERT:
        raise TaskBoardKVSegmentError("segment_plan stage is unknown")
    if bindings.get("expert") != STAGE_TO_EXPERT[bindings["stage"]]:
        raise TaskBoardKVSegmentError("segment_plan stage/expert binding changed")
    if bindings.get("split") not in {"train", "calibration"}:
        raise TaskBoardKVSegmentError("segment_plan split is unknown")
    if bindings.get("variant") not in {"clean", "noisy"}:
        raise TaskBoardKVSegmentError("segment_plan variant is unknown")

    if outer is not None:
        bound_record = record if record is not None else outer.training_record
        expected_bindings = _binding_from_outer(outer, bound_record)
        if dict(bindings) != expected_bindings:
            raise TaskBoardKVSegmentError(
                "segment_plan bindings do not match the authenticated outer sidecar"
            )
        if outer.segment_plan_schema_sha256 != bindings["segment_plan_schema_sha256"]:
            raise TaskBoardKVSegmentError("segment-plan schema cross-binding changed")

    fixed_mappings = (
        ("shared_prefix_policy", SHARED_PREFIX_POLICY),
        ("target_delta_policy", TARGET_DELTA_POLICY),
        ("cache_compatibility", CACHE_COMPATIBILITY),
    )
    for field, expected in fixed_mappings:
        raw = plan.get(field)
        if not isinstance(raw, Mapping) or dict(raw) != expected:
            raise TaskBoardKVSegmentError(f"segment_plan.{field} contract changed")

    raw_segments = plan.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise TaskBoardKVSegmentError("segment_plan.segments must not be empty")

    prior_ids: list[str] = []
    parent_segment_id: str | None = None
    parent_lineage = _canonical_sha256(
        {
            "task_bundle_sha256": bindings["task_bundle_sha256"],
            "execution_mode": EXECUTION_MODE,
            "root": "ordered_prefix_genesis",
        }
    )
    source_ids: list[str] = []
    causal_orders: list[int] = []
    shared_prefix_open = True
    for serialization_order, raw_segment in enumerate(raw_segments):
        path = f"segment_plan.segments[{serialization_order}]"
        segment = _require_exact_mapping(raw_segment, SEGMENT_FIELDS, path)
        source_id = _require_text(segment.get("source_block_id"), f"{path}.source_block_id")
        content_sha = _require_sha256(segment.get("content_sha256"), f"{path}.content_sha256")
        producer_role = _require_text(segment.get("producer_role"), f"{path}.producer_role")
        cache_scope = _require_text(segment.get("cache_scope"), f"{path}.cache_scope")
        if producer_role not in {"task_source", *ROLES}:
            raise TaskBoardKVSegmentError(f"{path}.producer_role is unknown")
        if cache_scope not in CACHE_SCOPES:
            raise TaskBoardKVSegmentError(f"{path}.cache_scope is unknown")
        if cache_scope == "task_shared_prefix":
            if not shared_prefix_open:
                raise TaskBoardKVSegmentError(
                    "task-shared segments must form one leading ordered prefix"
                )
        else:
            shared_prefix_open = False
        raw_serialization_order = segment.get("serialization_order")
        if (
            isinstance(raw_serialization_order, bool)
            or not isinstance(raw_serialization_order, int)
            or raw_serialization_order != serialization_order
        ):
            raise TaskBoardKVSegmentError(f"{path}.serialization_order changed")
        causal_order = segment.get("causal_order")
        if isinstance(causal_order, bool) or not isinstance(causal_order, int) or causal_order < 0:
            raise TaskBoardKVSegmentError(f"{path}.causal_order must be non-negative")
        if causal_orders and causal_order <= causal_orders[-1]:
            raise TaskBoardKVSegmentError("segment causal order must be strictly increasing")
        causal_orders.append(causal_order)
        visibility = _require_unique_text_list(segment.get("visibility"), f"{path}.visibility")
        if any(role not in ROLES for role in visibility):
            raise TaskBoardKVSegmentError(f"{path}.visibility contains an unknown expert")
        dependencies = _require_unique_text_list(
            segment.get("dependencies"), f"{path}.dependencies"
        )
        if dependencies != prior_ids:
            raise TaskBoardKVSegmentError(f"{path}.dependencies changed")
        if segment.get("parent_segment_id") != parent_segment_id:
            raise TaskBoardKVSegmentError(f"{path}.parent_segment_id changed")
        if segment.get("parent_lineage_sha256") != parent_lineage:
            raise TaskBoardKVSegmentError(f"{path}.parent_lineage_sha256 changed")
        expected_id = _segment_id(
            task_bundle_sha256=bindings["task_bundle_sha256"],
            source_block_id=source_id,
            content_sha256=content_sha,
            producer_role=producer_role,
            cache_scope=cache_scope,
        )
        if segment.get("segment_id") != expected_id:
            raise TaskBoardKVSegmentError(f"{path}.segment_id changed")
        expected_lineage = _canonical_sha256(
            {
                "parent_lineage_sha256": parent_lineage,
                "segment_id": expected_id,
                "serialization_order": serialization_order,
                "causal_order": causal_order,
            }
        )
        if segment.get("prefix_lineage_sha256") != expected_lineage:
            raise TaskBoardKVSegmentError(f"{path}.prefix_lineage_sha256 changed")
        if cache_scope == "task_shared_prefix" and (
            producer_role != "task_source"
            or segment.get("commit_state") != "committed"
            or visibility != list(ROLES)
        ):
            raise TaskBoardKVSegmentError(f"{path}: invalid shared-prefix metadata")
        if cache_scope == "downstream_task_shared_immutable" and (
            producer_role not in ROLES or segment.get("commit_state") != "committed"
        ):
            raise TaskBoardKVSegmentError(f"{path}: invalid downstream metadata")
        if cache_scope == "expert_private_delta" and (
            producer_role != bindings["expert"]
            or segment.get("commit_state") not in {"candidate", "verified"}
            or visibility != [bindings["expert"]]
        ):
            raise TaskBoardKVSegmentError(f"{path}: invalid private-delta metadata")
        if source_id in source_ids:
            raise TaskBoardKVSegmentError("segment source block ids must be unique")
        source_ids.append(source_id)
        prior_ids.append(expected_id)
        parent_segment_id = expected_id
        parent_lineage = expected_lineage

    if record is not None:
        if bindings["task_id"] != record.task_id or bindings["expert"] != record.role:
            raise TaskBoardKVSegmentError("segment_plan does not bind its inner record")
        by_id = {block.block_id: block for block in record.blocks}
        board_order = {block.block_id: index for index, block in enumerate(record.blocks)}
        allowed_ids = [*record.targets.relevant, *record.targets.distractors]
        expected_source_ids = sorted(allowed_ids, key=board_order.__getitem__)
        forbidden_ids = set(record.targets.forbidden)
        if not forbidden_ids.issubset(by_id):
            raise TaskBoardKVSegmentError("forbidden ids are not physically present")
        if set(source_ids) & forbidden_ids:
            raise TaskBoardKVSegmentError(
                "forbidden/current/future blocks must never enter a segment plan"
            )
        if source_ids != expected_source_ids:
            raise TaskBoardKVSegmentError(
                "segments must exactly cover relevant and distractor blocks in board order"
            )
        view = build_training_view(record)
        if list(view.visible_block_ids) != expected_source_ids:
            raise TaskBoardKVSegmentError(
                "real training-view materializer disagrees with segment sources"
            )
        try:
            prompt = json.loads(view.prompt)
            target = json.loads(record.target_output)
        except json.JSONDecodeError as exc:  # pragma: no cover - parser invariant
            raise TaskBoardKVSegmentError("training view is not canonical JSON") from exc
        prompt_blocks = prompt.get("blocks") if isinstance(prompt, Mapping) else None
        if not isinstance(prompt_blocks, list):
            raise TaskBoardKVSegmentError("training view omitted its visible blocks")
        prompt_ids = [block.get("id") for block in prompt_blocks if isinstance(block, Mapping)]
        if prompt_ids != expected_source_ids or forbidden_ids & set(prompt_ids):
            raise TaskBoardKVSegmentError(
                "training prompt includes a forbidden block or changed ordering"
            )
        answer = target.get("answer") if isinstance(target, Mapping) else None
        if not isinstance(answer, str) or not answer:
            raise TaskBoardKVSegmentError("training target answer is missing")
        if answer in view.prompt:
            raise TaskBoardKVSegmentError("current target answer leaked into the prompt")
        forbidden_content_hashes = {
            _sha256_bytes(by_id[block_id].content.encode("utf-8"))
            for block_id in forbidden_ids
        }
        segment_content_hashes = {
            str(segment["content_sha256"]) for segment in raw_segments
        }
        if forbidden_content_hashes & segment_content_hashes:
            raise TaskBoardKVSegmentError(
                "forbidden/current/future content was inserted under another block id"
            )
        distractors = set(record.targets.distractors)
        for index, (segment, source_id) in enumerate(
            zip(raw_segments, expected_source_ids, strict=True)
        ):
            block = by_id[source_id]
            expected_content_sha = _sha256_bytes(block.content.encode("utf-8"))
            if segment["content_sha256"] != expected_content_sha:
                raise TaskBoardKVSegmentError(
                    f"segment_plan.segments[{index}].content_sha256 changed"
                )
            if segment["causal_order"] != board_order[source_id]:
                raise TaskBoardKVSegmentError(
                    f"segment_plan.segments[{index}].causal_order changed"
                )
            if source_id in distractors:
                expected_scope = "expert_private_delta"
                expected_producer = record.role
            elif block.visible_to == ROLES:
                expected_scope = "task_shared_prefix"
                expected_producer = "task_source"
            else:
                expected_scope = "downstream_task_shared_immutable"
                try:
                    first_visible = min(ROLES.index(role) for role in block.visible_to)
                except ValueError as exc:
                    raise TaskBoardKVSegmentError(
                        "downstream visibility contains an unknown role"
                    ) from exc
                if first_visible < 1:
                    raise TaskBoardKVSegmentError(
                        "downstream segment has no causally prior producer"
                    )
                expected_producer = ROLES[first_visible - 1]
            if (
                segment["cache_scope"] != expected_scope
                or segment["producer_role"] != expected_producer
                or segment["visibility"] != list(block.visible_to)
                or segment["commit_state"] != block.commit_state
            ):
                raise TaskBoardKVSegmentError(
                    f"segment_plan.segments[{index}] source semantics changed"
                )

    return _canonical_sha256(plan)


@dataclass(frozen=True)
class TaskBoardKVSegmentPlan:
    """Content-free authenticated index for one producer-native plan."""

    source_sidecar_record_id: str
    source_gold_record_id: str
    source_gold_sha256: str
    source_gold_file_sha256: str
    source_snapshot_sha256: str
    source_snapshot_manifest_sha256: str
    task_bundle_sha256: str
    base_task_board_sha256: str
    projector_version: str
    projector_config_sha256: str
    sidecar_schema_sha256: str
    segment_plan_schema_sha256: str
    canonical_segment_plan_sha256: str
    split: str
    stage: str
    expert: str
    variant: str
    task_id: str
    segment_count: int
    segment_ids_sha256: str
    terminal_prefix_lineage_sha256: str
    segments_by_cache_scope: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "source_plan_location": "outer_sidecar.segment_plan",
            "source_sidecar_record_id": self.source_sidecar_record_id,
            "source_gold_record_id": self.source_gold_record_id,
            "source_gold_sha256": self.source_gold_sha256,
            "source_gold_file_sha256": self.source_gold_file_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "source_snapshot_manifest_sha256": self.source_snapshot_manifest_sha256,
            "task_bundle_sha256": self.task_bundle_sha256,
            "base_task_board_sha256": self.base_task_board_sha256,
            "projector_version": self.projector_version,
            "projector_config_sha256": self.projector_config_sha256,
            "sidecar_schema_sha256": self.sidecar_schema_sha256,
            "segment_plan_schema_version": PLAN_SCHEMA_VERSION,
            "segment_plan_schema_sha256": self.segment_plan_schema_sha256,
            "canonical_segment_plan_sha256": self.canonical_segment_plan_sha256,
            "split": self.split,
            "stage": self.stage,
            "expert": self.expert,
            "variant": self.variant,
            "task_id": self.task_id,
            "segment_count": self.segment_count,
            "segment_ids_sha256": self.segment_ids_sha256,
            "terminal_prefix_lineage_sha256": self.terminal_prefix_lineage_sha256,
            "segments_by_cache_scope": dict(self.segments_by_cache_scope),
            "cache_compatibility_status": "identity_unbound",
            "cache_reuse_allowed": False,
        }


def validate_index_mapping(value: Any) -> None:
    """Validate the content-free materialized consumer index."""

    index = _require_exact_mapping(value, INDEX_FIELDS, "consumer_index")
    _reject_content_fields(index, "consumer_index")
    fixed = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "source_plan_location": "outer_sidecar.segment_plan",
        "segment_plan_schema_version": PLAN_SCHEMA_VERSION,
        "cache_compatibility_status": "identity_unbound",
        "cache_reuse_allowed": False,
    }
    if any(index.get(key) != expected for key, expected in fixed.items()):
        raise TaskBoardKVSegmentError("consumer index fixed contract changed")
    for key in (
        "source_gold_sha256",
        "source_gold_file_sha256",
        "source_snapshot_sha256",
        "source_snapshot_manifest_sha256",
        "task_bundle_sha256",
        "base_task_board_sha256",
        "projector_config_sha256",
        "sidecar_schema_sha256",
        "segment_plan_schema_sha256",
        "canonical_segment_plan_sha256",
        "segment_ids_sha256",
        "terminal_prefix_lineage_sha256",
    ):
        _require_sha256(index.get(key), f"consumer_index.{key}")
    count = index.get("segment_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise TaskBoardKVSegmentError("consumer_index.segment_count must be positive")
    scopes = index.get("segments_by_cache_scope")
    if not isinstance(scopes, Mapping) or set(scopes) != set(CACHE_SCOPES):
        raise TaskBoardKVSegmentError("consumer_index cache-scope counts changed")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in scopes.values()):
        raise TaskBoardKVSegmentError("consumer_index cache-scope count is invalid")
    if sum(scopes.values()) != count:
        raise TaskBoardKVSegmentError("consumer_index segment count is inconsistent")


def _index_native_plan(sidecar: TaskBoardSidecar) -> TaskBoardKVSegmentPlan:
    plan = sidecar.segment_plan
    plan_sha = validate_plan_mapping(
        plan, outer=sidecar, record=sidecar.training_record
    )
    segments = plan["segments"]
    scope_counts = Counter(str(segment["cache_scope"]) for segment in segments)
    index = TaskBoardKVSegmentPlan(
        source_sidecar_record_id=sidecar.record_id,
        source_gold_record_id=sidecar.source_gold_record_id,
        source_gold_sha256=sidecar.source_gold_sha256,
        source_gold_file_sha256=sidecar.source_gold_file_sha256,
        source_snapshot_sha256=sidecar.source_snapshot_sha256,
        source_snapshot_manifest_sha256=sidecar.source_snapshot_manifest_sha256,
        task_bundle_sha256=sidecar.task_bundle_sha256,
        base_task_board_sha256=sidecar.base_task_board_sha256,
        projector_version=sidecar.projector_version,
        projector_config_sha256=sidecar.config_sha256,
        sidecar_schema_sha256=sidecar.sidecar_schema_sha256,
        segment_plan_schema_sha256=sidecar.segment_plan_schema_sha256,
        canonical_segment_plan_sha256=plan_sha,
        split=sidecar.split,
        stage=sidecar.stage,
        expert=sidecar.expert,
        variant=sidecar.variant,
        task_id=sidecar.training_record.task_id,
        segment_count=len(segments),
        segment_ids_sha256=_canonical_sha256(
            [segment["segment_id"] for segment in segments]
        ),
        terminal_prefix_lineage_sha256=segments[-1]["prefix_lineage_sha256"],
        segments_by_cache_scope=tuple(
            (scope, scope_counts.get(scope, 0)) for scope in CACHE_SCOPES
        ),
    )
    validate_index_mapping(index.to_dict())
    return index


def project_taskboard_kv_segment_plans(
    sidecars: Sequence[TaskBoardSidecar],
) -> tuple[TaskBoardKVSegmentPlan, ...]:
    """Compatibility name: consume native plans; never project new plans."""

    if not sidecars:
        raise TaskBoardKVSegmentError("native TaskBoard sidecars must not be empty")
    return tuple(_index_native_plan(sidecar) for sidecar in sidecars)


consume_native_taskboard_kv_segment_plans = project_taskboard_kv_segment_plans


def validate_native_sidecar_dataset(
    sidecars: Sequence[TaskBoardSidecar],
) -> dict[str, Any]:
    """Enforce bundle isolation and immutable segment-id metadata globally."""

    indexes = project_taskboard_kv_segment_plans(sidecars)
    record_ids: set[str] = set()
    bundle_splits: dict[str, set[str]] = {}
    bundle_task_ids: dict[str, set[str]] = {}
    task_bindings: dict[str, set[tuple[str, str]]] = {}
    source_hash_bindings: dict[str, set[tuple[str, str, str]]] = {}
    groups: dict[tuple[str, str], list[TaskBoardSidecar]] = {}
    segment_catalog: dict[str, bytes] = {}
    unique_scopes: dict[str, set[str]] = {scope: set() for scope in CACHE_SCOPES}
    segment_references = 0
    for sidecar in sidecars:
        if sidecar.record_id in record_ids:
            raise TaskBoardKVSegmentError("duplicate outer sidecar id")
        record_ids.add(sidecar.record_id)
        bundle_splits.setdefault(sidecar.task_bundle_sha256, set()).add(sidecar.split)
        bundle_task_ids.setdefault(sidecar.task_bundle_sha256, set()).add(
            sidecar.training_record.task_id
        )
        task_bindings.setdefault(sidecar.training_record.task_id, set()).add(
            (sidecar.task_bundle_sha256, sidecar.split)
        )
        source_hash_bindings.setdefault(sidecar.source_gold_sha256, set()).add(
            (
                sidecar.source_gold_record_id,
                sidecar.source_gold_file_sha256,
                sidecar.split,
            )
        )
        groups.setdefault((sidecar.task_bundle_sha256, sidecar.variant), []).append(
            sidecar
        )
        for segment in sidecar.segment_plan["segments"]:
            segment_id = segment["segment_id"]
            encoded = _canonical_bytes(segment)
            prior = segment_catalog.setdefault(segment_id, encoded)
            if prior != encoded:
                raise TaskBoardKVSegmentError(
                    "one segment id aliases different immutable metadata"
                )
            unique_scopes[segment["cache_scope"]].add(segment_id)
            segment_references += 1
    if any(len(splits) != 1 for splits in bundle_splits.values()):
        raise TaskBoardKVSegmentError("task_bundle_sha256 crossed dataset splits")
    if any(len(task_ids) != 1 for task_ids in bundle_task_ids.values()):
        raise TaskBoardKVSegmentError("task bundle aliases multiple task ids")
    if any(len(bindings) != 1 for bindings in task_bindings.values()):
        raise TaskBoardKVSegmentError("task id aliases bundles or crosses splits")
    if any(len(bindings) != 1 for bindings in source_hash_bindings.values()):
        raise TaskBoardKVSegmentError(
            "source Gold hash aliases records/files or crosses splits"
        )
    global_bindings = {
        (
            sidecar.source_snapshot_sha256,
            sidecar.source_snapshot_manifest_sha256,
            sidecar.projector_version,
            sidecar.config_sha256,
            sidecar.sidecar_schema_sha256,
            sidecar.segment_plan_schema_sha256,
        )
        for sidecar in sidecars
    }
    if len(global_bindings) != 1:
        raise TaskBoardKVSegmentError(
            "dataset mixes snapshot, projector, config, or schema bindings"
        )
    for (bundle, variant), group in groups.items():
        roles = [sidecar.expert for sidecar in group]
        if len(group) != len(ROLES) or set(roles) != set(ROLES):
            raise TaskBoardKVSegmentError(
                f"bundle {bundle}/{variant} does not contain exactly five role views"
            )
        if len({sidecar.split for sidecar in group}) != 1:
            raise TaskBoardKVSegmentError("five role views crossed splits")
    return {
        "records": len(indexes),
        "task_bundles": len(bundle_splits),
        "segment_references": segment_references,
        "unique_segments": len(segment_catalog),
        "unique_segments_by_cache_scope": {
            scope: len(unique_scopes[scope]) for scope in CACHE_SCOPES
        },
        "by_split": dict(sorted(Counter(index.split for index in indexes).items())),
        "by_variant": dict(
            sorted(Counter(index.variant for index in indexes).items())
        ),
        "by_expert": dict(sorted(Counter(index.expert for index in indexes).items())),
        "all_five_role_views_same_split": True,
        "canonical_native_plans_consumed": True,
    }


def content_free_plan_summary(
    plans: Sequence[TaskBoardKVSegmentPlan],
) -> dict[str, Any]:
    """Summarize authenticated indexes without copying canonical plans."""

    scopes = Counter()
    for plan in plans:
        scopes.update(dict(plan.segments_by_cache_scope))
    return {
        "records": len(plans),
        "task_bundles": len({plan.task_bundle_sha256 for plan in plans}),
        "segment_references": sum(plan.segment_count for plan in plans),
        "segments_by_cache_scope": {
            scope: scopes.get(scope, 0) for scope in CACHE_SCOPES
        },
        "by_split": dict(sorted(Counter(plan.split for plan in plans).items())),
        "by_variant": dict(sorted(Counter(plan.variant for plan in plans).items())),
        "cache_reuse_allowed": False,
        "claim_scope": CLAIM_SCOPE,
    }


def _read_snapshot(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            snapshot = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise TaskBoardKVSegmentError(f"{path}: could not read bytes snapshot") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(snapshot) != after.st_size:
        raise TaskBoardKVSegmentError(f"{path}: file changed during bytes snapshot")
    return snapshot


def _parse_json_bytes(snapshot: bytes, source: str) -> Mapping[str, Any]:
    try:
        value = json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskBoardKVSegmentError(f"{source}: invalid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise TaskBoardKVSegmentError(f"{source}: expected a JSON object")
    return value


def _reject_external_refs(value: Any, source: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "$ref" and (not isinstance(item, str) or not item.startswith("#/")):
                raise TaskBoardKVSegmentError(
                    f"{source}: external JSON-Schema resolution is forbidden"
                )
            _reject_external_refs(item, source)
    elif isinstance(value, list):
        for item in value:
            _reject_external_refs(item, source)


def _validate_schema_snapshot(
    snapshot: bytes,
    *,
    source: str,
    expected_version: str,
    required_property: str,
) -> str:
    schema = _parse_json_bytes(snapshot, source)
    _reject_external_refs(schema, source)
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise TaskBoardKVSegmentError(f"{source}: schema root must be closed")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or required_property not in properties:
        raise TaskBoardKVSegmentError(f"{source}: required property is absent")
    schema_version = properties.get("schema_version")
    if not isinstance(schema_version, Mapping) or schema_version.get("const") != expected_version:
        raise TaskBoardKVSegmentError(f"{source}: schema version changed")
    required = schema.get("required")
    if not isinstance(required, list) or required_property not in required:
        raise TaskBoardKVSegmentError(f"{source}: required-field contract changed")
    return _sha256_bytes(snapshot)


def _validate_plan_schema_snapshot(
    snapshot: bytes, *, source: str, expected_sha256: str | None = None
) -> str:
    digest = _validate_schema_snapshot(
        snapshot,
        source=source,
        expected_version=PLAN_SCHEMA_VERSION,
        required_property="segments",
    )
    schema = _parse_json_bytes(snapshot, source)
    if set(schema.get("required", ())) != PLAN_FIELDS:
        raise TaskBoardKVSegmentError("native plan schema root fields changed")
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, "expected_plan_schema_sha256"
    ):
        raise TaskBoardKVSegmentError("native plan schema hash mismatch")
    return digest


def validate_plan_schema(
    path: str | Path, *, expected_sha256: str | None = None
) -> str:
    """Authenticate the closed native plan schema without remote resolution."""

    resolved = Path(path).expanduser().resolve()
    return _validate_plan_schema_snapshot(
        _read_snapshot(resolved),
        source=str(resolved),
        expected_sha256=expected_sha256,
    )


def _load_yaml_snapshot(path: Path) -> tuple[Mapping[str, Any], bytes, str]:
    snapshot = _read_snapshot(path)
    try:
        value = yaml.safe_load(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TaskBoardKVSegmentError(f"{path}: invalid UTF-8 YAML") from exc
    if not isinstance(value, Mapping):
        raise TaskBoardKVSegmentError(f"{path}: expected a YAML mapping")
    return value, snapshot, _sha256_bytes(snapshot)


def _validate_producer_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != PROJECTOR_CONFIG_SCHEMA_VERSION
        or config.get("projector_version") != PROJECTOR_VERSION
    ):
        raise TaskBoardKVSegmentError("producer config version changed")
    output = config.get("output_contract")
    hierarchical = config.get("hierarchical_task_kv")
    partitions = config.get("partitions")
    if not isinstance(output, Mapping) or not isinstance(hierarchical, Mapping) or not isinstance(partitions, Mapping):
        raise TaskBoardKVSegmentError("producer config omitted a required contract")
    expected_output = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "manifest_schema_version": PROJECTOR_MANIFEST_SCHEMA_VERSION,
        "segment_plan_schema_version": PLAN_SCHEMA_VERSION,
        "canonical_gold_written": False,
        "provider_requests": 0,
        "heldout_content_emitted": False,
    }
    if any(output.get(key) != value for key, value in expected_output.items()):
        raise TaskBoardKVSegmentError("producer output contract changed")
    expected_hierarchical = {
        "segment_plan_location": "outer_sidecar.segment_plan",
        "architecture": "hierarchical_task_kv",
        "execution_mode": EXECUTION_MODE,
        "materialization": "metadata_only_no_tensor_or_kv",
        "full_generation_kv_shared_claimed": False,
        "token_level_moe_claimed": False,
        "tensors_emitted": False,
        "kv_payloads_emitted": False,
        "forbidden_current_future_preinsert_allowed": False,
        "shared_then_mask_allowed": False,
    }
    if any(hierarchical.get(key) != value for key, value in expected_hierarchical.items()):
        raise TaskBoardKVSegmentError("producer hierarchical Task-KV contract changed")
    if (
        partitions.get("split_group_key") != "task_bundle_sha256"
        or partitions.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or partitions.get("all_five_role_views_same_split") is not True
        or partitions.get("split_before_augmentation") is not True
    ):
        raise TaskBoardKVSegmentError("producer split grouping contract changed")


def _validate_consumer_config(config: Mapping[str, Any]) -> int:
    if config.get("schema_version") != CONSUMER_CONFIG_SCHEMA_VERSION:
        raise TaskBoardKVSegmentError(
            "consumer config is not the producer-native v2 Task-KV contract"
        )
    source = config.get("source_of_truth")
    partition = config.get("partition_contract")
    architecture = config.get("architecture_contract")
    causal = config.get("causal_visibility")
    runtime = config.get("runtime_compatibility")
    historical = config.get("historical_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (source, partition, architecture, causal, runtime, historical)
    ):
        raise TaskBoardKVSegmentError("consumer config omitted a required contract")
    expected_sources = {
        "segment_plan_location": "outer_sidecar.segment_plan",
        "projector_config": PROJECTOR_CONFIG_SCHEMA_VERSION,
        "sidecar_schema": SIDECAR_SCHEMA_VERSION,
        "segment_plan_schema": PLAN_SCHEMA_VERSION,
        "manifest_schema": PROJECTOR_MANIFEST_SCHEMA_VERSION,
    }
    if source.get("segment_plan_location") != expected_sources["segment_plan_location"]:
        raise TaskBoardKVSegmentError("consumer config source plan location changed")
    for key in (
        "projector_config",
        "sidecar_schema",
        "segment_plan_schema",
        "manifest_schema",
    ):
        descriptor = source.get(key)
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("schema_version") != expected_sources[key]
            or not isinstance(descriptor.get("path"), str)
            or not descriptor["path"]
        ):
            raise TaskBoardKVSegmentError(
                f"consumer config source_of_truth.{key} changed"
            )
    if (
        historical.get("accepted_by_current_fixture") is not False
        or historical.get("derived_taskboard_kv_plan_accepted") is not False
    ):
        raise TaskBoardKVSegmentError("consumer config accepted a legacy derived plan")
    if (
        partition.get("split_before_augmentation") is not True
        or partition.get("split_group_key") != "task_bundle_sha256"
        or partition.get("task_id_cross_binding_key")
        != "training_record.task_board.task_id"
        or partition.get("all_five_role_views_same_split") is not True
    ):
        raise TaskBoardKVSegmentError("consumer config partition contract changed")
    if partition.get("required_role_sequence") != list(ROLES):
        raise TaskBoardKVSegmentError("consumer config role order changed")
    scopes = architecture.get("cache_scopes")
    if not isinstance(scopes, Mapping) or set(scopes.values()) != set(CACHE_SCOPES):
        raise TaskBoardKVSegmentError("consumer config cache scopes changed")
    expected_architecture = {
        "architecture": "hierarchical_task_kv",
        "execution_mode": EXECUTION_MODE,
        "materialization": "metadata_only_no_tensor_or_kv",
        "segment_serialization": "ordered_prefix_chain",
        "independent_segment_concatenation_allowed": False,
        "shared_prefix_membership": "strict_all_five_role_visibility_intersection",
        "promotion_requires": "explicit_committed_and_causally_visible_downstream",
    }
    if any(architecture.get(key) != value for key, value in expected_architecture.items()):
        raise TaskBoardKVSegmentError("consumer architecture contract changed")
    expected_causal = {
        "current_target_segment_emitted": False,
        "shared_then_mask_allowed": False,
        "forbidden_current_future_preinsert_allowed": False,
        "candidate_and_verified_default_scope": "expert_private_delta",
    }
    if any(causal.get(key) != value for key, value in expected_causal.items()):
        raise TaskBoardKVSegmentError("consumer causal visibility contract changed")
    if dict(runtime) != CACHE_COMPATIBILITY:
        raise TaskBoardKVSegmentError("consumer runtime compatibility changed")
    expected_flags = {
        "claim_scope": "research_proxy_only",
        "provider_requests": 0,
        "canonical_gold_written": False,
        "heldout_content_read": False,
    }
    if any(config.get(key) != value for key, value in expected_flags.items()):
        raise TaskBoardKVSegmentError("consumer claim or safety flags changed")
    max_shard_bytes = config.get(
        "max_shard_bytes", DEFAULT_CONSUMER_MAX_SHARD_BYTES
    )
    if (
        isinstance(max_shard_bytes, bool)
        or not isinstance(max_shard_bytes, int)
        or max_shard_bytes < 1
        or max_shard_bytes >= 50_000_000
    ):
        raise TaskBoardKVSegmentError(
            "consumer config max_shard_bytes must be positive and below 50,000,000"
        )
    return max_shard_bytes


def _path_has_declared_suffix(actual: Path, declared: Any) -> bool:
    if not isinstance(declared, str) or not declared:
        return False
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    actual_parts = tuple(part.casefold() for part in actual.parts)
    declared_parts = tuple(part.casefold() for part in relative.parts)
    return (
        len(actual_parts) >= len(declared_parts)
        and actual_parts[-len(declared_parts) :] == declared_parts
    )


def load_authenticated_taskboard_kv_dataset(
    root: str | Path,
    *,
    manifest_path: str | Path | None,
    producer_config_path: str | Path,
    manifest_schema_path: str | Path,
    sidecar_schema_path: str | Path,
    segment_plan_schema_path: str | Path,
    consumer_config_path: str | Path,
    expected_consumer_config_sha256: str | None = None,
) -> tuple[tuple[TaskBoardSidecar, ...], dict[str, Any], dict[str, Any]]:
    """Authenticate all producer/consumer contracts and the fixed dataset."""

    paths = {
        "producer_config": Path(producer_config_path).expanduser().resolve(),
        "manifest_schema": Path(manifest_schema_path).expanduser().resolve(),
        "sidecar_schema": Path(sidecar_schema_path).expanduser().resolve(),
        "segment_plan_schema": Path(segment_plan_schema_path).expanduser().resolve(),
        "consumer_config": Path(consumer_config_path).expanduser().resolve(),
    }
    producer_config, producer_bytes, producer_sha = _load_yaml_snapshot(
        paths["producer_config"]
    )
    _validate_producer_config(producer_config)
    consumer_config, consumer_bytes, consumer_sha = _load_yaml_snapshot(
        paths["consumer_config"]
    )
    config_max_shard_bytes = _validate_consumer_config(consumer_config)
    source_descriptors = consumer_config["source_of_truth"]
    configured_paths = {
        "producer_config": "projector_config",
        "manifest_schema": "manifest_schema",
        "sidecar_schema": "sidecar_schema",
        "segment_plan_schema": "segment_plan_schema",
    }
    for actual_name, descriptor_name in configured_paths.items():
        descriptor = source_descriptors[descriptor_name]
        if not _path_has_declared_suffix(paths[actual_name], descriptor.get("path")):
            raise TaskBoardKVSegmentError(
                f"consumer config does not bind the supplied {actual_name} path"
            )
    expected_consumer_sha = (
        FROZEN_CONSUMER_CONFIG_SHA256
        if expected_consumer_config_sha256 is None
        else _require_sha256(
            expected_consumer_config_sha256, "expected_consumer_config_sha256"
        )
    )
    if consumer_sha != expected_consumer_sha:
        raise TaskBoardKVSegmentError("consumer config hash mismatch")

    manifest_schema_bytes = _read_snapshot(paths["manifest_schema"])
    manifest_schema_sha = _validate_schema_snapshot(
        manifest_schema_bytes,
        source=str(paths["manifest_schema"]),
        expected_version=PROJECTOR_MANIFEST_SCHEMA_VERSION,
        required_property="hierarchical_task_kv",
    )
    sidecar_schema_bytes = _read_snapshot(paths["sidecar_schema"])
    sidecar_schema_sha = _validate_schema_snapshot(
        sidecar_schema_bytes,
        source=str(paths["sidecar_schema"]),
        expected_version=SIDECAR_SCHEMA_VERSION,
        required_property="segment_plan",
    )
    segment_schema_bytes = _read_snapshot(paths["segment_plan_schema"])
    segment_schema_sha = _validate_plan_schema_snapshot(
        segment_schema_bytes, source=str(paths["segment_plan_schema"])
    )
    actual_frozen_hashes = {
        "producer_config": producer_sha,
        "manifest_schema": manifest_schema_sha,
        "sidecar_schema": sidecar_schema_sha,
        "segment_plan_schema": segment_schema_sha,
    }
    if actual_frozen_hashes != FROZEN_PRODUCER_CONTRACT_SHA256:
        raise TaskBoardKVSegmentError(
            "producer contract bytes do not match the frozen v2 release set"
        )

    try:
        sidecars, manifest, source_validation = load_taskboard_sidecar_dataset(
            root,
            manifest_path,
            expected_config_sha256=producer_sha,
            expected_sidecar_schema_sha256=sidecar_schema_sha,
            expected_manifest_schema_sha256=manifest_schema_sha,
            expected_segment_plan_schema_sha256=segment_schema_sha,
        )
    except QuerySpecializationError as exc:
        raise TaskBoardKVSegmentError(f"producer dataset authentication failed: {exc}") from exc

    dataset_summary = validate_native_sidecar_dataset(sidecars)
    for sidecar in sidecars:
        if sidecar.config_sha256 != producer_sha:
            raise TaskBoardKVSegmentError("sidecar producer config binding changed")
        if sidecar.sidecar_schema_sha256 != sidecar_schema_sha:
            raise TaskBoardKVSegmentError("sidecar schema binding changed")
        if sidecar.segment_plan_schema_sha256 != segment_schema_sha:
            raise TaskBoardKVSegmentError("segment-plan schema binding changed")

    # Re-read every local contract after parsing.  A concurrent replacement
    # cannot make authenticated bytes differ from the bytes consumed here.
    expected_contracts = {
        paths["producer_config"]: _sha256_bytes(producer_bytes),
        paths["manifest_schema"]: _sha256_bytes(manifest_schema_bytes),
        paths["sidecar_schema"]: _sha256_bytes(sidecar_schema_bytes),
        paths["segment_plan_schema"]: _sha256_bytes(segment_schema_bytes),
        paths["consumer_config"]: _sha256_bytes(consumer_bytes),
    }
    for path, expected_sha in expected_contracts.items():
        if _sha256_bytes(_read_snapshot(path)) != expected_sha:
            raise TaskBoardKVSegmentError(
                f"{path}: contract changed during authentication"
            )

    validation = {
        "source_manifest_sha256": source_validation["manifest_sha256"],
        "producer_config_sha256": producer_sha,
        "producer_manifest_schema_sha256": manifest_schema_sha,
        "producer_sidecar_schema_sha256": sidecar_schema_sha,
        "producer_segment_plan_schema_sha256": segment_schema_sha,
        "consumer_config_schema_version": CONSUMER_CONFIG_SCHEMA_VERSION,
        "consumer_config_sha256": consumer_sha,
        "consumer_config_max_shard_bytes": config_max_shard_bytes,
        "dataset_summary": dataset_summary,
        "source_authenticated_file_sha256": dict(
            source_validation["authenticated_file_sha256"]
        ),
        "authenticated_contract_paths": {
            name: str(path) for name, path in paths.items()
        },
        "authenticated_contract_sha256": {
            name: expected_contracts[path] for name, path in paths.items()
        },
    }
    return sidecars, manifest, validation


@dataclass(frozen=True)
class ExpertOutputPlacement:
    """Compatibility helper for the native private-then-commit rule."""

    content_sha256: str
    cache_scope: str
    owner_expert_id: str
    downstream_expert_ids: tuple[str, ...]
    explicitly_committed: bool


def place_expert_output(
    *,
    content: str,
    owner_expert_id: str,
    commit_metadata: Mapping[str, Any] | None = None,
) -> ExpertOutputPlacement:
    """Keep new expert output private unless an exact commit promotes it."""

    if owner_expert_id not in ROLES:
        raise TaskBoardKVSegmentError("owner_expert_id is unknown")
    if not isinstance(content, str) or not content:
        raise TaskBoardKVSegmentError("expert output content must not be empty")
    digest = _sha256_bytes(content.encode("utf-8"))
    if commit_metadata is None:
        return ExpertOutputPlacement(
            content_sha256=digest,
            cache_scope="expert_private_delta",
            owner_expert_id=owner_expert_id,
            downstream_expert_ids=(owner_expert_id,),
            explicitly_committed=False,
        )
    fields = {
        "schema_version",
        "commit_id",
        "committed",
        "approved_cache_scope",
        "owner_expert_id",
        "content_sha256",
        "downstream_expert_ids",
    }
    commit = _require_exact_mapping(commit_metadata, fields, "commit_metadata")
    downstream = _require_unique_text_list(
        commit.get("downstream_expert_ids"), "commit_metadata.downstream_expert_ids"
    )
    if (
        commit.get("schema_version") != "anchor.taskboard-kv-explicit-commit.v1"
        or commit.get("committed") is not True
        or commit.get("approved_cache_scope")
        != "downstream_task_shared_immutable"
        or commit.get("owner_expert_id") != owner_expert_id
        or commit.get("content_sha256") != digest
        or any(role not in ROLES for role in downstream)
    ):
        raise TaskBoardKVSegmentError("commit metadata does not exactly bind output")
    return ExpertOutputPlacement(
        content_sha256=digest,
        cache_scope="downstream_task_shared_immutable",
        owner_expert_id=owner_expert_id,
        downstream_expert_ids=tuple(downstream),
        explicitly_committed=True,
    )
