"""Authenticated natural-language scaffolds for prefix-branch training.

The producer consumes a pinned TaskBoard projector artifact in two passes.
The first pass proves source/bundle/split/role invariants without retaining
source bodies.  The second pass renders only explicit, causally allowed
segment references plus synthetic, auditable control metadata.  Current,
future, and forbidden bodies are never serialized into an output scaffold.

This module deliberately describes a two-request protocol.  A planner result
is validated and committed, a frozen base with adapters disabled re-encodes
the short committed scaffold, and only a subsequent expert request may scan
an aLoRA invocation in its input.  It makes no mid-request switching, physical
KV sharing, hidden-chain-of-thought, or training-quality claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import yaml

from anchor_mvp.swebench import long_context_preflight as _lc
from anchor_mvp.swebench import taskboard_projector as _projector


CONFIG_SCHEMA = "anchor.natural-language-scaffold-config.v1"
PRODUCER_VERSION = "anchor.natural-language-scaffold-producer.v1"
RECORD_SCHEMA = "anchor.natural-language-scaffold.v1"
MANIFEST_SCHEMA = "anchor.natural-language-scaffold-manifest.v1"
ROUTING_SCHEMA = "anchor.natural-language-scaffold-routing.v1"
SMOKE_SCHEMA = "anchor.natural-language-scaffold-smoke-contract.v1"

FIXED_FILES = (
    ("train/json_only.jsonl", "train", "noisy", "json_only"),
    (
        "train/concise_rationale_plus_json.jsonl",
        "train",
        "noisy",
        "concise_rationale_plus_json",
    ),
    ("calibration/json_only.jsonl", "calibration", "clean", "json_only"),
    (
        "calibration/concise_rationale_plus_json.jsonl",
        "calibration",
        "clean",
        "concise_rationale_plus_json",
    ),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_RECORD_ID_RE = re.compile(r"^natural-language-scaffold-v1:[0-9a-f]{64}$")
_PAIR_ID_RE = re.compile(r"^natural-language-scaffold-pair-v1:[0-9a-f]{64}$")
_DENIED_KEYS = {
    "answer",
    "chain_of_thought",
    "content",
    "heldout",
    "messages",
    "preview",
    "prompt",
    "task_board",
    "token_ids",
    "token_index",
    "training_record",
}


class NaturalLanguageScaffoldError(RuntimeError):
    """A stable, body-free producer error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise NaturalLanguageScaffoldError(code)


def _lc_error(exc: _lc.LongContextPreflightError) -> NaturalLanguageScaffoldError:
    suffix = exc.code.removeprefix("long_context_")
    return NaturalLanguageScaffoldError("natural_language_scaffold_" + suffix)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_id(value: object) -> bool:
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def _contains_denied_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _DENIED_KEYS or _contains_denied_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_denied_key(item) for item in value)
    return False


def _load_yaml(snapshot: _lc._BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise NaturalLanguageScaffoldError(code) from exc
    return _mapping(value, code)


@dataclass(frozen=True)
class NaturalLanguageScaffoldConfig:
    """Hash-bound policy and schemas for a scaffold build."""

    path: Path
    sha256: str
    record_schema_sha256: str
    manifest_schema_sha256: str
    smoke_schema_sha256: str
    smoke_config_sha256: str
    implementation_sha256: str
    max_input_records: int
    max_input_file_bytes: int
    max_output_file_bytes: int
    raw: Mapping[str, Any]
    architecture_contract_sha256: str
    adapter_control_policy_sha256: str
    serialization_policy_sha256: str

    @classmethod
    def load(
        cls, value: str | Path
    ) -> tuple["NaturalLanguageScaffoldConfig", dict[Path, _lc._BytesSnapshot]]:
        raw_path = Path(value).expanduser()
        if raw_path.is_symlink():
            _fail("natural_language_scaffold_config_invalid")
        path = raw_path.resolve()
        if path != raw_path.absolute():
            _fail("natural_language_scaffold_config_invalid")
        try:
            snapshot = _lc._read_snapshot(
                path,
                "natural_language_scaffold_config_invalid",
                max_bytes=1_000_000,
            )
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        raw = _load_yaml(snapshot, "natural_language_scaffold_config_invalid")
        expected_top = {
            "schema_version",
            "producer_version",
            "input_contract",
            "output_contract",
            "architecture_contract",
            "visibility_contract",
            "scaffold_contract",
            "route_boundary_contract",
            "cache_contract",
            "adapter_control_contract",
            "alora_capability_contract",
            "serialization_contract",
            "fixture_contract",
            "limits",
            "safety",
        }
        _exact_keys(raw, expected_top, "natural_language_scaffold_config_invalid")
        output = _mapping(
            raw.get("output_contract"), "natural_language_scaffold_config_invalid"
        )
        architecture = _mapping(
            raw.get("architecture_contract"),
            "natural_language_scaffold_config_invalid",
        )
        visibility = _mapping(
            raw.get("visibility_contract"),
            "natural_language_scaffold_config_invalid",
        )
        route = _mapping(
            raw.get("route_boundary_contract"),
            "natural_language_scaffold_config_invalid",
        )
        cache = _mapping(
            raw.get("cache_contract"), "natural_language_scaffold_config_invalid"
        )
        adapter = _mapping(
            raw.get("adapter_control_contract"),
            "natural_language_scaffold_config_invalid",
        )
        alora = _mapping(
            raw.get("alora_capability_contract"),
            "natural_language_scaffold_config_invalid",
        )
        fixture = _mapping(
            raw.get("fixture_contract"), "natural_language_scaffold_config_invalid"
        )
        limits = _mapping(raw.get("limits"), "natural_language_scaffold_config_invalid")
        safety = _mapping(raw.get("safety"), "natural_language_scaffold_config_invalid")
        fixed = [item[0] for item in FIXED_FILES]
        if (
            raw.get("schema_version") != CONFIG_SCHEMA
            or raw.get("producer_version") != PRODUCER_VERSION
            or output.get("record_schema_version") != RECORD_SCHEMA
            or output.get("manifest_schema_version") != MANIFEST_SCHEMA
            or output.get("fixed_files") != fixed
            or output.get("mandatory_manifest_sha256_sidecar") is not True
            or architecture.get("name")
            != "frozen_prefix_q_reader__prefix_branch_producer_consumer"
            or architecture.get("adapter_state_on_prefix") != "off"
            or architecture.get("adapter_state_after_boundary") != "expert_only"
            or architecture.get("hidden_chain_of_thought_exposed") is not False
            or architecture.get("cross_attention_q_reader_implemented") is not False
            or visibility.get("split_group_key") != "task_bundle_sha256"
            or visibility.get("split_before_augmentation") is not True
            or visibility.get("all_five_role_views_same_split") is not True
            or visibility.get("current_target_body_serialized") is not False
            or visibility.get("future_block_body_serialized") is not False
            or visibility.get("forbidden_block_body_serialized") is not False
            or visibility.get("whole_taskboard_stringification_allowed") is not False
            or route.get("semantics") != "explicit_two_request_commit_boundary"
            or route.get("committed_scaffold_reencode_required") is not True
            or architecture.get("private_tail_kv_required") is not True
            or cache.get("full_generation_kv_shared_claimed") is not False
            or cache.get("exact_reuse_scope") != "identical_ordered_prefix_lineage_only"
            or adapter.get("labels") != ["q_only", "q_plus_o", "wide_lora"]
            or adapter.get("producer_claims_training_outcome") is not False
            or alora.get("activation_semantics") != "next_request_input_activation_only"
            or alora.get("same_request_activation_allowed") is not False
            or alora.get("mid_request_generated_activation_allowed") is not False
            or alora.get("mid_request_generated_trigger_switch_claimed") is not False
            or fixture.get("source_task_bundles") != 2
            or fixture.get("roles_per_bundle") != 5
            or _mapping(
                raw.get("scaffold_contract"),
                "natural_language_scaffold_config_invalid",
            ).get("variants")
            != ["json_only", "concise_rationale_plus_json"]
            or fixture.get("expected_records") != 20
            or safety.get("provider_requests") != 0
            or safety.get("model_loaded") is not False
            or safety.get("canonical_gold_written") is not False
        ):
            _fail("natural_language_scaffold_config_invalid")
        for key in (
            "max_input_records",
            "max_input_file_bytes",
            "max_output_file_bytes",
        ):
            number = limits.get(key)
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                _fail("natural_language_scaffold_config_invalid")

        record_schema_path = (
            path.parent / "swebench_natural_language_scaffold_sidecar.schema.json"
        )
        manifest_schema_path = (
            path.parent / "swebench_natural_language_scaffold_manifest.schema.json"
        )
        smoke_schema_path = (
            path.parent
            / "swebench_natural_language_scaffold_smoke_contract.schema.json"
        )
        smoke_config_path = (
            path.parent / "swebench_natural_language_scaffold_smoke_v1.yaml"
        )
        implementation_path = (
            path.parents[2]
            / "src"
            / "anchor_mvp"
            / "swebench"
            / "natural_language_scaffold.py"
        )
        schema_paths = (record_schema_path, manifest_schema_path, smoke_schema_path)
        schema_snapshots: list[_lc._BytesSnapshot] = []
        try:
            for schema_path in schema_paths:
                schema_snapshots.append(
                    _lc._read_snapshot(
                        schema_path,
                        "natural_language_scaffold_schema_invalid",
                        max_bytes=1_000_000,
                    )
                )
            smoke_snapshot = _lc._read_snapshot(
                smoke_config_path,
                "natural_language_scaffold_smoke_contract_invalid",
                max_bytes=1_000_000,
            )
            implementation_snapshot = _lc._read_snapshot(
                implementation_path,
                "natural_language_scaffold_implementation_invalid",
                max_bytes=2_000_000,
            )
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        schema_constants = (RECORD_SCHEMA, MANIFEST_SCHEMA, SMOKE_SCHEMA)
        for item, constant in zip(schema_snapshots, schema_constants, strict=True):
            try:
                parsed = _lc._json(item, "natural_language_scaffold_schema_invalid")
            except _lc.LongContextPreflightError as exc:
                raise _lc_error(exc) from exc
            if (
                parsed.get("properties", {}).get("schema_version", {}).get("const")
                != constant
            ):
                _fail("natural_language_scaffold_schema_invalid")
        smoke = _load_yaml(
            smoke_snapshot, "natural_language_scaffold_smoke_contract_invalid"
        )
        smoke_model = _mapping(
            smoke.get("model_artifact"),
            "natural_language_scaffold_smoke_contract_invalid",
        )
        smoke_runtime = _mapping(
            smoke.get("runtime_capability"),
            "natural_language_scaffold_smoke_contract_invalid",
        )
        smoke_current = _mapping(
            smoke.get("current_execution"),
            "natural_language_scaffold_smoke_contract_invalid",
        )
        if (
            smoke.get("schema_version") != SMOKE_SCHEMA
            or smoke.get("contract_scope") != "behavior_smoke_only"
            or smoke.get("execution_mode") != "contract_only_unexecuted"
            or smoke_model.get("basename") != "qwen2.5-1.5b-instruct-q4_k_m.gguf"
            or smoke_model.get("sha256_status") != "runtime_required"
            or smoke_model.get("bytes_status") != "runtime_required"
            or smoke_model.get("trainable_weights") is not False
            or smoke_runtime.get("activation_semantics")
            != "next_request_input_activation_only"
            or smoke_runtime.get("mid_request_generated_trigger_switch_claimed")
            is not False
            or smoke_current.get("model_loaded") is not False
            or smoke_current.get("provider_requests") != 0
            or smoke_current.get("network_requests") != 0
        ):
            _fail("natural_language_scaffold_smoke_contract_invalid")
        loaded = cls(
            path=path,
            sha256=snapshot.sha256,
            record_schema_sha256=schema_snapshots[0].sha256,
            manifest_schema_sha256=schema_snapshots[1].sha256,
            smoke_schema_sha256=schema_snapshots[2].sha256,
            smoke_config_sha256=smoke_snapshot.sha256,
            implementation_sha256=implementation_snapshot.sha256,
            max_input_records=int(limits["max_input_records"]),
            max_input_file_bytes=int(limits["max_input_file_bytes"]),
            max_output_file_bytes=int(limits["max_output_file_bytes"]),
            raw=raw,
            architecture_contract_sha256=_sha256_value(architecture),
            adapter_control_policy_sha256=_sha256_value(adapter),
            serialization_policy_sha256=_sha256_value(raw["serialization_contract"]),
        )
        inventory = {
            path: snapshot,
            smoke_config_path: smoke_snapshot,
            implementation_path: implementation_snapshot,
        }
        inventory.update(dict(zip(schema_paths, schema_snapshots, strict=True)))
        return loaded, inventory


def _stable_goal(language: str) -> str:
    del language
    return "complete_bound_stage_using_only_declared_evidence_references"


def _stable_rationale(language: str, expert: str) -> str:
    if language == "zh-CN":
        return (
            f"决策依据摘要：仅核验已认证的允许证据引用与验收条件，再路由至 {expert}。"
        )
    return (
        "Decision-basis summary: verify only authenticated allowed evidence "
        f"references and acceptance criteria before routing to {expert}."
    )


def _segment_ref(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": item["segment_id"],
        "source_block_id": item["source_block_id"],
        "content_sha256": item["content_sha256"],
        "causal_order": item["causal_order"],
        "cache_scope": item["cache_scope"],
    }


def _render_source(
    cfg: NaturalLanguageScaffoldConfig,
    source: _lc._SanitizedSourceRow,
    line: bytes,
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256_bytes(line) != source.source_line_sha256:
        _fail("natural_language_scaffold_source_line_changed")
    inner = _mapping(
        value.get("training_record"), "natural_language_scaffold_source_invalid"
    )
    board = _mapping(
        inner.get("task_board"), "natural_language_scaffold_source_invalid"
    )
    targets = _mapping(
        inner.get("attention_targets"),
        "natural_language_scaffold_source_invalid",
    )
    blocks = board.get("blocks")
    relevant = targets.get("relevant_block_ids")
    distractors = targets.get("distractor_block_ids")
    forbidden = targets.get("forbidden_block_ids")
    if (
        not isinstance(blocks, list)
        or not isinstance(relevant, list)
        or not isinstance(distractors, list)
        or not isinstance(forbidden, list)
    ):
        _fail("natural_language_scaffold_source_invalid")
    by_id = {
        str(item["id"]): item
        for item in blocks
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    plan = _mapping(
        value.get("segment_plan"), "natural_language_scaffold_source_invalid"
    )
    try:
        _projector._validate_segment_plan(
            plan,
            wrapper=value,
            inner=inner,
            by_id=by_id,
            relevant=relevant,
            distractors=distractors,
            forbidden=forbidden,
        )
    except _projector.TaskBoardProjectorError as exc:
        raise NaturalLanguageScaffoldError(
            "natural_language_scaffold_segment_plan_invalid"
        ) from exc
    if _sha256_value(plan) != source.segment_plan_sha256:
        _fail("natural_language_scaffold_segment_plan_changed")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        _fail("natural_language_scaffold_segment_plan_invalid")
    forbidden_set = set(str(item) for item in forbidden)
    allowed_refs: list[dict[str, Any]] = []
    evidence_refs: list[dict[str, Any]] = []
    shared_prefix_lineage: str | None = None
    for index, segment in enumerate(segments):
        item = _mapping(segment, "natural_language_scaffold_segment_plan_invalid")
        source_id = item.get("source_block_id")
        if source_id != source.source_block_ids[index] or source_id in forbidden_set:
            _fail("natural_language_scaffold_forbidden_selection")
        block = _mapping(
            by_id.get(str(source_id)),
            "natural_language_scaffold_segment_plan_invalid",
        )
        body = block.get("content")
        if (
            not isinstance(body, str)
            or not body
            or _sha256_bytes(body.encode("utf-8")) != item.get("content_sha256")
        ):
            _fail("natural_language_scaffold_segment_plan_invalid")
        ref = _segment_ref(item)
        allowed_refs.append(ref)
        if item.get("cache_scope") != "expert_private_delta":
            evidence_refs.append(ref)
            shared_prefix_lineage = str(item.get("prefix_lineage_sha256"))
    if not shared_prefix_lineage:
        _fail("natural_language_scaffold_shared_prefix_invalid")
    target = _mapping(inner.get("target"), "natural_language_scaffold_source_invalid")
    target_body = target.get("answer")
    if not isinstance(target_body, str) or not target_body:
        _fail("natural_language_scaffold_source_invalid")
    target_sha256 = _sha256_bytes(target_body.encode("utf-8"))
    allowed_evidence_sha256 = _sha256_value(allowed_refs)
    forbidden_evidence_sha256 = _sha256_value(
        {
            "source_line_sha256": source.source_line_sha256,
            "ordered_forbidden_block_ids": [str(item) for item in forbidden],
        }
    )
    target_binding_sha256 = _sha256_value(
        {
            "source_gold_sha256": source.source_gold_sha256,
            "target_sha256": target_sha256,
            "stage": source.stage,
            "expert": source.expert,
        }
    )
    ordered_segment_ids_sha256 = _sha256_value(
        [item["segment_id"] for item in allowed_refs]
    )
    pair_identity = {
        "task_bundle_sha256": source.task_bundle_sha256,
        "task_id_sha256": source.task_id_sha256,
        "source_gold_sha256": source.source_gold_sha256,
        "split": source.split,
        "source_variant": source.variant,
        "stage": source.stage,
        "expert": source.expert,
        "target_binding_sha256": target_binding_sha256,
        "allowed_evidence_sha256": allowed_evidence_sha256,
        "forbidden_evidence_sha256": forbidden_evidence_sha256,
        "segment_plan_sha256": source.segment_plan_sha256,
    }
    pair_id = "natural-language-scaffold-pair-v1:" + _sha256_value(pair_identity)
    trigger_text = f"<|anchor_expert:{source.expert}|>"
    trigger_sha256 = _sha256_bytes(trigger_text.encode("utf-8"))
    expert_trigger = {
        "kind": "next_request_expert_invocation_candidate",
        "expert": source.expert,
        "trigger_text": trigger_text,
        "trigger_text_sha256": trigger_sha256,
        "instruction": "execute_bound_stage_using_committed_scaffold_and_allowed_refs_only",
        "tokenizer_binding_status": "unbound",
    }
    route_boundary = {
        "semantics": "explicit_two_request_commit_boundary",
        "prefix_lineage_sha256": source.terminal_prefix_lineage_sha256,
        "planner_request_phase": "rationale_route_and_sentinel_candidate",
        "validation_required": True,
        "commit_required": True,
        "commit_promotes_text_only": True,
        "planner_private_tail_kv_transfer_allowed": False,
        "committed_scaffold_reencode_required": True,
        "committed_scaffold_reencode_producer": "frozen_base",
        "committed_scaffold_reencode_adapter_state": "off",
        "expert_request_phase": "next_request",
        "expert_request_requires_committed_scaffold_as_input": True,
        "token_boundary_status": "tokenizer_binding_required",
    }
    cache_metadata = {
        "prefix_lineage_sha256": source.terminal_prefix_lineage_sha256,
        "shared_prefix_scope": "task_shared_prefix",
        "private_tail_scope": "expert_private_delta",
        "adapter_state_on_prefix": "off",
        "adapter_state_after_boundary": "expert_only",
        "private_tail_kv_required": True,
        "full_generation_kv_shared_claimed": False,
        "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
        "cache_identity_status": "identity_unbound",
        "exact_cache_reuse_enabled": False,
        "reuse_savings_tokens": 0,
        "planner_private_tail_kv_reused_by_expert": False,
        "physical_kv_tensor_emitted": False,
        "committed_scaffold_reencode_executed": False,
        "downstream_immutable_segment_emitted": False,
    }
    alora_invocation = {
        "optional": True,
        "trigger_text": trigger_text,
        "trigger_text_sha256": trigger_sha256,
        "capability_binding_status": "unbound",
        "activation_semantics": "next_request_input_activation_only",
        "invocation_scan_scope": "new_request_input_tokens_only",
        "same_request_activation_allowed": False,
        "mid_request_generated_activation_allowed": False,
        "mid_request_generated_trigger_switch_claimed": False,
        "explicit_commit_required": True,
        "tokenizer_binding_status": "unbound",
        "adapter_available": False,
        "adapter_loaded": False,
        "activation_executed": False,
        "cross_attention_q_reader_claimed": False,
        "physical_shared_kv_claimed": False,
    }
    ordered_segment_ids = [item["segment_id"] for item in allowed_refs]
    tool_calls = [
        {
            "call_id": "scaffold-call-v1:" + _sha256_value([pair_id, "inspect"]),
            "tool_name": "fixture.inspect_segment_refs",
            "arguments": {
                "segment_ids": ordered_segment_ids,
                "checks": [
                    "ordered_lineage",
                    "visibility_allowlist",
                    "target_exclusion",
                    "forbidden_exclusion",
                ],
            },
        },
        {
            "call_id": "scaffold-call-v1:" + _sha256_value([pair_id, "validate"]),
            "tool_name": "fixture.validate_constraints",
            "arguments": {
                "segment_ids": ordered_segment_ids,
                "checks": [
                    "visibility_allowlist",
                    "target_exclusion",
                    "forbidden_exclusion",
                    "pair_binding",
                ],
            },
        },
    ]
    tool_results = [
        {
            "call_id": tool_calls[0]["call_id"],
            "status": "synthetic_ok",
            "result_code": "segment_refs_verified",
        },
        {
            "call_id": tool_calls[1]["call_id"],
            "status": "synthetic_ok",
            "result_code": "constraints_verified",
        },
    ]
    routing_json = {
        "role": source.stage,
        "expert": source.expert,
        "goal": _stable_goal(source.language),
        "constraints": [
            "use_only_declared_allowed_segment_refs",
            "exclude_current_target_future_and_forbidden_bodies",
            "do_not_stringify_taskboard",
            "require_explicit_validation_and_commit_before_expert_request",
            "treat_expert_tail_as_private_kv",
        ],
        "allowed_segment_refs": allowed_refs,
        "evidence_segment_refs": evidence_refs,
        "tool_plan": [item["call_id"] for item in tool_calls],
        "acceptance_criteria": [
            "route_schema_valid",
            "source_target_and_evidence_bindings_match",
            "tool_trace_is_auditable",
            "scaffold_committed_before_next_request",
            "expert_private_tail_not_claimed_shared",
        ],
    }
    routing_json_sha256 = _sha256_value(routing_json)
    payload = {
        "routing_json": routing_json,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "expert_trigger": expert_trigger,
    }
    shared = {
        "schema_version": RECORD_SCHEMA,
        "pair_id": pair_id,
        "task_bundle_sha256": source.task_bundle_sha256,
        "task_id_sha256": source.task_id_sha256,
        "source_gold_sha256": source.source_gold_sha256,
        "split": source.split,
        "source_variant": source.variant,
        "stage": source.stage,
        "expert": source.expert,
        "language": source.language,
        "source_partition_sha256": source.source_partition_sha256,
        "source_line_sha256": source.source_line_sha256,
        "target_sha256": target_sha256,
        "target_binding_sha256": target_binding_sha256,
        "allowed_evidence_sha256": allowed_evidence_sha256,
        "forbidden_evidence_sha256": forbidden_evidence_sha256,
        "segment_plan_sha256": source.segment_plan_sha256,
        "ordered_segment_ids_sha256": ordered_segment_ids_sha256,
        "terminal_prefix_lineage_sha256": source.terminal_prefix_lineage_sha256,
        "architecture_contract_sha256": cfg.architecture_contract_sha256,
        "adapter_control_policy_sha256": cfg.adapter_control_policy_sha256,
        "serialization_policy_sha256": cfg.serialization_policy_sha256,
        "route_boundary": route_boundary,
        "cache_metadata": cache_metadata,
        "routing_json": routing_json,
        "routing_json_sha256": routing_json_sha256,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "expert_trigger": expert_trigger,
        "alora_invocation": alora_invocation,
        "adapter_control_labels": ["q_only", "q_plus_o", "wide_lora"],
        "training_outcome_claimed": False,
        "canonical_json_payload_sha256": _sha256_value(payload),
        "evaluation_status": "not_evaluated",
        "quality_validated": False,
        "execution_authorized": False,
        "provider_requests": 0,
    }
    records: dict[str, dict[str, Any]] = {}
    payload_text = _canonical_bytes(payload).decode("utf-8")
    for scaffold_variant in ("json_only", "concise_rationale_plus_json"):
        record = dict(shared)
        record["scaffold_variant"] = scaffold_variant
        if scaffold_variant == "concise_rationale_plus_json":
            rationale = _stable_rationale(source.language, source.expert)
            if len(rationale.encode("utf-8")) > 512:
                _fail("natural_language_scaffold_rationale_invalid")
            record["concise_rationale_summary"] = rationale
            scaffold_text = rationale + "\n" + payload_text
        else:
            scaffold_text = payload_text
        record["scaffold_text"] = scaffold_text
        record["scaffold_text_sha256"] = _sha256_bytes(scaffold_text.encode("utf-8"))
        record_identity = {
            "pair_id": pair_id,
            "scaffold_variant": scaffold_variant,
            "scaffold_text_sha256": record["scaffold_text_sha256"],
        }
        record["record_id"] = "natural-language-scaffold-v1:" + _sha256_value(
            record_identity
        )
        ordered = {key: record[key] for key in sorted(record)}
        _validate_record(ordered)
        records[scaffold_variant] = ordered
    return records["json_only"], records["concise_rationale_plus_json"]


def _validate_record(value: Mapping[str, Any]) -> None:
    common = {
        "schema_version",
        "record_id",
        "pair_id",
        "task_bundle_sha256",
        "task_id_sha256",
        "source_gold_sha256",
        "split",
        "source_variant",
        "stage",
        "expert",
        "language",
        "scaffold_variant",
        "source_partition_sha256",
        "source_line_sha256",
        "target_sha256",
        "target_binding_sha256",
        "allowed_evidence_sha256",
        "forbidden_evidence_sha256",
        "segment_plan_sha256",
        "ordered_segment_ids_sha256",
        "terminal_prefix_lineage_sha256",
        "architecture_contract_sha256",
        "adapter_control_policy_sha256",
        "serialization_policy_sha256",
        "route_boundary",
        "cache_metadata",
        "routing_json",
        "routing_json_sha256",
        "tool_calls",
        "tool_results",
        "expert_trigger",
        "alora_invocation",
        "adapter_control_labels",
        "training_outcome_claimed",
        "canonical_json_payload_sha256",
        "scaffold_text",
        "scaffold_text_sha256",
        "evaluation_status",
        "quality_validated",
        "execution_authorized",
        "provider_requests",
    }
    variant = value.get("scaffold_variant")
    expected = common | (
        {"concise_rationale_summary"}
        if variant == "concise_rationale_plus_json"
        else set()
    )
    _exact_keys(value, expected, "natural_language_scaffold_record_invalid")
    hashes = {
        "task_bundle_sha256",
        "task_id_sha256",
        "source_gold_sha256",
        "source_partition_sha256",
        "source_line_sha256",
        "target_sha256",
        "target_binding_sha256",
        "allowed_evidence_sha256",
        "forbidden_evidence_sha256",
        "segment_plan_sha256",
        "ordered_segment_ids_sha256",
        "terminal_prefix_lineage_sha256",
        "architecture_contract_sha256",
        "adapter_control_policy_sha256",
        "serialization_policy_sha256",
        "routing_json_sha256",
        "canonical_json_payload_sha256",
        "scaffold_text_sha256",
    }
    route = _mapping(
        value.get("route_boundary"), "natural_language_scaffold_record_invalid"
    )
    cache = _mapping(
        value.get("cache_metadata"), "natural_language_scaffold_record_invalid"
    )
    alora = _mapping(
        value.get("alora_invocation"), "natural_language_scaffold_record_invalid"
    )
    routing = _mapping(
        value.get("routing_json"), "natural_language_scaffold_record_invalid"
    )
    trigger = _mapping(
        value.get("expert_trigger"), "natural_language_scaffold_record_invalid"
    )
    if (
        value.get("schema_version") != RECORD_SCHEMA
        or not isinstance(value.get("record_id"), str)
        or _RECORD_ID_RE.fullmatch(str(value["record_id"])) is None
        or not isinstance(value.get("pair_id"), str)
        or _PAIR_ID_RE.fullmatch(str(value["pair_id"])) is None
        or any(not _is_sha256(value.get(key)) for key in hashes)
        or value.get("split") not in {"train", "calibration"}
        or value.get("source_variant") not in {"clean", "noisy"}
        or (value.get("split"), value.get("source_variant"))
        not in {("train", "noisy"), ("calibration", "clean")}
        or value.get("stage") not in _projector.STAGE_EXPERTS
        or _projector.STAGE_EXPERTS[str(value.get("stage"))] != value.get("expert")
        or value.get("language") not in {"en", "zh-CN"}
        or variant not in {"json_only", "concise_rationale_plus_json"}
        or not isinstance(value.get("scaffold_text"), str)
        or not value["scaffold_text"]
        or _sha256_bytes(str(value["scaffold_text"]).encode("utf-8"))
        != value.get("scaffold_text_sha256")
        or _sha256_value(routing) != value.get("routing_json_sha256")
        or route.get("semantics") != "explicit_two_request_commit_boundary"
        or route.get("commit_required") is not True
        or route.get("commit_promotes_text_only") is not True
        or route.get("planner_private_tail_kv_transfer_allowed") is not False
        or route.get("committed_scaffold_reencode_required") is not True
        or route.get("committed_scaffold_reencode_producer") != "frozen_base"
        or route.get("committed_scaffold_reencode_adapter_state") != "off"
        or route.get("expert_request_phase") != "next_request"
        or route.get("expert_request_requires_committed_scaffold_as_input") is not True
        or cache.get("private_tail_kv_required") is not True
        or cache.get("full_generation_kv_shared_claimed") is not False
        or cache.get("exact_reuse_scope") != "identical_ordered_prefix_lineage_only"
        or cache.get("planner_private_tail_kv_reused_by_expert") is not False
        or cache.get("committed_scaffold_reencode_executed") is not False
        or alora.get("activation_semantics") != "next_request_input_activation_only"
        or alora.get("same_request_activation_allowed") is not False
        or alora.get("mid_request_generated_activation_allowed") is not False
        or alora.get("mid_request_generated_trigger_switch_claimed") is not False
        or alora.get("explicit_commit_required") is not True
        or alora.get("adapter_available") is not False
        or alora.get("activation_executed") is not False
        or value.get("adapter_control_labels") != ["q_only", "q_plus_o", "wide_lora"]
        or value.get("training_outcome_claimed") is not False
        or routing.get("role") != value.get("stage")
        or routing.get("expert") != value.get("expert")
        or trigger.get("expert") != value.get("expert")
        or value.get("evaluation_status") != "not_evaluated"
        or value.get("quality_validated") is not False
        or value.get("execution_authorized") is not False
        or value.get("provider_requests") != 0
        or _contains_denied_key(value)
    ):
        _fail("natural_language_scaffold_record_invalid")
    rationale = value.get("concise_rationale_summary")
    if variant == "json_only":
        if rationale is not None:
            _fail("natural_language_scaffold_record_invalid")
    elif (
        not isinstance(rationale, str)
        or not rationale
        or len(rationale.encode("utf-8")) > 512
        or not str(value["scaffold_text"]).startswith(rationale + "\n")
    ):
        _fail("natural_language_scaffold_record_invalid")


def _pair_invariant(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "record_id",
        "scaffold_variant",
        "concise_rationale_summary",
        "scaffold_text",
        "scaffold_text_sha256",
    }
    return {key: record[key] for key in sorted(record) if key not in excluded}


def _verify_output_groups(
    records: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> None:
    flat = [row for rows in records.values() for row in rows]
    if len(flat) != 20 or len({row["record_id"] for row in flat}) != 20:
        _fail("natural_language_scaffold_fixture_count_invalid")
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in flat:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    if len(pairs) != 10:
        _fail("natural_language_scaffold_pair_invalid")
    for rows in pairs.values():
        if (
            len(rows) != 2
            or {row["scaffold_variant"] for row in rows}
            != {"json_only", "concise_rationale_plus_json"}
            or _pair_invariant(rows[0]) != _pair_invariant(rows[1])
        ):
            _fail("natural_language_scaffold_pair_invalid")
    groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
    bundle_split: dict[str, str] = {}
    for row in flat:
        prior = bundle_split.setdefault(row["task_bundle_sha256"], row["split"])
        if prior != row["split"]:
            _fail("natural_language_scaffold_bundle_split_invalid")
        groups.setdefault(
            (row["task_bundle_sha256"], row["scaffold_variant"]), set()
        ).add((row["stage"], row["expert"]))
    expected_roles = {
        (stage, _projector.STAGE_EXPERTS[stage]) for stage in _projector.STAGES
    }
    if len(bundle_split) != 2 or any(
        group != expected_roles for group in groups.values()
    ):
        _fail("natural_language_scaffold_role_group_invalid")


def _records_second_pass(
    cfg: NaturalLanguageScaffoldConfig,
    snapshots: Mapping[tuple[str, str], _lc._BytesSnapshot],
    source_rows: Mapping[tuple[str, str], list[_lc._SanitizedSourceRow]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = {
        (split, scaffold): [] for _path, split, _source, scaffold in FIXED_FILES
    }
    for split, source_variant in (("train", "noisy"), ("calibration", "clean")):
        expected = source_rows[(split, source_variant)]
        try:
            parsed = list(
                _lc._iter_jsonl(
                    snapshots[(split, source_variant)],
                    "natural_language_scaffold_source_invalid",
                )
            )
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        if len(parsed) != len(expected):
            _fail("natural_language_scaffold_source_line_changed")
        for source, (line_number, line, value) in zip(expected, parsed, strict=True):
            if source.line_number != line_number:
                _fail("natural_language_scaffold_source_line_changed")
            json_only, rationale = _render_source(cfg, source, line, value)
            output[(split, "json_only")].append(json_only)
            output[(split, "concise_rationale_plus_json")].append(rationale)
    _verify_output_groups(output)
    return output


def _manifest_counts(
    records: Mapping[tuple[str, str], list[dict[str, Any]]],
    source_rows: Mapping[tuple[str, str], list[_lc._SanitizedSourceRow]],
) -> dict[str, Any]:
    flat = [row for rows in records.values() for row in rows]
    selected_source = (
        source_rows[("train", "noisy")] + source_rows[("calibration", "clean")]
    )
    allowed_refs = [
        ref for row in flat for ref in row["routing_json"]["allowed_segment_refs"]
    ]
    return {
        "total": len(flat),
        "pairs": len({row["pair_id"] for row in flat}),
        "unique_task_bundles": len({row["task_bundle_sha256"] for row in flat}),
        "task_ids_sha256": _sha256_bytes(
            "\n".join(sorted({row.task_id for row in selected_source})).encode("utf-8")
        ),
        "allowed_segment_references": len(allowed_refs),
        "unique_allowed_segments": len({ref["segment_id"] for ref in allowed_refs}),
        "by_split": dict(sorted(Counter(row["split"] for row in flat).items())),
        "by_scaffold_variant": dict(
            sorted(Counter(row["scaffold_variant"] for row in flat).items())
        ),
        "by_source_variant": dict(
            sorted(Counter(row["source_variant"] for row in flat).items())
        ),
        "by_stage": dict(sorted(Counter(row["stage"] for row in flat).items())),
        "by_expert": dict(sorted(Counter(row["expert"] for row in flat).items())),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> bytes:
    data = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _verify_bytes_unchanged(
    snapshots: Mapping[Path, _lc._BytesSnapshot], code: str
) -> None:
    for path, expected in snapshots.items():
        try:
            current = _lc._read_snapshot(path, code, max_bytes=max(expected.size, 1))
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        if current.sha256 != expected.sha256 or current.size != expected.size:
            _fail(code)


def build_natural_language_scaffold(
    config: NaturalLanguageScaffoldConfig | str | Path,
    projector_dir: str | Path,
    expected_projector_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build and atomically publish a deterministic 20-record fixture."""

    if isinstance(config, NaturalLanguageScaffoldConfig):
        cfg = config
        current, inventory = NaturalLanguageScaffoldConfig.load(cfg.path)
        if current != cfg:
            _fail("natural_language_scaffold_config_changed")
    else:
        cfg, inventory = NaturalLanguageScaffoldConfig.load(config)
    long_config_path = cfg.path.parent / "swebench_long_context_preflight_v1.yaml"
    try:
        long_cfg, long_inventory = _lc.LongContextTokenInventoryConfig.load(
            long_config_path
        )
    except _lc.LongContextPreflightError as exc:
        raise _lc_error(exc) from exc
    inventory.update(long_inventory)
    try:
        (
            projector_root,
            projector_manifest,
            snapshots,
            projector_config,
            projector_manifest_sidecar_sha256,
        ) = _lc._load_projector_artifact(
            long_cfg,
            projector_dir,
            expected_projector_manifest_sha256,
            inventory,
        )
    except _lc.LongContextPreflightError as exc:
        raise _lc_error(exc) from exc
    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink() or raw_output.parent.is_symlink():
        _fail("natural_language_scaffold_output_exists_or_overlaps_input")
    try:
        output_parent = raw_output.parent.resolve(strict=True)
    except OSError as exc:
        raise NaturalLanguageScaffoldError(
            "natural_language_scaffold_output_parent_invalid"
        ) from exc
    if not output_parent.is_dir() or output_parent != raw_output.parent.absolute():
        _fail("natural_language_scaffold_output_parent_invalid")
    output_parent_identity = _lc._directory_identity(output_parent.stat())
    output = output_parent / raw_output.name
    try:
        _lc._check_output(output, [projector_root, *inventory])
    except _lc.LongContextPreflightError as exc:
        raise _lc_error(exc) from exc
    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        _fail("natural_language_scaffold_temporary_conflict")
    try:
        try:
            source_rows = _lc._source_rows_first_pass(
                snapshots,
                projector_config=projector_config,
                projector_manifest=projector_manifest,
                max_records=cfg.max_input_records,
            )
            _lc._verify_groups(source_rows, projector_manifest)
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        records = _records_second_pass(cfg, snapshots, source_rows)
        temporary.mkdir(parents=True)
        files: list[dict[str, Any]] = []
        output_snapshots: dict[Path, _lc._BytesSnapshot] = {}
        for relative, split, source_variant, scaffold_variant in FIXED_FILES:
            path = temporary.joinpath(*Path(relative).parts)
            data = _write_jsonl(path, records[(split, scaffold_variant)])
            if not data or len(data) > cfg.max_output_file_bytes:
                _fail("natural_language_scaffold_output_file_invalid")
            try:
                snapshot = _lc._read_snapshot(
                    path,
                    "natural_language_scaffold_output_file_invalid",
                    max_bytes=cfg.max_output_file_bytes,
                )
                parsed = list(
                    _lc._iter_jsonl(
                        snapshot, "natural_language_scaffold_output_file_invalid"
                    )
                )
            except _lc.LongContextPreflightError as exc:
                raise _lc_error(exc) from exc
            if len(parsed) != 5:
                _fail("natural_language_scaffold_output_file_invalid")
            for _line_number, _line, row in parsed:
                _validate_record(row)
            output_snapshots[path] = snapshot
            files.append(
                {
                    "path": relative,
                    "sha256": snapshot.sha256,
                    "bytes": snapshot.size,
                    "records": len(parsed),
                    "split": split,
                    "source_variant": source_variant,
                    "scaffold_variant": scaffold_variant,
                }
            )
        architecture_contract = {
            "sha256": cfg.architecture_contract_sha256,
            **dict(cfg.raw["architecture_contract"]),
        }
        adapter_contract = {
            "sha256": cfg.adapter_control_policy_sha256,
            **dict(cfg.raw["adapter_control_contract"]),
        }
        serialization_contract = {
            "sha256": cfg.serialization_policy_sha256,
            **dict(cfg.raw["serialization_contract"]),
        }
        route_config = _mapping(
            cfg.raw["route_boundary_contract"],
            "natural_language_scaffold_config_invalid",
        )
        alora_config = _mapping(
            cfg.raw["alora_capability_contract"],
            "natural_language_scaffold_config_invalid",
        )
        route_activation_contract = {
            "semantics": route_config["semantics"],
            "planner_request_output": route_config["planner_request_output"],
            "validation_required": route_config["validation_required"],
            "commit_required": route_config["commit_required"],
            "commit_promotes_text_only": route_config["commit_promotes_text_only"],
            "planner_private_tail_kv_transfer_allowed": route_config[
                "planner_private_tail_kv_transfer_allowed"
            ],
            "committed_scaffold_reencode_required": route_config[
                "committed_scaffold_reencode_required"
            ],
            "committed_scaffold_reencode_producer": route_config[
                "committed_scaffold_reencode_producer"
            ],
            "committed_scaffold_reencode_adapter_state": route_config[
                "committed_scaffold_reencode_adapter_state"
            ],
            "committed_scaffold_output": route_config["committed_scaffold_output"],
            "expert_activation_request": route_config["expert_activation_request"],
            "expert_request_requires_committed_scaffold_as_input": route_config[
                "expert_request_requires_committed_scaffold_as_input"
            ],
            "alora_activation_semantics": alora_config["activation_semantics"],
            "same_request_activation_allowed": alora_config[
                "same_request_activation_allowed"
            ],
            "mid_request_generated_activation_allowed": alora_config[
                "mid_request_generated_activation_allowed"
            ],
            "mid_request_generated_trigger_switch_claimed": alora_config[
                "mid_request_generated_trigger_switch_claimed"
            ],
            "token_boundary_status": route_config["token_boundary_status"],
        }
        cache_config = _mapping(
            cfg.raw["cache_contract"],
            "natural_language_scaffold_config_invalid",
        )
        cache_contract = {
            key: cache_config[key]
            for key in (
                "shared_prefix_scope",
                "private_tail_scope",
                "exact_reuse_scope",
                "identical_token_order_positions_rope_required",
                "cache_identity_status",
                "exact_cache_reuse_enabled",
                "reuse_savings_tokens",
                "planner_private_tail_kv_reused_by_expert",
                "physical_kv_tensor_emitted",
                "full_generation_kv_shared_claimed",
                "committed_scaffold_reencode_executed",
                "downstream_immutable_segment_emitted",
            )
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "synthetic_fixture_ready",
            "input": {
                "projector_manifest_schema_version": (
                    "anchor.swebench-taskboard-projector-manifest.v2"
                ),
                "projector_manifest_sha256": expected_projector_manifest_sha256,
                "projector_manifest_sha256_sidecar_sha256": (
                    projector_manifest_sidecar_sha256
                ),
                "projector_manifest_schema_sha256": (
                    projector_config.manifest_schema_sha256
                ),
                "projector_config_sha256": projector_config.sha256,
                "projector_sidecar_schema_sha256": (
                    projector_config.sidecar_schema_sha256
                ),
                "segment_plan_schema_sha256": (
                    projector_config.segment_plan_schema_sha256
                ),
                "partitions": [
                    {
                        "path": relative,
                        "sha256": snapshots[(split, source_variant)].sha256,
                        "bytes": snapshots[(split, source_variant)].size,
                        "records": len(source_rows[(split, source_variant)]),
                        "split": split,
                        "variant": source_variant,
                    }
                    for relative, split, source_variant in (
                        ("train/clean.jsonl", "train", "clean"),
                        ("train/noisy.jsonl", "train", "noisy"),
                        ("calibration/clean.jsonl", "calibration", "clean"),
                    )
                ],
                "selected_source_partitions": [
                    "train/noisy.jsonl",
                    "calibration/clean.jsonl",
                ],
            },
            "producer": {
                "name": "anchor.natural-language-scaffold",
                "producer_version": PRODUCER_VERSION,
                "config_schema_version": CONFIG_SCHEMA,
                "config_sha256": cfg.sha256,
                "implementation_sha256": cfg.implementation_sha256,
                "record_schema_version": RECORD_SCHEMA,
                "record_schema_sha256": cfg.record_schema_sha256,
                "manifest_schema_sha256": cfg.manifest_schema_sha256,
                "smoke_contract_schema_version": SMOKE_SCHEMA,
                "smoke_contract_schema_sha256": cfg.smoke_schema_sha256,
                "smoke_contract_sha256": cfg.smoke_config_sha256,
            },
            "architecture_contract": architecture_contract,
            "visibility_contract": dict(cfg.raw["visibility_contract"]),
            "route_activation_contract": route_activation_contract,
            "cache_contract": cache_contract,
            "adapter_control_contract": adapter_contract,
            "serialization_contract": serialization_contract,
            "files": files,
            "counts": _manifest_counts(records, source_rows),
            "smoke_contract": {
                "schema_version": SMOKE_SCHEMA,
                "contract_sha256": cfg.smoke_config_sha256,
                "contract_schema_sha256": cfg.smoke_schema_sha256,
                "execution_mode": "contract_only_unexecuted",
                "model_file_basename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                "model_sha256_status": "runtime_required",
                "model_bytes_status": "runtime_required",
                "model_loaded": False,
                "smoke_executed": False,
                "provider_requests": 0,
                "network_requests": 0,
            },
            "manifest_sha256_sidecar_required": True,
            "canonical_gold_written": False,
            "heldout_written": False,
            "provider_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "network_requests": 0,
            "evaluation_status": "not_evaluated",
            "quality_validated": False,
            "execution_authorized": False,
            "claim_scope": "synthetic_fixture_contract_only",
        }
        if _contains_denied_key(manifest):
            _fail("natural_language_scaffold_body_exclusion_invalid")
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        sidecar_bytes = f"{manifest_sha256}  manifest.json\n".encode("ascii")
        sidecar_path = temporary / "manifest.json.sha256"
        sidecar_path.write_bytes(sidecar_bytes)
        try:
            manifest_snapshot = _lc._read_snapshot(
                manifest_path,
                "natural_language_scaffold_output_manifest_invalid",
                max_bytes=cfg.max_output_file_bytes,
            )
            sidecar_snapshot = _lc._read_snapshot(
                sidecar_path,
                "natural_language_scaffold_output_manifest_invalid",
                max_bytes=256,
            )
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        if (
            manifest_snapshot.sha256 != manifest_sha256
            or sidecar_snapshot.data != sidecar_bytes
        ):
            _fail("natural_language_scaffold_output_manifest_invalid")

        # Public hook used by tests and operators to prove one authenticated
        # bytes snapshot survived until atomic publication.
        try:
            _lc._verify_inventory_unchanged(inventory)
        except _lc.LongContextPreflightError as exc:
            raise _lc_error(exc) from exc
        _verify_bytes_unchanged(
            output_snapshots
            | {manifest_path: manifest_snapshot, sidecar_path: sidecar_snapshot},
            "natural_language_scaffold_output_changed",
        )
        if _lc._directory_identity(output_parent.stat()) != output_parent_identity:
            _fail("natural_language_scaffold_output_parent_changed")
        os.replace(temporary, output)
        return {
            "output_dir": str(output),
            "manifest_sha256": manifest_sha256,
            "record_schema_sha256": cfg.record_schema_sha256,
            "manifest_schema_sha256": cfg.manifest_schema_sha256,
            "config_sha256": cfg.sha256,
            "implementation_sha256": cfg.implementation_sha256,
            "smoke_schema_sha256": cfg.smoke_schema_sha256,
            "smoke_config_sha256": cfg.smoke_config_sha256,
            "records": 20,
            "pairs": 10,
            "unique_task_bundles": 2,
            "provider_requests": 0,
            "model_loads": 0,
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
