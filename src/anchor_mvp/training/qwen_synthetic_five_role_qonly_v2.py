"""Strict five-role consumer and preflight for the 1,000-record proxy fixture.

This additive v2 module never executes training.  It authenticates the complete
fixture before selecting one explicit role, supports a local tokenizer-only
preflight, and emits a declarative serial five-adapter plan.  It does not change
or weaken the frozen v1 consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from anchor_mvp.research import synthetic_five_role_qonly_diagnostic_v1 as producer

from . import qwen_lora_diagnostic as qdiag
from . import qwen_synthetic_scaffold_diagnostic as base
from .config import ConfigError, _expand_env


CONFIG_VERSION = "anchor.qwen25-1.5b-synthetic-five-role-qonly-consumer-config.v2"
PREFLIGHT_VERSION = (
    "anchor.qwen25-1.5b-synthetic-five-role-qonly-tokenizer-preflight.v2"
)
PLAN_VERSION = "anchor.synthetic-five-role-qonly-serial-adapter-plan.v2"
CONFIG_PATH = "configs/training/qwen2_5_1_5b_synthetic_five_role_qonly_v2.yaml"
IMPLEMENTATION_PATH = "src/anchor_mvp/training/qwen_synthetic_five_role_qonly_v2.py"
DATASET_ROOT = "fixtures/research/synthetic_five_role_qonly_diagnostic_v1"
PRODUCER_CONFIG_PATH = "configs/research/synthetic_five_role_qonly_diagnostic_v1.yaml"
RECORD_SCHEMA_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1_manifest.schema.json"
)
PARTITION_PATHS = (
    "train/concise_rationale_plus_json.jsonl",
    "eval_proxy/concise_rationale_plus_json.jsonl",
)
ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
ROLE_STAGE = {role: index for index, role in enumerate(ROLES)}
ROLE_CANONICAL_STAGE = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "frontend_gen": "domain_builder",
    "frontend_review": "domain_review",
    "security_gate": "security",
}
LANGUAGES = ("en", "zh-CN")
STRATA = (
    "prefix_evidence_selection",
    "prefix_evidence_plus_structured_private_writeback",
    "conflicting_allowed_evidence_resolution",
    "tool_result_commit_then_expert_private_tail",
    "ordered_long_prefix_retrieval",
)
PRIMARY_MODULES = ("q_proj",)
CONTROL_LABELS = ("o_only", "q_plus_o")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 2_000_000
_MAX_PARTITION_BYTES = 50_000_000


@dataclass(frozen=True)
class RoleExample:
    record_id: str
    role_view_id: str
    task_bundle_id: str
    task_bundle_sha256: str
    task_semantic_sha256: str
    inner_task_id: str
    chain_root_sha256: str
    role: str
    canonical_stage: str
    split: str
    language: str
    stratum: str
    prompt: str
    target: str


@dataclass(frozen=True)
class RoleDataset:
    role: str
    manifest_sha256: str
    partition_sha256: Mapping[str, str]
    global_records_authenticated: int
    global_task_bundles_authenticated: int
    global_task_semantics_authenticated: int
    train: tuple[RoleExample, ...]
    eval_proxy: tuple[RoleExample, ...]


def _fail(code: str) -> None:
    raise ConfigError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _require_sha(value: object, code: str, *, allow_pending: bool = False) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    if not allow_pending and value == "0" * 64:
        _fail("five_role_fixture_identity_pending")
    return value


def _canonical_config_path(path: str | Path) -> Path:
    root = qdiag._project_root_from_module()
    canonical = root.joinpath(*PurePosixPath(CONFIG_PATH).parts)
    qdiag._assert_physical_path(
        canonical, require_file=True, label="five-role v2 config"
    )
    requested = Path(path)
    resolved = (
        Path(os.path.abspath(requested))
        if requested.is_absolute()
        else root.joinpath(*PurePosixPath(requested.as_posix()).parts)
    )
    resolved = Path(os.path.abspath(resolved))
    qdiag._assert_physical_path(
        resolved, require_file=True, label="five-role v2 config"
    )
    if os.path.normcase(str(resolved)) != os.path.normcase(str(canonical)):
        _fail("five_role_config_path_invalid")
    return canonical


def _repo_path(value: object, expected: str, *, directory: bool = False) -> Path:
    if value != expected:
        _fail("five_role_repo_path_drift")
    relative = PurePosixPath(expected)
    if relative.is_absolute() or ".." in relative.parts:
        _fail("five_role_repo_path_unsafe")
    return qdiag._assert_physical_path(
        qdiag._project_root_from_module().joinpath(*relative.parts),
        require_file=not directory,
        require_directory=directory,
        label=expected,
    )


def validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "claim_scope",
            "paths",
            "model",
            "dataset",
            "roles",
            "lora",
            "training",
            "precision",
            "kv_runtime_boundary",
            "output",
            "controls",
            "claims",
            "_config_path",
        },
        "five_role_config_fields_drift",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "synthetic_diagnostic_only_no_formal_or_training_authority"
        or _mapping(config.get("paths"), "five_role_paths_invalid")
        != {"project_root": "../.."}
    ):
        _fail("five_role_config_identity_drift")
    model = _mapping(config.get("model"), "five_role_model_invalid")
    _exact_keys(
        model,
        {
            "id",
            "local_path",
            "local_files_only",
            "allow_network",
            "trust_remote_code",
            "expected_source_revision",
            "expected_source_repo",
            "expected_config_json_sha256",
            "expected_model_safetensors_sha256",
            "expected_tokenizer_json_sha256",
            "expected_tokenizer_config_sha256",
        },
        "five_role_model_fields_drift",
    )
    if (
        model.get("id") != qdiag.EXPECTED_MODEL_ID
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("expected_source_revision") != qdiag.EXPECTED_SOURCE_REVISION
        or model.get("expected_source_repo") != qdiag.EXPECTED_SOURCE_REPO
    ):
        _fail("five_role_model_contract_drift")
    for field in (
        "expected_config_json_sha256",
        "expected_model_safetensors_sha256",
        "expected_tokenizer_json_sha256",
        "expected_tokenizer_config_sha256",
    ):
        _require_sha(model.get(field), f"five_role_model_{field}_invalid")
    dataset = _mapping(config.get("dataset"), "five_role_dataset_config_invalid")
    required_dataset = {
        "kind",
        "root",
        "producer_config",
        "manifest",
        "manifest_sidecar",
        "record_schema",
        "manifest_schema",
        "expected_manifest_sha256",
        "expected_record_schema_sha256",
        "expected_manifest_schema_sha256",
        "expected_records",
        "expected_task_bundles",
        "records_per_role",
        "train_records_per_role",
        "eval_proxy_records_per_role",
        "formal_inputs_allowed",
        "heldout_allowed",
        "protected_source_paths_allowed",
        "replaces_v1",
    }
    _exact_keys(dataset, required_dataset, "five_role_dataset_fields_drift")
    if (
        dataset.get("kind") != "synthetic_five_role_qonly_diagnostic_v1"
        or dataset.get("root") != DATASET_ROOT
        or dataset.get("producer_config") != PRODUCER_CONFIG_PATH
        or dataset.get("manifest") != f"{DATASET_ROOT}/manifest.json"
        or dataset.get("manifest_sidecar") != f"{DATASET_ROOT}/manifest.json.sha256"
        or dataset.get("record_schema") != RECORD_SCHEMA_PATH
        or dataset.get("manifest_schema") != MANIFEST_SCHEMA_PATH
        or dataset.get("expected_records") != 1000
        or dataset.get("expected_task_bundles") != 200
        or dataset.get("records_per_role") != 200
        or dataset.get("train_records_per_role") != 160
        or dataset.get("eval_proxy_records_per_role") != 40
        or dataset.get("formal_inputs_allowed") is not False
        or dataset.get("heldout_allowed") is not False
        or dataset.get("protected_source_paths_allowed") is not False
        or dataset.get("replaces_v1") is not False
    ):
        _fail("five_role_dataset_contract_drift")
    for field in (
        "expected_manifest_sha256",
        "expected_record_schema_sha256",
        "expected_manifest_schema_sha256",
    ):
        _require_sha(dataset.get(field), f"five_role_{field}_invalid")
    roles = _mapping(config.get("roles"), "five_role_roles_invalid")
    if roles != {"ordered": list(ROLES), "stage_index": ROLE_STAGE}:
        _fail("five_role_role_map_drift")
    if _mapping(config.get("lora"), "five_role_lora_invalid") != {
        "profile": "q_only",
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj"],
    }:
        _fail("five_role_primary_must_be_q_proj_only_rank4")
    if _mapping(config.get("training"), "five_role_training_invalid") != {
        "optimizer_steps_per_role": 160,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_length": 512,
        "learning_rate": 0.00005,
        "seed": 1337,
        "gradient_checkpointing": True,
        "use_cache": False,
    }:
        _fail("five_role_training_contract_drift")
    if _mapping(config.get("precision"), "five_role_precision_invalid") != {
        "compute_dtype": "bfloat16",
        "tf32": True,
        "float32_matmul_precision": "high",
    }:
        _fail("five_role_precision_contract_drift")
    if _mapping(
        config.get("kv_runtime_boundary"), "five_role_kv_runtime_boundary_invalid"
    ) != {
        "shared_prefix_adapter_mode": "off",
        "shared_prefix_read_only": True,
        "expert_activation": "q_proj_only",
        "expert_private_tail_append_only": True,
        "private_tail_includes_post_activation_prompt_and_generated_tokens": True,
        "private_tail_cross_expert_reuse": False,
        "committed_text_reencoded_into_next_shared_context": True,
        "full_generation_kv_shared": False,
        "ordinary_in_stack_q_lora_exact_kv_sharing": False,
        "runtime_private_tail_materialized": False,
        "execution_authorized": False,
    }:
        _fail("five_role_kv_runtime_boundary_drift")
    if _mapping(config.get("output"), "five_role_output_invalid") != {
        "adapter_dir_template": (
            "artifacts/diagnostics/"
            "qwen2_5_1_5b_synthetic_five_role_qonly_v2/{role}/adapter"
        ),
        "preflight_dir_template": (
            "artifacts/diagnostics/"
            "qwen2_5_1_5b_synthetic_five_role_qonly_v2/{role}/preflight"
        ),
    }:
        _fail("five_role_output_contract_drift")
    if _mapping(config.get("controls"), "five_role_controls_invalid") != {
        "execution_overlay_only": True,
        "duplicate_dataset_rows": False,
        "labels": list(CONTROL_LABELS),
        "admitted_to_primary_runner": False,
    }:
        _fail("five_role_controls_must_remain_overlay_only")
    if _mapping(config.get("claims"), "five_role_claims_invalid") != {
        "diagnostic_only": True,
        "dataset_proxy_ready": True,
        "records_materialized": True,
        "runtime_private_tail_materialized": False,
        "execution_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
        "two_track_600_record_contract_satisfied": False,
    }:
        _fail("five_role_claims_must_remain_blocked")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    canonical = _canonical_config_path(path)
    snapshot = base._read_snapshot(canonical, max_bytes=_MAX_METADATA_BYTES)
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError("five_role_config_invalid_utf8_yaml") from exc
    if not isinstance(value, Mapping):
        _fail("five_role_config_not_mapping")
    config = _expand_env(dict(value))
    config["_config_path"] = str(canonical)
    validate_config(config)
    snapshot.assert_unchanged()
    return config


def _strict_record(value: Mapping[str, Any]) -> RoleExample:
    if any(field in value for field in ("pair", "pair_id", "variant")):
        _fail("five_role_legacy_pair_or_variant_semantics_forbidden")
    role = value.get("role")
    split = value.get("split")
    stage = value.get("stage_index")
    canonical_stage = value.get("canonical_stage")
    view = value.get("view")
    claims = _mapping(value.get("claims"), "five_role_record_claims_invalid")
    audit = _mapping(value.get("audit"), "five_role_record_audit_invalid")
    if (
        role not in ROLES
        or split not in {"train", "eval_proxy"}
        or stage != ROLE_STAGE[role]
        or canonical_stage != ROLE_CANONICAL_STAGE[role]
        or view != "concise_rationale_plus_json"
        or claims.get("diagnostic_only") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal") is not False
        or any(
            audit.get(field) != 0
            for field in (
                "protected_body_reads",
                "provider_requests",
                "network_requests",
                "model_loads",
                "gpu_requests",
                "real_tool_executions",
            )
        )
    ):
        _fail("five_role_record_boundary_invalid")
    input_value = _mapping(value.get("input"), "five_role_record_input_invalid")
    target = _mapping(value.get("target"), "five_role_record_target_invalid")
    fields = {
        "record_id": value.get("record_id"),
        "role_view_id": value.get("role_view_id"),
        "task_bundle_id": value.get("task_bundle_id"),
        "task_bundle_sha256": value.get("task_bundle_sha256"),
        "task_semantic_sha256": value.get("task_semantic_sha256"),
        "inner_task_id": value.get("inner_task_id"),
        "chain_root_sha256": value.get("chain_root_sha256"),
        "role": role,
        "canonical_stage": canonical_stage,
        "split": split,
        "language": value.get("language"),
        "stratum": value.get("stratum"),
        "prompt": input_value.get("materialized_prompt"),
        "target": target.get("serialized_assistant_output"),
    }
    if not all(isinstance(item, str) and item for item in fields.values()):
        _fail("five_role_record_training_view_invalid")
    for field in (
        "task_bundle_sha256",
        "task_semantic_sha256",
        "chain_root_sha256",
    ):
        _require_sha(fields[field], f"five_role_record_{field}_invalid")
    return RoleExample(**fields)  # type: ignore[arg-type]


def _validate_causal_materialization(
    records: Sequence[Mapping[str, Any]],
) -> None:
    by_bundle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_bundle[str(record["task_bundle_sha256"])].append(record)
    for bundle_records in by_bundle.values():
        if len(bundle_records) != 5:
            _fail("five_role_bundle_cardinality_invalid")
        by_stage = {int(item["stage_index"]): item for item in bundle_records}
        if set(by_stage) != set(range(5)):
            _fail("five_role_bundle_stage_map_invalid")
        inventories: dict[int, tuple[Mapping[str, Any], ...]] = {}
        canonical_inventory_identity: (
            tuple[tuple[object, object, object, object, object, object], ...] | None
        ) = None
        for view_stage, view_record in by_stage.items():
            raw_inventory = _sequence(
                view_record.get("board_segment_inventory"),
                "five_role_segment_inventory_invalid",
            )
            if len(raw_inventory) != 5:
                _fail("five_role_segment_inventory_cardinality_invalid")
            inventory = tuple(
                _mapping(item, "five_role_segment_entry_invalid")
                for item in raw_inventory
            )
            identities: list[tuple[object, object, object, object, object, object]] = []
            for slot, entry in enumerate(inventory):
                if (
                    entry.get("stage_index") != slot
                    or entry.get("segment_ref") != f"S{slot}"
                    or entry.get("role") != ROLES[slot]
                    or entry.get("canonical_stage") != ROLE_CANONICAL_STAGE[ROLES[slot]]
                ):
                    _fail("five_role_segment_slot_binding_invalid")
                segment_id = entry.get("segment_id")
                content_sha = entry.get("content_sha256")
                if (
                    not isinstance(segment_id, str)
                    or not segment_id
                    or not isinstance(content_sha, str)
                ):
                    _fail("five_role_segment_identity_invalid")
                _require_sha(
                    content_sha,
                    "five_role_segment_content_sha256_invalid",
                    allow_pending=True,
                )
                identities.append(
                    (
                        segment_id,
                        entry.get("segment_ref"),
                        entry.get("role"),
                        entry.get("canonical_stage"),
                        entry.get("stage_index"),
                        content_sha,
                    )
                )
            identity_tuple = tuple(identities)
            if len({item[0] for item in identity_tuple}) != 5:
                _fail("five_role_segment_id_collision")
            if canonical_inventory_identity is None:
                canonical_inventory_identity = identity_tuple
            elif identity_tuple != canonical_inventory_identity:
                _fail("five_role_segment_inventory_cross_binding_invalid")
            target = _mapping(
                view_record.get("target"), "five_role_record_target_invalid"
            )
            canonical_route = _mapping(
                target.get("canonical_routing_json"),
                "five_role_canonical_route_invalid",
            )
            canonical_route_sha = _sha256(
                json.dumps(
                    canonical_route,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if (
                target.get("canonical_json_sha256") != canonical_route_sha
                or inventory[view_stage].get("content_sha256") != canonical_route_sha
            ):
                _fail("five_role_target_segment_cross_binding_invalid")
            summary = target.get("concise_rationale_summary")
            serialized = target.get("serialized_assistant_output")
            if (
                not isinstance(summary, str)
                or not isinstance(serialized, str)
                or serialized
                != f"{summary}\n"
                + json.dumps(
                    canonical_route,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                or target.get("output_sha256") != _sha256(serialized.encode("utf-8"))
            ):
                _fail("five_role_target_serialization_binding_invalid")
            inventories[view_stage] = inventory
        for stage, record in by_stage.items():
            prompt = str(
                _mapping(record["input"], "five_role_input_invalid")[
                    "materialized_prompt"
                ]
            )
            inventory = inventories[stage]
            forbidden = _sequence(
                record.get("forbidden_segment_ids"),
                "five_role_forbidden_inventory_invalid",
            )
            forbidden_entries: list[Mapping[str, Any]] = []
            previous_entries: list[Mapping[str, Any]] = []
            for raw_entry in inventory:
                entry = _mapping(raw_entry, "five_role_segment_entry_invalid")
                entry_stage = entry.get("stage_index")
                visibility = entry.get("visibility")
                if not isinstance(entry_stage, int) or entry_stage not in range(5):
                    _fail("five_role_segment_stage_invalid")
                if entry_stage < stage:
                    if visibility != "previous_committed":
                        _fail("five_role_previous_visibility_invalid")
                    previous_entries.append(entry)
                else:
                    expected = (
                        "current_target" if entry_stage == stage else "future_target"
                    )
                    if visibility != expected:
                        _fail("five_role_forbidden_visibility_invalid")
                    forbidden_entries.append(entry)
            expected_forbidden = [
                entry.get("segment_id") for entry in forbidden_entries
            ]
            if list(forbidden) != expected_forbidden:
                _fail("five_role_forbidden_inventory_mismatch")
            allowed_context = _sequence(
                _mapping(record["input"], "five_role_input_invalid").get(
                    "allowed_context_segments"
                ),
                "five_role_allowed_context_invalid",
            )
            expected_allowed: list[dict[str, Any]] = []
            for previous_stage, previous_entry in enumerate(previous_entries):
                previous_target = _mapping(
                    by_stage[previous_stage].get("target"),
                    "five_role_previous_target_invalid",
                )
                expected_allowed.append(
                    {
                        "segment_ref": previous_entry.get("segment_ref"),
                        "role": previous_entry.get("role"),
                        "canonical_stage": previous_entry.get("canonical_stage"),
                        "stage_index": previous_entry.get("stage_index"),
                        "committed_summary": previous_target.get(
                            "concise_rationale_summary"
                        ),
                    }
                )
            materialized_allowed = [
                dict(
                    _mapping(
                        item,
                        "five_role_allowed_context_entry_invalid",
                    )
                )
                for item in allowed_context
            ]
            if materialized_allowed != expected_allowed:
                _fail("five_role_allowed_context_mismatch")
            if any(
                str(item["segment_ref"]) not in prompt
                or str(item["committed_summary"]) not in prompt
                for item in expected_allowed
            ):
                _fail("five_role_allowed_context_not_materialized")
            forbidden_needles: set[str] = set()
            for entry in forbidden_entries:
                for key in (
                    "segment_id",
                    "segment_ref",
                    "content",
                    "content_sha256",
                ):
                    value = entry.get(key)
                    if isinstance(value, str) and value:
                        forbidden_needles.add(value)
            for future_stage in range(stage, 5):
                future = by_stage[future_stage]
                target = _mapping(
                    future.get("target"), "five_role_future_target_invalid"
                )
                for key in (
                    "concise_rationale_summary",
                    "serialized_assistant_output",
                    "canonical_json_sha256",
                    "output_sha256",
                ):
                    value = target.get(key)
                    if isinstance(value, str) and value:
                        forbidden_needles.add(value)
                canonical = target.get("canonical_routing_json")
                if isinstance(canonical, Mapping):
                    forbidden_needles.add(
                        json.dumps(
                            canonical,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
            if any(needle in prompt for needle in forbidden_needles):
                _fail("five_role_forbidden_content_reached_prompt")


def _validate_global_records(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> tuple[RoleExample, ...]:
    if len(records) != 1000:
        _fail("five_role_global_record_count_invalid")
    examples = tuple(_strict_record(record) for record in records)
    if (
        len({item.record_id for item in examples}) != 1000
        or len({item.role_view_id for item in examples}) != 1000
    ):
        _fail("five_role_record_or_view_id_collision")
    if Counter(item.role for item in examples) != Counter(
        {role: 200 for role in ROLES}
    ):
        _fail("five_role_global_role_count_invalid")
    bundles: dict[str, list[RoleExample]] = defaultdict(list)
    semantics_by_role: dict[str, set[str]] = defaultdict(set)
    semantics_by_language: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        bundles[item.task_bundle_sha256].append(item)
        semantics_by_role[item.role].add(item.task_semantic_sha256)
        semantics_by_language[item.language].add(item.task_semantic_sha256)
    if len(bundles) != 200:
        _fail("five_role_global_bundle_count_invalid")
    if (
        set(semantics_by_language) != set(LANGUAGES)
        or semantics_by_language["en"] & semantics_by_language["zh-CN"]
    ):
        _fail("five_role_language_semantic_overlap_invalid")
    global_semantics = set().union(*semantics_by_role.values())
    if len(global_semantics) != 200 or any(
        values != global_semantics for values in semantics_by_role.values()
    ):
        _fail("five_role_task_semantic_role_inventory_invalid")
    for items in bundles.values():
        if (
            {item.role for item in items} != set(ROLES)
            or len({item.task_bundle_id for item in items}) != 1
            or len({item.task_semantic_sha256 for item in items}) != 1
            or len({item.inner_task_id for item in items}) != 1
            or len({item.chain_root_sha256 for item in items}) != 1
            or len({item.language for item in items}) != 1
            or len({item.stratum for item in items}) != 1
            or len({item.split for item in items}) != 1
        ):
            _fail("five_role_bundle_cross_binding_invalid")
    split_bundle_counts = Counter(items[0].split for items in bundles.values())
    if split_bundle_counts != Counter({"train": 160, "eval_proxy": 40}):
        _fail("five_role_bundle_split_count_invalid")
    cell_counts = Counter(
        (item.language, item.stratum, item.role, item.split) for item in examples
    )
    expected_cells = Counter()
    for language in LANGUAGES:
        for stratum in STRATA:
            for role in ROLES:
                expected_cells[(language, stratum, role, "train")] = 16
                expected_cells[(language, stratum, role, "eval_proxy")] = 4
    if cell_counts != expected_cells:
        _fail("five_role_language_stratum_role_quota_invalid")
    split_contract = _mapping(
        manifest.get("split_contract"), "five_role_manifest_split_invalid"
    )
    generation_contract = _mapping(
        manifest.get("generation_contract"),
        "five_role_manifest_generation_invalid",
    )
    semantic_contract = _mapping(
        manifest.get("semantic_identity_contract"),
        "five_role_manifest_semantic_contract_invalid",
    )
    ablation_contract = _mapping(
        manifest.get("ablation_contract"),
        "five_role_manifest_ablation_invalid",
    )
    compatibility = _mapping(
        manifest.get("compatibility_boundary"),
        "five_role_manifest_compatibility_invalid",
    )
    claims = _mapping(manifest.get("claims"), "five_role_manifest_claims_invalid")
    counts = _mapping(manifest.get("counts"), "five_role_manifest_counts_invalid")
    if (
        split_contract.get("group_key") != "task_bundle_sha256"
        or split_contract.get("eval_proxy_is_heldout") is not False
        or generation_contract.get("source_namespace")
        != "anchor.synthetic-five-role-qonly-diagnostic.v1"
        or semantic_contract.get("unique_task_semantics") != 200
        or semantic_contract.get("each_role_covers_same_200_semantics") is not True
        or semantic_contract.get("en_zh_intersection_count") != 0
        or semantic_contract.get("translation_pair_count") != 0
        or ablation_contract.get("primary_label") != "q_only"
        or ablation_contract.get("q_only_is_only_primary") is not True
        or ablation_contract.get("diagnostic_control_labels") != list(CONTROL_LABELS)
        or ablation_contract.get("control_arms_are_execution_overlays_only") is not True
        or ablation_contract.get("control_arm_rows_materialized") is not False
        or compatibility.get("pair_count") != 0
        or compatibility.get("replaces_100_v1") is not False
        or compatibility.get("satisfies_independent_600_materialization") is not False
        or compatibility.get("satisfies_factorial_600_materialization") is not False
        or counts.get("records") != 1000
        or counts.get("task_bundles") != 200
        or claims.get("diagnostic_only") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal") is not False
        or claims.get("eval_proxy_is_heldout") is not False
    ):
        _fail("five_role_manifest_boundary_invalid")
    _validate_causal_materialization(records)
    return examples


def load_role_dataset(config: Mapping[str, Any], role: str) -> RoleDataset:
    if role not in ROLES:
        _fail("five_role_explicit_role_required")
    dataset_config = _mapping(config.get("dataset"), "five_role_dataset_config_invalid")
    expected_manifest_sha = _require_sha(
        dataset_config.get("expected_manifest_sha256"),
        "five_role_manifest_sha_invalid",
    )
    expected_record_schema_sha = _require_sha(
        dataset_config.get("expected_record_schema_sha256"),
        "five_role_record_schema_sha_invalid",
    )
    expected_manifest_schema_sha = _require_sha(
        dataset_config.get("expected_manifest_schema_sha256"),
        "five_role_manifest_schema_sha_invalid",
    )
    root = _repo_path(dataset_config["root"], DATASET_ROOT, directory=True)
    producer_config = _repo_path(
        dataset_config["producer_config"], PRODUCER_CONFIG_PATH
    )
    manifest_snapshot = base._read_snapshot(
        _repo_path(dataset_config["manifest"], f"{DATASET_ROOT}/manifest.json"),
        max_bytes=_MAX_METADATA_BYTES,
    )
    sidecar_snapshot = base._read_snapshot(
        _repo_path(
            dataset_config["manifest_sidecar"],
            f"{DATASET_ROOT}/manifest.json.sha256",
        ),
        max_bytes=1024,
    )
    record_schema_snapshot = base._read_snapshot(
        _repo_path(dataset_config["record_schema"], RECORD_SCHEMA_PATH),
        max_bytes=_MAX_METADATA_BYTES,
    )
    manifest_schema_snapshot = base._read_snapshot(
        _repo_path(dataset_config["manifest_schema"], MANIFEST_SCHEMA_PATH),
        max_bytes=_MAX_METADATA_BYTES,
    )
    if (
        manifest_snapshot.sha256 != expected_manifest_sha
        or record_schema_snapshot.sha256 != expected_record_schema_sha
        or manifest_schema_snapshot.sha256 != expected_manifest_schema_sha
    ):
        _fail("five_role_source_identity_mismatch")
    if sidecar_snapshot.data != (
        f"{manifest_snapshot.sha256}  manifest.json\n".encode("ascii")
    ):
        _fail("five_role_manifest_sidecar_invalid")
    try:
        manifest = base._strict_json(manifest_snapshot.data, "five-role manifest")
        record_schema = base._strict_json(
            record_schema_snapshot.data, "five-role record schema"
        )
        manifest_schema = base._strict_json(
            manifest_schema_snapshot.data, "five-role manifest schema"
        )
    except ConfigError:
        raise ConfigError(
            "five_role_metadata_json_invalid_without_record_content"
        ) from None
    try:
        Draft202012Validator.check_schema(record_schema)
        Draft202012Validator.check_schema(manifest_schema)
        Draft202012Validator(manifest_schema).validate(manifest)
    except (SchemaError, ValidationError):
        raise ConfigError("five_role_schema_or_manifest_validation_failed") from None
    partitions = _sequence(
        manifest.get("partitions"), "five_role_partition_inventory_invalid"
    )
    by_path = {
        str(_mapping(item, "five_role_partition_invalid").get("path")): _mapping(
            item, "five_role_partition_invalid"
        )
        for item in partitions
    }
    if set(by_path) != set(PARTITION_PATHS):
        _fail("five_role_partition_paths_invalid")
    partition_snapshots: dict[str, base.FileSnapshot] = {}
    records: list[Mapping[str, Any]] = []
    validator = Draft202012Validator(record_schema)
    for relative in PARTITION_PATHS:
        entry = by_path[relative]
        snapshot = base._read_snapshot(
            qdiag._assert_physical_path(
                root.joinpath(*PurePosixPath(relative).parts),
                require_file=True,
                label="five-role partition",
            ),
            max_bytes=_MAX_PARTITION_BYTES,
        )
        if snapshot.sha256 != entry.get("sha256") or len(snapshot.data) != entry.get(
            "bytes"
        ):
            _fail("five_role_partition_identity_mismatch")
        try:
            raw_records = base._strict_jsonl(snapshot.data, relative)
        except ConfigError:
            raise ConfigError(
                "five_role_partition_jsonl_invalid_without_record_content"
            ) from None
        if len(raw_records) != entry.get("records"):
            _fail("five_role_partition_count_mismatch")
        for record in raw_records:
            try:
                validator.validate(record)
            except ValidationError:
                raise ConfigError("five_role_record_schema_validation_failed") from None
        partition_snapshots[relative] = snapshot
        records.extend(raw_records)
    try:
        audited = producer.audit_dataset(
            qdiag._project_root_from_module(), producer_config, root
        )
    except (ConfigError, RuntimeError):
        raise ConfigError(
            "five_role_producer_audit_failed_without_record_content"
        ) from None
    if dict(audited) != dict(manifest):
        _fail("five_role_producer_consumer_manifest_disagreement")
    examples = _validate_global_records(records, manifest)
    selected = tuple(item for item in examples if item.role == role)
    train = tuple(
        sorted(
            (item for item in selected if item.split == "train"),
            key=lambda item: item.record_id,
        )
    )
    eval_proxy = tuple(
        sorted(
            (item for item in selected if item.split == "eval_proxy"),
            key=lambda item: item.record_id,
        )
    )
    if len(train) != 160 or len(eval_proxy) != 40:
        _fail("five_role_selected_role_count_invalid")
    for snapshot in (
        manifest_snapshot,
        sidecar_snapshot,
        record_schema_snapshot,
        manifest_schema_snapshot,
        *partition_snapshots.values(),
    ):
        snapshot.assert_unchanged()
    return RoleDataset(
        role=role,
        manifest_sha256=manifest_snapshot.sha256,
        partition_sha256={
            path: snapshot.sha256 for path, snapshot in partition_snapshots.items()
        },
        global_records_authenticated=1000,
        global_task_bundles_authenticated=200,
        global_task_semantics_authenticated=200,
        train=train,
        eval_proxy=eval_proxy,
    )


def _role_output(config: Mapping[str, Any], role: str, kind: str) -> str:
    if role not in ROLES or kind not in {"adapter", "preflight"}:
        _fail("five_role_output_request_invalid")
    output = _mapping(config.get("output"), "five_role_output_invalid")
    key = f"{kind}_dir_template"
    relative = str(output[key]).format(role=role)
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or role not in path.parts:
        _fail("five_role_output_not_role_isolated")
    return path.as_posix()


def build_serial_launcher_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    jobs = []
    for index, role in enumerate(ROLES):
        jobs.append(
            {
                "serial_index": index,
                "role": role,
                "train_records": 160,
                "eval_proxy_records": 40,
                "adapter_output": _role_output(config, role, "adapter"),
                "preflight_output": _role_output(config, role, "preflight"),
                "preflight_argv": [
                    "python",
                    "scripts/research/prepare_synthetic_five_role_qonly_v2.py",
                    "--config",
                    CONFIG_PATH,
                    "--role",
                    role,
                    "--tokenizer-only",
                    "--publish-preflight",
                ],
                "training_argv": None,
                "training_execution_supported": False,
                "lora": {
                    "target_modules": list(PRIMARY_MODULES),
                    "rank": 4,
                    "alpha": 8,
                },
                "training": {
                    "optimizer_steps": 160,
                    "micro_batch_size": 1,
                    "sequence_length": 512,
                    "use_cache": False,
                },
                "kv_runtime_boundary": dict(
                    _mapping(
                        config.get("kv_runtime_boundary"),
                        "five_role_kv_runtime_boundary_invalid",
                    )
                ),
            }
        )
    return {
        "schema_version": PLAN_VERSION,
        "status": "declarative_plan_only_execution_blocked",
        "execution_mode": "strictly_serial_one_adapter_at_a_time",
        "jobs": jobs,
        "controls": {
            "labels": list(CONTROL_LABELS),
            "execution_overlay_only": True,
            "duplicated_rows": False,
            "admitted_to_primary_runner": False,
        },
        "claims": {
            "diagnostic_only": True,
            "models_loaded": False,
            "gpu_requested": False,
            "training_executed": False,
            "runtime_private_tail_materialized": False,
            "execution_authorized": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
        },
    }


def _dry_run_from_dataset(
    config: Mapping[str, Any], role: str, dataset: RoleDataset
) -> dict[str, Any]:
    if dataset.role != role:
        _fail("five_role_selected_dataset_role_mismatch")
    return {
        "schema_version": PREFLIGHT_VERSION,
        "status": "passed_dataset_only_dry_run_training_blocked",
        "role": role,
        "dataset": {
            "manifest_sha256": dataset.manifest_sha256,
            "partition_sha256": dict(dataset.partition_sha256),
            "global_records_authenticated_before_role_filter": (
                dataset.global_records_authenticated
            ),
            "global_task_bundles_authenticated": (
                dataset.global_task_bundles_authenticated
            ),
            "global_task_semantics_authenticated": (
                dataset.global_task_semantics_authenticated
            ),
            "role_train_records": len(dataset.train),
            "role_eval_proxy_records": len(dataset.eval_proxy),
        },
        "selected_role_contract": {
            "single_role_only": True,
            "role": role,
            "mixed_role_training_rejected": True,
        },
        "lora": {
            "profile": "q_only",
            "target_modules": ["q_proj"],
            "rank": 4,
            "o_proj_allowed": False,
        },
        "numerics": {
            "tf32": True,
            "compute_dtype": "bfloat16",
            "micro_batch_size": 1,
            "sequence_length": 512,
            "use_cache": False,
        },
        "kv_runtime_boundary": dict(
            _mapping(
                config.get("kv_runtime_boundary"),
                "five_role_kv_runtime_boundary_invalid",
            )
        ),
        "serial_launcher_plan": build_serial_launcher_plan(config),
        "claims": {
            "diagnostic_only": True,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "gpu_requested": False,
            "training_executed": False,
            "runtime_private_tail_materialized": False,
            "execution_authorized": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
        },
        "audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_body_reads": 0,
        },
    }


def build_dry_run(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    return _dry_run_from_dataset(config, role, load_role_dataset(config, role))


def tokenizer_only_preflight(
    config: Mapping[str, Any], role: str, tokenizer: Any
) -> dict[str, Any]:
    dataset = load_role_dataset(config, role)
    converted = base.ScaffoldDataset(
        manifest={},
        manifest_sha256=dataset.manifest_sha256,
        partition_sha256=dataset.partition_sha256,
        train=tuple(
            base.ScaffoldExample(
                record_id=item.record_id,
                split=item.split,
                variant="concise_rationale_plus_json",
                source_bundle_id=item.task_bundle_sha256,
                prompt=item.prompt,
                target=item.target,
            )
            for item in dataset.train
        ),
        eval_proxy=tuple(
            base.ScaffoldExample(
                record_id=item.record_id,
                split=item.split,
                variant="concise_rationale_plus_json",
                source_bundle_id=item.task_bundle_sha256,
                prompt=item.prompt,
                target=item.target,
            )
            for item in dataset.eval_proxy
        ),
    )
    try:
        token_report = base.token_length_preflight(
            tokenizer, converted, sequence_length=512
        )
    except Exception:
        raise RuntimeError(
            "five_role_tokenizer_preflight_failed_without_record_content"
        ) from None
    report = _dry_run_from_dataset(config, role, dataset)
    report["status"] = "passed_tokenizer_only_preflight_training_blocked"
    report["token_lengths"] = token_report
    report["claims"]["tokenizer_loaded"] = True
    report["audit"]["tokenizer_loads"] = 1
    return report


def _load_local_tokenizer(config: Mapping[str, Any]) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("five_role_tokenizer_runtime_unavailable") from None

    model = _mapping(config.get("model"), "five_role_model_invalid")
    model_path = Path(str(model["local_path"])).expanduser()
    if not model_path.is_absolute():
        model_path = qdiag._project_root_from_module() / model_path
    qdiag._assert_physical_path(
        model_path, require_directory=True, label="local tokenizer directory"
    )
    tokenizer_artifacts = {
        "tokenizer.json": model["expected_tokenizer_json_sha256"],
        "tokenizer_config.json": model["expected_tokenizer_config_sha256"],
    }
    for name, expected_sha in tokenizer_artifacts.items():
        snapshot = base._read_snapshot(
            qdiag._assert_physical_path(
                model_path / name,
                require_file=True,
                label="local tokenizer artifact",
            ),
            max_bytes=_MAX_PARTITION_BYTES,
        )
        if snapshot.sha256 != expected_sha:
            _fail("five_role_tokenizer_artifact_identity_mismatch")
        snapshot.assert_unchanged()
    try:
        return AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
    except Exception:
        raise RuntimeError(
            "five_role_local_tokenizer_load_failed_without_record_content"
        ) from None


def _publish_receipt(
    config: Mapping[str, Any], role: str, value: Mapping[str, Any]
) -> Path:
    relative = _role_output(config, role, "preflight")
    output = qdiag._project_root_from_module().joinpath(*PurePosixPath(relative).parts)
    if os.path.lexists(output):
        _fail("five_role_preflight_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = _sha256(raw)
    sidecar = f"{digest}  preflight.json\n".encode("ascii")
    with qdiag._adapter_staging_directory(output) as temporary:
        receipt_path = temporary / "preflight.json"
        sidecar_path = temporary / "preflight.json.sha256"
        receipt_path.write_bytes(raw)
        sidecar_path.write_bytes(sidecar)
        if (
            base._read_snapshot(receipt_path, max_bytes=_MAX_METADATA_BYTES).data != raw
            or base._read_snapshot(sidecar_path, max_bytes=1024).data != sidecar
        ):
            _fail("five_role_preflight_staging_authentication_failed")
        qdiag._rename_directory_noreplace(temporary, output)
    published = base._read_snapshot(
        output / "preflight.json", max_bytes=_MAX_METADATA_BYTES
    )
    published_sidecar = base._read_snapshot(
        output / "preflight.json.sha256", max_bytes=1024
    )
    if published.data != raw or published_sidecar.data != (
        f"{digest}  preflight.json\n".encode("ascii")
    ):
        _fail("five_role_preflight_publish_authentication_failed")
    return output / "preflight.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate all 1,000 synthetic records, then select one explicit "
            "five-role Q-only partition. Default is dataset-only and never loads "
            "a model, CUDA, provider, or training runtime."
        ),
        epilog=(
            "Example: python scripts/research/"
            "prepare_synthetic_five_role_qonly_v2.py --role planner. "
            "Use --tokenizer-only for the separately authenticated local tokenizer."
        ),
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Pinned consumer YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=ROLES,
        help="Select exactly one role after global authentication.",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Load only the local tokenizer; models and CUDA remain forbidden.",
    )
    parser.add_argument(
        "--publish-preflight",
        action="store_true",
        help="Atomically publish a role-isolated tokenizer preflight receipt.",
    )
    return parser


def _failure_hint(code: str) -> str:
    if "fixture_identity_pending" in code:
        return "The final fixture hashes are not locked in the consumer config."
    if "identity_mismatch" in code or "sha" in code:
        return "A pinned artifact changed; rebuild/audit the fixture and update all hashes together."
    if "tokenizer" in code:
        return "Check the configured local tokenizer directory and its two pinned file hashes."
    if "preflight_output_exists" in code:
        return "Receipts are no-replace; use a fresh output namespace or intentionally archive the old receipt."
    if "forbidden" in code or "causal" in code:
        return "Causal materialization failed closed before tokenization; rebuild the dataset."
    return "The strict preflight stopped before model/GPU/training execution."


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.publish_preflight and not args.tokenizer_only:
            _fail("five_role_publish_requires_tokenizer_only")
        if args.tokenizer_only:
            result = tokenizer_only_preflight(
                config, args.role, _load_local_tokenizer(config)
            )
            if args.publish_preflight:
                receipt = _publish_receipt(config, args.role, result)
                result["published_receipt"] = str(
                    receipt.relative_to(qdiag._project_root_from_module())
                ).replace("\\", "/")
        else:
            result = build_dry_run(config, args.role)
    except (ConfigError, RuntimeError) as exc:
        code = str(exc)
        print(
            json.dumps(
                {
                    "schema_version": PREFLIGHT_VERSION,
                    "status": "blocked",
                    "error_code": code,
                    "hint": _failure_hint(code),
                    "claims": {
                        "model_loaded": False,
                        "gpu_requested": False,
                        "training_executed": False,
                        "execution_authorized": False,
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
