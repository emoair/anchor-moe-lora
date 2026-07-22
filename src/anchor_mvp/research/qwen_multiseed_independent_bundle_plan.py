"""Deterministic, zero-request planner for the next controlled proxy study.

The module authenticates the existing non-authorizing risk consumer and its
Producer follow-up metadata.  It plans seeds, balanced arm-order controls,
checkpoints, bundle-first split semantics, and the confirmation evaluation
matrix.  It never loads a model or authorizes training.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import yaml


CONFIG_VERSION = "anchor.qwen-multiseed-independent-bundle-plan-config.v1"
PLAN_VERSION = "anchor.qwen-multiseed-independent-bundle-plan.v1"
CONFIG_PATH = "configs/research/qwen_multiseed_independent_bundle_plan_v1.yaml"
CONFIG_SHA256 = "198ea3741b738cd2560b016499ef0d0d7a717b3b0cf8ecff482595e4e688888d"
IMPLEMENTATION_PATH = (
    "src/anchor_mvp/research/qwen_multiseed_independent_bundle_plan.py"
)

MASTER_SEEDS = (1337, 7331, 104729, 130363, 20260723)
SEED_DOMAINS = ("adapter_init", "record_order", "cuda")
DISCOVERY_ARMS = ("q_only", "q_plus_o", "wide_budget_matched")
MECHANISM_ARMS = (
    "q_only",
    "q_plus_o",
    "wide_budget_matched",
    "o_only",
    "k_plus_v",
)
CHECKPOINTS = (5, 10, 20, 40, 80)
EVAL_CELLS = (
    "old_task_new_template",
    "new_task_old_template",
    "new_task_new_template",
)
INFORMATION_FLOW_STRATA = (
    "prefix_evidence_selection",
    "prefix_evidence_plus_structured_private_writeback",
    "conflicting_allowed_evidence_resolution",
    "tool_result_commit_then_expert_private_tail",
    "ordered_long_prefix_retrieval",
)

_ROOT = Path(__file__).resolve().parents[3]
_THIS_FILE = Path(__file__).resolve()
_MAX_BYTES = 2 * 1024 * 1024
_REPARSE_POINT = 0x0400


class MultiSeedPlanError(RuntimeError):
    """Stable, content-free fail-closed planning error."""


def _fail(code: str) -> None:
    raise MultiSeedPlanError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compact_json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _is_link(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE_POINT
    )


def _safe_path(relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(code)
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        _fail(code)
    current = _ROOT
    if _is_link(current):
        _fail(code)
    for part in candidate.parts:
        current /= part
        if current.exists() and _is_link(current):
            _fail(code)
    try:
        current.resolve(strict=False).relative_to(_ROOT.resolve())
    except ValueError:
        _fail(code)
    return current


def _read_bytes(path: Path, code: str) -> tuple[bytes, str, tuple[int, int, int, int]]:
    try:
        if not path.is_file() or _is_link(path):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > _MAX_BYTES:
                _fail(code)
            data = handle.read(_MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except MultiSeedPlanError:
        raise
    except OSError as exc:
        raise MultiSeedPlanError(code) from exc
    identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        len(data) > _MAX_BYTES
        or len(data) != after.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != identity
        or (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != identity
        or _is_link(path)
    ):
        _fail(code)
    return data, _sha256(data), identity


def _assert_unchanged(
    path: Path,
    expected: tuple[bytes, str, tuple[int, int, int, int]],
    code: str,
) -> None:
    if _read_bytes(path, code) != expected:
        _fail(code)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            _fail("config_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _list(value: object, code: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _load_yaml(data: bytes) -> Mapping[str, Any]:
    try:
        value = yaml.load(data.decode("utf-8"), Loader=_UniqueLoader)
    except MultiSeedPlanError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise MultiSeedPlanError("config_invalid") from exc
    return _mapping(value, "config_invalid")


def _load_json(data: bytes, code: str) -> Mapping[str, Any]:
    def pairs(pairs_: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs_:
            if key in result:
                _fail(code)
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _: _fail(code),
        )
    except MultiSeedPlanError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiSeedPlanError(code) from exc
    return _mapping(value, code)


def _evaluate_authenticated_risk_consumer(
    snapshot: tuple[bytes, str, tuple[int, int, int, int]],
    implementation_path: Path,
    config_path: object,
    expected_sha256: object,
) -> Mapping[str, Any]:
    """Execute only the already-authenticated consumer source snapshot."""

    if snapshot[1] != expected_sha256:
        _fail("risk_consumer_implementation_sha256_mismatch")
    module_name = "anchor_mvp.research._authenticated_qwen_risk_consumer_" + snapshot[1]
    module = ModuleType(module_name)
    module.__file__ = str(implementation_path)
    module.__package__ = "anchor_mvp.research"
    previous = sys.modules.get(module_name)
    try:
        sys.modules[module_name] = module
        code = compile(
            snapshot[0].decode("utf-8"),
            str(implementation_path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        evaluator = getattr(module, "evaluate_risk_evidence", None)
        if not callable(evaluator):
            _fail("risk_consumer_evaluator_missing")
        decision = evaluator(config_path)
    except MultiSeedPlanError:
        raise
    except Exception as exc:
        raise MultiSeedPlanError("risk_consumer_evaluation_failed") from exc
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return _mapping(decision, "risk_consumer_decision_invalid")


def _derived_seed(domain: str, master: int) -> int:
    preimage = (f"anchor.controlled-proxy-followup.seed.v1\0{domain}\0{master}").encode(
        "utf-8"
    )
    digest = hashlib.sha256(preimage).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _seeded_arm_order(master: int, arms: Sequence[str]) -> list[str]:
    return sorted(
        arms,
        key=lambda arm: hashlib.sha256(
            b"anchor\0arm_order\0"
            + str(master).encode("ascii")
            + b"\0"
            + arm.encode("ascii")
        ).digest(),
    )


def _discovery_orders() -> list[list[str]]:
    return [list(order) for order in permutations(DISCOVERY_ARMS)]


def _mechanism_orders(master: int, seed_index: int) -> list[list[str]]:
    base = _seeded_arm_order(master, MECHANISM_ARMS)
    orders = [base[offset:] + base[:offset] for offset in range(5)]
    canonical = list(MECHANISM_ARMS)
    extra = canonical[seed_index:] + canonical[:seed_index]
    orders.append(extra)
    return orders


def _validate_orders(orders: Sequence[Sequence[str]], arms: Sequence[str]) -> None:
    if len(orders) != 6 or len({tuple(order) for order in orders}) != 6:
        _fail("arm_order_schedule_invalid")
    if any(tuple(sorted(order)) != tuple(sorted(arms)) for order in orders):
        _fail("arm_order_schedule_invalid")


def _factor_quota_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = _mapping(config["confirmation_dataset"], "dataset_invalid")
    matrix = _mapping(config["evaluation_matrix"], "eval_matrix_invalid")
    factors = list(EVAL_CELLS)
    domain = str(matrix["quota_domain"])
    rotation = (
        int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[:4], "big") % 3
    )
    if rotation != matrix["quota_rotation"]:
        _fail("factor_quota_rotation_drift")
    strata = list(dataset["information_flow_strata"])
    table: list[dict[str, Any]] = []
    for language_index, language in enumerate(dataset["languages"]):
        for stratum_index, stratum in enumerate(strata):
            omitted_index = (stratum_index + language_index + rotation) % 3
            omitted = factors[omitted_index]
            eval_counts = {factor: 0 if factor == omitted else 1 for factor in factors}
            train_counts = {factor: 2 - eval_counts[factor] for factor in factors}
            table.append(
                {
                    "language": language,
                    "stratum": stratum,
                    "omitted_eval_factor": omitted,
                    "factor_total": {factor: 2 for factor in factors},
                    "train": train_counts,
                    "eval_proxy": eval_counts,
                    "within_non_omitted_factor_pair_assignment": (
                        "pending_task_bundle_identities_low_hash_eval_high_hash_train"
                    ),
                }
            )
    return table


def _validate_global_order_balance(seed_plans: Sequence[Mapping[str, Any]]) -> None:
    for position in range(len(MECHANISM_ARMS)):
        counts = {arm: 0 for arm in MECHANISM_ARMS}
        for seed in seed_plans:
            for order in seed["mechanism_controls_throughput_arm_orders"]:
                counts[order[position]] += 1
        if set(counts.values()) != {6}:
            _fail("mechanism_arm_order_global_balance_invalid")


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "claim_scope",
        "paths",
        "source_bindings",
        "study_boundaries",
        "execution_fairness",
        "performance_contract",
        "replication",
        "confirmation_dataset",
        "evaluation_matrix",
        "gates",
        "claims",
        "audit",
    }:
        _fail("config_shape_invalid")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "zero_request_non_authorizing_diagnostic_plan_only"
    ):
        _fail("config_identity_invalid")
    paths = _mapping(config["paths"], "paths_invalid")
    if paths.get("repository_root") != "../.." or set(paths) != {
        "repository_root",
        "risk_consumer_config",
        "risk_consumer_implementation",
        "producer_followup_contract",
        "producer_followup_sidecar",
        "producer_risk_contract",
        "producer_risk_sidecar",
    }:
        _fail("paths_invalid")
    if set(_mapping(config["source_bindings"], "source_bindings_invalid")) != {
        "risk_consumer_config_sha256",
        "risk_consumer_implementation_sha256",
        "producer_followup_contract_sha256",
        "producer_followup_sidecar_sha256",
        "producer_risk_contract_sha256",
        "producer_risk_sidecar_sha256",
    }:
        _fail("source_bindings_invalid")
    boundaries = _mapping(config["study_boundaries"], "study_boundaries_invalid")
    if dict(boundaries) != {
        "producer_independent_confirmation": {
            "status": ("blocked_missing_confirmation_inventory_and_zero_overlap_proof"),
            "required_for_bundle_generalization": True,
            "satisfied_by_secondary_controlled_factorial_probe": False,
            "independent_confirmation_validated": False,
            "bundle_generalization_validated": False,
        },
        "secondary_controlled_factorial_probe": {
            "status": "blocked_metadata_blueprint_only",
            "scope": "factor_isolation_eval_proxy_only",
            "may_satisfy_independent_confirmation": False,
            "may_satisfy_bundle_generalization": False,
            "controlled_factorial_confirmation_validated": False,
        },
    }:
        _fail("study_boundaries_invalid")
    fairness = _mapping(config["execution_fairness"], "execution_fairness_invalid")
    if dict(fairness) != {
        "optimizer_steps": 80,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_length": 512,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "learning_rate": 0.00005,
            "betas": [0.9, 0.999],
            "eps": 0.00000001,
            "weight_decay": 0.01,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "zero_grad_set_to_none": True,
        },
        "precision": {
            "compute_dtype": "bfloat16",
            "tf32": True,
            "float32_matmul_precision": "high",
        },
        "gradient_checkpointing": {
            "enabled": True,
            "use_reentrant": False,
            "use_cache": False,
        },
        "determinism": {
            "torch_use_deterministic_algorithms": True,
            "warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
        "lora": {
            "alpha_over_rank": 2.0,
            "dropout": 0.0,
            "bias": "none",
            "use_rslora": False,
            "use_dora": False,
        },
        "reset_policy": {
            "fresh_frozen_base_per_arm": True,
            "fresh_zero_delta_adapter_per_arm": True,
            "fresh_optimizer_per_arm": True,
            "resume_allowed": False,
        },
        "identity": {
            "run_key_format": "track_id__master_seed__arm_id",
            "artifact_key_format": "track_id/master_seed/arm_id/checkpoint_step",
            "track_qualified_keys_required": True,
        },
    }:
        _fail("execution_fairness_invalid")
    performance = _mapping(
        config["performance_contract"], "performance_contract_invalid"
    )
    if dict(performance) != {
        "arm_order_scope": "throughput_only_not_training_repetition",
        "warmup_runs_per_arm_per_seed": 1,
        "timed_repetitions_per_arm_per_seed": 6,
        "timing_separate_from_training": True,
        "cuda_synchronize_before_after_timing": True,
        "runtime_environment_receipt_required": True,
        "gpu_thermal_clock_state_receipt_required": True,
        "single_gpu_serial_execution": True,
        "max_concurrent_gpu_jobs": 1,
        "peak_vram_cap_bytes": 5 * 1024**3,
    }:
        _fail("performance_contract_invalid")
    replication = _mapping(config["replication"], "replication_invalid")
    if set(replication) != {
        "master_seeds",
        "seed_derivation_algorithm",
        "seed_domains",
        "checkpoint_steps",
        "primary_endpoint",
        "primary_metric",
        "primary_aggregation",
        "uncertainty_method",
        "uncertainty_resamples",
        "uncertainty_seed",
        "confidence_level",
        "torch_deterministic_algorithms_required",
        "cuda_synchronize_before_after_timing",
        "budget_math",
        "shared_reference",
        "tracks",
        "cross_track_comparison_allowed",
        "retained_o_branch_may_be_relabelled_o_only",
    } or (
        tuple(replication.get("master_seeds", ())) != MASTER_SEEDS
        or tuple(replication.get("seed_domains", ())) != SEED_DOMAINS
        or replication.get("seed_derivation_algorithm")
        != "sha256_utf8_anchor_domain_nul_decimal_master_first_u32_mask31_v1"
        or replication.get("checkpoint_steps") != list(CHECKPOINTS)
        or replication.get("primary_endpoint") != 80
        or replication.get("uncertainty_resamples") != 10_000
        or replication.get("uncertainty_seed") != 20260723
        or replication.get("confidence_level") != 0.95
        or replication.get("torch_deterministic_algorithms_required") is not True
        or replication.get("cuda_synchronize_before_after_timing") is not True
        or replication.get("cross_track_comparison_allowed") is not False
        or replication.get("retained_o_branch_may_be_relabelled_o_only") is not False
    ):
        _fail("replication_invalid")
    budget_math = _mapping(replication.get("budget_math"), "budget_math_invalid")
    if dict(budget_math) != {
        "decoder_layers": 28,
        "hidden_size": 1536,
        "kv_size": 256,
        "q_or_o_parameters_per_all_layer_rank": 86_016,
        "k_or_v_parameters_per_all_layer_rank": 50_176,
    }:
        _fail("budget_math_invalid")
    if dict(_mapping(replication["shared_reference"], "shared_reference_invalid")) != {
        "id": "adapter_off",
        "kind": "shared_frozen_base_eval_reference",
        "independently_trained": False,
        "expected_trainable_parameters": 0,
    }:
        _fail("shared_reference_invalid")
    expected_tracks = {
        "discovery_replication": {
            "purpose": "replicate_the_original_three_arm_proxy_ranking",
            "common_budget_trainable_parameters": 1_376_256,
            "orders_per_seed": 6,
            "arm_order_algorithm": "all_six_permutations_lexicographic_v1",
            "arms": [
                {
                    "id": "q_only",
                    "independently_trained": True,
                    "rank_policy": {"q_proj": 16},
                    "expected_trainable_parameters": 1_376_256,
                    "expected_trainable_tensors": 56,
                },
                {
                    "id": "q_plus_o",
                    "independently_trained": True,
                    "rank_policy": {"q_proj": 8, "o_proj": 8},
                    "expected_trainable_parameters": 1_376_256,
                    "expected_trainable_tensors": 112,
                },
                {
                    "id": "wide_budget_matched",
                    "independently_trained": True,
                    "rank_policy": {"q_proj": 5, "o_proj": 4, "k_proj": 6, "v_proj": 6},
                    "expected_trainable_parameters": 1_376_256,
                    "expected_trainable_tensors": 224,
                },
            ],
        },
        "mechanism_controls": {
            "purpose": "compare_independently_trained_projection_controls",
            "common_budget_trainable_parameters": 1_204_224,
            "orders_per_seed": 6,
            "arm_order_algorithm": "five_seeded_cyclic_orders_plus_cross_seed_balanced_extra_v1",
            "arm_order_balance": "exact_global_position_balance_across_all_five_master_seeds",
            "arms": [
                {
                    "id": "q_only",
                    "independently_trained": True,
                    "rank_policy": {"q_proj": 14},
                    "expected_trainable_parameters": 1_204_224,
                    "expected_trainable_tensors": 56,
                },
                {
                    "id": "q_plus_o",
                    "independently_trained": True,
                    "rank_policy": {"q_proj": 7, "o_proj": 7},
                    "expected_trainable_parameters": 1_204_224,
                    "expected_trainable_tensors": 112,
                },
                {
                    "id": "wide_budget_matched",
                    "independently_trained": True,
                    "rank_policy": {"q_proj": 4, "o_proj": 3, "k_proj": 6, "v_proj": 6},
                    "expected_trainable_parameters": 1_204_224,
                    "expected_trainable_tensors": 224,
                },
                {
                    "id": "o_only",
                    "independently_trained": True,
                    "rank_policy": {"o_proj": 14},
                    "expected_trainable_parameters": 1_204_224,
                    "expected_trainable_tensors": 56,
                },
                {
                    "id": "k_plus_v",
                    "independently_trained": True,
                    "rank_policy": {"k_proj": 12, "v_proj": 12},
                    "expected_trainable_parameters": 1_204_224,
                    "expected_trainable_tensors": 112,
                },
            ],
        },
    }
    tracks = _mapping(replication.get("tracks"), "tracks_invalid")
    if dict(tracks) != expected_tracks:
        _fail("tracks_invalid")
    q_o = 86_016
    k_v = 50_176
    for track in expected_tracks.values():
        budget = int(track["common_budget_trainable_parameters"])
        for arm in track["arms"]:
            policy = arm["rank_policy"]
            actual = sum(
                int(rank) * (q_o if module in {"q_proj", "o_proj"} else k_v)
                for module, rank in policy.items()
            )
            expected_tensor_count = 2 * 28 * len(policy)
            if (
                actual != budget
                or arm.get("expected_trainable_parameters") != actual
                or arm.get("expected_trainable_tensors") != expected_tensor_count
            ):
                _fail("track_budget_mismatch")
    dataset = _mapping(config["confirmation_dataset"], "dataset_invalid")
    if (
        set(dataset)
        != {
            "status",
            "scope",
            "source_bundles_total",
            "train_bundles",
            "eval_proxy_bundles",
            "roles",
            "paired_variants",
            "expected_records",
            "split_group_key",
            "split_before_role_variant_augmentation",
            "all_role_variant_views_same_split",
            "languages",
            "information_flow_strata",
            "language_stratum_cells",
            "bundles_per_language_stratum_cell",
            "train_bundles_per_language_stratum_cell",
            "eval_bundles_per_language_stratum_cell",
            "task_inventory_status",
            "template_inventory_status",
            "task_template_pair_inventory_status",
            "eval_proxy_is_heldout",
        }
        or dataset.get("status") != "unavailable_not_generated"
        or dataset.get("scope") != "secondary_controlled_factorial_probe_only"
        or dataset.get("source_bundles_total") != 60
        or dataset.get("train_bundles") != 40
        or dataset.get("eval_proxy_bundles") != 20
        or dataset.get("roles") != 5
        or dataset.get("paired_variants") != 2
        or dataset.get("expected_records") != 600
        or dataset.get("split_group_key") != "task_bundle_sha256"
        or dataset.get("split_before_role_variant_augmentation") is not True
        or dataset.get("all_role_variant_views_same_split") is not True
        or dataset.get("languages") != ["en", "zh-CN"]
        or tuple(dataset.get("information_flow_strata", ())) != INFORMATION_FLOW_STRATA
        or len(set(dataset.get("information_flow_strata", ()))) != 5
        or dataset.get("language_stratum_cells") != 10
        or dataset.get("bundles_per_language_stratum_cell") != 6
        or dataset.get("train_bundles_per_language_stratum_cell") != 4
        or dataset.get("eval_bundles_per_language_stratum_cell") != 2
        or any(
            dataset.get(key) != "unavailable"
            for key in (
                "task_inventory_status",
                "template_inventory_status",
                "task_template_pair_inventory_status",
            )
        )
        or dataset.get("eval_proxy_is_heldout") is not False
        or dataset.get("language_stratum_cells")
        != len(dataset.get("languages", ())) * len(INFORMATION_FLOW_STRATA)
        or dataset.get("source_bundles_total")
        != dataset.get("language_stratum_cells")
        * dataset.get("bundles_per_language_stratum_cell")
        or dataset.get("train_bundles")
        != dataset.get("language_stratum_cells")
        * dataset.get("train_bundles_per_language_stratum_cell")
        or dataset.get("eval_proxy_bundles")
        != dataset.get("language_stratum_cells")
        * dataset.get("eval_bundles_per_language_stratum_cell")
        or dataset.get("expected_records")
        != dataset.get("source_bundles_total")
        * dataset.get("roles")
        * dataset.get("paired_variants")
    ):
        _fail("dataset_invalid")
    matrix = _mapping(config["evaluation_matrix"], "eval_matrix_invalid")
    cells = _list(matrix.get("cells"), "eval_matrix_invalid")
    expected_truth = (
        (True, False, False, True, True),
        (False, True, True, False, True),
        (False, True, False, True, True),
    )
    expected_cell_identities = (
        ("discovery_task", "confirmation_template"),
        ("confirmation_task", "discovery_template"),
        ("confirmation_task", "confirmation_template"),
    )
    if (
        set(matrix)
        != {
            "status",
            "design",
            "bundles_per_factor_cell",
            "factor_views_total",
            "allocation_order",
            "quota_algorithm",
            "quota_domain",
            "quota_rotation_derivation",
            "quota_rotation",
            "within_factor_pair_assignment_algorithm",
            "bundles_per_factor",
            "train_factor_quotas",
            "eval_factor_quotas",
            "six_eval_factor",
            "common_domain_inventories",
            "cells",
            "require_task_inventory_binding",
            "require_template_inventory_binding",
            "require_task_template_pair_inventory_binding",
            "global_task_nonoverlap_required",
            "global_template_nonoverlap_required",
            "global_task_template_pair_nonoverlap_required",
        }
        or matrix.get("status") != "metadata_blueprint_only"
        or matrix.get("design") != "controlled_factorial_confirmation"
        or matrix.get("bundles_per_factor_cell") != 20
        or matrix.get("factor_views_total") != 60
        or matrix.get("allocation_order")
        != "freeze_task_and_template_identities_then_pair_hash_then_language_stratum_factor_quota_split_then_role_variant_noise_length"
        or matrix.get("quota_algorithm")
        != "language_stratum_rotation_omits_one_eval_factor_then_bundle_hash_orders_each_remaining_pair_v1"
        or matrix.get("quota_domain") != "anchor.controlled-factorial-eval-quota.v1"
        or matrix.get("quota_rotation_derivation") != "uint32_be_sha256_domain_mod_3"
        or matrix.get("quota_rotation") != 0
        or matrix.get("within_factor_pair_assignment_algorithm")
        != "sha256_utf8_domain_nul_task_bundle_sha256_low_eval_high_train_v1"
        or matrix.get("bundles_per_factor") != 20
        or matrix.get("train_factor_quotas") != [13, 14, 13]
        or matrix.get("eval_factor_quotas") != [7, 6, 7]
        or matrix.get("six_eval_factor") != "new_task_old_template"
        or matrix.get("common_domain_inventories")
        != {
            "task_hash": "anchor.controlled-factorial-task-identity.v1",
            "template_hash": "anchor.controlled-factorial-template-identity.v1",
            "task_template_pair_hash": (
                "anchor.controlled-factorial-task-template-pair.v1"
            ),
        }
        or [cell.get("id") for cell in cells if isinstance(cell, Mapping)]
        != list(EVAL_CELLS)
        or any(
            set(_mapping(cell, "eval_matrix_invalid"))
            != {
                "id",
                "task_identity",
                "template_identity",
                "old_task_membership_required",
                "new_task_nonoverlap_required",
                "old_template_membership_required",
                "new_template_nonoverlap_required",
                "task_template_pair_nonoverlap_required",
            }
            or (cell.get("task_identity"), cell.get("template_identity"))
            != expected_cell_identities[index]
            or tuple(
                cell.get(key)
                for key in (
                    "old_task_membership_required",
                    "new_task_nonoverlap_required",
                    "old_template_membership_required",
                    "new_template_nonoverlap_required",
                    "task_template_pair_nonoverlap_required",
                )
            )
            != expected_truth[index]
            for index, cell in enumerate(cells)
        )
        or matrix.get("global_task_nonoverlap_required") is not False
        or matrix.get("global_template_nonoverlap_required") is not False
        or matrix.get("global_task_template_pair_nonoverlap_required") is not True
        or any(
            matrix.get(key) is not True
            for key in (
                "require_task_inventory_binding",
                "require_template_inventory_binding",
                "require_task_template_pair_inventory_binding",
            )
        )
    ):
        _fail("eval_matrix_invalid")
    table = _factor_quota_table(config)
    train_totals = {factor: 0 for factor in EVAL_CELLS}
    eval_totals = {factor: 0 for factor in EVAL_CELLS}
    for row in table:
        for factor in EVAL_CELLS:
            train_totals[factor] += row["train"][factor]
            eval_totals[factor] += row["eval_proxy"][factor]
    if list(train_totals.values()) != [13, 14, 13] or list(eval_totals.values()) != [
        7,
        6,
        7,
    ]:
        _fail("factor_quota_totals_invalid")
    gates = _mapping(config["gates"], "gates_invalid")
    if (
        set(gates)
        != {
            "missing_confirmation_dataset",
            "missing_factor_inventories_and_membership_proofs",
            "formal_v3_ready_count",
            "formal_v3_total",
            "protected_inventory_ready_count",
            "protected_inventory_total",
            "execution_ready",
            "materialization_ready",
            "training_authorized",
            "formal_training_authorized",
            "formal",
            "multi_seed_validated",
            "producer_independent_confirmation_validated",
            "controlled_factorial_confirmation_validated",
            "bundle_generalization_validated",
            "statistical_significance_claimed",
            "independently_trained_o_only_result_available",
            "independently_trained_k_plus_v_result_available",
        }
        or gates.get("missing_confirmation_dataset") is not True
        or gates.get("missing_factor_inventories_and_membership_proofs") is not True
        or gates.get("formal_v3_ready_count") != 0
        or gates.get("formal_v3_total") != 5
        or gates.get("protected_inventory_ready_count") != 2
        or gates.get("protected_inventory_total") != 6
        or any(
            gates.get(key) is not False
            for key in (
                "execution_ready",
                "materialization_ready",
                "training_authorized",
                "formal_training_authorized",
                "formal",
                "multi_seed_validated",
                "producer_independent_confirmation_validated",
                "controlled_factorial_confirmation_validated",
                "bundle_generalization_validated",
                "statistical_significance_claimed",
                "independently_trained_o_only_result_available",
                "independently_trained_k_plus_v_result_available",
            )
        )
    ):
        _fail("gates_invalid")
    claims = _mapping(config["claims"], "claims_invalid")
    if set(claims) != {
        "execution_ready",
        "materialization_ready",
        "training_authorized",
        "formal_training_authorized",
        "formal",
        "multi_seed_validated",
        "producer_independent_confirmation_validated",
        "controlled_factorial_confirmation_validated",
        "bundle_generalization_validated",
        "statistical_significance_claimed",
        "sample_efficiency_claimed",
        "compute_matched_across_tracks_claimed",
        "quality_validated",
        "performance_validated",
        "throughput_superiority_claimed",
    } or any(value is not False for value in claims.values()):
        _fail("claims_invalid")
    if dict(_mapping(config["audit"], "audit_invalid")) != {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_content_reads": 0,
        "dataset_body_reads": 0,
        "training_runs": 0,
    }:
        _fail("audit_invalid")


def _validate_sidecar(document_sha: str, data: bytes, filename: str, code: str) -> None:
    if data != f"{document_sha}  {filename}\n".encode("ascii"):
        _fail(code)


def _validate_producer_contract(
    contract: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    followup = _mapping(contract.get("producer_followup"), "followup_invalid")
    replication = _mapping(followup.get("replication_phase"), "followup_invalid")
    fixture = _mapping(followup.get("stratified_fixture_phase"), "followup_invalid")
    local_replication = _mapping(config["replication"], "replication_invalid")
    local_dataset = _mapping(config["confirmation_dataset"], "dataset_invalid")
    if (
        replication.get("seed_schedule") != list(MASTER_SEEDS)
        or replication.get("checkpoint_steps") != list(CHECKPOINTS)
        or replication.get("additional_mechanism_controls_required")
        != ["o_only_budget_matched", "k_plus_v_budget_matched"]
        or replication.get("formal_claim_allowed") is not False
        or replication.get("primary_metric") != local_replication.get("primary_metric")
        or replication.get("primary_aggregation")
        != local_replication.get("primary_aggregation")
        or replication.get("uncertainty_method")
        != local_replication.get("uncertainty_method")
    ):
        _fail("producer_replication_drift")
    schedules = _mapping(
        replication.get("seed_domain_schedules"), "producer_seed_schedule_invalid"
    )
    for domain in SEED_DOMAINS:
        expected = [_derived_seed(domain, seed) for seed in MASTER_SEEDS]
        schedule = _mapping(schedules.get(domain), "producer_seed_schedule_invalid")
        if schedule.get("values") != expected or schedule.get("sha256") != (
            _compact_json_sha256(expected)
        ):
            _fail("producer_seed_schedule_drift")
    if (
        fixture.get("status") != "design_only_not_generated"
        or fixture.get("source_bundles_total")
        != local_dataset.get("source_bundles_total")
        or fixture.get("train_bundles") != local_dataset.get("train_bundles")
        or fixture.get("eval_proxy_bundles") != local_dataset.get("eval_proxy_bundles")
        or fixture.get("expected_records") != local_dataset.get("expected_records")
        or fixture.get("group_key") != "task_bundle_sha256"
        or fixture.get("confirmation_blueprint_inventory_status")
        != "pending_not_generated"
        or fixture.get("namespace_neutral_blueprint_zero_overlap_status")
        != "unavailable_until_both_blueprint_inventories_exist"
        or fixture.get("independent_confirmation_claimed") is not False
    ):
        _fail("producer_confirmation_blueprint_drift")


def _requested_config(path: str | Path) -> Path:
    requested = Path(path)
    if ".." in requested.parts:
        _fail("config_path_invalid")
    candidate = requested if requested.is_absolute() else _ROOT / requested
    canonical = (_ROOT / CONFIG_PATH).resolve(strict=False)
    if candidate.resolve(strict=False) != canonical:
        _fail("config_path_invalid")
    return canonical


def build_dry_run_plan(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Authenticate dependencies and return a deterministic blocked plan."""

    path = _requested_config(config_path)
    config_snapshot = _read_bytes(path, "config_unreadable")
    implementation_snapshot = _read_bytes(
        _THIS_FILE, "planner_implementation_unreadable"
    )
    if config_snapshot[1] != CONFIG_SHA256 or b"\r" in config_snapshot[0]:
        _fail("config_sha256_mismatch")
    if b"\r" in implementation_snapshot[0]:
        _fail("planner_implementation_line_endings_invalid")
    config = _load_yaml(config_snapshot[0])
    _validate_config(config)
    paths = _mapping(config["paths"], "paths_invalid")
    bindings = _mapping(config["source_bindings"], "source_bindings_invalid")
    source_roles = {
        "risk_consumer_config": "risk_consumer_config_sha256",
        "risk_consumer_implementation": "risk_consumer_implementation_sha256",
        "producer_followup_contract": "producer_followup_contract_sha256",
        "producer_followup_sidecar": "producer_followup_sidecar_sha256",
        "producer_risk_contract": "producer_risk_contract_sha256",
        "producer_risk_sidecar": "producer_risk_sidecar_sha256",
    }
    snapshots: dict[str, tuple[bytes, str, tuple[int, int, int, int]]] = {}
    for role, digest_role in source_roles.items():
        snapshots[role] = _read_bytes(
            _safe_path(paths[role], f"{role}_path_invalid"), f"{role}_unreadable"
        )
        if snapshots[role][1] != bindings.get(digest_role):
            _fail(f"{role}_sha256_mismatch")
    _validate_sidecar(
        snapshots["producer_followup_contract"][1],
        snapshots["producer_followup_sidecar"][0],
        "synthetic_scaffold_controlled_proxy_followup_v1.json",
        "producer_followup_sidecar_invalid",
    )
    _validate_sidecar(
        snapshots["producer_risk_contract"][1],
        snapshots["producer_risk_sidecar"][0],
        "synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json",
        "producer_risk_sidecar_invalid",
    )
    contract = _load_json(
        snapshots["producer_followup_contract"][0], "producer_followup_invalid"
    )
    _validate_producer_contract(contract, config)

    decision = _evaluate_authenticated_risk_consumer(
        snapshots["risk_consumer_implementation"],
        _safe_path(
            paths["risk_consumer_implementation"],
            "risk_consumer_implementation_path_invalid",
        ),
        paths["risk_consumer_config"],
        bindings["risk_consumer_implementation_sha256"],
    )
    if (
        decision.get("status") != "blocked"
        or decision.get("training_authorized") is not False
        or decision.get("formal_training_authorized") is not False
        or decision.get("formal") is not False
        or decision.get("promotion_gates")
        != {
            "formal_v3_ready_count": 0,
            "formal_v3_total": 5,
            "protected_inventory_ready_count": 2,
            "protected_inventory_total": 6,
            "multi_seed_validated": False,
            "bundle_generalization_validated": False,
        }
    ):
        _fail("risk_consumer_gate_drift")

    seed_plans: list[dict[str, Any]] = []
    discovery_orders = _discovery_orders()
    _validate_orders(discovery_orders, DISCOVERY_ARMS)
    for seed_index, master in enumerate(MASTER_SEEDS):
        mechanism_orders = _mechanism_orders(master, seed_index)
        _validate_orders(mechanism_orders, MECHANISM_ARMS)
        seed_plans.append(
            {
                "master_seed": master,
                "derived_seeds": {
                    domain: _derived_seed(domain, master) for domain in SEED_DOMAINS
                },
                "discovery_replication_throughput_arm_orders": discovery_orders,
                "discovery_replication_throughput_arm_orders_sha256": (
                    _compact_json_sha256(discovery_orders)
                ),
                "mechanism_controls_throughput_arm_orders": mechanism_orders,
                "mechanism_controls_throughput_arm_orders_sha256": (
                    _compact_json_sha256(mechanism_orders)
                ),
            }
        )
    _validate_global_order_balance(seed_plans)
    quota_table = _factor_quota_table(config)
    quota_table_sha256 = _compact_json_sha256(quota_table)

    plan: dict[str, Any] = {
        "schema_version": PLAN_VERSION,
        "status": "blocked_controlled_factorial_confirmation_inputs_unavailable",
        "claim_scope": "zero_request_non_authorizing_diagnostic_plan_only",
        "self_bindings": {
            "config": {"path": CONFIG_PATH, "sha256": config_snapshot[1]},
            "implementation": {
                "path": IMPLEMENTATION_PATH,
                "sha256": implementation_snapshot[1],
            },
        },
        "source_bindings": {key: bindings[key] for key in sorted(bindings)},
        "study_boundaries": dict(config["study_boundaries"]),
        "execution_fairness": dict(config["execution_fairness"]),
        "performance_contract": dict(config["performance_contract"]),
        "replication": {
            "master_seed_count": len(MASTER_SEEDS),
            "master_seeds": list(MASTER_SEEDS),
            "seed_plans": seed_plans,
            "trainable_arm_ids_by_track": {
                "discovery_replication": list(DISCOVERY_ARMS),
                "mechanism_controls": list(MECHANISM_ARMS),
            },
            "shared_eval_reference_id": "adapter_off",
            "tracks": dict(config["replication"])["tracks"],
            "cross_track_comparison_allowed": False,
            "checkpoint_steps": list(CHECKPOINTS),
            "primary_endpoint": 80,
            "planned_independent_training_jobs": len(MASTER_SEEDS)
            * (len(DISCOVERY_ARMS) + len(MECHANISM_ARMS)),
            "planned_throughput_order_slots": len(MASTER_SEEDS)
            * 6
            * (len(DISCOVERY_ARMS) + len(MECHANISM_ARMS)),
            "planned_trainable_checkpoint_receipts": len(MASTER_SEEDS)
            * (len(DISCOVERY_ARMS) + len(MECHANISM_ARMS))
            * len(CHECKPOINTS),
        },
        "bundle_first_split": dict(config["confirmation_dataset"]),
        "evaluation_matrix": {
            **dict(config["evaluation_matrix"]),
            "language_stratum_factor_rotation_table": quota_table,
            "language_stratum_factor_rotation_table_sha256": quota_table_sha256,
            "actual_task_template_pair_inventories_available": False,
            "actual_bundle_assignment_materialized": False,
        },
        "gates": dict(config["gates"]),
        "claims": dict(config["claims"]),
        "audit": dict(config["audit"]),
    }
    plan["plan_sha256"] = _compact_json_sha256(plan)

    for role, snapshot in snapshots.items():
        _assert_unchanged(
            _safe_path(paths[role], f"{role}_path_invalid"),
            snapshot,
            f"{role}_changed",
        )
    _assert_unchanged(path, config_snapshot, "config_changed")
    _assert_unchanged(
        _THIS_FILE, implementation_snapshot, "planner_implementation_changed"
    )
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_dry_run_plan(args.config)
    except MultiSeedPlanError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
