"""Run the split-safe CPU proxy for role-conditioned Query specialization.

The executable path trains tiny role-specific low-rank Query residuals over
frozen synthetic block keys.  Contract fixtures are validated but never used
for gradient updates.  Train and evaluation task groups are generated
separately, every task has all five role views, and all signal gates are
computed only on unseen evaluation tasks.

This is deliberately not a language-model result: it does not load a
foundation model, optimize strict JSON, establish attention causality, or
measure KV-cache reuse and throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anchor_mvp.research.query_specialization import (  # noqa: E402
    BLOCK_KINDS,
    ROLES,
    QuerySpecializationError,
    QueryTrainingRecord,
    TaskBoardSidecar,
    block_attention_auxiliary_loss,
    build_training_view,
    dataset_summary,
    load_taskboard_sidecar_dataset,
    lora_target_modules,
)
from anchor_mvp.research.training_release_consumer import (  # noqa: E402
    QUERY_SPECIALIZATION_CONSUMER_ID,
    QUERY_SPECIALIZATION_CONSUMER_VERSION,
    QUERY_SPECIALIZATION_IMPLEMENTATION_FILES,
    QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT,
    TrainingReleaseConsumerError,
    load_training_release_lock,
    validate_release_lock_schema,
    validate_source_disjoint_schema,
)


CLAIM_SCOPE = "experimental_proxy_scaffold_only"
TOP_LEVEL_CONFIG_FIELDS = {
    "schema_version",
    "claim_scope",
    "dataset_contract",
    "materialization",
    "proxy",
    "full_model_followup",
}
DATASET_CONTRACT_FIELDS = {
    "root",
    "manifest",
    "files",
    "sidecar_schema",
    "manifest_schema",
    "segment_plan_schema",
    "projector_config",
    "expected_sidecar_schema_sha256",
    "expected_manifest_schema_sha256",
    "expected_segment_plan_schema_sha256",
    "expected_projector_config_sha256",
    "split_group_key",
    "require_clean_noisy_pairs",
    "require_disjoint_source_tasks",
    "require_all_role_views",
    "hard_exclude_forbidden_blocks",
    "canonical_gold_written",
    "provider_requests",
    "heldout_content_read",
    "heldout_content_emitted",
    "split_preserved",
    "augmentation_applied_after_split",
}
MATERIALIZATION_FIELDS = {
    "schema_version",
    "record_schema",
    "output_root",
    "max_shard_bytes",
    "require_train_and_calibration_for_q1_smoke",
}
FIXED_SIDECAR_FILES = (
    "train/clean.jsonl",
    "train/noisy.jsonl",
    "calibration/clean.jsonl",
)
PROXY_FIELDS = {
    "profile",
    "hidden_size",
    "rank",
    "alpha",
    "seed",
    "train_tasks",
    "eval_tasks",
    "epochs",
    "learning_rate",
    "attention_temperature",
    "task_variation",
    "distractor_copies",
    "losses",
    "metrics_schema",
    "output",
    "signal_thresholds",
}
PROXY_LOSS_FIELDS = {"distractor_weight", "clean_noisy_consistency_weight"}
SIGNAL_THRESHOLD_FIELDS = {
    "minimum_top1_relevant_rate",
    "minimum_relevant_mass_gain",
    "minimum_mean_min_relevant_block_mass",
    "maximum_distractor_mass",
    "minimum_correct_role_mass_gap",
    "minimum_clean_noisy_query_cosine",
}
EVAL_METRIC_FIELDS = {
    "top1_relevant_rate",
    "mean_relevant_mass",
    "mean_min_relevant_block_mass",
    "mean_distractor_mass",
    "mean_clean_noisy_query_cosine",
    "mean_correct_role_mass_gap",
    "mean_inter_role_query_cosine_diagnostic",
}
SIGNAL_CHECK_FIELDS = {
    "eval_top1_relevant",
    "eval_relevant_mass_gain",
    "eval_min_relevant_block_mass",
    "eval_distractor_mass",
    "eval_correct_role_mass_gap",
    "eval_clean_noisy_consistency",
}
FULL_MODEL_FIELDS = {
    "enabled",
    "release_lock",
    "base_config",
    "lora",
    "optimization",
    "attention_implementation",
    "direct_attention_supervision",
    "token_to_block_aggregation",
    "evidence_selection_target",
}
FULL_MODEL_RELEASE_LOCK_FIELDS = {
    "status",
    "root",
    "schema",
    "expected_schema_sha256",
    "expected_manifest_sha256",
    "expected_consumer_contract_sha256",
    "expected_consumer_id",
    "expected_consumer_version",
    "source_disjoint_schema",
    "expected_source_disjoint_schema_sha256",
}
FULL_MODEL_LORA_FIELDS = {
    "primary_profile",
    "control_profiles",
    "rank",
    "alpha",
    "dropout",
    "dtype",
    "freeze_base_model",
}
FULL_MODEL_OPTIMIZATION_FIELDS = {"learning_rate", "max_steps"}


@dataclass(frozen=True)
class ProxyExample:
    """One synthetic, content-free task/role/variant view."""

    task_id: str
    split: str
    role: str
    variant: str
    pair_id: str
    role_index: int
    keys: Any
    context: Any
    relevant: Any
    distractor: Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split-safe CPU Query-LoRA task-board specialization proxy"
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/research/query_specialization_mvp.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    mode.add_argument("--execute", action="store_true", help="train the tiny CPU proxy")
    parser.add_argument("--output", help="override the content-free metrics path")
    return parser


def _decode_utf8_snapshot(snapshot: bytes, source: str) -> str:
    try:
        return snapshot.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuerySpecializationError(f"{source}: invalid UTF-8") from exc


def _load_config_snapshot(snapshot: bytes, source: str) -> dict[str, Any]:
    """Parse the exact config bytes whose digest is recorded by the run."""

    value = yaml.safe_load(_decode_utf8_snapshot(snapshot, source))
    if not isinstance(value, dict):
        raise QuerySpecializationError("experiment config must be a mapping")
    _reject_unknown_fields(value, TOP_LEVEL_CONFIG_FIELDS, "config")
    if value.get("schema_version") != "anchor.query-specialization-experiment.v1":
        raise QuerySpecializationError("unsupported query-specialization config")
    if value.get("claim_scope") != CLAIM_SCOPE:
        raise QuerySpecializationError(f"claim_scope must be {CLAIM_SCOPE!r}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    return _load_config_snapshot(path.read_bytes(), str(path))


def _resolve(root: Path, value: Any, path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QuerySpecializationError(f"{path} must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuerySpecializationError(f"{path} must be a mapping")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], path: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise QuerySpecializationError(
            f"{path} contains unknown fields: {sorted(unknown)}"
        )


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QuerySpecializationError(f"{path} must be a positive integer")
    return value


def _bounded_float(
    value: Any,
    path: str,
    *,
    lower: float = 0.0,
    upper: float,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuerySpecializationError(f"{path} must be numeric")
    result = float(value)
    valid_lower = result >= lower if allow_zero else result > lower
    if not valid_lower or result > upper:
        bracket = "[" if allow_zero else "("
        raise QuerySpecializationError(
            f"{path} must be in {bracket}{lower}, {upper}]"
        )
    return result


def _require_true(value: Any, path: str) -> None:
    if value is not True:
        raise QuerySpecializationError(f"{path} must remain true for this MVP")


def _required_sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QuerySpecializationError(f"{path} must be a lowercase SHA-256")
    return value


def _load_contract_dataset(
    config: Mapping[str, Any],
) -> tuple[
    tuple[TaskBoardSidecar, ...],
    tuple[QueryTrainingRecord, ...],
    dict[str, Any],
    dict[str, Any],
]:
    """Load and bind the fixed producer output without touching canonical Gold."""

    contract = _mapping(config.get("dataset_contract"), "dataset_contract")
    _reject_unknown_fields(contract, DATASET_CONTRACT_FIELDS, "dataset_contract")
    files = contract.get("files")
    if not isinstance(files, list) or tuple(files) != FIXED_SIDECAR_FILES:
        raise QuerySpecializationError(
            f"dataset_contract.files must be exactly {list(FIXED_SIDECAR_FILES)!r}"
        )
    root = _resolve(REPO_ROOT, contract.get("root"), "dataset_contract.root")
    manifest_path = _resolve(
        REPO_ROOT, contract.get("manifest"), "dataset_contract.manifest"
    )
    if manifest_path != (root / "manifest.json").resolve():
        raise QuerySpecializationError(
            "dataset_contract.manifest must be <dataset_contract.root>/manifest.json"
        )

    expected_hashes: dict[str, tuple[Path, str]] = {
        "sidecar_schema": (
            _resolve(
                REPO_ROOT,
                contract.get("sidecar_schema"),
                "dataset_contract.sidecar_schema",
            ),
            _required_sha256(
                contract.get("expected_sidecar_schema_sha256"),
                "dataset_contract.expected_sidecar_schema_sha256",
            ),
        ),
        "manifest_schema": (
            _resolve(
                REPO_ROOT,
                contract.get("manifest_schema"),
                "dataset_contract.manifest_schema",
            ),
            _required_sha256(
                contract.get("expected_manifest_schema_sha256"),
                "dataset_contract.expected_manifest_schema_sha256",
            ),
        ),
        "segment_plan_schema": (
            _resolve(
                REPO_ROOT,
                contract.get("segment_plan_schema"),
                "dataset_contract.segment_plan_schema",
            ),
            _required_sha256(
                contract.get("expected_segment_plan_schema_sha256"),
                "dataset_contract.expected_segment_plan_schema_sha256",
            ),
        ),
        "projector_config": (
            _resolve(
                REPO_ROOT,
                contract.get("projector_config"),
                "dataset_contract.projector_config",
            ),
            _required_sha256(
                contract.get("expected_projector_config_sha256"),
                "dataset_contract.expected_projector_config_sha256",
            ),
        ),
    }
    actual_hashes: dict[str, str] = {}
    contract_snapshots: dict[str, bytes] = {}
    for label, (path, expected) in expected_hashes.items():
        snapshot = path.read_bytes()
        contract_snapshots[label] = snapshot
        actual = hashlib.sha256(snapshot).hexdigest()
        if actual != expected:
            raise QuerySpecializationError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )
        actual_hashes[f"{label}_sha256"] = actual
    for label in ("sidecar_schema", "manifest_schema", "segment_plan_schema"):
        schema = json.loads(
            _decode_utf8_snapshot(
                contract_snapshots[label], str(expected_hashes[label][0])
            )
        )
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise QuerySpecializationError(f"{label} must use JSON Schema 2020-12")

    for key in (
        "require_clean_noisy_pairs",
        "require_disjoint_source_tasks",
        "require_all_role_views",
        "hard_exclude_forbidden_blocks",
        "split_preserved",
        "augmentation_applied_after_split",
    ):
        _require_true(contract.get(key), f"dataset_contract.{key}")
    if contract.get("split_group_key") != "task_bundle_sha256":
        raise QuerySpecializationError(
            "dataset_contract.split_group_key must be task_bundle_sha256"
        )
    for key in (
        "canonical_gold_written",
        "heldout_content_read",
        "heldout_content_emitted",
    ):
        if contract.get(key) is not False:
            raise QuerySpecializationError(f"dataset_contract.{key} must remain false")
    if contract.get("provider_requests") != 0:
        raise QuerySpecializationError("dataset_contract.provider_requests must be 0")

    sidecars, manifest, validation = load_taskboard_sidecar_dataset(
        root,
        manifest_path=manifest_path,
        expected_config_sha256=expected_hashes["projector_config"][1],
        expected_sidecar_schema_sha256=expected_hashes["sidecar_schema"][1],
        expected_manifest_schema_sha256=expected_hashes["manifest_schema"][1],
        expected_segment_plan_schema_sha256=expected_hashes[
            "segment_plan_schema"
        ][1],
    )
    records = tuple(sidecar.training_record for sidecar in sidecars)
    authenticated = validation.get("authenticated_file_sha256")
    if not isinstance(authenticated, Mapping):
        raise QuerySpecializationError(
            "sidecar loader omitted authenticated snapshot hashes"
        )
    expected_authenticated = ("manifest.json", *FIXED_SIDECAR_FILES)
    if set(authenticated) != set(expected_authenticated):
        raise QuerySpecializationError(
            "sidecar loader authenticated-file set does not match the contract"
        )
    combined = hashlib.sha256()
    for relative in expected_authenticated:
        digest = _required_sha256(
            authenticated.get(relative),
            f"sidecar_loader.authenticated_file_sha256[{relative!r}]",
        )
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
    actual_hashes["dataset_contract_sha256"] = combined.hexdigest()
    return tuple(sidecars), records, manifest, {**validation, **actual_hashes}


def _role_relevant_kinds(
    records: Sequence[QueryTrainingRecord],
) -> dict[str, tuple[str, ...]]:
    by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    for record in records:
        if record.variant != "clean":
            continue
        by_id = {block.block_id: block for block in record.blocks}
        by_role[record.role].update(
            by_id[block_id].kind for block_id in record.targets.relevant
        )
    missing = [role for role, kinds in by_role.items() if not kinds]
    if missing:
        raise QuerySpecializationError(
            f"contract fixture lacks relevant block kinds for roles: {missing}"
        )
    return {role: tuple(sorted(kinds)) for role, kinds in by_role.items()}


def _digest_strings(values: Sequence[str]) -> str:
    canonical = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_proxy_result_contract(result: Mapping[str, Any]) -> None:
    """Fail closed on the checked-in metrics contract without extra packages."""

    expected = {
        "schema_version",
        "probe_kind",
        "claim_scope",
        "metrics_scope",
        "experiment_config_sha256",
        "contract_fixture_sha256",
        "metrics_schema_sha256",
        "runner_sha256",
        "query_contract_module_sha256",
        "runtime_versions",
        "signal_thresholds",
        "foundation_model_loaded",
        "strict_json_optimized",
        "contract_fixture_used_for_gradient_training",
        "base_projection_frozen",
        "frozen_parameters",
        "trainable_parameters",
        "epochs",
        "learning_rate",
        "attention_temperature",
        "task_variation",
        "distractor_copies",
        "final_loss",
        "split_audit",
        "before_eval",
        "after_eval",
        "eval_delta",
        "signal_checks",
        "proxy_signal_passed",
        "non_claims",
    }
    if set(result) != expected:
        raise QuerySpecializationError(
            "proxy result fields do not match the metrics contract"
        )
    for key in (
        "experiment_config_sha256",
        "contract_fixture_sha256",
        "metrics_schema_sha256",
        "runner_sha256",
        "query_contract_module_sha256",
    ):
        value = result[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise QuerySpecializationError(f"proxy result {key} is not SHA-256")
    thresholds = _mapping(result["signal_thresholds"], "result.signal_thresholds")
    if set(thresholds) != SIGNAL_THRESHOLD_FIELDS:
        raise QuerySpecializationError("proxy result thresholds do not match config")
    checks = _mapping(result["signal_checks"], "result.signal_checks")
    if set(checks) != SIGNAL_CHECK_FIELDS or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise QuerySpecializationError("proxy result signal checks are malformed")
    if result["proxy_signal_passed"] is not all(checks.values()):
        raise QuerySpecializationError("proxy signal aggregate disagrees with checks")
    split_audit = _mapping(result["split_audit"], "result.split_audit")
    if split_audit.get("task_overlap") != 0:
        raise QuerySpecializationError("proxy result reports task leakage")
    for section in ("before_eval", "after_eval"):
        metrics = _mapping(result[section], f"result.{section}")
        if set(metrics) != EVAL_METRIC_FIELDS or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in metrics.values()
        ):
            raise QuerySpecializationError(f"proxy result {section} is malformed")


def proxy_task_ids(proxy: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return deterministic, disjoint task-group IDs for direct testing."""

    train_count = _positive_int(proxy.get("train_tasks"), "proxy.train_tasks")
    eval_count = _positive_int(proxy.get("eval_tasks"), "proxy.eval_tasks")
    train = tuple(f"proxy-train-{index:04d}" for index in range(train_count))
    evaluate = tuple(f"proxy-eval-{index:04d}" for index in range(eval_count))
    if set(train) & set(evaluate):
        raise QuerySpecializationError("proxy train/eval task groups overlap")
    return {"train": train, "eval": evaluate}


def _plan(
    config: Mapping[str, Any],
    records: Sequence[QueryTrainingRecord],
    *,
    producer_manifest: Mapping[str, Any],
    dataset_validation: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(config.get("dataset_contract"), "dataset_contract")
    proxy = _mapping(config.get("proxy"), "proxy")
    full_model = _mapping(config.get("full_model_followup"), "full_model_followup")
    materialization = _mapping(config.get("materialization"), "materialization")
    _reject_unknown_fields(contract, DATASET_CONTRACT_FIELDS, "dataset_contract")
    _reject_unknown_fields(
        materialization, MATERIALIZATION_FIELDS, "materialization"
    )
    _reject_unknown_fields(proxy, PROXY_FIELDS, "proxy")
    _reject_unknown_fields(full_model, FULL_MODEL_FIELDS, "full_model_followup")
    release_lock = _mapping(
        full_model.get("release_lock"), "full_model_followup.release_lock"
    )
    _reject_unknown_fields(
        release_lock,
        FULL_MODEL_RELEASE_LOCK_FIELDS,
        "full_model_followup.release_lock",
    )
    full_model_lora = _mapping(
        full_model.get("lora"), "full_model_followup.lora"
    )
    full_model_optimization = _mapping(
        full_model.get("optimization"), "full_model_followup.optimization"
    )
    _reject_unknown_fields(
        full_model_lora,
        FULL_MODEL_LORA_FIELDS,
        "full_model_followup.lora",
    )
    _reject_unknown_fields(
        full_model_optimization,
        FULL_MODEL_OPTIMIZATION_FIELDS,
        "full_model_followup.optimization",
    )

    metrics_schema_path = _resolve(
        REPO_ROOT, proxy.get("metrics_schema"), "proxy.metrics_schema"
    )
    metrics_schema_raw = metrics_schema_path.read_bytes()
    metrics_schema = json.loads(metrics_schema_raw)
    if metrics_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise QuerySpecializationError("metrics schema must use JSON Schema 2020-12")

    for record in records:
        build_training_view(record)
    relevant_kinds = _role_relevant_kinds(records)

    profile = str(proxy.get("profile", ""))
    if profile != "q_only":
        raise QuerySpecializationError("the proxy primary profile must remain q_only")
    modules = lora_target_modules(profile)
    width = _positive_int(proxy.get("hidden_size"), "proxy.hidden_size")
    rank = _positive_int(proxy.get("rank"), "proxy.rank")
    if rank > width:
        raise QuerySpecializationError("proxy.rank must not exceed proxy.hidden_size")
    alpha = _positive_int(proxy.get("alpha"), "proxy.alpha")
    task_ids = proxy_task_ids(proxy)
    _positive_int(proxy.get("epochs"), "proxy.epochs")
    _bounded_float(proxy.get("learning_rate"), "proxy.learning_rate", upper=0.1)
    _bounded_float(
        proxy.get("attention_temperature"),
        "proxy.attention_temperature",
        upper=1.0,
    )
    _bounded_float(
        proxy.get("task_variation"),
        "proxy.task_variation",
        upper=1.0,
        allow_zero=True,
    )
    _positive_int(proxy.get("distractor_copies"), "proxy.distractor_copies")
    losses = _mapping(proxy.get("losses"), "proxy.losses")
    _reject_unknown_fields(losses, PROXY_LOSS_FIELDS, "proxy.losses")
    _bounded_float(
        losses.get("distractor_weight"),
        "proxy.losses.distractor_weight",
        upper=10.0,
        allow_zero=True,
    )
    _bounded_float(
        losses.get("clean_noisy_consistency_weight"),
        "proxy.losses.clean_noisy_consistency_weight",
        upper=10.0,
        allow_zero=True,
    )
    thresholds = _mapping(proxy.get("signal_thresholds"), "proxy.signal_thresholds")
    _reject_unknown_fields(
        thresholds, SIGNAL_THRESHOLD_FIELDS, "proxy.signal_thresholds"
    )
    for key in SIGNAL_THRESHOLD_FIELDS:
        _bounded_float(
            thresholds.get(key),
            f"proxy.signal_thresholds.{key}",
            upper=1.0,
            allow_zero=True,
        )
    enabled = full_model.get("enabled")
    if not isinstance(enabled, bool):
        raise QuerySpecializationError("full_model_followup.enabled must be boolean")
    release_schema_path = _resolve(
        REPO_ROOT,
        release_lock.get("schema"),
        "full_model_followup.release_lock.schema",
    )
    release_schema_sha256 = _required_sha256(
        release_lock.get("expected_schema_sha256"),
        "full_model_followup.release_lock.expected_schema_sha256",
    )
    try:
        validated_schema_sha256 = validate_release_lock_schema(
            release_schema_path, release_schema_sha256
        )
    except TrainingReleaseConsumerError as exc:
        raise QuerySpecializationError(str(exc)) from exc
    source_disjoint_schema_path = _resolve(
        REPO_ROOT,
        release_lock.get("source_disjoint_schema"),
        "full_model_followup.release_lock.source_disjoint_schema",
    )
    source_disjoint_schema_sha256 = _required_sha256(
        release_lock.get("expected_source_disjoint_schema_sha256"),
        "full_model_followup.release_lock.expected_source_disjoint_schema_sha256",
    )
    try:
        validated_source_disjoint_schema_sha256 = validate_source_disjoint_schema(
            source_disjoint_schema_path, source_disjoint_schema_sha256
        )
    except TrainingReleaseConsumerError as exc:
        raise QuerySpecializationError(str(exc)) from exc
    expected_consumer_id = release_lock.get("expected_consumer_id")
    expected_consumer_version = release_lock.get("expected_consumer_version")
    if expected_consumer_id != QUERY_SPECIALIZATION_CONSUMER_ID:
        raise QuerySpecializationError(
            "full_model_followup.release_lock.expected_consumer_id is invalid"
        )
    if expected_consumer_version != QUERY_SPECIALIZATION_CONSUMER_VERSION:
        raise QuerySpecializationError(
            "full_model_followup.release_lock.expected_consumer_version is invalid"
        )
    if not enabled:
        if (
            release_lock.get("status") != "unavailable"
            or release_lock.get("root") is not None
            or release_lock.get("expected_manifest_sha256") is not None
            or release_lock.get("expected_consumer_contract_sha256") is not None
        ):
            raise QuerySpecializationError(
                "disabled full_model_followup requires an unavailable release lock"
            )
        release_validation: dict[str, Any] = {
            "schema_version": "anchor.generic-train-release-lock.v2",
            "status": "unavailable",
            "formal_training_authorized": False,
            "schema_sha256": validated_schema_sha256,
            "source_disjoint_schema_sha256": (
                validated_source_disjoint_schema_sha256
            ),
            "reason": "real_frozen_formal_v3_release_unavailable",
        }
    else:
        if release_lock.get("status") != "ready":
            raise QuerySpecializationError(
                "enabled full_model_followup requires release_lock.status=ready"
            )
        release_root = _resolve(
            REPO_ROOT,
            release_lock.get("root"),
            "full_model_followup.release_lock.root",
        )
        expected_release_sha256 = _required_sha256(
            release_lock.get("expected_manifest_sha256"),
            "full_model_followup.release_lock.expected_manifest_sha256",
        )
        expected_consumer_contract_sha256 = _required_sha256(
            release_lock.get("expected_consumer_contract_sha256"),
            "full_model_followup.release_lock.expected_consumer_contract_sha256",
        )
        authenticated_files = _mapping(
            dataset_validation.get("authenticated_file_sha256"),
            "sidecar_dataset_validation.authenticated_file_sha256",
        )
        projector_manifest_sha256 = _required_sha256(
            authenticated_files.get("manifest.json"),
            "sidecar_dataset_validation.authenticated_file_sha256['manifest.json']",
        )
        authenticated_partitions = {
            relative: _required_sha256(
                authenticated_files.get(relative),
                f"sidecar_dataset_validation.authenticated_file_sha256[{relative!r}]",
            )
            for relative in FIXED_SIDECAR_FILES
        }
        try:
            release_validation = load_training_release_lock(
                release_root=release_root,
                dataset_root=_resolve(
                    REPO_ROOT,
                    contract.get("root"),
                    "dataset_contract.root",
                ),
                schema_path=release_schema_path,
                expected_manifest_sha256=expected_release_sha256,
                expected_schema_sha256=release_schema_sha256,
                expected_projector_manifest_sha256=projector_manifest_sha256,
                expected_projector_manifest_schema_sha256=_required_sha256(
                    contract.get("expected_manifest_schema_sha256"),
                    "dataset_contract.expected_manifest_schema_sha256",
                ),
                expected_projector_sidecar_schema_sha256=_required_sha256(
                    contract.get("expected_sidecar_schema_sha256"),
                    "dataset_contract.expected_sidecar_schema_sha256",
                ),
                expected_projector_segment_plan_schema_sha256=_required_sha256(
                    contract.get("expected_segment_plan_schema_sha256"),
                    "dataset_contract.expected_segment_plan_schema_sha256",
                ),
                expected_consumer_contract_sha256=(
                    expected_consumer_contract_sha256
                ),
                repository_root=REPO_ROOT,
                expected_consumer_id=expected_consumer_id,
                expected_consumer_version=expected_consumer_version,
                required_implementation_files=(
                    QUERY_SPECIALIZATION_IMPLEMENTATION_FILES
                ),
                required_launch_entrypoint=(
                    QUERY_SPECIALIZATION_LAUNCH_ENTRYPOINT
                ),
                authenticated_partition_sha256=authenticated_partitions,
            ).as_dict()
        except TrainingReleaseConsumerError as exc:
            raise QuerySpecializationError(str(exc)) from exc
    return {
        "claim_scope": CLAIM_SCOPE,
        "contract_fixture": dataset_summary(records),
        "producer_manifest": {
            "schema_version": producer_manifest["schema_version"],
            "claim_scope": producer_manifest["claim_scope"],
            "provider_requests": producer_manifest["provider_requests"],
            "canonical_gold_written": producer_manifest["canonical_gold_written"],
            "heldout_content_read": producer_manifest["heldout_content_read"],
            "heldout_content_emitted": producer_manifest[
                "heldout_content_emitted"
            ],
            "counts": producer_manifest["counts"],
        },
        "sidecar_dataset_validation": dict(dataset_validation),
        "metrics_schema_sha256": hashlib.sha256(metrics_schema_raw).hexdigest(),
        "role_relevant_block_kinds": {
            role: list(relevant_kinds[role]) for role in ROLES
        },
        "proxy": {
            "profile": profile,
            "target_modules": list(modules),
            "hidden_size": width,
            "rank": rank,
            "alpha": alpha,
            "train_tasks": len(task_ids["train"]),
            "eval_tasks": len(task_ids["eval"]),
            "train_task_ids_sha256": _digest_strings(task_ids["train"]),
            "eval_task_ids_sha256": _digest_strings(task_ids["eval"]),
            "task_overlap": 0,
            "all_roles_per_task": True,
            "foundation_model_loaded": False,
            "strict_json_optimized": False,
        },
        "full_model_followup_enabled": enabled,
        "formal_training_authorized": release_validation[
            "formal_training_authorized"
        ],
        "release_lock_status": release_validation["status"],
        "release_lock_manifest_sha256": release_validation.get(
            "manifest_sha256"
        ),
        "release_lock_validation": release_validation,
    }


def _stable_vector(torch: Any, text: str, width: int) -> Any:
    values: list[float] = []
    counter = 0
    while len(values) < width:
        digest = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    tensor = torch.tensor(values[:width], dtype=torch.float32)
    return tensor / tensor.norm().clamp_min(1e-8)


def _build_proxy_examples(
    torch: Any,
    *,
    task_ids: Mapping[str, Sequence[str]],
    role_kinds: Mapping[str, Sequence[str]],
    width: int,
    task_variation: float,
    distractor_copies: int,
) -> dict[str, tuple[ProxyExample, ...]]:
    """Build shared boards with all role views after task-group splitting."""

    result: dict[str, list[ProxyExample]] = {"train": [], "eval": []}
    shared_context = _stable_vector(torch, "shared-task-context", width)
    for split in ("train", "eval"):
        for task_id in task_ids[split]:
            task_context = _stable_vector(torch, f"task-content:{task_id}", width)
            base_keys: dict[str, Any] = {}
            for kind in BLOCK_KINDS:
                key = _stable_vector(torch, f"kind:{kind}", width)
                key = key + task_variation * _stable_vector(
                    torch, f"task-kind:{task_id}:{kind}", width
                )
                base_keys[kind] = key / key.norm().clamp_min(1e-8)
            for role_index, role in enumerate(ROLES):
                relevant_kinds = set(role_kinds[role])
                pair_id = f"{task_id}:{role}"
                for variant in ("clean", "noisy"):
                    key_rows: list[Any] = []
                    relevant_mask: list[bool] = []
                    distractor_mask: list[bool] = []
                    for kind in BLOCK_KINDS:
                        is_relevant = kind in relevant_kinds
                        if variant == "clean" and not is_relevant:
                            continue
                        copies = 1 if is_relevant else distractor_copies
                        for copy_index in range(copies):
                            key = base_keys[kind]
                            if copy_index:
                                key = key + 0.02 * _stable_vector(
                                    torch,
                                    f"distractor-copy:{task_id}:{kind}:{copy_index}",
                                    width,
                                )
                                key = key / key.norm().clamp_min(1e-8)
                            key_rows.append(key)
                            relevant_mask.append(is_relevant)
                            distractor_mask.append(not is_relevant)
                    keys = torch.stack(key_rows)
                    context = (
                        shared_context
                        + task_variation * task_context
                        + 0.1 * keys.mean(dim=0)
                    )
                    context = context / context.norm().clamp_min(1e-8)
                    result[split].append(
                        ProxyExample(
                            task_id=task_id,
                            split=split,
                            role=role,
                            variant=variant,
                            pair_id=pair_id,
                            role_index=role_index,
                            keys=keys,
                            context=context,
                            relevant=torch.tensor(
                                [relevant_mask], dtype=torch.bool
                            ),
                            distractor=torch.tensor(
                                [distractor_mask], dtype=torch.bool
                            ),
                        )
                    )
    return {key: tuple(value) for key, value in result.items()}


def _execute_probe(
    config: Mapping[str, Any],
    records: Sequence[QueryTrainingRecord],
    *,
    experiment_config_sha256: str,
    contract_fixture_sha256: str,
    metrics_schema_sha256: str,
    runner_sha256: str,
    query_contract_module_sha256: str,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    proxy = _mapping(config["proxy"], "proxy")
    width = int(proxy["hidden_size"])
    rank = int(proxy["rank"])
    alpha = float(proxy["alpha"])
    seed = int(proxy["seed"])
    learning_rate = float(proxy["learning_rate"])
    temperature = float(proxy["attention_temperature"])
    task_variation = float(proxy["task_variation"])
    distractor_copies = int(proxy["distractor_copies"])
    losses_config = _mapping(proxy["losses"], "proxy.losses")
    distractor_weight = float(losses_config["distractor_weight"])
    consistency_weight = float(
        losses_config["clean_noisy_consistency_weight"]
    )
    torch.manual_seed(seed)

    class RoleQueryLoRA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.empty(width, width)
            torch.nn.init.orthogonal_(base)
            self.register_buffer("base_projection", base)
            self.lora_a = torch.nn.Parameter(
                torch.empty(len(ROLES), width, rank)
            )
            self.lora_b = torch.nn.Parameter(
                torch.zeros(len(ROLES), rank, width)
            )
            torch.nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
            self.scale = alpha / rank

        def forward(self, role_index: int, context: Any) -> Any:
            base_query = context @ self.base_projection
            delta = (context @ self.lora_a[role_index]) @ self.lora_b[role_index]
            query = base_query + self.scale * delta
            return query / query.norm().clamp_min(1e-8)

    task_ids = proxy_task_ids(proxy)
    examples = _build_proxy_examples(
        torch,
        task_ids=task_ids,
        role_kinds=_role_relevant_kinds(records),
        width=width,
        task_variation=task_variation,
        distractor_copies=distractor_copies,
    )
    train_task_ids = {example.task_id for example in examples["train"]}
    eval_task_ids = {example.task_id for example in examples["eval"]}
    overlap = train_task_ids & eval_task_ids
    if overlap:
        raise QuerySpecializationError(
            f"proxy task leakage detected: {sorted(overlap)}"
        )
    for split in ("train", "eval"):
        for task_id in {example.task_id for example in examples[split]}:
            task_examples = [
                example for example in examples[split] if example.task_id == task_id
            ]
            if {example.role for example in task_examples} != set(ROLES):
                raise QuerySpecializationError(
                    f"proxy task {task_id!r} does not contain all role views"
                )
            for role in ROLES:
                variants = {
                    example.variant
                    for example in task_examples
                    if example.role == role
                }
                if variants != {"clean", "noisy"}:
                    raise QuerySpecializationError(
                        f"proxy task {task_id!r}/{role} lacks clean/noisy views"
                    )

    model = RoleQueryLoRA()
    frozen_parameters = int(model.base_projection.numel())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )

    def metrics(evaluation_examples: Sequence[ProxyExample]) -> dict[str, float]:
        top1 = 0
        noisy_count = 0
        relevant_masses: list[float] = []
        relevant_min_masses: list[float] = []
        distractor_masses: list[float] = []
        pair_cosines: list[float] = []
        correct_role_gaps: list[float] = []
        inter_role_cosines: list[float] = []
        by_pair: dict[str, dict[str, ProxyExample]] = {}
        with torch.no_grad():
            for example in evaluation_examples:
                by_pair.setdefault(example.pair_id, {})[example.variant] = example
                if example.variant != "noisy":
                    continue
                noisy_count += 1
                query = model(example.role_index, example.context)
                probabilities = torch.softmax(
                    (example.keys @ query) / temperature, dim=-1
                )
                relevant = example.relevant[0]
                distractor = example.distractor[0]
                top1 += int(bool(relevant[int(probabilities.argmax())]))
                relevant_masses.append(float(probabilities[relevant].sum()))
                relevant_min_masses.append(float(probabilities[relevant].min()))
                distractor_masses.append(float(probabilities[distractor].sum()))
                correct_mass = probabilities[relevant].sum()
                wrong_masses = []
                role_queries = []
                for role_index in range(len(ROLES)):
                    role_query = model(role_index, example.context)
                    role_queries.append(role_query)
                    if role_index != example.role_index:
                        wrong_probabilities = torch.softmax(
                            (example.keys @ role_query) / temperature, dim=-1
                        )
                        wrong_masses.append(wrong_probabilities[relevant].sum())
                correct_role_gaps.append(
                    float(correct_mass - torch.stack(wrong_masses).max())
                )
                for left in range(len(role_queries)):
                    for right in range(left + 1, len(role_queries)):
                        inter_role_cosines.append(
                            float(
                                torch.nn.functional.cosine_similarity(
                                    role_queries[left], role_queries[right], dim=0
                                )
                            )
                        )
            for pair in by_pair.values():
                clean = pair["clean"]
                noisy = pair["noisy"]
                clean_query = model(clean.role_index, clean.context)
                noisy_query = model(noisy.role_index, noisy.context)
                pair_cosines.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            clean_query, noisy_query, dim=0
                        )
                    )
                )
        return {
            "top1_relevant_rate": top1 / max(1, noisy_count),
            "mean_relevant_mass": sum(relevant_masses)
            / max(1, len(relevant_masses)),
            "mean_min_relevant_block_mass": sum(relevant_min_masses)
            / max(1, len(relevant_min_masses)),
            "mean_distractor_mass": sum(distractor_masses)
            / max(1, len(distractor_masses)),
            "mean_clean_noisy_query_cosine": sum(pair_cosines)
            / max(1, len(pair_cosines)),
            "mean_correct_role_mass_gap": sum(correct_role_gaps)
            / max(1, len(correct_role_gaps)),
            "mean_inter_role_query_cosine_diagnostic": sum(inter_role_cosines)
            / max(1, len(inter_role_cosines)),
        }

    # Both snapshots are evaluation-only; gradient updates consume train only.
    before = metrics(examples["eval"])
    epochs = int(proxy["epochs"])
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        attention_losses: list[Any] = []
        pair_queries: dict[str, list[Any]] = {}
        for example in examples["train"]:
            query = model(example.role_index, example.context)
            pair_queries.setdefault(example.pair_id, []).append(query)
            probabilities = torch.softmax(
                (example.keys @ query) / temperature, dim=-1
            )
            loss, _ = block_attention_auxiliary_loss(
                probabilities.unsqueeze(0),
                example.relevant,
                example.distractor,
                distractor_weight=distractor_weight,
            )
            attention_losses.append(loss)
        consistency = torch.stack(
            [
                1.0
                - torch.nn.functional.cosine_similarity(pair[0], pair[1], dim=0)
                for pair in pair_queries.values()
            ]
        ).mean()
        total = (
            torch.stack(attention_losses).mean()
            + consistency_weight * consistency
        )
        total.backward()
        optimizer.step()
        final_loss = float(total.detach())

    after = metrics(examples["eval"])
    thresholds = _mapping(proxy["signal_thresholds"], "proxy.signal_thresholds")
    checks = {
        "eval_top1_relevant": after["top1_relevant_rate"]
        >= float(thresholds["minimum_top1_relevant_rate"]),
        "eval_relevant_mass_gain": (
            after["mean_relevant_mass"] - before["mean_relevant_mass"]
        )
        >= float(thresholds["minimum_relevant_mass_gain"]),
        "eval_min_relevant_block_mass": after["mean_min_relevant_block_mass"]
        >= float(thresholds["minimum_mean_min_relevant_block_mass"]),
        "eval_distractor_mass": after["mean_distractor_mass"]
        <= float(thresholds["maximum_distractor_mass"]),
        "eval_correct_role_mass_gap": after["mean_correct_role_mass_gap"]
        >= float(thresholds["minimum_correct_role_mass_gap"]),
        "eval_clean_noisy_consistency": after[
            "mean_clean_noisy_query_cosine"
        ]
        >= float(thresholds["minimum_clean_noisy_query_cosine"]),
    }
    result = {
        "schema_version": "anchor.query-specialization-proxy-metrics.v1",
        "probe_kind": "split_safe_tiny_frozen_key_role_query_lora",
        "claim_scope": CLAIM_SCOPE,
        "metrics_scope": "unseen_eval_task_groups_only",
        "experiment_config_sha256": experiment_config_sha256,
        "contract_fixture_sha256": contract_fixture_sha256,
        "metrics_schema_sha256": metrics_schema_sha256,
        "runner_sha256": runner_sha256,
        "query_contract_module_sha256": query_contract_module_sha256,
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "signal_thresholds": {
            key: float(value) for key, value in sorted(thresholds.items())
        },
        "foundation_model_loaded": False,
        "strict_json_optimized": False,
        "contract_fixture_used_for_gradient_training": False,
        "base_projection_frozen": True,
        "frozen_parameters": frozen_parameters,
        "trainable_parameters": trainable_parameters,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "attention_temperature": temperature,
        "task_variation": task_variation,
        "distractor_copies": distractor_copies,
        "final_loss": final_loss,
        "split_audit": {
            "group_key": "synthetic_task_id",
            "train_tasks": len(train_task_ids),
            "eval_tasks": len(eval_task_ids),
            "train_examples": len(examples["train"]),
            "eval_examples": len(examples["eval"]),
            "task_overlap": len(overlap),
            "train_task_ids_sha256": _digest_strings(tuple(train_task_ids)),
            "eval_task_ids_sha256": _digest_strings(tuple(eval_task_ids)),
            "all_five_roles_per_task": True,
            "clean_noisy_after_split": True,
        },
        "before_eval": before,
        "after_eval": after,
        "eval_delta": {
            "relevant_mass": after["mean_relevant_mass"]
            - before["mean_relevant_mass"],
            "distractor_mass": after["mean_distractor_mass"]
            - before["mean_distractor_mass"],
        },
        "signal_checks": checks,
        "proxy_signal_passed": all(checks.values()),
        "non_claims": [
            "strict_json_instruction_following",
            "foundation_model_qlora_quality",
            "causal_attention_interpretation",
            "shared_kv_correctness_or_speedup",
        ],
    }
    _validate_proxy_result_contract(result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config_bytes = config_path.read_bytes()
        config = _load_config_snapshot(config_bytes, str(config_path))
        _, records, producer_manifest, dataset_validation = _load_contract_dataset(
            config
        )
        plan = _plan(
            config,
            records,
            producer_manifest=producer_manifest,
            dataset_validation=dataset_validation,
        )
        response: dict[str, Any] = {"ok": True, "mode": "dry_run", "plan": plan}
        if args.execute:
            proxy = _mapping(config["proxy"], "proxy")
            result = _execute_probe(
                config,
                records,
                experiment_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                contract_fixture_sha256=dataset_validation[
                    "dataset_contract_sha256"
                ],
                metrics_schema_sha256=plan["metrics_schema_sha256"],
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                query_contract_module_sha256=hashlib.sha256(
                    (SRC_ROOT / "anchor_mvp/research/query_specialization.py").read_bytes()
                ).hexdigest(),
            )
            output = (
                Path(args.output).expanduser().resolve()
                if args.output
                else _resolve(REPO_ROOT, proxy["output"], "proxy.output")
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            response.update(
                {
                    "mode": "execute",
                    "result": result,
                    "output": str(output),
                }
            )
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    except (
        OSError,
        KeyError,
        json.JSONDecodeError,
        QuerySpecializationError,
        TrainingReleaseConsumerError,
    ) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
