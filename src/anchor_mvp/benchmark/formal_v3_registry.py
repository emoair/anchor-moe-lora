"""Immutable formal-v3 A--F evaluation registry finalizer.

This module is deliberately independent from the formal-v1/v2 registry.  It
never opens held-out cases or fixtures.  It binds only the external held-out
metadata hashes, the immutable formal-v3 training snapshot, snapshot-sized
training schedules, frozen calibration allocations and completed adapter
artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..training.config import load_training_config, select_adapter
from ..training.manifest import config_fingerprint, sha256_file


CONTROL_SCHEMA = "anchor.formal-v3-af-finalizer-config.v1"
REGISTRY_SCHEMA = "anchor.formal-v3-af-registry.v1"
BENCHMARK_SCHEMA = "anchor.formal-v3-af-benchmark.v1"
BUNDLE_SCHEMA = "anchor.formal-v3-af-registry-bundle.v1"
SNAPSHOT_SCHEMA = "anchor.training-snapshot.v2"
ALLOCATION_SCHEMA = "anchor.lora-allocation.v1"
STAGES = ("planner", "tool_policy", "frontend", "review", "security")
TRAINING_NAMES = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "frontend": "frontend_gen",
    "review": "frontend_review",
    "security": "security_gate",
}
ARMS = ("A", "B", "C", "D", "E", "F")
REGISTRY_CONTRACT_KEYS = (
    "formal_version",
    "version_id",
    "control_path",
    "control_sha256",
    "snapshot",
    "base",
    "heldout",
    "protocol",
    "allocations",
    "arms",
    "normalization",
    "output_namespace",
    "resume_scope",
    "formal_v2_inputs_allowed",
)
PARAMETERS_PER_RANK = 649_216
B_RANK = 16
B_PARAMETERS = PARAMETERS_PER_RANK * B_RANK
D_RANKS = {
    "planner": 3,
    "tool_policy": 3,
    "frontend_gen": 4,
    "frontend_review": 3,
    "security_gate": 3,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")
_RUNTIME_ID = re.compile(r"[0-9a-f]{32}")
_FORBIDDEN_V2 = re.compile(r"formal[-_]v2", re.IGNORECASE)
_NEGATIVE_V2_CAPABILITIES = {
    "formal_v2_artifacts_allowed",
    "formal_v2_inputs_allowed",
}


class FormalV3RegistryError(ValueError):
    """A formal-v3 evaluation source is missing, stale or inconsistent."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FormalV3RegistryError(
            "invalid_binding", f"{label} must be a lowercase SHA-256"
        )
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FormalV3RegistryError(
            "training_incomplete", f"{label} is missing: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalV3RegistryError(
            "invalid_metadata", f"{label} is invalid: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FormalV3RegistryError(
            "invalid_metadata", f"{label} must contain one JSON object"
        )
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalV3RegistryError(
            "invalid_metadata", f"{label} must be an object"
        )
    return value


def _inside(root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise FormalV3RegistryError("invalid_path", f"{label} path is missing")
    path = Path(raw)
    candidate = (path if path.is_absolute() else root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FormalV3RegistryError(
            "invalid_path", f"{label} escapes the project root"
        ) from exc
    return candidate


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _forbid_formal_v2(value: object, label: str) -> None:
    def walk(item: object) -> None:
        if isinstance(item, str):
            if _FORBIDDEN_V2.search(item):
                raise FormalV3RegistryError(
                    "formal_v2_input_forbidden",
                    f"{label} contains a forbidden formal-v2 reference",
                )
        elif isinstance(item, Mapping):
            for key, child in item.items():
                if isinstance(key, str) and _FORBIDDEN_V2.search(key):
                    if key in _NEGATIVE_V2_CAPABILITIES and child is False:
                        continue
                    raise FormalV3RegistryError(
                        "formal_v2_input_forbidden",
                        f"{label} contains a forbidden formal-v2 capability",
                    )
                walk(key)
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                walk(child)

    walk(value)


def _verify_sidecar(path: Path, sidecar: Path, label: str) -> str:
    if not path.is_file() or not sidecar.is_file():
        raise FormalV3RegistryError(
            "training_incomplete", f"{label} or its SHA-256 sidecar is missing"
        )
    try:
        tokens = sidecar.read_text(encoding="ascii").split()
    except OSError as exc:
        raise FormalV3RegistryError(
            "invalid_metadata", f"cannot read {label} sidecar"
        ) from exc
    observed = sha256_file(path)
    if not tokens or tokens[0].lower() != observed:
        raise FormalV3RegistryError(
            "hash_mismatch", f"{label} SHA-256 sidecar mismatch"
        )
    return observed


def _sidecar_record(root: Path, sidecar: Path) -> dict[str, Any]:
    """Bind the sidecar itself, not only the digest it declared once."""

    return {
        "path": _relative(root, sidecar),
        "bytes": sidecar.stat().st_size,
        "sha256": sha256_file(sidecar),
    }


def _weight_set_digest(weights: Sequence[Mapping[str, Any]]) -> str:
    return _digest(
        [
            {
                "path": item.get("path"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for item in weights
        ]
    )


def _base_binding(root: Path, control: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, control.get("base_manifest"), "base manifest")
    value = _load_json(path, "base manifest")
    if value.get("schema_version") != "anchor.bnb-nf4-export.v1":
        raise FormalV3RegistryError(
            "base_mismatch", "formal-v3 requires the audited NF4 base manifest"
        )
    quantization = _mapping(value.get("quantization"), "base quantization")
    if not (
        quantization.get("type") == "nf4"
        and quantization.get("double_quant") is True
        and quantization.get("compute_dtype") == "bfloat16"
    ):
        raise FormalV3RegistryError(
            "base_mismatch", "formal-v3 base quantization contract changed"
        )
    raw_weights = value.get("weights")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise FormalV3RegistryError("base_mismatch", "base has no weight inventory")
    weights: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_weights):
        item = _mapping(raw, f"base weight {index}")
        weight_path = _inside(path.parent, item.get("path"), f"base weight {index}")
        expected_sha = _require_sha(item.get("sha256"), f"base weight {index}")
        expected_bytes = item.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or not weight_path.is_file()
            or weight_path.stat().st_size != expected_bytes
        ):
            raise FormalV3RegistryError(
                "base_mismatch", f"base weight {index} is missing or has the wrong size"
            )
        if sha256_file(weight_path) != expected_sha:
            raise FormalV3RegistryError(
                "base_mismatch", f"base weight {index} digest changed"
            )
        weights.append(
            {
                "path": _relative(root, weight_path),
                "bytes": expected_bytes,
                "sha256": expected_sha,
            }
        )
    return {
        "format": "transformers-bitsandbytes-nf4",
        "base_contract_id": str(control.get("base_contract_id", "")),
        "manifest_path": _relative(root, path),
        "manifest_sha256": sha256_file(path),
        "source_weight_sha256": _require_sha(
            value.get("source_weight_sha256"), "base source weight"
        ),
        "weight_set_sha256": _weight_set_digest(raw_weights),
        "model_dir": _relative(root, path.parent),
        "weights": weights,
    }


def _snapshot_binding(root: Path, control: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, control.get("snapshot_manifest"), "snapshot manifest")
    sidecar = _inside(
        root,
        control.get("snapshot_sidecar", f"{control.get('snapshot_manifest', '')}.sha256"),
        "snapshot sidecar",
    )
    manifest_sha = _verify_sidecar(path, sidecar, "formal-v3 snapshot")
    value = _load_json(path, "formal-v3 snapshot")
    if value.get("schema_version") != SNAPSHOT_SCHEMA:
        raise FormalV3RegistryError(
            "snapshot_mismatch", "formal-v3 requires anchor.training-snapshot.v2"
        )
    snapshot_sha = _require_sha(value.get("snapshot_sha256"), "snapshot identifier")
    split = _mapping(value.get("split_contract"), "snapshot split contract")
    partitions = _mapping(split.get("partitions"), "snapshot partitions")
    heldout = _mapping(partitions.get("heldout"), "snapshot heldout partition")
    calibration = _mapping(
        partitions.get("calibration"), "snapshot calibration partition"
    )
    if not (
        split.get("schema_version") == "anchor.formal-v3-gold-splits.v1"
        and split.get("pairwise_disjoint") is True
        and split.get("heldout_content_read") is False
        and split.get("heldout_content_emitted") is False
        and heldout.get("role") == "evaluation_only_hash_metadata"
        and heldout.get("content_present") is False
        and heldout.get("content_read") is False
        and heldout.get("content_emitted") is False
    ):
        raise FormalV3RegistryError(
            "snapshot_mismatch", "snapshot does not preserve heldout hash-only isolation"
        )
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "manifest_path": _relative(root, path),
        "manifest_sha256": manifest_sha,
        "sidecar": _sidecar_record(root, sidecar),
        "snapshot_sha256": snapshot_sha,
        "calibration_snapshot_sha256": _require_sha(
            calibration.get("snapshot_sha256"), "calibration snapshot"
        ),
        "heldout_ids_sha256": _require_sha(
            heldout.get("ids_sha256"), "heldout ID set"
        ),
        "heldout_manifest_sha256": _require_sha(
            heldout.get("manifest_sha256"), "heldout manifest"
        ),
        "leakage_audit_sha256": _require_sha(
            split.get("leakage_audit_sha256"), "heldout leakage audit"
        ),
        "heldout_content_read": False,
    }


def _heldout_binding(
    root: Path,
    control: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(control.get("heldout_metadata"), "heldout_metadata")
    if "cases_path" in value or "fixtures_root" in value:
        raise FormalV3RegistryError(
            "heldout_body_forbidden",
            "finalizer control must contain heldout metadata only, never case/fixture paths",
        )
    manifest = _inside(root, value.get("manifest_path"), "heldout manifest metadata")
    leak = _inside(root, value.get("leak_audit_path"), "heldout leak metadata")
    manifest_sha = sha256_file(manifest) if manifest.is_file() else ""
    leak_sha = sha256_file(leak) if leak.is_file() else ""
    if manifest_sha != snapshot["heldout_manifest_sha256"]:
        raise FormalV3RegistryError(
            "heldout_metadata_mismatch",
            "external heldout manifest metadata differs from the formal-v3 snapshot",
        )
    if leak_sha != snapshot["leakage_audit_sha256"]:
        raise FormalV3RegistryError(
            "heldout_metadata_mismatch",
            "external heldout leakage metadata differs from the formal-v3 snapshot",
        )
    return {
        "manifest_path": _relative(root, manifest),
        "manifest_sha256": manifest_sha,
        "leak_audit_path": _relative(root, leak),
        "leak_audit_sha256": leak_sha,
        "ids_sha256": snapshot["heldout_ids_sha256"],
        "metadata_only": True,
        "case_content_read": False,
    }


def _schedule_binding(
    root: Path,
    schedule_root: Path,
    snapshot: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    path = schedule_root / str(snapshot["snapshot_sha256"]) / f"{arm}.json"
    path = _inside(root, str(path), f"arm {arm} schedule")
    sidecar = Path(f"{path}.sha256")
    schedule_sha = _verify_sidecar(path, sidecar, f"arm {arm} training schedule")
    value = _load_json(path, f"arm {arm} training schedule")
    _forbid_formal_v2(value, f"arm {arm} schedule")
    training = _mapping(value.get("training"), f"arm {arm} schedule training")
    exposure = _mapping(
        training.get("resolved_exposure"), f"arm {arm} resolved exposure"
    )
    evaluation = _mapping(
        exposure.get("evaluation_contract"), f"arm {arm} evaluation contract"
    )
    if not (
        exposure.get("arm") == arm
        and exposure.get("dataset_snapshot_sha256") == snapshot["snapshot_sha256"]
        and exposure.get("snapshot_manifest_sha256") == snapshot["manifest_sha256"]
        and exposure.get("heldout_content_read") is False
        and evaluation.get("schema_version") == "anchor.formal-v3-af-evaluation.v1"
        and evaluation.get("heldout_ids_sha256") == snapshot["heldout_ids_sha256"]
        and evaluation.get("heldout_manifest_sha256")
        == snapshot["heldout_manifest_sha256"]
        and evaluation.get("leakage_audit_sha256")
        == snapshot["leakage_audit_sha256"]
        and evaluation.get("normalization") == "A_equals_100"
        and evaluation.get("formal_v2_artifacts_allowed") is False
    ):
        raise FormalV3RegistryError(
            "schedule_mismatch", f"arm {arm} schedule is not bound to formal-v3"
        )
    max_steps = training.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise FormalV3RegistryError(
            "schedule_mismatch", f"arm {arm} schedule has no positive max_steps"
        )
    return {
        "path": _relative(root, path),
        "sha256": schedule_sha,
        "sidecar": _sidecar_record(root, sidecar),
        "max_steps_per_adapter_job": max_steps,
        "resolved_exposure_sha256": _digest(exposure),
        "evaluation_contract_sha256": _digest(evaluation),
    }


def _allocation_binding(
    root: Path,
    path: Path,
    snapshot: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    sidecar = Path(f"{path}.sha256")
    manifest_sha = _verify_sidecar(path, sidecar, f"arm {arm} calibration allocation")
    value = _load_json(path, f"arm {arm} calibration allocation")
    _forbid_formal_v2(value, f"arm {arm} allocation")
    selection = _mapping(
        value.get("selection_algorithm"), f"arm {arm} selection algorithm"
    )
    attempted = value.get("attempted_allocations")
    if not (
        value.get("schema_version") == ALLOCATION_SCHEMA
        and value.get("arm") == arm
        and value.get("dataset_snapshot_sha256") == snapshot["snapshot_sha256"]
        and value.get("calibration_snapshot_sha256")
        == snapshot["calibration_snapshot_sha256"]
        and value.get("allocation_frozen_before_heldout") is True
        and value.get("heldout_opened") is False
        and value.get("heldout_opened_at") is None
        and value.get("heldout_access") == "forbidden_until_allocation_frozen"
        and value.get("selection_status") == "calibration_selected_frozen"
        and value.get("calibration_metrics_available") is True
        and isinstance(value.get("calibration_record_count"), int)
        and not isinstance(value.get("calibration_record_count"), bool)
        and value["calibration_record_count"] > 0
        and isinstance(selection.get("algorithm_id"), str)
        and bool(selection["algorithm_id"])
        and selection.get("calibration_performance_used") is True
        and value.get("target_modules") == ["q_proj", "v_proj"]
        and isinstance(attempted, list)
        and bool(attempted)
    ):
        raise FormalV3RegistryError(
            "allocation_mismatch",
            f"arm {arm} allocation is not frozen to this calibration split",
        )
    ranks = _mapping(value.get("selected_ranks"), f"arm {arm} selected ranks")
    if set(ranks) != set(TRAINING_NAMES.values()):
        raise FormalV3RegistryError(
            "allocation_mismatch", f"arm {arm} must allocate all five experts"
        )
    normalized: dict[str, int] = {}
    for expert in TRAINING_NAMES.values():
        rank = ranks[expert]
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 16:
            raise FormalV3RegistryError(
                "allocation_mismatch", f"arm {arm} rank for {expert} is outside 1..16"
            )
        normalized[expert] = rank
    if len(set(normalized.values())) == 1:
        raise FormalV3RegistryError(
            "allocation_mismatch", f"arm {arm} adaptive allocation must be non-uniform"
        )
    rank_sum = sum(normalized.values())
    parameters_per_rank = value.get("parameters_per_rank")
    parameters = value.get("materialized_trainable_parameters")
    if (
        parameters_per_rank != PARAMETERS_PER_RANK
        or value.get("rank_sum") != rank_sum
        or parameters != rank_sum * PARAMETERS_PER_RANK
    ):
        raise FormalV3RegistryError(
            "allocation_mismatch", f"arm {arm} materialized budget is inconsistent"
        )
    if arm == "F" and (rank_sum != B_RANK or parameters != B_PARAMETERS):
        raise FormalV3RegistryError(
            "allocation_mismatch", "arm F must exactly match B's rank and parameter budget"
        )
    selected_signature = tuple(normalized[name] for name in TRAINING_NAMES.values())
    attempted_signatures: set[tuple[int, ...]] = set()
    for index, raw_attempt in enumerate(attempted):
        attempt = _mapping(raw_attempt, f"arm {arm} attempted allocation {index}")
        attempt_ranks = _mapping(
            attempt.get("selected_ranks"),
            f"arm {arm} attempted allocation {index} ranks",
        )
        if set(attempt_ranks) != set(TRAINING_NAMES.values()):
            raise FormalV3RegistryError(
                "allocation_mismatch",
                f"arm {arm} attempted allocation {index} lacks the five experts",
            )
        signature: list[int] = []
        for expert in TRAINING_NAMES.values():
            attempt_rank = attempt_ranks[expert]
            if (
                isinstance(attempt_rank, bool)
                or not isinstance(attempt_rank, int)
                or not 1 <= attempt_rank <= 16
            ):
                raise FormalV3RegistryError(
                    "allocation_mismatch",
                    f"arm {arm} attempted allocation {index} has an invalid rank",
                )
            signature.append(attempt_rank)
        frozen_signature = tuple(signature)
        metrics = attempt.get("calibration_metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise FormalV3RegistryError(
                "allocation_mismatch",
                f"arm {arm} attempted allocation {index} has no calibration metrics",
            )
        if frozen_signature in attempted_signatures:
            raise FormalV3RegistryError(
                "allocation_mismatch",
                f"arm {arm} attempted allocations contain a duplicate rank plan",
            )
        attempted_signatures.add(frozen_signature)
    if selected_signature not in attempted_signatures:
        raise FormalV3RegistryError(
            "allocation_mismatch",
            f"arm {arm} selected ranks were not measured on calibration",
        )
    try:
        created = datetime.fromisoformat(str(value.get("created_at")))
        frozen = datetime.fromisoformat(str(value.get("allocation_frozen_at")))
    except ValueError as exc:
        raise FormalV3RegistryError(
            "allocation_mismatch", f"arm {arm} freeze timestamps are invalid"
        ) from exc
    if created.tzinfo is None or frozen.tzinfo is None:
        raise FormalV3RegistryError(
            "allocation_mismatch", f"arm {arm} freeze timestamps need a timezone"
        )
    if frozen < created:
        raise FormalV3RegistryError(
            "allocation_mismatch", f"arm {arm} was frozen before it was created"
        )
    return {
        "path": _relative(root, path),
        "sha256": manifest_sha,
        "sidecar": _sidecar_record(root, sidecar),
        "mechanism_id": str(value.get("mechanism_id", "")),
        "selection_algorithm_id": str(selection["algorithm_id"]),
        "selection_status": "calibration_selected_frozen",
        "calibration_metrics_available": True,
        "calibration_record_count": int(value["calibration_record_count"]),
        "selected_ranks": normalized,
        "rank_sum": rank_sum,
        "materialized_trainable_parameters": parameters,
        "calibration_snapshot_sha256": snapshot["calibration_snapshot_sha256"],
    }


def _expected_ranks(
    allocations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    return {
        "A": {},
        "B": {"mixed_all": 16},
        "C": {expert: 16 for expert in TRAINING_NAMES.values()},
        "D": dict(D_RANKS),
        "E": dict(allocations["E"]["selected_ranks"]),
        "F": dict(allocations["F"]["selected_ranks"]),
    }


def _adapter_binding(
    root: Path,
    snapshot: Mapping[str, Any],
    schedule: Mapping[str, Any],
    arm: str,
    adapter_name: str,
    rank: int,
    artifact_root: Path,
) -> dict[str, Any]:
    artifact_name = f"{adapter_name}-r{rank}"
    adapter_dir = artifact_root / "adapters" / artifact_name
    manifest_path = artifact_root / "manifests" / f"{artifact_name}.execute.json"
    progress_path = artifact_root / "adapters" / f"{artifact_name}.progress" / "status.json"
    required = {
        "adapter_config": adapter_dir / "adapter_config.json",
        "checkpoint_metadata": adapter_dir / "checkpoint_metadata.json",
        "execute_manifest": manifest_path,
        "progress_status": progress_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    weights = [
        path
        for path in (
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        )
        if path.is_file()
    ]
    if missing or len(weights) != 1 or weights[0].stat().st_size <= 0:
        raise FormalV3RegistryError(
            "training_incomplete",
            f"arm {arm} adapter {artifact_name} is incomplete",
            details={"missing": missing, "weight_file_count": len(weights)},
        )
    adapter_config = _load_json(required["adapter_config"], "adapter config")
    metadata = _load_json(required["checkpoint_metadata"], "checkpoint metadata")
    execute = _load_json(required["execute_manifest"], "execute manifest")
    progress = _load_json(required["progress_status"], "progress status")
    _forbid_formal_v2(execute, f"arm {arm} execute manifest")
    targets = sorted(adapter_config.get("target_modules", []))
    config_path = _inside(root, execute.get("config_path"), "execute config path")
    expected_schedule = _inside(root, schedule["path"], f"arm {arm} schedule path")
    if config_path != expected_schedule:
        raise FormalV3RegistryError(
            "checkpoint_mismatch", f"arm {arm} adapter used another training schedule"
        )
    selected = select_adapter(
        load_training_config(expected_schedule), adapter_name, rank
    )
    expected_config_sha = config_fingerprint(selected)
    expected_steps = schedule["max_steps_per_adapter_job"]
    expected_parameters = PARAMETERS_PER_RANK * rank
    if not (
        adapter_config.get("r") == rank
        and adapter_config.get("lora_alpha") == 2 * rank
        and targets == ["q_proj", "v_proj"]
        and execute.get("mode") == "execute"
        and execute.get("stage") == "train"
        and execute.get("run_name") == artifact_name
        and execute.get("adapter_name") == adapter_name
        and execute.get("config_sha256") == expected_config_sha
        and _mapping(execute.get("preflight"), "execute preflight").get("passed")
        is True
        and _mapping(execute["preflight"].get("dataset_snapshot_manifest"), "snapshot preflight").get("passed")
        is True
        and execute["preflight"].get("dataset_snapshot_sha256")
        == snapshot["snapshot_sha256"]
        and metadata.get("run_name") == artifact_name
        and metadata.get("adapter_name") == adapter_name
        and metadata.get("config_sha256") == expected_config_sha
        and metadata.get("global_step") == expected_steps
        and metadata.get("trainable_parameters") == expected_parameters
        and metadata.get("artifact_type") == "peft_adapter"
        and metadata.get("merge_status") == "unmerged"
        and progress.get("state") == "completed"
        and progress.get("step") == expected_steps
        and isinstance(progress.get("run_id"), str)
        and _RUNTIME_ID.fullmatch(progress["run_id"])
    ):
        raise FormalV3RegistryError(
            "checkpoint_mismatch",
            f"arm {arm} adapter {artifact_name} metadata is not a completed formal-v3 job",
        )
    file_records: dict[str, dict[str, Any]] = {}
    for label, path in {**required, "adapter_model": weights[0]}.items():
        file_records[label] = {
            "path": _relative(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "artifact_name": artifact_name,
        "adapter_name": adapter_name,
        "rank": rank,
        "trainable_parameters": expected_parameters,
        "global_step": expected_steps,
        "runtime_run_id": progress["run_id"],
        "config_sha256": expected_config_sha,
        "adapter_dir": _relative(root, adapter_dir),
        "files": file_records,
        "adapter_sha256": _digest(file_records),
    }


def _protocol(control: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(_mapping(control.get("protocol"), "protocol"))
    if tuple(value.get("stages", ())) != STAGES:
        raise FormalV3RegistryError(
            "protocol_mismatch", "formal-v3 requires the five ordered MVP stages"
        )
    sampling = _mapping(value.get("sampling"), "protocol sampling")
    if not (
        value.get("review_protocol") in {"verdict_v2", "repair_code_v1"}
        and isinstance(value.get("max_tokens_per_call"), int)
        and not isinstance(value.get("max_tokens_per_call"), bool)
        and value["max_tokens_per_call"] > 0
        and sampling.get("temperature") == 0.0
        and sampling.get("top_p") == 1.0
    ):
        raise FormalV3RegistryError(
            "protocol_mismatch", "formal-v3 protocol/token/sampling contract is invalid"
        )
    return value


def build_registry(
    control_path: str | Path,
    project_root: str | Path,
    *,
    version_id: str,
) -> dict[str, Any]:
    """Build and deeply verify a registry value without writing it."""

    root = Path(project_root).resolve()
    control_path = _inside(root, str(Path(control_path).resolve()), "finalizer control")
    control = _load_json(control_path, "formal-v3 finalizer control")
    if control.get("schema_version") != CONTROL_SCHEMA:
        raise FormalV3RegistryError("invalid_control", "unsupported finalizer control")
    if not _VERSION.fullmatch(version_id) or _FORBIDDEN_V2.search(version_id):
        raise FormalV3RegistryError(
            "invalid_version", "version_id must be safe and must not name formal-v2"
        )
    _forbid_formal_v2(control, "formal-v3 finalizer control")
    snapshot = _snapshot_binding(root, control)
    heldout = _heldout_binding(root, control, snapshot)
    protocol = _protocol(control)
    schedule_root = _inside(root, control.get("schedule_root"), "schedule root")
    schedules = {
        arm: _schedule_binding(root, schedule_root, snapshot, arm)
        for arm in ARMS[1:]
    }
    if len({item["evaluation_contract_sha256"] for item in schedules.values()}) != 1:
        raise FormalV3RegistryError(
            "schedule_mismatch", "B--F schedules do not share one evaluation contract"
        )
    allocation_paths = _mapping(control.get("allocation_manifests"), "allocation manifests")
    if set(allocation_paths) != {"E", "F"}:
        raise FormalV3RegistryError(
            "invalid_control", "allocation_manifests must contain exactly E and F"
        )
    allocations = {
        arm: _allocation_binding(
            root,
            _inside(root, allocation_paths[arm], f"arm {arm} allocation"),
            snapshot,
            arm,
        )
        for arm in ("E", "F")
    }
    if allocations["E"]["mechanism_id"] != allocations["F"]["mechanism_id"]:
        raise FormalV3RegistryError(
            "allocation_mismatch", "E and F must use the same frozen mechanism"
        )
    if (
        allocations["E"]["selection_algorithm_id"]
        != allocations["F"]["selection_algorithm_id"]
    ):
        raise FormalV3RegistryError(
            "allocation_mismatch", "E and F must use the same calibration algorithm"
        )
    ranks = _expected_ranks(allocations)
    artifact_roots = _mapping(control.get("artifact_roots"), "artifact roots")
    if set(artifact_roots) != set(ARMS[1:]):
        raise FormalV3RegistryError(
            "invalid_control", "artifact_roots must contain exactly B through F"
        )
    base = _base_binding(root, control)
    arms: dict[str, Any] = {
        "A": {
            "topology": "frozen_q4_base_no_adapter",
            "adapters": [],
            "rank_total": 0,
            "trainable_parameter_total": 0,
        }
    }
    for arm in ARMS[1:]:
        root_path = _inside(root, artifact_roots[arm], f"arm {arm} artifact root")
        records = [
            _adapter_binding(
                root,
                snapshot,
                schedules[arm],
                arm,
                adapter_name,
                rank,
                root_path,
            )
            for adapter_name, rank in ranks[arm].items()
        ]
        arms[arm] = {
            "topology": (
                "single_mixed_adapter_reused_five_stages"
                if arm == "B"
                else "five_stage_serial_runtime_lora_hot_swap"
            ),
            "schedule": schedules[arm],
            "adapters": records,
            "ranks": ranks[arm],
            "rank_total": sum(ranks[arm].values()),
            "trainable_parameter_total": sum(
                item["trainable_parameters"] for item in records
            ),
        }
    if arms["B"]["trainable_parameter_total"] != B_PARAMETERS:
        raise FormalV3RegistryError("budget_mismatch", "B is not the rank-16 budget")
    if arms["D"]["trainable_parameter_total"] != B_PARAMETERS:
        raise FormalV3RegistryError("budget_mismatch", "D no longer matches B")
    if arms["F"]["trainable_parameter_total"] != B_PARAMETERS:
        raise FormalV3RegistryError("budget_mismatch", "F no longer matches B")
    contract = {
        "formal_version": "formal-v3",
        "version_id": version_id,
        "control_path": _relative(root, control_path),
        "control_sha256": sha256_file(control_path),
        "snapshot": snapshot,
        "base": base,
        "heldout": heldout,
        "protocol": protocol,
        "allocations": allocations,
        "arms": arms,
        "normalization": {"baseline": "A", "index": 100},
        "output_namespace": f"runs/formal-v3/evaluation/{version_id}",
        "resume_scope": "same_version_same_registry_same_heldout_only",
        "formal_v2_inputs_allowed": False,
    }
    return {
        "schema_version": REGISTRY_SCHEMA,
        "created_at": _now(),
        **contract,
        "contract_sha256": _digest(contract),
        "immutable": True,
        "write_policy": "exclusive_bundle_create_no_overwrite",
    }


def _model_id(version_id: str, arm: str, stage: str, artifact: str | None) -> str:
    if arm == "A":
        return f"fmv3-{version_id}-base-q4"
    if arm == "B":
        return f"fmv3-{version_id}-b-mixed-r16"
    assert artifact is not None
    return f"fmv3-{version_id}-{arm.lower()}-{stage}-{artifact.rsplit('-r', 1)[-1]}"


def benchmark_from_registry(
    registry: Mapping[str, Any], *, registry_path: str, registry_sha256: str
) -> dict[str, Any]:
    """Materialize a BaselineSpec-compatible benchmark without heldout bodies."""

    version_id = str(registry["version_id"])
    base = _mapping(registry["base"], "registry base")
    protocol = _mapping(registry["protocol"], "registry protocol")
    heldout = _mapping(registry["heldout"], "registry heldout")
    arms = _mapping(registry["arms"], "registry arms")
    baseline_names = {
        "A": "base_matched_calls",
        "B": "mixed_matched_calls",
        "C": "c_pipeline",
        "D": "d_budget_matched_pipeline",
        "E": "e_adaptive_pareto_pipeline",
        "F": "f_adaptive_budget_matched_pipeline",
    }
    group_names = {
        arm: f"{arm}_FORMAL_V3_{'Q4_BASE' if arm == 'A' else 'LORA'}"
        for arm in ARMS
    }
    baselines: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_value = _mapping(arms[arm], f"registry arm {arm}")
        records = arm_value.get("adapters", [])
        by_name = {
            str(item["adapter_name"]): item
            for item in records
            if isinstance(item, Mapping)
        }
        stage_models: dict[str, str] = {}
        stage_ranks: dict[str, int] = {}
        stage_artifacts: dict[str, str] = {}
        for stage in STAGES:
            training_name = TRAINING_NAMES[stage]
            if arm == "A":
                artifact = None
                rank = None
            elif arm == "B":
                artifact = str(by_name["mixed_all"]["artifact_name"])
                rank = int(by_name["mixed_all"]["rank"])
            else:
                artifact = str(by_name[training_name]["artifact_name"])
                rank = int(by_name[training_name]["rank"])
            stage_models[stage] = _model_id(version_id, arm, stage, artifact)
            if artifact is not None and rank is not None:
                stage_artifacts[stage] = artifact
                stage_ranks[stage] = rank
        baseline: dict[str, Any] = {
            "name": baseline_names[arm],
            "group": group_names[arm],
            "registry_group": arm,
            "workflow": "pipeline",
            "review_protocol": protocol["review_protocol"],
            "model": _model_id(version_id, "A", "planner", None),
            "base_contract_id": base["base_contract_id"],
            "base_source_sha256": base["source_weight_sha256"],
            "q4_artifact_sha256": base["manifest_sha256"],
            "stage_adapter_artifacts": stage_artifacts,
            "stage_adapter_ranks": stage_ranks,
            "stage_models": stage_models,
            "max_tokens_per_call": protocol["max_tokens_per_call"],
            "adapter_trainable_parameters": arm_value[
                "trainable_parameter_total"
            ],
            "status": "ready",
        }
        if arm in {"E", "F"}:
            allocation = registry["allocations"][arm]
            baseline.update(
                {
                    "allocation_method": allocation["mechanism_id"],
                    "selection_split": "calibration_only",
                    "allocation_frozen": True,
                    "allocation_manifest_sha256": allocation["sha256"],
                    "maximum_stage_rank": 16,
                }
            )
        if arm == "F":
            baseline.update(
                {
                    "rank_sum_constraint": B_RANK,
                    "parameter_budget_constraint": B_PARAMETERS,
                }
            )
        baselines.append(baseline)
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "formal_version": "formal-v3",
        "version_id": version_id,
        "run_id": version_id,
        "registry_binding": {
            "path": registry_path,
            "sha256": registry_sha256,
            "contract_sha256": registry["contract_sha256"],
        },
        "base_binding": base,
        "dataset_binding": registry["snapshot"],
        "heldout_binding": heldout,
        "token_contract": protocol,
        "metrics_plan": {
            "index_baseline": "A",
            "index_value": 100,
            "equal_budget_comparison": ["B", "D", "F"],
            "capacity_comparison": ["C", "E"],
        },
        "backend_audit": {
            "serial_runtime_lora": {
                "required": True,
                "maximum_active_loras": 1,
                "maximum_cpu_loras": 1,
                "allow_static_lora_modules": False,
                "require_localhost_admin": True,
                "base_model_only_before_heldout": True,
            }
        },
        "baselines": baselines,
        "formal_v2_inputs_allowed": False,
    }


def _atomic_exclusive_bundle(
    project_root: Path,
    destination: Path,
    registry: Mapping[str, Any],
) -> dict[str, str]:
    if destination.exists():
        raise FormalV3RegistryError(
            "immutable_output_exists",
            f"formal-v3 registry bundle already exists: {destination}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        registry_path = temporary / "registry.json"
        registry_path.write_bytes(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        registry_sha = sha256_file(registry_path)
        registry_relative = (
            destination / "registry.json"
        ).resolve().relative_to(project_root.resolve()).as_posix()
        benchmark = benchmark_from_registry(
            registry,
            registry_path=registry_relative,
            registry_sha256=registry_sha,
        )
        benchmark_path = temporary / "benchmark.json"
        benchmark_path.write_bytes(
            json.dumps(benchmark, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        benchmark_sha = sha256_file(benchmark_path)
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "version_id": registry["version_id"],
            "registry_sha256": registry_sha,
            "benchmark_sha256": benchmark_sha,
            "formal_v2_inputs_allowed": False,
        }
        (temporary / "bundle_manifest.json").write_bytes(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        os.replace(temporary, destination)
        return {
            "registry": str(destination / "registry.json"),
            "registry_sha256": registry_sha,
            "benchmark": str(destination / "benchmark.json"),
            "benchmark_sha256": benchmark_sha,
            "bundle_manifest": str(destination / "bundle_manifest.json"),
        }
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


def finalize_bundle(
    control_path: str | Path,
    project_root: str | Path,
    *,
    version_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = _inside(root, str(output_dir), "registry output directory")
    registry = build_registry(control_path, root, version_id=version_id)
    paths = _atomic_exclusive_bundle(root, output, registry)
    return {
        "status": "READY",
        "version_id": version_id,
        "heldout_case_content_read": False,
        **paths,
    }


def inspect_readiness(
    control_path: str | Path,
    project_root: str | Path,
    *,
    version_id: str,
) -> dict[str, Any]:
    """Return a machine-readable READY/BLOCKED report without writing anything."""

    try:
        registry = build_registry(control_path, project_root, version_id=version_id)
    except FormalV3RegistryError as exc:
        return {
            "schema_version": "anchor.formal-v3-af-readiness.v1",
            "status": "BLOCKED",
            "code": exc.code,
            "message": str(exc),
            "details": exc.details,
            "heldout_case_content_read": False,
            "gpu_started": False,
            "api_called": False,
        }
    return {
        "schema_version": "anchor.formal-v3-af-readiness.v1",
        "status": "READY_TO_FINALIZE",
        "version_id": version_id,
        "registry_contract_sha256": registry["contract_sha256"],
        "heldout_case_content_read": False,
        "gpu_started": False,
        "api_called": False,
    }
