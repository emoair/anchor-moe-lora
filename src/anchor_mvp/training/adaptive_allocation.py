"""Freeze and verify calibration-isolated E/F LoRA rank preregistrations.

The current partial-data experiment has no aggregate calibration measurements yet.
This module therefore freezes an explicitly heuristic, calibration-pending rank plan
that is eligible for training but *not* for held-out evaluation or Pareto claims.
It never opens benchmark cases or sample records; only small aggregate manifests are
read.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "anchor.lora-allocation.v1"
CALIBRATION_SCHEMA = "anchor.adaptive-calibration-snapshot.v1"
MECHANISM_ID = "stage_complexity_calibration_pareto_v1"
ALGORITHM_ID = "deterministic_stage_complexity_utility_v1"
BASE_CONTRACT_ID = "gemma4-12b-r56820d7-bnb-nf4-doublequant-bf16-v1"
PARAMETERS_PER_RANK = 649_216
B_RANK = 16
B_PARAMETERS = PARAMETERS_PER_RANK * B_RANK
SEED = 20_260_715
STAGES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
RANK_MENU = (1, 2, 3, 4, 6, 8, 12, 16)
STAGE_WEIGHTS = {
    "planner": 2,
    "tool_policy": 1,
    "frontend_gen": 4,
    "frontend_review": 3,
    "security_gate": 1,
}
UTILITY_POINTS = {1: 100, 2: 158, 3: 200, 4: 232, 6: 281, 8: 317, 12: 370, 16: 409}
E_CANDIDATE_RANKS = (
    (2, 1, 4, 3, 1),
    (4, 2, 8, 6, 2),
    (6, 3, 12, 8, 3),
    (8, 4, 16, 12, 4),
)
E_SELECTED_RANKS = dict(zip(STAGES, E_CANDIDATE_RANKS[-1], strict=True))


class AllocationError(ValueError):
    """Raised when an adaptive allocation or one of its bindings is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AllocationError(f"expected a JSON object: {path}")
    return value


def _safe_project_file(root: Path, relative: str, *, label: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise AllocationError(f"{label} must be a project-relative file")
    # Rank selection must never gain a path to held-out material.
    if "heldout" in relative.lower() or "held-out" in relative.lower():
        raise AllocationError(f"{label} must not reference held-out material")
    root = root.resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise AllocationError(f"{label} escapes the project root")
    if not path.is_file():
        raise AllocationError(f"{label} does not exist: {path}")
    return path


def _rank_mapping(values: tuple[int, ...]) -> dict[str, int]:
    return dict(zip(STAGES, values, strict=True))


def _signature(ranks: Mapping[str, int]) -> str:
    return ";".join(f"{stage}={ranks[stage]}" for stage in STAGES)


def _utility(ranks: Mapping[str, int]) -> int:
    return sum(STAGE_WEIGHTS[stage] * UTILITY_POINTS[ranks[stage]] for stage in STAGES)


def _candidate(ranks: Mapping[str, int]) -> dict[str, Any]:
    rank_sum = sum(ranks.values())
    signature = _signature(ranks)
    return {
        "selected_ranks": dict(ranks),
        "rank_sum": rank_sum,
        "materialized_trainable_parameters": rank_sum * PARAMETERS_PER_RANK,
        "heuristic_utility_points": _utility(ranks),
        "seeded_tiebreak_sha256": _sha256_bytes(f"{SEED}:{signature}".encode()),
        "calibration_metrics": None,
    }


def e_candidates() -> list[dict[str, Any]]:
    """Return the preregistered increasing-capacity E candidate ladder."""

    return [_candidate(_rank_mapping(values)) for values in E_CANDIDATE_RANKS]


def f_candidates() -> list[dict[str, Any]]:
    """Enumerate the complete non-uniform, exact-B-rank F search space."""

    candidates = []
    for values in itertools.product(RANK_MENU, repeat=len(STAGES)):
        if sum(values) != B_RANK or len(set(values)) == 1:
            continue
        candidates.append(_candidate(_rank_mapping(values)))
    return sorted(
        candidates,
        key=lambda item: (
            -item["heuristic_utility_points"],
            item["seeded_tiebreak_sha256"],
        ),
    )


def _weight_set_digest(quantization_manifest: Mapping[str, Any]) -> str:
    weights = quantization_manifest.get("weights")
    if not isinstance(weights, list) or not weights:
        raise AllocationError("NF4 quantization manifest has no weight bindings")
    lines: list[str] = []
    for entry in weights:
        if not isinstance(entry, Mapping):
            raise AllocationError("NF4 weight binding must be an object")
        path, size, digest = entry.get("path"), entry.get("bytes"), entry.get("sha256")
        if (
            not isinstance(path, str)
            or Path(path).name != path
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise AllocationError("NF4 weight binding is malformed")
        lines.append(f"{path}:{size}:{digest}")
    return _sha256_bytes("\n".join(lines).encode())


def _selection_algorithm() -> dict[str, Any]:
    return {
        "algorithm_id": ALGORITHM_ID,
        "seed": SEED,
        "rank_menu": list(RANK_MENU),
        "stage_weights": dict(STAGE_WEIGHTS),
        "utility_points": {str(key): value for key, value in UTILITY_POINTS.items()},
        "score": "sum(stage_weight * utility_points[rank])",
        "tie_break": "ascending sha256(seed:stage-rank-signature)",
        "calibration_performance_used": False,
    }


def build_preregistered_allocations(
    root: Path,
    *,
    calibration_manifest: str,
    dataset_manifest: str,
    quantization_manifest: str,
    created_at: str,
) -> dict[str, dict[str, Any]]:
    """Build E/F manifests from aggregate bindings without opening any samples."""

    root = root.resolve()
    calibration_path = _safe_project_file(
        root, calibration_manifest, label="calibration_manifest"
    )
    dataset_path = _safe_project_file(root, dataset_manifest, label="dataset_manifest")
    quantization_path = _safe_project_file(
        root, quantization_manifest, label="quantization_manifest"
    )
    calibration = _read_json_object(calibration_path)
    dataset = _read_json_object(dataset_path)
    quantization = _read_json_object(quantization_path)
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise AllocationError("calibration snapshot schema mismatch")
    if calibration.get("split") != "calibration_only":
        raise AllocationError("rank allocation accepts calibration_only aggregates")
    if calibration.get("heldout_accessed") is not False:
        raise AllocationError("calibration snapshot must attest heldout_accessed=false")
    if calibration.get("record_count") != 0 or calibration.get("metrics") != []:
        raise AllocationError(
            "this preregistration builder is only for the explicit empty-calibration state"
        )
    if dataset.get("schema_version") != "anchor.per-expert-partial-training-snapshot.v1":
        raise AllocationError("partial dataset snapshot schema mismatch")
    if dataset.get("not_for_end_to_end_claim") is not True:
        raise AllocationError("partial dataset claim waiver is missing")
    if quantization.get("schema_version") != "anchor.bnb-nf4-export.v1":
        raise AllocationError("NF4 quantization manifest schema mismatch")

    common = {
        "schema_version": SCHEMA_VERSION,
        "mechanism_id": MECHANISM_ID,
        "base_contract_id": BASE_CONTRACT_ID,
        "base_source_sha256": quantization.get("source_weight_sha256"),
        "q4_artifact_sha256": _sha256_file(quantization_path),
        "base_q4_nf4_weight_set_sha256": _weight_set_digest(quantization),
        "quantization_manifest_path": quantization_manifest,
        "target_modules": ["q_proj", "v_proj"],
        "parameters_per_rank": PARAMETERS_PER_RANK,
        "dataset_snapshot_sha256": dataset.get("snapshot_sha256"),
        "dataset_snapshot_manifest_sha256": _sha256_file(dataset_path),
        "dataset_snapshot_manifest_path": dataset_manifest,
        "calibration_snapshot_sha256": _sha256_file(calibration_path),
        "calibration_snapshot_path": calibration_manifest,
        "calibration_record_count": 0,
        "calibration_metrics_available": False,
        "selection_status": "heuristic_preregistered_calibration_pending",
        "training_eligible": True,
        "heldout_evaluation_eligible": False,
        "allocation_frozen_before_heldout": True,
        "allocation_frozen_at": created_at,
        "created_at": created_at,
        "heldout_access": "forbidden_until_calibration_supported_allocation_frozen",
        "heldout_opened": False,
        "heldout_opened_at": None,
        "selection_algorithm": _selection_algorithm(),
        "attempted_allocations": [],
        "limitation": (
            "No calibration measurements exist. Ranks are a complexity-prior "
            "preregistration for training only; they are not a measured Pareto optimum."
        ),
    }
    selected_e = e_candidates()[-1]
    all_f = f_candidates()
    selected_f = all_f[0]
    allocations = {
        "E": {
            **common,
            "arm": "E",
            "selection_objectives": [
                "maximize_per_stage_calibration_quality",
                "minimize_materialized_parameters",
                "minimize_routed_latency",
                "minimize_peak_vram",
            ],
            "candidate_allocations": e_candidates(),
            "candidate_space_sha256": _sha256_bytes(_canonical_json(e_candidates())),
            "selected_ranks": selected_e["selected_ranks"],
            "rank_sum": selected_e["rank_sum"],
            "materialized_trainable_parameters": selected_e[
                "materialized_trainable_parameters"
            ],
            "selection_reason": (
                "Highest preregistered tier-cap candidate: highest=16, high=12, "
                "medium=8, low=4. Calibration support remains pending."
            ),
        },
        "F": {
            **common,
            "arm": "F",
            "selection_objectives": [
                "maximize_per_stage_calibration_quality",
                "minimize_routed_latency",
                "minimize_peak_vram",
            ],
            "candidate_allocations": all_f,
            "candidate_space_sha256": _sha256_bytes(_canonical_json(all_f)),
            "selected_ranks": selected_f["selected_ranks"],
            "rank_sum": selected_f["rank_sum"],
            "materialized_trainable_parameters": selected_f[
                "materialized_trainable_parameters"
            ],
            "reference_B": {
                "rank": B_RANK,
                "materialized_trainable_parameters": B_PARAMETERS,
            },
            "selection_reason": (
                "Top deterministic complexity-prior utility candidate among every "
                "non-uniform rank-menu allocation with exact total rank 16. "
                "Calibration support remains pending."
            ),
        },
    }
    return allocations


def write_allocations(output_dir: Path, allocations: Mapping[str, Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("E", "F"):
        path = output_dir / f"formal_partial_v1_{arm}.heuristic-calibration-pending.json"
        if path.exists() or Path(f"{path}.sha256").exists():
            raise AllocationError(f"refusing to overwrite immutable allocation: {path}")
        payload = _canonical_json(allocations[arm])
        digest = _sha256_bytes(payload)
        path.write_bytes(payload)
        Path(f"{path}.sha256").write_text(
            f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
        )


def validate_preregistered_allocation(
    path: Path, *, root: Path, expected_arm: str
) -> dict[str, Any]:
    """Validate all immutable bindings and return the frozen manifest."""

    root, path = root.resolve(), path.resolve()
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise AllocationError("allocation manifest or SHA-256 sidecar is missing")
    tokens = sidecar.read_text(encoding="ascii").split()
    if not tokens or tokens[0] != _sha256_file(path):
        raise AllocationError("allocation manifest SHA-256 sidecar mismatch")
    value = _read_json_object(path)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("arm") != expected_arm:
        raise AllocationError("allocation schema/arm mismatch")
    if value.get("selection_status") != "heuristic_preregistered_calibration_pending":
        raise AllocationError("unexpected selection status")
    if not (
        value.get("mechanism_id") == MECHANISM_ID
        and value.get("base_contract_id") == BASE_CONTRACT_ID
        and value.get("target_modules") == ["q_proj", "v_proj"]
        and value.get("parameters_per_rank") == PARAMETERS_PER_RANK
        and value.get("selection_algorithm") == _selection_algorithm()
    ):
        raise AllocationError("allocation mechanism/base/algorithm contract mismatch")
    if not (
        value.get("training_eligible") is True
        and value.get("heldout_evaluation_eligible") is False
        and value.get("heldout_opened") is False
        and value.get("calibration_metrics_available") is False
        and value.get("calibration_record_count") == 0
        and value.get("attempted_allocations") == []
    ):
        raise AllocationError("calibration-pending training/heldout contract is invalid")
    bindings = (
        ("dataset_snapshot_manifest_path", "dataset_snapshot_manifest_sha256"),
        ("quantization_manifest_path", "q4_artifact_sha256"),
        ("calibration_snapshot_path", "calibration_snapshot_sha256"),
    )
    opened: dict[str, dict[str, Any]] = {}
    for path_field, digest_field in bindings:
        relative = value.get(path_field)
        if not isinstance(relative, str):
            raise AllocationError(f"{path_field} is missing")
        bound = _safe_project_file(root, relative, label=path_field)
        if value.get(digest_field) != _sha256_file(bound):
            raise AllocationError(f"{digest_field} binding mismatch")
        opened[path_field] = _read_json_object(bound)
    dataset = opened["dataset_snapshot_manifest_path"]
    quantization = opened["quantization_manifest_path"]
    calibration = opened["calibration_snapshot_path"]
    if value.get("dataset_snapshot_sha256") != dataset.get("snapshot_sha256"):
        raise AllocationError("dataset snapshot identifier mismatch")
    if value.get("base_source_sha256") != quantization.get("source_weight_sha256"):
        raise AllocationError("base source SHA-256 mismatch")
    if value.get("base_q4_nf4_weight_set_sha256") != _weight_set_digest(quantization):
        raise AllocationError("NF4 weight-set digest mismatch")
    if not (
        calibration.get("schema_version") == CALIBRATION_SCHEMA
        and calibration.get("split") == "calibration_only"
        and calibration.get("record_count") == 0
        and calibration.get("metrics") == []
        and calibration.get("heldout_accessed") is False
    ):
        raise AllocationError("calibration aggregate manifest is invalid")
    ranks = value.get("selected_ranks")
    if not isinstance(ranks, Mapping) or set(ranks) != set(STAGES):
        raise AllocationError("selected_ranks must name exactly the five specialists")
    normalized = {stage: ranks[stage] for stage in STAGES}
    if any(rank not in RANK_MENU for rank in normalized.values()):
        raise AllocationError("selected rank is outside the preregistered menu")
    if len(set(normalized.values())) == 1:
        raise AllocationError("adaptive allocation must be non-uniform")
    rank_sum = sum(normalized.values())
    if value.get("rank_sum") != rank_sum:
        raise AllocationError("rank_sum mismatch")
    if value.get("materialized_trainable_parameters") != rank_sum * PARAMETERS_PER_RANK:
        raise AllocationError("materialized parameter count mismatch")
    expected_candidates = e_candidates() if expected_arm == "E" else f_candidates()
    expected = (
        E_SELECTED_RANKS
        if expected_arm == "E"
        else expected_candidates[0]["selected_ranks"]
    )
    if value.get("candidate_allocations") != expected_candidates:
        raise AllocationError("candidate allocation ledger mismatch")
    if value.get("candidate_space_sha256") != _sha256_bytes(
        _canonical_json(expected_candidates)
    ):
        raise AllocationError("candidate allocation space hash mismatch")
    if normalized != expected:
        raise AllocationError("selected ranks do not match the deterministic preregistration")
    if expected_arm == "F" and (
        rank_sum != B_RANK or rank_sum * PARAMETERS_PER_RANK != B_PARAMETERS
    ):
        raise AllocationError("F does not exactly match B's materialized parameter budget")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--quantization-manifest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    allocations = build_preregistered_allocations(
        args.root,
        calibration_manifest=args.calibration_manifest,
        dataset_manifest=args.dataset_manifest,
        quantization_manifest=args.quantization_manifest,
        created_at=args.created_at,
    )
    write_allocations(args.output_dir, allocations)
    print(
        json.dumps(
            {arm: value["selected_ranks"] for arm, value in allocations.items()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
