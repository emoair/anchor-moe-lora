"""Contract-only preflight for compact-v2b segmented A--F evaluation.

This preflight reads the machine-readable segment contract and aggregate compact
training manifest only.  It never opens held-out cases, benchmark records, adapters,
or a model backend.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .segment_protocol import SegmentContract


SCHEMA = "anchor.segmented-eval-contract.v1"


class SegmentPreflightError(ValueError):
    """Raised when the frozen segment contract is incomplete or inconsistent."""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentPreflightError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise SegmentPreflightError(f"{label} must contain one JSON object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SegmentPreflightError(f"{label} escapes the project root") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_segment_contract(
    contract_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract_file = Path(contract_path)
    if not contract_file.is_absolute():
        contract_file = _inside(root, contract_file.as_posix(), "segment contract")
    contract = _load_object(contract_file, "segment contract")
    if contract.get("schema_version") != SCHEMA:
        raise SegmentPreflightError("unsupported segment contract schema")

    binding = SegmentContract(
        artifact_protocol=str(contract.get("artifact_protocol", "")),
        contract_version=str(contract.get("segment_contract_version", "")),
        frontend_segments=int(contract.get("frontend_segment_count", 0)),
        review_segments=int(contract.get("review_segment_count", 0)),
    )
    if contract.get("review_protocol") != "segmented_repair_v1":
        raise SegmentPreflightError("segment contract has the wrong review protocol")
    if int(contract.get("expected_physical_calls", 0)) != binding.expected_calls:
        raise SegmentPreflightError("segment contract physical call count is inconsistent")
    token_cap = int(contract.get("max_completion_tokens_per_physical_call", 0))
    if token_cap < 1:
        raise SegmentPreflightError("segment contract requires a positive token cap")

    selection = contract.get("count_selection")
    if not isinstance(selection, Mapping):
        raise SegmentPreflightError("segment count selection metadata is missing")
    if selection.get("heldout_content_read") is not False or selection.get(
        "benchmark_record_content_read"
    ) is not False:
        raise SegmentPreflightError("segment counts must be selected without evaluation data")
    manifest_file = _inside(
        root,
        str(selection.get("coverage_manifest", "")),
        "compact coverage manifest",
    )
    expected_manifest_sha = str(selection.get("coverage_manifest_sha256", ""))
    if _sha256(manifest_file) != expected_manifest_sha:
        raise SegmentPreflightError("compact coverage manifest digest changed")
    manifest = _load_object(manifest_file, "compact coverage manifest")
    if manifest.get("artifact_protocol") != binding.artifact_protocol:
        raise SegmentPreflightError("compact manifest artifact protocol changed")
    if manifest.get("heldout_content_read") is not False or manifest.get(
        "benchmark_records_content_read"
    ) is not False:
        raise SegmentPreflightError("compact manifest is not evaluation-data isolated")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise SegmentPreflightError("compact coverage metadata is missing")
    target_maxima: list[int] = []
    for expert in ("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate"):
        expert_coverage = coverage.get(expert)
        if not isinstance(expert_coverage, Mapping):
            raise SegmentPreflightError(f"compact coverage is missing {expert}")
        target_tokens = expert_coverage.get("target_tokens")
        if not isinstance(target_tokens, Mapping):
            raise SegmentPreflightError(f"compact target coverage is missing {expert}")
        target_maxima.append(int(target_tokens.get("max", 0)))
    maximum_target = max(target_maxima)
    if token_cap < maximum_target:
        raise SegmentPreflightError(
            "physical-call token cap is below the audited compact target maximum"
        )

    fairness = contract.get("fairness")
    if not isinstance(fairness, Mapping):
        raise SegmentPreflightError("segment fairness contract is missing")
    if fairness.get("same_contract_required_for_arms") != ["A", "B", "C", "D", "E", "F"]:
        raise SegmentPreflightError("segment fairness contract must cover exactly A--F")
    required_true = (
        "same_base_q4_required",
        "same_logical_stage_order_required",
        "same_segment_counts_required",
        "same_per_call_completion_cap_required",
    )
    if any(fairness.get(key) is not True for key in required_true):
        raise SegmentPreflightError("segment fairness requirements are incomplete")
    if fairness.get("target_artifact_digest_may_be_used_as_input") is not False:
        raise SegmentPreflightError("target artifact digests must remain unavailable")
    if fairness.get("index_baseline") != "A" or fairness.get("index_value") != 100:
        raise SegmentPreflightError("native Q4 A must be indexed at 100")

    execution = contract.get("execution_gate")
    if not isinstance(execution, Mapping):
        raise SegmentPreflightError("segment execution gate is missing")
    if execution.get("gpu_execution_authorized") is not False or execution.get(
        "heldout_execution_authorized"
    ) is not False:
        raise SegmentPreflightError("contract-only preflight cannot authorize execution")

    return {
        "status": "contract_valid_training_artifacts_pending",
        "schema_version": SCHEMA,
        "artifact_protocol": binding.artifact_protocol,
        "segment_contract_version": binding.contract_version,
        "frontend_segment_count": binding.frontend_segments,
        "review_segment_count": binding.review_segments,
        "expected_physical_calls": binding.expected_calls,
        "max_completion_tokens_per_physical_call": token_cap,
        "audited_maximum_target_tokens": maximum_target,
        "a100_baseline": "A",
        "heldout_content_read": False,
        "benchmark_record_content_read": False,
        "gpu_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate segmented formal-v2 wiring")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)
    result = preflight_segment_contract(args.contract, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
