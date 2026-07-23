"""Metadata-only Gemma 3 1B IT five-expert Q-only budget audit.

This module never imports a tensor runtime and never opens model weights.  It
validates the versioned contract from a single bytes snapshot, checks its
mandatory sha256sum-style sidecar, and recomputes every parameter/memory row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


CONTRACT_VERSION = "anchor.gemma3-1b-it-five-expert-qonly-parameter-budget.v1"
CONTRACT_RELATIVE = (
    "configs/research/gemma3_1b_it_five_expert_qonly_parameter_budget_v1.json"
)
SCHEMA_RELATIVE = (
    "configs/research/gemma3_1b_it_five_expert_qonly_parameter_budget_v1.schema.json"
)
SIDECAR_RELATIVE = f"{CONTRACT_RELATIVE}.sha256"

BASE_PARAMETERS = 999_885_952
LAYERS = 26
Q_INPUT = 1152
Q_OUTPUT = 1024
EXPERTS = 5
PER_EXPERT_PARAMS_PER_RANK = LAYERS * (Q_INPUT + Q_OUTPUT)
DENSE_Q_PER_EXPERT_PARAMS = LAYERS * Q_INPUT * Q_OUTPUT
EFFECTIVE_RANK_CEILING = min(Q_INPUT, Q_OUTPUT)
EXPECTED_RANKS = (4, 8, 16, 32, 64, 256, 512, 542, 1024, 3535)


class BudgetContractError(RuntimeError):
    """Raised when the metadata-only contract fails closed."""


@dataclass(frozen=True)
class BytesSnapshot:
    path: Path
    data: bytes
    sha256: str


def _snapshot(path: Path) -> BytesSnapshot:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BudgetContractError(f"required artifact is unreadable: {path}") from exc
    return BytesSnapshot(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _parse_json(snapshot: BytesSnapshot) -> dict[str, Any]:
    if snapshot.data.startswith(b"\xef\xbb\xbf"):
        raise BudgetContractError(f"UTF-8 BOM is forbidden: {snapshot.path}")
    if b"\r" in snapshot.data:
        raise BudgetContractError(f"CR/CRLF is forbidden: {snapshot.path}")
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BudgetContractError(f"invalid UTF-8 JSON: {snapshot.path}") from exc
    if not isinstance(value, dict):
        raise BudgetContractError(f"JSON root must be an object: {snapshot.path}")
    return value


def calculate_rank_row(rank: int) -> dict[str, int]:
    """Return the exact integer parameter and byte estimates for one rank."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    per_expert = PER_EXPERT_PARAMS_PER_RANK * rank
    aggregate = EXPERTS * per_expert
    base_bf16 = 2 * BASE_PARAMETERS
    return {
        "rank": rank,
        "per_expert_params": per_expert,
        "five_expert_aggregate_params": aggregate,
        "single_route_active_params": per_expert,
        "per_expert_bf16_checkpoint_bytes": 2 * per_expert,
        "five_expert_bf16_checkpoint_bytes": 2 * aggregate,
        "single_active_training_state_16bpp_bytes": 16 * per_expert,
        "all_experts_training_state_16bpp_bytes": 16 * aggregate,
        "rough_peak_active_only_bytes": base_bf16 + 16 * per_expert,
        "rough_peak_sequential_all_checkpoints_resident_bytes": (
            base_bf16 + 24 * per_expert
        ),
        "rough_peak_all_experts_optimizer_resident_bytes": (base_bf16 + 16 * aggregate),
    }


def _audit_rank_table(contract: dict[str, Any]) -> None:
    rows = contract["rank_table"]
    observed_ranks = tuple(row["rank"] for row in rows)
    if observed_ranks != EXPECTED_RANKS:
        raise BudgetContractError(
            f"rank table must be ordered exactly as {EXPECTED_RANKS}"
        )
    for row in rows:
        expected = calculate_rank_row(row["rank"])
        for key, expected_value in expected.items():
            if row.get(key) != expected_value:
                raise BudgetContractError(
                    f"rank {row['rank']} field {key} is not derivable from metadata"
                )


def audit_contract(repo_root: Path) -> dict[str, Any]:
    """Validate contract, schema, sidecar, and all pure-math identities."""

    root = repo_root.resolve(strict=True)
    contract_snapshot = _snapshot(root / CONTRACT_RELATIVE)
    schema_snapshot = _snapshot(root / SCHEMA_RELATIVE)
    sidecar_snapshot = _snapshot(root / SIDECAR_RELATIVE)

    expected_sidecar = (
        f"{contract_snapshot.sha256}  {Path(CONTRACT_RELATIVE).name}\n"
    ).encode("ascii")
    if sidecar_snapshot.data != expected_sidecar:
        raise BudgetContractError("mandatory contract SHA-256 sidecar is invalid")

    contract = _parse_json(contract_snapshot)
    schema = _parse_json(schema_snapshot)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)

    if contract["schema_version"] != CONTRACT_VERSION:
        raise BudgetContractError("unexpected contract version")
    binding = contract["audit"]["schema_binding"]
    if (
        binding["path"] != SCHEMA_RELATIVE
        or binding["sha256"] != schema_snapshot.sha256
    ):
        raise BudgetContractError("schema physical SHA-256 binding mismatch")

    math = contract["parameter_math"]
    if math["per_expert_params_per_rank"] != PER_EXPERT_PARAMS_PER_RANK:
        raise BudgetContractError("per-expert rank slope mismatch")
    if math["five_expert_params_per_rank"] != (EXPERTS * PER_EXPERT_PARAMS_PER_RANK):
        raise BudgetContractError("five-expert rank slope mismatch")
    if math["dense_q_per_expert_params"] != DENSE_Q_PER_EXPERT_PARAMS:
        raise BudgetContractError("dense-Q control mismatch")
    if math["effective_rank_ceiling"] != EFFECTIVE_RANK_CEILING:
        raise BudgetContractError("effective-rank ceiling mismatch")
    nearest = round(BASE_PARAMETERS / (EXPERTS * PER_EXPERT_PARAMS_PER_RANK))
    if nearest != 3535 or math["nearest_base_parity_rank"] != nearest:
        raise BudgetContractError("nearest base-parity rank mismatch")

    _audit_rank_table(contract)
    return {
        "status": contract["status"],
        "contract_sha256": contract_snapshot.sha256,
        "sidecar_sha256": sidecar_snapshot.sha256,
        "schema_sha256": schema_snapshot.sha256,
        "base_parameters": BASE_PARAMETERS,
        "rank_count": len(contract["rank_table"]),
        "training_authorized": contract["claims"]["training_authorized"],
        "model_loads": contract["counters"]["model_loads"],
        "gpu_operations": contract["counters"]["gpu_operations"],
        "provider_requests": contract["counters"]["provider_requests"],
        "network_requests": contract["counters"]["network_requests"],
        "weight_tensor_reads": contract["counters"]["weight_tensor_reads"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the metadata-only Gemma 3 1B IT five-expert Q-only "
            "parameter budget contract"
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_contract(args.repo_root)
    except BudgetContractError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
