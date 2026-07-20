"""Read-only scale-up gates for data, base artifacts, GPU capacity, and probes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .manifest import sha256_file
from .schema import DatasetValidationError, iter_jsonl, validate_jsonl
from .config import (
    PARTIAL_SNAPSHOT_SCHEMA,
    PARTIAL_TRAINING_MODE,
)
from .formal_v3_schedule import (
    CANDIDATE_TASKS_PER_STAGE,
    CANDIDATE_WORK_ORDERS,
    SPLIT_SCHEMA,
)


REQUIRED_EXPERTS = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
EXPERT_TASKS = {
    "planner": "plan",
    "tool_policy": "tool_policy",
    "frontend_gen": "frontend",
    "frontend_review": "review",
    "security_gate": "security",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_BANK_FILENAME = "task_bank.jsonl"
_PARTIAL_EXCLUSIONS = frozenset(
    {"negative", "reject", "oracle_label_only", "heldout"}
)


def _gate(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def _safe_nonnegative(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _task_card_coverage_valid(
    value: object, *, complete_chain_count: int | None
) -> bool:
    if not isinstance(value, Mapping) or complete_chain_count is None:
        return False
    return bool(
        value.get("passed") is True
        and value.get("cardinality_equal") is True
        and _safe_nonnegative(value.get("complete_chain_count")) == complete_chain_count
        and _safe_nonnegative(value.get("card_count")) == complete_chain_count
        and _safe_nonnegative(value.get("unique_alignment_id_count"))
        == complete_chain_count
    )


def _task_bank_source_binding_valid(
    value: object, *, complete_chain_count: int | None
) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"path", "records", "bytes", "sha256"}
        and value.get("path") == TASK_BANK_FILENAME
        and _safe_nonnegative(value.get("records")) == complete_chain_count
        and _safe_nonnegative(value.get("bytes")) is not None
        and isinstance(value.get("sha256"), str)
        and bool(_SHA256_RE.fullmatch(value["sha256"]))
    )


def _task_bank_binding(path: Path) -> dict[str, Any]:
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("task bank rows must be JSON objects")
            records += 1
    return {
        "path": TASK_BANK_FILENAME,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _dataset_stats(path: Path, expected_expert: str) -> dict[str, Any]:
    validation = validate_jsonl(path, allowed_experts=[expected_expert])
    assistant_lengths: list[int] = []
    identifiers: list[str] = []
    live_records = 0
    for _, record in iter_jsonl(path):
        identifiers.append(str(record["id"]))
        assistant_lengths.append(len(str(record["messages"][-1]["content"]).strip()))
        provenance = record.get("provenance", {})
        teacher = (
            provenance.get("teacher", {}) if isinstance(provenance, Mapping) else {}
        )
        model = (
            str(teacher.get("model", "")).strip().casefold()
            if isinstance(teacher, Mapping)
            else ""
        )
        base_url = (
            str(teacher.get("base_url", "")).strip().casefold()
            if isinstance(teacher, Mapping)
            else ""
        )
        if (
            model
            and model not in {"mock", "mock-teacher", "fixture"}
            and not base_url.startswith("mock:")
        ):
            live_records += 1
    count = len(assistant_lengths)
    return {
        **validation,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "assistant_target_chars": {
            "min": min(assistant_lengths),
            "max": max(assistant_lengths),
            "mean": round(sum(assistant_lengths) / count, 2),
            "empty": sum(length == 0 for length in assistant_lengths),
        },
        "live_records": live_records,
        "all_records_live": live_records == count,
        "ids": identifiers,
    }


def inspect_gate_datasets(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    gate_config = config["scale_gate"]
    required = gate_config["required_datasets"]
    reports: dict[str, dict[str, Any]] = {}
    all_ids: list[str] = []
    for expert in REQUIRED_EXPERTS:
        relative = required.get(expert)
        path = (
            (root / relative).resolve()
            if isinstance(relative, str)
            else root / "<missing>"
        )
        if not isinstance(relative, str) or not path.is_file():
            reports[expert] = {
                "path": str(path),
                "exists": False,
                "valid": False,
                "error": "required canonical live-smoke dataset is missing",
            }
            continue
        try:
            stats = _dataset_stats(path, expert)
            identifiers = stats.pop("ids")
            all_ids.extend(identifiers)
            reports[expert] = {
                "path": str(path),
                "exists": True,
                "valid": True,
                **stats,
            }
        except (DatasetValidationError, OSError, ValueError) as exc:
            reports[expert] = {
                "path": str(path),
                "exists": True,
                "valid": False,
                "error": str(exc),
            }

    duplicate_count = len(all_ids) - len(set(all_ids))
    complete = all(report.get("exists") for report in reports.values())
    schemas_valid = complete and all(report.get("valid") for report in reports.values())
    live = schemas_valid and all(
        report.get("all_records_live") for report in reports.values()
    )
    nonempty = schemas_valid and all(
        report.get("assistant_target_chars", {}).get("empty") == 0
        for report in reports.values()
    )
    digest_parts = [
        f"{expert}:{reports[expert].get('sha256', 'missing')}"
        for expert in REQUIRED_EXPERTS
    ]
    snapshot_sha256 = hashlib.sha256(
        "\n".join(digest_parts).encode("utf-8")
    ).hexdigest()
    return {
        "reports": reports,
        "snapshot_sha256": snapshot_sha256,
        "complete": complete,
        "schemas_valid": schemas_valid,
        "all_records_live": live,
        "assistant_targets_nonempty": nonempty,
        "cross_file_duplicate_ids": duplicate_count,
    }


def _inspect_partial_dataset_snapshot_manifest(
    config: Mapping[str, Any], root: Path, datasets: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a balanced partial-Gold snapshot without granting DAG claims."""

    snapshot_config = config["scale_gate"]["dataset_snapshot"]
    manifest_path = (root / snapshot_config["manifest"]).resolve()
    sidecar_path = (root / snapshot_config["sidecar"]).resolve()
    report: dict[str, Any] = {
        "required": True,
        "passed": False,
        "manifest_path": str(manifest_path),
        "sidecar_path": str(sidecar_path),
        "training_mode": PARTIAL_TRAINING_MODE,
        "not_for_end_to_end_claim": True,
        "errors": [],
    }
    errors: list[str] = report["errors"]
    if sidecar_path != Path(str(manifest_path) + ".sha256"):
        errors.append("snapshot sidecar must be manifest.json.sha256 beside the manifest")
    if not manifest_path.is_file():
        errors.append("immutable partial snapshot manifest is missing")
    if not sidecar_path.is_file():
        errors.append("immutable partial snapshot SHA-256 sidecar is missing")
    if errors:
        return report

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar_tokens = sidecar_path.read_text(encoding="ascii").split()
        if not sidecar_tokens or sidecar_tokens[0] != manifest_sha256:
            errors.append("snapshot manifest SHA-256 sidecar mismatch")
        value = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(value, Mapping):
            errors.append("snapshot manifest must be a JSON object")
            return report

        per_expert = snapshot_config["balanced_records_per_expert"]
        exclusions = value.get("excluded")
        contract_checks = {
            "schema": value.get("schema_version") == PARTIAL_SNAPSHOT_SCHEMA,
            "training_mode": value.get("training_mode") == PARTIAL_TRAINING_MODE,
            "claim_waiver": value.get("not_for_end_to_end_claim") is True,
            "selection_size": value.get("per_expert") == per_expert,
            "selection_algorithm": value.get("selection")
            == "sha256(seed:id), ascending",
            "selection_seed": bool(
                isinstance(value.get("seed"), int)
                and not isinstance(value.get("seed"), bool)
                and value["seed"] >= 0
            ),
            "total_records": value.get("total_records")
            == per_expert * len(REQUIRED_EXPERTS),
            "source_export_schema": value.get("source_export_schema_version")
            == "anchor.per-expert-partial-gold-export.v1",
            "source_export_hash": bool(
                isinstance(value.get("source_export_manifest_sha256"), str)
                and _SHA256_RE.fullmatch(value["source_export_manifest_sha256"])
            ),
            "source_partition_hash": bool(
                isinstance(value.get("source_partition_manifest_sha256"), str)
                and _SHA256_RE.fullmatch(value["source_partition_manifest_sha256"])
            ),
            "waivers_preserved": isinstance(value.get("waivers"), Mapping),
            "strict_chain_count_metadata": _safe_nonnegative(
                value.get("strict_complete_chains")
            )
            is not None,
            "exclusions": bool(
                isinstance(exclusions, Mapping)
                and all(exclusions.get(name) is True for name in _PARTIAL_EXCLUSIONS)
            ),
        }
        if not all(contract_checks.values()):
            errors.append("partial snapshot scope or exclusion contract is invalid")

        files = value.get("files")
        if not isinstance(files, Mapping) or set(files) != set(REQUIRED_EXPERTS):
            errors.append("snapshot manifest files must map exactly the five experts")
            files = {}

        digest_parts: list[str] = []
        file_checks: dict[str, dict[str, bool]] = {}
        for expert in REQUIRED_EXPERTS:
            entry = files.get(expert)
            observed = datasets["reports"].get(expert, {})
            checks: dict[str, bool] = {}
            if not isinstance(entry, Mapping):
                errors.append(f"snapshot manifest is missing files.{expert}")
                file_checks[expert] = checks
                continue
            relative = entry.get("path")
            safe_basename = bool(
                isinstance(relative, str)
                and relative
                and Path(relative).name == relative
            )
            configured = (
                root / config["scale_gate"]["required_datasets"][expert]
            ).resolve()
            bound = (
                (manifest_path.parent / relative).resolve() if safe_basename else None
            )
            checks.update(
                {
                    "safe_basename": safe_basename,
                    "configured_path": bound == configured,
                    "observed_path": observed.get("path") == str(configured),
                    "sha256": entry.get("sha256") == observed.get("sha256"),
                    "bytes": entry.get("bytes") == observed.get("bytes"),
                    "records": entry.get("records")
                    == observed.get("valid_records")
                    == per_expert,
                    "source_records": bool(
                        isinstance(entry.get("source_records"), int)
                        and not isinstance(entry.get("source_records"), bool)
                        and entry["source_records"] >= per_expert
                    ),
                    "source_bytes": bool(
                        isinstance(entry.get("source_bytes"), int)
                        and not isinstance(entry.get("source_bytes"), bool)
                        and entry["source_bytes"] > 0
                    ),
                    "source_sha256": bool(
                        isinstance(entry.get("source_sha256"), str)
                        and _SHA256_RE.fullmatch(entry["source_sha256"])
                    ),
                }
            )
            if not all(checks.values()):
                errors.append(f"partial snapshot binding failed for {expert}")
            file_checks[expert] = checks
            if (
                safe_basename
                and isinstance(entry.get("sha256"), str)
                and isinstance(entry.get("records"), int)
                and not isinstance(entry.get("records"), bool)
            ):
                digest_parts.append(
                    f"{expert}:{relative}:{entry['sha256']}:{entry['records']}"
                )

        computed_snapshot = (
            hashlib.sha256("\n".join(digest_parts).encode()).hexdigest()
            if len(digest_parts) == len(REQUIRED_EXPERTS)
            else None
        )
        if value.get("snapshot_sha256") != computed_snapshot:
            errors.append("snapshot_sha256 does not match immutable file bindings")
        report.update(
            {
                "manifest_sha256": manifest_sha256,
                "source_partition_manifest_sha256": value.get(
                    "source_partition_manifest_sha256"
                ),
                "declared_snapshot_sha256": value.get("snapshot_sha256"),
                "computed_snapshot_sha256": computed_snapshot,
                "contract_checks": contract_checks,
                "file_checks": file_checks,
            }
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not errors
    return report


def _inspect_formal_v3_split_contract(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    train_complete_chain_count: int | None,
) -> dict[str, Any]:
    """Verify train/calibration/heldout roles without opening heldout bodies."""

    report: dict[str, Any] = {
        "required": True,
        "passed": False,
        "heldout_content_read": False,
        "errors": [],
    }
    errors: list[str] = report["errors"]
    population = manifest.get("population_contract")
    split = manifest.get("split_contract")
    if not isinstance(population, Mapping):
        errors.append("snapshot population_contract is missing")
        return report
    if not isinstance(split, Mapping):
        errors.append("snapshot train/calibration/heldout split_contract is missing")
        return report

    candidate_tasks = _safe_nonnegative(population.get("candidate_tasks_per_stage"))
    candidate_work_orders = _safe_nonnegative(
        population.get("candidate_work_orders")
    )
    gold_accepted = _safe_nonnegative(population.get("gold_accepted_tasks"))
    if (
        candidate_tasks != CANDIDATE_TASKS_PER_STAGE
        or candidate_work_orders != CANDIDATE_WORK_ORDERS
        or population.get("work_orders_per_task") != len(REQUIRED_EXPERTS)
        or gold_accepted is None
        or gold_accepted > CANDIDATE_TASKS_PER_STAGE
    ):
        errors.append("snapshot candidate/Gold population contract is invalid")

    roles = split.get("partitions")
    if (
        split.get("schema_version") != SPLIT_SCHEMA
        or split.get("assignment") != "source_bank_split_then_gold_gate_v1"
        or split.get("pairwise_disjoint") is not True
        or split.get("gold_coverage_complete") is not True
        or split.get("heldout_content_read") is not False
        or split.get("heldout_content_emitted") is not False
        or not isinstance(split.get("leakage_audit_sha256"), str)
        or not _SHA256_RE.fullmatch(str(split.get("leakage_audit_sha256")))
        or not isinstance(roles, Mapping)
        or set(roles) != {"train", "calibration", "heldout"}
    ):
        errors.append("snapshot split proof is invalid")
        return report

    train = roles["train"]
    calibration = roles["calibration"]
    heldout = roles["heldout"]
    if not all(isinstance(item, Mapping) for item in (train, calibration, heldout)):
        errors.append("snapshot split partitions must be objects")
        return report
    assert isinstance(train, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(heldout, Mapping)

    def parse_expert_counts(value: Mapping[str, Any], label: str) -> dict[str, int]:
        raw = value.get("gold_records_per_expert")
        if not isinstance(raw, Mapping) or set(raw) != set(REQUIRED_EXPERTS):
            errors.append(f"snapshot {label} expert counts are invalid")
            return {}
        parsed: dict[str, int] = {}
        for expert in REQUIRED_EXPERTS:
            count = _safe_nonnegative(raw.get(expert))
            if count is None:
                errors.append(f"snapshot {label} count is invalid for {expert}")
                return {}
            parsed[expert] = count
        if len(set(parsed.values())) != 1:
            errors.append(f"snapshot {label} Gold counts must be balanced")
        return parsed

    train_counts = parse_expert_counts(train, "train")
    calibration_counts = parse_expert_counts(calibration, "calibration")
    train_tasks = _safe_nonnegative(train.get("gold_task_count"))
    calibration_tasks = _safe_nonnegative(calibration.get("gold_task_count"))
    if (
        train.get("role") != "training_only"
        or train.get("source_partition") != "train"
        or train.get("candidate_task_count") != 17_105
        or train_tasks is None
        or not train_counts
        or any(value != train_tasks for value in train_counts.values())
        or train_tasks != train_complete_chain_count
        or not isinstance(train.get("ids_sha256"), str)
        or not _SHA256_RE.fullmatch(str(train.get("ids_sha256")))
    ):
        errors.append("snapshot train split binding is invalid")
    if (
        calibration.get("role") != "rank_allocation_only"
        or calibration.get("source_partition") != "validation-from-train"
        or calibration.get("candidate_task_count") != 1_903
        or calibration_tasks is None
        or calibration_tasks < 1
        or not calibration_counts
        or any(value != calibration_tasks for value in calibration_counts.values())
        or not isinstance(calibration.get("ids_sha256"), str)
        or not _SHA256_RE.fullmatch(str(calibration.get("ids_sha256")))
        or not isinstance(calibration.get("snapshot_sha256"), str)
        or not _SHA256_RE.fullmatch(str(calibration.get("snapshot_sha256")))
    ):
        errors.append("snapshot calibration split binding is invalid")

    calibration_files = calibration.get("files")
    calibration_file_checks: dict[str, Any] = {}
    if not isinstance(calibration_files, Mapping) or set(calibration_files) != set(
        REQUIRED_EXPERTS
    ):
        errors.append("snapshot calibration files must map exactly five experts")
    else:
        for expert in REQUIRED_EXPERTS:
            entry = calibration_files[expert]
            checks: dict[str, bool] = {}
            if not isinstance(entry, Mapping):
                errors.append(f"snapshot calibration file is invalid for {expert}")
                calibration_file_checks[expert] = checks
                continue
            relative = entry.get("path")
            safe_relative = bool(
                isinstance(relative, str)
                and relative.replace("\\", "/").startswith("calibration/")
                and ".." not in Path(relative).parts
                and not Path(relative).is_absolute()
            )
            path = (manifest_path.parent / relative).resolve() if safe_relative else None
            inside_snapshot = bool(
                path is not None and path.is_relative_to(manifest_path.parent.resolve())
            )
            exists = bool(path is not None and inside_snapshot and path.is_file())
            checks.update(
                {
                    "safe_relative_path": safe_relative and inside_snapshot,
                    "regular_file": exists and not path.is_symlink() if path else False,
                    "records": entry.get("records") == calibration_tasks,
                    "bytes": bool(
                        exists
                        and isinstance(entry.get("bytes"), int)
                        and not isinstance(entry.get("bytes"), bool)
                        and path.stat().st_size == entry.get("bytes")
                    ),
                    "sha256": bool(
                        exists
                        and isinstance(entry.get("sha256"), str)
                        and _SHA256_RE.fullmatch(entry["sha256"])
                        and sha256_file(path) == entry["sha256"]
                    ),
                }
            )
            if not all(checks.values()):
                errors.append(f"snapshot calibration file binding failed for {expert}")
            calibration_file_checks[expert] = checks

    heldout_valid = bool(
        heldout.get("role") == "evaluation_only_hash_metadata"
        and heldout.get("source_partition") == "external-heldout"
        and heldout.get("content_present") is False
        and heldout.get("content_read") is False
        and heldout.get("content_emitted") is False
        and isinstance(heldout.get("ids_sha256"), str)
        and _SHA256_RE.fullmatch(str(heldout.get("ids_sha256")))
        and isinstance(heldout.get("manifest_sha256"), str)
        and _SHA256_RE.fullmatch(str(heldout.get("manifest_sha256")))
        and "files" not in heldout
    )
    if not heldout_valid:
        errors.append("snapshot heldout must remain hash-only and unread")

    if (
        gold_accepted is not None
        and train_tasks is not None
        and calibration_tasks is not None
        and gold_accepted != train_tasks + calibration_tasks
    ):
        errors.append("snapshot Gold population does not equal train+calibration")

    report.update(
        {
            "population": {
                "candidate_tasks_per_stage": candidate_tasks,
                "candidate_work_orders": candidate_work_orders,
                "gold_accepted_tasks": gold_accepted,
            },
            "train_records_per_expert": train_counts,
            "calibration_records_per_expert": calibration_counts,
            "calibration_snapshot_sha256": calibration.get("snapshot_sha256"),
            "calibration_file_checks": calibration_file_checks,
            "heldout_hash_only": heldout_valid,
            "heldout_ids_sha256": heldout.get("ids_sha256"),
            "heldout_manifest_sha256": heldout.get("manifest_sha256"),
            "leakage_audit_sha256": split.get("leakage_audit_sha256"),
        }
    )
    report["passed"] = not errors
    return report


def inspect_dataset_snapshot_manifest(
    config: Mapping[str, Any], root: Path, datasets: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the immutable formal-v3 manifest, sidecar, and every file binding."""

    snapshot_config = config["scale_gate"].get("dataset_snapshot")
    if snapshot_config is None:
        return {"required": False, "passed": True}
    if snapshot_config.get("schema_version") == PARTIAL_SNAPSHOT_SCHEMA:
        return _inspect_partial_dataset_snapshot_manifest(config, root, datasets)

    manifest_path = (root / snapshot_config["manifest"]).resolve()
    sidecar_path = (root / snapshot_config["sidecar"]).resolve()
    report: dict[str, Any] = {
        "required": True,
        "passed": False,
        "manifest_path": str(manifest_path),
        "sidecar_path": str(sidecar_path),
        "errors": [],
    }
    errors: list[str] = report["errors"]
    if sidecar_path != Path(str(manifest_path) + ".sha256"):
        errors.append(
            "snapshot sidecar must be manifest.json.sha256 beside the manifest"
        )
    if not manifest_path.is_file():
        errors.append("immutable snapshot manifest is missing")
    if not sidecar_path.is_file():
        errors.append("immutable snapshot SHA-256 sidecar is missing")
    if errors:
        return report

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar_tokens = sidecar_path.read_text(encoding="ascii").split()
        if not sidecar_tokens or sidecar_tokens[0] != manifest_sha256:
            errors.append("snapshot manifest SHA-256 sidecar mismatch")
        value = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(value, Mapping):
            errors.append("snapshot manifest must be a JSON object")
            return report
        if value.get("schema_version") != snapshot_config["schema_version"]:
            errors.append("snapshot manifest schema mismatch")
        source_sha = value.get("source_partition_manifest_sha256")
        if not isinstance(source_sha, str) or not _SHA256_RE.fullmatch(source_sha):
            errors.append("source partition manifest SHA-256 is missing or invalid")

        minimum = int(snapshot_config["minimum_records_per_expert"])
        source_gate = value.get("source_gate")
        source_gold_files: Mapping[str, Any] = {}
        source_task_bank_file: Mapping[str, Any] = {}
        source_complete_chain_count: int | None = None
        if not isinstance(source_gate, Mapping):
            errors.append("snapshot source_gate lineage proof is missing")
        else:
            complete_chain_count = source_gate.get("complete_chain_count")
            source_complete_chain_count = _safe_nonnegative(complete_chain_count)
            minimum_complete_chain_count = source_gate.get(
                "minimum_complete_chain_count"
            )
            valid_chain_count = (
                isinstance(complete_chain_count, int)
                and not isinstance(complete_chain_count, bool)
                and complete_chain_count >= minimum
            )
            valid_chain_minimum = (
                isinstance(minimum_complete_chain_count, int)
                and not isinstance(minimum_complete_chain_count, bool)
                and minimum_complete_chain_count == minimum
            )
            lineage_edge_error_count = source_gate.get("lineage_edge_error_count")
            lineage_chain_error_count = source_gate.get("lineage_chain_error_count")
            zero_lineage_errors = all(
                isinstance(count, int) and not isinstance(count, bool) and count == 0
                for count in (
                    lineage_edge_error_count,
                    lineage_chain_error_count,
                )
            )
            if not (
                source_gate.get("lineage_complete") is True
                and source_gate.get("complete_chain_count_sufficient") is True
                and valid_chain_count
                and valid_chain_minimum
                and zero_lineage_errors
            ):
                errors.append("snapshot source_gate lineage proof is invalid")
            raw_source_gold_files = source_gate.get("gold_files")
            if not isinstance(raw_source_gold_files, Mapping) or set(
                raw_source_gold_files
            ) != set(EXPERT_TASKS.values()):
                errors.append("snapshot source_gate gold file bindings are invalid")
            else:
                source_gold_files = raw_source_gold_files
            near_duplicate_gate = source_gate.get("near_duplicate_gate")
            if (
                not isinstance(near_duplicate_gate, Mapping)
                or near_duplicate_gate.get("passed") is not True
            ):
                errors.append("snapshot source_gate near-duplicate proof is invalid")
            if not _task_card_coverage_valid(
                source_gate.get("task_card_coverage"),
                complete_chain_count=source_complete_chain_count,
            ):
                errors.append(
                    "snapshot source_gate task-card coverage proof is invalid"
                )
            raw_source_task_bank = source_gate.get("task_bank_file")
            if not _task_bank_source_binding_valid(
                raw_source_task_bank,
                complete_chain_count=source_complete_chain_count,
            ):
                errors.append("snapshot source_gate task bank binding is invalid")
            else:
                assert isinstance(raw_source_task_bank, Mapping)
                source_task_bank_file = raw_source_task_bank

        if str(config.get("experiment", "")).startswith(
            "anchor-moe-lora-formal-v3"
        ):
            split_report = _inspect_formal_v3_split_contract(
                value,
                manifest_path=manifest_path,
                train_complete_chain_count=source_complete_chain_count,
            )
            if not split_report["passed"]:
                errors.extend(split_report["errors"])
        else:
            split_report = {
                "required": False,
                "passed": True,
                "heldout_content_read": False,
                "errors": [],
            }

        files = value.get("files")
        if not isinstance(files, Mapping) or set(files) != set(REQUIRED_EXPERTS):
            errors.append("snapshot manifest files must map exactly the five experts")
            files = {}

        digest_parts: list[str] = []
        file_checks: dict[str, Any] = {}
        for expert in REQUIRED_EXPERTS:
            entry = files.get(expert)
            observed = datasets["reports"].get(expert, {})
            checks: dict[str, bool] = {}
            if not isinstance(entry, Mapping):
                errors.append(f"snapshot manifest is missing files.{expert}")
                file_checks[expert] = checks
                continue
            relative = entry.get("path")
            safe_basename = (
                isinstance(relative, str)
                and bool(relative)
                and Path(relative).name == relative
            )
            checks["safe_basename"] = safe_basename
            configured = (
                root / config["scale_gate"]["required_datasets"][expert]
            ).resolve()
            bound = (
                (manifest_path.parent / relative).resolve() if safe_basename else None
            )
            checks["configured_path"] = bound == configured
            checks["observed_path"] = observed.get("path") == str(configured)
            checks["sha256"] = entry.get("sha256") == observed.get("sha256")
            checks["bytes"] = entry.get("bytes") == observed.get("bytes")
            checks["records"] = entry.get("records") == observed.get("valid_records")
            checks["minimum_records"] = (
                isinstance(entry.get("records"), int)
                and not isinstance(entry.get("records"), bool)
                and entry["records"] >= minimum
            )
            checks["source_sha256"] = isinstance(
                entry.get("source_sha256"), str
            ) and bool(_SHA256_RE.fullmatch(entry["source_sha256"]))
            source_binding = source_gold_files.get(EXPERT_TASKS[expert])
            checks["source_gate_gold_binding"] = bool(
                isinstance(source_binding, Mapping)
                and set(source_binding) == {"path", "records", "bytes", "sha256"}
                and source_binding.get("path") == relative
                and source_binding.get("records") == entry.get("records")
                and source_binding.get("bytes") == entry.get("bytes")
                and source_binding.get("sha256") == entry.get("source_sha256")
                and entry.get("source_sha256") == entry.get("sha256")
            )
            if not all(checks.values()):
                errors.append(f"snapshot manifest binding failed for {expert}")
            file_checks[expert] = checks
            if (
                safe_basename
                and isinstance(entry.get("sha256"), str)
                and isinstance(entry.get("records"), int)
            ):
                digest_parts.append(
                    f"{expert}:{relative}:{entry['sha256']}:{entry['records']}"
                )

        task_bank_entry = value.get("task_bank_file")
        task_bank_checks: dict[str, bool] = {}
        if not isinstance(task_bank_entry, Mapping):
            errors.append("snapshot manifest task bank binding is missing")
        else:
            relative = task_bank_entry.get("path")
            safe_basename = (
                relative == TASK_BANK_FILENAME
                and isinstance(relative, str)
                and Path(relative).name == relative
            )
            task_bank_checks["safe_basename"] = safe_basename
            task_bank_path = manifest_path.parent / TASK_BANK_FILENAME
            task_bank_checks["regular_file"] = bool(
                task_bank_path.is_file() and not task_bank_path.is_symlink()
            )
            task_bank_checks["schema"] = set(task_bank_entry) == {
                "path",
                "records",
                "bytes",
                "sha256",
                "source_sha256",
            }
            observed_before: dict[str, Any] | None = None
            observed_after: dict[str, Any] | None = None
            if task_bank_checks["regular_file"]:
                try:
                    observed_before = _task_bank_binding(task_bank_path)
                    observed_after = _task_bank_binding(task_bank_path)
                except (
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                ):
                    observed_before = None
                    observed_after = None
            task_bank_checks["readable"] = observed_before is not None
            task_bank_checks["unchanged_during_read"] = bool(
                observed_before is not None and observed_before == observed_after
            )
            declared_copy_binding = {
                "path": task_bank_entry.get("path"),
                "records": task_bank_entry.get("records"),
                "bytes": task_bank_entry.get("bytes"),
                "sha256": task_bank_entry.get("sha256"),
            }
            task_bank_checks["copy_binding"] = bool(
                observed_after is not None and observed_after == declared_copy_binding
            )
            task_bank_checks["source_sha256"] = bool(
                isinstance(task_bank_entry.get("source_sha256"), str)
                and _SHA256_RE.fullmatch(task_bank_entry["source_sha256"])
                and task_bank_entry.get("source_sha256")
                == task_bank_entry.get("sha256")
            )
            task_bank_checks["source_gate_binding"] = bool(
                source_task_bank_file
                and dict(source_task_bank_file)
                == {
                    "path": TASK_BANK_FILENAME,
                    "records": task_bank_entry.get("records"),
                    "bytes": task_bank_entry.get("bytes"),
                    "sha256": task_bank_entry.get("source_sha256"),
                }
            )
            task_bank_checks["cardinality"] = bool(
                _safe_nonnegative(task_bank_entry.get("records"))
                == source_complete_chain_count
            )
            if not all(task_bank_checks.values()):
                errors.append("snapshot task bank binding failed")
            if (
                safe_basename
                and isinstance(task_bank_entry.get("sha256"), str)
                and isinstance(task_bank_entry.get("records"), int)
                and not isinstance(task_bank_entry.get("records"), bool)
            ):
                digest_parts.append(
                    "task_bank:"
                    f"{relative}:"
                    f"{task_bank_entry['sha256']}:"
                    f"{task_bank_entry['records']}"
                )

        computed_snapshot = (
            hashlib.sha256("\n".join(digest_parts).encode()).hexdigest()
            if len(digest_parts) == len(REQUIRED_EXPERTS) + 1
            else None
        )
        if value.get("snapshot_sha256") != computed_snapshot:
            errors.append("snapshot_sha256 does not match immutable file bindings")
        report.update(
            {
                "manifest_sha256": manifest_sha256,
                "source_partition_manifest_sha256": source_sha,
                "declared_snapshot_sha256": value.get("snapshot_sha256"),
                "computed_snapshot_sha256": computed_snapshot,
                "file_checks": file_checks,
                "task_bank_checks": task_bank_checks,
                "split_contract": split_report,
            }
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not errors
    return report


def inspect_base_artifact(
    config: Mapping[str, Any], root: Path, *, deep_checksum: bool = False
) -> dict[str, Any]:
    expected = config["scale_gate"]["base_artifact"]
    local_dir = (root / expected["local_path"]).resolve()
    manifest_path = local_dir / expected["download_manifest"]
    weight_path = local_dir / expected["weight_file"]
    report: dict[str, Any] = {
        "local_path": str(local_dir),
        "manifest_path": str(manifest_path),
        "weight_path": str(weight_path),
        "deep_checksum": deep_checksum,
    }
    if not manifest_path.is_file() or not weight_path.is_file():
        report.update(
            {"passed": False, "error": "base manifest or weight file is missing"}
        )
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = manifest.get("verification", {})
        observed_size = weight_path.stat().st_size
        expected_sha = expected["sha256"]
        manifest_sha = verification.get("sha256")
        observed_sha = sha256_file(weight_path) if deep_checksum else manifest_sha
        checks = {
            "repo_id": manifest.get("repo_id")
            == config["model"]["id"]
            == expected["repo_id"],
            "revision": manifest.get("revision")
            == config["model"]["revision"]
            == expected["revision"],
            "manifest_sha256": manifest_sha == expected_sha,
            "observed_sha256": observed_sha == expected_sha,
            "bytes": observed_size == verification.get("bytes") == expected["bytes"],
            "lfs_oid_verified": verification.get("matches_hugging_face_lfs_oid")
            is True,
        }
        report.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "sha256": observed_sha,
                "checksum_source": "deep-file-hash"
                if deep_checksum
                else "verified-download-manifest",
                "bytes": observed_size,
                "revision": manifest.get("revision"),
                "repo_id": manifest.get("repo_id"),
            }
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return report


def inspect_training_artifact(
    config: Mapping[str, Any], root: Path, *, deep_checksum: bool = False
) -> dict[str, Any]:
    """Verify the actual reloadable bitsandbytes NF4 directory used by PEFT."""

    expected = config["scale_gate"]["training_artifact"]
    local_dir = (root / expected["local_path"]).resolve()
    manifest_path = (root / expected["manifest"]).resolve()
    config_path = local_dir / "config.json"
    index_path = local_dir / "model.safetensors.index.json"
    report: dict[str, Any] = {
        "passed": False,
        "local_path": str(local_dir),
        "manifest_path": str(manifest_path),
        "deep_checksum": deep_checksum,
        "errors": [],
    }
    errors: list[str] = report["errors"]
    if local_dir != (root / config["model"]["local_path"]).resolve():
        errors.append("training artifact local path does not match model.local_path")
    if manifest_path.parent != local_dir:
        errors.append(
            "NF4 export manifest must be inside the training artifact directory"
        )
    for required in (manifest_path, config_path, index_path):
        if not required.is_file():
            errors.append(f"required NF4 artifact file is missing: {required.name}")
    if errors:
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not all(
            isinstance(item, Mapping) for item in (manifest, model_config, index)
        ):
            errors.append("NF4 manifest, config, and index must be JSON objects")
            return report

        quant = manifest.get("quantization")
        model_quant = model_config.get("quantization_config")
        checks = {
            "schema": manifest.get("schema_version") == "anchor.bnb-nf4-export.v1",
            "format": expected.get("format") == "transformers-bitsandbytes-nf4",
            "source_path": str(manifest.get("source", "")).replace("\\", "/")
            == str(config["scale_gate"]["base_artifact"]["local_path"]).replace(
                "\\", "/"
            ),
            "source_weight_sha256": manifest.get("source_weight_sha256")
            == config["scale_gate"]["base_artifact"]["sha256"],
            "model_footprint_bytes": manifest.get("model_footprint_bytes")
            == expected.get("model_footprint_bytes"),
            "quantization_manifest": isinstance(quant, Mapping)
            and quant.get("type") == config["quantization"]["quant_type"] == "nf4"
            and quant.get("double_quant") is config["quantization"]["double_quant"]
            and quant.get("compute_dtype") == config["quantization"]["compute_dtype"]
            and quant.get("storage_dtype")
            == config["quantization"]["quant_storage_dtype"],
            "transformers_config": isinstance(model_quant, Mapping)
            and model_quant.get("quant_method") == "bitsandbytes"
            and model_quant.get("load_in_4bit") is True
            and model_quant.get("load_in_8bit") is False
            and model_quant.get("bnb_4bit_quant_type") == "nf4"
            and model_quant.get("bnb_4bit_use_double_quant") is True
            and model_quant.get("bnb_4bit_compute_dtype") == "bfloat16"
            and model_quant.get("bnb_4bit_quant_storage") == "bfloat16"
            and model_quant.get("llm_int8_enable_fp32_cpu_offload") is False,
            "frozen_peft_contract": config["quantization"]["freeze_base_model"] is True
            and config["model"]["training_format"] == "transformers_or_peft_4bit"
            and config["model"]["load_strategy"] == "prequantized_peft_4bit",
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(f"NF4 training artifact contract failed: {name}")

        weights = manifest.get("weights")
        weight_reports: list[dict[str, Any]] = []
        declared_names: set[str] = set()
        declared_total = 0
        if not isinstance(weights, list) or not weights:
            errors.append("NF4 manifest must bind at least one safetensors shard")
            weights = []
        for position, entry in enumerate(weights):
            if not isinstance(entry, Mapping):
                errors.append(f"NF4 weights[{position}] must be an object")
                continue
            name, declared_bytes, declared_sha = (
                entry.get("path"),
                entry.get("bytes"),
                entry.get("sha256"),
            )
            safe_name = (
                isinstance(name, str)
                and Path(name).name == name
                and name.endswith(".safetensors")
            )
            shard = local_dir / name if safe_name else local_dir / "<invalid>"
            exists = safe_name and shard.is_file()
            size_matches = (
                exists
                and isinstance(declared_bytes, int)
                and not isinstance(declared_bytes, bool)
                and shard.stat().st_size == declared_bytes
            )
            sha_shape = isinstance(declared_sha, str) and bool(
                _SHA256_RE.fullmatch(declared_sha)
            )
            sha_matches = bool(
                not deep_checksum
                or (exists and sha_shape and sha256_file(shard) == declared_sha)
            )
            if not all((safe_name, exists, size_matches, sha_shape, sha_matches)):
                errors.append(f"NF4 weight binding failed at index {position}")
            if safe_name:
                declared_names.add(name)
            if isinstance(declared_bytes, int) and not isinstance(declared_bytes, bool):
                declared_total += declared_bytes
            weight_reports.append(
                {
                    "path": name,
                    "exists": exists,
                    "bytes_match": size_matches,
                    "sha256_shape": sha_shape,
                    "sha256_verified": deep_checksum and sha_matches,
                }
            )

        weight_map = index.get("weight_map")
        index_names = (
            set(weight_map.values()) if isinstance(weight_map, Mapping) else set()
        )
        metadata = index.get("metadata")
        index_total = (
            metadata.get("total_size") if isinstance(metadata, Mapping) else None
        )
        quant_state_present = bool(
            isinstance(weight_map, Mapping)
            and any("quant_state.bitsandbytes__nf4" in str(key) for key in weight_map)
        )
        index_checks = {
            "shards_exact": bool(index_names) and index_names == declared_names,
            # Safetensors index total_size is tensor payload bytes; shard file
            # bytes also include per-file headers. Bind it to a tight plausible
            # range instead of incorrectly requiring byte-for-byte equality.
            "total_size_plausible": isinstance(index_total, int)
            and not isinstance(index_total, bool)
            and 0 < index_total <= declared_total
            and declared_total - index_total
            <= max(16 * 1024 * 1024, declared_total // 100),
            "nf4_quant_state": quant_state_present,
        }
        for name, passed in index_checks.items():
            if not passed:
                errors.append(f"NF4 safetensors index contract failed: {name}")
        report.update(
            {
                "checks": checks,
                "index_checks": index_checks,
                "weights": weight_reports,
                "weight_bytes": declared_total,
                "checksum_source": (
                    "deep-file-hash" if deep_checksum else "manifest-and-file-size"
                ),
            }
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not errors
    return report


def load_heldout_cases(
    config: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (root / config["scale_gate"]["heldout_cases"]).resolve()
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return cases, {
            "path": str(path),
            "passed": False,
            "errors": ["file is missing"],
        }
    seen: set[str] = set()
    try:
        for line_number, value in iter_jsonl(path):
            source = f"{path}:{line_number}"
            if not isinstance(value, Mapping):
                errors.append(f"{source}: case must be an object")
                continue
            identifier, expert, prompt = (
                value.get("id"),
                value.get("expert"),
                value.get("prompt"),
            )
            if (
                not isinstance(identifier, str)
                or not identifier.strip()
                or identifier in seen
            ):
                errors.append(f"{source}: id must be unique non-empty text")
                continue
            seen.add(identifier)
            if expert not in REQUIRED_EXPERTS:
                errors.append(f"{source}: invalid expert")
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"{source}: prompt must be non-empty text")
                continue
            max_new_tokens = value.get("max_new_tokens", 16)
            if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 64:
                errors.append(f"{source}: max_new_tokens must be in [1, 64]")
                continue
            cases.append(
                {
                    "id": identifier,
                    "expert": expert,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                }
            )
    except (DatasetValidationError, OSError) as exc:
        errors.append(str(exc))
    covered = {case["expert"] for case in cases}
    if covered != set(REQUIRED_EXPERTS):
        errors.append(
            f"held-out cases must cover all experts; covered={sorted(covered)}"
        )
    return cases, {
        "path": str(path),
        "passed": not errors,
        "case_count": len(cases),
        "covered_experts": sorted(covered),
        "sha256": sha256_file(path) if path.is_file() else None,
        "errors": errors,
    }


def build_preflight_report(
    config: Mapping[str, Any],
    root: Path,
    dependencies: Mapping[str, Any],
    *,
    deep_checksum: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets = inspect_gate_datasets(config, root)
    snapshot = inspect_dataset_snapshot_manifest(config, root, datasets)
    base = inspect_base_artifact(config, root, deep_checksum=deep_checksum)
    training_artifact = inspect_training_artifact(
        config, root, deep_checksum=deep_checksum
    )
    heldout_cases, heldout = load_heldout_cases(config, root)
    device = dependencies.get("device", {})
    minimum_free = float(config["scale_gate"]["minimum_free_vram_gib"])
    free_vram = device.get("free_memory_gib")
    gpu_passed = bool(
        device.get("cuda_available")
        and device.get("bf16_supported")
        and isinstance(free_vram, (int, float))
        and free_vram >= minimum_free
    )
    host_memory = dependencies.get("host_memory", {})
    minimum_free_host = float(config["scale_gate"]["minimum_free_host_memory_gib"])
    free_host_memory = host_memory.get("available_memory_gib")
    host_memory_passed = bool(
        host_memory.get("probed")
        and isinstance(free_host_memory, (int, float))
        and free_host_memory >= minimum_free_host
    )
    gates = {
        "five_live_datasets_present": _gate(
            datasets["complete"], reports=datasets["reports"]
        ),
        "canonical_schema_valid": _gate(datasets["schemas_valid"]),
        "real_teacher_samples": _gate(datasets["all_records_live"]),
        "assistant_targets_nonempty": _gate(datasets["assistant_targets_nonempty"]),
        "dataset_ids_unique": _gate(
            datasets["cross_file_duplicate_ids"] == 0,
            duplicates=datasets["cross_file_duplicate_ids"],
        ),
        "base_revision_and_checksum": _gate(bool(base.get("passed")), report=base),
        "training_nf4_artifact": _gate(
            bool(training_artifact.get("passed")), report=training_artifact
        ),
        "training_dependencies": _gate(
            bool(dependencies.get("ready")),
            missing=dependencies.get("missing", []),
            incompatible=dependencies.get("incompatible", []),
        ),
        "gpu_free_vram": _gate(
            gpu_passed,
            free_gib=free_vram,
            required_gib=minimum_free,
            device=device.get("name"),
        ),
        "host_free_memory": _gate(
            host_memory_passed,
            free_gib=free_host_memory,
            required_gib=minimum_free_host,
            total_gib=host_memory.get("total_memory_gib"),
            probe_error=host_memory.get("probe_error"),
        ),
        "heldout_cases": _gate(bool(heldout.get("passed")), report=heldout),
    }
    if snapshot["required"]:
        gates["immutable_dataset_snapshot"] = _gate(
            bool(snapshot.get("passed")), report=snapshot
        )
    passed = all(item["passed"] for item in gates.values())
    dataset_snapshot_sha256 = (
        snapshot.get("computed_snapshot_sha256")
        if snapshot["required"]
        else datasets["snapshot_sha256"]
    )
    return (
        {
            "passed": passed,
            "gates": gates,
            "dataset_snapshot_sha256": dataset_snapshot_sha256,
            "dataset_snapshot_manifest": snapshot,
            "base": base,
            "training_artifact": training_artifact,
            "host_memory": host_memory,
            "heldout": heldout,
        },
        heldout_cases,
    )


def verify_prior_smoke_gate(
    config: Mapping[str, Any], root: Path, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    relative = config["scale_gate"]["required_smoke_gate_manifest"]
    path = (root / relative).resolve()
    if not path.is_file():
        return {
            "passed": False,
            "path": str(path),
            "error": "executed smoke-gate manifest is missing",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "stage": value.get("stage") == "smoke-gate",
            "mode": value.get("mode") == "execute",
            "gate_passed": value.get("smoke_gate", {}).get("passed") is True,
            "base_revision": value.get("base_model_revision")
            == config["model"]["revision"],
            "dataset_snapshot": value.get("preflight", {}).get(
                "dataset_snapshot_sha256"
            )
            == preflight.get("dataset_snapshot_sha256"),
        }
        return {"passed": all(checks.values()), "path": str(path), "checks": checks}
    except (OSError, ValueError, TypeError) as exc:
        return {
            "passed": False,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
