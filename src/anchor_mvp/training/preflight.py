"""Read-only scale-up gates for data, base artifacts, GPU capacity, and probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .manifest import sha256_file
from .schema import DatasetValidationError, iter_jsonl, validate_jsonl


REQUIRED_EXPERTS = ("frontend_gen", "code_review", "security_audit")


def _gate(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def _dataset_stats(path: Path, expected_expert: str) -> dict[str, Any]:
    validation = validate_jsonl(path, allowed_experts=[expected_expert])
    assistant_lengths: list[int] = []
    identifiers: list[str] = []
    live_records = 0
    for _, record in iter_jsonl(path):
        identifiers.append(str(record["id"]))
        assistant_lengths.append(len(str(record["messages"][-1]["content"]).strip()))
        provenance = record.get("provenance", {})
        teacher = provenance.get("teacher", {}) if isinstance(provenance, Mapping) else {}
        model = str(teacher.get("model", "")).strip().casefold() if isinstance(teacher, Mapping) else ""
        base_url = str(teacher.get("base_url", "")).strip().casefold() if isinstance(teacher, Mapping) else ""
        if model and model not in {"mock", "mock-teacher", "fixture"} and not base_url.startswith("mock:"):
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
        path = (root / relative).resolve() if isinstance(relative, str) else root / "<missing>"
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
            reports[expert] = {"path": str(path), "exists": True, "valid": True, **stats}
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
    live = schemas_valid and all(report.get("all_records_live") for report in reports.values())
    nonempty = schemas_valid and all(
        report.get("assistant_target_chars", {}).get("empty") == 0 for report in reports.values()
    )
    digest_parts = [
        f"{expert}:{reports[expert].get('sha256', 'missing')}" for expert in REQUIRED_EXPERTS
    ]
    snapshot_sha256 = hashlib.sha256("\n".join(digest_parts).encode("utf-8")).hexdigest()
    return {
        "reports": reports,
        "snapshot_sha256": snapshot_sha256,
        "complete": complete,
        "schemas_valid": schemas_valid,
        "all_records_live": live,
        "assistant_targets_nonempty": nonempty,
        "cross_file_duplicate_ids": duplicate_count,
    }


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
        report.update({"passed": False, "error": "base manifest or weight file is missing"})
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = manifest.get("verification", {})
        observed_size = weight_path.stat().st_size
        expected_sha = expected["sha256"]
        manifest_sha = verification.get("sha256")
        observed_sha = sha256_file(weight_path) if deep_checksum else manifest_sha
        checks = {
            "repo_id": manifest.get("repo_id") == config["model"]["id"] == expected["repo_id"],
            "revision": manifest.get("revision") == config["model"]["revision"] == expected["revision"],
            "manifest_sha256": manifest_sha == expected_sha,
            "observed_sha256": observed_sha == expected_sha,
            "bytes": observed_size == verification.get("bytes") == expected["bytes"],
            "lfs_oid_verified": verification.get("matches_hugging_face_lfs_oid") is True,
        }
        report.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "sha256": observed_sha,
                "checksum_source": "deep-file-hash" if deep_checksum else "verified-download-manifest",
                "bytes": observed_size,
                "revision": manifest.get("revision"),
                "repo_id": manifest.get("repo_id"),
            }
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return report


def load_heldout_cases(config: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (root / config["scale_gate"]["heldout_cases"]).resolve()
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return cases, {"path": str(path), "passed": False, "errors": ["file is missing"]}
    seen: set[str] = set()
    try:
        for line_number, value in iter_jsonl(path):
            source = f"{path}:{line_number}"
            if not isinstance(value, Mapping):
                errors.append(f"{source}: case must be an object")
                continue
            identifier, expert, prompt = value.get("id"), value.get("expert"), value.get("prompt")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
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
        errors.append(f"held-out cases must cover all experts; covered={sorted(covered)}")
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
    base = inspect_base_artifact(config, root, deep_checksum=deep_checksum)
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
        "three_live_datasets_present": _gate(datasets["complete"], reports=datasets["reports"]),
        "canonical_schema_valid": _gate(datasets["schemas_valid"]),
        "real_teacher_samples": _gate(datasets["all_records_live"]),
        "assistant_targets_nonempty": _gate(datasets["assistant_targets_nonempty"]),
        "dataset_ids_unique": _gate(datasets["cross_file_duplicate_ids"] == 0, duplicates=datasets["cross_file_duplicate_ids"]),
        "base_revision_and_checksum": _gate(bool(base.get("passed")), report=base),
        "training_dependencies": _gate(bool(dependencies.get("ready")), missing=dependencies.get("missing", []), incompatible=dependencies.get("incompatible", [])),
        "gpu_free_vram": _gate(gpu_passed, free_gib=free_vram, required_gib=minimum_free, device=device.get("name")),
        "host_free_memory": _gate(
            host_memory_passed,
            free_gib=free_host_memory,
            required_gib=minimum_free_host,
            total_gib=host_memory.get("total_memory_gib"),
            probe_error=host_memory.get("probe_error"),
        ),
        "heldout_cases": _gate(bool(heldout.get("passed")), report=heldout),
    }
    passed = all(item["passed"] for item in gates.values())
    return (
        {
            "passed": passed,
            "gates": gates,
            "dataset_snapshot_sha256": datasets["snapshot_sha256"],
            "base": base,
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
        return {"passed": False, "path": str(path), "error": "executed smoke-gate manifest is missing"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "stage": value.get("stage") == "smoke-gate",
            "mode": value.get("mode") == "execute",
            "gate_passed": value.get("smoke_gate", {}).get("passed") is True,
            "base_revision": value.get("base_model_revision") == config["model"]["revision"],
            "dataset_snapshot": value.get("preflight", {}).get("dataset_snapshot_sha256")
            == preflight.get("dataset_snapshot_sha256"),
        }
        return {"passed": all(checks.values()), "path": str(path), "checks": checks}
    except (OSError, ValueError, TypeError) as exc:
        return {"passed": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
