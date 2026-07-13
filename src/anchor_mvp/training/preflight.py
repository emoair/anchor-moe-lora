"""Read-only scale-up gates for data, base artifacts, GPU capacity, and probes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .manifest import sha256_file
from .schema import DatasetValidationError, iter_jsonl, validate_jsonl


REQUIRED_EXPERTS = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def inspect_dataset_snapshot_manifest(
    config: Mapping[str, Any], root: Path, datasets: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the immutable formal-v3 manifest, sidecar, and every file binding."""

    snapshot_config = config["scale_gate"].get("dataset_snapshot")
    if snapshot_config is None:
        return {"required": False, "passed": True}

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
        errors.append("snapshot sidecar must be manifest.json.sha256 beside the manifest")
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

        files = value.get("files")
        if not isinstance(files, Mapping) or set(files) != set(REQUIRED_EXPERTS):
            errors.append("snapshot manifest files must map exactly the five experts")
            files = {}

        minimum = int(snapshot_config["minimum_records_per_expert"])
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
            configured = (root / config["scale_gate"]["required_datasets"][expert]).resolve()
            bound = (manifest_path.parent / relative).resolve() if safe_basename else None
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
                "source_partition_manifest_sha256": source_sha,
                "declared_snapshot_sha256": value.get("snapshot_sha256"),
                "computed_snapshot_sha256": computed_snapshot,
                "file_checks": file_checks,
            }
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
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
        errors.append("NF4 export manifest must be inside the training artifact directory")
    for required in (manifest_path, config_path, index_path):
        if not required.is_file():
            errors.append(f"required NF4 artifact file is missing: {required.name}")
    if errors:
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not all(isinstance(item, Mapping) for item in (manifest, model_config, index)):
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
            and quant.get("compute_dtype")
            == config["quantization"]["compute_dtype"]
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
            and config["model"]["training_format"]
            == "transformers_or_peft_4bit"
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
                not deep_checksum or (exists and sha_shape and sha256_file(shard) == declared_sha)
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
        index_names = set(weight_map.values()) if isinstance(weight_map, Mapping) else set()
        metadata = index.get("metadata")
        index_total = metadata.get("total_size") if isinstance(metadata, Mapping) else None
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
            and declared_total - index_total <= max(16 * 1024 * 1024, declared_total // 100),
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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not errors
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
        "five_live_datasets_present": _gate(datasets["complete"], reports=datasets["reports"]),
        "canonical_schema_valid": _gate(datasets["schemas_valid"]),
        "real_teacher_samples": _gate(datasets["all_records_live"]),
        "assistant_targets_nonempty": _gate(datasets["assistant_targets_nonempty"]),
        "dataset_ids_unique": _gate(datasets["cross_file_duplicate_ids"] == 0, duplicates=datasets["cross_file_duplicate_ids"]),
        "base_revision_and_checksum": _gate(bool(base.get("passed")), report=base),
        "training_nf4_artifact": _gate(
            bool(training_artifact.get("passed")), report=training_artifact
        ),
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
