"""Immutable A--F experiment-run registries.

The registry is deliberately separate from the trainer.  It can index completed
legacy artifacts without moving them, and it gives future runs exclusive,
versioned output roots.  Registry files are created with ``O_EXCL`` semantics;
there is no update-in-place path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import sha256_file


GROUPS = ("A", "B", "C", "D", "E", "F")
SPECIALISTS = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
GROUP_SCHEMA = "anchor.af-group-registry.v1"
RUN_SCHEMA = "anchor.af-run-manifest.v1"
LAYOUTS = ("legacy-layout-v1", "versioned-layout-v2")
BUDGET_PARAMETERS = 10_387_456
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExperimentRegistryError(ValueError):
    """Raised when an immutable run cannot be safely registered."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentRegistryError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentRegistryError(f"{label} must be a JSON object: {path}")
    return value


def _sha(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ExperimentRegistryError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ExperimentRegistryError("run_id is not a safe version identifier")
    return run_id


def _inside(root: Path, candidate: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ExperimentRegistryError(
            f"{label} escapes project root: {resolved}"
        ) from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise ExperimentRegistryError(
            f"immutable registry already exists; refusing overwrite: {path}"
        ) from exc
    return path


def _base_binding(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_path = _inside(project_root, manifest_path, "base manifest")
    value = _load_json(manifest_path, "base artifact manifest")
    if value.get("schema_version") != "anchor.bnb-nf4-export.v1":
        raise ExperimentRegistryError("base artifact must be the audited NF4 export")
    source_sha = _require_sha(value.get("source_weight_sha256"), "base source digest")
    weights = value.get("weights")
    if not isinstance(weights, list) or not weights:
        raise ExperimentRegistryError("base manifest has no weight shard inventory")
    for index, item in enumerate(weights):
        if not isinstance(item, Mapping):
            raise ExperimentRegistryError(f"base weight {index} must be an object")
        _require_sha(item.get("sha256"), f"base weight {index} digest")
    return {
        "format": "transformers-bitsandbytes-nf4",
        "manifest_path": _relative(project_root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_weight_sha256": source_sha,
        "weight_set_sha256": _sha(
            [
                {
                    "path": item.get("path"),
                    "bytes": item.get("bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in weights
            ]
        ),
    }


def _dataset_binding(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_path = _inside(project_root, manifest_path, "dataset manifest")
    value = _load_json(manifest_path, "dataset snapshot manifest")
    snapshot_sha = _require_sha(value.get("snapshot_sha256"), "dataset snapshot hash")
    if value.get("not_for_end_to_end_claim") is not True:
        raise ExperimentRegistryError(
            "this partial snapshot must retain not_for_end_to_end_claim=true"
        )
    return {
        "schema_version": value.get("schema_version"),
        "manifest_path": _relative(project_root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "snapshot_sha256": snapshot_sha,
        "not_for_end_to_end_claim": True,
    }


def initialize_run(
    project_root: str | Path,
    registry_parent: str | Path,
    *,
    run_id: str,
    profile: str,
    layout: str,
    base_manifest: str | Path,
    dataset_manifest: str | Path,
    legacy_artifact_parent: str | Path | None = None,
    group_artifact_roots: Mapping[str, str | Path] | None = None,
) -> Path:
    """Create a six-arm run namespace and immutable common contract.

    ``legacy-layout-v1`` keeps artifacts in the pre-existing sibling ``A``--``F``
    roots and stores only read-only indexes below the registry.  New runs should
    use ``versioned-layout-v2``, where every output lives below ``runs/<run_id>``.
    """

    root = Path(project_root).resolve()
    run_id = _safe_run_id(run_id)
    if layout not in LAYOUTS:
        raise ExperimentRegistryError(f"layout must be one of {LAYOUTS}")
    if not isinstance(profile, str) or not profile.strip():
        raise ExperimentRegistryError("profile must be non-empty")
    parent = _inside(root, Path(registry_parent), "registry parent")
    run_root = _inside(root, parent / run_id, "run root")
    if layout == "legacy-layout-v1":
        if legacy_artifact_parent is None:
            raise ExperimentRegistryError(
                "legacy-layout-v1 requires legacy_artifact_parent"
            )
        artifact_parent = _inside(
            root, Path(legacy_artifact_parent), "legacy artifact parent"
        )
        overrides = dict(group_artifact_roots or {})
        invalid_overrides = set(overrides) - set(GROUPS[1:])
        if invalid_overrides:
            raise ExperimentRegistryError(
                f"artifact root overrides are invalid: {sorted(invalid_overrides)}"
            )
    else:
        if legacy_artifact_parent is not None:
            raise ExperimentRegistryError(
                "versioned-layout-v2 cannot point at a legacy artifact parent"
            )
        artifact_parent = run_root
        if group_artifact_roots:
            raise ExperimentRegistryError(
                "versioned-layout-v2 cannot override per-group artifact roots"
            )
        overrides = {}

    # Validate immutable inputs before claiming the run namespace.  A malformed
    # source must not leave an empty run ID that looks like a training attempt.
    base_binding = _base_binding(root, Path(base_manifest))
    dataset_binding = _dataset_binding(root, Path(dataset_manifest))

    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ExperimentRegistryError(
            f"run namespace already exists; choose a new run_id: {run_root}"
        ) from exc

    groups: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        group_root = run_root / group
        group_root.mkdir()
        artifact_root = (
            None
            if group == "A"
            else _inside(
                root,
                Path(overrides[group])
                if group in overrides
                else artifact_parent / group,
                f"group {group} artifact root",
            )
        )
        if layout == "versioned-layout-v2" and artifact_root is not None:
            (group_root / "adapters").mkdir()
            (group_root / "manifests").mkdir()
        if group != "A":
            (group_root / "reservations").mkdir()
        groups[group] = {
            "registry_path": _relative(root, group_root / "group_registry.json"),
            "artifact_root": (
                None if artifact_root is None else _relative(root, artifact_root)
            ),
            "storage": (
                "base-only"
                if group == "A"
                else "immutable-external"
                if layout == "legacy-layout-v1"
                else "versioned-local"
            ),
        }

    contract = {
        "run_id": run_id,
        "profile": profile,
        "layout": layout,
        "base_artifact": base_binding,
        "dataset_snapshot": dataset_binding,
        "groups": groups,
    }
    manifest = {
        "schema_version": RUN_SCHEMA,
        "created_at": _now(),
        **contract,
        "contract_sha256": _sha(contract),
        "write_policy": "exclusive-create-no-overwrite",
    }
    _exclusive_json(run_root / "run_manifest.json", manifest)
    return run_root


def _load_run(project_root: Path, run_root: Path) -> dict[str, Any]:
    run_root = _inside(project_root, run_root, "run root")
    manifest = _load_json(run_root / "run_manifest.json", "run manifest")
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ExperimentRegistryError("unsupported run manifest schema")
    run_id = _safe_run_id(str(manifest.get("run_id", "")))
    if run_root.name != run_id:
        raise ExperimentRegistryError("run directory and run_id differ")
    contract = {
        key: manifest.get(key)
        for key in (
            "run_id",
            "profile",
            "layout",
            "base_artifact",
            "dataset_snapshot",
            "groups",
        )
    }
    if manifest.get("contract_sha256") != _sha(contract):
        raise ExperimentRegistryError("run contract hash mismatch")
    base = manifest.get("base_artifact")
    dataset = manifest.get("dataset_snapshot")
    if not isinstance(base, Mapping) or not isinstance(dataset, Mapping):
        raise ExperimentRegistryError("run bindings are malformed")
    base_path = project_root / str(base.get("manifest_path"))
    dataset_path = project_root / str(dataset.get("manifest_path"))
    if sha256_file(base_path) != base.get("manifest_sha256"):
        raise ExperimentRegistryError(
            "base artifact manifest changed after run creation"
        )
    if sha256_file(dataset_path) != dataset.get("manifest_sha256"):
        raise ExperimentRegistryError("dataset manifest changed after run creation")
    return manifest


def _group_path(project_root: Path, run: Mapping[str, Any], group: str) -> Path:
    if group not in GROUPS:
        raise ExperimentRegistryError(f"group must be one of {GROUPS}")
    groups = run.get("groups")
    entry = groups.get(group) if isinstance(groups, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ExperimentRegistryError(f"run has no group {group}")
    registry_path = project_root / str(entry.get("registry_path"))
    return _inside(project_root, registry_path.parent, f"group {group} root")


def _common_group_fields(run: Mapping[str, Any], group: str) -> dict[str, Any]:
    return {
        "schema_version": GROUP_SCHEMA,
        "created_at": _now(),
        "run_id": run["run_id"],
        "profile": run["profile"],
        "layout": run["layout"],
        "group": group,
        "base_artifact": run["base_artifact"],
        "dataset_snapshot": run["dataset_snapshot"],
        "immutable": True,
        "write_policy": "exclusive-create-no-overwrite",
    }


def register_base_group(
    project_root: str | Path,
    run_root: str | Path,
    *,
    config_path: str | Path,
) -> Path:
    """Register A as a native-Q4, zero-adapter baseline."""

    root = Path(project_root).resolve()
    run = _load_run(root, Path(run_root))
    group_root = _group_path(root, run, "A")
    registry_path = group_root / "group_registry.json"
    unexpected = [path for path in group_root.iterdir() if path != registry_path]
    if unexpected:
        raise ExperimentRegistryError("group A must never contain training artifacts")
    config = _inside(root, Path(config_path), "A config")
    manifest = {
        **_common_group_fields(run, "A"),
        "status": "registered",
        "arm_type": "native_q4_base",
        "training_performed": False,
        "configuration": {
            "source_path": _relative(root, config),
            "source_sha256": sha256_file(config),
            "resolved_config_sha256": sha256_file(config),
        },
        "adapters": [],
        "adapter_summary": {
            "count": 0,
            "rank_total": 0,
            "trainable_parameter_total": 0,
            "steps_total": 0,
        },
    }
    return _exclusive_json(registry_path, manifest)


def _expected_adapters(group: str) -> set[str]:
    if group == "B":
        return {"mixed_all"}
    if group in {"C", "D", "E", "F"}:
        return set(SPECIALISTS)
    return set()


def _adapter_record(
    project_root: Path,
    run: Mapping[str, Any],
    adapter_dir: Path,
    manifests_dir: Path,
) -> dict[str, Any]:
    name = adapter_dir.name
    required = {
        "adapter_config": adapter_dir / "adapter_config.json",
        "adapter_model": adapter_dir / "adapter_model.safetensors",
        "checkpoint_metadata": adapter_dir / "checkpoint_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ExperimentRegistryError(f"adapter {name} is incomplete: {missing}")
    config = _load_json(required["adapter_config"], f"{name} adapter config")
    metadata = _load_json(required["checkpoint_metadata"], f"{name} metadata")
    execute_path = manifests_dir / f"{name}.execute.json"
    execute = _load_json(execute_path, f"{name} execute manifest")
    progress_path = adapter_dir.parent / f"{name}.progress" / "status.json"
    progress = _load_json(progress_path, f"{name} progress status")

    adapter_name = metadata.get("adapter_name")
    if adapter_name != execute.get("adapter_name"):
        raise ExperimentRegistryError(f"adapter identity mismatch for {name}")
    if execute.get("run_name") != name or metadata.get("run_name") != name:
        raise ExperimentRegistryError(f"run name mismatch for {name}")
    if execute.get("mode") != "execute" or execute.get("stage") != "train":
        raise ExperimentRegistryError(f"{name} is not a completed training manifest")
    if progress.get("state") != "completed":
        raise ExperimentRegistryError(f"{name} progress is not completed")
    rank = config.get("r")
    step = metadata.get("global_step")
    parameters = metadata.get("trainable_parameters")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (rank, step, parameters)
    ):
        raise ExperimentRegistryError(
            f"invalid rank/step/parameter metadata for {name}"
        )
    if progress.get("step") != step:
        raise ExperimentRegistryError(
            f"progress and checkpoint steps differ for {name}"
        )
    runtime_run_id = progress.get("run_id")
    if not isinstance(runtime_run_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", runtime_run_id
    ):
        raise ExperimentRegistryError(f"invalid runtime run id for {name}")
    config_sha = _require_sha(metadata.get("config_sha256"), f"{name} config hash")
    if execute.get("config_sha256") != config_sha:
        raise ExperimentRegistryError(
            f"execute and checkpoint config hashes differ for {name}"
        )
    observed_snapshot = execute.get("preflight", {}).get("dataset_snapshot_sha256")
    if observed_snapshot != run["dataset_snapshot"]["snapshot_sha256"]:
        raise ExperimentRegistryError(f"dataset snapshot mismatch for {name}")

    config_path = _inside(
        root := project_root,
        Path(str(execute.get("config_path"))),
        f"{name} config path",
    )
    file_hashes = {
        label: {
            "path": _relative(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in required.items()
    }
    tree_sha = _sha(
        [{"name": label, **file_hashes[label]} for label in sorted(file_hashes)]
    )
    return {
        "artifact_name": name,
        "adapter_name": adapter_name,
        "runtime_run_id": runtime_run_id,
        "rank": rank,
        "global_step": step,
        "trainable_parameters": parameters,
        "resolved_config_sha256": config_sha,
        "config_source": {
            "path": _relative(root, config_path),
            "sha256": sha256_file(config_path),
        },
        "execute_manifest": {
            "path": _relative(root, execute_path),
            "sha256": sha256_file(execute_path),
        },
        "final_files": file_hashes,
        "adapter_sha256": tree_sha,
    }


def _validate_group_shape(group: str, records: Sequence[Mapping[str, Any]]) -> None:
    names = {str(item.get("adapter_name")) for item in records}
    expected = _expected_adapters(group)
    if names != expected or len(records) != len(expected):
        raise ExperimentRegistryError(
            f"group {group} adapter set mismatch: expected={sorted(expected)}, "
            f"observed={sorted(names)}"
        )
    ranks = {str(item["adapter_name"]): int(item["rank"]) for item in records}
    total_parameters = sum(int(item["trainable_parameters"]) for item in records)
    if group == "B" and (
        ranks != {"mixed_all": 16} or total_parameters != BUDGET_PARAMETERS
    ):
        raise ExperimentRegistryError("group B must be the one rank-16 mixed adapter")
    if group == "C" and any(rank != 16 for rank in ranks.values()):
        raise ExperimentRegistryError("group C requires five rank-16 specialists")
    if group == "D" and ranks != {
        "planner": 3,
        "tool_policy": 3,
        "frontend_gen": 4,
        "frontend_review": 3,
        "security_gate": 3,
    }:
        raise ExperimentRegistryError("group D requires frozen ranks 3/3/4/3/3")
    if group == "E" and (
        any(rank > 16 for rank in ranks.values()) or len(set(ranks.values())) == 1
    ):
        raise ExperimentRegistryError("group E requires non-uniform ranks capped at 16")
    if group == "F" and (
        sum(ranks.values()) != 16 or total_parameters != BUDGET_PARAMETERS
    ):
        raise ExperimentRegistryError(
            "group F must exactly match B's rank/parameter budget"
        )


def index_completed_group(
    project_root: str | Path,
    run_root: str | Path,
    *,
    group: str,
) -> Path:
    """Create an immutable read-only index over one completed B--F arm."""

    if group not in {"B", "C", "D", "E", "F"}:
        raise ExperimentRegistryError("only B--F contain trainable adapters")
    root = Path(project_root).resolve()
    run = _load_run(root, Path(run_root))
    group_root = _group_path(root, run, group)
    group_entry = run["groups"][group]
    artifact_root = _inside(
        root, root / str(group_entry["artifact_root"]), f"group {group} artifact root"
    )
    adapters_dir = artifact_root / "adapters"
    manifests_dir = artifact_root / "manifests"
    if not adapters_dir.is_dir() or not manifests_dir.is_dir():
        raise ExperimentRegistryError(f"group {group} artifact directories are missing")
    adapter_dirs = sorted(
        path
        for path in adapters_dir.iterdir()
        if path.is_dir() and not path.name.endswith(".progress")
    )
    records = [_adapter_record(root, run, path, manifests_dir) for path in adapter_dirs]
    _validate_group_shape(group, records)
    total_rank = sum(int(item["rank"]) for item in records)
    total_steps = sum(int(item["global_step"]) for item in records)
    total_parameters = sum(int(item["trainable_parameters"]) for item in records)
    manifest = {
        **_common_group_fields(run, group),
        "status": "completed",
        "arm_type": {
            "B": "mixed_single_adapter",
            "C": "full_capacity_routed",
            "D": "manual_budget_matched_routed",
            "E": "adaptive_pareto_routed",
            "F": "adaptive_budget_matched_routed",
        }[group],
        "source": {
            "mode": str(group_entry["storage"]),
            "artifact_root": _relative(root, artifact_root),
            "read_only_index": True,
        },
        "configuration": {
            "resolved_config_sha256": sorted(
                {str(item["resolved_config_sha256"]) for item in records}
            ),
            "source_sha256": sorted(
                {str(item["config_source"]["sha256"]) for item in records}
            ),
        },
        "adapters": records,
        "adapter_summary": {
            "count": len(records),
            "rank_total": total_rank,
            "trainable_parameter_total": total_parameters,
            "steps_total": total_steps,
            "ranks": {str(item["adapter_name"]): int(item["rank"]) for item in records},
            "steps": {
                str(item["adapter_name"]): int(item["global_step"]) for item in records
            },
            "adapter_sha256": {
                str(item["adapter_name"]): str(item["adapter_sha256"])
                for item in records
            },
        },
    }
    return _exclusive_json(group_root / "group_registry.json", manifest)


def reserve_output(
    project_root: str | Path,
    run_root: str | Path,
    *,
    group: str,
    artifact_name: str,
) -> Path:
    """Reserve a fresh adapter/manifest name without touching the output root."""

    if group == "A":
        raise ExperimentRegistryError(
            "group A is base-only and cannot reserve training"
        )
    if group not in GROUPS:
        raise ExperimentRegistryError(f"group must be one of {GROUPS}")
    if not _ARTIFACT_NAME.fullmatch(artifact_name):
        raise ExperimentRegistryError("artifact_name is unsafe")
    root = Path(project_root).resolve()
    run = _load_run(root, Path(run_root))
    group_root = _group_path(root, run, group)
    if (group_root / "group_registry.json").exists():
        raise ExperimentRegistryError("completed group registry is immutable")
    artifact_root = _inside(
        root,
        root / str(run["groups"][group]["artifact_root"]),
        f"group {group} artifact root",
    )
    collisions = [
        artifact_root / "adapters" / artifact_name,
        artifact_root / "adapters" / f"{artifact_name}.progress",
        artifact_root / "manifests" / f"{artifact_name}.execute.json",
        artifact_root / "manifests" / f"{artifact_name}.dry-run.json",
    ]
    existing = [str(path) for path in collisions if path.exists()]
    if existing:
        raise ExperimentRegistryError(
            "refusing to overwrite existing adapter/manifest output: "
            + ", ".join(existing)
        )
    reservation = {
        "schema_version": "anchor.af-output-reservation.v1",
        "created_at": _now(),
        "run_id": run["run_id"],
        "group": group,
        "artifact_name": artifact_name,
        "base_artifact_manifest_sha256": run["base_artifact"]["manifest_sha256"],
        "dataset_snapshot_sha256": run["dataset_snapshot"]["snapshot_sha256"],
        "collisions_checked": [_relative(root, path) for path in collisions],
    }
    return _exclusive_json(
        group_root / "reservations" / f"{artifact_name}.json", reservation
    )


def verify_registry(
    project_root: str | Path, run_root: str | Path, *, group: str
) -> dict[str, Any]:
    """Verify immutable bindings and all indexed final adapter files."""

    root = Path(project_root).resolve()
    run = _load_run(root, Path(run_root))
    group_root = _group_path(root, run, group)
    registry_path = group_root / "group_registry.json"
    registry = _load_json(registry_path, f"group {group} registry")
    if registry.get("schema_version") != GROUP_SCHEMA:
        raise ExperimentRegistryError("unsupported group registry schema")
    for key in ("run_id", "profile", "layout", "base_artifact", "dataset_snapshot"):
        if registry.get(key) != run.get(key):
            raise ExperimentRegistryError(f"group {group} binding changed: {key}")
    if registry.get("group") != group:
        raise ExperimentRegistryError("group registry is stored under the wrong arm")
    if group == "A":
        source = registry.get("configuration", {})
        config = root / str(source.get("source_path"))
        if sha256_file(config) != source.get("source_sha256"):
            raise ExperimentRegistryError("group A config changed after registration")
    else:
        records = registry.get("adapters")
        if not isinstance(records, list):
            raise ExperimentRegistryError("adapter registry must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise ExperimentRegistryError("adapter registry entry is malformed")
            for item in record.get("final_files", {}).values():
                path = root / str(item.get("path"))
                if sha256_file(path) != item.get("sha256"):
                    raise ExperimentRegistryError(
                        f"indexed adapter file changed: {path}"
                    )
            execute = record.get("execute_manifest", {})
            path = root / str(execute.get("path"))
            if sha256_file(path) != execute.get("sha256"):
                raise ExperimentRegistryError(f"execute manifest changed: {path}")
            source = record.get("config_source", {})
            path = root / str(source.get("path"))
            if sha256_file(path) != source.get("sha256"):
                raise ExperimentRegistryError(f"training config changed: {path}")
        _validate_group_shape(group, records)
    return {
        "ok": True,
        "run_id": run["run_id"],
        "group": group,
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Immutable A--F run registry")
    parser.add_argument("--project-root", default=".")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--registry-parent", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--profile", required=True)
    init.add_argument("--layout", choices=LAYOUTS, required=True)
    init.add_argument("--base-manifest", required=True)
    init.add_argument("--dataset-manifest", required=True)
    init.add_argument("--legacy-artifact-parent")
    init.add_argument(
        "--group-artifact-root",
        action="append",
        default=[],
        metavar="GROUP=PATH",
        help="legacy-only B--F artifact-root override; may be repeated",
    )
    base = commands.add_parser("register-a")
    base.add_argument("--run-root", required=True)
    base.add_argument("--config", required=True)
    index = commands.add_parser("index")
    index.add_argument("--run-root", required=True)
    index.add_argument("--group", choices=GROUPS[1:], required=True)
    reserve = commands.add_parser("reserve")
    reserve.add_argument("--run-root", required=True)
    reserve.add_argument("--group", choices=GROUPS[1:], required=True)
    reserve.add_argument("--artifact-name", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--group", choices=GROUPS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    try:
        if args.command == "init":
            overrides: dict[str, Path] = {}
            for raw in args.group_artifact_root:
                if "=" not in raw:
                    raise ExperimentRegistryError(
                        "--group-artifact-root must use GROUP=PATH"
                    )
                group, path = raw.split("=", 1)
                if not path:
                    raise ExperimentRegistryError(
                        "--group-artifact-root path cannot be empty"
                    )
                if group in overrides:
                    raise ExperimentRegistryError(
                        f"duplicate artifact-root override for group {group}"
                    )
                overrides[group] = root / path
            result: object = initialize_run(
                root,
                root / args.registry_parent,
                run_id=args.run_id,
                profile=args.profile,
                layout=args.layout,
                base_manifest=root / args.base_manifest,
                dataset_manifest=root / args.dataset_manifest,
                legacy_artifact_parent=(
                    None
                    if args.legacy_artifact_parent is None
                    else root / args.legacy_artifact_parent
                ),
                group_artifact_roots=overrides,
            )
        elif args.command == "register-a":
            result = register_base_group(
                root, root / args.run_root, config_path=root / args.config
            )
        elif args.command == "index":
            result = index_completed_group(root, root / args.run_root, group=args.group)
        elif args.command == "reserve":
            result = reserve_output(
                root,
                root / args.run_root,
                group=args.group,
                artifact_name=args.artifact_name,
            )
        else:
            result = verify_registry(root, root / args.run_root, group=args.group)
        print(
            json.dumps(
                result
                if isinstance(result, Mapping)
                else {"ok": True, "path": str(result)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ExperimentRegistryError, OSError) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
