from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.swebench import frozen_prefix_qreader_release as release
from anchor_mvp.swebench.frozen_prefix_qreader_release import (
    FrozenPrefixQReaderReleaseError,
    freeze_frozen_prefix_qreader_release,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/frozen_prefix_qreader_release_overlay_v1.json"
OVERLAY_SCHEMA = (
    ROOT / "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json"
)
MIRROR_FILES = (
    "configs/research/frozen_prefix_qreader_release_overlay_v1.json",
    "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json",
    "configs/research/generic_train_release_lock.schema.json",
    "configs/orchestration/frozen_prefix_qreader_profile_freeze_manifest.schema.json",
    "configs/research/swebench_natural_language_scaffold_v2_manifest.schema.json",
    "configs/research/swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json",
    "configs/research/swebench_taskboard_projector_v2.yaml",
    "src/anchor_mvp/swebench/taskboard_projector.py",
    "scripts/data/project_swebench_taskboard.py",
    "configs/research/taskboard_projector_manifest.schema.json",
    "configs/research/taskboard_projector_sidecar.schema.json",
    "configs/research/hierarchical_task_kv_segment_plan.schema.json",
    "configs/research/swebench_natural_language_scaffold_v2.yaml",
    "src/anchor_mvp/swebench/natural_language_scaffold_v2.py",
    "configs/research/swebench_natural_language_scaffold_v2_config.schema.json",
    "configs/research/swebench_natural_language_scaffold_v2_record.schema.json",
    "configs/research/swebench_natural_language_scaffold_v2_bundle_profile.schema.json",
    "configs/research/swebench_natural_language_scaffold_v2_bundle_profile_descriptor.schema.json",
    "scripts/data/build_swebench_natural_language_scaffold_v2.py",
    "scripts/data/audit_swebench_natural_language_scaffold_v2.py",
    "src/anchor_mvp/swebench/frozen_prefix_qreader_release.py",
    "scripts/data/freeze_frozen_prefix_qreader_release.py",
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_artifact(directory: Path, value: Mapping[str, Any]) -> str:
    directory.mkdir(parents=True)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    digest = sha256(encoded).hexdigest()
    (directory / "manifest.json").write_bytes(encoded)
    (directory / "manifest.json.sha256").write_bytes(
        f"{digest}  manifest.json\n".encode("ascii")
    )
    return digest


def _resolve_ref(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    assert ref.startswith("#/")
    value: Any = root
    for part in ref[2:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    assert isinstance(value, Mapping)
    return value


def _merged(schema: Mapping[str, Any], root: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(schema)
    ref = value.pop("$ref", None)
    if ref is not None:
        base = _merged(_resolve_ref(root, str(ref)), root)
        for key, item in value.items():
            if key == "properties" and isinstance(item, Mapping):
                base.setdefault("properties", {})
                base["properties"].update(item)
            else:
                base[key] = item
        value = base
    for item in value.pop("allOf", []):
        merged = _merged(item, root)
        for key, nested in merged.items():
            if key == "properties" and isinstance(nested, Mapping):
                value.setdefault("properties", {})
                value["properties"].update(nested)
            elif key == "required":
                value["required"] = list(
                    dict.fromkeys([*value.get("required", []), *nested])
                )
            else:
                value.setdefault(key, nested)
    return value


def _sample(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    value = _merged(schema, root)
    if "oneOf" in value:
        choice = _merged(value.pop("oneOf")[0], root)
        value.setdefault("properties", {})
        value["properties"].update(choice.get("properties", {}))
        value["required"] = list(
            dict.fromkeys([*value.get("required", []), *choice.get("required", [])])
        )
        for key, item in choice.items():
            if key not in {"properties", "required"}:
                value.setdefault(key, item)
    if "const" in value:
        return deepcopy(value["const"])
    if "enum" in value:
        return deepcopy(value["enum"][0])
    value_type = value.get("type")
    if value_type == "object" or "properties" in value:
        properties = value.get("properties", {})
        result = {
            key: _sample(properties[key], root) for key in value.get("required", [])
        }
        if len(result) < int(value.get("minProperties", 0)):
            additional = value.get("additionalProperties", {})
            while len(result) < int(value["minProperties"]):
                result[f"key_{len(result)}"] = _sample(additional, root)
        return result
    if value_type == "array" or "prefixItems" in value:
        prefix = value.get("prefixItems", [])
        result = [_sample(item, root) for item in prefix]
        minimum = int(value.get("minItems", len(result)))
        item_schema = value.get("items", {})
        while len(result) < minimum:
            result.append(_sample(item_schema, root))
        return result
    if value_type == "boolean":
        return False
    if value_type == "integer":
        return max(int(value.get("minimum", 0)), 1)
    if value_type == "number":
        return float(value.get("minimum", 0))
    if value_type == "string" or "pattern" in value:
        pattern = value.get("pattern")
        if pattern == "^[0-9a-f]{64}$":
            return "a" * 64
        if isinstance(pattern, str) and pattern.startswith("^[a-z]"):
            return "role_id"
        if isinstance(pattern, str) and ("A-Za-z]:" in pattern or "\\.\\." in pattern):
            return "file.json"
        if "path" in str(value.get("title", "")).lower():
            return "file.json"
        return "x"
    return {}


def _schema(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, Mapping)
    Draft202012Validator.check_schema(value)
    return value


def _valid_from_schema(path: Path) -> dict[str, Any]:
    schema = _schema(path)
    value = _sample(schema, schema)
    assert isinstance(value, dict)
    return value


def _mirror_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in MIRROR_FILES:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    config_path = (
        root / "configs/research/frozen_prefix_qreader_release_overlay_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for binding in config["producer_bindings"].values():
        path = root / binding["path"]
        binding["sha256"] = _sha(path)
        binding["bytes"] = path.stat().st_size
    for contract in config["input_contracts"].values():
        contract["schema_sha256"] = _sha(root / contract["schema_path"])
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return root


def _consumer_reference() -> dict[str, Any]:
    return {
        "schema_version": "anchor.synthetic-five-role-qonly-diagnostic-manifest.v1",
        "status": "dataset_proxy_ready_training_not_authorized",
        "claim_scope": "synthetic_diagnostic_only_no_formal_or_training_authority",
        "counts": {
            "records": 1000,
            "role_views": 1000,
            "pair_count": 0,
            "task_bundles": 200,
            "roles": 5,
            "languages": 2,
            "variants": 1,
            "variants_per_role": 1,
            "train_records": 800,
            "eval_proxy_records": 200,
        },
        "semantic_identity_contract": {
            "en_unique_task_semantics": 100,
            "zh_cn_unique_task_semantics": 100,
            "translation_pair_count": 0,
        },
        "ablation_contract": {
            "primary_label": "q_only",
            "q_only_is_only_primary": True,
        },
        "claims": {
            "diagnostic_only": True,
            "training_authorized": False,
            "formal": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
            "physical_kv_reuse_claimed": False,
            "numeric_equivalence_claimed": False,
        },
        "source_disjoint_boundary": {
            "zero_intersection_claimed": False,
            "formal_source_disjoint_proven": False,
        },
        "compatibility_boundary": {
            "pair_count": 0,
            "variants_per_role": 1,
        },
        "audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_body_reads": 0,
            "real_tool_executions": 0,
            "forbidden_current_future_content_excluded": True,
        },
    }


def _inputs(tmp_path: Path) -> dict[str, Any]:
    root = _mirror_project(tmp_path)
    generic_schema = root / "configs/research/generic_train_release_lock.schema.json"
    profile_schema = (
        root
        / "configs/orchestration/frozen_prefix_qreader_profile_freeze_manifest.schema.json"
    )
    view_schema = (
        root
        / "configs/research/swebench_natural_language_scaffold_v2_manifest.schema.json"
    )
    bundle_schema = (
        root
        / "configs/research/swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json"
    )
    generic = _valid_from_schema(generic_schema)
    profile = _valid_from_schema(profile_schema)
    bundle = _valid_from_schema(bundle_schema)
    view = _valid_from_schema(view_schema)
    dirs = {
        "generic_release_lock": (
            root / "artifacts/formal_v3/training_release/release_lock"
        ),
        "profile_freeze": (
            root / "artifacts/distillation-profiles/frozen-prefix-qreader-v2"
        ),
        "training_view": (root / "artifacts/swebench/frozen-prefix-qreader-view-v2"),
        "bundle_profile": (
            root / "artifacts/swebench/frozen-prefix-qreader-bundle-profile-v2"
        ),
        "consumer_reference": tmp_path / "consumer",
    }

    generic["bindings"]["projector_manifest_sha256"] = "1" * 64
    profile_schema_value = _schema(profile_schema)
    dependency_choices = profile_schema_value["$defs"]["dependency_role_path"]["oneOf"]
    profile["dependencies"] = [
        {
            "role": item["properties"]["role"]["const"],
            "path": item["properties"]["path"]["const"],
            "sha256": "a" * 64,
            "bytes": 1,
        }
        for item in dependency_choices
    ]
    dependency_by_role = {
        dependency["role"]: dependency for dependency in profile["dependencies"]
    }
    for role in release._CROSS_BOUND_PROFILE_DEPENDENCY_ROLES:
        dependency = dependency_by_role[role]
        path = root / dependency["path"]
        dependency["sha256"] = _sha(path)
        dependency["bytes"] = path.stat().st_size
    generic["bindings"]["projector_manifest_schema_sha256"] = dependency_by_role[
        "taskboard_projector_manifest_schema"
    ]["sha256"]
    generic["bindings"]["projector_sidecar_schema_sha256"] = dependency_by_role[
        "taskboard_projector_sidecar_schema"
    ]["sha256"]
    generic["bindings"]["projector_segment_plan_schema_sha256"] = dependency_by_role[
        "taskboard_segment_plan_schema"
    ]["sha256"]
    generic["hierarchical_task_kv"]["segment_plan_schema_sha256"] = dependency_by_role[
        "taskboard_segment_plan_schema"
    ]["sha256"]
    overlay_dependencies = {
        "release_overlay_schema": (
            root
            / "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json"
        ),
        "release_overlay_implementation": (
            root / "src/anchor_mvp/swebench/frozen_prefix_qreader_release.py"
        ),
        "release_overlay_cli": (
            root / "scripts/data/freeze_frozen_prefix_qreader_release.py"
        ),
    }
    for dependency in profile["dependencies"]:
        path = overlay_dependencies.get(dependency["role"])
        if path is not None:
            dependency["sha256"] = _sha(path)
            dependency["bytes"] = path.stat().st_size
    dependency_by_role = {
        dependency["role"]: dependency for dependency in profile["dependencies"]
    }
    bundle["source"]["projector_manifest_sha256"] = "1" * 64
    bundle["producer"]["config_sha256"] = dependency_by_role[
        "training_view_materializer_config"
    ]["sha256"]
    bundle["producer"]["implementation_sha256"] = dependency_by_role[
        "training_view_materializer_implementation"
    ]["sha256"]
    bundle["producer"]["record_schema_sha256"] = dependency_by_role[
        "bundle_profile_record_schema"
    ]["sha256"]
    bundle["producer"]["manifest_schema_sha256"] = dependency_by_role[
        "bundle_profile_manifest_schema"
    ]["sha256"]
    bundle["producer"]["descriptor_schema_sha256"] = dependency_by_role[
        "bundle_profile_descriptor_schema"
    ]["sha256"]
    bundle_dir = dirs["bundle_profile"]
    bundle_sha = _write_artifact(bundle_dir, bundle)
    bundle_sidecar_sha = _sha(bundle_dir / "manifest.json.sha256")
    view["input"]["projector_manifest_sha256"] = "1" * 64
    view["input"]["projector_manifest_sidecar_sha256"] = sha256(
        f"{'1' * 64}  manifest.json\n".encode("ascii")
    ).hexdigest()
    view["input"]["projector_manifest_schema_sha256"] = dependency_by_role[
        "taskboard_projector_manifest_schema"
    ]["sha256"]
    view["input"]["projector_record_schema_sha256"] = dependency_by_role[
        "taskboard_projector_sidecar_schema"
    ]["sha256"]
    view["input"]["segment_plan_schema_sha256"] = dependency_by_role[
        "taskboard_segment_plan_schema"
    ]["sha256"]
    view["input"]["bundle_profile_manifest_sha256"] = bundle_sha
    view["input"]["bundle_profile_manifest_sidecar_sha256"] = bundle_sidecar_sha
    view["input"]["bundle_profile_manifest_schema_sha256"] = _sha(bundle_schema)
    view["input"]["bundle_profile_record_schema_sha256"] = dependency_by_role[
        "bundle_profile_record_schema"
    ]["sha256"]
    view["producer"]["config_sha256"] = dependency_by_role[
        "training_view_materializer_config"
    ]["sha256"]
    view["producer"]["implementation_sha256"] = dependency_by_role[
        "training_view_materializer_implementation"
    ]["sha256"]
    view["producer"]["record_schema_sha256"] = dependency_by_role[
        "training_view_record_schema"
    ]["sha256"]
    view["producer"]["manifest_schema_sha256"] = dependency_by_role[
        "training_view_manifest_schema"
    ]["sha256"]

    hashes = {
        "generic_release_lock": _write_artifact(dirs["generic_release_lock"], generic),
        "profile_freeze": _write_artifact(dirs["profile_freeze"], profile),
        "training_view": _write_artifact(dirs["training_view"], view),
        "bundle_profile": bundle_sha,
        "consumer_reference": _write_artifact(
            dirs["consumer_reference"], _consumer_reference()
        ),
    }
    # The production config pins the real consumer fixture hashes. Tests keep
    # those semantics but use a tiny metadata-only fixture, so update only the
    # mirrored config and authenticate its resulting bytes.
    config_path = (
        root / "configs/research/frozen_prefix_qreader_release_overlay_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["consumer_diagnostic_reference"]["manifest_sha256"] = hashes[
        "consumer_reference"
    ]
    config["consumer_diagnostic_reference"]["sidecar_physical_sha256"] = _sha(
        dirs["consumer_reference"] / "manifest.json.sha256"
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "root": root,
        "config": config_path,
        "config_sha": _sha(config_path),
        "dirs": dirs,
        "hashes": hashes,
    }


def _freeze(
    inputs: Mapping[str, Any], output: str = "artifacts/release"
) -> dict[str, Any]:
    dirs = inputs["dirs"]
    hashes = inputs["hashes"]
    return freeze_frozen_prefix_qreader_release(
        project_root=inputs["root"],
        config_path="configs/research/frozen_prefix_qreader_release_overlay_v1.json",
        expected_config_sha256=inputs["config_sha"],
        generic_release_dir=dirs["generic_release_lock"],
        expected_generic_release_sha256=hashes["generic_release_lock"],
        profile_freeze_dir=dirs["profile_freeze"],
        expected_profile_freeze_sha256=hashes["profile_freeze"],
        training_view_dir=dirs["training_view"],
        expected_training_view_sha256=hashes["training_view"],
        bundle_profile_dir=dirs["bundle_profile"],
        expected_bundle_profile_sha256=hashes["bundle_profile"],
        consumer_reference_dir=dirs["consumer_reference"],
        expected_consumer_reference_sha256=hashes["consumer_reference"],
        output_dir=output,
    )


def _cli_command(
    inputs: Mapping[str, Any],
    *,
    script: Path | None = None,
    output: str = "artifacts/cli-release",
) -> list[str]:
    dirs = inputs["dirs"]
    hashes = inputs["hashes"]
    return [
        sys.executable,
        str(
            script
            or (inputs["root"] / "scripts/data/freeze_frozen_prefix_qreader_release.py")
        ),
        "--config",
        str(inputs["config"]),
        "--config-sha256",
        inputs["config_sha"],
        "--generic-release-dir",
        str(dirs["generic_release_lock"]),
        "--generic-release-manifest-sha256",
        hashes["generic_release_lock"],
        "--profile-freeze-dir",
        str(dirs["profile_freeze"]),
        "--profile-freeze-manifest-sha256",
        hashes["profile_freeze"],
        "--training-view-dir",
        str(dirs["training_view"]),
        "--training-view-manifest-sha256",
        hashes["training_view"],
        "--bundle-profile-dir",
        str(dirs["bundle_profile"]),
        "--bundle-profile-manifest-sha256",
        hashes["bundle_profile"],
        "--consumer-reference-dir",
        str(dirs["consumer_reference"]),
        "--consumer-reference-manifest-sha256",
        hashes["consumer_reference"],
        "--output-dir",
        output,
    ]


def test_release_overlay_success_is_blocked_and_body_free(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = _freeze(inputs)
    output = inputs["root"] / "artifacts/release"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    schema = _schema(
        inputs["root"]
        / "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json"
    )
    Draft202012Validator(schema).validate(manifest)
    assert result["status"] == "profile_materialized_execution_blocked"
    assert result["manifest_files_read"] == 5
    assert result["profile_dependency_files_authenticated"] == 17
    assert result["partition_files_read"] == 0
    assert result["provider_requests"] == 0
    assert result["model_loads"] == 0
    assert result["gpu_requests"] == 0
    assert result["training_authorized"] is False
    assert result["formal_training_authorized"] is False
    assert result["release_authorized"] is False
    assert manifest["base_release"]["self_reported_status"] == "ready"
    assert manifest["base_release"]["authority_inherited"] is False
    assert manifest["consumer_diagnostic_reference"]["reference_only"] is True
    assert manifest["gemma_runner_compatibility"]["sequence_length"] == 768
    assert manifest["gemma_runner_compatibility"]["observed_records_over_512"] == 514
    assert (output / "manifest.json.sha256").read_text(encoding="ascii") == (
        f"{result['manifest_sha256']}  manifest.json\n"
    )


def test_release_overlay_never_reads_partition_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    for directory in inputs["dirs"].values():
        (directory / "do-not-read.jsonl").write_text(
            '{"opaque_body":"must-not-be-read"}\n', encoding="utf-8"
        )
    observed: list[Path] = []
    original = release._read_snapshot

    def audit(path: Path, code: str) -> release.BytesSnapshot:
        observed.append(path)
        return original(path, code)

    monkeypatch.setattr(release, "_read_snapshot", audit)
    _freeze(inputs)
    assert not any(path.suffix == ".jsonl" for path in observed)
    assert not any("Gold" in str(path) or "heldout" in str(path) for path in observed)


def test_release_overlay_rejects_strict_sidecar_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    sidecar = inputs["dirs"]["training_view"] / "manifest.json.sha256"
    sidecar.write_text(
        f"{inputs['hashes']['training_view']} *manifest.json\n",
        encoding="ascii",
    )
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="training_view_artifact_invalid",
    ):
        _freeze(inputs)
    assert not (inputs["root"] / "artifacts/release").exists()


def test_release_overlay_rejects_bundle_cross_binding_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    path = inputs["dirs"]["training_view"] / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["input"]["bundle_profile_manifest_sha256"] = "0" * 64
    inputs["hashes"]["training_view"] = _write_artifact_replace(
        inputs["dirs"]["training_view"], value
    )
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="training_view_semantics_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_training_view_producer_identity_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    directory = inputs["dirs"]["training_view"]
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    value["producer"]["implementation_sha256"] = "0" * 64
    inputs["hashes"]["training_view"] = _write_artifact_replace(directory, value)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="training_view_semantics_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_bundle_producer_identity_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    directory = inputs["dirs"]["bundle_profile"]
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    value["producer"]["config_sha256"] = "0" * 64
    inputs["hashes"]["bundle_profile"] = _write_artifact_replace(directory, value)
    view_dir = inputs["dirs"]["training_view"]
    view = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
    view["input"]["bundle_profile_manifest_sha256"] = inputs["hashes"]["bundle_profile"]
    view["input"]["bundle_profile_manifest_sidecar_sha256"] = _sha(
        directory / "manifest.json.sha256"
    )
    inputs["hashes"]["training_view"] = _write_artifact_replace(view_dir, view)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="bundle_profile_semantics_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_projector_schema_transitive_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    generic_dir = inputs["dirs"]["generic_release_lock"]
    generic = json.loads((generic_dir / "manifest.json").read_text(encoding="utf-8"))
    generic["bindings"]["projector_manifest_schema_sha256"] = "0" * 64
    inputs["hashes"]["generic_release_lock"] = _write_artifact_replace(
        generic_dir, generic
    )
    view_dir = inputs["dirs"]["training_view"]
    view = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
    view["input"]["projector_manifest_schema_sha256"] = "0" * 64
    inputs["hashes"]["training_view"] = _write_artifact_replace(view_dir, view)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="training_view_semantics_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_projector_sidecar_identity_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    view_dir = inputs["dirs"]["training_view"]
    view = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
    view["input"]["projector_manifest_sidecar_sha256"] = "0" * 64
    inputs["hashes"]["training_view"] = _write_artifact_replace(view_dir, view)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="training_view_semantics_invalid",
    ):
        _freeze(inputs)


@pytest.mark.parametrize(
    "role",
    [
        "taskboard_projector_config",
        "taskboard_projector_implementation",
        "training_view_materializer_config",
        "training_view_materializer_implementation",
        "training_view_materializer_builder",
        "training_view_materializer_auditor",
    ],
)
def test_release_overlay_rejects_profile_dependency_physical_drift(
    tmp_path: Path,
    role: str,
) -> None:
    inputs = _inputs(tmp_path)
    profile_path = inputs["dirs"]["profile_freeze"] / "manifest.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    dependency = next(item for item in profile["dependencies"] if item["role"] == role)
    path = inputs["root"] / dependency["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="profile_dependency_binding_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_profile_overlay_code_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    directory = inputs["dirs"]["profile_freeze"]
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dependency = next(
        item
        for item in value["dependencies"]
        if item["role"] == "release_overlay_implementation"
    )
    dependency["sha256"] = "0" * 64
    inputs["hashes"]["profile_freeze"] = _write_artifact_replace(directory, value)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="profile_overlay_dependency_binding_invalid",
    ):
        _freeze(inputs)


def _write_artifact_replace(directory: Path, value: Mapping[str, Any]) -> str:
    shutil.rmtree(directory)
    return _write_artifact(directory, value)


def test_release_overlay_rejects_consumer_count_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    directory = inputs["dirs"]["consumer_reference"]
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    value["counts"]["records"] = 999
    inputs["hashes"]["consumer_reference"] = _write_artifact_replace(directory, value)
    config = json.loads(inputs["config"].read_text(encoding="utf-8"))
    config["consumer_diagnostic_reference"]["manifest_sha256"] = inputs["hashes"][
        "consumer_reference"
    ]
    config["consumer_diagnostic_reference"]["sidecar_physical_sha256"] = _sha(
        directory / "manifest.json.sha256"
    )
    inputs["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inputs["config_sha"] = _sha(inputs["config"])
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="consumer_reference_semantics_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_nested_consumer_authority_claims(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    directory = inputs["dirs"]["consumer_reference"]
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    value["release_authorized"] = True
    value["claims"]["formal_training_authorized"] = True
    inputs["hashes"]["consumer_reference"] = _write_artifact_replace(directory, value)
    config = json.loads(inputs["config"].read_text(encoding="utf-8"))
    config["consumer_diagnostic_reference"]["manifest_sha256"] = inputs["hashes"][
        "consumer_reference"
    ]
    config["consumer_diagnostic_reference"]["sidecar_physical_sha256"] = _sha(
        directory / "manifest.json.sha256"
    )
    inputs["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inputs["config_sha"] = _sha(inputs["config"])
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="consumer_reference_authority_claim_invalid",
    ):
        _freeze(inputs)


@pytest.mark.parametrize(
    "claim",
    [
        "authorized",
        "release_ready",
        "release_eligible",
        "training_authority",
        "formal_authority",
        "formal_authorization_granted",
    ],
)
def test_release_overlay_rejects_unknown_positive_authority_gate(
    tmp_path: Path,
    claim: str,
) -> None:
    inputs = _inputs(tmp_path)
    directory = inputs["dirs"]["consumer_reference"]
    value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    value["claims"][claim] = True
    inputs["hashes"]["consumer_reference"] = _write_artifact_replace(directory, value)
    config = json.loads(inputs["config"].read_text(encoding="utf-8"))
    config["consumer_diagnostic_reference"]["manifest_sha256"] = inputs["hashes"][
        "consumer_reference"
    ]
    config["consumer_diagnostic_reference"]["sidecar_physical_sha256"] = _sha(
        directory / "manifest.json.sha256"
    )
    inputs["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inputs["config_sha"] = _sha(inputs["config"])
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="consumer_reference_authority_claim_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_input_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    original = release._recheck
    invoked = False

    def tamper(
        inventory: Mapping[str, tuple[Path, release.BytesSnapshot]],
        artifacts: Mapping[str, release.InputArtifact],
    ) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            path = artifacts["profile_freeze"].manifest_path
            path.write_bytes(path.read_bytes() + b" ")
        original(inventory, artifacts)

    monkeypatch.setattr(release, "_recheck", tamper)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="profile_freeze_changed_during_operation",
    ):
        _freeze(inputs)
    assert not (inputs["root"] / "artifacts/release").exists()


def test_release_overlay_removes_output_on_terminal_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    original = release._recheck
    calls = 0

    def tamper_after_publish(
        inventory: Mapping[str, tuple[Path, release.BytesSnapshot]],
        artifacts: Mapping[str, release.InputArtifact],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            path = artifacts["generic_release_lock"].manifest_path
            path.write_bytes(path.read_bytes() + b" ")
        original(inventory, artifacts)

    monkeypatch.setattr(release, "_recheck", tamper_after_publish)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="generic_release_lock_changed_during_operation",
    ):
        _freeze(inputs)
    assert calls == 2
    assert not (inputs["root"] / "artifacts/release").exists()


def test_release_overlay_rejects_dependency_cycle(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = json.loads(inputs["config"].read_text(encoding="utf-8"))
    config["dependency_dag"]["edges"].append(["release_overlay", "overlay_config"])
    inputs["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inputs["config_sha"] = _sha(inputs["config"])
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="release_overlay_config_invalid",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_cross_checkout_execution_identity(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    implementation = (
        inputs["root"] / "src/anchor_mvp/swebench/frozen_prefix_qreader_release.py"
    )
    implementation.write_bytes(implementation.read_bytes() + b"\n# drift\n")
    config = json.loads(inputs["config"].read_text(encoding="utf-8"))
    binding = config["producer_bindings"]["implementation"]
    binding["sha256"] = _sha(implementation)
    binding["bytes"] = implementation.stat().st_size
    inputs["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inputs["config_sha"] = _sha(inputs["config"])
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="executing_implementation_binding_invalid",
    ):
        _freeze(inputs)


def test_release_cli_executes_authenticated_snapshot_successfully(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    completed = subprocess.run(
        _cli_command(inputs),
        cwd=inputs["root"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "profile_materialized_execution_blocked"
    assert payload["training_authorized"] is False
    assert (inputs["root"] / "artifacts/cli-release/manifest.json").is_file()


def test_release_cli_never_executes_unbound_implementation_bytes(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    marker = tmp_path / "unverified-module-executed"
    implementation = (
        inputs["root"] / "src/anchor_mvp/swebench/frozen_prefix_qreader_release.py"
    )
    implementation.write_bytes(
        implementation.read_bytes()
        + f"\nPath({str(marker)!r}).write_text('executed')\n".encode("utf-8")
    )
    completed = subprocess.run(
        _cli_command(inputs),
        cwd=inputs["root"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stderr)
    assert payload["error"] == "release_bootstrap_implementation_binding_invalid"
    assert marker.exists() is False
    assert (inputs["root"] / "artifacts/cli-release").exists() is False


def test_release_cli_rejects_noncanonical_executing_copy(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    canonical = inputs["root"] / "scripts/data/freeze_frozen_prefix_qreader_release.py"
    sibling = canonical.with_name("freeze_frozen_prefix_qreader_release_copy.py")
    shutil.copyfile(canonical, sibling)
    completed = subprocess.run(
        _cli_command(inputs, script=sibling),
        cwd=inputs["root"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stderr)
    assert payload["error"] == "release_bootstrap_executing_cli_path_invalid"
    assert (inputs["root"] / "artifacts/cli-release").exists() is False


def test_release_overlay_rejects_existing_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = inputs["root"] / "artifacts/release"
    output.mkdir(parents=True)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="release_overlay_output_exists",
    ):
        _freeze(inputs)


def test_release_overlay_rejects_noncanonical_producer_artifact_path(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    alternate = tmp_path / "alternate-view"
    shutil.copytree(inputs["dirs"]["training_view"], alternate)
    inputs["dirs"]["training_view"] = alternate
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="training_view_artifact_path_invalid",
    ):
        _freeze(inputs)


@pytest.mark.parametrize(
    "relative",
    [
        "configs/research/.",
        "configs/research/..",
        "configs/./research/file.json",
        "configs/../research/file.json",
    ],
)
def test_project_binding_rejects_dot_segments(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / "project"
    (root / "configs/research").mkdir(parents=True)
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="test_path_invalid",
    ):
        release._safe_project_file(root, relative, "test_path_invalid")


def test_release_overlay_rejects_symlinked_artifact(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    source = inputs["dirs"]["bundle_profile"]
    link = tmp_path / "bundle-link"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available")
    inputs["dirs"]["bundle_profile"] = link
    with pytest.raises(
        FrozenPrefixQReaderReleaseError,
        match="bundle_profile_artifact_invalid",
    ):
        _freeze(inputs)


def test_checked_in_config_contains_no_runtime_manifest_hash_placeholders() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["dependency_dag"]["config_excludes_runtime_manifest_hashes"] is True
    for binding in value["producer_bindings"].values():
        path = ROOT / binding["path"]
        assert binding["sha256"] == _sha(path)
        assert binding["bytes"] == path.stat().st_size
    for contract in value["input_contracts"].values():
        assert set(contract) == {
            "schema_version",
            "schema_path",
            "schema_sha256",
            "canonical_runtime_dir",
        }
        assert contract["schema_sha256"] == _sha(ROOT / contract["schema_path"])
    assert value["consumer_diagnostic_reference"]["manifest_sha256"] == (
        "a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed"
    )
    assert (
        value["consumer_diagnostic_reference"]["sidecar_physical_sha256"]
        == "7f238be47cc60af808421bbbdaefb6bbc5d5c0f617d976b66d5b2a87d767b0a0"
    )
    assert "PLACEHOLDER" not in CONFIG.read_text(encoding="utf-8")


def test_overlay_schema_is_valid_draft_2020_12() -> None:
    schema = _schema(OVERLAY_SCHEMA)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
