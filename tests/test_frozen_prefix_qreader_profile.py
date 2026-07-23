from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from anchor_mvp.swebench import frozen_prefix_qreader_profile as profile_module
from anchor_mvp.swebench.frozen_prefix_qreader_profile import (
    DEPENDENCY_LAYERS,
    EXPECTED_DEPENDENCY_PATHS,
    EXPECTED_DEPENDENCY_ROLES,
    FREEZE_MANIFEST_SCHEMA_PATH,
    PROFILE_PATH,
    PROFILE_SCHEMA_PATH,
    SHARED_CORE_ROLES,
    FrozenPrefixQReaderProfileError,
    core_command_metadata,
    freeze_profile,
    load_profile,
    preflight_profile,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / PROFILE_PATH
SCHEMA = ROOT / PROFILE_SCHEMA_PATH
FREEZE_SCHEMA = ROOT / FREEZE_MANIFEST_SCHEMA_PATH
RUNNER = ROOT / "scripts/data/run_frozen_prefix_qreader_profile.py"
PLACEHOLDER = "PLACEHOLDER_PENDING_PHYSICAL_FREEZE"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _identity(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return sha256(data).hexdigest(), len(data)


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    value = _load_json(PROFILE)

    fixed_paths = {
        PROFILE_PATH.as_posix(),
        PROFILE_SCHEMA_PATH.as_posix(),
        FREEZE_MANIFEST_SCHEMA_PATH.as_posix(),
        str(value["v1_reference"]["path"]),  # type: ignore[index]
    }
    dependencies = value["dependencies"]
    assert isinstance(dependencies, list)
    for raw in dependencies:
        assert isinstance(raw, dict)
        fixed_paths.add(str(raw["path"]))

    for relative in sorted(fixed_paths):
        source = ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            destination.write_bytes(
                f"fixture-only dependency: {relative}\n".encode("utf-8")
            )

    for field in ("profile_schema", "freeze_manifest_schema", "v1_reference"):
        raw = value[field]
        assert isinstance(raw, dict)
        digest, size = _identity(root / str(raw["path"]))
        raw["sha256"] = digest
        raw["bytes"] = size
    for raw in dependencies:
        assert isinstance(raw, dict)
        digest, size = _identity(root / str(raw["path"]))
        raw["state"] = "bound"
        raw["sha256"] = digest
        raw["bytes"] = size
    _write_json(root / PROFILE_PATH, value)
    return root


def _profile_value(root: Path = ROOT) -> dict[str, object]:
    return _load_json(root / PROFILE_PATH)


def test_published_schema_validates_profile_and_exact_architecture_contract() -> None:
    from jsonschema import Draft202012Validator

    schema = _load_json(SCHEMA)
    profile = _profile_value()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(profile)

    assert profile["profile_id"] == "frozen-prefix-qreader-v2"
    pipeline = profile["post_gold_pipeline"]
    assert isinstance(pipeline, dict)
    assert pipeline["training_view_record_schema_version"] == (
        "anchor.frozen-prefix-qreader-training-view.v2"
    )
    assert pipeline["training_view_manifest_schema_version"] == (
        "anchor.frozen-prefix-qreader-training-view-manifest.v2"
    )
    assert pipeline["bundle_profile_record_schema_version"] == (
        "anchor.frozen-prefix-qreader-bundle-profile.v2"
    )
    assert pipeline["bundle_profile_manifest_schema_version"] == (
        "anchor.frozen-prefix-qreader-bundle-profile-manifest.v2"
    )
    assert pipeline["primary_adapter_label"] == "q_only"
    assert pipeline["diagnostic_adapter_labels"] == ["o_only", "q_plus_o"]
    assert pipeline["adapter_state_on_prefix"] == "off"
    assert pipeline["adapter_state_after_boundary"] == "expert_only"
    assert pipeline["private_tail_kv_required"] is True
    assert pipeline["full_generation_kv_shared_claimed"] is False
    assert pipeline["mid_request_generated_trigger_switch_claimed"] is False


def test_bound_fixture_preflight_authenticates_shared_core_and_is_non_authorizing(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    report = preflight_profile(root)
    assert report["profile_ready"] is True
    assert report["profile_id"] == "frozen-prefix-qreader-v2"
    assert report["authenticated_dependency_count"] == 39
    assert report["shared_core_blob_count"] == 10
    assert report["companion_dependency_count"] == 26
    assert report["dependency_dag_acyclic"] is True
    assert report["diagnostic_fixture_record_count"] == 1000
    assert report["gemma_sequence_length"] == 768
    assert report["gemma_strict_no_truncation"] is True
    for key in (
        "canonical_gold_mutated",
        "credentials_read",
        "gold_bodies_read",
        "heldout_bodies_read",
        "live_authorized",
        "training_authorized",
        "formal_training_authorized",
        "release_authorized",
    ):
        assert report[key] is False
    for key in (
        "provider_requests",
        "model_loads",
        "gpu_requests",
        "network_requests",
    ):
        assert report[key] == 0


def test_freeze_is_atomic_hash_bound_and_non_authorizing(tmp_path: Path) -> None:
    from jsonschema import Draft202012Validator

    root = _fixture_root(tmp_path)
    output = root / "artifacts/frozen-prefix-qreader-profile-v2"
    result = freeze_profile(root, PROFILE_PATH, output)
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest_sha256 = sha256(manifest_bytes).hexdigest()
    assert (output / "manifest.json.sha256").read_bytes() == (
        f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    manifest = json.loads(manifest_bytes)
    Draft202012Validator(_load_json(root / FREEZE_MANIFEST_SCHEMA_PATH)).validate(
        manifest
    )
    assert result["manifest_sha256"] == manifest_sha256
    assert manifest["authenticated_dependency_count"] == 39
    assert manifest["shared_core_blob_count"] == 10
    assert manifest["all_dependencies_bound"] is True
    assert manifest["canonical_gold_written"] is False
    assert manifest["canonical_gold_mutated"] is False
    assert manifest["training_authorized"] is False
    assert manifest["formal_training_authorized"] is False
    assert manifest["release_authorized"] is False
    with pytest.raises(FrozenPrefixQReaderProfileError, match="freeze_output_exists"):
        freeze_profile(root, PROFILE_PATH, output)


def test_shared_core_must_equal_v1_physical_identities(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    value = _profile_value(root)
    dependencies = value["dependencies"]
    assert isinstance(dependencies, list)
    anchor = dependencies[0]
    assert isinstance(anchor, dict)
    anchor_path = root / str(anchor["path"])
    anchor_path.write_bytes(anchor_path.read_bytes() + b"\n")
    digest, size = _identity(anchor_path)
    anchor["sha256"] = digest
    anchor["bytes"] = size
    _write_json(root / PROFILE_PATH, value)
    with pytest.raises(
        FrozenPrefixQReaderProfileError,
        match="shared_core_identity_mismatch",
    ):
        load_profile(root)


def test_dependency_bytes_and_terminal_recheck_fail_closed(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    loaded = load_profile(root)
    dependency = root / EXPECTED_DEPENDENCY_PATHS["taskboard_projector_config"]
    dependency.write_bytes(dependency.read_bytes() + b"\n")
    with pytest.raises(
        FrozenPrefixQReaderProfileError,
        match="taskboard_projector_config_changed_during_operation",
    ):
        loaded.recheck()
    with pytest.raises(
        FrozenPrefixQReaderProfileError,
        match="taskboard_projector_config_binding_invalid",
    ):
        load_profile(root)


def test_placeholder_or_missing_companion_never_becomes_ready(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    value = _profile_value(root)
    dependencies = value["dependencies"]
    assert isinstance(dependencies, list)
    item = next(
        raw
        for raw in dependencies
        if isinstance(raw, dict)
        and raw.get("role") == "training_view_materializer_config"
    )
    item["state"] = "pending_physical_freeze"
    item["sha256"] = PLACEHOLDER
    item["bytes"] = 0
    _write_json(root / PROFILE_PATH, value)
    with pytest.raises(
        FrozenPrefixQReaderProfileError,
        match="profile_dependencies_pending_physical_freeze",
    ):
        load_profile(root)


def test_schema_rejects_authorization_or_kv_overclaim(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    value = _profile_value(root)
    authorization = value["authorization"]
    pipeline = value["post_gold_pipeline"]
    assert isinstance(authorization, dict)
    assert isinstance(pipeline, dict)
    authorization["training_authorized"] = True
    pipeline["full_generation_kv_shared_claimed"] = True
    _write_json(root / PROFILE_PATH, value)
    with pytest.raises(
        FrozenPrefixQReaderProfileError,
        match="profile_schema_validation_failed",
    ):
        load_profile(root)


def test_published_schemas_cross_bind_every_role_to_one_canonical_path(
    tmp_path: Path,
) -> None:
    from jsonschema import Draft202012Validator, ValidationError

    profile = _profile_value()
    dependencies = profile["dependencies"]
    assert isinstance(dependencies, list)
    first = dependencies[0]
    second = dependencies[1]
    assert isinstance(first, dict) and isinstance(second, dict)
    first["path"] = second["path"]
    with pytest.raises(ValidationError):
        Draft202012Validator(_load_json(SCHEMA)).validate(profile)

    repeated_profile = _profile_value()
    repeated_dependencies = repeated_profile["dependencies"]
    assert isinstance(repeated_dependencies, list)
    repeated_profile["dependencies"] = [
        repeated_dependencies[0] for _ in repeated_dependencies
    ]
    with pytest.raises(ValidationError):
        Draft202012Validator(_load_json(SCHEMA)).validate(repeated_profile)
    missing_profile = _profile_value()
    missing_dependencies = missing_profile["dependencies"]
    assert isinstance(missing_dependencies, list)
    missing_profile["dependencies"] = missing_dependencies[:-1]
    with pytest.raises(ValidationError):
        Draft202012Validator(_load_json(SCHEMA)).validate(missing_profile)

    root = _fixture_root(tmp_path)
    output = root / "artifacts/profile-schema-cross-binding"
    freeze_profile(root, PROFILE_PATH, output)
    manifest = _load_json(output / "manifest.json")
    manifest_dependencies = manifest["dependencies"]
    assert isinstance(manifest_dependencies, list)
    first_manifest = manifest_dependencies[0]
    second_manifest = manifest_dependencies[1]
    assert isinstance(first_manifest, dict)
    assert isinstance(second_manifest, dict)
    first_manifest["path"] = second_manifest["path"]
    with pytest.raises(ValidationError):
        Draft202012Validator(_load_json(root / FREEZE_MANIFEST_SCHEMA_PATH)).validate(
            manifest
        )
    repeated_manifest = _load_json(output / "manifest.json")
    repeated_manifest_dependencies = repeated_manifest["dependencies"]
    assert isinstance(repeated_manifest_dependencies, list)
    repeated_manifest["dependencies"] = [
        repeated_manifest_dependencies[0] for _ in repeated_manifest_dependencies
    ]
    with pytest.raises(ValidationError):
        Draft202012Validator(_load_json(root / FREEZE_MANIFEST_SCHEMA_PATH)).validate(
            repeated_manifest
        )
    missing_manifest = _load_json(output / "manifest.json")
    missing_manifest_dependencies = missing_manifest["dependencies"]
    assert isinstance(missing_manifest_dependencies, list)
    missing_manifest["dependencies"] = missing_manifest_dependencies[:-1]
    with pytest.raises(ValidationError):
        Draft202012Validator(_load_json(root / FREEZE_MANIFEST_SCHEMA_PATH)).validate(
            missing_manifest
        )


def test_duplicate_json_key_is_rejected_before_schema_validation(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    path = root / PROFILE_PATH
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '  "profile_id": "frozen-prefix-qreader-v2",',
        (
            '  "profile_id": "frozen-prefix-qreader-v2",\n'
            '  "profile_id": "frozen-prefix-qreader-v2",'
        ),
        1,
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    with pytest.raises(FrozenPrefixQReaderProfileError, match="profile_invalid"):
        load_profile(root)


def test_post_publish_recheck_failure_removes_new_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_root(tmp_path)
    output = root / "artifacts/post-publish-recheck"
    original = profile_module.AuthenticatedFrozenPrefixQReaderProfile.recheck
    calls = 0

    def fail_terminal_recheck(
        profile: profile_module.AuthenticatedFrozenPrefixQReaderProfile,
    ) -> None:
        nonlocal calls
        calls += 1
        original(profile)
        if calls == 3:
            raise FrozenPrefixQReaderProfileError("profile_changed_during_operation")

    monkeypatch.setattr(
        profile_module.AuthenticatedFrozenPrefixQReaderProfile,
        "recheck",
        fail_terminal_recheck,
    )
    with pytest.raises(
        FrozenPrefixQReaderProfileError,
        match="profile_changed_during_operation",
    ):
        freeze_profile(root, PROFILE_PATH, output)
    assert calls == 3
    assert not output.exists()


def test_dependency_dag_has_unique_paths_and_no_reverse_release_binding() -> None:
    assert tuple(EXPECTED_DEPENDENCY_PATHS) == EXPECTED_DEPENDENCY_ROLES
    assert set(DEPENDENCY_LAYERS) == set(EXPECTED_DEPENDENCY_ROLES)
    assert len(set(EXPECTED_DEPENDENCY_PATHS.values())) == len(
        EXPECTED_DEPENDENCY_PATHS
    )
    assert "release_overlay_config" not in EXPECTED_DEPENDENCY_ROLES
    assert PROFILE_PATH.as_posix() not in EXPECTED_DEPENDENCY_PATHS.values()
    assert (
        FREEZE_MANIFEST_SCHEMA_PATH.as_posix() not in EXPECTED_DEPENDENCY_PATHS.values()
    )
    assert tuple(EXPECTED_DEPENDENCY_ROLES[:10]) == SHARED_CORE_ROLES
    layers = tuple(DEPENDENCY_LAYERS[role] for role in EXPECTED_DEPENDENCY_ROLES)
    assert layers == tuple(sorted(layers))


def test_core_command_is_description_only_and_never_adds_confirm_live(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    value = core_command_metadata(root)
    assert value["executes_command"] is False
    assert value["confirm_live_included"] is False
    assert "-ConfirmLive" not in value["command"]
    assert value["provider_requests"] == 0
    assert value["live_authorized"] is False
    assert value["training_authorized"] is False


def test_checked_in_profile_preflight_is_ready_or_explicitly_pending() -> None:
    value = _profile_value()
    dependencies = value["dependencies"]
    assert isinstance(dependencies, list)
    has_placeholder = any(
        isinstance(item, dict) and item.get("sha256") == PLACEHOLDER
        for item in dependencies
    )
    if has_placeholder:
        with pytest.raises(
            FrozenPrefixQReaderProfileError,
            match="profile_dependencies_pending_physical_freeze",
        ):
            preflight_profile(ROOT)
    else:
        assert preflight_profile(ROOT)["profile_ready"] is True


def test_cli_is_zero_request_and_fails_closed_while_companions_pending() -> None:
    value = _profile_value()
    dependencies = value["dependencies"]
    assert isinstance(dependencies, list)
    has_placeholder = any(
        isinstance(item, dict) and item.get("sha256") == PLACEHOLDER
        for item in dependencies
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "preflight",
            "--profile",
            PROFILE_PATH.as_posix(),
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if has_placeholder:
        assert completed.returncode == 2
        assert "profile_dependencies_pending_physical_freeze" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["provider_requests"] == 0
        assert report["gold_bodies_read"] is False
        assert report["heldout_bodies_read"] is False
        assert report["training_authorized"] is False


def test_v1_profile_contract_files_are_byte_identical_to_branch_point() -> None:
    expected = {
        "configs/orchestration/distillation_pipeline_profile.schema.json": (
            "98a181de536dfefda22c4bd35cba4ebd81d8c8ea67235465becbac665e4d4f94"
        ),
        "configs/orchestration/distillation_profile_freeze_manifest.schema.json": (
            "ef81bab1d5c4066077031846dd3cc6cadfbddcb62cee45fff91a76e9c48a8343"
        ),
        "configs/orchestration/profiles/task_level_moe_lora_v1.json": (
            "99dc24c35aa32bd1b121715e5d7b6d7d3652957fe4cb39f6fa8b7e2480ea4aa4"
        ),
        "src/anchor_mvp/swebench/distillation_profile.py": (
            "a274b78f16148b2ef83579dd6651d84e2fdda450b6dcbfc5ba16bb29b471c594"
        ),
        "scripts/data/run_distillation_profile.py": (
            "2499981b6438d233331ff68bf0359e227e2e6730103a959e134cf00ed276b3a6"
        ),
        "tests/test_distillation_profiles.py": (
            "2a5ba7a5d0906f8c9f6562502c0fbafd7125b07d6dcf53718eb43db8392c04bb"
        ),
        "docs/distillation_pipeline_profiles.md": (
            "6453bf3b69f6e982bcf4d72e2c6ce10f0c118bcb1d74be60e6f7343e51835726"
        ),
        "docs/distillation_pipeline_profiles.zh-CN.md": (
            "71d7ec1881a5e7bcb8667916d3af1f0492053001da9ff3a0487b4916263dd2c5"
        ),
    }
    for relative, expected_sha256 in expected.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected_sha256
