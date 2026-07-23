from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from anchor_mvp.swebench.distillation_profile import (
    DistillationProfileError,
    PROFILE_PATH,
    freeze_profile,
    load_profile,
    preflight_profile,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs/orchestration/distillation_pipeline_profile.schema.json"
FREEZE_SCHEMA = (
    ROOT / "configs/orchestration/distillation_profile_freeze_manifest.schema.json"
)
PROFILE = ROOT / PROFILE_PATH
RUNNER = ROOT / "scripts/data/run_distillation_profile.py"


def _profile_value(root: Path = ROOT) -> dict[str, object]:
    value = json.loads((root / PROFILE_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    profile = _profile_value()
    relative_paths = {
        PROFILE_PATH.as_posix(),
        str(profile["profile_schema"]["path"]),
        str(profile["freeze_manifest_schema"]["path"]),
        *(
            str(item["path"])
            for item in profile["dependencies"]
            if isinstance(item, dict)
        ),
    }
    for relative in sorted(relative_paths):
        source = ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root


def _write_profile(root: Path, value: object) -> None:
    destination = root / PROFILE_PATH
    destination.write_text(
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


def test_checked_in_profile_is_strict_and_content_free() -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    profile = _profile_value()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(profile)

    report = preflight_profile(ROOT)
    assert report == {
        "schema_version": "anchor.distillation-profile-preflight.v1",
        "profile_ready": True,
        "profile_id": "task-level-moe-lora-v1",
        "profile_sha256": sha256(PROFILE.read_bytes()).hexdigest(),
        "profile_schema_sha256": sha256(SCHEMA.read_bytes()).hexdigest(),
        "freeze_manifest_schema_sha256": sha256(FREEZE_SCHEMA.read_bytes()).hexdigest(),
        "execution_core_id": "anchor.swebench-five-stage-execution-core.v1",
        "post_gold_view_id": "anchor.task-level-moe-lora-view.v1",
        "full_bank_config_path": "configs/data/swebench_full_bank.formal.yaml",
        "coordinator_config_path": ("configs/data/swebench_five_stage.ccswitch.yaml"),
        "authenticated_dependency_count": 14,
        "canonical_gold_mutated": False,
        "provider_requests": 0,
        "credentials_read": False,
        "gold_bodies_read": False,
        "heldout_bodies_read": False,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "live_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }


def test_profile_freeze_is_atomic_hash_bound_and_non_authorizing(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    output = root / "artifacts/profile-freeze-v1"
    result = freeze_profile(root, PROFILE_PATH, output)

    manifest_path = output / "manifest.json"
    sidecar_path = output / "manifest.json.sha256"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256(manifest_bytes).hexdigest()
    assert sidecar_path.read_bytes() == (
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    manifest = json.loads(manifest_bytes)
    assert result["manifest_sha256"] == manifest_sha
    assert result["status"] == "frozen_non_authorizing"
    assert manifest["authenticated_dependency_count"] == 14
    from jsonschema import Draft202012Validator, ValidationError

    freeze_schema = json.loads(FREEZE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(freeze_schema)
    validator = Draft202012Validator(freeze_schema)
    validator.validate(manifest)
    tampered = json.loads(json.dumps(manifest))
    tampered["dependencies"][0]["path"] = tampered["dependencies"][1]["path"]
    with pytest.raises(ValidationError):
        validator.validate(tampered)
    assert manifest["profile_scope"] == "post_canonical_gold_view_only"
    for key in (
        "canonical_gold_written",
        "canonical_gold_mutated",
        "credentials_read",
        "gold_bodies_read",
        "heldout_bodies_read",
        "live_authorized",
        "training_authorized",
        "formal_training_authorized",
        "release_authorized",
    ):
        assert manifest[key] is False
    for key in (
        "provider_requests",
        "model_loads",
        "gpu_requests",
        "network_requests",
    ):
        assert manifest[key] == 0
    with pytest.raises(DistillationProfileError, match="freeze_output_exists"):
        freeze_profile(root, PROFILE_PATH, output)


def test_dependency_bytes_and_terminal_recheck_fail_closed(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    loaded = load_profile(root)
    anchor = root / "anchor.ps1"
    anchor.write_bytes(anchor.read_bytes() + b"\n")
    with pytest.raises(
        DistillationProfileError,
        match="anchor_launcher_changed_during_operation",
    ):
        loaded.recheck()
    with pytest.raises(
        DistillationProfileError,
        match="anchor_launcher_binding_invalid",
    ):
        load_profile(root)


def test_freeze_schema_bytes_and_terminal_recheck_fail_closed(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    loaded = load_profile(root)
    schema_path = root / FREEZE_SCHEMA.relative_to(ROOT)
    schema_path.write_bytes(schema_path.read_bytes() + b"\n")
    with pytest.raises(
        DistillationProfileError,
        match="freeze_manifest_schema_changed_during_operation",
    ):
        loaded.recheck()
    with pytest.raises(
        DistillationProfileError,
        match="freeze_manifest_schema_binding_invalid",
    ):
        load_profile(root)


def test_unknown_profile_field_is_rejected_by_published_schema(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    value = _profile_value(root)
    value["unexpected"] = True
    _write_profile(root, value)
    with pytest.raises(
        DistillationProfileError,
        match="profile_schema_validation_failed",
    ):
        load_profile(root)


def test_authorization_or_qreader_promotion_is_rejected(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    value = _profile_value(root)
    value["authorization"]["training_authorized"] = True
    value["post_gold_view"]["frozen_prefix_qreader_claimed"] = True
    _write_profile(root, value)
    with pytest.raises(
        DistillationProfileError,
        match="profile_schema_validation_failed",
    ):
        load_profile(root)


def test_dependency_role_cannot_redirect_to_another_project_file(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    value = _profile_value(root)
    dependencies = value["dependencies"]
    assert isinstance(dependencies, list)
    anchor = dependencies[0]
    full_bank = dependencies[1]
    assert isinstance(anchor, dict) and isinstance(full_bank, dict)
    anchor["path"] = full_bank["path"]
    anchor["sha256"] = full_bank["sha256"]
    anchor["bytes"] = full_bank["bytes"]
    _write_profile(root, value)
    with pytest.raises(
        DistillationProfileError,
        match="profile_schema_validation_failed",
    ):
        load_profile(root)


def test_duplicate_json_key_is_rejected_before_schema_validation(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    path = root / PROFILE_PATH
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '  "profile_id": "task-level-moe-lora-v1",',
        (
            '  "profile_id": "task-level-moe-lora-v1",\n'
            '  "profile_id": "task-level-moe-lora-v1",'
        ),
        1,
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    with pytest.raises(DistillationProfileError, match="profile_invalid"):
        load_profile(root)


def test_dependency_symlink_or_reparse_is_rejected(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    anchor = root / "anchor.ps1"
    target = root / "anchor-target.ps1"
    anchor.replace(target)
    try:
        anchor.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    with pytest.raises(
        DistillationProfileError,
        match="anchor_launcher_binding_invalid",
    ):
        load_profile(root)


def test_absolute_profile_symlink_is_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    canonical = root / PROFILE_PATH
    alias = canonical.with_name("profile-alias.json")
    try:
        alias.symlink_to(canonical)
    except OSError as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    with pytest.raises(DistillationProfileError, match="profile_path_invalid"):
        load_profile(root, alias)


def test_freeze_rejects_internal_symlink_parent(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    real = root / "artifacts-real"
    real.mkdir()
    link = root / "artifacts"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink capability unavailable: {exc}")
    with pytest.raises(
        DistillationProfileError,
        match="freeze_output_path_invalid",
    ):
        freeze_profile(root, PROFILE_PATH, link / "profile-freeze-v1")


def test_cli_preflight_is_exactly_metadata_only() -> None:
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
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["profile_ready"] is True
    assert report["provider_requests"] == 0
    assert report["gold_bodies_read"] is False
    assert report["heldout_bodies_read"] is False
    assert report["live_authorized"] is False
    assert report["training_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_anchor_rejects_profile_and_direct_config_conflict() -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    completed = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "anchor.ps1"),
            "-Action",
            "distill-swebench",
            "-DistillationProfile",
            "task-level-moe-lora-v1",
            "-SWEConfig",
            "configs/data/swebench_full_bank.formal.yaml",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    combined = completed.stdout + completed.stderr
    assert "distillation_profile_conflicts_with_direct_swe_config" in combined


def test_anchor_default_non_profile_action_remains_available() -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    completed = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "anchor.ps1"),
            "-Action",
            "docs",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "distillation_profile" not in (completed.stdout + completed.stderr)
