from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from anchor_mvp.research import qwen_train_prerequisite_consumer_v2 as overlay


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/qwen_train_prerequisite_consumer_v2.yaml"
MANIFEST = ROOT / "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json"
MANIFEST_SIDECAR = MANIFEST.with_name("manifest.json.sha256")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_bytes())


def _release_snapshots() -> dict[str, overlay._BytesSnapshot]:
    paths = {
        "companion_config": (
            "configs/research/qwen_toy_prerequisite_companion_v2.json"
        ),
        "companion_schema": (
            "configs/research/qwen_toy_prerequisite_companion_v2.schema.json"
        ),
        "companion_implementation": (
            "src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py"
        ),
        "companion_manifest": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json"
        ),
        "companion_manifest_sidecar": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json.sha256"
        ),
        "copied_source_receipt": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
            "qwen_request_local_trigger_receipt_v2/receipt.json"
        ),
        "copied_source_sidecar": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
            "qwen_request_local_trigger_receipt_v2/receipt.json.sha256"
        ),
    }
    return {
        role: overlay._read_bytes_snapshot(ROOT / path, "snapshot_failed")
        for role, path in paths.items()
    }


def test_conjunctive_overlay_authenticates_trigger_but_remains_blocked() -> None:
    decision = overlay.evaluate_prerequisites(CONFIG)
    assert decision["schema_version"] == overlay.DECISION_VERSION
    assert decision["status"] == "blocked"
    assert decision["effective_condition"] == (
        "frozen_v1_and_authenticated_companion_v2"
    )
    assert decision["training_authorized"] is False
    assert decision["formal_training_authorized"] is False
    assert decision["trigger_materialization"] == {
        "status": "ready_diagnostic_only",
        "frozen_toy_v1_status": "pending_request_local_materialization",
        "companion_v2_status": "ready_diagnostic_only",
        "total_tokens": 44,
        "trigger_span_zero_based_exclusive": {"start": 25, "end": 33},
        "trigger_span_width": 8,
        "activation_semantics": "next_request_input_activation_only",
    }
    assert decision["inventory_status"] == {
        "coverage_ready_count": 2,
        "coverage_total": 6,
        "ready_source_classes": ["swebench_source", "heldout"],
        "missing_source_classes": list(overlay.MISSING_SOURCE_CLASSES),
        "zero_intersection_claimed": False,
        "v1_attestation_emitted": False,
    }
    assert decision["audit"] == {
        "protected_dataset_files_read": 0,
        "protected_content_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }
    assert decision["on_disk_consumer_implementation_sha256"] == _sha(
        ROOT / "src/anchor_mvp/research/qwen_train_prerequisite_consumer_v2.py"
    )


def test_frozen_v1_consumers_are_not_modified() -> None:
    expected = {
        "configs/research/qwen_train_prerequisite_consumer_v1.yaml": (
            overlay.V1_CONFIG_SHA256
        ),
        "src/anchor_mvp/research/qwen_train_prerequisite_consumer.py": (
            overlay.V1_IMPLEMENTATION_SHA256
        ),
        "configs/research/qwen_toy_prerequisite_consumer_v1.yaml": (
            overlay.TOY_V1_CONFIG_SHA256
        ),
        "src/anchor_mvp/research/qwen_toy_prerequisite_consumer.py": (
            overlay.TOY_V1_IMPLEMENTATION_SHA256
        ),
    }
    assert {path: _sha(ROOT / path) for path in expected} == expected


def test_overlay_and_companion_physical_identities_are_exact() -> None:
    assert _sha(CONFIG) == overlay.CONFIG_SHA256
    expected = {
        "configs/research/qwen_toy_prerequisite_companion_v2.json": (
            overlay.COMPANION_CONFIG_SHA256
        ),
        "configs/research/qwen_toy_prerequisite_companion_v2.schema.json": (
            overlay.COMPANION_SCHEMA_SHA256
        ),
        "src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py": (
            overlay.COMPANION_IMPLEMENTATION_SHA256
        ),
        "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json": (
            overlay.COMPANION_MANIFEST_SHA256
        ),
        "fixtures/research/qwen_toy_prerequisite_companion_v2/"
        "manifest.json.sha256": overlay.COMPANION_SIDECAR_SHA256,
        "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
        "qwen_request_local_trigger_receipt_v2/receipt.json": (
            overlay.SOURCE_RECEIPT_SHA256
        ),
        "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
        "qwen_request_local_trigger_receipt_v2/receipt.json.sha256": (
            overlay.SOURCE_SIDECAR_SHA256
        ),
    }
    assert {path: _sha(ROOT / path) for path in expected} == expected


def test_mandatory_sidecars_are_exact() -> None:
    expected = f"{overlay.COMPANION_MANIFEST_SHA256}  manifest.json\n".encode("ascii")
    assert MANIFEST_SIDECAR.read_bytes() == expected
    receipt = (
        ROOT / "fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json"
    )
    copied = ROOT / (
        "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
        "qwen_request_local_trigger_receipt_v2/receipt.json"
    )
    assert copied.read_bytes() == receipt.read_bytes()
    assert copied.with_name("receipt.json.sha256").read_bytes() == (
        receipt.with_name("receipt.json.sha256").read_bytes()
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claims", "training_authorized"), True),
        (("claims", "formal"), True),
        (("claims", "trigger_materialization_ready"), False),
        (("proof", "zero_intersection_claimed"), True),
        (("proof", "v1_attestation_emitted"), True),
        (("inventory_status", "coverage_ready_count"), 6),
        (("trigger_materialization", "raw_token_ids_emitted"), True),
        (("trigger_materialization", "planner_request1_private_kv_reused"), True),
    ],
)
def test_semantic_promotions_fail_closed(path: tuple[str, str], value: object) -> None:
    manifest = _manifest()
    section = manifest[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    with pytest.raises(overlay.QwenPrerequisiteOverlayError):
        overlay._validate_companion_semantics(manifest)


def test_config_binding_or_policy_drift_fails_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    config["bindings"]["companion_manifest_sha256"] = "0" * 64
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError, match="overlay_bindings_drift"
    ):
        overlay._validate_config(config)
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    config["policy"]["training_authorized"] = True
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError, match="overlay_policy_drift"
    ):
        overlay._validate_config(config)


def test_snapshot_detects_mid_evaluation_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay, "_REPOSITORY_ROOT", tmp_path)
    target = tmp_path / "identity.json"
    target.write_bytes(b'{"value":1}\n')
    snapshot = overlay._read_bytes_snapshot(target, "snapshot_failed")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"value":2}\n')
    os.replace(replacement, target)
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match="changed"):
        snapshot.assert_unchanged("changed")


def test_symlink_ancestor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(physical, linked, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    monkeypatch.setattr(overlay, "_REPOSITORY_ROOT", tmp_path)
    target = linked / "contract.json"
    target.write_bytes(b"{}\n")
    with pytest.raises(overlay.QwenPrerequisiteOverlayError):
        overlay._read_bytes_snapshot(target, "symlink_rejected")


@pytest.mark.parametrize("kind", ["config", "artifact", "receipt"])
def test_lexical_symlink_ancestors_are_rejected_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(physical, linked, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    monkeypatch.setattr(overlay, "_REPOSITORY_ROOT", tmp_path)
    if kind == "config":
        config = physical / "config.yaml"
        config.write_bytes(b"schema_version: test\n")
        monkeypatch.setattr(overlay, "CONFIG_PATH", "linked/config.yaml")
        with pytest.raises(
            overlay.QwenPrerequisiteOverlayError,
            match="overlay_config_path_invalid",
        ):
            overlay._requested_config_path("linked/config.yaml")
    else:
        target = physical / ("artifact" if kind == "artifact" else "receipt.json")
        target.write_bytes(b"{}\n")
        with pytest.raises(overlay.QwenPrerequisiteOverlayError, match="path_invalid"):
            overlay._safe_path(f"linked/{target.name}", f"{kind}_path_invalid")


def test_lexical_reparse_check_happens_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked = tmp_path / "linked"
    linked.mkdir()
    monkeypatch.setattr(overlay, "_REPOSITORY_ROOT", tmp_path)
    original = overlay._is_reparse_or_symlink

    def mark_lexical_link(path: Path) -> bool:
        if path == linked:
            return True
        return original(path)

    monkeypatch.setattr(overlay, "_is_reparse_or_symlink", mark_lexical_link)
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match="path_invalid"):
        overlay._safe_path("linked/contract.json", "path_invalid")


@pytest.mark.parametrize("consumer", ["train", "toy"])
def test_frozen_v1_decisions_cannot_be_promoted(
    monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    if consumer == "train":
        monkeypatch.setattr(
            overlay,
            "_evaluate_frozen_v1",
            lambda *_: {
                "schema_version": "anchor.qwen-train-prerequisite-decision.v1",
                "status": "ready",
                "training_authorized": False,
                "formal_training_authorized": False,
            },
        )
        expected = "frozen_v1_decision_drift"
    else:
        monkeypatch.setattr(
            overlay,
            "_evaluate_frozen_toy_v1",
            lambda *_: {
                "schema_version": ("anchor.qwen-toy-prerequisite-consumer-decision.v1"),
                "status": "ready",
                "training_authorized": False,
                "formal_training_authorized": False,
                "zero_intersection_claimed": False,
                "v1_attestation_emitted": False,
            },
        )
        expected = "frozen_toy_v1_decision_drift"
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match=expected):
        overlay.evaluate_prerequisites(CONFIG)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_tokens", 48),
        ("trigger_span_width", 7),
        ("boundary_overhang.leading_utf8_bytes", 1),
        ("boundary_overhang.trailing_codepoints", 0),
    ],
)
def test_trigger_shape_or_overhang_drift_fails_closed(field: str, value: int) -> None:
    manifest = _manifest()
    trigger = manifest["trigger_materialization"]
    assert isinstance(trigger, dict)
    if field.startswith("boundary_overhang."):
        overhang = trigger["boundary_overhang"]
        assert isinstance(overhang, dict)
        overhang[field.split(".", 1)[1]] = value
    else:
        trigger[field] = value
    with pytest.raises(overlay.QwenPrerequisiteOverlayError):
        overlay._validate_companion_semantics(manifest)


def test_missing_companion_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = overlay._read_bytes_snapshot

    def fail_manifest(path: Path, code: str, **kwargs: object):
        if path == MANIFEST:
            raise overlay.QwenPrerequisiteOverlayError("missing_companion")
        return original(path, code, **kwargs)

    monkeypatch.setattr(overlay, "_read_bytes_snapshot", fail_manifest)
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match="missing_companion"):
        overlay.evaluate_prerequisites(CONFIG)


@pytest.mark.parametrize(
    "missing",
    [
        MANIFEST_SIDECAR,
        ROOT
        / (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
            "qwen_request_local_trigger_receipt_v2/receipt.json.sha256"
        ),
        ROOT
        / "fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json.sha256",
    ],
)
def test_each_mandatory_sidecar_missing_is_fail_closed(
    missing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = overlay._read_bytes_snapshot

    def fail_sidecar(path: Path, code: str, **kwargs: object):
        if path == missing:
            raise overlay.QwenPrerequisiteOverlayError("sidecar_missing")
        return original(path, code, **kwargs)

    monkeypatch.setattr(overlay, "_read_bytes_snapshot", fail_sidecar)
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match="sidecar_missing"):
        overlay.evaluate_prerequisites(CONFIG)


def test_sidecar_content_tamper_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = overlay._read_bytes_snapshot

    def tamper(path: Path, code: str, **kwargs: object):
        snapshot = original(path, code, **kwargs)
        if path == MANIFEST_SIDECAR:
            data = b"0" * 64 + b"  manifest.json\n"
            return overlay._BytesSnapshot(
                path=snapshot.path,
                data=data,
                sha256=hashlib.sha256(data).hexdigest(),
                identity=snapshot.identity,
            )
        return snapshot

    monkeypatch.setattr(overlay, "_read_bytes_snapshot", tamper)
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError,
        match="companion_manifest_sidecar_sha256_mismatch",
    ):
        overlay.evaluate_prerequisites(CONFIG)


def test_companion_audit_failure_cannot_be_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_audit(*_: object, **__: object) -> dict[str, object]:
        raise overlay.QwenPrerequisiteOverlayError("audit_failed")

    monkeypatch.setattr(overlay, "_audit_companion", fail_audit)
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match="audit_failed"):
        overlay.evaluate_prerequisites(CONFIG)


def test_cli_overrides_accept_only_frozen_identities() -> None:
    accepted = argparse.Namespace(
        frozen_v1_config=("configs/research/qwen_train_prerequisite_consumer_v1.yaml"),
        frozen_v1_config_sha256=overlay.V1_CONFIG_SHA256,
        frozen_toy_v1_config=(
            "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"
        ),
        frozen_toy_v1_config_sha256=overlay.TOY_V1_CONFIG_SHA256,
        companion_manifest=(
            "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json"
        ),
        companion_manifest_sha256=overlay.COMPANION_MANIFEST_SHA256,
        companion_manifest_sidecar_sha256=overlay.COMPANION_SIDECAR_SHA256,
        producer_companion_release_commit=overlay.COMPANION_RELEASE_COMMIT,
    )
    overlay._validate_cli_overrides(accepted)
    accepted.companion_manifest_sha256 = "0" * 64
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError, match="overlay_cli_override_drift"
    ):
        overlay._validate_cli_overrides(accepted)


def test_cli_override_groups_are_all_or_none() -> None:
    incomplete = argparse.Namespace(
        frozen_v1_config=("configs/research/qwen_train_prerequisite_consumer_v1.yaml"),
        frozen_v1_config_sha256=None,
        frozen_toy_v1_config=None,
        frozen_toy_v1_config_sha256=None,
        companion_manifest=None,
        companion_manifest_sha256=None,
        companion_manifest_sidecar_sha256=None,
        producer_companion_release_commit=None,
    )
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError,
        match="overlay_cli_override_incomplete",
    ):
        overlay._validate_cli_overrides(incomplete)


def test_release_commit_tree_parent_and_blobs_are_authenticated() -> None:
    decision = overlay.evaluate_prerequisites(CONFIG)
    assert decision["bindings"]["producer_companion_release_commit"] == (
        overlay.COMPANION_RELEASE_COMMIT
    )
    assert decision["bindings"]["producer_companion_release_tree"] == (
        overlay.COMPANION_RELEASE_TREE
    )


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("replace", "companion_git_replace_refs_present"),
        ("tree", "companion_release_commit_identity_mismatch"),
        ("blob", "companion_release_blob_mismatch"),
    ],
)
def test_release_git_control_or_identity_drift_fails_closed(
    mode: str, error: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = _release_snapshots()
    original = overlay._run_git

    def drift(
        arguments: list[str] | tuple[str, ...], code: str, *, binary: bool = False
    ) -> bytes | str:
        if mode == "replace" and arguments[0] == "for-each-ref":
            return "refs/replace/unsafe\n"
        if mode == "tree" and arguments[:2] == [
            "rev-parse",
            f"{overlay.COMPANION_RELEASE_COMMIT}^{{tree}}",
        ]:
            return "0" * 40 + "\n"
        if mode == "blob" and arguments[:2] == ["cat-file", "blob"]:
            return b"drift"
        return original(arguments, code, binary=binary)

    monkeypatch.setattr(overlay, "_run_git", drift)
    with pytest.raises(overlay.QwenPrerequisiteOverlayError, match=error):
        overlay._validate_release_git_blobs(snapshots)


def test_nonempty_grafts_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = _release_snapshots()
    grafts = tmp_path / "grafts"
    grafts.write_text("unsafe\n", encoding="ascii")
    original = overlay._run_git

    def point_at_grafts(
        arguments: list[str] | tuple[str, ...], code: str, *, binary: bool = False
    ) -> bytes | str:
        if arguments[:3] == ["rev-parse", "--git-path", "info/grafts"]:
            return str(grafts) + "\n"
        return original(arguments, code, binary=binary)

    monkeypatch.setattr(overlay, "_run_git", point_at_grafts)
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError, match="companion_git_grafts_present"
    ):
        overlay._validate_release_git_blobs(snapshots)


def test_end_to_end_final_snapshot_recheck_detects_inode_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = overlay._read_bytes_snapshot
    reads = 0

    def swap_on_recheck(path: Path, code: str, **kwargs: object):
        nonlocal reads
        snapshot = original(path, code, **kwargs)
        if path == MANIFEST:
            reads += 1
            if reads == 2:
                identity = (
                    snapshot.identity[0],
                    snapshot.identity[1] + 1,
                    snapshot.identity[2],
                    snapshot.identity[3],
                )
                return overlay._BytesSnapshot(
                    path=snapshot.path,
                    data=snapshot.data,
                    sha256=snapshot.sha256,
                    identity=identity,
                )
        return snapshot

    monkeypatch.setattr(overlay, "_read_bytes_snapshot", swap_on_recheck)
    with pytest.raises(
        overlay.QwenPrerequisiteOverlayError,
        match="companion_manifest_changed_during_evaluation",
    ):
        overlay.evaluate_prerequisites(CONFIG)


def test_cli_returns_blocked_json_and_exit_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        overlay,
        "evaluate_prerequisites",
        lambda _: {"status": "blocked", "training_authorized": False},
    )
    code = overlay.main(["--config", overlay.CONFIG_PATH])
    assert code == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "training_authorized": False,
    }
