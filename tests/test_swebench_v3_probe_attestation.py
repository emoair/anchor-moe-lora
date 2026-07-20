from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from anchor_mvp.tooling import swebench_execution_v3 as execution


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _sha(value)


def _write_json(path: Path, value: Any) -> str:
    return _write(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode(),
    )


def _fixture(root: Path) -> tuple[dict[str, Any], str, Path]:
    lock_sha = "a" * 64
    baseline = "b" * 40
    binary_path = root / "artifacts/opencode/linux/opencode-anchor"
    binary_sha = _write(binary_path, b"binary")
    patch_manifest_path = root / "patches/opencode/patch-manifest.json"
    patch_manifest_sha = _write_json(
        patch_manifest_path,
        {
            "baseline_commit": baseline,
            "tool_contract_version": execution.EXECUTION_TOOL_CONTRACT_V3,
        },
    )
    bundle_path = root / "artifacts/opencode/bundle-manifest.json"
    bundle_sha = _write_json(
        bundle_path,
        {
            "source": {
                "baseline_commit": baseline,
                "tool_contract_version": execution.EXECUTION_TOOL_CONTRACT_V3,
                "patch_source_manifest_sha256": patch_manifest_sha,
            },
            "platforms": {
                "linux-x64": {
                    "binary": {
                        "path": "linux/opencode-anchor",
                        "sha256": binary_sha,
                    }
                }
            },
        },
    )
    route_path = root / "artifacts/ccswitch/route-manifest.json"
    route_sha = _write_json(
        route_path,
        {"schema_version": "anchor.ccswitch-route-manifest.v1", "ready": True},
    )
    patch_path = root / "artifacts/private/final.patch"
    patch_sha = _write(patch_path, b"diff --git a/a b/a\n")
    receipt_path = root / "artifacts/private/official-eval-receipt.json"
    receipt_sha = _write_json(receipt_path, {"private": True})
    attestation_path = root / "artifacts/probe/attestation.json"
    value: dict[str, Any] = {
        "schema_version": execution.REPRESENTATIVE_PROBE_ATTESTATION_SCHEMA,
        "execution_lock_sha256": lock_sha,
        "opencode": {
            "baseline_commit": baseline,
            "patch_manifest": "patches/opencode/patch-manifest.json",
            "patch_manifest_sha256": patch_manifest_sha,
            "bundle_manifest": "artifacts/opencode/bundle-manifest.json",
            "bundle_manifest_sha256": bundle_sha,
            "linux_binary": "artifacts/opencode/linux/opencode-anchor",
            "linux_binary_sha256": binary_sha,
        },
        "ccswitch": {
            "route_manifest": "artifacts/ccswitch/route-manifest.json",
            "route_manifest_sha256": route_sha,
        },
        "representative": {
            "checkpoint_id": "c" * 64,
            "task_id_sha256": "d" * 64,
            "revision": 1,
            "instance_id_sha256": "e" * 64,
            "image_key_sha256": "f" * 64,
            "image_digest": "sha256:" + "1" * 64,
            "image_cache_binding_sha256": "2" * 64,
            "base_commit": "3" * 40,
            "final_patch": "artifacts/private/final.patch",
            "final_patch_sha256": patch_sha,
            "official_receipt": "artifacts/private/official-eval-receipt.json",
            "official_receipt_sha256": receipt_sha,
        },
        "content_free": True,
    }
    value["content_sha256"] = _sha(execution._canonical(value).encode())
    _write_json(attestation_path, value)
    lock = {
        "runtime": {
            "representative_probe_attestation": "artifacts/probe/attestation.json",
            "native_probe_root": "/var/lib/anchor/swebench-v3",
            "receipt_key_path": "/var/lib/anchor/keys/official-eval-hmac-v1",
            "wsl_distro": "Ubuntu-22.04",
        },
        "dataset": {"revision": "4" * 40},
        "opencode": {
            "patch_manifest": "patches/opencode/patch-manifest.json",
            "bundle_manifest": "artifacts/opencode/bundle-manifest.json",
        },
        "ccswitch": {"route_manifest": "artifacts/ccswitch/route-manifest.json"},
    }
    return lock, lock_sha, attestation_path


def test_representative_probe_requires_artifacts_image_ledger_and_hmac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, lock_sha, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        execution,
        "_wsl_verify_representative_private_bindings",
        lambda *args, **kwargs: (True, True),
    )
    patched, official, status = execution._representative_probe_attestation(
        tmp_path, lock, lock_sha256=lock_sha
    )
    assert patched is True and official is True
    assert status["artifact_bindings_valid"] is True
    assert status["image_binding_verified"] is True
    assert status["official_receipt_authenticated"] is True


@pytest.mark.parametrize("private_result", [(False, True), (True, False)])
def test_representative_probe_fails_closed_if_private_binding_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    private_result: tuple[bool, bool],
) -> None:
    lock, lock_sha, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        execution,
        "_wsl_verify_representative_private_bindings",
        lambda *args, **kwargs: private_result,
    )
    patched, official, _ = execution._representative_probe_attestation(
        tmp_path, lock, lock_sha256=lock_sha
    )
    assert patched is False or official is False


def test_representative_probe_rejects_recomputed_attestation_with_tampered_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, lock_sha, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        execution,
        "_wsl_verify_representative_private_bindings",
        lambda *args, **kwargs: (True, True),
    )
    (tmp_path / "artifacts/opencode/linux/opencode-anchor").write_bytes(b"tampered")
    patched, official, status = execution._representative_probe_attestation(
        tmp_path, lock, lock_sha256=lock_sha
    )
    assert patched is False and official is False
    assert status["artifact_bindings_valid"] is False


def test_representative_probe_runner_requires_both_explicit_confirmations() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/tooling/run_swebench_v3_representative_probe.py"),
            "--control-run-id",
            "test-probe-0001",
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout.strip() == (
        "representative_probe=blocked reason=explicit_confirmation_required"
    )
