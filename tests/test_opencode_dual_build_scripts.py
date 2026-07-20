from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from anchor_mvp.tooling.tool_contract import (
    EXECUTION_TOOL_CONTRACT_V3_VERSION,
    v3_contract_descriptor,
)


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SCRIPT = ROOT / "scripts" / "tooling" / "build_patched_opencode.ps1"
WSL_SCRIPT = ROOT / "scripts" / "tooling" / "build_patched_opencode_wsl.sh"
BUNDLE_SCRIPT = ROOT / "scripts" / "tooling" / "assemble_opencode_bundle.py"
BUILD_DOC = ROOT / "docs" / "opencode_dual_build.md"

PLATFORM_BINARY_PATHS = {
    "windows-x64": "windows-x64/opencode-anchor.exe",
    "linux-x64": "linux-x64/opencode-anchor",
}


def _bundle_module():
    spec = importlib.util.spec_from_file_location("anchor_opencode_bundle", BUNDLE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> dict[str, object]:
    return {
        "repository": "https://github.com/anomalyco/opencode.git",
        "baseline_commit": "a" * 40,
        "opencode_version": "1.17.18",
        "patch_sha256": "b" * 64,
        "patch_source_manifest_sha256": "c" * 64,
        "bun_version": "1.3.14",
        "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3_VERSION,
        "tool_contract": v3_contract_descriptor(),
        "lockfile_sha256": "d" * 64,
    }


def _write_platform_manifest(
    root: Path, *, target: str, binary_relative: str, source: dict[str, object]
) -> None:
    binary = root / binary_relative
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(target.encode("ascii"))
    platform = {
        "schema_version": "anchor.patched-opencode.platform.v1",
        "target": target,
        "platform": {
            "os": "windows" if target == "windows-x64" else "linux",
            "arch": "x64",
            "libc": None if target == "windows-x64" else "glibc",
        },
        "source": source,
        "bun": {"version": "1.3.14", "sha256": "e" * 64},
        "node_gyp_version": "v13.0.1" if target == "windows-x64" else None,
        "install": {"executed": True, "linker": "test", "cache_scope": target},
        "checks": {"tests_executed": True, "required_tests": {}, "test_exclusions": []},
        "binary": {"path": binary_relative, "sha256": _sha256(binary)},
        "global_install_modified": False,
    }
    (root / f"{target}.manifest.json").write_text(
        json.dumps(platform, sort_keys=True), encoding="utf-8"
    )


def test_dual_build_scripts_use_clean_platform_specific_worktrees():
    windows = WINDOWS_SCRIPT.read_text(encoding="utf-8")
    wsl = WSL_SCRIPT.read_text(encoding="utf-8")

    assert "worktrees\\$Target" in windows
    assert "worktrees/linux-x64" in wsl
    assert "BunSha256" in windows
    assert "--bun-sha256" in wsl
    assert "BUN_INSTALL_CACHE_DIR" in windows
    assert "BUN_INSTALL_CACHE_DIR" in wsl
    assert '"$Target.manifest.json"' in windows
    assert "linux-x64.manifest.json" in wsl
    assert 'path = "$Target/opencode-anchor.exe"' in windows
    assert '"path": "linux-x64/opencode-anchor"' in wsl
    assert "workspace_link_audit" in windows
    assert "workspace_link_audit" in wsl
    assert "Workspace package resolves outside the current checkout" in windows
    assert "Workspace package resolves outside the current checkout" in wsl
    assert 'required_root = "checkout/packages"' in windows
    assert '"required_root": "checkout/packages"' in wsl
    assert windows.count('"--timeout", "60000"') == 4
    assert "test --timeout 15000" in wsl
    assert "--fresh-install-failure failed_external_tls" in wsl
    assert "--third-party-deps-reused" in wsl
    assert "reused_deps_lock_sha256" in wsl
    assert "Reused dependency lock hash does not match" in wsl
    assert "--models-dev-json" in wsl
    assert "models.dev snapshot SHA-256 mismatch" in wsl
    assert "models_dev_snapshot_sha256" in wsl
    for script in (windows, wsl):
        lowered = script.casefold()
        assert "git reset --" not in lowered
        assert "git clean " not in lowered
        assert "status --porcelain" in lowered
    assert '"apply", "--check"' in windows
    assert "apply --check" in wsl
    for script in (windows, wsl):
        assert EXECUTION_TOOL_CONTRACT_V3_VERSION in script
        assert "v2 artifacts" in script


def test_dual_build_document_records_portable_linux_bun_with_full_binary_hash():
    document = BUILD_DOC.read_text(encoding="utf-8")

    assert EXECUTION_TOOL_CONTRACT_V3_VERSION in document
    assert "not-ready" in document
    assert '$HOME/.cache/anchor-moe-lora/toolchains/bun-1.3.14/bun' in document
    assert "/home/is/" not in document
    assert "9fd36f87e4b90b07632b987a2e4ec81ca15a62c81bf983190cea6d715be2ad74" in document
    assert "951ee2aee855f08595aeec6225226a298d3fea83a3dcd6465c09cbccdf7e848f" in document
    assert "35969274" in document
    assert "sha256sum" in document
    assert "assemble_opencode_bundle.py" in document


def test_bundle_manifest_requires_two_matching_platform_contracts(tmp_path: Path):
    module = _bundle_module()
    source = _source()
    _write_platform_manifest(
        tmp_path,
        target="windows-x64",
        binary_relative=PLATFORM_BINARY_PATHS["windows-x64"],
        source=source,
    )
    _write_platform_manifest(
        tmp_path,
        target="linux-x64",
        binary_relative=PLATFORM_BINARY_PATHS["linux-x64"],
        source=source,
    )

    destination = module.write_bundle(tmp_path)
    bundle = json.loads(destination.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "anchor.patched-opencode.bundle.v1"
    assert set(bundle["platforms"]) == {"windows-x64", "linux-x64"}
    assert bundle["source"] == source

    linux_manifest = tmp_path / "linux-x64.manifest.json"
    changed = json.loads(linux_manifest.read_text(encoding="utf-8"))
    changed["source"]["patch_sha256"] = "f" * 64
    linux_manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="source contract differs"):
        module.build_bundle(tmp_path)


def test_bundle_manifest_rejects_binary_hash_drift(tmp_path: Path):
    module = _bundle_module()
    source = _source()
    _write_platform_manifest(
        tmp_path,
        target="windows-x64",
        binary_relative=PLATFORM_BINARY_PATHS["windows-x64"],
        source=source,
    )
    _write_platform_manifest(
        tmp_path,
        target="linux-x64",
        binary_relative=PLATFORM_BINARY_PATHS["linux-x64"],
        source=source,
    )
    (tmp_path / "linux-x64" / "opencode-anchor").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="binary SHA-256 mismatch"):
        module.build_bundle(tmp_path)


def test_bundle_manifest_rejects_legacy_v2_contract(tmp_path: Path):
    module = _bundle_module()
    source = _source()
    source["tool_contract_version"] = "anchor.execution-tool-contract.v2"
    source["tool_contract"] = {
        "version": "anchor.execution-tool-contract.v2",
        "tools": [],
    }
    _write_platform_manifest(
        tmp_path,
        target="windows-x64",
        binary_relative=PLATFORM_BINARY_PATHS["windows-x64"],
        source=source,
    )
    _write_platform_manifest(
        tmp_path,
        target="linux-x64",
        binary_relative=PLATFORM_BINARY_PATHS["linux-x64"],
        source=source,
    )

    with pytest.raises(ValueError, match="only anchor.execution-tool-contract.v3"):
        module.build_bundle(tmp_path)


def test_bundle_manifest_rejects_asymmetric_or_root_binary_layout(tmp_path: Path):
    module = _bundle_module()
    source = _source()
    _write_platform_manifest(
        tmp_path,
        target="windows-x64",
        binary_relative="opencode-anchor.exe",
        source=source,
    )
    _write_platform_manifest(
        tmp_path,
        target="linux-x64",
        binary_relative=PLATFORM_BINARY_PATHS["linux-x64"],
        source=source,
    )

    with pytest.raises(ValueError, match="binary layout must be windows-x64/opencode-anchor.exe"):
        module.build_bundle(tmp_path)
