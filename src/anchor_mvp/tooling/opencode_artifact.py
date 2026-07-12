"""Patched OpenCode binary provenance and launch-time attestation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .tool_contract import EXECUTION_TOOL_CONTRACT_VERSION, contract_descriptor


BUILD_MANIFEST_SCHEMA = "anchor.patched-opencode.v1"
PATCH_MANIFEST_SCHEMA = "anchor.opencode-patch-source.v1"
PLATFORM_MANIFEST_SCHEMA = "anchor.patched-opencode.platform.v1"
BUNDLE_MANIFEST_SCHEMA = "anchor.patched-opencode.bundle.v1"
AUDITED_REPOSITORY = "https://github.com/anomalyco/opencode.git"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_TARGETS = {
    "windows-x64": {
        "manifest": "windows-x64.manifest.json",
        "binary": "windows-x64/opencode-anchor.exe",
        "platform": {"os": "windows", "arch": "x64", "libc": None},
    },
    "linux-x64": {
        "manifest": "linux-x64.manifest.json",
        "binary": "linux-x64/opencode-anchor",
        "platform": {"os": "linux", "arch": "x64", "libc": "glibc"},
    },
}
_BUNDLE_SOURCE_KEYS = {
    "repository",
    "baseline_commit",
    "opencode_version",
    "patch_sha256",
    "patch_source_manifest_sha256",
    "bun_version",
    "tool_contract_version",
    "tool_contract",
    "lockfile_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value: object, *, label: str) -> str:
    normalized = str(value).casefold()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _bundle_file(root: Path, value: object, *, expected: str, label: str) -> Path:
    if not isinstance(value, str) or value.replace("\\", "/") != expected:
        raise ValueError(f"{label} must be {expected}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the artifact root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the artifact root") from exc
    if not path.is_file() or (root / relative).is_symlink():
        raise ValueError(f"{label} is missing or is a symlink")
    return path


@dataclass(frozen=True)
class BinaryAttestation:
    executable: Path
    binary_sha256: str
    build_manifest: Path
    build_manifest_sha256: str
    patch_sha256: str
    baseline_commit: str
    opencode_version: str
    behavioral_probe: bool = False
    linux_executable: Path | None = None
    linux_binary_sha256: str | None = None
    bundle_manifest: Path | None = None
    bundle_manifest_sha256: str | None = None
    bound_files: tuple[tuple[Path, str], ...] = ()

    def with_behavioral_probe(self) -> "BinaryAttestation":
        return replace(self, behavioral_probe=True)


def verify_binary_attestation(
    executable: str | Path,
    *,
    patch_manifest: str | Path,
    linux_executable: str | Path | None = None,
) -> BinaryAttestation:
    binary = Path(executable).expanduser().resolve()
    if not binary.is_file():
        raise ValueError("patched OpenCode executable is missing")
    source_path = Path(patch_manifest).expanduser().resolve()
    source = _load_object(source_path, label="patch source manifest")
    if source.get("schema_version") != PATCH_MANIFEST_SCHEMA:
        raise ValueError("unsupported patch source manifest schema")
    if source.get("repository") != AUDITED_REPOSITORY:
        raise ValueError("patch source repository is not the audited upstream")
    if source.get("tool_contract_version") != EXECUTION_TOOL_CONTRACT_VERSION:
        raise ValueError("patch source tool contract version differs from the converter")
    if source.get("tool_contract") != contract_descriptor():
        raise ValueError("patch source tool contract differs from the converter")
    required_tests = source.get("required_tests")
    if not isinstance(required_tests, dict):
        raise ValueError("patch source manifest has no required behavioral tests")
    for suite in ("core", "opencode"):
        tests = required_tests.get(suite)
        if not isinstance(tests, list) or not tests or not all(
            isinstance(item, str) and item.endswith(".test.ts") for item in tests
        ):
            raise ValueError(f"patch source manifest has invalid {suite} tests")
    if linux_executable is None:
        raise ValueError("mounted Linux OpenCode executable is required for bundle attestation")
    mounted_linux = Path(linux_executable).expanduser().resolve()
    build_path = binary.parent / "manifest.json"
    build = _load_object(build_path, label="patched OpenCode build manifest")
    if build.get("schema_version") != BUILD_MANIFEST_SCHEMA:
        raise ValueError("unsupported patched OpenCode build manifest schema")
    expected: dict[str, object] = {
        "repository": source.get("repository"),
        "baseline_commit": source.get("baseline_commit"),
        "opencode_version": source.get("upstream_version"),
        "patch_sha256": source.get("patch_sha256"),
        "bun_version": source.get("bun_version"),
        "tool_contract_version": EXECUTION_TOOL_CONTRACT_VERSION,
        "tool_contract": contract_descriptor(),
        "binary": binary.name,
        "tests_executed": True,
        "required_tests": required_tests,
        "typecheck_executed": True,
        "global_install_modified": False,
    }
    mismatches = [name for name, value in expected.items() if build.get(name) != value]
    if mismatches:
        raise ValueError("patched OpenCode build manifest mismatch: " + ",".join(mismatches))
    patch_path = source_path.parent / str(source.get("patch", ""))
    if not patch_path.is_file() or sha256_file(patch_path) != source.get("patch_sha256"):
        raise ValueError("audited OpenCode patch digest mismatch")
    observed_binary = sha256_file(binary)
    if observed_binary != build.get("binary_sha256"):
        raise ValueError("patched OpenCode binary digest mismatch")
    source_digest = sha256_file(source_path)
    if build.get("patch_source_manifest_sha256") != source_digest:
        raise ValueError("build does not bind the audited patch source manifest")

    artifact_root = binary.parent.resolve()
    bundle_path = artifact_root / "bundle-manifest.json"
    bundle = _load_object(bundle_path, label="patched OpenCode bundle manifest")
    if bundle.get("schema_version") != BUNDLE_MANIFEST_SCHEMA:
        raise ValueError("unsupported patched OpenCode bundle manifest schema")
    bundle_source = _mapping(bundle.get("source"), label="bundle source")
    if set(bundle_source) != _BUNDLE_SOURCE_KEYS:
        raise ValueError("bundle source contract has unexpected or missing fields")
    expected_bundle_source = {
        "repository": source.get("repository"),
        "baseline_commit": source.get("baseline_commit"),
        "opencode_version": source.get("upstream_version"),
        "patch_sha256": source.get("patch_sha256"),
        "patch_source_manifest_sha256": source_digest,
        "bun_version": source.get("bun_version"),
        "tool_contract_version": EXECUTION_TOOL_CONTRACT_VERSION,
        "tool_contract": contract_descriptor(),
    }
    source_mismatches = [
        name
        for name, expected_value in expected_bundle_source.items()
        if bundle_source.get(name) != expected_value
    ]
    _sha256(bundle_source.get("lockfile_sha256"), label="bundle lockfile_sha256")
    if source_mismatches:
        raise ValueError(
            "bundle source differs from the audited patch source: "
            + ",".join(source_mismatches)
        )

    bundle_platforms = _mapping(bundle.get("platforms"), label="bundle platforms")
    if set(bundle_platforms) != set(_BUNDLE_TARGETS):
        raise ValueError("bundle must contain exactly windows-x64 and linux-x64")
    bound_files: list[tuple[Path, str]] = [
        (source_path, source_digest),
        (patch_path.resolve(), str(source["patch_sha256"])),
        (bundle_path.resolve(), sha256_file(bundle_path)),
    ]
    member_binary_paths: dict[str, Path] = {}
    member_binary_hashes: dict[str, str] = {}
    for target, layout in _BUNDLE_TARGETS.items():
        entry = _mapping(bundle_platforms.get(target), label=f"bundle {target}")
        if set(entry) != {"manifest", "manifest_sha256", "binary"}:
            raise ValueError(f"bundle {target} has unexpected or missing fields")
        member_manifest_path = _bundle_file(
            artifact_root,
            entry.get("manifest"),
            expected=str(layout["manifest"]),
            label=f"bundle {target} manifest",
        )
        expected_manifest_sha = _sha256(
            entry.get("manifest_sha256"), label=f"bundle {target} manifest_sha256"
        )
        observed_manifest_sha = sha256_file(member_manifest_path)
        if observed_manifest_sha != expected_manifest_sha:
            raise ValueError(f"bundle {target} member manifest SHA-256 mismatch")
        member = _load_object(member_manifest_path, label=f"{target} platform manifest")
        if member.get("schema_version") != PLATFORM_MANIFEST_SCHEMA:
            raise ValueError(f"{target}: unsupported platform manifest schema")
        if member.get("target") != target:
            raise ValueError(f"{target}: target does not match member manifest")
        if member.get("platform") != layout["platform"]:
            raise ValueError(f"{target}: platform identity mismatch")
        if member.get("source") != bundle_source:
            raise ValueError(f"{target}: source contract differs from the bundle")
        checks = _mapping(member.get("checks"), label=f"{target} checks")
        required_checks = {
            "tests_executed": True,
            "required_tests": required_tests,
            "typecheck_executed": True,
            "build_smoke_executed": True,
        }
        failed_checks = [
            name for name, expected_value in required_checks.items() if checks.get(name) != expected_value
        ]
        if failed_checks:
            raise ValueError(f"{target}: required build checks mismatch: {','.join(failed_checks)}")
        if member.get("global_install_modified") is not False:
            raise ValueError(f"{target}: build modified the global installation")
        member_binary = _mapping(member.get("binary"), label=f"{target} binary")
        bundle_binary = _mapping(entry.get("binary"), label=f"bundle {target} binary")
        if member_binary != bundle_binary or set(member_binary) != {"path", "sha256"}:
            raise ValueError(f"{target}: bundle and member binary contracts differ")
        member_binary_path = _bundle_file(
            artifact_root,
            member_binary.get("path"),
            expected=str(layout["binary"]),
            label=f"{target} binary path",
        )
        member_binary_sha = _sha256(
            member_binary.get("sha256"), label=f"{target} binary sha256"
        )
        if sha256_file(member_binary_path) != member_binary_sha:
            raise ValueError(f"{target}: binary SHA-256 mismatch")
        member_binary_paths[target] = member_binary_path
        member_binary_hashes[target] = member_binary_sha
        bound_files.extend(
            (
                (member_manifest_path, observed_manifest_sha),
                (member_binary_path, member_binary_sha),
            )
        )

    if observed_binary != member_binary_hashes["windows-x64"]:
        raise ValueError("host Windows executable differs from the bundle member")
    expected_linux = member_binary_paths["linux-x64"]
    if mounted_linux != expected_linux:
        raise ValueError("mounted Linux executable is not the attested bundle member")
    return BinaryAttestation(
        executable=binary,
        binary_sha256=observed_binary,
        build_manifest=build_path,
        build_manifest_sha256=sha256_file(build_path),
        patch_sha256=str(source["patch_sha256"]),
        baseline_commit=str(source["baseline_commit"]),
        opencode_version=str(source["upstream_version"]),
        linux_executable=expected_linux,
        linux_binary_sha256=member_binary_hashes["linux-x64"],
        bundle_manifest=bundle_path.resolve(),
        bundle_manifest_sha256=sha256_file(bundle_path),
        bound_files=tuple(bound_files),
    )


def verify_binary_identity(attestation: BinaryAttestation) -> None:
    """Fail if the attested executable or its build manifest changed."""
    if not attestation.executable.is_file():
        raise ValueError("attested OpenCode executable disappeared")
    if sha256_file(attestation.executable) != attestation.binary_sha256:
        raise ValueError("attested OpenCode executable changed before launch")
    if sha256_file(attestation.build_manifest) != attestation.build_manifest_sha256:
        raise ValueError("patched OpenCode build manifest changed before launch")
    for path, expected_sha256 in attestation.bound_files:
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"attested bundle file changed before launch: {path.name}")


def verify_launch_identity(attestation: BinaryAttestation) -> None:
    """Fail if identity changed or the offline behavior was not probed."""

    if not attestation.behavioral_probe:
        raise ValueError("patched OpenCode behavioral attestation is missing")
    verify_binary_identity(attestation)
