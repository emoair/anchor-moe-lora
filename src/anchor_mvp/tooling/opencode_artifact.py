"""Patched OpenCode binary provenance and launch-time attestation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .tool_contract import EXECUTION_TOOL_CONTRACT_VERSION, contract_descriptor


BUILD_MANIFEST_SCHEMA = "anchor.patched-opencode.v1"
PATCH_MANIFEST_SCHEMA = "anchor.opencode-patch-source.v1"
AUDITED_REPOSITORY = "https://github.com/anomalyco/opencode.git"


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

    def with_behavioral_probe(self) -> "BinaryAttestation":
        return BinaryAttestation(
            executable=self.executable,
            binary_sha256=self.binary_sha256,
            build_manifest=self.build_manifest,
            build_manifest_sha256=self.build_manifest_sha256,
            patch_sha256=self.patch_sha256,
            baseline_commit=self.baseline_commit,
            opencode_version=self.opencode_version,
            behavioral_probe=True,
        )


def verify_binary_attestation(
    executable: str | Path,
    *,
    patch_manifest: str | Path,
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
    return BinaryAttestation(
        executable=binary,
        binary_sha256=observed_binary,
        build_manifest=build_path,
        build_manifest_sha256=sha256_file(build_path),
        patch_sha256=str(source["patch_sha256"]),
        baseline_commit=str(source["baseline_commit"]),
        opencode_version=str(source["upstream_version"]),
    )


def verify_binary_identity(attestation: BinaryAttestation) -> None:
    """Fail if the attested executable or its build manifest changed."""
    if not attestation.executable.is_file():
        raise ValueError("attested OpenCode executable disappeared")
    if sha256_file(attestation.executable) != attestation.binary_sha256:
        raise ValueError("attested OpenCode executable changed before launch")
    if sha256_file(attestation.build_manifest) != attestation.build_manifest_sha256:
        raise ValueError("patched OpenCode build manifest changed before launch")


def verify_launch_identity(attestation: BinaryAttestation) -> None:
    """Fail if identity changed or the offline behavior was not probed."""

    if not attestation.behavioral_probe:
        raise ValueError("patched OpenCode behavioral attestation is missing")
    verify_binary_identity(attestation)
