"""Validate two platform artifacts and write one deterministic OpenCode bundle manifest.

This utility is intentionally independent of the distillation executors. It consumes only
the platform manifests emitted by the Windows and WSL build scripts, verifies their binary
hashes, and refuses to combine artifacts that do not share an identical pinned source
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PLATFORM_SCHEMA = "anchor.patched-opencode.platform.v1"
BUNDLE_SCHEMA = "anchor.patched-opencode.bundle.v1"
REQUIRED_TOOL_CONTRACT = "anchor.execution-tool-contract.v3"
TARGET_MANIFESTS = {
    "windows-x64": "windows-x64.manifest.json",
    "linux-x64": "linux-x64.manifest.json",
}
TARGET_BINARY_PATHS = {
    "windows-x64": "windows-x64/opencode-anchor.exe",
    "linux-x64": "linux-x64/opencode-anchor",
}
SOURCE_KEYS = (
    "repository",
    "baseline_commit",
    "opencode_version",
    "patch_sha256",
    "patch_source_manifest_sha256",
    "bun_version",
    "tool_contract_version",
    "tool_contract",
    "lockfile_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"manifest is missing or invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the artifact root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the artifact root") from error
    if not resolved.is_file():
        raise ValueError(f"artifact binary is missing: {relative.as_posix()}")
    return resolved


def _validate_platform(
    artifact_root: Path, target: str, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _load_object(manifest_path)
    if value.get("schema_version") != PLATFORM_SCHEMA:
        raise ValueError(f"{target}: unsupported platform manifest schema")
    if value.get("target") != target:
        raise ValueError(f"{target}: target does not match manifest path")
    platform = _mapping(value.get("platform"), f"{target}.platform")
    if platform.get("arch") != "x64":
        raise ValueError(f"{target}: artifact architecture is not x64")
    expected_os = "windows" if target == "windows-x64" else "linux"
    if platform.get("os") != expected_os:
        raise ValueError(f"{target}: platform operating system mismatch")
    source = dict(_mapping(value.get("source"), f"{target}.source"))
    missing = [key for key in SOURCE_KEYS if key not in source]
    if missing:
        raise ValueError(f"{target}: source contract is missing {','.join(missing)}")
    if source.get("tool_contract_version") != REQUIRED_TOOL_CONTRACT:
        raise ValueError(
            f"{target}: only {REQUIRED_TOOL_CONTRACT} artifacts may enter the formal bundle"
        )
    contract = _mapping(source.get("tool_contract"), f"{target}.source.tool_contract")
    model_policy = _mapping(
        contract.get("model_bash_policy"),
        f"{target}.source.tool_contract.model_bash_policy",
    )
    hidden_eval = _mapping(
        contract.get("hidden_official_eval"),
        f"{target}.source.tool_contract.hidden_official_eval",
    )
    if (
        contract.get("version") != REQUIRED_TOOL_CONTRACT
        or model_policy.get("workdir") != "/testbed"
        or model_policy.get("network")
        != "none-with-supervisor-unix-socket-loopback-bridge"
        or hidden_eval.get("network") != "none"
        or hidden_eval.get("fresh_container") is not True
    ):
        raise ValueError(f"{target}: formal v3 isolation contract is incomplete")
    binary = _mapping(value.get("binary"), f"{target}.binary")
    binary_path = _relative_file(artifact_root, binary.get("path"), f"{target}.binary.path")
    observed_relative = binary_path.relative_to(artifact_root.resolve()).as_posix()
    expected_relative = TARGET_BINARY_PATHS[target]
    if observed_relative != expected_relative:
        raise ValueError(
            f"{target}: binary layout must be {expected_relative}, got {observed_relative}"
        )
    observed_sha = sha256_file(binary_path)
    expected_sha = binary.get("sha256")
    if not isinstance(expected_sha, str) or observed_sha != expected_sha.lower():
        raise ValueError(f"{target}: binary SHA-256 mismatch")
    return source, {
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "binary": dict(binary),
    }


def build_bundle(artifact_root: str | Path) -> dict[str, Any]:
    """Return a deterministic bundle after checking both platform artifacts."""

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError(f"artifact root is missing: {root}")
    common_source: dict[str, Any] | None = None
    platforms: dict[str, Any] = {}
    for target, filename in TARGET_MANIFESTS.items():
        source, artifact = _validate_platform(root, target, root / filename)
        if common_source is None:
            common_source = source
        elif source != common_source:
            mismatched = [key for key in SOURCE_KEYS if source.get(key) != common_source.get(key)]
            raise ValueError(
                f"{target}: source contract differs from the other platform: {','.join(mismatched)}"
            )
        platforms[target] = artifact
    return {
        "schema_version": BUNDLE_SCHEMA,
        "source": common_source,
        "platforms": platforms,
    }


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_bundle(artifact_root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(artifact_root).resolve()
    bundle = build_bundle(root)
    destination = Path(output).resolve() if output else root / "bundle-manifest.json"
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError("bundle output must stay inside the artifact root") from error
    destination.write_text(canonical_json(bundle), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Windows and Linux OpenCode artifacts and write their bundle manifest."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate both platform manifests and binaries without writing bundle-manifest.json.",
    )
    args = parser.parse_args()
    try:
        if args.check:
            build_bundle(args.artifact_root)
            print("bundle manifests and binary hashes are valid")
        else:
            print(write_bundle(args.artifact_root, args.output))
    except ValueError as error:
        print(f"OpenCode bundle refused: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
