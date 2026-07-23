"""Freeze the non-authorizing frozen-prefix Q-reader release overlay.

This entrypoint deliberately uses only the Python standard library until the
overlay implementation, this exact CLI, the config, and the profile freeze
manifest have been authenticated from single byte snapshots.  The verified
implementation bytes are then compiled directly; the normal import system
never gets a chance to execute an unverified checkout or cached module.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(os.path.abspath(Path(__file__).parent.parent.parent))
CONFIG_PATH = Path("configs/research/frozen_prefix_qreader_release_overlay_v1.json")
IMPLEMENTATION_PATH = Path("src/anchor_mvp/swebench/frozen_prefix_qreader_release.py")
CLI_PATH = Path("scripts/data/freeze_frozen_prefix_qreader_release.py")
PROFILE_RUNTIME_DIR = Path("artifacts/distillation-profiles/frozen-prefix-qreader-v2")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_PATH = re.compile(
    r"^(?!/)(?![A-Za-z]:)(?!.*\\)(?!.*(?:^|/)\.\.?/)(?!.*//)"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)
_REPARSE_POINT = 0x400
_MAX_METADATA_BYTES = 50_000_000


class ReleaseBootstrapError(RuntimeError):
    """A stable refusal code raised before any producer module executes."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ReleaseBootstrapError(code)


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _plain_snapshot(path: Path, code: str) -> tuple[bytes, str, int]:
    """Read one immutable lexical-path snapshot and reject link indirection."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            observed = os.lstat(current)
            if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
                _fail(code)
        with absolute.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size < 1 or before.st_size >= _MAX_METADATA_BYTES:
                _fail(code)
            data = handle.read()
            after = os.fstat(handle.fileno())
        terminal = os.lstat(absolute)
    except ReleaseBootstrapError:
        raise
    except OSError as exc:
        raise ReleaseBootstrapError(code) from exc
    identity = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_reparse(before)
        or identity != _identity(after)
        or identity != _identity(terminal)
        or len(data) != before.st_size
    ):
        _fail(code)
    return data, sha256(data).hexdigest(), len(data)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _parse_json(data: bytes, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        raise ReleaseBootstrapError(code) from exc
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _canonical_project_file(relative: object, expected: Path, code: str) -> Path:
    if (
        not isinstance(relative, str)
        or not _PROJECT_PATH.fullmatch(relative)
        or PurePosixPath(relative).is_absolute()
        or relative != expected.as_posix()
    ):
        _fail(code)
    return PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)


def _binding(
    value: object,
    expected_path: Path,
    code: str,
    *,
    expected_role: str | None = None,
) -> Mapping[str, Any]:
    binding = _mapping(value, code)
    expected_keys = {"path", "sha256", "bytes"}
    if expected_role is not None:
        expected_keys.add("role")
    if (
        set(binding) != expected_keys
        or (expected_role is not None and binding.get("role") != expected_role)
        or binding.get("path") != expected_path.as_posix()
        or not isinstance(binding.get("sha256"), str)
        or not _SHA256.fullmatch(str(binding["sha256"]))
        or isinstance(binding.get("bytes"), bool)
        or not isinstance(binding.get("bytes"), int)
        or int(binding["bytes"]) < 1
    ):
        _fail(code)
    return binding


def _same_lexical_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _config_path(value: Path) -> Path:
    if value.is_absolute():
        candidate = Path(os.path.abspath(value))
    else:
        if value.as_posix() != CONFIG_PATH.as_posix():
            _fail("release_bootstrap_config_path_invalid")
        candidate = PROJECT_ROOT / value
    expected = PROJECT_ROOT / CONFIG_PATH
    if not _same_lexical_path(candidate, expected):
        _fail("release_bootstrap_config_path_invalid")
    return expected


def _profile_path(value: Path) -> Path:
    candidate = Path(os.path.abspath(value))
    expected = PROJECT_ROOT / PROFILE_RUNTIME_DIR
    if not _same_lexical_path(candidate, expected):
        _fail("release_bootstrap_profile_path_invalid")
    return expected


def _load_authenticated_release_module(args: argparse.Namespace) -> ModuleType:
    """Authenticate the loader chain before compiling exact producer bytes."""

    executing_cli = Path(os.path.abspath(__file__))
    canonical_cli = PROJECT_ROOT / CLI_PATH
    if not _same_lexical_path(executing_cli, canonical_cli):
        _fail("release_bootstrap_executing_cli_path_invalid")
    cli_bytes, cli_sha256, cli_size = _plain_snapshot(
        executing_cli, "release_bootstrap_executing_cli_invalid"
    )

    config_file = _config_path(args.config)
    config_bytes, config_sha256, _config_size = _plain_snapshot(
        config_file, "release_bootstrap_config_invalid"
    )
    if (
        not isinstance(args.config_sha256, str)
        or not _SHA256.fullmatch(args.config_sha256)
        or config_sha256 != args.config_sha256
    ):
        _fail("release_bootstrap_config_hash_mismatch")
    config = _parse_json(config_bytes, "release_bootstrap_config_invalid")
    if (
        config.get("schema_version")
        != "anchor.frozen-prefix-qreader-release-overlay-config.v1"
    ):
        _fail("release_bootstrap_config_invalid")
    producer = _mapping(
        config.get("producer_bindings"), "release_bootstrap_config_invalid"
    )
    config_implementation = _binding(
        producer.get("implementation"),
        IMPLEMENTATION_PATH,
        "release_bootstrap_config_invalid",
    )
    config_cli = _binding(
        producer.get("freeze_script"),
        CLI_PATH,
        "release_bootstrap_config_invalid",
    )

    profile_dir = _profile_path(args.profile_freeze_dir)
    profile_bytes, profile_sha256, _profile_size = _plain_snapshot(
        profile_dir / "manifest.json",
        "release_bootstrap_profile_invalid",
    )
    if (
        not isinstance(args.profile_freeze_manifest_sha256, str)
        or not _SHA256.fullmatch(args.profile_freeze_manifest_sha256)
        or profile_sha256 != args.profile_freeze_manifest_sha256
    ):
        _fail("release_bootstrap_profile_hash_mismatch")
    sidecar_bytes, _sidecar_sha256, _sidecar_size = _plain_snapshot(
        profile_dir / "manifest.json.sha256",
        "release_bootstrap_profile_invalid",
    )
    if sidecar_bytes != f"{profile_sha256}  manifest.json\n".encode("ascii"):
        _fail("release_bootstrap_profile_invalid")
    profile = _parse_json(profile_bytes, "release_bootstrap_profile_invalid")
    if (
        profile.get("schema_version")
        != "anchor.frozen-prefix-qreader-profile-freeze-manifest.v2"
        or profile.get("profile_id") != "frozen-prefix-qreader-v2"
    ):
        _fail("release_bootstrap_profile_invalid")
    dependencies = profile.get("dependencies")
    if not isinstance(dependencies, list):
        _fail("release_bootstrap_profile_invalid")
    dependency_by_role: dict[str, Mapping[str, Any]] = {}
    for item in dependencies:
        dependency = _mapping(item, "release_bootstrap_profile_invalid")
        role = dependency.get("role")
        if not isinstance(role, str) or role in dependency_by_role:
            _fail("release_bootstrap_profile_invalid")
        dependency_by_role[role] = dependency
    profile_implementation = _binding(
        dependency_by_role.get("release_overlay_implementation"),
        IMPLEMENTATION_PATH,
        "release_bootstrap_profile_invalid",
        expected_role="release_overlay_implementation",
    )
    profile_cli = _binding(
        dependency_by_role.get("release_overlay_cli"),
        CLI_PATH,
        "release_bootstrap_profile_invalid",
        expected_role="release_overlay_cli",
    )

    if (
        cli_sha256 != config_cli["sha256"]
        or cli_size != config_cli["bytes"]
        or cli_sha256 != profile_cli["sha256"]
        or cli_size != profile_cli["bytes"]
    ):
        _fail("release_bootstrap_executing_cli_binding_invalid")

    implementation_file = _canonical_project_file(
        config_implementation.get("path"),
        IMPLEMENTATION_PATH,
        "release_bootstrap_implementation_binding_invalid",
    )
    implementation_bytes, implementation_sha256, implementation_size = _plain_snapshot(
        implementation_file,
        "release_bootstrap_implementation_binding_invalid",
    )
    if (
        implementation_sha256 != config_implementation["sha256"]
        or implementation_size != config_implementation["bytes"]
        or implementation_sha256 != profile_implementation["sha256"]
        or implementation_size != profile_implementation["bytes"]
        or profile_implementation["path"] != config_implementation["path"]
    ):
        _fail("release_bootstrap_implementation_binding_invalid")

    module_name = (
        f"_anchor_authenticated_frozen_prefix_qreader_release_{implementation_sha256}"
    )
    module = ModuleType(module_name)
    module.__file__ = str(implementation_file)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(
            implementation_bytes,
            str(implementation_file),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ReleaseBootstrapError(
            "release_bootstrap_implementation_load_failed"
        ) from exc
    error_type = getattr(module, "FrozenPrefixQReaderReleaseError", None)
    if (
        not isinstance(getattr(module, "CONFIG_PATH", None), Path)
        or module.CONFIG_PATH != CONFIG_PATH
        or not isinstance(error_type, type)
        or not issubclass(error_type, Exception)
        or not callable(getattr(module, "freeze_frozen_prefix_qreader_release", None))
    ):
        sys.modules.pop(module_name, None)
        _fail("release_bootstrap_implementation_exports_invalid")
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate five manifest-only conjuncts and create a blocked, "
            "non-authorizing Q-reader v2 release overlay. No partition, Gold, "
            "held-out, provider, model, or GPU access is performed."
        )
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--generic-release-dir", type=Path, required=True)
    parser.add_argument("--generic-release-manifest-sha256", required=True)
    parser.add_argument("--profile-freeze-dir", type=Path, required=True)
    parser.add_argument("--profile-freeze-manifest-sha256", required=True)
    parser.add_argument("--training-view-dir", type=Path, required=True)
    parser.add_argument("--training-view-manifest-sha256", required=True)
    parser.add_argument("--bundle-profile-dir", type=Path, required=True)
    parser.add_argument("--bundle-profile-manifest-sha256", required=True)
    parser.add_argument("--consumer-reference-dir", type=Path, required=True)
    parser.add_argument("--consumer-reference-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _refusal(code: str) -> int:
    print(
        json.dumps(
            {
                "ok": False,
                "error": code,
                "training_authorized": False,
                "formal_training_authorized": False,
                "release_authorized": False,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        release = _load_authenticated_release_module(args)
    except ReleaseBootstrapError as exc:
        return _refusal(exc.code)
    release_error = release.FrozenPrefixQReaderReleaseError
    try:
        result = release.freeze_frozen_prefix_qreader_release(
            project_root=PROJECT_ROOT,
            config_path=args.config,
            expected_config_sha256=args.config_sha256,
            generic_release_dir=args.generic_release_dir,
            expected_generic_release_sha256=(args.generic_release_manifest_sha256),
            profile_freeze_dir=args.profile_freeze_dir,
            expected_profile_freeze_sha256=args.profile_freeze_manifest_sha256,
            training_view_dir=args.training_view_dir,
            expected_training_view_sha256=args.training_view_manifest_sha256,
            bundle_profile_dir=args.bundle_profile_dir,
            expected_bundle_profile_sha256=(args.bundle_profile_manifest_sha256),
            consumer_reference_dir=args.consumer_reference_dir,
            expected_consumer_reference_sha256=(
                args.consumer_reference_manifest_sha256
            ),
            output_dir=args.output_dir,
        )
    except release_error as exc:
        return _refusal(exc.code)
    except (OSError, ValueError, json.JSONDecodeError):
        return _refusal("invalid_local_artifact")
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
