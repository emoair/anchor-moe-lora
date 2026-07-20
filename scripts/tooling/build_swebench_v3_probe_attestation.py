"""Build the content-free attestation for an already completed v3 live probe.

This command never starts a provider, pulls an image, or runs a task.  It only
binds artifacts produced by one explicit representative run.  The regular
execution preflight subsequently re-authenticates the private receipt and
image-cache ledger inside the root-owned WSL supervisor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.tooling.swebench_execution_v3 import (  # noqa: E402
    REPRESENTATIVE_PROBE_ATTESTATION_SCHEMA,
    load_execution_lock,
    sha256_file,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(label)
    return value


def _project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label)
    return _mapping(json.loads(path.read_text(encoding="utf-8")), label)


def build(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = args.lock.resolve()
    lock_path.relative_to(ROOT)
    lock_sha256 = sha256_file(lock_path)
    lock = load_execution_lock(ROOT, lock_path, expected_sha256=lock_sha256)
    opencode = _mapping(lock["opencode"], "lock opencode invalid")
    ccswitch = _mapping(lock["ccswitch"], "lock ccswitch invalid")

    patch_manifest = _project_path(str(opencode["patch_manifest"]))
    bundle_manifest = _project_path(str(opencode["bundle_manifest"]))
    route_manifest = _project_path(str(ccswitch["route_manifest"]))
    patch_value = _read_json(patch_manifest, "patch manifest missing")
    bundle_value = _read_json(bundle_manifest, "bundle manifest missing")
    _read_json(route_manifest, "route manifest missing")
    platforms = _mapping(bundle_value.get("platforms"), "bundle platforms invalid")
    linux = _mapping(platforms.get("linux-x64"), "linux bundle missing")
    binary = _mapping(linux.get("binary"), "linux binary binding missing")
    linux_binary = (bundle_manifest.parent / str(binary.get("path", ""))).resolve()
    linux_binary.relative_to(bundle_manifest.parent.resolve())
    if linux_binary.is_symlink() or not linux_binary.is_file():
        raise ValueError("linux binary missing")

    runtime_binding_path = args.runtime_binding.resolve()
    runtime_binding_path.relative_to(ROOT)
    runtime_binding = _read_json(
        runtime_binding_path, "representative runtime binding missing"
    )
    expected_binding_fields = {
        "schema_version",
        "checkpoint_id",
        "task_id_sha256",
        "revision",
        "instance_id_sha256",
        "image_key_sha256",
        "image_digest",
        "image_cache_binding_sha256",
        "base_commit",
        "final_patch_sha256",
        "official_receipt_sha256",
        "lock_sha256",
        "content_free",
        "content_sha256",
    }
    if set(runtime_binding) != expected_binding_fields:
        raise ValueError("representative runtime binding shape invalid")
    unsigned_binding = {
        name: runtime_binding[name]
        for name in runtime_binding
        if name != "content_sha256"
    }
    if (
        runtime_binding.get("schema_version")
        != "anchor.swebench-representative-runtime-binding.v1"
        or runtime_binding.get("content_free") is not True
        or runtime_binding.get("lock_sha256") != lock_sha256
        or runtime_binding.get("content_sha256")
        != hashlib.sha256(_canonical(unsigned_binding).encode("utf-8")).hexdigest()
    ):
        raise ValueError("representative runtime binding invalid")
    final_patch = runtime_binding_path.with_name("final.patch")
    official_receipt = runtime_binding_path.with_name("official-eval-receipt.json")
    final_patch.relative_to(ROOT)
    official_receipt.relative_to(ROOT)
    if final_patch.is_symlink() or not final_patch.is_file() or final_patch.stat().st_size < 1:
        raise ValueError("final patch missing")
    receipt = _read_json(official_receipt, "official receipt missing")
    for value in (
        runtime_binding.get("task_id_sha256"),
        runtime_binding.get("instance_id_sha256"),
        runtime_binding.get("image_key_sha256"),
        runtime_binding.get("image_cache_binding_sha256"),
    ):
        if not isinstance(value, str):
            raise ValueError("representative hash invalid")
        if not _SHA256.fullmatch(value):
            raise ValueError("representative hash invalid")
    patch_sha256 = sha256_file(final_patch)
    if (
        receipt.get("task_id_sha256") != runtime_binding["task_id_sha256"]
        or receipt.get("instance_id_sha256")
        != runtime_binding["instance_id_sha256"]
        or receipt.get("patch_sha256") != patch_sha256
        or receipt.get("lock_sha256") != lock_sha256
        or runtime_binding.get("final_patch_sha256") != patch_sha256
        or runtime_binding.get("official_receipt_sha256")
        != sha256_file(official_receipt)
    ):
        raise ValueError("receipt binding mismatch")

    value: dict[str, Any] = {
        "schema_version": REPRESENTATIVE_PROBE_ATTESTATION_SCHEMA,
        "execution_lock_sha256": lock_sha256,
        "opencode": {
            "baseline_commit": patch_value.get("baseline_commit"),
            "patch_manifest": _relative(patch_manifest),
            "patch_manifest_sha256": sha256_file(patch_manifest),
            "bundle_manifest": _relative(bundle_manifest),
            "bundle_manifest_sha256": sha256_file(bundle_manifest),
            "linux_binary": _relative(linux_binary),
            "linux_binary_sha256": sha256_file(linux_binary),
        },
        "ccswitch": {
            "route_manifest": _relative(route_manifest),
            "route_manifest_sha256": sha256_file(route_manifest),
        },
        "representative": {
            "checkpoint_id": receipt.get("checkpoint_id"),
            "task_id_sha256": runtime_binding["task_id_sha256"],
            "revision": receipt.get("revision"),
            "instance_id_sha256": runtime_binding["instance_id_sha256"],
            "image_key_sha256": runtime_binding["image_key_sha256"],
            "image_digest": receipt.get("image_digest"),
            "image_cache_binding_sha256": runtime_binding[
                "image_cache_binding_sha256"
            ],
            "base_commit": receipt.get("base_commit"),
            "final_patch": _relative(final_patch),
            "final_patch_sha256": patch_sha256,
            "official_receipt": _relative(official_receipt),
            "official_receipt_sha256": sha256_file(official_receipt),
        },
        "content_free": True,
    }
    value["content_sha256"] = hashlib.sha256(
        _canonical(value).encode("utf-8")
    ).hexdigest()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "configs/tooling/swebench_execution_v3.lock.json",
    )
    parser.add_argument(
        "--runtime-binding",
        type=Path,
        required=True,
        help="System-private representative-runtime-binding.json from one live probe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/tooling/swebench-v3/representative-probe-attestation.json",
    )
    args = parser.parse_args()
    try:
        value = build(args)
        output = args.output.resolve()
        output.relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"representative_probe_attestation=blocked reason={type(exc).__name__}")
        return 2
    print(f"representative_probe_attestation=written path={output}")
    print("live_execution=not_started_by_this_command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
