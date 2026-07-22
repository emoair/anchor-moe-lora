"""Build authenticated metadata prerequisites for Qwen toy diagnostics.

The producer authenticates only metadata manifests and identifier-only files.
It never opens Gold, held-out, or scaffold sample bodies.  Missing per-ID
metadata is frozen as unavailable instead of being represented by a zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

from anchor_mvp.swebench.toy_diagnostic_auditor import audit_toy_partition
from anchor_mvp.swebench.toy_diagnostic_generator import (
    canonical_bytes,
    generate_toy_records,
    sha256_bytes,
    sha256_value,
    source_id_inventory_sha256,
    source_id_token,
)


PRODUCER_VERSION = "anchor.qwen-toy-prerequisite-producer.v1"
MANIFEST_SCHEMA = "anchor.qwen-toy-prerequisite-manifest.v1"
INVENTORY_SCHEMA = "anchor.protected-source-id-inventory.v1"
SOURCE_ORDER = (
    "swebench_source",
    "gold_partition",
    "partial_gold_export",
    "heldout",
    "legacy_heldout_cases",
    "synthetic_scaffold",
)
READY_SOURCES = ("swebench_source", "heldout")
MISSING_SOURCES = (
    "gold_partition",
    "partial_gold_export",
    "legacy_heldout_cases",
    "synthetic_scaffold",
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_LINE_RE = re.compile(rb"^[0-9a-f]{64}\n$")


class QwenToyPrerequisiteError(RuntimeError):
    """A stable, content-free producer or audit failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise QwenToyPrerequisiteError(code)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class BytesSnapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


def _has_reparse_attribute(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _reject_reparse_chain(root: Path, path: Path, code: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise QwenToyPrerequisiteError(code) from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current_stat = current.lstat()
        except OSError as exc:
            raise QwenToyPrerequisiteError(code) from exc
        if stat.S_ISLNK(current_stat.st_mode) or _has_reparse_attribute(current_stat):
            _fail(code)


def _safe_existing_file(root: Path, relative_value: object, code: str) -> Path:
    if (
        not isinstance(relative_value, str)
        or not relative_value
        or "\\" in relative_value
    ):
        _fail(code)
    relative = Path(relative_value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        _fail(code)
    path = root.joinpath(*relative.parts)
    _reject_reparse_chain(root, path, code)
    try:
        if path.resolve(strict=True) != path.absolute() or not path.is_file():
            _fail(code)
    except OSError as exc:
        raise QwenToyPrerequisiteError(code) from exc
    return path


def _snapshot(path: Path, code: str, *, max_bytes: int) -> BytesSnapshot:
    try:
        if path.is_symlink():
            _fail(code)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or _has_reparse_attribute(before)
                or before.st_size > max_bytes
            ):
                _fail(code)
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except QwenToyPrerequisiteError:
        raise
    except OSError as exc:
        raise QwenToyPrerequisiteError(code) from exc
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(after)
        or identity != _stat_identity(current)
        or len(data) != after.st_size
    ):
        _fail(code)
    return BytesSnapshot(path, data, sha256_bytes(data), len(data), identity)


def _json(snapshot: BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        first = json.loads(snapshot.data.decode("utf-8", errors="strict"))
        second = json.loads(snapshot.data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenToyPrerequisiteError(code) from exc
    if first != second:
        _fail(code)
    return _mapping(first, code)


def _verify_unchanged(inventory: Mapping[Path, BytesSnapshot]) -> None:
    for path, expected in inventory.items():
        current = _snapshot(
            path,
            "qwen_toy_prerequisite_input_changed",
            max_bytes=max(expected.size, 1),
        )
        if current.sha256 != expected.sha256 or current.size != expected.size:
            _fail("qwen_toy_prerequisite_input_changed")


def _sidecar_bytes(digest: str, filename: str) -> bytes:
    if not _is_sha(digest) or "/" in filename or "\\" in filename:
        _fail("qwen_toy_prerequisite_sidecar_invalid")
    return f"{digest}  {filename}\n".encode("ascii")


def _canonical_document(value: object) -> bytes:
    return canonical_bytes(value) + b"\n"


def _file_ref(relative: str, data: bytes, records: int) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "records": records,
    }


def _parse_config(snapshot: BytesSnapshot) -> Mapping[str, Any]:
    config = _json(snapshot, "qwen_toy_prerequisite_config_invalid")
    _exact_keys(
        config,
        {
            "schema_version",
            "producer_id",
            "generator",
            "closed_grammar_path",
            "schemas",
            "implementations",
            "protected_sources",
            "limits",
            "safety",
        },
        "qwen_toy_prerequisite_config_invalid",
    )
    schemas = _mapping(config["schemas"], "qwen_toy_prerequisite_config_invalid")
    implementations = _mapping(
        config["implementations"], "qwen_toy_prerequisite_config_invalid"
    )
    limits = _mapping(config["limits"], "qwen_toy_prerequisite_config_invalid")
    safety = _mapping(config["safety"], "qwen_toy_prerequisite_config_invalid")
    sources = config["protected_sources"]
    if (
        config["schema_version"] != "anchor.qwen-toy-prerequisite-config.v1"
        or config["producer_id"] != PRODUCER_VERSION
        or set(schemas) != {"inventory", "record", "manifest", "trigger_receipt"}
        or set(implementations) != {"generator", "attester", "builder"}
        or not isinstance(sources, list)
        or tuple(
            item.get("source_class") if isinstance(item, Mapping) else None
            for item in sources
        )
        != SOURCE_ORDER
        or limits
        != {
            "max_input_file_bytes": 1_000_000,
            "max_records": 32,
            "max_output_file_bytes": 2_000_000,
            "max_total_output_bytes": 5_000_000,
        }
        or safety
        != {
            "diagnostic_only": True,
            "formal_training_authorized": False,
            "consumable_by_formal_release": False,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_content_reads": 0,
        }
    ):
        _fail("qwen_toy_prerequisite_config_invalid")
    return config


def _path_sha(
    root: Path, relative: str, inventory: dict[Path, BytesSnapshot]
) -> dict[str, str]:
    path = _safe_existing_file(root, relative, "qwen_toy_prerequisite_input_invalid")
    snapshot = _snapshot(
        path, "qwen_toy_prerequisite_input_invalid", max_bytes=1_000_000
    )
    inventory[path] = snapshot
    return {"path": relative, "sha256": snapshot.sha256}


def _authenticated_metadata(
    root: Path,
    entry: Mapping[str, Any],
    inventory: dict[Path, BytesSnapshot],
    *,
    open_canonical: bool,
) -> tuple[list[dict[str, object]], Mapping[str, Any] | None]:
    metadata: list[dict[str, object]] = []
    canonical: Mapping[str, Any] | None = None
    if open_canonical:
        relative = entry["canonical_path"]
        path = _safe_existing_file(
            root, relative, "qwen_toy_prerequisite_source_metadata_invalid"
        )
        snapshot = _snapshot(
            path,
            "qwen_toy_prerequisite_source_metadata_invalid",
            max_bytes=1_000_000,
        )
        if (
            snapshot.sha256 != entry["expected_sha256"]
            or snapshot.size != entry["expected_bytes"]
        ):
            _fail("qwen_toy_prerequisite_source_metadata_invalid")
        canonical = _json(snapshot, "qwen_toy_prerequisite_source_metadata_invalid")
        inventory[path] = snapshot
        metadata.append(
            {
                "role": "canonical_metadata_manifest",
                "path": relative,
                "sha256": snapshot.sha256,
                "bytes": snapshot.size,
            }
        )
    supplemental = entry.get("metadata_inputs", [])
    if not isinstance(supplemental, list):
        _fail("qwen_toy_prerequisite_source_metadata_invalid")
    for raw in supplemental:
        item = _mapping(raw, "qwen_toy_prerequisite_source_metadata_invalid")
        _exact_keys(
            item,
            {"role", "path", "sha256", "bytes"},
            "qwen_toy_prerequisite_source_metadata_invalid",
        )
        path = _safe_existing_file(
            root, item["path"], "qwen_toy_prerequisite_source_metadata_invalid"
        )
        snapshot = _snapshot(
            path,
            "qwen_toy_prerequisite_source_metadata_invalid",
            max_bytes=1_000_000,
        )
        if snapshot.sha256 != item["sha256"] or snapshot.size != item["bytes"]:
            _fail("qwen_toy_prerequisite_source_metadata_invalid")
        _json(snapshot, "qwen_toy_prerequisite_source_metadata_invalid")
        inventory[path] = snapshot
        metadata.append(
            {
                "role": item["role"],
                "path": item["path"],
                "sha256": snapshot.sha256,
                "bytes": snapshot.size,
            }
        )
    return metadata, canonical


def _identifier_contract(entry: Mapping[str, Any]) -> dict[str, object]:
    policy = {
        "algorithm": "sha256_utf8_namespace_nul_native_identifier_v1",
        "domain": entry["domain"],
        "namespace": entry["namespace"],
        "input_representation": entry["input_representation"],
    }
    namespaces = {"namespaces": [entry["namespace"]]}
    return {
        "canonical_preimage": "utf8(namespace) || NUL || utf8(native_identifier)",
        "inventory_digest_algorithm": (
            "sha256_sorted_unique_hex_lines_no_trailing_lf_v1"
        ),
        "identifier_domain_policy": policy,
        "identifier_domain_policy_sha256": sha256_value(policy),
        "namespace_inventory": namespaces,
        "namespace_inventory_sha256": sha256_value(namespaces),
    }


def _base_inventory_manifest(
    entry: Mapping[str, Any],
    *,
    status: str,
    authenticated: bool,
    inventory_schema_ref: Mapping[str, str],
    config_ref: Mapping[str, str],
    builder_ref: Mapping[str, str],
    metadata_inputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": INVENTORY_SCHEMA,
        "source_class": entry["source_class"],
        "status": status,
        "claim_scope": (
            "metadata_only_source_identifier_provenance_no_content_or_semantic_uniqueness_claim"
        ),
        "canonical_source": {
            "path": entry["canonical_path"],
            "expected_sha256": entry["expected_sha256"],
            "authenticated_from_metadata": authenticated,
            "content_read_count": 0,
        },
        "inventory_schema": dict(inventory_schema_ref),
        "producer": {
            "producer_id": PRODUCER_VERSION,
            "config": dict(config_ref),
            "implementation": dict(builder_ref),
            "canonical_json_policy": "utf8_sort_keys_compact_no_normalization_v1",
            "single_bytes_snapshot": True,
            "same_bytes_reparse": True,
            "final_source_recheck": True,
        },
        "identifier_contract": _identifier_contract(entry),
        "extraction": {
            "mode": entry["mode"],
            "metadata_inputs": list(metadata_inputs),
            "metadata_files_read": len(metadata_inputs),
            "body_files_read": 0,
        },
        "safety": {
            "metadata_only": True,
            "sample_content_emitted": False,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "formal_training_authorized": False,
        },
    }


def _ready_tokens(
    entry: Mapping[str, Any],
    canonical: Mapping[str, Any],
    metadata_inputs: Sequence[Mapping[str, object]],
    snapshots: Mapping[Path, BytesSnapshot],
    root: Path,
) -> list[str]:
    native: list[str] = []
    if entry["source_class"] == "swebench_source":
        declared_files = canonical.get("files")
        if not isinstance(declared_files, list):
            _fail("qwen_toy_prerequisite_swebench_metadata_invalid")
        declared_by_path = {
            item.get("path"): item
            for item in declared_files
            if isinstance(item, Mapping)
        }
        for item in metadata_inputs:
            if item["role"] != "identifier_allowlist":
                continue
            relative = str(item["path"])
            manifest_relative = relative.removeprefix(
                "datasets/public/swebench-full-bank-v1/"
            )
            declared = declared_by_path.get(manifest_relative)
            if (
                not isinstance(declared, Mapping)
                or declared.get("sha256") != item["sha256"]
                or declared.get("bytes") != item["bytes"]
            ):
                _fail("qwen_toy_prerequisite_swebench_metadata_invalid")
            path = root / relative
            allowlist = _json(
                snapshots[path], "qwen_toy_prerequisite_swebench_metadata_invalid"
            )
            if set(allowlist) != {
                "dataset_id",
                "dataset_revision",
                "instance_ids",
                "schema_version",
                "split",
            }:
                _fail("qwen_toy_prerequisite_swebench_metadata_invalid")
            identifiers = allowlist["instance_ids"]
            if (
                allowlist["dataset_id"] != canonical.get("dataset_id")
                or allowlist["dataset_revision"] != canonical.get("dataset_revision")
                or allowlist["schema_version"] != "anchor.swebench-train-allowlist.v1"
                or allowlist["split"] != "train"
                or not isinstance(identifiers, list)
                or len(identifiers) != declared.get("records")
                or any(not isinstance(value, str) or not value for value in identifiers)
            ):
                _fail("qwen_toy_prerequisite_swebench_metadata_invalid")
            native.extend(identifiers)
        if len(native) != 19_008 or len(set(native)) != 19_008:
            _fail("qwen_toy_prerequisite_swebench_metadata_invalid")
    elif entry["source_class"] == "heldout":
        raw = canonical.get("case_ids_sha256")
        if (
            not isinstance(raw, list)
            or canonical.get("case_count") != 6
            or len(raw) != 6
            or len(set(raw)) != 6
            or any(not _is_sha(item) for item in raw)
        ):
            _fail("qwen_toy_prerequisite_heldout_metadata_invalid")
        native.extend(raw)
    else:
        _fail("qwen_toy_prerequisite_ready_source_invalid")
    return sorted(source_id_token(str(entry["namespace"]), item) for item in native)


def _write_inventory(
    temporary: Path,
    entry: Mapping[str, Any],
    manifest: dict[str, object],
    tokens: list[str] | None,
) -> tuple[dict[str, object], bytes, bytes]:
    source_class = str(entry["source_class"])
    directory = temporary / "inventories" / source_class
    directory.mkdir(parents=True, exist_ok=False)
    if tokens is not None:
        token_bytes = b"".join(token.encode("ascii") + b"\n" for token in tokens)
        token_relative = f"inventories/{source_class}/source_ids.sha256.jsonl"
        manifest.update(
            {
                "source_id_count": len(tokens),
                "source_id_inventory_sha256": source_id_inventory_sha256(tokens),
                "inventory_file": _file_ref(token_relative, token_bytes, len(tokens)),
                "source_ids_recomputed": True,
            }
        )
        (directory / "source_ids.sha256.jsonl").write_bytes(token_bytes)
    else:
        manifest.update(
            {
                "source_ids_recomputed": False,
                "reason_codes": list(entry["reason_codes"]),
                "missing_fields": [
                    "source_id_count",
                    "source_id_inventory_sha256",
                    "inventory_file",
                ],
            }
        )
    manifest_bytes = _canonical_document(manifest)
    digest = sha256_bytes(manifest_bytes)
    sidecar_bytes = _sidecar_bytes(digest, "manifest.json")
    (directory / "manifest.json").write_bytes(manifest_bytes)
    (directory / "manifest.json.sha256").write_bytes(sidecar_bytes)
    return manifest, manifest_bytes, sidecar_bytes


def _inventory_ref(
    source_class: str,
    manifest: Mapping[str, object],
    manifest_bytes: bytes,
    sidecar_bytes: bytes,
) -> dict[str, object]:
    base: dict[str, object] = {
        "source_class": source_class,
        "status": manifest["status"],
        "manifest": {
            "path": f"inventories/{source_class}/manifest.json",
            "sha256": sha256_bytes(manifest_bytes),
        },
        "sidecar": {
            "path": f"inventories/{source_class}/manifest.json.sha256",
            "sha256": sha256_bytes(sidecar_bytes),
        },
        "sidecar_declared_sha256": sha256_bytes(manifest_bytes),
    }
    if manifest["status"] == "ready":
        base["source_id_count"] = manifest["source_id_count"]
        base["source_id_inventory_sha256"] = manifest["source_id_inventory_sha256"]
    return base


def build_qwen_toy_prerequisite(
    repo_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
) -> Mapping[str, Any]:
    """Build one immutable small fixture and publish it atomically."""

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        _fail("qwen_toy_prerequisite_root_invalid")
    config_raw = Path(config_path)
    if config_raw.is_absolute():
        try:
            config_relative = (
                config_raw.resolve(strict=True).relative_to(root).as_posix()
            )
        except (OSError, ValueError) as exc:
            raise QwenToyPrerequisiteError(
                "qwen_toy_prerequisite_config_invalid"
            ) from exc
    else:
        config_relative = config_raw.as_posix()
    config_file = _safe_existing_file(
        root, config_relative, "qwen_toy_prerequisite_config_invalid"
    )
    config_snapshot = _snapshot(
        config_file, "qwen_toy_prerequisite_config_invalid", max_bytes=1_000_000
    )
    config = _parse_config(config_snapshot)
    snapshots: dict[Path, BytesSnapshot] = {config_file: config_snapshot}

    schema_paths = _mapping(config["schemas"], "qwen_toy_prerequisite_config_invalid")
    implementation_paths = _mapping(
        config["implementations"], "qwen_toy_prerequisite_config_invalid"
    )
    schema_refs = {
        name: _path_sha(root, str(relative), snapshots)
        for name, relative in schema_paths.items()
    }
    implementation_refs = {
        name: _path_sha(root, str(relative), snapshots)
        for name, relative in implementation_paths.items()
    }
    grammar_relative = str(config["closed_grammar_path"])
    grammar_file = _safe_existing_file(
        root, grammar_relative, "qwen_toy_prerequisite_grammar_invalid"
    )
    grammar_snapshot = _snapshot(
        grammar_file, "qwen_toy_prerequisite_grammar_invalid", max_bytes=1_000_000
    )
    grammar = _json(grammar_snapshot, "qwen_toy_prerequisite_grammar_invalid")
    snapshots[grammar_file] = grammar_snapshot

    generated_records, toy_tokens = generate_toy_records(config["generator"], grammar)
    records_bytes = b"".join(
        canonical_bytes(record) + b"\n" for record in generated_records
    )
    toy_inventory_bytes = b"".join(
        token.encode("ascii") + b"\n" for token in sorted(toy_tokens)
    )
    independent_audit = audit_toy_partition(
        config["generator"], grammar, records_bytes, toy_inventory_bytes
    )

    output = Path(output_root)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    if output.exists() or output.parent.is_symlink():
        _fail("qwen_toy_prerequisite_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "toy").mkdir()
        (temporary / "toy" / "diagnostic.jsonl").write_bytes(records_bytes)
        (temporary / "toy" / "source_ids.sha256.jsonl").write_bytes(toy_inventory_bytes)

        config_ref = {"path": config_relative, "sha256": config_snapshot.sha256}
        builder_ref = implementation_refs["builder"]
        source_manifests: list[Mapping[str, Any]] = []
        protected_refs: list[dict[str, object]] = []
        protected_preimage: list[dict[str, object]] = []
        for raw_entry in config["protected_sources"]:
            entry = _mapping(raw_entry, "qwen_toy_prerequisite_config_invalid")
            source_class = str(entry["source_class"])
            open_canonical = source_class != "legacy_heldout_cases"
            metadata_inputs, canonical = _authenticated_metadata(
                root,
                entry,
                snapshots,
                open_canonical=open_canonical,
            )
            status = "ready" if source_class in READY_SOURCES else "unavailable"
            manifest = _base_inventory_manifest(
                entry,
                status=status,
                authenticated=open_canonical,
                inventory_schema_ref=schema_refs["inventory"],
                config_ref=config_ref,
                builder_ref=builder_ref,
                metadata_inputs=metadata_inputs,
            )
            tokens = None
            if status == "ready":
                if canonical is None:
                    _fail("qwen_toy_prerequisite_ready_source_invalid")
                tokens = _ready_tokens(
                    entry, canonical, metadata_inputs, snapshots, root
                )
            manifest, manifest_bytes, sidecar_bytes = _write_inventory(
                temporary, entry, manifest, tokens
            )
            source_manifests.append(manifest)
            protected_refs.append(
                _inventory_ref(source_class, manifest, manifest_bytes, sidecar_bytes)
            )
            contract = _mapping(
                manifest["identifier_contract"],
                "qwen_toy_prerequisite_internal_invalid",
            )
            preimage: dict[str, object] = {
                "source_class": source_class,
                "status": status,
                "manifest_sha256": sha256_bytes(manifest_bytes),
                "sidecar_sha256": sha256_bytes(sidecar_bytes),
                "identifier_domain_policy_sha256": contract[
                    "identifier_domain_policy_sha256"
                ],
                "namespace_inventory_sha256": contract["namespace_inventory_sha256"],
            }
            if status == "ready":
                preimage["source_id_count"] = manifest["source_id_count"]
                preimage["source_id_inventory_sha256"] = manifest[
                    "source_id_inventory_sha256"
                ]
            else:
                preimage["reason_codes"] = manifest["reason_codes"]
            protected_preimage.append(preimage)

        _verify_unchanged(snapshots)
        read_inputs = [
            {
                "role": "generator_implementation",
                "path": implementation_refs["generator"]["path"],
                "sha256": implementation_refs["generator"]["sha256"],
                "bytes": snapshots[
                    root / implementation_refs["generator"]["path"]
                ].size,
            },
            {
                "role": "generator_config",
                "path": config_relative,
                "sha256": config_snapshot.sha256,
                "bytes": config_snapshot.size,
            },
            {
                "role": "closed_grammar",
                "path": grammar_relative,
                "sha256": grammar_snapshot.sha256,
                "bytes": grammar_snapshot.size,
            },
        ]
        generation_read_set = {
            "scope": "declared_semantic_generation_inputs_only",
            "inputs": read_inputs,
            "inventory_sha256": sha256_value(read_inputs),
            "unexpected_reads": 0,
            "protected_content_reads": 0,
        }
        protected_set_sha = sha256_value(protected_preimage)
        audit_document = {
            **independent_audit,
            "claim_scope": (
                "toy_generation_rebuild_and_metadata_inventory_authentication_only"
            ),
            "attester": implementation_refs["attester"],
            "builder": implementation_refs["builder"],
            "generation_read_set_sha256": generation_read_set["inventory_sha256"],
            "protected_inventory_set_sha256": protected_set_sha,
            "protected_inventory_coverage": {
                "ready": len(READY_SOURCES),
                "total": len(SOURCE_ORDER),
                "missing_source_classes": list(MISSING_SOURCES),
            },
            "v1_attestation_emitted": False,
            "zero_intersection_claimed": False,
            "final_recheck_passed": True,
        }
        audit_bytes = _canonical_document(audit_document)
        audit_sha = sha256_bytes(audit_bytes)
        audit_sidecar = _sidecar_bytes(audit_sha, "audit.json")
        (temporary / "audit.json").write_bytes(audit_bytes)
        (temporary / "audit.json.sha256").write_bytes(audit_sidecar)

        manifest_document = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "toy_generation_verified_protected_inventory_incomplete",
            "claim_scope": (
                "diagnostic_only_toy_generation_and_metadata_prerequisite_status"
            ),
            "producer": {
                "producer_id": PRODUCER_VERSION,
                "config": config_ref,
                "closed_grammar": {
                    "path": grammar_relative,
                    "sha256": grammar_snapshot.sha256,
                },
                "inventory_schema": schema_refs["inventory"],
                "record_schema": schema_refs["record"],
                "manifest_schema": schema_refs["manifest"],
                "trigger_receipt_schema": schema_refs["trigger_receipt"],
                "generator_implementation": implementation_refs["generator"],
                "attester_implementation": implementation_refs["attester"],
                "builder_implementation": implementation_refs["builder"],
                "canonical_json_policy": ("utf8_sort_keys_compact_no_normalization_v1"),
                "manifest_sidecar_required": True,
                "single_bytes_snapshot": True,
                "same_bytes_reparse": True,
                "final_source_recheck": True,
            },
            "toy": {
                "namespace": "anchor.qwen-toy-diagnostic.v1",
                "partition": "diagnostic_only",
                "record_count": len(generated_records),
                "records": _file_ref(
                    "toy/diagnostic.jsonl", records_bytes, len(generated_records)
                ),
                "source_id_inventory": _file_ref(
                    "toy/source_ids.sha256.jsonl",
                    toy_inventory_bytes,
                    len(toy_tokens),
                ),
                "source_id_inventory_sha256": source_id_inventory_sha256(toy_tokens),
                "deterministic_rebuild_matches": True,
                "tokenizer_bound": False,
            },
            "protected_inventories": protected_refs,
            "generation_read_set": generation_read_set,
            "request_local_trigger_binding": {
                "schema_version": (
                    "anchor.qwen-request-local-trigger-materialization.v1"
                ),
                "status": "pending_request_local_materialization",
                "activation_semantics": "next_request_input_activation_only",
                "serialization_scope": (
                    "exact_full_chat_templated_request2_bytes_single_tokenization"
                ),
                "tokenizer_binding_sha256": None,
                "chat_template_sha256": None,
                "exact_r2_serialization_sha256": None,
                "ordered_input_token_ids_sha256": None,
                "trigger_span_zero_based_exclusive": None,
                "boundary_overhang": None,
                "isolated_trigger_encoding_authoritative": False,
                "full_r2_single_tokenization_required": True,
                "global_token_index_emitted": False,
                "token_ids_emitted": False,
                "planner_request1_private_kv_reused": False,
                "formal_training_authorized": False,
            },
            "audit": {
                "path": "audit.json",
                "sha256": audit_sha,
                "sidecar_path": "audit.json.sha256",
                "sidecar_sha256": sha256_bytes(audit_sidecar),
                "independent_rebuild_passed": True,
                "same_snapshot_reparse_passed": True,
                "final_recheck_passed": True,
            },
            "proof": {
                "status": "toy_generation_verified_protected_inventory_incomplete",
                "coverage_ready_count": len(READY_SOURCES),
                "coverage_total": len(SOURCE_ORDER),
                "missing_source_classes": list(MISSING_SOURCES),
                "protected_inventory_set_sha256": protected_set_sha,
                "v1_attestation_emitted": False,
                "zero_intersection_claimed": False,
                "formal_training_authorized": False,
            },
            "execution": {
                "provider_requests": 0,
                "network_requests": 0,
                "model_loads": 0,
                "gpu_requests": 0,
                "protected_content_reads": 0,
            },
            "safety": {
                "diagnostic_only": True,
                "formal_training_authorized": False,
                "consumable_by_formal_release": False,
                "canonical_gold_written": False,
                "heldout_written": False,
                "sample_content_emitted": False,
            },
        }
        manifest_bytes = _canonical_document(manifest_document)
        manifest_sha = sha256_bytes(manifest_bytes)
        manifest_sidecar = _sidecar_bytes(manifest_sha, "manifest.json")
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "manifest.json.sha256").write_bytes(manifest_sidecar)

        total_size = sum(
            path.stat().st_size for path in temporary.rglob("*") if path.is_file()
        )
        if total_size > int(
            _mapping(config["limits"], "invalid")["max_total_output_bytes"]
        ):
            _fail("qwen_toy_prerequisite_output_too_large")
        os.replace(temporary, output)
        return manifest_document
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_artifact_file(
    root: Path, relative: str, inventory: dict[Path, BytesSnapshot], max_bytes: int
) -> BytesSnapshot:
    path = _safe_existing_file(root, relative, "qwen_toy_prerequisite_artifact_invalid")
    snapshot = _snapshot(
        path, "qwen_toy_prerequisite_artifact_invalid", max_bytes=max_bytes
    )
    inventory[path] = snapshot
    return snapshot


def audit_qwen_toy_prerequisite(
    repo_root: str | Path,
    config_path: str | Path,
    artifact_root: str | Path,
) -> Mapping[str, Any]:
    """Fail closed while authenticating a published prerequisite artifact."""

    root = Path(repo_root).resolve(strict=True)
    artifact = Path(artifact_root)
    if not artifact.is_absolute():
        artifact = root / artifact
    artifact = artifact.resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_snapshot = _snapshot(
        config_file.resolve(strict=True),
        "qwen_toy_prerequisite_config_invalid",
        max_bytes=1_000_000,
    )
    config = _parse_config(config_snapshot)
    grammar_path = _safe_existing_file(
        root, config["closed_grammar_path"], "qwen_toy_prerequisite_grammar_invalid"
    )
    grammar_snapshot = _snapshot(
        grammar_path, "qwen_toy_prerequisite_grammar_invalid", max_bytes=1_000_000
    )
    grammar = _json(grammar_snapshot, "qwen_toy_prerequisite_grammar_invalid")
    observed: dict[Path, BytesSnapshot] = {
        config_snapshot.path: config_snapshot,
        grammar_snapshot.path: grammar_snapshot,
    }
    schema_paths = _mapping(config["schemas"], "qwen_toy_prerequisite_config_invalid")
    implementation_paths = _mapping(
        config["implementations"], "qwen_toy_prerequisite_config_invalid"
    )
    schema_refs = {
        name: _path_sha(root, str(relative), observed)
        for name, relative in schema_paths.items()
    }
    implementation_refs = {
        name: _path_sha(root, str(relative), observed)
        for name, relative in implementation_paths.items()
    }
    manifest_snapshot = _read_artifact_file(
        artifact, "manifest.json", observed, 1_000_000
    )
    sidecar_snapshot = _read_artifact_file(
        artifact, "manifest.json.sha256", observed, 256
    )
    if sidecar_snapshot.data != _sidecar_bytes(
        manifest_snapshot.sha256, "manifest.json"
    ):
        _fail("qwen_toy_prerequisite_sidecar_mismatch")
    manifest = _json(manifest_snapshot, "qwen_toy_prerequisite_manifest_invalid")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status")
        != "toy_generation_verified_protected_inventory_incomplete"
    ):
        _fail("qwen_toy_prerequisite_manifest_invalid")
    producer = _mapping(
        manifest.get("producer"), "qwen_toy_prerequisite_manifest_invalid"
    )
    if producer.get("config", {}).get("sha256") != config_snapshot.sha256:
        _fail("qwen_toy_prerequisite_config_drift")
    if producer.get("closed_grammar", {}).get("sha256") != grammar_snapshot.sha256:
        _fail("qwen_toy_prerequisite_grammar_drift")
    expected_producer_refs = {
        "inventory_schema": schema_refs["inventory"],
        "record_schema": schema_refs["record"],
        "manifest_schema": schema_refs["manifest"],
        "trigger_receipt_schema": schema_refs["trigger_receipt"],
        "generator_implementation": implementation_refs["generator"],
        "attester_implementation": implementation_refs["attester"],
        "builder_implementation": implementation_refs["builder"],
    }
    if any(producer.get(key) != value for key, value in expected_producer_refs.items()):
        _fail("qwen_toy_prerequisite_implementation_or_schema_drift")

    toy = _mapping(manifest.get("toy"), "qwen_toy_prerequisite_manifest_invalid")
    records_ref = _mapping(toy.get("records"), "qwen_toy_prerequisite_manifest_invalid")
    tokens_ref = _mapping(
        toy.get("source_id_inventory"), "qwen_toy_prerequisite_manifest_invalid"
    )
    records = _read_artifact_file(
        artifact, str(records_ref["path"]), observed, 2_000_000
    )
    tokens = _read_artifact_file(artifact, str(tokens_ref["path"]), observed, 2_000_000)
    if (
        records.sha256 != records_ref.get("sha256")
        or records.size != records_ref.get("bytes")
        or tokens.sha256 != tokens_ref.get("sha256")
        or tokens.size != tokens_ref.get("bytes")
    ):
        _fail("qwen_toy_prerequisite_toy_file_mismatch")
    independent = audit_toy_partition(
        config["generator"], grammar, records.data, tokens.data
    )
    if independent["source_id_inventory_sha256"] != toy.get(
        "source_id_inventory_sha256"
    ):
        _fail("qwen_toy_prerequisite_toy_inventory_mismatch")

    protected = manifest.get("protected_inventories")
    if (
        not isinstance(protected, list)
        or tuple(item.get("source_class") for item in protected) != SOURCE_ORDER
    ):
        _fail("qwen_toy_prerequisite_inventory_order_invalid")
    for item, raw_entry in zip(protected, config["protected_sources"], strict=True):
        entry = _mapping(raw_entry, "qwen_toy_prerequisite_config_invalid")
        source_class = str(item["source_class"])
        manifest_ref = _mapping(
            item.get("manifest"), "qwen_toy_prerequisite_inventory_invalid"
        )
        sidecar_ref = _mapping(
            item.get("sidecar"), "qwen_toy_prerequisite_inventory_invalid"
        )
        source_manifest_snapshot = _read_artifact_file(
            artifact, str(manifest_ref["path"]), observed, 1_000_000
        )
        source_sidecar_snapshot = _read_artifact_file(
            artifact, str(sidecar_ref["path"]), observed, 256
        )
        if (
            source_manifest_snapshot.sha256 != manifest_ref.get("sha256")
            or source_sidecar_snapshot.sha256 != sidecar_ref.get("sha256")
            or source_sidecar_snapshot.data
            != _sidecar_bytes(source_manifest_snapshot.sha256, "manifest.json")
        ):
            _fail("qwen_toy_prerequisite_inventory_invalid")
        source_manifest = _json(
            source_manifest_snapshot, "qwen_toy_prerequisite_inventory_invalid"
        )
        if source_manifest.get("source_class") != source_class or source_manifest.get(
            "status"
        ) != item.get("status"):
            _fail("qwen_toy_prerequisite_inventory_invalid")
        if item.get("status") == "ready":
            file_ref = _mapping(
                source_manifest.get("inventory_file"),
                "qwen_toy_prerequisite_inventory_invalid",
            )
            source_tokens = _read_artifact_file(
                artifact, str(file_ref["path"]), observed, 2_000_000
            )
            token_lines = source_tokens.data.splitlines(keepends=True)
            if (
                source_tokens.sha256 != file_ref.get("sha256")
                or source_tokens.size != file_ref.get("bytes")
                or len(token_lines) != file_ref.get("records")
                or any(_TOKEN_LINE_RE.fullmatch(line) is None for line in token_lines)
            ):
                _fail("qwen_toy_prerequisite_inventory_invalid")
            opaque_tokens = [line[:-1].decode("ascii") for line in token_lines]
            metadata_inputs, canonical = _authenticated_metadata(
                root,
                entry,
                observed,
                open_canonical=True,
            )
            if canonical is None:
                _fail("qwen_toy_prerequisite_inventory_invalid")
            expected_tokens = _ready_tokens(
                entry, canonical, metadata_inputs, observed, root
            )
            if (
                opaque_tokens != sorted(set(opaque_tokens))
                or opaque_tokens != expected_tokens
                or source_id_inventory_sha256(opaque_tokens)
                != source_manifest.get("source_id_inventory_sha256")
            ):
                _fail("qwen_toy_prerequisite_inventory_invalid")
        else:
            metadata_inputs, _ = _authenticated_metadata(
                root,
                entry,
                observed,
                open_canonical=source_class != "legacy_heldout_cases",
            )
            if (
                any(
                    key in source_manifest
                    for key in (
                        "source_id_count",
                        "source_id_inventory_sha256",
                        "inventory_file",
                    )
                )
                or source_manifest.get("reason_codes") != entry.get("reason_codes")
                or source_manifest.get("extraction", {}).get("metadata_inputs")
                != metadata_inputs
            ):
                _fail("qwen_toy_prerequisite_unavailable_claim_invalid")

    audit_ref = _mapping(
        manifest.get("audit"), "qwen_toy_prerequisite_manifest_invalid"
    )
    audit_snapshot = _read_artifact_file(
        artifact, str(audit_ref["path"]), observed, 1_000_000
    )
    audit_sidecar = _read_artifact_file(
        artifact, str(audit_ref["sidecar_path"]), observed, 256
    )
    if (
        audit_snapshot.sha256 != audit_ref.get("sha256")
        or audit_sidecar.sha256 != audit_ref.get("sidecar_sha256")
        or audit_sidecar.data != _sidecar_bytes(audit_snapshot.sha256, "audit.json")
    ):
        _fail("qwen_toy_prerequisite_audit_invalid")
    audit_document = _json(audit_snapshot, "qwen_toy_prerequisite_audit_invalid")
    if (
        audit_document.get("status") != "passed"
        or audit_document.get("v1_attestation_emitted") is not False
        or audit_document.get("zero_intersection_claimed") is not False
    ):
        _fail("qwen_toy_prerequisite_audit_invalid")
    proof = _mapping(manifest.get("proof"), "qwen_toy_prerequisite_manifest_invalid")
    if (
        proof.get("coverage_ready_count") != 2
        or proof.get("coverage_total") != 6
        or proof.get("missing_source_classes") != list(MISSING_SOURCES)
        or proof.get("v1_attestation_emitted") is not False
        or proof.get("zero_intersection_claimed") is not False
        or any(
            key in proof
            for key in (
                "intersection_count",
                "intersection_proof_sha256",
                "proof_input_inventory_sha256",
            )
        )
    ):
        _fail("qwen_toy_prerequisite_proof_invalid")
    trigger_binding = _mapping(
        manifest.get("request_local_trigger_binding"),
        "qwen_toy_prerequisite_trigger_binding_invalid",
    )
    if trigger_binding != {
        "schema_version": "anchor.qwen-request-local-trigger-materialization.v1",
        "status": "pending_request_local_materialization",
        "activation_semantics": "next_request_input_activation_only",
        "serialization_scope": (
            "exact_full_chat_templated_request2_bytes_single_tokenization"
        ),
        "tokenizer_binding_sha256": None,
        "chat_template_sha256": None,
        "exact_r2_serialization_sha256": None,
        "ordered_input_token_ids_sha256": None,
        "trigger_span_zero_based_exclusive": None,
        "boundary_overhang": None,
        "isolated_trigger_encoding_authoritative": False,
        "full_r2_single_tokenization_required": True,
        "global_token_index_emitted": False,
        "token_ids_emitted": False,
        "planner_request1_private_kv_reused": False,
        "formal_training_authorized": False,
    }:
        _fail("qwen_toy_prerequisite_trigger_binding_invalid")
    _verify_unchanged(observed)
    return manifest
