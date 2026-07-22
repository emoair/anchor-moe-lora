"""Materialize the authenticated 44-token Qwen request-local trigger receipt.

This module is deliberately tokenizer-only. It reads four small tokenizer
assets into one authenticated byte snapshot, reconstructs a private local
tokenizer directory from those bytes, and tokenizes the complete chat-templated
request exactly once. It never loads model weights, requests a GPU, reads a
project dataset, or emits raw token IDs or a global token index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from anchor_mvp.research import qwen_alora_prefix_kv_diagnostic as common


CONFIG_VERSION = "anchor.qwen-request-local-trigger-receipt-config.v2"
RECEIPT_VERSION = "anchor.qwen-request-local-trigger-receipt.v2"
PRODUCER_ID = "anchor.qwen-request-local-trigger-materializer.v2"
ORDERED_IDS_DIGEST_ALGORITHM = "sha256_concat_signed_int64_big_endian_v1"
EXPECTED_PRODUCER_COMMIT = "744e23f975b13923903f5fabe04c32e74ea25dc4"
EXPECTED_CONSUMER_BASELINE = "b0441e6beaa07b180d7fc69e462b4d2babf21792"
CONSUMER_BASELINE_SEMANTICS = "required_ancestor_or_equal_dependency"
DEFAULT_CONFIG = "configs/research/qwen_request_local_trigger_receipt_v2.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMPORTED_IMPLEMENTATION_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


class TriggerReceiptError(RuntimeError):
    """The tokenizer-only materialization failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def ordered_token_ids_sha256(values: Sequence[int]) -> str:
    """Hash signed int64 big-endian token IDs without a JSON preimage."""

    payload = bytearray()
    for value in values:
        if isinstance(value, bool):
            raise TriggerReceiptError("token IDs must be integers, not booleans")
        integer = int(value)
        try:
            payload.extend(integer.to_bytes(8, "big", signed=True))
        except OverflowError as exc:
            raise TriggerReceiptError("token ID is outside signed int64") from exc
    return _sha256(bytes(payload))


def _legacy_json_array_sha256(values: Sequence[int]) -> str:
    raw = json.dumps([int(value) for value in values], separators=(",", ":")).encode(
        "utf-8"
    )
    return _sha256(raw)


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _assert_physical_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if os.path.normcase(os.path.realpath(absolute)) != os.path.normcase(str(absolute)):
        raise TriggerReceiptError(f"{label} has realpath drift: {absolute}")
    current = absolute
    while True:
        if os.path.lexists(current) and _is_reparse_or_symlink(current):
            raise TriggerReceiptError(
                f"{label} contains a symlink/reparse component: {current}"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not absolute.is_file():
        raise TriggerReceiptError(f"{label} must be a physical file: {absolute}")
    return absolute


def _read_snapshot(path: Path, *, label: str) -> dict[str, Any]:
    physical = _assert_physical_file(path, label=label)
    before = physical.stat()
    raw = physical.read_bytes()
    after = physical.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    observed = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity != observed or len(raw) != before.st_size:
        raise TriggerReceiptError(f"{label} changed while being read")
    return {
        "path": physical,
        "raw": raw,
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "identity": identity,
    }


def _recheck_snapshot(snapshot: Mapping[str, Any], *, label: str) -> None:
    current = _read_snapshot(Path(snapshot["path"]), label=label)
    if (
        current["sha256"] != snapshot["sha256"]
        or current["bytes"] != snapshot["bytes"]
        or current["identity"] != snapshot["identity"]
    ):
        raise TriggerReceiptError(f"{label} changed after authentication")


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TriggerReceiptError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TriggerReceiptError(f"{label} must contain a JSON object")
    return value


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        observed = os.environ.get(name)
        if observed is not None:
            return observed
        if fallback is not None:
            return fallback
        raise TriggerReceiptError(f"required environment variable is unset: {name}")

    return _ENV_RE.sub(replace, value)


def _repo_path(value: str, *, label: str) -> Path:
    path = (_REPO_ROOT / value).resolve(strict=False)
    try:
        path.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise TriggerReceiptError(f"{label} escapes repository root") from exc
    return path


def _tokenizer_root(config: Mapping[str, Any]) -> Path:
    raw = _expand_env_string(str(config["tokenizer"]["local_path"]))
    path = Path(raw)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    absolute = Path(os.path.abspath(path))
    if os.path.normcase(os.path.realpath(absolute)) != os.path.normcase(str(absolute)):
        raise TriggerReceiptError("tokenizer root has realpath drift")
    if not absolute.is_dir() or _is_reparse_or_symlink(absolute):
        raise TriggerReceiptError("tokenizer root must be a physical directory")
    return absolute


def _schema_snapshot(binding: Mapping[str, Any], *, label: str) -> tuple[dict, dict]:
    snapshot = _read_snapshot(
        _repo_path(str(binding["path"]), label=label), label=label
    )
    if snapshot["sha256"] != binding["sha256"]:
        raise TriggerReceiptError(f"{label} SHA-256 mismatch")
    schema = _json_object(snapshot["raw"], label=label)
    Draft202012Validator.check_schema(schema)
    return schema, snapshot


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict, dict[str, dict]]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    config_snapshot = _read_snapshot(config_path, label="materializer config")
    try:
        value = yaml.safe_load(config_snapshot["raw"].decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TriggerReceiptError(
            "materializer config must be UTF-8 YAML/JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TriggerReceiptError("materializer config root must be an object")
    config_schema, config_schema_snapshot = _schema_snapshot(
        value.get("schemas", {}).get("config", {}), label="config schema"
    )
    errors = sorted(
        Draft202012Validator(config_schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        raise TriggerReceiptError(f"config schema rejected input: {errors[0].message}")
    receipt_schema, receipt_schema_snapshot = _schema_snapshot(
        value["schemas"]["receipt"], label="receipt schema"
    )
    if value["schema_version"] != CONFIG_VERSION:
        raise TriggerReceiptError("config version mismatch")
    if value["schemas"]["receipt"]["version"] != RECEIPT_VERSION:
        raise TriggerReceiptError("receipt version binding mismatch")
    if value["source_bindings"] != {
        "producer_commit": EXPECTED_PRODUCER_COMMIT,
        "consumer_baseline_commit": EXPECTED_CONSUMER_BASELINE,
        "consumer_baseline_semantics": CONSUMER_BASELINE_SEMANTICS,
    }:
        raise TriggerReceiptError("source commit binding drift")
    return value, {
        "config": config_snapshot,
        "config_schema": config_schema_snapshot,
        "receipt_schema": receipt_schema_snapshot,
        "receipt_schema_value": {"value": receipt_schema},
    }


def _git_run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _authenticate_consumer_baseline() -> None:
    """Require the frozen consumer dependency to be an ancestor of HEAD.

    The materializer's executable identity is authenticated independently by
    physical config/schema/implementation snapshots.  The consumer commit is a
    dependency baseline, not the forever-current checkout identity; requiring
    equality would make a committed materializer impossible to reproduce.
    """

    head_result = _git_run("rev-parse", "HEAD")
    observed_head = head_result.stdout.strip()
    if head_result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", observed_head):
        raise TriggerReceiptError("unable to authenticate consumer Git HEAD")

    baseline_result = _git_run(
        "rev-parse", "--verify", f"{EXPECTED_CONSUMER_BASELINE}^{{commit}}"
    )
    observed_baseline = baseline_result.stdout.strip()
    if (
        baseline_result.returncode != 0
        or observed_baseline != EXPECTED_CONSUMER_BASELINE
    ):
        raise TriggerReceiptError(
            "consumer dependency baseline is unavailable or drifted"
        )

    ancestor_result = _git_run(
        "merge-base", "--is-ancestor", EXPECTED_CONSUMER_BASELINE, observed_head
    )
    if ancestor_result.returncode != 0:
        raise TriggerReceiptError(
            "consumer dependency baseline is not an ancestor of HEAD"
        )


def _asset_snapshots(config: Mapping[str, Any]) -> tuple[Path, dict[str, dict]]:
    root = _tokenizer_root(config)
    observed: dict[str, dict] = {}
    for entry in config["tokenizer"]["assets"]:
        name = str(entry["path"])
        snapshot = _read_snapshot(root / name, label=f"tokenizer asset {name}")
        if snapshot["sha256"] != entry["sha256"] or snapshot["bytes"] != entry["bytes"]:
            raise TriggerReceiptError(f"tokenizer asset identity mismatch: {name}")
        observed[name] = snapshot
    return root, observed


def _binding_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = config["tokenizer"]
    runtime_descriptor = dict(tokenizer["runtime"])
    serialization_policy = dict(tokenizer["serialization_policy"])
    special_token_policy = dict(tokenizer["special_token_policy"])
    return {
        "backend": "local_offline_hf_fast_tokenizer",
        "tokenizer_id": tokenizer["id"],
        "tokenizer_revision": tokenizer["revision"],
        "asset_inventory": [dict(item) for item in tokenizer["assets"]],
        "tokenizer_assets_sha256": tokenizer["tokenizer_assets_sha256"],
        "runtime_descriptor": runtime_descriptor,
        "tokenizer_runtime_descriptor_sha256": _sha256(
            _canonical_json_bytes(runtime_descriptor)
        ),
        "chat_template": dict(tokenizer["chat_template"]),
        "serialization_policy": serialization_policy,
        "serialization_policy_sha256": _sha256(
            _canonical_json_bytes(serialization_policy)
        ),
        "special_token_policy": special_token_policy,
        "special_token_policy_sha256": _sha256(
            _canonical_json_bytes(special_token_policy)
        ),
        "network_access": False,
        "local_files_only": True,
        "trust_remote_code": False,
        "model_weight_files_read": False,
        "gguf_files_read": False,
        "exact_tokenization_ready": True,
        "training_authorized": False,
    }


def tokenizer_binding_sha256(config: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(_binding_from_config(config)))


def _implementation_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    implementation_path = _repo_path(
        str(config["implementation"]["path"]), label="implementation"
    )
    snapshot = _read_snapshot(implementation_path, label="implementation")
    if snapshot["sha256"] != _IMPORTED_IMPLEMENTATION_SHA256:
        raise TriggerReceiptError(
            "implementation bytes differ from the module imported for this run"
        )
    return snapshot


def _materialize_with_authenticated_assets(
    config: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import tokenizers  # imported only for a version identity
    import transformers
    from transformers import AutoTokenizer

    expected_runtime = config["tokenizer"]["runtime"]
    observed_versions = {
        "python": sys.version.split()[0],
        "transformers": str(transformers.__version__),
        "tokenizers": str(tokenizers.__version__),
    }
    for name, observed in observed_versions.items():
        if observed != expected_runtime[name]:
            raise TriggerReceiptError(
                f"tokenizer runtime {name} mismatch: {observed!r}"
            )

    with tempfile.TemporaryDirectory(prefix="anchor-qwen-tokenizer-v2-") as raw_dir:
        snapshot_root = Path(raw_dir)
        for name, snapshot in assets.items():
            (snapshot_root / name).write_bytes(snapshot["raw"])
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_root,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if type(tokenizer).__name__ != expected_runtime["tokenizer_class"]:
            raise TriggerReceiptError("tokenizer class mismatch")
        if (
            bool(getattr(tokenizer, "is_fast", False))
            is not expected_runtime["is_fast"]
        ):
            raise TriggerReceiptError("fast-tokenizer identity mismatch")
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise TriggerReceiptError("tokenizer exposes no chat template")
        if (
            _sha256(chat_template.encode("utf-8"))
            != config["tokenizer"]["chat_template"]["exact_utf8_sha256"]
        ):
            raise TriggerReceiptError("chat-template UTF-8 identity mismatch")

        request2 = config["request2"]
        trigger = str(request2["trigger_text"])
        user = "\n".join(
            (
                str(request2["user_prefix"]),
                trigger,
                str(request2["user_suffix"]),
            )
        )
        messages = [
            {"role": "system", "content": str(request2["system_content"])},
            {"role": "user", "content": user},
        ]
        serialized = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(serialized, str) or not serialized:
            raise TriggerReceiptError("chat template produced no request2")
        if serialized.count(trigger) != 1:
            raise TriggerReceiptError("trigger text must occur exactly once")
        encoded = tokenizer(
            serialized,
            add_special_tokens=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )

    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise TriggerReceiptError("tokenizer returned invalid input IDs")
    if not isinstance(raw_offsets, Sequence) or len(raw_offsets) != len(raw_ids):
        raise TriggerReceiptError("tokenizer returned invalid offset mapping")
    input_ids = [int(value) for value in raw_ids]
    offsets: list[tuple[int, int]] = []
    for item in raw_offsets:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise TriggerReceiptError("offset entry is invalid")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start or end > len(serialized):
            raise TriggerReceiptError("offset entry is out of range")
        offsets.append((start, end))

    char_start = serialized.index(trigger)
    char_end = char_start + len(trigger)
    overlaps = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < char_end and end > char_start
    ]
    if not overlaps or overlaps != list(range(overlaps[0], overlaps[-1] + 1)):
        raise TriggerReceiptError("trigger covering span is missing/noncontiguous")
    token_start, token_end = overlaps[0], overlaps[-1] + 1
    cover_start, cover_end = offsets[token_start][0], offsets[token_end - 1][1]
    if cover_start > char_start or cover_end < char_end:
        raise TriggerReceiptError("token span does not cover trigger text")
    trigger_ids = input_ids[token_start:token_end]
    matches = sum(
        input_ids[index : index + len(trigger_ids)] == trigger_ids
        for index in range(len(input_ids) - len(trigger_ids) + 1)
    )
    leading = serialized[cover_start:char_start]
    trailing = serialized[char_end:cover_end]
    overhang = {
        "leading_utf8_bytes": len(leading.encode("utf-8")),
        "trailing_utf8_bytes": len(trailing.encode("utf-8")),
        "leading_codepoints": len(leading),
        "trailing_codepoints": len(trailing),
    }
    return {
        "activation_semantics": "next_request_input_activation_only",
        "serialization_scope": (
            "exact_full_chat_templated_request2_bytes_single_tokenization"
        ),
        "exact_r2_serialization_sha256": _sha256(serialized.encode("utf-8")),
        "chat_template_sha256": _sha256(chat_template.encode("utf-8")),
        "ordered_input_token_ids_sha256": ordered_token_ids_sha256(input_ids),
        "ordered_input_token_ids_digest_algorithm": ORDERED_IDS_DIGEST_ALGORITHM,
        "legacy_manual_probe_ordered_ids_sha256": _legacy_json_array_sha256(input_ids),
        "trigger_text_sha256": _sha256(trigger.encode("utf-8")),
        "trigger_covering_token_ids_sha256": ordered_token_ids_sha256(trigger_ids),
        "legacy_manual_probe_trigger_ids_sha256": _legacy_json_array_sha256(
            trigger_ids
        ),
        "trigger_text_occurrences": serialized.count(trigger),
        "trigger_token_sequence_occurrences": matches,
        "trigger_span_zero_based_exclusive": {
            "index_base": "zero",
            "end_semantics": "exclusive",
            "start": token_start,
            "end": token_end,
        },
        "total_tokens": len(input_ids),
        "trigger_span_width": token_end - token_start,
        "boundary_overhang": overhang,
        "complete_r2_tokenization_count": 1,
        "isolated_trigger_encoding_authoritative": False,
        "raw_token_ids_emitted": False,
        "global_token_index_emitted": False,
        "planner_request1_private_kv_reused": False,
    }


def _validate_expected_materialization(
    config: Mapping[str, Any], materialization: Mapping[str, Any]
) -> None:
    expected = config["expected_materialization"]
    flat_pairs = {
        "exact_r2_serialization_sha256": materialization[
            "exact_r2_serialization_sha256"
        ],
        "ordered_input_token_ids_sha256": materialization[
            "ordered_input_token_ids_sha256"
        ],
        "ordered_input_token_ids_digest_algorithm": materialization[
            "ordered_input_token_ids_digest_algorithm"
        ],
        "legacy_json_array_ordered_ids_sha256": materialization[
            "legacy_manual_probe_ordered_ids_sha256"
        ],
        "trigger_text_sha256": materialization["trigger_text_sha256"],
        "trigger_covering_token_ids_sha256": materialization[
            "trigger_covering_token_ids_sha256"
        ],
        "legacy_json_array_trigger_ids_sha256": materialization[
            "legacy_manual_probe_trigger_ids_sha256"
        ],
        "total_tokens": materialization["total_tokens"],
        "trigger_span_start": materialization["trigger_span_zero_based_exclusive"][
            "start"
        ],
        "trigger_span_end": materialization["trigger_span_zero_based_exclusive"]["end"],
        "trigger_span_width": materialization["trigger_span_width"],
        "trigger_text_occurrences": materialization["trigger_text_occurrences"],
        "trigger_token_sequence_occurrences": materialization[
            "trigger_token_sequence_occurrences"
        ],
        "boundary_overhang": materialization["boundary_overhang"],
    }
    for name, observed in flat_pairs.items():
        if observed != expected[name]:
            raise TriggerReceiptError(
                f"request2 materialization drift in {name}: {observed!r}"
            )


def build_receipt(config: Mapping[str, Any], snapshots: dict[str, dict]) -> dict:
    _authenticate_consumer_baseline()
    implementation_snapshot = _implementation_snapshot(config)
    snapshots["implementation"] = implementation_snapshot
    _root, assets = _asset_snapshots(config)
    snapshots.update({f"asset:{name}": value for name, value in assets.items()})

    tf32 = config["tf32_proxy_source"]
    tf32_snapshot = _read_snapshot(
        _repo_path(str(tf32["path"]), label="TF32 proxy receipt"),
        label="TF32 proxy receipt",
    )
    if tf32_snapshot["sha256"] != tf32["sha256"]:
        raise TriggerReceiptError("TF32 proxy receipt SHA-256 mismatch")
    tf32_value = _json_object(tf32_snapshot["raw"], label="TF32 proxy receipt")
    if (
        tf32_value.get("schema_version") != tf32["schema_version"]
        or tf32_value.get("claims", {}).get("proxy_signal_passed") is not True
        or tf32_value.get("claims", {}).get("numeric_equivalence") is not False
        or tf32_value.get("claims", {}).get("thresholds_formal") is not False
    ):
        raise TriggerReceiptError("TF32 proxy receipt claim drift")
    snapshots["tf32_proxy"] = tf32_snapshot

    binding = _binding_from_config(config)
    binding_sha = _sha256(_canonical_json_bytes(binding))
    if binding_sha != config["tokenizer"]["expected_tokenizer_binding_sha256"]:
        raise TriggerReceiptError(
            f"tokenizer binding SHA-256 mismatch: observed {binding_sha}"
        )
    materialization = _materialize_with_authenticated_assets(config, assets)
    _validate_expected_materialization(config, materialization)
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "status": "ready_diagnostic_only",
        "source_bindings": dict(config["source_bindings"]),
        "producer": {
            "producer_id": PRODUCER_ID,
            "config": {
                "path": DEFAULT_CONFIG,
                "sha256": snapshots["config"]["sha256"],
            },
            "config_schema": {
                "path": config["schemas"]["config"]["path"],
                "sha256": snapshots["config_schema"]["sha256"],
            },
            "receipt_schema": {
                "path": config["schemas"]["receipt"]["path"],
                "sha256": snapshots["receipt_schema"]["sha256"],
            },
            "implementation": {
                "path": config["implementation"]["path"],
                "sha256": implementation_snapshot["sha256"],
            },
            "canonical_json_policy": ("utf8_sort_keys_compact_no_normalization_lf_v1"),
            "receipt_sha256_sidecar_required": True,
            "receipt_self_sha256_in_body": False,
        },
        "tokenizer_binding_sha256": binding_sha,
        "tokenizer_binding": binding,
        "request2_materialization": materialization,
        "tf32_proxy_source": dict(tf32),
        "audit": dict(config["audit"]),
        "claims": dict(config["claim_scope"])
        | {
            "quality_validated": False,
            "physical_kv_claimed": False,
            "multistream_claimed": False,
        },
    }
    schema = snapshots["receipt_schema_value"]["value"]
    errors = sorted(
        Draft202012Validator(schema).iter_errors(receipt),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        raise TriggerReceiptError(
            f"receipt schema rejected output: {errors[0].message}"
        )
    return receipt


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where the platform exposes directory handles."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_physical_path(
    path: Path,
    *,
    label: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    try:
        return common._assert_physical(
            path,
            label=label,
            require_file=require_file,
            require_directory=require_directory,
        )
    except common.DiagnosticError as exc:
        raise TriggerReceiptError(str(exc)) from exc


def _cleanup_unpublished_directory(temp_dir: Path, temp_files: Sequence[Path]) -> None:
    """Remove only known files from this invocation's private temp directory."""

    if not os.path.lexists(temp_dir):
        return
    _assert_physical_path(
        temp_dir,
        label="unpublished request-local receipt directory",
        require_directory=True,
    )
    for path in temp_files:
        if os.path.lexists(path):
            _assert_physical_path(
                path,
                label="unpublished request-local receipt file",
                require_file=True,
            )
            path.unlink()
    temp_dir.rmdir()


def _publish_pair(
    config: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    receipt: dict,
    *,
    output: Path | None = None,
) -> tuple[Path, str, str]:
    destination = output or _repo_path(
        str(config["output"]["receipt_path"]), label="receipt output"
    )
    if not destination.is_absolute():
        destination = _REPO_ROOT / destination
    destination = Path(os.path.abspath(destination))
    if destination.name != "receipt.json":
        raise TriggerReceiptError("receipt output filename must be receipt.json")
    target_dir = destination.parent
    publish_root = target_dir.parent
    publish_root.mkdir(parents=True, exist_ok=True)
    _assert_physical_path(
        publish_root,
        label="request-local receipt publish root",
        require_directory=True,
    )
    _assert_physical_path(
        target_dir,
        label="request-local receipt output directory",
    )
    if os.path.lexists(target_dir):
        raise TriggerReceiptError(
            f"receipt output directory already exists: {target_dir}"
        )
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{target_dir.name}.tmp-", dir=publish_root)
    )
    temp_receipt = temp_dir / "receipt.json"
    temp_sidecar = temp_dir / "receipt.json.sha256"
    temp_files = (temp_receipt, temp_sidecar)
    try:
        _assert_physical_path(
            temp_dir,
            label="temporary request-local receipt directory",
            require_directory=True,
        )
        raw = _canonical_json_bytes(receipt, newline=True)
        receipt_sha = _sha256(raw)
        sidecar_raw = f"{receipt_sha}  receipt.json\n".encode("ascii")
        with temp_receipt.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        with temp_sidecar.open("xb") as handle:
            handle.write(sidecar_raw)
            handle.flush()
            os.fsync(handle.fileno())
        observed_receipt = _read_snapshot(
            temp_receipt, label="temporary request-local receipt"
        )
        observed_sidecar = _read_snapshot(
            temp_sidecar, label="temporary request-local receipt sidecar"
        )
        if observed_receipt["raw"] != raw or observed_sidecar["raw"] != sidecar_raw:
            raise TriggerReceiptError("temporary receipt bytes changed")
        for name, snapshot in snapshots.items():
            if name == "receipt_schema_value":
                continue
            _recheck_snapshot(snapshot, label=f"final source {name}")
        _authenticate_consumer_baseline()
        _fsync_directory(temp_dir)
        try:
            common._rename_noreplace(temp_dir, target_dir)
        except common.DiagnosticError as exc:
            raise TriggerReceiptError(str(exc)) from exc
        _fsync_directory(publish_root)
        final_receipt = _read_snapshot(
            destination, label="published request-local receipt"
        )
        final_sidecar = _read_snapshot(
            destination.with_name("receipt.json.sha256"),
            label="published request-local receipt sidecar",
        )
        if final_receipt["raw"] != raw or final_sidecar["raw"] != sidecar_raw:
            raise TriggerReceiptError("published receipt bytes changed")
        return destination, receipt_sha, _sha256(sidecar_raw)
    finally:
        _cleanup_unpublished_directory(temp_dir, temp_files)


def materialize(
    config_path: str | Path = DEFAULT_CONFIG, *, output: Path | None = None
) -> dict[str, Any]:
    config, snapshots = load_config(config_path)
    receipt = build_receipt(config, snapshots)
    destination, receipt_sha, sidecar_sha = _publish_pair(
        config, snapshots, receipt, output=output
    )
    return {
        "status": "ready_diagnostic_only",
        "receipt_path": str(destination),
        "receipt_sha256": receipt_sha,
        "receipt_sidecar_physical_sha256": sidecar_sha,
        "total_tokens": receipt["request2_materialization"]["total_tokens"],
        "trigger_span": receipt["request2_materialization"][
            "trigger_span_zero_based_exclusive"
        ],
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "formal": False,
        "training_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = materialize(args.config, output=args.output)
    except (TriggerReceiptError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": str(exc),
                    "formal": False,
                    "training_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
