"""Diagnostic-only Gemma 3 1B IT TF32 q_proj aLoRA Prefix-KV probe.

This module has its own versioned config, schemas, receipt, and CLI.  It does
not mutate the Qwen probes.  Contract and preflight modes do not import an ML
runtime; only ``--execute`` may load the locally exported model and touch CUDA.
No project dataset is opened by any mode.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from anchor_mvp.research import qwen_alora_prefix_kv_diagnostic as common


CONFIG_VERSION = "anchor.gemma3-alora-prefix-kv-diagnostic-config.v1"
RECEIPT_VERSION = "anchor.gemma3-alora-prefix-kv-diagnostic-receipt.v1"
PROFILE_ID = "gemma3_1b_it_tf32_qproj_alora_v1"
MODEL_ID = "google/gemma-3-1b-it-keras-v3-hf-export"
CONFIG_NAME = "gemma3_1b_it_alora_prefix_kv_tf32_v1.yaml"
CONFIG_SCHEMA_NAME = "gemma3_alora_prefix_kv_diagnostic_config.schema.json"
RECEIPT_SCHEMA_NAME = "gemma3_alora_prefix_kv_diagnostic_receipt.schema.json"
EXPECTED_RECEIPT = (
    "artifacts/diagnostics/gemma3_1b_it_alora_prefix_kv_tf32_v1/diagnostic_receipt.json"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ML_MODULES = ("torch", "transformers", "peft", "sentencepiece")

DiagnosticError = common.DiagnosticError


def _capture_imported_source() -> dict[str, Any]:
    """Bind the source bytes/stat observed by this imported module instance."""

    path = common._assert_physical(
        Path(__file__), label="imported Gemma diagnostic source", require_file=True
    )
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    stat_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stat_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stat_before != stat_after or len(raw) != before.st_size:
        raise RuntimeError("Gemma diagnostic source changed during module import")
    return {
        "path": path,
        "raw": raw,
        "sha256": common._sha256_bytes(raw),
        "stat": stat_before,
    }


_IMPORTED_SOURCE = _capture_imported_source()


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        observed = os.environ.get(name)
        if observed is not None:
            return observed
        if fallback is not None:
            return fallback
        raise DiagnosticError(f"required environment variable is unset: {name}")

    return _ENV_RE.sub(replace, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def config_sha256(config: Mapping[str, Any]) -> str:
    return common._sha256_bytes(common._canonical_json_bytes(_public_config(config)))


def implementation_sha256() -> str:
    return str(_IMPORTED_SOURCE["sha256"])


def _identity_file_snapshot(path: Path, *, label: str) -> dict[str, Any]:
    physical = common._assert_physical(path, label=label, require_file=True)
    before = physical.stat()
    raw = physical.read_bytes()
    after = physical.stat()
    stat_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stat_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stat_before != stat_after or len(raw) != before.st_size:
        raise DiagnosticError(f"{label} changed during identity snapshot")
    return {
        "path": physical,
        "raw": raw,
        "sha256": common._sha256_bytes(raw),
        "stat": stat_before,
    }


def _parse_config_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DiagnosticError(f"{label} must be UTF-8 YAML/JSON") from exc
    if not isinstance(parsed, dict):
        raise DiagnosticError(f"{label} must contain an object")
    expanded = _expand_env(parsed)
    if not isinstance(expanded, dict):
        raise DiagnosticError(f"{label} must expand to an object")
    return expanded


def _freeze_execution_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze code/config/schema identities before any model or GPU execution."""

    config_path = config.get("_config_path")
    if not isinstance(config_path, str):
        raise DiagnosticError("loaded config has no physical path")
    snapshots = {
        "implementation": _identity_file_snapshot(
            Path(str(_IMPORTED_SOURCE["path"])),
            label="Gemma diagnostic implementation",
        ),
        "config": _identity_file_snapshot(
            Path(config_path), label="Gemma diagnostic config"
        ),
        "config_schema": _identity_file_snapshot(
            _schema_path(CONFIG_SCHEMA_NAME), label="Gemma config schema"
        ),
        "receipt_schema": _identity_file_snapshot(
            _schema_path(RECEIPT_SCHEMA_NAME), label="Gemma receipt schema"
        ),
    }
    imported = snapshots["implementation"]
    if (
        imported["raw"] != _IMPORTED_SOURCE["raw"]
        or imported["sha256"] != _IMPORTED_SOURCE["sha256"]
        or imported["stat"] != _IMPORTED_SOURCE["stat"]
    ):
        raise DiagnosticError(
            "physical implementation no longer matches imported module source snapshot"
        )
    parsed = _parse_config_bytes(
        snapshots["config"]["raw"], label="frozen Gemma diagnostic config"
    )
    if parsed != _public_config(config):
        raise DiagnosticError(
            "physical config bytes do not match the parsed runtime config"
        )
    bindings = config["schemas"]
    if snapshots["config_schema"]["sha256"] != bindings["config"]["sha256"]:
        raise DiagnosticError("frozen config schema SHA-256 mismatch")
    if snapshots["receipt_schema"]["sha256"] != bindings["receipt"]["sha256"]:
        raise DiagnosticError("frozen receipt schema SHA-256 mismatch")
    return snapshots


def _reverify_execution_identity(identity: Mapping[str, Any]) -> None:
    """Fail closed if any execution-bound bytes or file identity drifted."""

    for name, frozen in identity.items():
        observed = _identity_file_snapshot(
            Path(str(frozen["path"])), label=f"reverified Gemma {name}"
        )
        if (
            observed["raw"] != frozen["raw"]
            or observed["sha256"] != frozen["sha256"]
            or observed["stat"] != frozen["stat"]
        ):
            raise DiagnosticError(f"execution-bound {name} identity drifted")


def _schema_path(name: str) -> Path:
    return _REPO_ROOT / "configs" / "research" / name


def _bound_schema(config: Mapping[str, Any], kind: str) -> tuple[dict[str, Any], str]:
    expected_name = CONFIG_SCHEMA_NAME if kind == "config" else RECEIPT_SCHEMA_NAME
    expected_version = CONFIG_VERSION if kind == "config" else RECEIPT_VERSION
    schemas = config.get("schemas")
    if not isinstance(schemas, Mapping) or not isinstance(schemas.get(kind), Mapping):
        raise DiagnosticError(f"config.schemas.{kind} must be an object")
    binding = schemas[kind]
    if binding.get("path") != f"configs/research/{expected_name}":
        raise DiagnosticError(f"config.schemas.{kind}.path is not canonical")
    if binding.get("version") != expected_version:
        raise DiagnosticError(f"config.schemas.{kind}.version is not canonical")
    expected_sha = binding.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        raise DiagnosticError(f"config.schemas.{kind}.sha256 is invalid")
    raw, observed_sha = common._read_bytes_snapshot(
        _schema_path(expected_name), label=f"Gemma {kind} schema"
    )
    if observed_sha != expected_sha:
        raise DiagnosticError(f"Gemma {kind} schema physical SHA-256 mismatch")
    schema = common._json_object(raw, label=f"Gemma {kind} schema")
    return schema, observed_sha


def validate_config(config: Mapping[str, Any]) -> dict[str, str]:
    config_schema, config_schema_sha = _bound_schema(config, "config")
    receipt_schema, receipt_schema_sha = _bound_schema(config, "receipt")
    common._validate_jsonschema(config, config_schema, label="Gemma diagnostic config")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(receipt_schema)
    except Exception as exc:
        raise DiagnosticError(f"Gemma receipt schema is invalid: {exc}") from exc
    if config.get("schema_version") != CONFIG_VERSION:
        raise DiagnosticError("Gemma config schema_version is not canonical")
    if config.get("profile_id") != PROFILE_ID:
        raise DiagnosticError("Gemma profile_id is not canonical")
    claims = config.get("claim_scope")
    if not isinstance(claims, Mapping):
        raise DiagnosticError("claim_scope must be an object")
    if claims.get("diagnostic_only") is not True:
        raise DiagnosticError("claim_scope.diagnostic_only must remain true")
    for key in ("formal", "training_authorized", "quality_validated"):
        if claims.get(key) is not False:
            raise DiagnosticError(f"claim_scope.{key} must remain false")
    return {"config": config_schema_sha, "receipt": receipt_schema_sha}


def load_config(path: str | Path) -> dict[str, Any]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    expected = _schema_path(CONFIG_NAME)
    if candidate != expected:
        raise DiagnosticError(f"config must be the canonical physical file: {expected}")
    raw, _ = common._read_bytes_snapshot(candidate, label="Gemma diagnostic config")
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DiagnosticError("Gemma config must be UTF-8 YAML/JSON") from exc
    if not isinstance(parsed, dict):
        raise DiagnosticError("Gemma config must contain an object")
    config = _expand_env(parsed)
    validate_config(config)
    config["_config_path"] = str(candidate)
    return config


def _model_root(config: Mapping[str, Any]) -> Path:
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("id") != MODEL_ID:
        raise DiagnosticError(f"model.id must remain {MODEL_ID}")
    value = model.get("local_path")
    if not isinstance(value, str) or not value:
        raise DiagnosticError("model.local_path must be non-empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return Path(os.path.abspath(path))


def _spm_path(config: Mapping[str, Any]) -> Path:
    value = config.get("tokenizer", {}).get("sentencepiece_path")
    if not isinstance(value, str) or not value:
        raise DiagnosticError("tokenizer.sentencepiece_path must be non-empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return Path(os.path.abspath(path))


def _receipt_path(config: Mapping[str, Any]) -> Path:
    relative = config.get("output", {}).get("receipt_path")
    if relative != EXPECTED_RECEIPT:
        raise DiagnosticError("output.receipt_path is not canonical")
    destination = Path(os.path.abspath(_REPO_ROOT / str(relative)))
    allowed = Path(os.path.abspath(_REPO_ROOT / "artifacts" / "diagnostics"))
    if destination == allowed or not destination.is_relative_to(allowed):
        raise DiagnosticError("receipt path escapes artifacts/diagnostics")
    return destination


def _asset_bindings(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    assets = config.get("model", {}).get("assets")
    if not isinstance(assets, list) or len(assets) != 3:
        raise DiagnosticError(
            "model.assets must contain config, model, and export manifest"
        )
    expected = ("config.json", "model.safetensors", "EXPORT_MANIFEST.json")
    result: list[Mapping[str, Any]] = []
    for index, (item, name) in enumerate(zip(assets, expected, strict=True)):
        if not isinstance(item, Mapping) or item.get("path") != name:
            raise DiagnosticError(f"model.assets[{index}] must bind {name}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise DiagnosticError(f"model.assets[{index}].sha256 is invalid")
        result.append(item)
    return tuple(result)


def build_contract_report(config: Mapping[str, Any]) -> dict[str, Any]:
    schemas = validate_config(_public_config(config))
    return {
        "schema_version": CONFIG_VERSION,
        "profile_id": PROFILE_ID,
        "mode": "contract_only",
        "status": "contract_valid",
        "config_sha256": config_sha256(config),
        "schema_sha256": schemas,
        "implementation_sha256": implementation_sha256(),
        "model_loaded": False,
        "gpu_touched": False,
        "provider_requests": 0,
        "network_requests": 0,
        "dataset_inputs": 0,
        "formal": False,
        "training_authorized": False,
    }


def build_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = build_contract_report(config)
    model_root = _model_root(config)
    spm_path = _spm_path(config)
    destination = _receipt_path(config)
    observed: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, bytes] = {}
    all_match = True
    for binding in _asset_bindings(config):
        name = str(binding["path"])
        path = model_root / name
        try:
            if name == "model.safetensors":
                digest = common._stream_sha256_snapshot(path, label=name)
                raw = b""
            else:
                raw, digest = common._read_bytes_snapshot(path, label=name)
            snapshots[name] = raw
            physical = True
        except (DiagnosticError, OSError):
            digest, physical = None, False
        matched = physical and digest == binding["sha256"]
        observed[name] = {
            "path": str(path),
            "expected_sha256": binding["sha256"],
            "observed_sha256": digest,
            "matched": matched,
        }
        all_match = all_match and matched
    try:
        _, spm_sha = common._read_bytes_snapshot(spm_path, label="SentencePiece model")
        spm_match = spm_sha == config["tokenizer"]["sentencepiece_sha256"]
    except (DiagnosticError, OSError):
        spm_sha, spm_match = None, False
    try:
        architecture = common._json_object(
            snapshots["config.json"], label="config.json"
        )
    except (DiagnosticError, KeyError):
        architecture = {}
    architecture_match = (
        architecture.get("architectures") == ["Gemma3ForCausalLM"]
        and architecture.get("model_type") == "gemma3_text"
        and architecture.get("num_hidden_layers") == 26
        and architecture.get("hidden_size") == 1152
        and architecture.get("num_attention_heads") == 4
        and architecture.get("num_key_value_heads") == 1
        and architecture.get("sliding_window") == 512
        and architecture.get("_sliding_window_pattern") == 6
    )
    gates = {
        "contract_valid": True,
        "model_root_physical": model_root.is_dir(),
        "model_artifact_hashes_match": all_match,
        "sentencepiece_hash_matches": spm_match,
        "gemma3_text_architecture_matches": architecture_match,
        "receipt_absent": not common._lexists(destination),
        "network_disabled": config["audit"]["network_requests"] == 0,
        "dataset_reads_zero": config["audit"]["dataset_reads"] == 0,
    }
    return {
        **contract,
        "mode": "preflight",
        "status": "ready" if all(gates.values()) else "blocked",
        "model_root": str(model_root),
        "sentencepiece_path": str(spm_path),
        "receipt_path": str(destination),
        "artifact_identity": observed,
        "sentencepiece_observed_sha256": spm_sha,
        "gates": gates,
        "ready": all(gates.values()),
    }


def _serialize_request2(protocol: Mapping[str, Any]) -> str:
    request = protocol.get("request2")
    if not isinstance(request, Mapping):
        raise DiagnosticError("protocol.request2 must be an object")
    fields = {}
    for key in (
        "turn_start",
        "turn_end",
        "user_role",
        "model_role",
        "header",
        "planner_scaffold_summary",
        "task_summary",
        "invocation_text",
    ):
        value = request.get(key)
        if not isinstance(value, str) or not value:
            raise DiagnosticError(f"protocol.request2.{key} must be non-empty")
        fields[key] = value
    return (
        f"{fields['turn_start']}{fields['user_role']}\n"
        f"{fields['header']}\n"
        f"Planner scaffold: {fields['planner_scaffold_summary']}\n"
        f"Activation: {fields['invocation_text']}\n"
        f"Task: {fields['task_summary']}{fields['turn_end']}\n"
        f"{fields['turn_start']}{fields['model_role']}\n"
    )


def materialize_request2(processor: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Tokenize the complete plain-text request once and locate trigger coverage."""

    serialized = _serialize_request2(config["protocol"])
    trigger = str(config["protocol"]["request2"]["invocation_text"])
    if serialized.count(trigger) != 1:
        raise DiagnosticError("trigger text must occur exactly once")
    proto = processor.encode_as_immutable_proto(serialized)
    pieces = tuple(proto.pieces)
    if not pieces:
        raise DiagnosticError("SentencePiece returned no pieces")
    ids_without_bos = [int(piece.id) for piece in pieces]
    bos_id = int(processor.bos_id())
    if bos_id != int(config["tokenizer"]["explicit_bos_id"]):
        raise DiagnosticError("SentencePiece BOS ID does not match config")
    input_ids = [bos_id, *ids_without_bos]
    char_start = serialized.index(trigger)
    char_end = char_start + len(trigger)
    overlapping = [
        index
        for index, piece in enumerate(pieces)
        if int(piece.begin) < char_end and int(piece.end) > char_start
    ]
    if not overlapping or overlapping != list(
        range(overlapping[0], overlapping[-1] + 1)
    ):
        raise DiagnosticError("trigger piece coverage is empty or non-contiguous")
    piece_start, piece_end = overlapping[0], overlapping[-1] + 1
    cover_start = int(pieces[piece_start].begin)
    cover_end = int(pieces[piece_end - 1].end)
    if cover_start > char_start or cover_end < char_end:
        raise DiagnosticError("trigger pieces do not cover the complete trigger text")
    token_start, token_end = piece_start + 1, piece_end + 1
    invocation_ids = input_ids[token_start:token_end]
    if not invocation_ids or token_start <= 0 or token_end >= len(input_ids):
        raise DiagnosticError("request must have prefix, trigger, and suffix")
    if common._count_subsequence(input_ids, invocation_ids) != 1:
        raise DiagnosticError("trigger token subsequence is not request-local unique")
    left = serialized[cover_start:char_start]
    right = serialized[char_end:cover_end]
    return {
        "serialized_text": serialized,
        "input_ids": input_ids,
        "invocation_ids": invocation_ids,
        "token_start": token_start,
        "token_end": token_end,
        "character_start": char_start,
        "character_end": char_end,
        "cover_start": cover_start,
        "cover_end": cover_end,
        "left_overhang_codepoints": len(left),
        "right_overhang_codepoints": len(right),
        "left_overhang_utf8_bytes": len(left.encode("utf-8")),
        "right_overhang_utf8_bytes": len(right.encode("utf-8")),
        "left_overhang_sha256": common._sha256_bytes(left.encode("utf-8")),
        "right_overhang_sha256": common._sha256_bytes(right.encode("utf-8")),
        "total_tokens": len(input_ids),
        "prefix_tokens": token_start,
        "invocation_tokens": token_end - token_start,
        "post_invocation_tokens": len(input_ids) - token_end,
        "ordered_token_ids_sha256": common._token_ids_sha256(input_ids),
        "prefix_token_ids_sha256": common._token_ids_sha256(input_ids[:token_start]),
        "invocation_token_ids_sha256": common._token_ids_sha256(invocation_ids),
        "serialized_text_sha256": common._sha256_bytes(serialized.encode("utf-8")),
    }


def evaluate_gates(
    metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    paired = float(metrics["paired_differential_max_abs"])
    effect = float(metrics["adapter_effect_max_abs"])
    missing = float(metrics["missing_trigger_effect_max_abs"])
    peak = float(metrics["peak_allocated_mib"])
    relative = paired / effect if effect > 0 else math.inf
    finite_scalars = all(
        math.isfinite(value) for value in (paired, effect, missing, peak, relative)
    )
    gates = {
        "paired_differential": finite_scalars
        and paired <= float(config["gates"]["paired_differential_max_abs"]),
        "paired_relative": finite_scalars
        and relative <= float(config["gates"]["paired_to_adapter_effect_ratio_max"]),
        "adapter_effect": finite_scalars
        and effect >= float(config["gates"]["adapter_effect_min_abs"]),
        "missing_trigger": missing == 0.0,
        "prefix_kv": metrics.get("prefix_kv_bit_equal") is True,
        "finite": metrics.get("finite") is True,
        "shapes": metrics.get("shape_equal") is True,
        "argmax": metrics.get("argmax_equal") is True,
        "greedy": metrics.get("greedy_equal") is True,
        "memory": finite_scalars
        and peak <= float(config["gates"]["peak_allocated_mib_max"]),
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _initialize_alora(model: Any, torch: Any, seed: int) -> int:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    variants: set[str] = set()
    initialized = 0
    with torch.no_grad():
        for module in model.modules():
            lora_variant = getattr(module, "lora_variant", None)
            lora_b = getattr(module, "lora_B", None)
            if not isinstance(lora_variant, Mapping) or "default" not in lora_variant:
                continue
            variants.add(type(lora_variant["default"]).__name__)
            if lora_b is None or "default" not in lora_b:
                raise DiagnosticError("aLoRA layer has no default B matrix")
            weight = lora_b["default"].weight
            values = (
                torch.randn(
                    tuple(int(x) for x in weight.shape),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                * 0.01
            )
            values[values == 0] = torch.finfo(torch.float32).eps
            weight.copy_(values.to(device=weight.device, dtype=weight.dtype))
            initialized += 1
    if variants != {"ALoraLinearVariant"}:
        raise DiagnosticError("PEFT did not attach ALoraLinearVariant")
    if initialized != 26:
        raise DiagnosticError(
            f"expected 26 q_proj aLoRA matrices, observed {initialized}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return initialized


def _copy_authenticated_file(
    source: Path, destination: Path, *, expected_sha256: str, label: str
) -> None:
    """Create a private physical copy from one stable source-handle snapshot."""

    source = common._assert_physical(source, label=label, require_file=True)
    before = source.stat()
    digest = hashlib.sha256()
    with (
        source.open("rb", buffering=0) as reader,
        destination.open("xb", buffering=0) as writer,
    ):
        opened = os.fstat(reader.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise DiagnosticError(f"{label} changed before snapshot copy")
        while True:
            chunk = reader.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DiagnosticError(f"{label} changed during snapshot copy")
    if digest.hexdigest() != expected_sha256:
        raise DiagnosticError(f"{label} snapshot SHA-256 mismatch")
    observed = (
        common._stream_sha256_snapshot(destination, label=f"private {label}")
        if destination.name == "model.safetensors"
        else common._read_bytes_snapshot(destination, label=f"private {label}")[1]
    )
    if observed != expected_sha256:
        raise DiagnosticError(f"private {label} verification failed")


def _create_model_snapshot(config: Mapping[str, Any]) -> tuple[Path, tuple[Path, ...]]:
    """Copy authenticated model inputs into an exclusive private load directory."""

    root = _REPO_ROOT / "artifacts" / "diagnostics"
    common._assert_physical(root, label="Gemma model snapshot root")
    root.mkdir(parents=True, exist_ok=True)
    common._assert_physical(
        root, label="Gemma model snapshot root", require_directory=True
    )
    snapshot = root / f".gemma3-model-load-snapshot-{uuid.uuid4().hex}"
    snapshot.mkdir()
    common._assert_physical(
        snapshot, label="Gemma model load snapshot", require_directory=True
    )
    files: list[Path] = []
    try:
        for binding in _asset_bindings(config):
            name = str(binding["path"])
            destination = snapshot / name
            files.append(destination)
            _copy_authenticated_file(
                _model_root(config) / name,
                destination,
                expected_sha256=str(binding["sha256"]),
                label=f"model artifact {name}",
            )
        _fsync_directory(snapshot)
        return snapshot, tuple(files)
    except Exception:
        _cleanup_unpublished_directory(snapshot, tuple(files))
        raise


def _authenticate_after_load(config: Mapping[str, Any], snapshot: Path) -> None:
    for binding in _asset_bindings(config):
        name = str(binding["path"])
        path = snapshot / name
        digest = (
            common._stream_sha256_snapshot(path, label=f"post-load snapshot {name}")
            if name == "model.safetensors"
            else common._read_bytes_snapshot(path, label=f"post-load snapshot {name}")[
                1
            ]
        )
        if digest != binding["sha256"]:
            raise DiagnosticError(
                f"private model snapshot changed while loading: {name}"
            )


def _run_route(
    model: Any,
    ids: Any,
    torch: Any,
    *,
    greedy_tokens: int,
    adapter_enabled: bool,
    activation_continues: bool,
    past_key_values: Any | None = None,
) -> tuple[Any, tuple[int, ...], Any]:
    return common._greedy_route(
        model,
        ids,
        torch,
        greedy_tokens=greedy_tokens,
        adapter_enabled=adapter_enabled,
        activation_continues=activation_continues,
        past_key_values=past_key_values,
    )


def _execute_gpu(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    workspace = common._prepare_deterministic_cuda_environment()
    import peft
    import sentencepiece as spm
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise DiagnosticError("--execute requires CUDA")
    seed = int(config["numerics"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise DiagnosticError("TF32 controls did not remain enabled")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    model_snapshot: Path | None = None
    model_snapshot_files: tuple[Path, ...] = ()
    try:
        spm_raw, spm_sha = common._read_bytes_snapshot(
            _spm_path(config), label="authenticated SentencePiece model proto"
        )
        if (
            spm_sha != config["tokenizer"]["sentencepiece_sha256"]
            or spm_sha != preflight["sentencepiece_observed_sha256"]
        ):
            raise DiagnosticError("SentencePiece model proto identity mismatch")
        processor = spm.SentencePieceProcessor(model_proto=spm_raw)
        if int(processor.vocab_size()) != int(config["tokenizer"]["vocab_size"]):
            raise DiagnosticError("SentencePiece vocabulary size mismatch")
        materialized = materialize_request2(processor, config)
        values = list(materialized["input_ids"])
        trigger_ids = list(materialized["invocation_ids"])
        token_start = int(materialized["token_start"])
        full_ids = torch.tensor([values], dtype=torch.long, device="cuda")
        prefix_ids = full_ids[:, :token_start]
        tail_ids = full_ids[:, token_start:]
        missing_text = str(materialized["serialized_text"]).replace(
            str(config["protocol"]["request2"]["invocation_text"]),
            str(config["protocol"]["request2"]["missing_invocation_text"]),
        )
        missing_values = [
            int(processor.bos_id()),
            *map(int, processor.encode(missing_text, out_type=int)),
        ]
        if common._count_subsequence(missing_values, trigger_ids) != 0:
            raise DiagnosticError("missing-trigger control contains trigger IDs")
        missing_ids = torch.tensor([missing_values], dtype=torch.long, device="cuda")

        model_snapshot, model_snapshot_files = _create_model_snapshot(config)
        base = AutoModelForCausalLM.from_pretrained(
            model_snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda")
        model = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=["q_proj"],
                bias="none",
                task_type="CAUSAL_LM",
                alora_invocation_tokens=trigger_ids,
            ),
        )
        _initialize_alora(model, torch, seed)
        model.eval()
        model.config.use_cache = True
        _authenticate_after_load(config, model_snapshot)
        greedy_tokens = int(config["numerics"]["greedy_tokens"])

        base_full, base_full_greedy, cache = _run_route(
            model,
            full_ids,
            torch,
            greedy_tokens=greedy_tokens,
            adapter_enabled=False,
            activation_continues=False,
        )
        del cache
        alora_full, alora_full_greedy, cache = _run_route(
            model,
            full_ids,
            torch,
            greedy_tokens=greedy_tokens,
            adapter_enabled=True,
            activation_continues=True,
        )
        del cache
        prefix_off = common._prefill(model, prefix_ids, torch, adapter_enabled=False)
        prefix_active = common._prefill(model, prefix_ids, torch, adapter_enabled=True)
        prefix_equal, cache_layers = common._cache_bit_equal(
            prefix_off, prefix_active, torch
        )
        prefix_cache_tokens = common._cache_sequence_length(prefix_off)
        base_split, base_split_greedy, cache = _run_route(
            model,
            tail_ids,
            torch,
            greedy_tokens=greedy_tokens,
            adapter_enabled=False,
            activation_continues=False,
            past_key_values=prefix_off,
        )
        del cache, prefix_off
        alora_split, alora_split_greedy, cache = _run_route(
            model,
            tail_ids,
            torch,
            greedy_tokens=greedy_tokens,
            adapter_enabled=True,
            activation_continues=True,
            past_key_values=prefix_active,
        )
        del cache, prefix_active
        missing_off, missing_off_greedy, cache = _run_route(
            model,
            missing_ids,
            torch,
            greedy_tokens=greedy_tokens,
            adapter_enabled=False,
            activation_continues=False,
        )
        del cache
        missing_active, missing_active_greedy, cache = _run_route(
            model,
            missing_ids,
            torch,
            greedy_tokens=greedy_tokens,
            adapter_enabled=True,
            activation_continues=False,
        )
        del cache

        tensors = (
            base_full,
            base_split,
            alora_full,
            alora_split,
            missing_off,
            missing_active,
        )
        base_delta = base_split - base_full
        alora_delta = alora_split - alora_full
        paired = float(torch.max(torch.abs(alora_delta - base_delta)).item())
        effect = float(torch.max(torch.abs(alora_full - base_full)).item())
        missing = float(torch.max(torch.abs(missing_active - missing_off)).item())
        missing_bit_equal = bool(torch.equal(missing_active, missing_off))
        base_argmax_equal = bool(
            torch.equal(torch.argmax(base_full, -1), torch.argmax(base_split, -1))
        )
        alora_argmax_equal = bool(
            torch.equal(torch.argmax(alora_full, -1), torch.argmax(alora_split, -1))
        )
        greedy_equal = (
            base_full_greedy == base_split_greedy
            and alora_full_greedy == alora_split_greedy
            and missing_off_greedy == missing_active_greedy
        )
        peak_allocated = float(torch.cuda.max_memory_allocated() / 2**20)
        peak_reserved = float(torch.cuda.max_memory_reserved() / 2**20)
        metrics_for_gates = {
            "paired_differential_max_abs": paired,
            "adapter_effect_max_abs": effect,
            "missing_trigger_effect_max_abs": missing,
            "prefix_kv_bit_equal": prefix_equal,
            "finite": all(
                bool(torch.isfinite(value).all().item()) for value in tensors
            ),
            "shape_equal": len({tuple(value.shape) for value in tensors}) == 1,
            "argmax_equal": base_argmax_equal and alora_argmax_equal,
            "greedy_equal": greedy_equal,
            "peak_allocated_mib": peak_allocated,
        }
        gates = evaluate_gates(metrics_for_gates, config)
        gates["missing_trigger"] = gates["missing_trigger"] and missing_bit_equal
        gates["all_passed"] = all(
            value for key, value in gates.items() if key != "all_passed"
        )
        if not gates["all_passed"]:
            failed = [key for key, passed in gates.items() if not passed]
            raise DiagnosticError(
                "Gemma runtime gates failed: "
                + ", ".join(failed)
                + f"; paired={paired:.17g}; effect={effect:.17g}; ratio={paired / effect if effect else math.inf:.17g}"
            )
        _reverify_execution_identity(execution_identity)
        boundary = {
            key: materialized[key]
            for key in (
                "character_start",
                "character_end",
                "cover_start",
                "cover_end",
                "left_overhang_codepoints",
                "right_overhang_codepoints",
                "left_overhang_utf8_bytes",
                "right_overhang_utf8_bytes",
                "left_overhang_sha256",
                "right_overhang_sha256",
            )
        }
        receipt = {
            "schema_version": RECEIPT_VERSION,
            "status": "passed",
            "config_sha256": config_sha256(config),
            "config_file_sha256": execution_identity["config"]["sha256"],
            "implementation_sha256": execution_identity["implementation"]["sha256"],
            "schema_sha256": {
                "config": execution_identity["config_schema"]["sha256"],
                "receipt": execution_identity["receipt_schema"]["sha256"],
            },
            "model_identity": {
                "id": MODEL_ID,
                "config_json_sha256": preflight["artifact_identity"]["config.json"][
                    "observed_sha256"
                ],
                "model_safetensors_sha256": preflight["artifact_identity"][
                    "model.safetensors"
                ]["observed_sha256"],
                "export_manifest_sha256": preflight["artifact_identity"][
                    "EXPORT_MANIFEST.json"
                ]["observed_sha256"],
                "exporter": "keras_hub.models.Gemma3CausalLM.export_to_transformers",
                "sentencepiece_sha256": preflight["sentencepiece_observed_sha256"],
                "parameters": 999885952,
                "layers": 26,
                "load_strategy": "authenticated_private_physical_copy",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
                "peft": str(peft.__version__),
                "sentencepiece": str(spm.__version__),
                "cuda": torch.version.cuda,
                "gpu": str(torch.cuda.get_device_name(0)),
                "dtype": "float32",
                "attention_implementation": "eager",
                "tf32": True,
                "matmul_precision": "high",
                "cublas_workspace_config": workspace,
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
            },
            "tokenization": {
                "serialization_version": config["protocol"]["request2"][
                    "serialization_version"
                ],
                "request2_utf8_sha256": materialized["serialized_text_sha256"],
                "ordered_token_ids_sha256": materialized["ordered_token_ids_sha256"],
                "prefix_token_ids_sha256": materialized["prefix_token_ids_sha256"],
                "invocation_token_ids_sha256": materialized[
                    "invocation_token_ids_sha256"
                ],
                "trigger_text_sha256": common._sha256_bytes(
                    config["protocol"]["request2"]["invocation_text"].encode("utf-8")
                ),
                "full_tokens": materialized["total_tokens"],
                "prefix_tokens": materialized["prefix_tokens"],
                "invocation_tokens": materialized["invocation_tokens"],
                "post_invocation_tokens": materialized["post_invocation_tokens"],
                "trigger_span_start": materialized["token_start"],
                "trigger_span_end": materialized["token_end"],
                "trigger_text_occurrences": 1,
                "boundary": boundary,
                "sentencepiece_bos_id": int(processor.bos_id()),
                "sentencepiece_eos_id": int(processor.eos_id()),
                "hf_export_config_bos_id": 1,
                "hf_export_config_eos_id": 2,
                "bos_source": "sentencepiece_model_explicit_prepend",
                "token_ids_digest_algorithm": "sha256_signed_int64_big_endian_concatenation_v1",
                "trigger_span_semantics": "zero_based_start_inclusive_end_exclusive",
                "tokenizer_model_proto_authenticated": True,
                "complete_request_tokenized_once": True,
                "isolated_trigger_encoding_authoritative": False,
            },
            "routes": {
                "executed": [
                    "base_full",
                    "base_split",
                    "alora_full",
                    "alora_split",
                    "missing_trigger_active",
                    "missing_trigger_off",
                    "prefix_active",
                    "prefix_off",
                ],
                "adapter_kind": "alora",
                "target_modules": ["q_proj"],
                "rank": 4,
                "alpha": 8,
                "training_steps": 0,
                "adapter_saved": False,
                "planner_request_executed": False,
                "planner_private_kv_reused": False,
                "reuse_scope": "exact_plaintext_r2_pre_invocation_base_prefix_only",
                "cache_layers": cache_layers,
                "prefix_cache_tokens": prefix_cache_tokens,
                "full_cache_tokens": materialized["total_tokens"],
                "prefix_kv_bit_equal": prefix_equal,
            },
            "metrics": {
                "base_full_split_max_abs": float(
                    torch.max(torch.abs(base_delta)).item()
                ),
                "alora_full_split_max_abs": float(
                    torch.max(torch.abs(alora_delta)).item()
                ),
                "paired_differential_max_abs": paired,
                "paired_to_adapter_effect_ratio": paired / effect,
                "adapter_effect_max_abs": effect,
                "missing_trigger_effect_max_abs": missing,
                "base_argmax_equal": base_argmax_equal,
                "alora_argmax_equal": alora_argmax_equal,
                "greedy_tokens_equal": greedy_equal,
                "all_finite": metrics_for_gates["finite"],
                "shapes_equal": metrics_for_gates["shape_equal"],
            },
            "gates": gates,
            "audit": {
                "provider_requests": 0,
                "network_requests": 0,
                "dataset_reads": 0,
                "model_loads": 1,
            },
            "claims": {
                "diagnostic_only": True,
                "formal": False,
                "training_authorized": False,
                "quality_validated": False,
                "numeric_equivalence": False,
                "proxy_signal_passed": True,
                "thresholds_formal": False,
                "bf16_q4_bit_exact": False,
                "multi_stream": False,
                "zero_copy": False,
                "full_generation_kv_shared": False,
            },
        }
        return receipt
    finally:
        model = None
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()
        if model_snapshot is not None:
            _cleanup_unpublished_directory(model_snapshot, model_snapshot_files)


def _validate_receipt(
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    execution_identity: Mapping[str, Any] | None = None,
) -> None:
    identity = execution_identity or _freeze_execution_identity(config)
    _reverify_execution_identity(identity)
    schema, receipt_schema_sha = _bound_schema(config, "receipt")
    common._validate_jsonschema(receipt, schema, label="Gemma diagnostic receipt")
    _, config_schema_sha = _bound_schema(config, "config")
    if receipt.get("config_sha256") != config_sha256(config):
        raise DiagnosticError("receipt config canonical SHA-256 mismatch")
    if receipt.get("config_file_sha256") != identity["config"]["sha256"]:
        raise DiagnosticError("receipt config physical SHA-256 mismatch")
    if receipt.get("implementation_sha256") != identity["implementation"]["sha256"]:
        raise DiagnosticError("receipt implementation physical SHA-256 mismatch")
    if receipt.get("schema_sha256") != {
        "config": identity["config_schema"]["sha256"],
        "receipt": identity["receipt_schema"]["sha256"],
    }:
        raise DiagnosticError("receipt schema physical SHA-256 mismatch")
    if (
        config_schema_sha != identity["config_schema"]["sha256"]
        or receipt_schema_sha != identity["receipt_schema"]["sha256"]
    ):
        raise DiagnosticError("receipt validation schema snapshot drifted")
    if (
        receipt.get("status") != "passed"
        or receipt.get("gates", {}).get("all_passed") is not True
    ):
        raise DiagnosticError("only an all-passed Gemma receipt may be published")
    for key in ("provider_requests", "network_requests", "dataset_reads"):
        if receipt.get("audit", {}).get(key) != 0:
            raise DiagnosticError(f"receipt audit.{key} must remain zero")
    claims = receipt.get("claims", {})
    for key in (
        "formal",
        "training_authorized",
        "quality_validated",
        "numeric_equivalence",
        "thresholds_formal",
    ):
        if claims.get(key) is not False:
            raise DiagnosticError(f"receipt claims.{key} must remain false")

    tokenization = receipt.get("tokenization")
    routes = receipt.get("routes")
    if not isinstance(tokenization, Mapping) or not isinstance(routes, Mapping):
        raise DiagnosticError("receipt tokenization/routes must be objects")
    full_tokens = int(tokenization["full_tokens"])
    prefix_tokens = int(tokenization["prefix_tokens"])
    invocation_tokens = int(tokenization["invocation_tokens"])
    post_tokens = int(tokenization["post_invocation_tokens"])
    span_start = int(tokenization["trigger_span_start"])
    span_end = int(tokenization["trigger_span_end"])
    if full_tokens != prefix_tokens + invocation_tokens + post_tokens:
        raise DiagnosticError("receipt token counts do not partition full_tokens")
    if span_start != prefix_tokens or span_end != span_start + invocation_tokens:
        raise DiagnosticError("receipt trigger span/count cross-binding mismatch")
    if not 0 < span_start < span_end < full_tokens:
        raise DiagnosticError(
            "receipt trigger span is not bounded zero-based [start,end)"
        )
    if (
        routes.get("prefix_cache_tokens") != prefix_tokens
        or routes.get("full_cache_tokens") != full_tokens
    ):
        raise DiagnosticError("receipt cache/token count cross-binding mismatch")

    spm_raw, spm_sha = common._read_bytes_snapshot(
        _spm_path(config), label="receipt validation SentencePiece model proto"
    )
    if spm_sha != config["tokenizer"]["sentencepiece_sha256"]:
        raise DiagnosticError("receipt validation SentencePiece SHA-256 mismatch")
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise DiagnosticError(
            "sentencepiece is required to validate a receipt"
        ) from exc
    processor = spm.SentencePieceProcessor(model_proto=spm_raw)
    expected = materialize_request2(processor, config)
    expected_fields = {
        "request2_utf8_sha256": expected["serialized_text_sha256"],
        "ordered_token_ids_sha256": expected["ordered_token_ids_sha256"],
        "prefix_token_ids_sha256": expected["prefix_token_ids_sha256"],
        "invocation_token_ids_sha256": expected["invocation_token_ids_sha256"],
        "trigger_text_sha256": common._sha256_bytes(
            config["protocol"]["request2"]["invocation_text"].encode("utf-8")
        ),
        "full_tokens": expected["total_tokens"],
        "prefix_tokens": expected["prefix_tokens"],
        "invocation_tokens": expected["invocation_tokens"],
        "post_invocation_tokens": expected["post_invocation_tokens"],
        "trigger_span_start": expected["token_start"],
        "trigger_span_end": expected["token_end"],
    }
    for key, expected_value in expected_fields.items():
        if tokenization.get(key) != expected_value:
            raise DiagnosticError(f"receipt tokenization.{key} mismatch")
    expected_boundary = {
        key: expected[key]
        for key in (
            "character_start",
            "character_end",
            "cover_start",
            "cover_end",
            "left_overhang_codepoints",
            "right_overhang_codepoints",
            "left_overhang_utf8_bytes",
            "right_overhang_utf8_bytes",
            "left_overhang_sha256",
            "right_overhang_sha256",
        )
    }
    if tokenization.get("boundary") != expected_boundary:
        raise DiagnosticError("receipt token boundary/overhang mismatch")


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry on platforms that expose directory handles."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_unpublished_directory(
    temporary_directory: Path, temporary_files: Sequence[Path]
) -> None:
    """Remove only the known private files from an unpublished temp directory."""

    if not common._lexists(temporary_directory):
        return
    common._assert_physical(
        temporary_directory,
        label="unpublished Gemma receipt directory",
        require_directory=True,
    )
    for path in temporary_files:
        if common._lexists(path):
            common._assert_physical(
                path,
                label="unpublished Gemma receipt file",
                require_file=True,
            )
            path.unlink()
    temporary_directory.rmdir()


def publish_receipt(
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    execution_identity: Mapping[str, Any] | None = None,
) -> Path:
    identity = execution_identity or _freeze_execution_identity(config)
    _reverify_execution_identity(identity)
    _validate_receipt(config, receipt, identity)
    destination = _receipt_path(config)
    version_directory = destination.parent
    publish_root = version_directory.parent
    common._assert_physical(
        publish_root,
        label="Gemma diagnostic publish root before creation",
    )
    publish_root.mkdir(parents=True, exist_ok=True)
    common._assert_physical(
        publish_root,
        label="Gemma diagnostic publish root",
        require_directory=True,
    )
    common._assert_physical(
        version_directory,
        label="Gemma diagnostic version directory",
    )
    if common._lexists(version_directory):
        raise FileExistsError(
            f"Gemma diagnostic version directory exists: {version_directory}"
        )
    raw = common._canonical_json_bytes(receipt)
    digest = common._sha256_bytes(raw)
    sidecar_raw = f"{digest}  {destination.name}\n".encode("utf-8")
    temporary_directory = publish_root / (
        f".{version_directory.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary_receipt = temporary_directory / destination.name
    temporary_sidecar = temporary_directory / f"{destination.name}.sha256"
    temporary_files = (temporary_receipt, temporary_sidecar)
    try:
        temporary_directory.mkdir()
        common._assert_physical(
            temporary_directory,
            label="temporary Gemma receipt directory",
            require_directory=True,
        )
        with temporary_receipt.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary_sidecar.open("xb") as handle:
            handle.write(sidecar_raw)
            handle.flush()
            os.fsync(handle.fileno())
        observed_receipt, _ = common._read_bytes_snapshot(
            temporary_receipt,
            label="temporary Gemma diagnostic receipt",
        )
        observed_sidecar, _ = common._read_bytes_snapshot(
            temporary_sidecar,
            label="temporary Gemma diagnostic receipt sidecar",
        )
        if observed_receipt != raw or observed_sidecar != sidecar_raw:
            raise DiagnosticError("temporary Gemma receipt bytes changed")
        _fsync_directory(temporary_directory)
        _reverify_execution_identity(identity)
        common._rename_noreplace(temporary_directory, version_directory)
        _fsync_directory(publish_root)
        final_receipt, _ = common._read_bytes_snapshot(
            destination,
            label="published Gemma diagnostic receipt",
        )
        final_sidecar, _ = common._read_bytes_snapshot(
            destination.with_suffix(".json.sha256"),
            label="published Gemma diagnostic receipt sidecar",
        )
        if final_receipt != raw or final_sidecar != sidecar_raw:
            raise DiagnosticError("published Gemma receipt bytes changed")
    finally:
        _cleanup_unpublished_directory(temporary_directory, temporary_files)
    return destination


def execute_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    execution_identity = _freeze_execution_identity(config)
    preflight = build_preflight(config)
    if not preflight["ready"]:
        failed = [key for key, passed in preflight["gates"].items() if not passed]
        raise DiagnosticError("Gemma preflight failed: " + ", ".join(failed))
    _reverify_execution_identity(execution_identity)
    receipt = _execute_gpu(config, preflight, execution_identity)
    _reverify_execution_identity(execution_identity)
    _validate_receipt(config, receipt, execution_identity)
    publish_receipt(config, receipt, execution_identity)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemma 3 1B IT TF32 q_proj aLoRA Prefix-KV diagnostic"
    )
    parser.add_argument("--config", default=str(_schema_path(CONFIG_NAME)))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.execute:
            receipt = execute_diagnostic(config)
            report = {
                "mode": "execute",
                "status": "passed",
                "receipt_path": str(_receipt_path(config)),
                "receipt_sha256": common._sha256_bytes(
                    common._canonical_json_bytes(receipt)
                ),
                "metrics": receipt["metrics"],
                "runtime": receipt["runtime"],
                "claims": receipt["claims"],
            }
        elif args.preflight:
            report = build_preflight(config)
        else:
            report = build_contract_report(config)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (DiagnosticError, FileExistsError, OSError) as exc:
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
