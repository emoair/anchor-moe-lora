"""Native-Windows one-active-LoRA server for the frozen formal A--F run.

The module deliberately imports the CUDA/Transformers stack only after every
offline artifact and registry gate has passed.  Its HTTP surface matches the
small vLLM subset consumed by ``formal-run``:

* ``GET /v1/models``
* ``POST /v1/chat/completions``
* ``POST /v1/load_lora_adapter``
* ``POST /v1/unload_lora_adapter``

An additional, read-only ``/admin/probe`` endpoint makes backend state visible
without opening a held-out case or mutating the active adapter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from types import MethodType
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..benchmark.formal_af_preflight import (
    FormalAFPreflightError,
    preflight as formal_af_preflight,
)
from ..training.experiment_registry import (
    ExperimentRegistryError,
    verify_registry,
)


_HEX64 = re.compile(r"[0-9a-f]{64}")
_GROUPS = tuple("ABCDEF")
_STAGES = ("planner", "tool_policy", "frontend", "review", "security")
PROCESSOR_SCHEMA = "anchor.formal-af-processor.v1"
_DROP_DECODE_ATTENTION_MASK = "_anchor_drop_unpadded_decode_attention_mask"
_DECODE_FAST_PATH_INSTALLED = "_anchor_unpadded_decode_fast_path_installed"


class WindowsSerialServerError(RuntimeError):
    """A formal serving invariant was violated."""


class RequestContractError(WindowsSerialServerError):
    """An HTTP request is incompatible with the frozen formal contract."""

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowsSerialServerError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise WindowsSerialServerError(f"{label} must be a JSON object: {path}")
    return value


def _inside(root: Path, candidate: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise WindowsSerialServerError(f"{label} escapes the project root") from exc
    return resolved


def _resolve_from(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _generation_stop_token_ids(tokenizer: Any, model: Any) -> list[int]:
    """Return every public assistant-turn stop token known to the artifact.

    Gemma 4 serializes an assistant turn with ``<turn|>`` (``eot_token``), while
    its ordinary ``eos_token_id`` remains the document-level EOS token.  Passing
    only the latter to ``generate`` lets a completed assistant turn continue
    until ``max_new_tokens``.  Resolve the turn token from tokenizer metadata so
    the server remains artifact-driven rather than hard-coding Gemma's token id.
    """

    token_ids: list[int] = []

    def add(value: Any) -> None:
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        for item in values:
            if isinstance(item, bool):
                continue
            try:
                token_id = int(item)
            except (TypeError, ValueError):
                continue
            if token_id >= 0 and token_id not in token_ids:
                token_ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    for config_name in ("generation_config", "config"):
        add(getattr(getattr(model, config_name, None), "eos_token_id", None))

    eot_tokens: list[str] = []
    direct_eot = getattr(tokenizer, "eot_token", None)
    if isinstance(direct_eot, str) and direct_eot:
        eot_tokens.append(direct_eot)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        configured_eot = init_kwargs.get("eot_token")
        if isinstance(configured_eot, str) and configured_eot:
            eot_tokens.append(configured_eot)
        model_tokens = init_kwargs.get("model_specific_special_tokens")
        if isinstance(model_tokens, Mapping):
            configured_eot = model_tokens.get("eot_token")
            if isinstance(configured_eot, str) and configured_eot:
                eot_tokens.append(configured_eot)

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        unknown_token = getattr(tokenizer, "unk_token", None)
        for token in dict.fromkeys(eot_tokens):
            token_id = convert(token)
            if token_id == unknown_id and token != unknown_token:
                continue
            add(token_id)

    if not token_ids:
        raise WindowsSerialServerError(
            "tokenizer/model artifact declares no generation stop token"
        )
    return token_ids


@dataclass(frozen=True)
class AdapterBinding:
    model_id: str
    group: str
    artifact_name: str
    adapter_dir: Path
    adapter_sha256: str
    registry_sha256: str


RegistryVerifier = Callable[[str | Path, str | Path], Mapping[str, Any]]


class FormalAdapterCatalog:
    """Freeze and revalidate the preflight-produced runtime adapter map."""

    def __init__(
        self,
        project_root: str | Path,
        preflight_result: Mapping[str, Any],
        run_manifest_path: str | Path,
        *,
        registry_verifier: Callable[..., Mapping[str, Any]] = verify_registry,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.run_manifest_path = _inside(
            self.project_root,
            _resolve_from(self.project_root, run_manifest_path),
            "run manifest",
        )
        self.run_root = self.run_manifest_path.parent
        self._verify_registry = registry_verifier
        run = _load_object(self.run_manifest_path, "run manifest")
        run_groups = run.get("groups")
        if not isinstance(run_groups, Mapping) or set(run_groups) != set(_GROUPS):
            raise WindowsSerialServerError("run manifest must contain exactly A through F")
        self._registry_paths: dict[str, Path] = {}
        for group in _GROUPS:
            raw = run_groups[group]
            if not isinstance(raw, Mapping):
                raise WindowsSerialServerError(f"run group {group} is malformed")
            self._registry_paths[group] = _inside(
                self.project_root,
                _resolve_from(self.project_root, str(raw.get("registry_path", ""))),
                f"group {group} registry",
            )

        runtime = preflight_result.get("runtime_bindings")
        locks = preflight_result.get("registry_locks")
        serial = preflight_result.get("serial_runtime_contract")
        if not isinstance(runtime, Mapping) or set(runtime) != set(_GROUPS):
            raise WindowsSerialServerError("preflight runtime map must contain A through F")
        if not isinstance(locks, Mapping) or set(locks) != set(_GROUPS):
            raise WindowsSerialServerError("preflight registry locks must contain A through F")
        if not isinstance(serial, Mapping):
            raise WindowsSerialServerError("preflight serial contract is missing")
        if serial.get("maximum_active_loras") != 1:
            raise WindowsSerialServerError("formal server requires maximum_active_loras=1")
        self.base_model_id = str(serial.get("base_model_id", ""))
        if not self.base_model_id:
            raise WindowsSerialServerError("formal base model id is missing")

        bindings: dict[str, AdapterBinding] = {}
        for group in _GROUPS:
            stages = runtime[group]
            lock = locks[group]
            if not isinstance(stages, Mapping) or set(stages) != set(_STAGES):
                raise WindowsSerialServerError(
                    f"group {group} must bind exactly the five formal stages"
                )
            if not isinstance(lock, Mapping):
                raise WindowsSerialServerError(f"group {group} registry lock is malformed")
            registry_sha = str(lock.get("registry_sha256", ""))
            if not _HEX64.fullmatch(registry_sha):
                raise WindowsSerialServerError(f"group {group} registry digest is invalid")
            for stage in _STAGES:
                raw = stages[stage]
                if not isinstance(raw, Mapping):
                    raise WindowsSerialServerError(
                        f"group {group} stage {stage} binding is malformed"
                    )
                model_id = str(raw.get("model_id", ""))
                if group == "A":
                    if model_id != self.base_model_id or any(
                        raw.get(key) is not None
                        for key in ("adapter_artifact", "adapter_dir", "adapter_sha256")
                    ):
                        raise WindowsSerialServerError("group A must be the bare base only")
                    continue
                artifact = str(raw.get("adapter_artifact", ""))
                digest = str(raw.get("adapter_sha256", ""))
                relative = str(raw.get("adapter_dir", ""))
                if not model_id or not artifact or not _HEX64.fullmatch(digest):
                    raise WindowsSerialServerError(
                        f"group {group} stage {stage} adapter binding is incomplete"
                    )
                adapter_dir = _inside(
                    self.project_root,
                    _resolve_from(self.project_root, relative),
                    f"adapter {model_id}",
                )
                if not adapter_dir.is_dir():
                    raise WindowsSerialServerError(
                        f"adapter directory is missing for {model_id}"
                    )
                candidate = AdapterBinding(
                    model_id=model_id,
                    group=group,
                    artifact_name=artifact,
                    adapter_dir=adapter_dir,
                    adapter_sha256=digest,
                    registry_sha256=registry_sha,
                )
                previous = bindings.get(model_id)
                if previous is not None and previous != candidate:
                    raise WindowsSerialServerError(
                        f"model id {model_id!r} maps to multiple formal artifacts"
                    )
                bindings[model_id] = candidate
        if not bindings:
            raise WindowsSerialServerError("formal runtime map contains no adapters")
        self._bindings = bindings

    @property
    def adapter_model_ids(self) -> frozenset[str]:
        return frozenset(self._bindings)

    def validate_load(self, name: str, raw_path: str | Path) -> AdapterBinding:
        """Validate an admin load against the startup-verified frozen map.

        The full formal preflight hashes every indexed adapter before the GPU is
        loaded. Re-hashing large adapter files inside every measured stage call
        would contaminate latency comparisons, so the hot path re-hashes only
        the small immutable registry and checks exact paths/sizes. The explicit
        probe endpoint calls :meth:`rehash_load` for a fresh full-file audit.
        """

        binding = self._bindings.get(name)
        if binding is None:
            raise RequestContractError(
                f"unregistered formal adapter model id: {name}",
                status=404,
                code="unknown_adapter",
            )
        supplied = _resolve_from(self.project_root, raw_path)
        if supplied != binding.adapter_dir:
            raise RequestContractError(
                "adapter path does not match the frozen formal registry",
                status=409,
                code="adapter_path_mismatch",
            )
        registry_path = self._registry_paths[binding.group]
        if _sha256_file(registry_path) != binding.registry_sha256:
            raise RequestContractError(
                "adapter registry digest changed after server preflight",
                status=409,
                code="registry_digest_mismatch",
            )
        self._validate_registry_record(binding)
        return binding

    def rehash_load(self, name: str, raw_path: str | Path) -> AdapterBinding:
        """Repeat registry and indexed adapter-file hashes without loading PEFT."""

        binding = self.validate_load(name, raw_path)
        try:
            verified = self._verify_registry(
                self.project_root, self.run_root, group=binding.group
            )
        except (ExperimentRegistryError, OSError) as exc:
            raise RequestContractError(
                "adapter registry verification failed",
                status=409,
                code="registry_verification_failed",
            ) from exc
        if verified.get("registry_sha256") != binding.registry_sha256:
            raise RequestContractError(
                "adapter registry digest changed after server preflight",
                status=409,
                code="registry_digest_mismatch",
            )
        return binding

    def _validate_registry_record(self, binding: AdapterBinding) -> None:
        registry = _load_object(
            self._registry_paths[binding.group], f"group {binding.group} registry"
        )
        records = registry.get("adapters")
        if not isinstance(records, list):
            raise RequestContractError(
                "adapter registry has no artifact index",
                status=409,
                code="registry_artifact_missing",
            )
        match = next(
            (
                item
                for item in records
                if isinstance(item, Mapping)
                and item.get("artifact_name") == binding.artifact_name
            ),
            None,
        )
        if not isinstance(match, Mapping) or match.get("adapter_sha256") != (
            binding.adapter_sha256
        ):
            raise RequestContractError(
                "adapter artifact digest changed after server preflight",
                status=409,
                code="adapter_digest_mismatch",
            )
        final_files = match.get("final_files")
        if not isinstance(final_files, Mapping):
            raise RequestContractError(
                "adapter final-file index is missing",
                status=409,
                code="adapter_file_index_missing",
            )
        indexed_parents = {
            _resolve_from(self.project_root, str(item.get("path", ""))).parent
            for item in final_files.values()
            if isinstance(item, Mapping)
        }
        if indexed_parents != {binding.adapter_dir}:
            raise RequestContractError(
                "adapter file index points outside the frozen adapter directory",
                status=409,
                code="adapter_file_path_mismatch",
            )
        for label, item in final_files.items():
            if not isinstance(item, Mapping):
                raise RequestContractError(
                    "adapter final-file index is malformed",
                    status=409,
                    code="adapter_file_index_invalid",
                )
            path = _resolve_from(self.project_root, str(item.get("path", "")))
            expected_bytes = item.get("bytes")
            if (
                not path.is_file()
                or isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or path.stat().st_size != expected_bytes
            ):
                raise RequestContractError(
                    f"indexed adapter file changed size: {label}",
                    status=409,
                    code="adapter_file_size_mismatch",
                )


def verify_processor_artifact(
    project_root: str | Path,
    manifest_path: str | Path,
) -> Path:
    """Verify the exact Gemma-4-IT processor used while training the adapters."""

    root = Path(project_root).resolve()
    manifest_path = _inside(
        root, _resolve_from(root, manifest_path), "processor manifest"
    )
    manifest = _load_object(manifest_path, "processor manifest")
    if manifest.get("schema_version") != PROCESSOR_SCHEMA:
        raise WindowsSerialServerError("unsupported formal processor manifest")
    processor_dir = _inside(
        root,
        _resolve_from(root, str(manifest.get("processor_path", ""))),
        "processor directory",
    )
    if not processor_dir.is_dir():
        raise WindowsSerialServerError("formal processor directory is missing")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise WindowsSerialServerError("formal processor manifest has no files")
    observed: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise WindowsSerialServerError(f"processor file {index} is malformed")
        name = str(item.get("path", ""))
        if not name or Path(name).name != name or name in names:
            raise WindowsSerialServerError("processor manifest contains an unsafe path")
        names.add(name)
        path = processor_dir / name
        expected_sha = str(item.get("sha256", ""))
        expected_bytes = item.get("bytes")
        if (
            not path.is_file()
            or not _HEX64.fullmatch(expected_sha)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha
        ):
            raise WindowsSerialServerError(f"formal processor file changed: {name}")
        observed.append(
            {"path": name, "bytes": expected_bytes, "sha256": expected_sha}
        )
    actual_names = {path.name for path in processor_dir.iterdir() if path.is_file()}
    if actual_names != names:
        raise WindowsSerialServerError("formal processor directory inventory changed")
    if _canonical_sha256(sorted(observed, key=lambda item: item["path"])) != manifest.get(
        "tree_sha256"
    ):
        raise WindowsSerialServerError("formal processor tree digest mismatch")
    if "chat_template.jinja" not in names:
        raise WindowsSerialServerError("formal processor has no frozen chat template")
    return processor_dir


def verify_base_artifact(
    project_root: str | Path,
    base_model_path: str | Path,
    run_manifest_path: str | Path,
) -> Path:
    """Hash the registered prequantized NF4 base before importing CUDA code."""

    root = Path(project_root).resolve()
    run_path = _inside(root, _resolve_from(root, run_manifest_path), "run manifest")
    run = _load_object(run_path, "run manifest")
    base = run.get("base_artifact")
    if not isinstance(base, Mapping):
        raise WindowsSerialServerError("run manifest has no base artifact binding")
    expected_manifest = _inside(
        root,
        _resolve_from(root, str(base.get("manifest_path", ""))),
        "base quantization manifest",
    )
    model_dir = _inside(root, _resolve_from(root, base_model_path), "base model")
    if model_dir != expected_manifest.parent:
        raise WindowsSerialServerError("base model path differs from the run registry")
    manifest_sha = str(base.get("manifest_sha256", ""))
    if not _HEX64.fullmatch(manifest_sha) or _sha256_file(expected_manifest) != manifest_sha:
        raise WindowsSerialServerError("base quantization manifest digest mismatch")
    manifest = _load_object(expected_manifest, "base quantization manifest")
    if manifest.get("schema_version") != "anchor.bnb-nf4-export.v1":
        raise WindowsSerialServerError("base artifact is not the frozen NF4 export")
    quant = manifest.get("quantization")
    if not isinstance(quant, Mapping) or {
        "type": quant.get("type"),
        "double_quant": quant.get("double_quant"),
        "compute_dtype": quant.get("compute_dtype"),
        "storage_dtype": quant.get("storage_dtype"),
    } != {
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
        "storage_dtype": "bfloat16",
    }:
        raise WindowsSerialServerError("base NF4 quantization contract changed")
    weights = manifest.get("weights")
    if not isinstance(weights, list) or not weights:
        raise WindowsSerialServerError("base quantization manifest has no shards")
    for index, item in enumerate(weights):
        if not isinstance(item, Mapping):
            raise WindowsSerialServerError(f"base shard {index} is malformed")
        name = str(item.get("path", ""))
        expected_sha = str(item.get("sha256", ""))
        expected_bytes = item.get("bytes")
        if not name or Path(name).name != name or not _HEX64.fullmatch(expected_sha):
            raise WindowsSerialServerError(f"base shard {index} binding is invalid")
        path = model_dir / name
        if (
            not path.is_file()
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha
        ):
            raise WindowsSerialServerError(f"frozen base shard changed: {name}")
    config = _load_object(model_dir / "config.json", "base Transformers config")
    model_quant = config.get("quantization_config")
    if not isinstance(model_quant, Mapping) or not (
        model_quant.get("quant_method") == "bitsandbytes"
        and model_quant.get("load_in_4bit") is True
        and model_quant.get("load_in_8bit") is False
        and model_quant.get("bnb_4bit_quant_type") == "nf4"
        and model_quant.get("bnb_4bit_use_double_quant") is True
        and model_quant.get("bnb_4bit_compute_dtype") == "bfloat16"
    ):
        raise WindowsSerialServerError("base Transformers NF4 config changed")
    return model_dir


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[dict[str, Any], ...]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: Any = None
    agent_mode: bool = False


def parse_chat_request(
    payload: Any,
    *,
    allowed_models: frozenset[str],
    token_cap: int,
) -> ChatRequest:
    if not isinstance(payload, Mapping):
        raise RequestContractError("request body must be a JSON object")
    allowed_fields = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stream",
        "n",
    }
    unknown = set(payload) - allowed_fields
    if unknown:
        raise RequestContractError(
            f"unsupported formal request fields: {sorted(unknown)}",
            code="unsupported_request_field",
        )
    model = str(payload.get("model", ""))
    if model not in allowed_models:
        raise RequestContractError(
            f"unregistered formal model id: {model}",
            status=404,
            code="model_not_found",
        )
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise RequestContractError("messages must be a non-empty array")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise RequestContractError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise RequestContractError(
                f"message {index} requires a text content and a supported role"
            )
        messages.append({"role": str(role), "content": content})
    if not any(item["role"] == "user" for item in messages):
        raise RequestContractError("formal chat requires at least one user message")
    raw_max = payload.get("max_tokens", token_cap)
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or not (1 <= raw_max <= token_cap):
        raise RequestContractError(f"max_tokens must be between 1 and {token_cap}")
    raw_temperature = payload.get("temperature", 0.0)
    raw_top_p = payload.get("top_p", 1.0)
    if not isinstance(raw_temperature, (int, float)) or float(raw_temperature) != 0.0:
        raise RequestContractError("formal A--F inference requires temperature=0")
    if not isinstance(raw_top_p, (int, float)) or float(raw_top_p) != 1.0:
        raise RequestContractError("formal A--F inference requires top_p=1")
    raw_stream = payload.get("stream", False)
    if not isinstance(raw_stream, bool):
        raise RequestContractError("stream must be a boolean")
    if payload.get("n", 1) != 1:
        raise RequestContractError("formal A--F inference supports n=1 only")
    return ChatRequest(
        model=model,
        messages=tuple(messages),
        max_tokens=raw_max,
        temperature=0.0,
        top_p=1.0,
        stream=raw_stream,
    )


def _validate_agent_content(value: Any, *, index: int) -> Any:
    if value is None or isinstance(value, str):
        return value if value is not None else ""
    if not isinstance(value, list):
        raise RequestContractError(
            f"agent message {index} content must be text, null, or content parts"
        )
    normalized: list[dict[str, Any]] = []
    for part_index, part in enumerate(value):
        if not isinstance(part, Mapping):
            raise RequestContractError(
                f"agent message {index} content part {part_index} must be an object"
            )
        part_type = part.get("type")
        if part_type != "text":
            raise RequestContractError(
                f"agent message {index} is text-only; unsupported content part "
                f"{part_type!r}"
            )
        if part_type == "text" and not isinstance(part.get("text"), str):
            raise RequestContractError(
                f"agent message {index} text part requires text"
            )
        normalized.append(dict(part))
    return normalized


def parse_agent_chat_request(
    payload: Any,
    *,
    allowed_models: frozenset[str],
    token_cap: int,
) -> ChatRequest:
    """Parse the broader OpenAI subset used by the local OpenCode harness."""

    if not isinstance(payload, Mapping):
        raise RequestContractError("request body must be a JSON object")
    allowed_fields = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stream_options",
        "n",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "seed",
        "stop",
    }
    unknown = set(payload) - allowed_fields
    if unknown:
        raise RequestContractError(
            f"unsupported agent request fields: {sorted(unknown)}",
            code="unsupported_request_field",
        )
    model = str(payload.get("model", ""))
    if model not in allowed_models:
        raise RequestContractError(
            f"unregistered agent model id: {model}",
            status=404,
            code="model_not_found",
        )
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise RequestContractError("messages must be a non-empty array")
    messages: list[dict[str, Any]] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            raise RequestContractError(f"agent message {index} must be an object")
        role = raw_message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise RequestContractError(
                f"agent message {index} has unsupported role {role!r}"
            )
        message: dict[str, Any] = {
            "role": str(role),
            "content": _validate_agent_content(raw_message.get("content"), index=index),
        }
        if isinstance(raw_message.get("name"), str):
            message["name"] = str(raw_message["name"])
        if role == "assistant" and raw_message.get("tool_calls") is not None:
            raw_calls = raw_message.get("tool_calls")
            if not isinstance(raw_calls, list):
                raise RequestContractError(
                    f"agent message {index} tool_calls must be an array"
                )
            calls: list[dict[str, Any]] = []
            for call_index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, Mapping) or raw_call.get("type") != "function":
                    raise RequestContractError(
                        f"agent message {index} tool call {call_index} must be a function"
                    )
                function = raw_call.get("function")
                if not isinstance(function, Mapping) or not isinstance(
                    function.get("name"), str
                ):
                    raise RequestContractError(
                        f"agent message {index} tool call {call_index} has no function name"
                    )
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, (str, Mapping)):
                    raise RequestContractError(
                        f"agent message {index} tool call {call_index} arguments are invalid"
                    )
                call: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": str(function["name"]),
                        "arguments": dict(arguments) if isinstance(arguments, Mapping) else arguments,
                    },
                }
                if isinstance(raw_call.get("id"), str):
                    call["id"] = str(raw_call["id"])
                calls.append(call)
            message["tool_calls"] = calls
        if role == "tool":
            tool_call_id = raw_message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise RequestContractError(
                    f"agent tool message {index} requires tool_call_id"
                )
            message["tool_call_id"] = tool_call_id
        reasoning = raw_message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            message["reasoning_content"] = reasoning
        messages.append(message)
    if not any(item["role"] == "user" for item in messages):
        raise RequestContractError("agent chat requires at least one user message")

    raw_max = payload.get(
        "max_completion_tokens", payload.get("max_tokens", token_cap)
    )
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or not (1 <= raw_max <= token_cap):
        raise RequestContractError(f"max_tokens must be between 1 and {token_cap}")
    raw_temperature = payload.get("temperature", 0.0)
    raw_top_p = payload.get("top_p", 1.0)
    if (
        isinstance(raw_temperature, bool)
        or not isinstance(raw_temperature, (int, float))
        or not 0.0 <= float(raw_temperature) <= 2.0
    ):
        raise RequestContractError("temperature must be between 0 and 2")
    if (
        isinstance(raw_top_p, bool)
        or not isinstance(raw_top_p, (int, float))
        or not 0.0 < float(raw_top_p) <= 1.0
    ):
        raise RequestContractError("top_p must be greater than 0 and at most 1")
    raw_stream = payload.get("stream", False)
    if not isinstance(raw_stream, bool):
        raise RequestContractError("stream must be a boolean")
    stream_options = payload.get("stream_options")
    if stream_options is not None:
        if not isinstance(stream_options, Mapping) or set(stream_options) - {
            "include_usage"
        }:
            raise RequestContractError("stream_options is invalid")
        if not isinstance(stream_options.get("include_usage", False), bool):
            raise RequestContractError("stream_options.include_usage must be boolean")
    if payload.get("n", 1) != 1:
        raise RequestContractError("agent inference supports n=1 only")
    if payload.get("stop") not in (None, [], ""):
        raise RequestContractError("custom stop sequences are not supported")

    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list) or len(raw_tools) > 128:
        raise RequestContractError("tools must be an array with at most 128 entries")
    tools: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, Mapping) or raw_tool.get("type") != "function":
            raise RequestContractError(f"tool {index} must be a function declaration")
        function = raw_tool.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            raise RequestContractError(f"tool {index} requires a function name")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, Mapping):
            raise RequestContractError(f"tool {index} parameters must be an object")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(function["name"]),
                    "description": str(function.get("description", "")),
                    "parameters": dict(parameters),
                },
            }
        )
    tool_choice = payload.get("tool_choice", "auto")
    if tool_choice == "none":
        tools = []
    elif tool_choice not in (None, "auto", "required") and not isinstance(
        tool_choice, Mapping
    ):
        raise RequestContractError("tool_choice is invalid")
    return ChatRequest(
        model=model,
        messages=tuple(messages),
        max_tokens=raw_max,
        temperature=float(raw_temperature),
        top_p=float(raw_top_p),
        stream=raw_stream,
        tools=tuple(tools),
        tool_choice=tool_choice,
        agent_mode=True,
    )


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    message: dict[str, Any] | None = None


def _coalesce_agent_system_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map leading OpenAI system/developer messages to Gemma's system turn.

    The frozen Gemma chat template consumes one leading system/developer message.
    OpenAI clients may send several.  Preserve all leading instruction text in
    order instead of silently rendering later developer messages as unknown roles.
    """

    leading: list[str] = []
    remaining: list[dict[str, Any]] = []
    still_leading = True
    for raw in messages:
        message = dict(raw)
        if still_leading and message.get("role") in {"system", "developer"}:
            content = message.get("content", "")
            if isinstance(content, str):
                leading.append(content)
            elif isinstance(content, list):
                leading.append(
                    "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, Mapping) and part.get("type") == "text"
                    )
                )
            continue
        still_leading = False
        if message.get("role") == "developer":
            raise RequestContractError(
                "developer messages are supported only before the first user turn"
            )
        remaining.append(message)
    if leading:
        return [{"role": "system", "content": "\n\n".join(leading)}, *remaining]
    return remaining


def _openai_agent_message(parsed: Any) -> dict[str, Any]:
    """Normalize ``Gemma4UnifiedProcessor.parse_response`` to OpenAI format."""

    if not isinstance(parsed, Mapping):
        raise RequestContractError(
            "processor returned an invalid agent response",
            status=502,
            code="agent_response_parse_error",
        )
    message: dict[str, Any] = {"role": "assistant"}
    raw_calls = parsed.get("tool_calls")
    calls: list[dict[str, Any]] = []
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise RequestContractError(
                "processor returned invalid tool calls",
                status=502,
                code="agent_response_parse_error",
            )
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, Mapping):
                raise RequestContractError(
                    "processor returned an invalid tool call",
                    status=502,
                    code="agent_response_parse_error",
                )
            function = raw_call.get("function")
            if not isinstance(function, Mapping) or not isinstance(
                function.get("name"), str
            ):
                raise RequestContractError(
                    "processor returned a tool call without a function name",
                    status=502,
                    code="agent_response_parse_error",
                )
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                encoded_arguments = arguments
            else:
                encoded_arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            calls.append(
                {
                    "id": f"call_anchor_{index}_{secrets.token_hex(6)}",
                    "type": "function",
                    "function": {
                        "name": str(function["name"]),
                        "arguments": encoded_arguments,
                    },
                }
            )
    content = parsed.get("content")
    if content is not None and not isinstance(content, str):
        raise RequestContractError(
            "processor returned non-text agent content",
            status=502,
            code="agent_response_parse_error",
        )
    message["content"] = (
        content.strip() if isinstance(content, str) else (None if calls else "")
    )
    if calls:
        message["tool_calls"] = calls
    return message


class SerialEngine(Protocol):
    @property
    def active_adapter(self) -> str | None: ...

    @property
    def loaded(self) -> bool: ...

    def load_adapter(self, binding: AdapterBinding) -> None: ...

    def unload_adapter(self, name: str) -> None: ...

    def generate(self, request: ChatRequest) -> GenerationResult: ...

    def generate_stream(
        self,
        request: ChatRequest,
        emit: Callable[[str, Any], None],
    ) -> GenerationResult: ...

    def close(self) -> None: ...


def _install_unpadded_decode_fast_path(model: Any) -> None:
    """Drop a redundant all-one padding mask after the generation prefill.

    The formal server accepts one unpadded text sequence at a time. Transformers
    nevertheless grows its two-dimensional all-one ``attention_mask`` at every
    decoded token. Gemma 4 then checks ``padding_mask.all()`` while deciding
    whether SDPA can omit the causal mask; on CUDA that boolean check is a
    device-to-host synchronization on every token.

    Position ids and the KV cache have already been initialized by the prefill.
    For an unpadded single sequence, removing the padding mask from subsequent
    decode steps is therefore exactly equivalent to keeping an all-one mask.
    Gemma 4's SDPA path already reduces that all-one mask to ``None`` before the
    attention kernel. The per-request flag is enabled only after the CPU-side
    processor output has proved the request is unpadded, text-only, and batch 1.
    """

    if getattr(model, _DECODE_FAST_PATH_INSTALLED, False):
        return
    original = model._update_model_kwargs_for_generation

    def update_without_redundant_mask(
        bound_model: Any,
        outputs: Any,
        model_kwargs: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        updated = original(outputs, model_kwargs, *args, **kwargs)
        if getattr(bound_model, _DROP_DECODE_ATTENTION_MASK, False):
            updated.pop("attention_mask", None)
        return updated

    model._update_model_kwargs_for_generation = MethodType(
        update_without_redundant_mask, model
    )
    setattr(model, _DROP_DECODE_ATTENTION_MASK, False)
    setattr(model, _DECODE_FAST_PATH_INSTALLED, True)


def _is_unpadded_text_batch(encoded: Mapping[str, Any]) -> bool:
    """Return true only for the exact batch shape safe for the fast path."""

    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if (
        input_ids is None
        or attention_mask is None
        or getattr(input_ids, "ndim", None) != 2
        or getattr(attention_mask, "ndim", None) != 2
        or int(input_ids.shape[0]) != 1
        or tuple(attention_mask.shape) != tuple(input_ids.shape)
        or not bool(attention_mask.all().item())
    ):
        return False
    multimodal_types = encoded.get("mm_token_type_ids")
    return multimodal_types is None or not bool(multimodal_types.any().item())


class TransformersSerialEngine:
    """Keep one prequantized base and zero-or-one PEFT adapter on one GPU."""

    def __init__(
        self,
        *,
        base_model_path: Path,
        processor_path: Path,
        base_model_id: str,
        max_model_length: int,
        allow_tf32: bool = True,
        optimize_unpadded_decode: bool = True,
    ) -> None:
        self.base_model_path = base_model_path
        self.processor_path = processor_path
        self.base_model_id = base_model_id
        self.max_model_length = max_model_length
        self.allow_tf32 = allow_tf32
        self.optimize_unpadded_decode = optimize_unpadded_decode
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._peft_model_type: Any | None = None
        self._generation_model: Any | None = None
        self._active_adapter: str | None = None

    @property
    def active_adapter(self) -> str | None:
        with self._lock:
            return self._active_adapter

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None and self._processor is not None

    def load_base(self) -> None:
        with self._lock:
            if self.loaded:
                return
            import torch
            from peft import PeftModel
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            if not torch.cuda.is_available():
                raise WindowsSerialServerError("CUDA is required for the 12B NF4 server")
            if not torch.cuda.is_bf16_supported():
                raise WindowsSerialServerError("the frozen NF4 server requires CUDA BF16")
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.set_float32_matmul_precision("high" if self.allow_tf32 else "highest")
            processor = AutoProcessor.from_pretrained(
                str(self.processor_path), local_files_only=True
            )
            if not getattr(processor, "chat_template", None):
                raise WindowsSerialServerError("frozen processor chat template did not load")
            model = AutoModelForMultimodalLM.from_pretrained(
                str(self.base_model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": 0},
                local_files_only=True,
            )
            if not getattr(model, "is_loaded_in_4bit", False):
                raise WindowsSerialServerError("loaded base is not bitsandbytes 4-bit")
            model.eval()
            model.config.use_cache = True
            if self.optimize_unpadded_decode:
                _install_unpadded_decode_fast_path(model)
            self._torch = torch
            self._peft_model_type = PeftModel
            self._processor = processor
            self._model = model
            self._generation_model = model

    def load_adapter(self, binding: AdapterBinding) -> None:
        with self._lock:
            self._require_loaded()
            if self._active_adapter == binding.model_id:
                return
            if self._active_adapter is not None:
                self._unload_active()
            assert self._peft_model_type is not None
            self._model = self._peft_model_type.from_pretrained(
                self._model,
                str(binding.adapter_dir),
                adapter_name=binding.model_id,
                is_trainable=False,
                autocast_adapter_dtype=False,
                low_cpu_mem_usage=True,
            )
            self._model.set_adapter(binding.model_id)
            self._model.eval()
            self._model.config.use_cache = True
            configs = getattr(self._model, "peft_config", {})
            if set(configs) != {binding.model_id}:
                wrapper = self._model
                self._model = wrapper.unload()
                del wrapper
                raise WindowsSerialServerError("PEFT loaded more than one adapter")
            self._active_adapter = binding.model_id

    def unload_adapter(self, name: str) -> None:
        with self._lock:
            self._require_loaded()
            if self._active_adapter != name:
                raise RequestContractError(
                    f"adapter {name!r} is not active",
                    status=409,
                    code="adapter_not_active",
                )
            self._unload_active()

    def _unload_active(self) -> None:
        if self._active_adapter is None:
            return
        wrapper = self._model
        self._model = wrapper.unload()
        self._model.eval()
        self._model.config.use_cache = True
        self._active_adapter = None
        del wrapper
        gc.collect()
        assert self._torch is not None
        self._torch.cuda.empty_cache()

    def generate(self, request: ChatRequest) -> GenerationResult:
        with self._lock:
            self._require_loaded()
            expected = (
                self.base_model_id if self._active_adapter is None else self._active_adapter
            )
            if request.model != expected:
                raise RequestContractError(
                    f"requested model {request.model!r} is not the active formal model",
                    status=409,
                    code="active_adapter_mismatch",
                )
            assert self._processor is not None and self._torch is not None
            template_messages = (
                _coalesce_agent_system_messages(request.messages)
                if request.agent_mode
                else list(request.messages)
            )
            template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": False,
                "enable_thinking": False,
            }
            if request.agent_mode:
                template_kwargs["tools"] = list(request.tools)
            rendered = self._processor.apply_chat_template(
                template_messages,
                **template_kwargs,
            )
            encoded = self._processor(text=[rendered], return_tensors="pt")
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens + request.max_tokens > self.max_model_length:
                mode = "agent" if request.agent_mode else "formal"
                raise RequestContractError(
                    f"{mode} prompt uses {prompt_tokens} tokens and requests "
                    f"{request.max_tokens} completion tokens; context limit is "
                    f"{self.max_model_length}",
                    status=400,
                    code="context_length_exceeded",
                )
            device = self._model.device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            fast_decode = self.optimize_unpadded_decode and _is_unpadded_text_batch(
                encoded
            )
            if self._generation_model is not None:
                setattr(
                    self._generation_model,
                    _DROP_DECODE_ATTENTION_MASK,
                    fast_decode,
                )
            stop_token_ids = _generation_stop_token_ids(
                self._processor.tokenizer, self._model
            )
            try:
                with self._torch.inference_mode():
                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": request.max_tokens,
                        "do_sample": request.temperature > 0.0,
                        "use_cache": True,
                        "pad_token_id": self._processor.tokenizer.pad_token_id,
                        "eos_token_id": stop_token_ids,
                    }
                    if request.temperature > 0.0:
                        generation_kwargs.update(
                            temperature=request.temperature,
                            top_p=request.top_p,
                        )
                    output = self._model.generate(**inputs, **generation_kwargs)
            finally:
                if self._generation_model is not None:
                    setattr(
                        self._generation_model,
                        _DROP_DECODE_ATTENTION_MASK,
                        False,
                    )
            generated = output[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            stopped_on_declared_token = bool(completion_tokens) and int(
                generated[-1].item()
            ) in stop_token_ids
            message: dict[str, Any] | None = None
            if request.agent_mode:
                raw_response = self._processor.tokenizer.decode(
                    generated, skip_special_tokens=False
                )
                try:
                    parsed_response = self._processor.parse_response(raw_response)
                except Exception as exc:
                    raise RequestContractError(
                        "model produced an invalid Gemma tool-response envelope",
                        status=502,
                        code="agent_response_parse_error",
                    ) from exc
                message = _openai_agent_message(parsed_response)
                if request.tool_choice == "required" and not message.get("tool_calls"):
                    raise RequestContractError(
                        "tool_choice=required but the model did not call a tool",
                        status=502,
                        code="required_tool_call_missing",
                    )
                content = str(message.get("content") or "")
            else:
                content = self._processor.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
            finish_reason = (
                "tool_calls"
                if message and message.get("tool_calls")
                else (
                    "stop"
                    if stopped_on_declared_token
                    else (
                        "length"
                        if completion_tokens >= request.max_tokens
                        else "stop"
                    )
                )
            )
            return GenerationResult(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                message=message,
            )

    def generate_stream(
        self,
        request: ChatRequest,
        emit: Callable[[str, Any], None],
    ) -> GenerationResult:
        """Generate once while forwarding decoded text as it becomes available."""

        with self._lock:
            self._require_loaded()
            expected = (
                self.base_model_id if self._active_adapter is None else self._active_adapter
            )
            if request.model != expected:
                raise RequestContractError(
                    f"requested model {request.model!r} is not the active formal model",
                    status=409,
                    code="active_adapter_mismatch",
                )
            assert self._processor is not None and self._torch is not None
            rendered = self._processor.apply_chat_template(
                list(request.messages),
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            encoded = self._processor(text=[rendered], return_tensors="pt")
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens + request.max_tokens > self.max_model_length:
                raise RequestContractError(
                    "prompt tokens plus max_tokens exceed the formal context limit",
                    status=400,
                    code="context_length_exceeded",
                )
            device = self._model.device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            fast_decode = self.optimize_unpadded_decode and _is_unpadded_text_batch(
                encoded
            )
            if self._generation_model is not None:
                setattr(
                    self._generation_model,
                    _DROP_DECODE_ATTENTION_MASK,
                    fast_decode,
                )
            stop_token_ids = _generation_stop_token_ids(
                self._processor.tokenizer, self._model
            )

            from transformers import TextStreamer

            pieces: list[str] = []

            class CallbackTextStreamer(TextStreamer):
                def on_finalized_text(
                    self, text: str, stream_end: bool = False
                ) -> None:
                    if text:
                        pieces.append(text)
                        emit("content", text)

            streamer = CallbackTextStreamer(
                self._processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            emit("ready", prompt_tokens)
            try:
                with self._torch.inference_mode():
                    output = self._model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self._processor.tokenizer.pad_token_id,
                        eos_token_id=stop_token_ids,
                        streamer=streamer,
                    )
            finally:
                if self._generation_model is not None:
                    setattr(
                        self._generation_model,
                        _DROP_DECODE_ATTENTION_MASK,
                        False,
                    )
            generated = output[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            stopped_on_declared_token = bool(completion_tokens) and int(
                generated[-1].item()
            ) in stop_token_ids
            return GenerationResult(
                content="".join(pieces),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=(
                    "stop"
                    if stopped_on_declared_token
                    else (
                        "length"
                        if completion_tokens >= request.max_tokens
                        else "stop"
                    )
                ),
            )

    def close(self) -> None:
        with self._lock:
            if self._active_adapter is not None:
                self._unload_active()
            self._model = None
            self._processor = None
            self._generation_model = None
            gc.collect()
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    def _require_loaded(self) -> None:
        if self._model is None or self._processor is None:
            raise WindowsSerialServerError("base model is not loaded")


class FormalSerialService:
    """Protocol-facing state machine shared by aiohttp handlers and tests."""

    def __init__(
        self,
        catalog: FormalAdapterCatalog,
        engine: SerialEngine,
        *,
        token_cap: int,
    ) -> None:
        self.catalog = catalog
        self.engine = engine
        self.token_cap = token_cap
        self.started_at = time.time()
        self.load_count = 0
        self.unload_count = 0
        self.request_count = 0

    @property
    def allowed_models(self) -> frozenset[str]:
        return frozenset({self.catalog.base_model_id, *self.catalog.adapter_model_ids})

    def model_catalog(self) -> dict[str, Any]:
        # The frozen pre-heldout gate intentionally exposes the base only.
        return {
            "object": "list",
            "data": [
                {
                    "id": self.catalog.base_model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "anchor-formal-af",
                }
            ],
        }

    def agent_model_catalog(self) -> dict[str, Any]:
        active = self.engine.active_adapter or self.catalog.base_model_id
        return {
            "object": "list",
            "data": [
                {
                    "id": active,
                    "object": "model",
                    "created": 0,
                    "owned_by": "anchor-agent-diagnostic",
                }
            ],
        }

    def probe(self) -> dict[str, Any]:
        return {
            "ok": self.engine.loaded,
            "mode": "windows_transformers_peft_serial",
            "base_loaded": self.engine.loaded,
            "base_model_id": self.catalog.base_model_id,
            "active_adapter": self.engine.active_adapter,
            "maximum_active_loras": 1,
            "catalog_policy": "base_model_only_before_heldout",
            "unpadded_decode_fast_path": bool(
                getattr(self.engine, "optimize_unpadded_decode", False)
            ),
            "adapter_loads": self.load_count,
            "adapter_unloads": self.unload_count,
            "requests": self.request_count,
        }

    def load_adapter(self, payload: Any, *, dry_probe: bool = False) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise RequestContractError("adapter request body must be a JSON object")
        name = payload.get("lora_name")
        path = payload.get("lora_path")
        if not isinstance(name, str) or not name or not isinstance(path, str) or not path:
            raise RequestContractError("lora_name and lora_path are required strings")
        binding = (
            self.catalog.rehash_load(name, path)
            if dry_probe
            else self.catalog.validate_load(name, path)
        )
        if not dry_probe:
            self.engine.load_adapter(binding)
            self.load_count += 1
        return {
            "ok": True,
            "lora_name": name,
            "loaded": not dry_probe,
            "registry_verified": True,
            "maximum_active_loras": 1,
        }

    def unload_adapter(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("lora_name"), str
        ):
            raise RequestContractError("lora_name is required")
        name = str(payload["lora_name"])
        self.engine.unload_adapter(name)
        self.unload_count += 1
        return {"ok": True, "lora_name": name, "loaded": False}

    def parse_request(self, payload: Any) -> ChatRequest:
        return parse_chat_request(
            payload,
            allowed_models=self.allowed_models,
            token_cap=self.token_cap,
        )

    def parse_agent_request(self, payload: Any) -> ChatRequest:
        return parse_agent_chat_request(
            payload,
            allowed_models=self.allowed_models,
            token_cap=self.token_cap,
        )

    def complete_request(self, request: ChatRequest) -> dict[str, Any]:
        if request.stream:
            raise RequestContractError(
                "stream=true must use the streaming completion path"
            )
        result = self.engine.generate(request)
        self.request_count += 1
        created = int(time.time())
        return {
            "id": f"chatcmpl-anchor-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": result.message
                    or {"role": "assistant", "content": result.content},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    def complete(self, payload: Any) -> dict[str, Any]:
        return self.complete_request(self.parse_request(payload))

    def stream_complete(
        self,
        request: ChatRequest,
        emit: Callable[[str, Any], None],
    ) -> GenerationResult:
        if not request.stream:
            raise RequestContractError(
                "stream=false must use the non-streaming completion path"
            )
        result = self.engine.generate_stream(request, emit)
        self.request_count += 1
        return result


def create_app(service: FormalSerialService, *, api_key: str | None = None) -> Any:
    """Build the aiohttp application without importing the ML runtime."""

    from aiohttp import web

    @web.middleware
    async def contract_errors(request: Any, handler: Callable[..., Any]) -> Any:
        try:
            if api_key and request.path != "/health":
                supplied = request.headers.get("Authorization", "")
                if not secrets.compare_digest(supplied, f"Bearer {api_key}"):
                    raise RequestContractError(
                        "invalid API key", status=401, code="invalid_api_key"
                    )
            return await handler(request)
        except RequestContractError as exc:
            return web.json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "code": exc.code,
                    }
                },
                status=exc.status,
            )
        except json.JSONDecodeError:
            return web.json_response(
                {
                    "error": {
                        "message": "request body is not valid JSON",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
                status=400,
            )
        except WindowsSerialServerError as exc:
            return web.json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "code": "formal_backend_error",
                    }
                },
                status=500,
            )

    async def json_body(request: Any) -> Any:
        if request.content_length is not None and request.content_length > 2 * 1024 * 1024:
            raise RequestContractError("request body exceeds 2 MiB", status=413)
        return await request.json(loads=json.loads)

    async def health(_: Any) -> Any:
        status = 200 if service.engine.loaded else 503
        return web.json_response({"ok": service.engine.loaded}, status=status)

    async def models(_: Any) -> Any:
        return web.json_response(service.model_catalog())

    async def agent_models(_: Any) -> Any:
        return web.json_response(service.agent_model_catalog())

    async def admin_probe(_: Any) -> Any:
        return web.json_response(service.probe())

    async def load_adapter(request: Any) -> Any:
        payload = await json_body(request)
        result = await request.app["to_thread"](service.load_adapter, payload)
        return web.json_response(result)

    async def probe_adapter(request: Any) -> Any:
        payload = await json_body(request)
        result = await request.app["to_thread"](
            service.load_adapter, payload, dry_probe=True
        )
        return web.json_response(result)

    async def unload_adapter(request: Any) -> Any:
        payload = await json_body(request)
        result = await request.app["to_thread"](service.unload_adapter, payload)
        return web.json_response(result)

    def sse_data(value: Any) -> bytes:
        if value == "[DONE]":
            return b"data: [DONE]\n\n"
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return b"data: " + encoded + b"\n\n"

    async def stream_completion(request: Any, parsed: ChatRequest) -> Any:
        import asyncio

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(kind: str, value: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, value))

        async def run_worker() -> None:
            try:
                result = await request.app["to_thread"](
                    service.stream_complete, parsed, emit
                )
            except Exception as exc:  # forwarded before/inside the SSE boundary
                await queue.put(("error", exc))
            else:
                await queue.put(("done", result))

        worker = asyncio.create_task(run_worker())
        first_kind, first_value = await queue.get()
        if first_kind == "error":
            await worker
            raise first_value
        if first_kind != "ready":
            await worker
            raise WindowsSerialServerError(
                "streaming engine emitted content before readiness"
            )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        completion_id = f"chatcmpl-anchor-{secrets.token_hex(12)}"
        created = int(time.time())

        def chunk(
            delta: Mapping[str, Any],
            *,
            finish_reason: str | None = None,
            usage: Mapping[str, int] | None = None,
        ) -> dict[str, Any]:
            value: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": parsed.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": dict(delta),
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if usage is not None:
                value["usage"] = dict(usage)
            return value

        await response.write(sse_data(chunk({"role": "assistant", "content": ""})))
        try:
            while True:
                kind, value = await queue.get()
                if kind == "content":
                    await response.write(sse_data(chunk({"content": str(value)})))
                    continue
                if kind == "error":
                    code = (
                        value.code
                        if isinstance(value, RequestContractError)
                        else "formal_backend_error"
                    )
                    await response.write(
                        sse_data(
                            {
                                "error": {
                                    "message": str(value),
                                    "type": "server_error",
                                    "code": code,
                                }
                            }
                        )
                    )
                    await response.write(sse_data("[DONE]"))
                    break
                if kind == "done":
                    result = value
                    usage = {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": (
                            result.prompt_tokens + result.completion_tokens
                        ),
                    }
                    await response.write(
                        sse_data(
                            chunk(
                                {},
                                finish_reason=result.finish_reason,
                                usage=usage,
                            )
                        )
                    )
                    await response.write(sse_data("[DONE]"))
                    break
        finally:
            await worker
        return response

    async def completions(request: Any) -> Any:
        payload = await json_body(request)
        parsed = service.parse_request(payload)
        if parsed.stream:
            return await stream_completion(request, parsed)
        result = await request.app["to_thread"](service.complete_request, parsed)
        return web.json_response(result)

    async def agent_stream_completion(request: Any, parsed: ChatRequest) -> Any:
        """Emit OpenAI SSE deltas after parsing Gemma's complete tool envelope.

        Tool-call syntax cannot be decoded safely with the formal TextStreamer,
        because that streamer removes the special delimiters needed by
        ``processor.parse_response``.  Buffer one model turn, parse it, then emit
        standards-compatible content/tool-call deltas.  The formal endpoint keeps
        its true token stream unchanged.
        """

        completed = await request.app["to_thread"](
            service.complete_request,
            replace(parsed, stream=False),
        )
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        completion_id = str(completed["id"])
        created = int(completed["created"])
        choice = completed["choices"][0]
        message = choice["message"]

        def chunk(
            delta: Mapping[str, Any],
            *,
            finish_reason: str | None = None,
            usage: Mapping[str, int] | None = None,
        ) -> dict[str, Any]:
            value: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": parsed.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": dict(delta),
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if usage is not None:
                value["usage"] = dict(usage)
            return value

        await response.write(sse_data(chunk({"role": "assistant", "content": ""})))
        content = message.get("content")
        if isinstance(content, str) and content:
            await response.write(sse_data(chunk({"content": content})))
        for index, call in enumerate(message.get("tool_calls", [])):
            await response.write(
                sse_data(
                    chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": call["function"]["arguments"],
                                    },
                                }
                            ]
                        }
                    )
                )
            )
        await response.write(
            sse_data(
                chunk(
                    {},
                    finish_reason=str(choice["finish_reason"]),
                    usage=completed["usage"],
                )
            )
        )
        await response.write(sse_data("[DONE]"))
        return response

    async def agent_completions(request: Any) -> Any:
        payload = await json_body(request)
        parsed = service.parse_agent_request(payload)
        if parsed.stream:
            return await agent_stream_completion(request, parsed)
        result = await request.app["to_thread"](service.complete_request, parsed)
        return web.json_response(result)

    async def to_thread(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(function, *args, **kwargs)

    app = web.Application(middlewares=[contract_errors], client_max_size=2 * 1024**2)
    app["to_thread"] = to_thread
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/agent/v1/models", agent_models)
    app.router.add_get("/admin/probe", admin_probe)
    app.router.add_post("/v1/load_lora_adapter", load_adapter)
    app.router.add_post("/v1/probe_lora_adapter", probe_adapter)
    app.router.add_post("/v1/unload_lora_adapter", unload_adapter)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_post("/agent/v1/chat/completions", agent_completions)
    return app


def _require_loopback_host(host: str) -> str:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise WindowsSerialServerError("formal admin server must bind to loopback only")
    return host


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Native Windows Transformers+PEFT one-active-LoRA server for the "
            "frozen formal A-F benchmark"
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--formal-config", default="configs/benchmark/formal_partial_v1_af.json"
    )
    parser.add_argument(
        "--base-model", default="models/google-gemma-4-12B-bnb-nf4"
    )
    parser.add_argument(
        "--processor-manifest",
        default="configs/serving/formal_af_windows_processor.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-model-length",
        type=int,
        choices=(1024, 2048, 4096),
        default=2048,
    )
    parser.add_argument("--api-key-env", default="ANCHOR_VLLM_API_KEY")
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="disable TF32 acceleration for non-BF16 fallback operations",
    )
    parser.add_argument(
        "--disable-unpadded-decode-fast-path",
        action="store_true",
        help=(
            "retain Transformers' redundant all-one decode attention mask; "
            "use only for compatibility diagnostics"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify registries, adapter files, processor and all NF4 shards; do not import CUDA",
    )
    return parser


def _resolve_processor_manifest_binding(
    root: Path,
    requested_manifest: str | Path,
    config: Mapping[str, Any],
    preflight_result: Mapping[str, Any],
) -> Path:
    processor_manifest = _inside(
        root,
        _resolve_from(root, requested_manifest),
        "processor manifest",
    )
    configured_processor = config.get("processor_binding")
    if configured_processor is None:
        return processor_manifest
    if not isinstance(configured_processor, Mapping):
        raise WindowsSerialServerError("processor_binding must be an object")
    bound_processor_manifest = _inside(
        root,
        _resolve_from(root, str(configured_processor.get("manifest_path", ""))),
        "bound processor manifest",
    )
    if processor_manifest != bound_processor_manifest:
        raise WindowsSerialServerError(
            "processor manifest differs from the formal config binding"
        )
    expected_manifest_sha = str(configured_processor.get("manifest_sha256", ""))
    if (
        not _HEX64.fullmatch(expected_manifest_sha)
        or _sha256_file(processor_manifest) != expected_manifest_sha
        or preflight_result.get("processor_manifest_sha256")
        != expected_manifest_sha
    ):
        raise WindowsSerialServerError(
            "processor manifest digest differs from the formal preflight"
        )
    expected_tree_sha = str(configured_processor.get("tree_sha256", ""))
    if (
        not _HEX64.fullmatch(expected_tree_sha)
        or preflight_result.get("processor_tree_sha256") != expected_tree_sha
    ):
        raise WindowsSerialServerError(
            "processor tree digest differs from the formal preflight"
        )
    return processor_manifest


def _build_contract(args: argparse.Namespace) -> tuple[
    Path, Path, FormalAdapterCatalog, int
]:
    root = Path(args.project_root).resolve()
    config_path = _inside(root, _resolve_from(root, args.formal_config), "formal config")
    result = formal_af_preflight(config_path, root)
    config = _load_object(config_path, "formal config")
    run_manifest = _inside(
        root,
        _resolve_from(root, str(config.get("run_manifest_path", ""))),
        "run manifest",
    )
    catalog = FormalAdapterCatalog(root, result, run_manifest)
    base_path = verify_base_artifact(root, args.base_model, run_manifest)
    processor_manifest = _resolve_processor_manifest_binding(
        root, args.processor_manifest, config, result
    )
    processor_path = verify_processor_artifact(root, processor_manifest)
    token_cap = int(result.get("per_stage_token_cap", 0))
    if token_cap < 1:
        raise WindowsSerialServerError("formal token cap is invalid")
    return base_path, processor_path, catalog, token_cap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _require_loopback_host(args.host)
        if not (1 <= args.port <= 65535):
            raise WindowsSerialServerError("port must be between 1 and 65535")
        base, processor, catalog, token_cap = _build_contract(args)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "status": "ready",
                        "gpu_started": False,
                        "heldout_case_content_read": False,
                        "base_model_id": catalog.base_model_id,
                        "adapter_model_count": len(catalog.adapter_model_ids),
                        "maximum_active_loras": 1,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        engine = TransformersSerialEngine(
            base_model_path=base,
            processor_path=processor,
            base_model_id=catalog.base_model_id,
            max_model_length=args.max_model_length,
            allow_tf32=not args.disable_tf32,
            optimize_unpadded_decode=not args.disable_unpadded_decode_fast_path,
        )
        engine.load_base()
        service = FormalSerialService(catalog, engine, token_cap=token_cap)
        app = create_app(service, api_key=os.environ.get(args.api_key_env))
        from aiohttp import web

        print(
            f"Serving formal A-F on http://{args.host}:{args.port}; "
            "base=NF4, maximum_active_loras=1, "
            "unpadded_decode_fast_path="
            f"{'off' if args.disable_unpadded_decode_fast_path else 'on'}"
        )
        try:
            web.run_app(app, host=args.host, port=args.port, print=None)
        finally:
            engine.close()
        return 0
    except (
        WindowsSerialServerError,
        FormalAFPreflightError,
        ExperimentRegistryError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
