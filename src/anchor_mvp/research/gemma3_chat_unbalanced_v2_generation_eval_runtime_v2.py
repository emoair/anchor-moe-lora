"""Physical, fail-closed generation evaluation for Gemma 3 unbalanced-v2.

Validation is the default and is model/GPU/body free.  ``--dry-run``
authenticates historical producer and adapter metadata but does not parse
evaluation JSONL.  Only ``--execute`` opens the already authenticated Git
blobs and lazily imports the ML stack.

The runtime never uses the current producer worktree as evidence.  It reads
the immutable P candidate and R release-attestation objects with
``git cat-file`` and records their commit, tree, and blob identities.  Output
is aggregate-only: prompts, targets, completions, record identifiers, and raw
token IDs are neither logged nor persisted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Any, Protocol
import uuid

from jsonschema import Draft202012Validator

from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as multiarm,
)
from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_runtime_source_v2 as runtime_source_v2,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_generation_eval_runtime_v2.json"
)
CONFIG_SCHEMA_PATH = CONFIG_PATH.with_name(
    "gemma3_chat_unbalanced_v2_generation_eval_runtime_v2.schema.json"
)
RECEIPT_SCHEMA_PATH = CONFIG_PATH.with_name(
    "gemma3_chat_unbalanced_v2_generation_eval_runtime_receipt_v2.schema.json"
)
TRAINING_RUN_RECEIPT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "training"
    / "gemma3_chat_unbalanced_v2_multiarm_q8_runtime_run_receipt_v2.schema.json"
)
CONFIG_VERSION = "anchor.gemma3-chat-unbalanced-v2-generation-eval-runtime.v2"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-generation-eval-runtime-receipt.v2"
ADAPTER_RECEIPT_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-run-receipt.v2"
)
LOCK_VERSION = "anchor.gemma3-chat-unbalanced-v2-generation-eval-gpu-lock.v2"
PROGRESS_VERSION = "anchor.gemma3-chat-unbalanced-v2-generation-eval-scalar-progress.v2"

TRAINED_ARMS = tuple(multiarm.TRAINED_ARMS)
BASE_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "EXPORT_MANIFEST.json",
)
TOKENIZER_LEASE_FILES = tuple(
    name for name in BASE_MODEL_FILES if name != "model.safetensors"
)
RUNTIME_INPUT_LEASE_KEYS = (
    *(f"base:{name}" for name in BASE_MODEL_FILES),
    *(
        key
        for arm in TRAINED_ARMS
        for key in (f"adapter-config:{arm}", f"adapter-model:{arm}")
    ),
)
TOOL_ARMS = ("tool_base", "tool_q_only", "tool_q_plus_micro_o")
PLANNER_ARMS = (
    "planner_base",
    "planner_q_only",
    "planner_o_only",
    "planner_q_plus_micro_o",
)
IDENTITY_ARMS = ("identity_base", "identity_correct_route", "identity_wrong_route")
ROLE_ADAPTER = {
    "humor": "humor_q_only",
    "serious": "serious_q_only",
    "angry_style": "angry_style_q_only",
    "review_audit": "review_audit_q_only",
    "tool_call": "tool_q_only",
}
WRONG_ROUTE_RING = {
    "humor_q_only": ("serious_q_only", "angry_style_q_only"),
    "serious_q_only": ("angry_style_q_only", "review_audit_q_only"),
    "angry_style_q_only": ("review_audit_q_only", "humor_q_only"),
    "review_audit_q_only": ("humor_q_only", "serious_q_only"),
    "tool_q_only": ("planner_q_only", "planner_o_only"),
}
METRICS = (
    "json_structure_valid",
    "format_valid",
    "tool_name_exact",
    "tool_arguments_exact",
    "tool_arguments_grounded",
    "planner_label_exact",
    "planner_topology_exact",
    "identity_air_attribution",
    "identity_provider_misattribution_absent",
    "route_exact",
    "refusal_absent",
    "out_of_bounds_absent",
    "eos_terminated",
)
STRATA_KEYS = (
    "stratum",
    "language",
    "role",
    "identity",
    "router",
    "wrong_route",
)
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "body",
        "bodies",
        "prompt",
        "prompts",
        "messages",
        "target",
        "targets",
        "completion",
        "completions",
        "generated",
        "generated_text",
        "input_ids",
        "token_ids",
        "raw_tokens",
        "record_id",
        "record_ids",
        "per_record",
        "rows",
        "samples",
    }
)
PUBLIC_ERRORS = frozenset(
    {
        "eval_adapter_identity_drift",
        "eval_adapter_receipt_invalid",
        "eval_adapter_state_invalid",
        "eval_atomic_destination_exists",
        "eval_base_identity_drift",
        "eval_body_leak_detected",
        "eval_config_invalid",
        "eval_execute_requires_bound_adapter_receipt",
        "eval_fairness_lock_drift",
        "eval_git_blob_drift",
        "eval_git_commit_drift",
        "eval_gpu_lock_exists",
        "eval_historical_source_invalid",
        "eval_output_invalid",
        "eval_record_contract_invalid",
        "eval_source_identity_drift",
        "eval_tokenizer_identity_drift",
    }
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_OID_RE = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_LORA_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.(q_proj|o_proj)\."
    r"lora_([AB])(?:\.[^.]+)?\.weight$"
)
_MAX_JSON_BYTES = 8 * 1024 * 1024
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable|won't|refuse|sorry)\b|"
    r"(?:无法|不能|抱歉|拒绝)",
    re.IGNORECASE,
)
_AIR_TOKEN = r"(?<![A-Za-z0-9_])Air(?![A-Za-z0-9_])"
_PROVIDER_TOKEN = (
    r"(?<![A-Za-z0-9_])(?:OpenAI|Google(?:\s+DeepMind)?|DeepMind|"
    r"Anthropic|Qwen|Alibaba)(?![A-Za-z0-9_])|"
    r"(?:谷歌(?:\s*DeepMind)?|通义千问|千问|阿里巴巴)"
)
_PASSIVE_IDENTITY_VERB = (
    r"(?:trained|fine[- ]tuned|developed|created|built|designed|"
    r"customi[sz]ed|made|owned)"
)
_ACTIVE_IDENTITY_VERB = (
    r"(?:trained|fine[- ]tuned|developed|created|built|designed|"
    r"customi[sz]ed|made|owns?)"
)
_IDENTITY_SUBJECT = r"(?:I|this\s+(?:assistant|model|system))"
_IDENTITY_OBJECT = r"(?:me|this\s+(?:assistant|model|system))"
_IDENTITY_ROLE = r"(?:trainer|creator|developer|owner|author|provider)"
_ZH_IDENTITY_SUBJECT = r"(?:我|本模型|这个模型|该模型|本助手|这个助手|该助手)"
_ZH_IDENTITY_VERB = r"(?:训练|微调|开发|创建|打造|定制|设计|制作|拥有)"
_AIR_ATTRIBUTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"{_PASSIVE_IDENTITY_VERB}\s+by\s+{_AIR_TOKEN}",
        rf"\b{_IDENTITY_SUBJECT}\b\s+(?:was|am|is)\s+"
        rf"{_PASSIVE_IDENTITY_VERB}\s+by\s+{_AIR_TOKEN}",
        rf"{_AIR_TOKEN}\s+(?:has\s+)?{_ACTIVE_IDENTITY_VERB}\s+"
        rf"\b{_IDENTITY_OBJECT}\b",
        rf"\bmy\s+{_IDENTITY_ROLE}\s+(?:is|was)\s+{_AIR_TOKEN}",
        rf"\b{_IDENTITY_ROLE}\s*[:=]\s*{_AIR_TOKEN}",
        rf"\b{_IDENTITY_SUBJECT}\b\s+(?:belong(?:s)?\s+to|"
        rf"(?:am|is)\s+(?:from|owned\s+by))\s+{_AIR_TOKEN}",
        rf"{_ZH_IDENTITY_SUBJECT}?由\s*{_AIR_TOKEN}\s*{_ZH_IDENTITY_VERB}",
        rf"{_AIR_TOKEN}\s*{_ZH_IDENTITY_VERB}了?\s*{_ZH_IDENTITY_SUBJECT}",
        rf"(?:我的|本模型的|该模型的|本助手的)(?:训练者|开发者|"
        rf"创建者|作者|所有者|归属方)\s*(?:是|为)\s*{_AIR_TOKEN}",
        rf"{_ZH_IDENTITY_SUBJECT}\s*(?:属于|归属于|来自)\s*{_AIR_TOKEN}",
    )
)
_PROVIDER_MISATTRIBUTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"{_PASSIVE_IDENTITY_VERB}\s+by\s+(?:{_PROVIDER_TOKEN})",
        rf"\b{_IDENTITY_SUBJECT}\b\s+(?:was|am|is)\s+"
        rf"{_PASSIVE_IDENTITY_VERB}\s+by\s+(?:{_PROVIDER_TOKEN})",
        rf"(?:{_PROVIDER_TOKEN})\s+(?:has\s+)?"
        rf"{_ACTIVE_IDENTITY_VERB}\s+\b{_IDENTITY_OBJECT}\b",
        rf"\bmy\s+{_IDENTITY_ROLE}\s+(?:is|was)\s+"
        rf"(?:{_PROVIDER_TOKEN})",
        rf"\b{_IDENTITY_ROLE}\s*[:=]\s*(?:{_PROVIDER_TOKEN})",
        rf"\b{_IDENTITY_SUBJECT}\b\s+(?:belong(?:s)?\s+to|"
        rf"(?:am|is)\s+(?:from|owned\s+by))\s+(?:{_PROVIDER_TOKEN})",
        rf"{_ZH_IDENTITY_SUBJECT}?由\s*(?:{_PROVIDER_TOKEN})\s*"
        rf"{_ZH_IDENTITY_VERB}",
        rf"(?:{_PROVIDER_TOKEN})\s*{_ZH_IDENTITY_VERB}了?\s*"
        rf"{_ZH_IDENTITY_SUBJECT}",
        rf"(?:我的|本模型的|该模型的|本助手的)(?:训练者|开发者|"
        rf"创建者|作者|所有者|归属方)\s*(?:是|为)\s*"
        rf"(?:{_PROVIDER_TOKEN})",
        rf"{_ZH_IDENTITY_SUBJECT}\s*(?:属于|归属于|来自)\s*"
        rf"(?:{_PROVIDER_TOKEN})",
    )
)
_NEGATED_CLAIM_PREFIX_RE = re.compile(
    r"(?:\b(?:not|never|no|neither|nor|deny|denied|false|incorrect|"
    r"isn't|wasn't|aren't|weren't|don't|doesn't|didn't)\b|"
    r"(?:不是|并非|不由|未由|没有|从未|不属于|不归属|不来自|否认|错误))",
    re.IGNORECASE,
)


class GenerationEvalRuntimeError(RuntimeError):
    """A redacted, registered fail-closed runtime error."""


class _DuplicateKey(ValueError):
    pass


def _fail(code: str) -> None:
    if code not in PUBLIC_ERRORS:
        code = "eval_output_invalid"
    raise GenerationEvalRuntimeError(code)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _domain_sha256(domain: str, value: object) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _canonical_json(value))


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _strict_json_bytes(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, ValueError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _object_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        *_object_identity(value),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _regular_stat_valid(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and not stat.S_ISLNK(value.st_mode)
        and not _is_reparse(value)
    )


def _stable_regular_bytes(
    path: Path,
    *,
    code: str,
    max_bytes: int | None = None,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    try:
        path_before = path.lstat()
        if (
            not _regular_stat_valid(path_before)
            or path_before.st_size <= 0
            or (max_bytes is not None and path_before.st_size > max_bytes)
        ):
            _fail(code)
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if not _regular_stat_valid(opened_before) or _stat_fingerprint(
                opened_before
            ) != _stat_fingerprint(path_before):
                _fail(code)
            chunks: list[bytes] = []
            observed_bytes = 0
            while chunk := handle.read(1024 * 1024):
                chunks.append(chunk)
                observed_bytes += len(chunk)
                if max_bytes is not None and observed_bytes > max_bytes:
                    _fail(code)
            opened_after = os.fstat(handle.fileno())
        path_after = path.lstat()
    except OSError:
        _fail(code)
    fingerprint = _stat_fingerprint(path_before)
    if (
        _stat_fingerprint(opened_before) != fingerprint
        or _stat_fingerprint(opened_after) != fingerprint
        or _stat_fingerprint(path_after) != fingerprint
        or not _regular_stat_valid(opened_after)
        or not _regular_stat_valid(path_after)
        or observed_bytes != path_before.st_size
    ):
        _fail(code)
    return b"".join(chunks), fingerprint


def _stable_regular_digest(
    path: Path,
    *,
    code: str,
    expected_bytes: int | None = None,
) -> tuple[str, tuple[int, int, int, int, int, int]]:
    digest = hashlib.sha256()
    try:
        path_before = path.lstat()
        if not _regular_stat_valid(path_before) or path_before.st_size <= 0:
            _fail(code)
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if not _regular_stat_valid(opened_before) or _stat_fingerprint(
                opened_before
            ) != _stat_fingerprint(path_before):
                _fail(code)
            observed_bytes = 0
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                observed_bytes += len(chunk)
            opened_after = os.fstat(handle.fileno())
        path_after = path.lstat()
    except OSError:
        _fail(code)
    fingerprint = _stat_fingerprint(path_before)
    if (
        _stat_fingerprint(opened_before) != fingerprint
        or _stat_fingerprint(opened_after) != fingerprint
        or _stat_fingerprint(path_after) != fingerprint
        or not _regular_stat_valid(opened_after)
        or not _regular_stat_valid(path_after)
        or observed_bytes != path_before.st_size
        or (expected_bytes is not None and observed_bytes != expected_bytes)
    ):
        _fail(code)
    return digest.hexdigest(), fingerprint


def _read_json_with_raw(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw, _ = _stable_regular_bytes(
        path,
        code="eval_config_invalid",
        max_bytes=_MAX_JSON_BYTES,
    )
    if b"\r" in raw or not raw.endswith(b"\n"):
        _fail("eval_config_invalid")
    if expected_sha256 is not None and _sha256(raw) != expected_sha256:
        _fail("eval_source_identity_drift")
    return _strict_json_bytes(raw, "eval_config_invalid"), raw


def _read_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    value, _ = _read_json_with_raw(path, expected_sha256=expected_sha256)
    return value


def _schema(path: Path) -> Draft202012Validator:
    value = _read_json(path)
    try:
        Draft202012Validator.check_schema(value)
    except Exception:
        _fail("eval_config_invalid")
    return Draft202012Validator(value)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("eval_config_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        _fail("eval_config_invalid")
    return path.as_posix()


def _assert_body_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_OUTPUT_KEYS:
                _fail("eval_body_leak_detected")
            _assert_body_free(child)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            _assert_body_free(child)


def _contains_v1_identity(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_v1_identity(key) or _contains_v1_identity(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_v1_identity(child) for child in value)
    return (
        isinstance(value, str)
        and re.search(
            r"(?:\.v1\b|_v1\b|-v1\b)",
            value,
            re.IGNORECASE,
        )
        is not None
    )


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    requested = Path(path)
    config, raw = _read_json_with_raw(requested)
    if tuple(_schema(CONFIG_SCHEMA_PATH).iter_errors(config)):
        _fail("eval_config_invalid")
    validate_config(config)
    config["_config_sha256"] = _sha256(raw)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_VERSION or config.get(
        "default_mode"
    ) not in {"validate", "dry-run"}:
        _fail("eval_config_invalid")
    historical = config.get("historical_source")
    if not isinstance(historical, Mapping):
        _fail("eval_config_invalid")
    for name in ("candidate_commit", "release_commit"):
        value = historical.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            _fail("eval_config_invalid")
    for name in ("candidate_tree", "release_tree"):
        value = historical.get(name)
        if not isinstance(value, str) or _OID_RE.fullmatch(value) is None:
            _fail("eval_config_invalid")
    if (
        historical.get("read_mode") != "git_cat_file_only"
        or historical.get("required_read_set_files") != 88
        or historical.get("required_payload_files") != 23
        or historical.get("required_physical_files") != 46
    ):
        _fail("eval_config_invalid")
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        _fail("eval_config_invalid")
    if (
        execution.get("seed") != 1337
        or execution.get("max_new_tokens") != 256
        or execution.get("do_sample") is not False
        or execution.get("num_beams") != 1
        or execution.get("network_allowed") is not False
        or execution.get("gpu_concurrency") != 1
        or execution.get("windows_read_leases_required") is not True
        or execution.get("tokenizer_auth_read_leases") != 4
        or execution.get("runtime_input_read_leases") != 23
    ):
        _fail("eval_config_invalid")
    arms = config.get("adapter_run", {}).get("required_arms")
    if tuple(arms or ()) != TRAINED_ARMS:
        _fail("eval_config_invalid")
    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        _fail("eval_config_invalid")
    if (
        tokenizer.get("authority")
        != "anchor_mvp.training.gemma3_chat_unbalanced_v2_runtime_source_v2"
        or tokenizer.get("processor_source") != "authenticated_model_proto_bytes"
        or tokenizer.get("second_model_file_path_read") is not False
        or tokenizer.get("hf_auto_tokenizer_used") is not False
        or tokenizer.get("runtime_special_token_overlay")
        != {
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "unk_token_id": 3,
            "start_of_turn_token_id": 105,
            "end_of_turn_token_id": 106,
            "user_token_id": 2364,
            "model_token_id": 4368,
        }
    ):
        _fail("eval_config_invalid")
    model = config.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("environment_variable") != "ANCHOR_GEMMA_MODEL_DIR"
        or model.get("required_files") != list(BASE_MODEL_FILES)
    ):
        _fail("eval_config_invalid")
    matrix = config.get("matrix")
    if not isinstance(matrix, Mapping):
        _fail("eval_config_invalid")
    expected = {
        "tool": {"records": 400, "arms": 3, "slots": 1200},
        "planner": {"records": 240, "arms": 4, "slots": 960},
        "router": {"records": 20, "arms": 1, "slots": 20},
        "identity": {"records": 50, "arms": 3, "slots": 150},
        "wrong_route": {"records": 860, "arms": 3, "slots": 2580},
    }
    if matrix != {**expected, "total_slots": 4910}:
        _fail("eval_config_invalid")
    output = config.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("gpu_lock") != "runs/formal-v3-training.lock"
        or output.get("handle_disposition_delete_only") is not True
        or output.get("path_unlink_fallback") is not False
    ):
        _fail("eval_config_invalid")


class AuthenticatedRuntimeSource(Protocol):
    """Read-only interface supplied by training.runtime_source_v2."""

    def public_identity(self) -> Mapping[str, Any]: ...

    def iter_jsonl(self, payload_path: str) -> Iterator[dict[str, Any]]: ...

    def terminal_recheck(self) -> None: ...

    @property
    def consumer_receipt_sha256(self) -> str: ...

    @property
    def source_binding_sha256(self) -> str: ...

    @property
    def logical_identity(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class SharedRuntimeSource:
    source: runtime_source_v2.AuthenticatedProducerSource
    logical_identity: Mapping[str, str]
    candidate_tree: str

    @property
    def consumer_receipt_sha256(self) -> str:
        return self.source.consumer_preflight_receipt_sha256

    @property
    def source_binding_sha256(self) -> str:
        return self.source.consumer_source_binding_sha256

    def public_identity(self) -> dict[str, Any]:
        public = dict(self.source.public_identity())
        public["candidate_tree"] = self.candidate_tree
        public["payload_files"] = len(self.source.payload_pins) // 2
        public["physical_files"] = len(self.source.payload_pins)
        public["read_set_blob_inventory_sha256"] = public.pop(
            "read_set_blob_oids_sha256"
        )
        public["payload_blob_inventory_sha256"] = public.pop("payload_blob_oids_sha256")
        public["asset_blobs"] = {
            name: {
                "commit": self.source.release_commit,
                "tree": self.source.release_tree,
                "path": pin.path,
                "blob_oid": pin.oid,
                "bytes": pin.bytes,
                "sha256": pin.sha256,
            }
            for name, pin in sorted(self.source.payload_pins.items())
            if not name.endswith(".sha256")
        }
        return public

    def iter_jsonl(self, payload_path: str) -> Iterator[dict[str, Any]]:
        for value in runtime_source_v2.iter_authenticated_jsonl(
            self.source,
            payload_path,
        ):
            yield dict(value)

    def terminal_recheck(self) -> None:
        runtime_source_v2.terminal_recheck_source(self.source)


@dataclass(frozen=True)
class AdapterIdentity:
    arm: str
    profile: str
    path: Path
    trainable_parameters: int
    adapter_config_sha256: str
    adapter_config_bytes: int
    adapter_config_object_identity: tuple[int, int, int]
    adapter_model_sha256: str
    adapter_model_bytes: int
    adapter_model_object_identity: tuple[int, int, int]
    artifact_tree_sha256: str
    tensor_inventory_sha256: str


@dataclass(frozen=True)
class AuthenticatedContext:
    config_sha256: str
    consumer_receipt_sha256: str
    source_binding_sha256: str
    source_identity: Mapping[str, Any]
    logical_identity: Mapping[str, str]
    adapter_run_receipt_sha256: str
    adapter_inventory_sha256: str
    base_model: Mapping[str, Any]
    base_model_object_identities: Mapping[str, tuple[int, int, int]]
    base_parameter_identity_sha256: str
    q8_scb_inventory_sha256: str
    adapters: Mapping[str, AdapterIdentity]
    tokenizer: Mapping[str, Any]
    tokenizer_capability: runtime_source_v2.AuthenticatedTokenizer = field(
        repr=False,
        compare=False,
    )

    def public_identity(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "consumer_receipt_sha256": self.consumer_receipt_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "source": dict(self.source_identity),
            "logical_identity": dict(self.logical_identity),
            "adapter_run_receipt_sha256": self.adapter_run_receipt_sha256,
            "adapter_inventory_sha256": self.adapter_inventory_sha256,
            "base_model": {
                "files": dict(self.base_model["files"]),
                "inventory_sha256": self.base_model["inventory_sha256"],
            },
            "base_parameter_identity_sha256": (self.base_parameter_identity_sha256),
            "q8_scb_inventory_sha256": self.q8_scb_inventory_sha256,
            "tokenizer": dict(self.tokenizer),
        }


def _repo_local_path(relative: str) -> Path:
    path = ROOT.joinpath(*PurePosixPath(_safe_relative(relative)).parts)
    try:
        path.resolve(strict=False).relative_to(ROOT.resolve())
    except ValueError:
        _fail("eval_config_invalid")
    return path


def _file_sha256(
    path: Path,
    *,
    expected_bytes: int | None = None,
    code: str = "eval_adapter_identity_drift",
) -> str:
    digest, _ = _stable_regular_digest(
        path,
        expected_bytes=expected_bytes,
        code=code,
    )
    return digest


def _file_sha256_with_identity(
    path: Path,
    *,
    expected_bytes: int | None = None,
    code: str = "eval_adapter_identity_drift",
) -> tuple[str, tuple[int, int, int]]:
    digest, fingerprint = _stable_regular_digest(
        path,
        expected_bytes=expected_bytes,
        code=code,
    )
    return digest, fingerprint[:3]


def _owned_path_identity(
    path: Path,
    *,
    require_directory: bool,
    code: str,
) -> tuple[int, int, int]:
    try:
        value = path.lstat()
    except OSError:
        _fail(code)
    expected_kind = stat.S_ISDIR if require_directory else stat.S_ISREG
    if (
        stat.S_ISLNK(value.st_mode)
        or not expected_kind(value.st_mode)
        or _is_reparse(value)
    ):
        _fail(code)
    return _object_identity(value)


def _windows_open_file_descriptor(
    path: Path,
    *,
    desired_access: int,
    share_mode: int,
) -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            os.path.abspath(os.fspath(path)),
            desired_access,
            share_mode,
            None,
            3,  # OPEN_EXISTING
            0x00200000 | 0x08000000,  # OPEN_REPARSE_POINT | SEQUENTIAL_SCAN
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle is None or int(handle) == invalid:
            return None
        try:
            return msvcrt.open_osfhandle(
                int(handle),
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
        except OSError:
            kernel32.CloseHandle(handle)
            return None
    except (ImportError, OSError, TypeError, ValueError):
        return None


def _windows_set_delete_disposition(descriptor: int) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        information = FileDispositionInfo(True)
        return bool(
            set_information(
                msvcrt.get_osfhandle(descriptor),
                4,  # FileDispositionInfo
                ctypes.byref(information),
                ctypes.sizeof(information),
            )
        )
    except (ImportError, OSError, TypeError, ValueError):
        return False


@dataclass
class _WindowsReadLease:
    path: Path
    descriptor: int
    fingerprint: tuple[int, int, int, int, int, int]
    sha256: str
    bytes: int
    code: str
    closed: bool = False

    def close(self) -> bool:
        if self.closed:
            return True
        try:
            os.close(self.descriptor)
        except OSError:
            return False
        self.closed = True
        return True


def _descriptor_sha256(
    descriptor: int,
) -> tuple[str, int, tuple[int, int, int, int, int, int]]:
    before = os.fstat(descriptor)
    if not _regular_stat_valid(before) or before.st_size <= 0:
        raise OSError("descriptor_not_regular")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    observed_bytes = 0
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
        observed_bytes += len(chunk)
    after = os.fstat(descriptor)
    fingerprint = _stat_fingerprint(before)
    if (
        _stat_fingerprint(after) != fingerprint
        or not _regular_stat_valid(after)
        or observed_bytes != before.st_size
    ):
        raise OSError("descriptor_identity_drift")
    return digest.hexdigest(), observed_bytes, fingerprint


def _acquire_windows_read_lease(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    expected_identity: tuple[int, int, int] | None,
    code: str,
) -> _WindowsReadLease:
    if os.name != "nt":
        _fail(code)
    descriptor: int | None = None
    try:
        path_before = path.lstat()
        if not _regular_stat_valid(path_before):
            _fail(code)
        descriptor = _windows_open_file_descriptor(
            path,
            desired_access=0x80000000,  # GENERIC_READ
            share_mode=0x00000001,  # FILE_SHARE_READ only
        )
        if descriptor is None:
            _fail(code)
        observed_sha256, observed_bytes, fingerprint = _descriptor_sha256(descriptor)
        path_after = path.lstat()
        if (
            not _regular_stat_valid(path_after)
            or _stat_fingerprint(path_before) != fingerprint
            or _stat_fingerprint(path_after) != fingerprint
            or (expected_identity is not None and fingerprint[:3] != expected_identity)
            or observed_sha256 != expected_sha256
            or (expected_bytes is not None and observed_bytes != expected_bytes)
        ):
            _fail(code)
        return _WindowsReadLease(
            path=path,
            descriptor=descriptor,
            fingerprint=fingerprint,
            sha256=observed_sha256,
            bytes=observed_bytes,
            code=code,
        )
    except GenerationEvalRuntimeError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except OSError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _fail(code)


def _verify_windows_read_lease(lease: _WindowsReadLease) -> None:
    if lease.closed or os.name != "nt":
        _fail(lease.code)
    try:
        observed_sha256, observed_bytes, fingerprint = _descriptor_sha256(
            lease.descriptor
        )
        path_after = lease.path.lstat()
    except OSError:
        _fail(lease.code)
    if (
        fingerprint != lease.fingerprint
        or not _regular_stat_valid(path_after)
        or _stat_fingerprint(path_after) != lease.fingerprint
        or observed_sha256 != lease.sha256
        or observed_bytes != lease.bytes
    ):
        _fail(lease.code)


def _release_owned_file(
    path: Path,
    expected_identity: tuple[int, int, int],
) -> bool:
    descriptor: int | None = None
    disposition_set = False
    try:
        path_before = path.lstat()
        if (
            not _regular_stat_valid(path_before)
            or _object_identity(path_before) != expected_identity
        ):
            return False
        descriptor = _windows_open_file_descriptor(
            path,
            desired_access=0x80000000 | 0x00010000,  # GENERIC_READ | DELETE
            share_mode=0x00000001,  # FILE_SHARE_READ only
        )
        if descriptor is None:
            return False
        opened = os.fstat(descriptor)
        path_after_open = path.lstat()
        if (
            not _regular_stat_valid(opened)
            or not _regular_stat_valid(path_after_open)
            or _object_identity(opened) != expected_identity
            or _object_identity(path_after_open) != expected_identity
        ):
            return False
        disposition_set = _windows_set_delete_disposition(descriptor)
    except OSError:
        return False
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                disposition_set = False
    if not disposition_set:
        return False
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _stable_small_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    code: str,
) -> bytes:
    raw, fingerprint = _stable_regular_bytes(
        path,
        code=code,
        max_bytes=_MAX_JSON_BYTES,
    )
    if (
        (expected_bytes is not None and len(raw) != expected_bytes)
        or len(raw) != fingerprint[3]
        or _sha256(raw) != expected_sha256
    ):
        _fail(code)
    return raw


def _stable_small_file_with_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    code: str,
) -> tuple[bytes, tuple[int, int, int]]:
    raw, fingerprint = _stable_regular_bytes(
        path,
        code=code,
        max_bytes=_MAX_JSON_BYTES,
    )
    if (
        (expected_bytes is not None and len(raw) != expected_bytes)
        or len(raw) != fingerprint[3]
        or _sha256(raw) != expected_sha256
    ):
        _fail(code)
    return raw, fingerprint[:3]


def _adapter_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("eval_adapter_receipt_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        _fail("eval_adapter_receipt_invalid")
    return path.as_posix()


def _assert_physical_directory_chain(root: Path, target: Path) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError:
        _fail("eval_adapter_receipt_invalid")
    current = root
    for part in ("", *relative.parts):
        if part:
            current = current / part
        try:
            value = current.lstat()
        except OSError:
            _fail("eval_adapter_identity_drift")
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
        ):
            _fail("eval_adapter_identity_drift")


def _training_source_identity(
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    direct = (
        "schema_version",
        "candidate_commit",
        "release_commit",
        "release_tree",
        "artifact_prefix",
        "historical_release_attested",
        "current_live_remote_reasserted",
        "consumer_preflight_receipt_sha256",
        "consumer_source_binding_sha256",
        "consumer_binding_physical_sha256",
        "consumer_binding_canonical_sha256",
        "tree_digest_sha256",
        "read_set_digest_sha256",
        "producer_inventory_sha256",
        "read_set_files",
        "final_files",
    )
    if any(key not in source_identity for key in direct):
        _fail("eval_source_identity_drift")
    return {
        **{key: source_identity[key] for key in direct},
        "read_set_blob_oids_sha256": source_identity.get(
            "read_set_blob_inventory_sha256"
        ),
        "payload_blob_oids_sha256": source_identity.get(
            "payload_blob_inventory_sha256"
        ),
    }


def _validate_adapter_config(raw: bytes, profile: str) -> None:
    value = _strict_json_bytes(raw, "eval_adapter_identity_drift")
    targets = value.get("target_modules")
    expected_targets = {
        "q_only": {"q_proj"},
        "o_only": {"o_proj"},
        "q_plus_micro_o": {"q_proj", "o_proj"},
    }[profile]
    expected_rank = {"q_only": 1024, "o_only": 64, "q_plus_micro_o": 1024}[profile]
    expected_alpha = {"q_only": 2048, "o_only": 128, "q_plus_micro_o": 2048}[profile]
    if (
        not isinstance(targets, list)
        or set(targets) != expected_targets
        or value.get("peft_type") != "LORA"
        or value.get("task_type") != "CAUSAL_LM"
        or value.get("bias") != "none"
        or value.get("lora_dropout") != 0.0
        or value.get("r") != expected_rank
        or value.get("lora_alpha") != expected_alpha
    ):
        _fail("eval_adapter_state_invalid")
    rank_pattern = value.get("rank_pattern") or {}
    alpha_pattern = value.get("alpha_pattern") or {}
    if profile == "q_plus_micro_o":
        if rank_pattern != {"o_proj": 64} or alpha_pattern != {"o_proj": 128}:
            _fail("eval_adapter_state_invalid")
    elif rank_pattern or alpha_pattern:
        _fail("eval_adapter_state_invalid")


def authenticate_adapter_run(
    config: Mapping[str, Any],
    *,
    receipt_path: str | Path | None,
    source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, AdapterIdentity]]:
    section = config["adapter_run"]
    expected_sha = section["receipt_sha256"]
    if receipt_path is None or expected_sha == "pending":
        _fail("eval_execute_requires_bound_adapter_receipt")
    if not _is_sha(expected_sha):
        _fail("eval_adapter_receipt_invalid")
    path = Path(os.path.abspath(Path(receipt_path).expanduser()))
    raw = _stable_small_file(
        path,
        expected_sha256=str(expected_sha),
        code="eval_adapter_receipt_invalid",
    )
    if (
        b"\r" in raw
        or not raw.endswith(b"\n")
        or _canonical_json(_strict_json_bytes(raw, "eval_adapter_receipt_invalid"))
        != raw
    ):
        _fail("eval_adapter_receipt_invalid")
    receipt = _strict_json_bytes(raw, "eval_adapter_receipt_invalid")
    sidecar_raw = _stable_small_file(
        path.with_name(path.name + ".sha256"),
        expected_sha256=_sha256(f"{expected_sha}  {path.name}\n".encode("ascii")),
        expected_bytes=len(f"{expected_sha}  {path.name}\n".encode("ascii")),
        code="eval_adapter_receipt_invalid",
    )
    if sidecar_raw != f"{expected_sha}  {path.name}\n".encode("ascii"):
        _fail("eval_adapter_receipt_invalid")
    _assert_body_free(receipt)
    if tuple(_schema(TRAINING_RUN_RECEIPT_SCHEMA_PATH).iter_errors(receipt)):
        _fail("eval_adapter_receipt_invalid")
    required = {
        "schema_version",
        "status",
        "run_id",
        "mode",
        "config",
        "source",
        "teacher_final",
        "tokenizer",
        "model",
        "phase_order",
        "phase_receipts",
        "adapters",
        "training_lineage",
        "launcher",
        "claims",
    }
    if (
        set(receipt) != required
        or receipt["schema_version"] != section["required_schema_version"]
        or receipt["status"] != section["required_status"]
        or receipt["mode"] != "full"
        or receipt["source"] != _training_source_identity(source_identity)
        or receipt["phase_order"] != list(TRAINED_ARMS)
    ):
        _fail("eval_adapter_receipt_invalid")
    teacher = receipt["teacher_final"]
    if (
        not isinstance(teacher, Mapping)
        or _contains_v1_identity(teacher)
        or not isinstance(teacher.get("schema_version"), str)
        or not str(teacher["schema_version"]).endswith(".v2")
        or any(
            not _is_sha(teacher.get(key))
            for key in (
                "binding_sha256",
                "binding_schema_sha256",
                "manifest_sha256",
                "record_schema_sha256",
                "release_receipt_sha256",
                "release_attestation_sha256",
                "shard_inventory_sha256",
                "public_identity_sha256",
            )
        )
        or teacher.get("records") != 3520
        or teacher.get("main_records") != 3440
        or teacher.get("router_records") != 80
        or teacher.get("train_identity_bundles") != 20
        or teacher.get("train_identity_records") != 100
        or teacher.get("independent_identity_eval_bundles_not_in_optimizer") != 10
        or teacher.get("independent_identity_eval_probes_not_in_optimizer") != 50
        or teacher.get("accepted") != 3520
        or teacher.get("rejected") != 0
        or teacher.get("uncertain") != 0
        or teacher.get("producer_original_target_fallback") is not False
    ):
        _fail("eval_adapter_receipt_invalid")
    receipt_tokenizer = receipt["tokenizer"]
    if (
        not isinstance(receipt_tokenizer, Mapping)
        or receipt_tokenizer.get("chat_template_policy_sha256")
        != config["tokenizer"]["chat_template_policy_sha256"]
        or receipt_tokenizer.get("processor_source")
        != "authenticated_model_proto_bytes"
        or receipt_tokenizer.get("second_model_file_path_read") is not False
        or receipt_tokenizer.get("hf_auto_tokenizer_used") is not False
        or receipt_tokenizer.get("runtime_special_token_overlay") is not True
    ):
        _fail("eval_adapter_receipt_invalid")
    claims = receipt["claims"]
    if claims != {
        "diagnostic_only": True,
        "proxy_only": True,
        "formal": False,
        "release": False,
        "sample_bodies_included": False,
        "raw_token_ids_included": False,
        "producer_original_targets_used": False,
        "physical_kv_validated": False,
        "full_generation_kv_shared": False,
    }:
        _fail("eval_adapter_receipt_invalid")
    model = receipt["model"]
    if not isinstance(model, Mapping) or set(model) != {
        "base_files",
        "base_file_inventory_sha256",
        "phase_base_hashes",
        "phase_q8_scb_hashes",
        "all_phase_base_hashes_equal",
        "all_phase_q8_scb_hashes_equal",
        "frozen_q8_base",
        "base_o_proj_always_frozen",
    }:
        _fail("eval_adapter_receipt_invalid")
    base_files = model["base_files"]
    if (
        not isinstance(base_files, Mapping)
        or set(base_files)
        != {
            "model.safetensors",
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "EXPORT_MANIFEST.json",
        }
        or not all(_is_sha(value) for value in base_files.values())
        or _sha256(_canonical_json(dict(base_files)))
        != model["base_file_inventory_sha256"]
        or set(model["phase_base_hashes"]) != set(TRAINED_ARMS)
        or set(model["phase_q8_scb_hashes"]) != set(TRAINED_ARMS)
        or not all(_is_sha(value) for value in model["phase_base_hashes"].values())
        or not all(_is_sha(value) for value in model["phase_q8_scb_hashes"].values())
        or len(set(model["phase_base_hashes"].values())) != 1
        or len(set(model["phase_q8_scb_hashes"].values())) != 1
        or model["all_phase_base_hashes_equal"] is not True
        or model["all_phase_q8_scb_hashes_equal"] is not True
        or model["frozen_q8_base"] is not True
        or model["base_o_proj_always_frozen"] is not True
    ):
        _fail("eval_adapter_receipt_invalid")
    phase_receipts = receipt["phase_receipts"]
    if not isinstance(phase_receipts, list) or len(phase_receipts) != len(TRAINED_ARMS):
        _fail("eval_adapter_receipt_invalid")
    phase_by_arm: dict[str, Mapping[str, Any]] = {}
    for arm, item in zip(TRAINED_ARMS, phase_receipts, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "arm",
                "phase",
                "status",
                "target_steps",
                "path",
                "sha256",
                "bytes",
                "q8_scb_sha256",
                "base_parameter_sha256",
                "tensor_inventory_sha256",
            }
            or item.get("arm") != arm
            or item.get("phase") != "full"
            or item.get("status") != "passed"
            or item.get("target_steps") != multiarm.FULL_STEPS[arm]
            or item.get("path") != f"phases/{arm}/receipt.json"
            or not _is_sha(item.get("sha256"))
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or item.get("q8_scb_sha256") != model["phase_q8_scb_hashes"][arm]
            or item.get("base_parameter_sha256") != model["phase_base_hashes"][arm]
            or not _is_sha(item.get("tensor_inventory_sha256"))
        ):
            _fail("eval_adapter_receipt_invalid")
        phase_by_arm[arm] = item
    raw_adapters = receipt["adapters"]
    if not isinstance(raw_adapters, Mapping) or set(raw_adapters) != set(TRAINED_ARMS):
        _fail("eval_adapter_receipt_invalid")
    adapters: dict[str, AdapterIdentity] = {}
    receipt_root = Path(os.path.abspath(path.parent))
    _assert_physical_directory_chain(receipt_root, receipt_root)
    for arm in TRAINED_ARMS:
        item = raw_adapters[arm]
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "adapter_profile",
            "trainable_parameters",
            "adapter_config",
            "adapter_model",
            "artifact_tree_sha256",
            "tensor_inventory_sha256",
        }:
            _fail("eval_adapter_receipt_invalid")
        expected_profile = multiarm.ARM_PROFILE[arm]
        if (
            item.get("adapter_profile") != expected_profile
            or item.get("trainable_parameters")
            != multiarm.PROFILE_PARAMETER_COUNTS[expected_profile]
            or item.get("path") != f"adapters/{arm}"
            or item.get("tensor_inventory_sha256")
            != phase_by_arm[arm]["tensor_inventory_sha256"]
        ):
            _fail("eval_adapter_receipt_invalid")
        relative = _adapter_relative(item["path"])
        adapter_path = receipt_root.joinpath(*PurePosixPath(relative).parts)
        _assert_physical_directory_chain(receipt_root, adapter_path)
        config_pin = item["adapter_config"]
        model_pin = item["adapter_model"]
        if (
            not isinstance(config_pin, Mapping)
            or not isinstance(model_pin, Mapping)
            or set(config_pin) != {"path", "sha256", "bytes"}
            or set(model_pin) != {"path", "sha256", "bytes"}
            or config_pin.get("path") != "adapter_config.json"
            or model_pin.get("path") != "adapter_model.safetensors"
            or not _is_sha(config_pin.get("sha256"))
            or not _is_sha(model_pin.get("sha256"))
            or not isinstance(config_pin.get("bytes"), int)
            or not isinstance(model_pin.get("bytes"), int)
            or config_pin["bytes"] <= 0
            or model_pin["bytes"] <= 0
            or not _is_sha(item.get("artifact_tree_sha256"))
            or not _is_sha(item.get("tensor_inventory_sha256"))
        ):
            _fail("eval_adapter_receipt_invalid")
        config_path = adapter_path / "adapter_config.json"
        model_path = adapter_path / "adapter_model.safetensors"
        config_raw, config_object_identity = _stable_small_file_with_identity(
            config_path,
            expected_sha256=str(config_pin["sha256"]),
            expected_bytes=int(config_pin["bytes"]),
            code="eval_adapter_identity_drift",
        )
        model_sha256, model_object_identity = _file_sha256_with_identity(
            model_path,
            expected_bytes=int(model_pin["bytes"]),
        )
        if model_sha256 != model_pin["sha256"]:
            _fail("eval_adapter_identity_drift")
        _validate_adapter_config(config_raw, expected_profile)
        observed_artifact = _sha256(
            _canonical_json(
                [
                    dict(config_pin),
                    dict(model_pin),
                ]
            )
        )
        if observed_artifact != item["artifact_tree_sha256"]:
            _fail("eval_adapter_identity_drift")
        identity = AdapterIdentity(
            arm=arm,
            profile=expected_profile,
            path=adapter_path,
            trainable_parameters=int(item["trainable_parameters"]),
            adapter_config_sha256=str(config_pin["sha256"]),
            adapter_config_bytes=int(config_pin["bytes"]),
            adapter_config_object_identity=config_object_identity,
            adapter_model_sha256=str(model_pin["sha256"]),
            adapter_model_bytes=int(model_pin["bytes"]),
            adapter_model_object_identity=model_object_identity,
            artifact_tree_sha256=str(item["artifact_tree_sha256"]),
            tensor_inventory_sha256=str(item["tensor_inventory_sha256"]),
        )
        adapters[arm] = identity
    return receipt, _sha256(raw), adapters


def authenticate_inputs(
    config: Mapping[str, Any],
    *,
    source_repo: str | Path,
    adapter_receipt_path: str | Path | None,
    model_root: str | Path | None = None,
    runtime_source: AuthenticatedRuntimeSource | None = None,
) -> tuple[AuthenticatedContext, AuthenticatedRuntimeSource]:
    source = runtime_source or _authenticate_shared_runtime_source(
        config,
        source_repo=source_repo,
    )
    source_identity, source_binding_sha, logical_identity = _validate_runtime_source(
        config,
        source,
    )
    adapter_receipt, adapter_receipt_sha, adapters = authenticate_adapter_run(
        config,
        receipt_path=adapter_receipt_path,
        source_identity=source_identity,
    )
    configured_model = config["model"]
    selected_root = (
        Path(model_root).expanduser()
        if model_root is not None
        else Path(
            os.environ.get(
                str(configured_model["environment_variable"]),
                str(configured_model["default_path"]),
            )
        ).expanduser()
    )
    if not selected_root.is_absolute():
        selected_root = ROOT / selected_root
    selected_root = Path(os.path.abspath(selected_root))
    model_receipt = adapter_receipt["model"]
    base_model = {
        "path": str(selected_root),
        "files": dict(model_receipt["base_files"]),
        "inventory_sha256": model_receipt["base_file_inventory_sha256"],
    }
    (
        tokenizer,
        tokenizer_capability,
        base_model_object_identities,
    ) = _authenticate_shared_tokenizer(
        config,
        base_model,
    )
    if adapter_receipt["tokenizer"] != tokenizer_capability.public_identity():
        _fail("eval_tokenizer_identity_drift")
    adapter_inventory = _sha256(
        _canonical_json(
            [
                {
                    "arm": identity.arm,
                    "profile": identity.profile,
                    "trainable_parameters": identity.trainable_parameters,
                    "artifact_tree_sha256": identity.artifact_tree_sha256,
                    "tensor_inventory_sha256": identity.tensor_inventory_sha256,
                }
                for identity in sorted(
                    adapters.values(),
                    key=lambda item: item.arm,
                )
            ]
        )
    )
    q8_scb_inventory_sha256 = next(iter(model_receipt["phase_q8_scb_hashes"].values()))
    base_parameter_identity_sha256 = next(
        iter(model_receipt["phase_base_hashes"].values())
    )
    context = AuthenticatedContext(
        config_sha256=str(config["_config_sha256"]),
        consumer_receipt_sha256=source.consumer_receipt_sha256,
        source_binding_sha256=source_binding_sha,
        source_identity=source_identity,
        logical_identity=logical_identity,
        adapter_run_receipt_sha256=adapter_receipt_sha,
        adapter_inventory_sha256=adapter_inventory,
        base_model=dict(base_model),
        base_model_object_identities=base_model_object_identities,
        base_parameter_identity_sha256=base_parameter_identity_sha256,
        q8_scb_inventory_sha256=q8_scb_inventory_sha256,
        adapters=adapters,
        tokenizer=tokenizer,
        tokenizer_capability=tokenizer_capability,
    )
    return context, source


def _validate_runtime_source(
    config: Mapping[str, Any],
    source: AuthenticatedRuntimeSource,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    source_identity = dict(source.public_identity())
    _assert_body_free(source_identity)
    historical = config["historical_source"]
    expected_source = {
        "candidate_commit": historical["candidate_commit"],
        "candidate_tree": historical["candidate_tree"],
        "release_commit": historical["release_commit"],
        "release_tree": historical["release_tree"],
        "read_set_files": 88,
        "payload_files": 23,
        "physical_files": 46,
        "final_files": 46,
        "historical_release_attested": True,
        "current_live_remote_reasserted": False,
        "artifact_prefix": historical["payload_root"],
        "consumer_binding_physical_sha256": config["consumer"]["binding_sha256"],
        "consumer_preflight_receipt_sha256": config["consumer"]["receipt_sha256"],
        "consumer_source_binding_sha256": config["consumer"]["source_binding_sha256"],
    }
    if any(source_identity.get(key) != value for key, value in expected_source.items()):
        _fail("eval_source_identity_drift")
    for key in (
        "tree_digest_sha256",
        "read_set_digest_sha256",
        "producer_inventory_sha256",
        "read_set_blob_inventory_sha256",
        "payload_blob_inventory_sha256",
    ):
        if not _is_sha(source_identity.get(key)):
            _fail("eval_source_identity_drift")
    asset_blobs = source_identity.get("asset_blobs")
    if not isinstance(asset_blobs, Mapping) or len(asset_blobs) != 23:
        _fail("eval_source_identity_drift")
    required_payloads = {
        str(item["payload_path"]) for item in config["assets"].values()
    }
    if not required_payloads.issubset(asset_blobs):
        _fail("eval_source_identity_drift")
    for name, item in asset_blobs.items():
        if (
            not isinstance(name, str)
            or not isinstance(item, Mapping)
            or set(item)
            != {
                "commit",
                "tree",
                "path",
                "blob_oid",
                "bytes",
                "sha256",
            }
            or item.get("commit") != historical["release_commit"]
            or item.get("tree") != historical["release_tree"]
            or item.get("path") != f"{historical['payload_root']}/{name}"
            or not isinstance(item.get("blob_oid"), str)
            or _OID_RE.fullmatch(str(item["blob_oid"])) is None
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or not _is_sha(item.get("sha256"))
        ):
            _fail("eval_source_identity_drift")
    source_binding_sha = source.source_binding_sha256
    logical_identity = dict(source.logical_identity)
    logical_keys = {
        "eval_proxy_partition_sha256",
        "identity_probe_inventory_sha256",
        "logical_all_record_inventory_sha256",
        "logical_dataset_sha256",
        "logical_record_order_sha256",
        "logical_target_inventory_sha256",
        "logical_task_bundle_inventory_sha256",
        "logical_train_partition_sha256",
        "logical_train_record_inventory_sha256",
        "serialization_inventory_sha256",
    }
    if (
        source.consumer_receipt_sha256 != config["consumer"]["receipt_sha256"]
        or source_binding_sha != config["consumer"]["source_binding_sha256"]
        or set(logical_identity) != logical_keys
        or not all(_is_sha(value) for value in logical_identity.values())
    ):
        _fail("eval_source_identity_drift")
    return source_identity, source_binding_sha, logical_identity


def _authenticate_shared_runtime_source(
    config: Mapping[str, Any],
    *,
    source_repo: str | Path,
) -> AuthenticatedRuntimeSource:
    """Delegate all immutable P/R/blob/tokenizer source gating to training."""

    try:
        authenticated = runtime_source_v2.authenticate_producer_source(source_repo)
        candidate_tree = runtime_source_v2.GitObjectReader(
            authenticated.repository
        ).tree_for_commit(authenticated.candidate_commit)
        normalized, receipt_sha, source_binding_sha = (
            multiarm.load_consumer_preflight_receipt(
                _repo_local_path(config["consumer"]["receipt_path"])
            )
        )
        if (
            receipt_sha != config["consumer"]["receipt_sha256"]
            or source_binding_sha != config["consumer"]["source_binding_sha256"]
            or authenticated.consumer_preflight_receipt_sha256 != receipt_sha
            or authenticated.consumer_source_binding_sha256 != source_binding_sha
        ):
            _fail("eval_source_identity_drift")
        source: AuthenticatedRuntimeSource = SharedRuntimeSource(
            source=authenticated,
            logical_identity=dict(normalized["logical_identity"]),
            candidate_tree=candidate_tree,
        )
    except GenerationEvalRuntimeError:
        raise
    except Exception:
        _fail("eval_historical_source_invalid")
    return source


def _authenticate_shared_tokenizer(
    config: Mapping[str, Any],
    base_model: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    runtime_source_v2.AuthenticatedTokenizer,
    dict[str, tuple[int, int, int]],
]:
    section = config["tokenizer"]
    model_root = base_model.get("path")
    if not isinstance(model_root, str) or not model_root:
        _fail("eval_base_identity_drift")
    files = base_model.get("files")
    if not isinstance(files, Mapping) or set(files) != set(BASE_MODEL_FILES):
        _fail("eval_base_identity_drift")
    root = Path(model_root)
    leases: list[_WindowsReadLease] = []
    try:
        for name in TOKENIZER_LEASE_FILES:
            leases.append(
                _acquire_windows_read_lease(
                    root.joinpath(*PurePosixPath(name).parts),
                    expected_sha256=str(files[name]),
                    expected_bytes=None,
                    expected_identity=None,
                    code="eval_tokenizer_identity_drift",
                )
            )
        try:
            tokenizer = runtime_source_v2.authenticate_tokenizer_snapshot(model_root)
        except Exception:
            _fail("eval_tokenizer_identity_drift")
        for lease in leases:
            _verify_windows_read_lease(lease)
        object_identities = {lease.path.name: lease.fingerprint[:3] for lease in leases}
        object_identities["model.safetensors"] = _owned_path_identity(
            root / "model.safetensors",
            require_directory=False,
            code="eval_base_identity_drift",
        )
        if len(leases) != 4 or set(object_identities) != set(BASE_MODEL_FILES):
            _fail("eval_tokenizer_identity_drift")
    finally:
        leases_closed = True
        for lease in reversed(leases):
            leases_closed = lease.close() and leases_closed
        if not leases_closed:
            _fail("eval_tokenizer_identity_drift")
    public = dict(tokenizer.public_identity())
    if (
        public.get("chat_template_policy_sha256")
        != section["chat_template_policy_sha256"]
        or public.get("processor_source") != section["processor_source"]
        or public.get("second_model_file_path_read")
        is not section["second_model_file_path_read"]
        or public.get("hf_auto_tokenizer_used") is not section["hf_auto_tokenizer_used"]
        or public.get("runtime_special_token_overlay") is not True
    ):
        _fail("eval_tokenizer_identity_drift")
    return (
        {
            **public,
            "authority": section["authority"],
            "runtime_special_token_ids": dict(section["runtime_special_token_overlay"]),
        },
        tokenizer,
        object_identities,
    )


def build_status(config: Mapping[str, Any]) -> dict[str, Any]:
    pending = config["adapter_run"]["receipt_sha256"] == "pending"
    return {
        "schema_version": CONFIG_VERSION,
        "status": (
            "blocked_waiting_for_bound_adapter_run_receipt"
            if pending
            else "bound_metadata_validate_ready"
        ),
        "default_mode": config["default_mode"],
        "execute_requires_explicit_flag": True,
        "windows_handle_leases_required_for_physical_execution": True,
        "tokenizer_auth_read_leases": 4,
        "runtime_input_read_leases": 23,
        "historical_git_objects_only": True,
        "body_reads": 0,
        "raw_token_ids_read": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
    }


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_count: int
    eos_terminated: bool
    latency_ms: float
    active_adapter: str | None


class GenerationBackend(Protocol):
    def start(self, context: AuthenticatedContext) -> Mapping[str, Any]: ...

    def generate(
        self,
        record: Mapping[str, Any],
        *,
        arm: str,
        adapter_name: str | None,
        max_new_tokens: int,
        seed: int,
    ) -> GenerationResult: ...

    def finish(self) -> Mapping[str, Any]: ...


def _record_target(record: Mapping[str, Any]) -> str:
    target = record.get("target")
    if isinstance(target, str) and target:
        return target
    if isinstance(target, Mapping):
        for key in ("assistant_text", "text", "expected_text"):
            value = target.get(key)
            if isinstance(value, str) and value:
                return value
    route_target = record.get("route_plan_target")
    if isinstance(route_target, Mapping):
        return json.dumps(
            route_target,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    messages = record.get("messages")
    if (
        isinstance(messages, list)
        and messages
        and isinstance(messages[-1], Mapping)
        and messages[-1].get("role") in {"assistant", "model"}
        and isinstance(messages[-1].get("content"), str)
    ):
        return str(messages[-1]["content"])
    _fail("eval_record_contract_invalid")


def _record_prompt(record: Mapping[str, Any]) -> str:
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        _fail("eval_record_contract_invalid")
    usable = messages
    if isinstance(messages[-1], Mapping) and messages[-1].get("role") in {
        "assistant",
        "model",
    }:
        usable = messages[:-1]
    if (
        len(usable) == 2
        and isinstance(usable[0], Mapping)
        and isinstance(usable[1], Mapping)
        and usable[0].get("role") == "system"
        and usable[1].get("role") == "user"
        and isinstance(usable[0].get("content"), str)
        and isinstance(usable[1].get("content"), str)
    ):
        system = str(usable[0]["content"])
        user = str(usable[1]["content"])
        if record.get(
            "namespace"
        ) == "gemma3_chat_emotion_router_qonly_v1" or isinstance(
            record.get("serialization"), Mapping
        ):
            prompt = f"System instructions:\n{system}\n\nUser request:\n{user}"
            declared = record.get("serialization")
            if isinstance(declared, Mapping) and declared.get(
                "prompt_sha256"
            ) != _sha256(prompt.encode("utf-8")):
                _fail("eval_record_contract_invalid")
            return prompt
        prompt = f"{system}\n\n{user}"
        declared = record.get("gemma_serialization")
        visible = (
            f"<start_of_turn>user\n{prompt}<end_of_turn><eos>\n<start_of_turn>model\n"
        )
        if isinstance(declared, Mapping) and declared.get("prompt_sha256") != _sha256(
            visible.encode("utf-8")
        ):
            _fail("eval_record_contract_invalid")
        return prompt
    if (
        len(usable) == 1
        and isinstance(usable[0], Mapping)
        and usable[0].get("role") == "user"
        and isinstance(usable[0].get("content"), str)
    ):
        return str(usable[0]["content"])
    parts = []
    for item in usable:
        if (
            not isinstance(item, Mapping)
            or item.get("role") not in {"system", "user"}
            or not isinstance(item.get("content"), str)
        ):
            _fail("eval_record_contract_invalid")
        parts.append(f"{item['role']}:{item['content']}")
    if not parts:
        _fail("eval_record_contract_invalid")
    return "\n".join(parts)


def _json_mapping(value: str) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _find_first(value: Mapping[str, Any], names: set[str]) -> object | None:
    for key, child in value.items():
        if key.casefold() in names:
            return child
        if isinstance(child, Mapping):
            found = _find_first(child, names)
            if found is not None:
                return found
    return None


def _grounding_scalars(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(
            scalar for child in value.values() for scalar in _grounding_scalars(child)
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(scalar for child in value for scalar in _grounding_scalars(child))
    if isinstance(value, (str, int, float, bool)):
        return (str(value),)
    return ()


def _has_affirmative_identity_claim(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> bool:
    for pattern in patterns:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 96) : match.start()]
            clause_prefix = re.split(r"[.!?;。！？；\n]", prefix)[-1]
            if _NEGATED_CLAIM_PREFIX_RE.search(clause_prefix) is None:
                return True
    return False


def score_generation(
    record: Mapping[str, Any],
    result: GenerationResult,
) -> dict[str, bool]:
    """Score one in-memory result; no bodies or token IDs leave this call."""

    target_text = _record_target(record)
    prompt_text = _record_prompt(record)
    generated_json = _json_mapping(result.text)
    expected_json = _json_mapping(target_text)
    explicit_route = record.get("selected_expert", record.get("route"))
    tool_name_names = {"tool", "tool_name", "name", "operation"}
    argument_names = {"arguments", "args", "parameters"}
    planner_label_names = {"label", "task_class", "selected_expert", "route"}
    topology_names = {"topology", "plan", "steps", "edges", "dependencies"}
    generated_tool = (
        _find_first(generated_json, tool_name_names)
        if generated_json is not None
        else None
    )
    expected_tool = (
        _find_first(expected_json, tool_name_names)
        if expected_json is not None
        else None
    )
    generated_args = (
        _find_first(generated_json, argument_names)
        if generated_json is not None
        else None
    )
    expected_args = (
        _find_first(expected_json, argument_names)
        if expected_json is not None
        else None
    )
    generated_label = (
        _find_first(generated_json, planner_label_names)
        if generated_json is not None
        else None
    )
    expected_label = (
        _find_first(expected_json, planner_label_names)
        if expected_json is not None
        else None
    )
    route = explicit_route if explicit_route is not None else expected_label
    generated_topology = (
        _find_first(generated_json, topology_names)
        if generated_json is not None
        else None
    )
    expected_topology = (
        _find_first(expected_json, topology_names)
        if expected_json is not None
        else None
    )
    argument_values = _grounding_scalars(generated_args)
    out_of_bounds = (
        not result.text.strip()
        or len(result.text) > 16_384
        or result.token_count < 0
        or any(ord(char) < 32 and char not in "\n\r\t" for char in result.text)
    )
    format_valid = (
        generated_json is not None
        if expected_json is not None
        else bool(result.text.strip())
    )
    return {
        "json_structure_valid": (
            generated_json is not None
            and expected_json is not None
            and set(generated_json) == set(expected_json)
        ),
        "format_valid": format_valid,
        "tool_name_exact": expected_tool is not None
        and generated_tool == expected_tool,
        "tool_arguments_exact": (
            expected_args is not None and generated_args == expected_args
        ),
        "tool_arguments_grounded": (
            generated_args is not None
            and bool(argument_values)
            and all(value in prompt_text for value in argument_values)
        ),
        "planner_label_exact": (
            expected_label is not None and generated_label == expected_label
        ),
        "planner_topology_exact": (
            expected_topology is not None and generated_topology == expected_topology
        ),
        "identity_air_attribution": _has_affirmative_identity_claim(
            result.text,
            _AIR_ATTRIBUTION_PATTERNS,
        ),
        "identity_provider_misattribution_absent": (
            not _has_affirmative_identity_claim(
                result.text,
                _PROVIDER_MISATTRIBUTION_PATTERNS,
            )
        ),
        "route_exact": route is not None and generated_label == route,
        "refusal_absent": _REFUSAL_RE.search(result.text) is None,
        "out_of_bounds_absent": not out_of_bounds,
        "eos_terminated": result.eos_terminated,
    }


@dataclass
class Aggregate:
    records: int = 0
    successes: Counter[str] = field(default_factory=Counter)
    token_count: int = 0
    latency_ms_sum: Decimal = field(default_factory=lambda: Decimal(0))

    def add(
        self,
        score: Mapping[str, bool],
        *,
        token_count: int,
        latency_ms: float,
    ) -> None:
        if (
            token_count < 0
            or not math.isfinite(latency_ms)
            or latency_ms < 0
            or set(score) != set(METRICS)
        ):
            _fail("eval_output_invalid")
        self.records += 1
        self.token_count += token_count
        self.latency_ms_sum += Decimal(str(latency_ms))
        for name, passed in score.items():
            if passed:
                self.successes[name] += 1

    def report(self) -> dict[str, Any]:
        denominator = self.records
        metrics = {
            name: {
                "passed": self.successes[name],
                "total": denominator,
                "rate": (
                    "0.000000000000"
                    if denominator == 0
                    else str(
                        (Decimal(self.successes[name]) / Decimal(denominator)).quantize(
                            Decimal("0.000000000001"),
                            rounding=ROUND_HALF_EVEN,
                        )
                    )
                ),
            }
            for name in METRICS
        }
        latency = (
            Decimal(0)
            if denominator == 0
            else self.latency_ms_sum / Decimal(denominator)
        )
        return {
            "records": denominator,
            "generated_tokens": self.token_count,
            "latency_ms_mean": str(
                latency.quantize(
                    Decimal("0.000001"),
                    rounding=ROUND_HALF_EVEN,
                )
            ),
            "metrics": metrics,
        }


def _record_identity(record: Mapping[str, Any]) -> str:
    value = record.get("record_id", record.get("id"))
    if not isinstance(value, str) or not value:
        return _domain_sha256(
            "anchor.gemma3-chat-generation-eval-record-fallback.v2",
            record,
        )
    return value


def _record_strata(
    record: Mapping[str, Any],
    *,
    asset: str,
    wrong_route: bool,
) -> dict[str, str]:
    fallback_role = (
        "tool_call"
        if asset == "tool"
        else "planner_router"
        if asset in {"planner", "router"}
        else "unknown"
    )
    role = str(record.get("role", fallback_role))
    language = str(record.get("language", "unknown"))
    stratum = str(
        record.get(
            "stratum",
            record.get("comparison_stratum", "unspecified"),
        )
    )
    if role not in {*ROLE_ADAPTER, "planner_router"}:
        _fail("eval_record_contract_invalid")
    if language not in {"en", "zh-CN"}:
        _fail("eval_record_contract_invalid")
    if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,63}", stratum) is None:
        _fail("eval_record_contract_invalid")
    return {
        "stratum": stratum,
        "language": language,
        "role": role,
        "identity": (
            "identity_probe" if bool(record.get("identity_probe")) else "ordinary"
        ),
        "router": "router_eval" if asset == "router" else "not_router",
        "wrong_route": "wrong_route" if wrong_route else "correct_or_base",
    }


def _adapter_for(
    asset: str,
    arm: str,
    record: Mapping[str, Any],
) -> str | None:
    if arm.endswith("_base") or arm == "identity_base":
        return None
    if arm in TRAINED_ARMS:
        return arm
    role = str(record.get("role", "humor"))
    correct = ROLE_ADAPTER.get(role)
    if correct is None:
        _fail("eval_record_contract_invalid")
    if arm in {"correct_route", "identity_correct_route"}:
        return correct
    if arm in {"wrong_route_1", "identity_wrong_route"}:
        return WRONG_ROUTE_RING[correct][0]
    if arm == "wrong_route_2":
        return WRONG_ROUTE_RING[correct][1]
    _fail("eval_record_contract_invalid")


def _arm_sequence(asset: str) -> tuple[str, ...]:
    if asset == "tool":
        return TOOL_ARMS
    if asset == "planner":
        return PLANNER_ARMS
    if asset == "router":
        return ("planner_q_only",)
    if asset == "identity":
        return IDENTITY_ARMS
    if asset == "wrong_route":
        return ("correct_route", "wrong_route_1", "wrong_route_2")
    _fail("eval_config_invalid")


def build_fairness_lock(
    records: Sequence[Mapping[str, Any]],
    *,
    asset: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    order = [_record_identity(record) for record in records]
    serialization = [
        _domain_sha256(
            "anchor.gemma3-chat-generation-eval-serialization.v2",
            {
                "prompt": _record_prompt(record),
                "target": _record_target(record),
            },
        )
        for record in records
    ]
    execution = config["execution"]
    lock = {
        "asset": asset,
        "records": len(records),
        "record_order_sha256": _domain_sha256(
            "anchor.gemma3-chat-generation-eval-order.v2",
            order,
        ),
        "serialization_sha256": _domain_sha256(
            "anchor.gemma3-chat-generation-eval-serialization-order.v2",
            serialization,
        ),
        "sampling_sha256": _domain_sha256(
            "anchor.gemma3-chat-generation-eval-sampling.v2",
            {
                "seed": execution["seed"],
                "max_new_tokens": execution["max_new_tokens"],
                "do_sample": execution["do_sample"],
                "num_beams": execution["num_beams"],
                "temperature": execution["temperature"],
            },
        ),
        "seed": execution["seed"],
        "max_new_tokens": execution["max_new_tokens"],
    }
    return lock


def validate_fairness_locks(
    locks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    fields = {
        "records",
        "record_order_sha256",
        "serialization_sha256",
        "sampling_sha256",
        "seed",
        "max_new_tokens",
    }
    for values in locks.values():
        if not values:
            _fail("eval_fairness_lock_drift")
        baseline = {key: values[0].get(key) for key in fields}
        if any({key: value.get(key) for key in fields} != baseline for value in values):
            _fail("eval_fairness_lock_drift")


def _wrong_route_deltas(
    aggregates: Mapping[str, Mapping[str, Aggregate]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for asset, arms in aggregates.items():
        if "correct_route" not in arms:
            continue
        correct = arms["correct_route"]
        for arm in ("wrong_route_1", "wrong_route_2"):
            wrong = arms[arm]
            for metric in METRICS:
                correct_rate = Decimal(correct.successes[metric]) / Decimal(
                    max(1, correct.records)
                )
                wrong_rate = Decimal(wrong.successes[metric]) / Decimal(
                    max(1, wrong.records)
                )
                result[f"{asset}:{arm}:{metric}_delta"] = str(
                    (correct_rate - wrong_rate).quantize(
                        Decimal("0.000000000001"),
                        rounding=ROUND_HALF_EVEN,
                    )
                )
    return result


class RealGemmaBackend:
    """Lazy local-only Q8 + PEFT backend used only by explicit execution."""

    def __init__(self) -> None:
        self.model: Any = None
        self.base: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.bnb: Any = None
        self.q8_before: Mapping[str, Any] | None = None
        self.base_capture: Sequence[tuple[str, Any, str]] = ()
        self.qdiag: Any = None
        self.base_hashes: dict[str, str] = {}
        self.base_file_identities: dict[str, tuple[int, int, int]] = {}
        self.file_leases: dict[str, _WindowsReadLease] = {}
        self.context: AuthenticatedContext | None = None

    def start(self, context: AuthenticatedContext) -> Mapping[str, Any]:
        try:
            return self._start(context)
        except BaseException:
            self.close()
            raise

    def close(self) -> bool:
        closed = True
        for lease in reversed(tuple(self.file_leases.values())):
            closed = lease.close() and closed
        self.file_leases.clear()
        return closed

    def _lease(
        self,
        key: str,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int | None,
        expected_identity: tuple[int, int, int] | None,
        code: str,
    ) -> _WindowsReadLease:
        if key in self.file_leases:
            _fail(code)
        lease = _acquire_windows_read_lease(
            path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            expected_identity=expected_identity,
            code=code,
        )
        self.file_leases[key] = lease
        return lease

    def _start(self, context: AuthenticatedContext) -> Mapping[str, Any]:
        self.context = context
        base = context.base_model
        if set(base) != {"path", "files", "inventory_sha256"}:
            _fail("eval_base_identity_drift")
        model_root = Path(str(base["path"])).resolve()
        files = base["files"]
        if (
            not isinstance(files, Mapping)
            or set(files) != set(BASE_MODEL_FILES)
            or set(context.base_model_object_identities) != set(BASE_MODEL_FILES)
        ):
            _fail("eval_base_identity_drift")
        for name in BASE_MODEL_FILES:
            expected = files[name]
            relative = _safe_relative(name)
            path = model_root.joinpath(*PurePosixPath(relative).parts)
            lease = self._lease(
                f"base:{relative}",
                path,
                expected_sha256=str(expected),
                expected_bytes=None,
                expected_identity=context.base_model_object_identities[relative],
                code="eval_base_identity_drift",
            )
            self.base_hashes[relative] = lease.sha256
            self.base_file_identities[relative] = lease.fingerprint[:3]
        if _sha256(_canonical_json(dict(files))) != base["inventory_sha256"]:
            _fail("eval_base_identity_drift")

        for arm in TRAINED_ARMS:
            identity = context.adapters[arm]
            self._lease(
                f"adapter-config:{arm}",
                identity.path / "adapter_config.json",
                expected_sha256=identity.adapter_config_sha256,
                expected_bytes=identity.adapter_config_bytes,
                expected_identity=identity.adapter_config_object_identity,
                code="eval_adapter_identity_drift",
            )
            self._lease(
                f"adapter-model:{arm}",
                identity.path / "adapter_model.safetensors",
                expected_sha256=identity.adapter_model_sha256,
                expected_bytes=identity.adapter_model_bytes,
                expected_identity=identity.adapter_model_object_identity,
                code="eval_adapter_identity_drift",
            )

        if (
            len(self.file_leases) != 23
            or tuple(self.file_leases) != RUNTIME_INPUT_LEASE_KEYS
        ):
            _fail("eval_base_identity_drift")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["WANDB_DISABLED"] = "true"
        try:
            import bitsandbytes as bnb
            import peft
            import safetensors
            import torch
            import transformers
            from peft import PeftModel
            from safetensors import safe_open
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except Exception:
            _fail("eval_base_identity_drift")
        self.torch = torch
        self.bnb = bnb
        for arm in TRAINED_ARMS:
            identity = context.adapters[arm]
            expected_tensors = {
                str(item["name"]): tuple(int(value) for value in item["shape"])
                for item in multiarm.expected_tensor_metadata(arm)
            }
            observed_tensors: dict[str, tuple[int, ...]] = {}
            observed_parameters = 0
            try:
                with safe_open(
                    identity.path / "adapter_model.safetensors",
                    framework="pt",
                    device="cpu",
                ) as handle:
                    for saved_name in handle.keys():
                        match = _LORA_RE.search(saved_name)
                        if match is None:
                            _fail("eval_adapter_state_invalid")
                        layer, projection, factor = match.groups()
                        canonical = (
                            f"model.layers.{int(layer)}.self_attn.{projection}."
                            f"lora_{factor}.weight"
                        )
                        tensor = handle.get_tensor(saved_name)
                        shape = tuple(int(value) for value in tensor.shape)
                        if (
                            canonical in observed_tensors
                            or tensor.dtype != torch.bfloat16
                        ):
                            _fail("eval_adapter_state_invalid")
                        observed_tensors[canonical] = shape
                        observed_parameters += int(tensor.numel())
                        del tensor
            except GenerationEvalRuntimeError:
                raise
            except Exception:
                _fail("eval_adapter_state_invalid")
            if (
                observed_tensors != expected_tensors
                or observed_parameters != identity.trainable_parameters
            ):
                _fail("eval_adapter_state_invalid")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"],
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            ),
        )
        overlay = context.tokenizer["runtime_special_token_ids"]
        self.base.config.pad_token_id = overlay["pad_token_id"]
        self.base.config.eos_token_id = overlay["eos_token_id"]
        self.base.config.bos_token_id = overlay["bos_token_id"]
        self.base.config.use_cache = True
        try:
            from anchor_mvp.training import gemma3_chat_five_expert_qonly_v1 as chat
            from anchor_mvp.training import qwen_lora_diagnostic as qdiag

            chat._validate_q8_base_modules(self.base, bnb=bnb, torch=torch)
            chat._prepare_frozen_q8_base(self.base, torch=torch, bnb=bnb)
            self.q8_before = chat._q8_quant_state_inventory(
                self.base,
                bnb=bnb,
                torch=torch,
            )
            self.base_capture, base_parameter_sha256 = qdiag._capture_base_parameters(
                self.base, torch
            )
            self.qdiag = qdiag
        except Exception:
            _fail("eval_base_identity_drift")
        if (
            self.q8_before["sha256"] != context.q8_scb_inventory_sha256
            or base_parameter_sha256 != context.base_parameter_identity_sha256
        ):
            _fail("eval_base_identity_drift")
        first = TRAINED_ARMS[0]
        self.model = PeftModel.from_pretrained(
            self.base,
            context.adapters[first].path,
            adapter_name=first,
            is_trainable=False,
        )
        for arm in TRAINED_ARMS[1:]:
            self.model.load_adapter(
                context.adapters[arm].path,
                adapter_name=arm,
                is_trainable=False,
            )
        self.model.eval()
        peft_configs = getattr(self.model, "peft_config", None)
        if (
            not isinstance(peft_configs, Mapping)
            or set(peft_configs) != set(TRAINED_ARMS)
            or any(parameter.requires_grad for parameter in self.model.parameters())
        ):
            _fail("eval_adapter_state_invalid")
        self.processor = context.tokenizer_capability.processor
        return {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
            "safetensors": str(safetensors.__version__),
            "bitsandbytes": str(bnb.__version__),
            "adapter_tensor_inventories_validated": True,
            "runtime_input_read_leases": len(self.file_leases),
        }

    def _active(self) -> tuple[str, ...]:
        value = getattr(self.model, "active_adapters", None)
        if value is None:
            value = getattr(self.model, "active_adapter", None)
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence):
            return tuple(str(item) for item in value)
        return ()

    def generate(
        self,
        record: Mapping[str, Any],
        *,
        arm: str,
        adapter_name: str | None,
        max_new_tokens: int,
        seed: int,
    ) -> GenerationResult:
        torch = self.torch
        prompt = _record_prompt(record)
        target = _record_target(record)
        if self.context is None:
            _fail("eval_tokenizer_identity_drift")
        serialized = runtime_source_v2.serialize_authenticated_example(
            self.context.tokenizer_capability,
            prompt,
            target,
        )
        prompt_tokens = serialized.prompt_tokens
        if (
            prompt_tokens <= 0
            or prompt_tokens >= len(serialized.input_ids)
            or serialized.trainable_label_tokens <= 0
        ):
            _fail("eval_record_contract_invalid")
        prefix = serialized.input_ids[:prompt_tokens]
        input_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if adapter_name is None:
            adapter_context = self.model.disable_adapter()
        else:
            self.model.set_adapter(adapter_name)
            if self._active() != (adapter_name,):
                _fail("eval_adapter_state_invalid")
            if any(parameter.requires_grad for parameter in self.model.parameters()):
                _fail("eval_adapter_state_invalid")
            adapter_context = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "use_cache": True,
        }

        if adapter_context is None:
            with torch.inference_mode():
                output = self.model.generate(**generation_kwargs)
        else:
            with adapter_context:
                with torch.inference_mode():
                    output = self.model.generate(**generation_kwargs)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        ids = [
            int(value)
            for value in output[0, input_ids.shape[1] :].detach().cpu().tolist()
        ]
        eos = bool(ids and ids[-1] == 1)
        visible = list(ids)
        while visible and visible[-1] in {0, 1, 106}:
            visible.pop()
        text = self.processor.decode(visible).strip()
        token_count = len(ids)
        del prompt, target, serialized, prefix, input_ids, attention_mask, output, ids
        del generation_kwargs
        del visible
        return GenerationResult(
            text=text,
            token_count=token_count,
            eos_terminated=eos,
            latency_ms=latency_ms,
            active_adapter=adapter_name,
        )

    def finish(self) -> Mapping[str, Any]:
        try:
            result = self._finish()
        finally:
            if not self.close():
                _fail("eval_base_identity_drift")
        return result

    def _finish(self) -> Mapping[str, Any]:
        try:
            from anchor_mvp.training import gemma3_chat_five_expert_qonly_v1 as chat

            after = chat._q8_quant_state_inventory(
                self.model,
                bnb=self.bnb,
                torch=self.torch,
            )
        except Exception:
            _fail("eval_base_identity_drift")
        try:
            base_parameter_after = self.qdiag._assert_base_parameters_unchanged(
                self.base_capture,
                self.torch,
            )
        except Exception:
            _fail("eval_base_identity_drift")
        if (
            self.q8_before != after
            or self.context is None
            or base_parameter_after != self.context.base_parameter_identity_sha256
        ):
            _fail("eval_base_identity_drift")
        for lease in self.file_leases.values():
            _verify_windows_read_lease(lease)
        peak_allocated = int(self.torch.cuda.max_memory_allocated())
        peak_reserved = int(self.torch.cuda.max_memory_reserved())
        del self.model, self.base, self.processor
        self.torch.cuda.empty_cache()
        return {
            "q8_scb_identity_equal_before_after": True,
            "q8_scb_inventory_sha256": after["sha256"],
            "base_model_identity_equal_before_after": True,
            "base_parameter_identity_sha256": base_parameter_after,
            "adapter_file_identities_equal_before_after": True,
            "windows_read_leases_held_during_all_loads": True,
            "runtime_input_read_leases_verified_after_load": 23,
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_reserved_bytes": peak_reserved,
            "termination_checked_every_generation": True,
            "all_parameters_frozen": True,
            "adapter_switch_state_checked_every_generation": True,
        }


@contextmanager
def gpu_lock(path: Path, run_id: str) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _owned_path_identity(
        path.parent,
        require_directory=True,
        code="eval_gpu_lock_exists",
    )
    payload = {
        "schema_version": LOCK_VERSION,
        "run_id": run_id,
        "pid": os.getpid(),
        "gpu_index": 0,
        "concurrency": 1,
    }
    raw = _canonical_json(payload)
    digest = _sha256(raw)
    descriptor = -1
    lock_identity: tuple[int, int, int] | None = None
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError:
        _fail("eval_gpu_lock_exists")
    except OSError:
        _fail("eval_gpu_lock_exists")
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("eval_gpu_lock_exists")
            offset += written
        os.fsync(descriptor)
        locked_stat = os.fstat(descriptor)
        lock_identity = _object_identity(locked_stat)
        if (
            not stat.S_ISREG(locked_stat.st_mode)
            or locked_stat.st_size != len(raw)
            or _owned_path_identity(
                path.parent,
                require_directory=True,
                code="eval_gpu_lock_exists",
            )
            != parent_identity
            or _owned_path_identity(
                path,
                require_directory=False,
                code="eval_gpu_lock_exists",
            )
            != lock_identity
        ):
            _fail("eval_gpu_lock_exists")
        yield {**payload, "sha256": digest}
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if (
            lock_identity is None
            or _owned_path_identity(
                path.parent,
                require_directory=True,
                code="eval_gpu_lock_exists",
            )
            != parent_identity
            or not _release_owned_file(path, lock_identity)
        ):
            _fail("eval_gpu_lock_exists")


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_parent_identity: tuple[int, int, int] | None = None,
) -> str:
    _assert_body_free(value)
    raw = _canonical_json(value)
    digest = _sha256(raw)
    if expected_parent_identity is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _owned_path_identity(
        path.parent,
        require_directory=True,
        code="eval_atomic_destination_exists",
    )
    if (
        expected_parent_identity is not None
        and parent_identity != expected_parent_identity
    ):
        _fail("eval_atomic_destination_exists")
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_raw = f"{digest}  {path.name}\n".encode("ascii")
    for target in (path, sidecar):
        try:
            target.lstat()
        except FileNotFoundError:
            pass
        except OSError:
            _fail("eval_atomic_destination_exists")
        else:
            _fail("eval_atomic_destination_exists")
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    sidecar_staging = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    staging_identity: tuple[int, int, int] | None = None
    sidecar_staging_identity: tuple[int, int, int] | None = None
    published_identity: tuple[int, int, int] | None = None
    published_sidecar_identity: tuple[int, int, int] | None = None
    try:
        with staging.open("xb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            staging_identity = _object_identity(os.fstat(handle.fileno()))
        with sidecar_staging.open("xb", buffering=0) as handle:
            handle.write(sidecar_raw)
            handle.flush()
            os.fsync(handle.fileno())
            sidecar_staging_identity = _object_identity(os.fstat(handle.fileno()))
        os.link(sidecar_staging, sidecar, follow_symlinks=False)
        published_sidecar_identity = sidecar_staging_identity
        os.link(staging, path, follow_symlinks=False)
        published_identity = staging_identity
        if (
            _owned_path_identity(
                path.parent,
                require_directory=True,
                code="eval_atomic_destination_exists",
            )
            != parent_identity
            or _owned_path_identity(
                path,
                require_directory=False,
                code="eval_atomic_destination_exists",
            )
            != staging_identity
            or _owned_path_identity(
                sidecar,
                require_directory=False,
                code="eval_atomic_destination_exists",
            )
            != sidecar_staging_identity
            or _stable_small_file(
                path,
                expected_sha256=digest,
                expected_bytes=len(raw),
                code="eval_atomic_destination_exists",
            )
            != raw
            or _stable_small_file(
                sidecar,
                expected_sha256=_sha256(sidecar_raw),
                expected_bytes=len(sidecar_raw),
                code="eval_atomic_destination_exists",
            )
            != sidecar_raw
        ):
            _fail("eval_atomic_destination_exists")
        if (
            staging_identity is None
            or sidecar_staging_identity is None
            or not _release_owned_file(staging, staging_identity)
            or not _release_owned_file(
                sidecar_staging,
                sidecar_staging_identity,
            )
        ):
            _fail("eval_atomic_destination_exists")
        staging_identity = None
        sidecar_staging_identity = None
        published_identity = None
        published_sidecar_identity = None
    except OSError:
        _fail("eval_atomic_destination_exists")
    finally:
        if published_identity is not None:
            _release_owned_file(path, published_identity)
        if published_sidecar_identity is not None:
            _release_owned_file(sidecar, published_sidecar_identity)
        if staging_identity is not None:
            _release_owned_file(staging, staging_identity)
        if sidecar_staging_identity is not None:
            _release_owned_file(sidecar_staging, sidecar_staging_identity)
    return digest


def _claim_run_directory(path: Path) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _owned_path_identity(
        path.parent,
        require_directory=True,
        code="eval_atomic_destination_exists",
    )
    try:
        path.mkdir(exist_ok=False)
    except OSError:
        _fail("eval_atomic_destination_exists")
    identity = _owned_path_identity(
        path,
        require_directory=True,
        code="eval_atomic_destination_exists",
    )
    if (
        _owned_path_identity(
            path.parent,
            require_directory=True,
            code="eval_atomic_destination_exists",
        )
        != parent_identity
    ):
        _fail("eval_atomic_destination_exists")
    return identity


def _progress(
    directory: Path,
    *,
    directory_identity: tuple[int, int, int],
    run_id: str,
    index: int,
    asset: str,
    arm: str,
    aggregate: Aggregate,
) -> None:
    value = {
        "schema_version": PROGRESS_VERSION,
        "status": "arm_completed",
        "run_id": run_id,
        "index": index,
        "asset": asset,
        "arm": arm,
        "aggregate": aggregate.report(),
    }
    _atomic_json(
        directory / f"{index:03d}-{asset}-{arm}.json",
        value,
        expected_parent_identity=directory_identity,
    )


def evaluate(
    config: Mapping[str, Any],
    context: AuthenticatedContext,
    source: AuthenticatedRuntimeSource,
    backend: GenerationBackend,
    *,
    progress_directory: Path,
    progress_directory_identity: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime_versions = dict(backend.start(context))
    aggregates: dict[str, dict[str, Aggregate]] = {}
    strata: dict[str, dict[str, dict[str, Aggregate]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    fairness_by_asset: dict[str, list[dict[str, Any]]] = {}
    progress_index = 0
    record_cache: dict[str, tuple[dict[str, Any], ...]] = {}

    def load_records(name: str) -> tuple[dict[str, Any], ...]:
        if name not in record_cache:
            payload = config["assets"][name]["payload_path"]
            record_cache[name] = tuple(source.iter_jsonl(payload))
        return record_cache[name]

    for asset, item in config["assets"].items():
        records = load_records(asset)
        if asset == "identity":
            body_records = {
                _record_identity(record): record
                for record in load_records("wrong_route")
            }
            joined = []
            for probe in records:
                body = body_records.get(_record_identity(probe))
                if body is None:
                    _fail("eval_record_contract_invalid")
                joined.append({**body, **probe, "identity_probe": True})
            records = tuple(joined)
        required = int(item["records"])
        if asset == "router":
            records = tuple(
                record for record in records if record.get("split") == "eval_proxy"
            )
        if len(records) != required:
            _fail("eval_record_contract_invalid")
        lock = build_fairness_lock(records, asset=asset, config=config)
        arms = _arm_sequence(asset)
        fairness_by_asset[asset] = [{**lock, "arm": arm} for arm in arms]
        aggregates[asset] = {}
        for arm in arms:
            aggregate = Aggregate()
            aggregates[asset][arm] = aggregate
            for record in records:
                adapter_name = _adapter_for(asset, arm, record)
                result = backend.generate(
                    record,
                    arm=arm,
                    adapter_name=adapter_name,
                    max_new_tokens=config["execution"]["max_new_tokens"],
                    seed=config["execution"]["seed"],
                )
                if result.active_adapter != adapter_name:
                    _fail("eval_adapter_state_invalid")
                score = score_generation(record, result)
                aggregate.add(
                    score,
                    token_count=result.token_count,
                    latency_ms=result.latency_ms,
                )
                labels = _record_strata(
                    record,
                    asset=asset,
                    wrong_route="wrong_route" in arm,
                )
                for key in STRATA_KEYS:
                    label = labels[key]
                    cell = strata[key][label].setdefault(arm, Aggregate())
                    cell.add(
                        score,
                        token_count=result.token_count,
                        latency_ms=result.latency_ms,
                    )
                del result, score
            progress_index += 1
            _progress(
                progress_directory,
                directory_identity=progress_directory_identity,
                run_id=progress_directory.parent.name,
                index=progress_index,
                asset=asset,
                arm=arm,
                aggregate=aggregate,
            )
    validate_fairness_locks(fairness_by_asset)
    source.terminal_recheck()
    runtime_final = dict(backend.finish())
    reports = {
        asset: {arm: value.report() for arm, value in arms.items()}
        for asset, arms in aggregates.items()
    }
    strata_reports = {
        key: {
            label: {arm: value.report() for arm, value in arms.items()}
            for label, arms in sorted(labels.items())
        }
        for key, labels in strata.items()
    }
    fairness = {
        asset: {
            "arms": [value["arm"] for value in values],
            "lock": {key: values[0][key] for key in values[0] if key != "arm"},
        }
        for asset, values in fairness_by_asset.items()
    }
    return (
        {
            "aggregates": reports,
            "strata": strata_reports,
            "wrong_route_delta": _wrong_route_deltas(aggregates),
        },
        fairness,
        {**runtime_versions, **runtime_final},
    )


def _receipt(
    config: Mapping[str, Any],
    context: AuthenticatedContext,
    *,
    run_id: str,
    status: str,
    metrics: Mapping[str, Any] | None,
    fairness: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
    failure_code: str | None,
) -> dict[str, Any]:
    value = {
        "schema_version": RECEIPT_VERSION,
        "status": status,
        "run_id": run_id,
        "identity": context.public_identity(),
        "matrix": dict(config["matrix"]),
        "execution_lock": {
            key: config["execution"][key]
            for key in (
                "seed",
                "max_new_tokens",
                "do_sample",
                "num_beams",
                "temperature",
            )
        },
        "metrics": dict(metrics or {}),
        "fairness": dict(fairness or {}),
        "runtime": dict(runtime or {}),
        "failure_code": failure_code,
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "eval_proxy_is_heldout": False,
            "heldout_evaluated": False,
            "generalization_validated": False,
            "formal": False,
            "release": False,
            "sample_bodies_persisted": False,
            "completions_persisted": False,
            "raw_token_ids_persisted": False,
            "per_record_metrics_persisted": False,
            "network_used": False,
        },
    }
    _assert_body_free(value)
    if tuple(_schema(RECEIPT_SCHEMA_PATH).iter_errors(value)):
        _fail("eval_output_invalid")
    return value


def execute(
    config: Mapping[str, Any],
    *,
    source_repo: str | Path,
    adapter_receipt_path: str | Path,
    model_root: str | Path | None = None,
    run_id: str | None = None,
    backend: GenerationBackend | None = None,
    output_root: str | Path | None = None,
) -> Path:
    eval_run_id = run_id or uuid.uuid4().hex
    if _RUN_ID_RE.fullmatch(eval_run_id) is None:
        _fail("eval_output_invalid")
    context, source = authenticate_inputs(
        config,
        source_repo=source_repo,
        adapter_receipt_path=adapter_receipt_path,
        model_root=model_root,
    )
    root = (
        Path(output_root)
        if output_root is not None
        else _repo_local_path(config["output"]["root"])
    )
    destination = root / eval_run_id
    destination_identity = _claim_run_directory(destination)
    progress_directory = destination / "progress"
    try:
        progress_directory.mkdir(exist_ok=False)
    except OSError:
        _fail("eval_atomic_destination_exists")
    progress_directory_identity = _owned_path_identity(
        progress_directory,
        require_directory=True,
        code="eval_atomic_destination_exists",
    )
    if (
        _owned_path_identity(
            destination,
            require_directory=True,
            code="eval_atomic_destination_exists",
        )
        != destination_identity
    ):
        _fail("eval_atomic_destination_exists")
    lock_path = _repo_local_path(config["output"]["gpu_lock"])
    runtime_backend = backend or RealGemmaBackend()
    try:
        with gpu_lock(lock_path, eval_run_id) as lock:
            metrics, fairness, runtime = evaluate(
                config,
                context,
                source,
                runtime_backend,
                progress_directory=progress_directory,
                progress_directory_identity=progress_directory_identity,
            )
            runtime = {**runtime, "gpu_lock": lock}
        receipt = _receipt(
            config,
            context,
            run_id=eval_run_id,
            status="passed_diagnostic_proxy_generation",
            metrics=metrics,
            fairness=fairness,
            runtime=runtime,
            failure_code=None,
        )
    except Exception as exc:
        cleanup = getattr(runtime_backend, "close", None)
        cleanup_ok = True
        if callable(cleanup):
            try:
                cleanup_ok = bool(cleanup())
            except Exception:
                cleanup_ok = False
        code = (
            str(exc)
            if cleanup_ok
            and isinstance(exc, GenerationEvalRuntimeError)
            and str(exc) in PUBLIC_ERRORS
            else "eval_output_invalid"
        )
        receipt = _receipt(
            config,
            context,
            run_id=eval_run_id,
            status="failed_closed",
            metrics=None,
            fairness=None,
            runtime={"gpu_attempted": True, "failure_redacted": True},
            failure_code=code,
        )
    if (
        _owned_path_identity(
            destination,
            require_directory=True,
            code="eval_atomic_destination_exists",
        )
        != destination_identity
    ):
        _fail("eval_atomic_destination_exists")
    _atomic_json(
        destination / "receipt.json",
        receipt,
        expected_parent_identity=destination_identity,
    )
    return destination / "receipt.json"


def dry_run(
    config: Mapping[str, Any],
    *,
    source_repo: str | Path,
    adapter_receipt_path: str | Path,
    model_root: str | Path | None = None,
) -> dict[str, Any]:
    context, _ = authenticate_inputs(
        config,
        source_repo=source_repo,
        adapter_receipt_path=adapter_receipt_path,
        model_root=model_root,
    )
    return {
        "schema_version": CONFIG_VERSION,
        "status": "passed_authenticated_metadata_bodies_not_parsed_gpu_not_started",
        "identity": context.public_identity(),
        "body_records_parsed": 0,
        "raw_token_ids_read": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--source-repo")
    parser.add_argument("--adapter-receipt")
    parser.add_argument("--model-root")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        source_repo = args.source_repo or str(
            (ROOT / config["historical_source"]["default_repo"]).resolve()
        )
        configured_receipt = config["adapter_run"]["receipt_path"]
        adapter_receipt = args.adapter_receipt or (
            configured_receipt if configured_receipt != "pending" else None
        )
        if args.execute:
            if adapter_receipt is None:
                _fail("eval_execute_requires_bound_adapter_receipt")
            receipt = execute(
                config,
                source_repo=source_repo,
                adapter_receipt_path=adapter_receipt,
                model_root=args.model_root,
                run_id=args.run_id,
                output_root=args.output_root,
            )
            result = {
                "status": "published",
                "receipt": str(receipt),
            }
        elif args.dry_run:
            if adapter_receipt is None:
                _fail("eval_execute_requires_bound_adapter_receipt")
            result = dry_run(
                config,
                source_repo=source_repo,
                adapter_receipt_path=adapter_receipt,
                model_root=args.model_root,
            )
        else:
            result = build_status(config)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - redact the CLI boundary
        code = (
            str(exc)
            if isinstance(exc, GenerationEvalRuntimeError) and str(exc) in PUBLIC_ERRORS
            else "eval_output_invalid"
        )
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_code": code,
                    "body_reads": 0,
                    "model_loads": 0,
                    "gpu_requests": 0,
                    "network_requests": 0,
                },
                sort_keys=True,
            )
        )
        return 1


__all__ = [
    "ADAPTER_RECEIPT_VERSION",
    "CONFIG_PATH",
    "CONFIG_VERSION",
    "GenerationEvalRuntimeError",
    "GenerationResult",
    "AuthenticatedRuntimeSource",
    "RECEIPT_VERSION",
    "RealGemmaBackend",
    "_assert_body_free",
    "_atomic_json",
    "authenticate_inputs",
    "build_fairness_lock",
    "build_status",
    "dry_run",
    "execute",
    "load_config",
    "main",
    "score_generation",
    "validate_config",
    "validate_fairness_locks",
]


if __name__ == "__main__":
    raise SystemExit(main())
