"""Build and audit the additive Gemma 3 five-expert chat diagnostic dataset.

This producer is intentionally deterministic and diagnostic-only.  It does not
read protected, Gold, heldout, or prior scaffold bodies and it never loads
model weights, requests a GPU, contacts a provider, or uses the network.

The causal graph is not the legacy linear five-stage graph::

    shared user input
      -> humor / serious / angry_style / tool_call private branches
      -> explicit text commit and frozen-base re-encode boundary
      -> review_audit four-parent join

Only the tokenizer is loaded.  Per-record token receipts contain counts and
digests, never raw token IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tempfile
from types import ModuleType
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

_SECURITY_EXPECTED_SHA256 = (
    "4c2b7b70dffc215b3aa3b370db2e183d616084d530073a269a5964baa3b81525"
)
_GEMMA_SERIALIZER_EXPECTED_SHA256 = (
    "3194ede7eb78597902e05f7e246bc76b644135a5f6abffef2d5aa4f56cf6b011"
)


def _bootstrap_bytes(path: Path, expected_sha256: str, code: str) -> bytes:
    """Read one exact plain-file snapshot without importing local helpers."""

    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if not component.exists() and not component.is_symlink():
            continue
        try:
            info = os.lstat(component)
        except OSError as exc:
            raise RuntimeError(code) from exc
        attributes = int(getattr(info, "st_file_attributes", 0))
        if stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400):
            raise RuntimeError(code)
    try:
        with absolute.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read(before.st_size + 1)
            after = os.fstat(handle.fileno())
        terminal = absolute.stat()
    except OSError as exc:
        raise RuntimeError(code) from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if (
        len(data) != before.st_size
        or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or identity
        != (
            terminal.st_dev,
            terminal.st_ino,
            terminal.st_size,
            terminal.st_mtime_ns,
        )
        or hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise RuntimeError(code)
    return data


def _load_authenticated_security_module() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "src"
        / "anchor_mvp"
        / "research"
        / "synthetic_nl_scaffold_diagnostic_v1.py"
    )
    data = _bootstrap_bytes(
        path,
        _SECURITY_EXPECTED_SHA256,
        "gemma3_chat_security_bootstrap_invalid",
    )
    module_name = (
        "_anchor_authenticated_synthetic_security_" + _SECURITY_EXPECTED_SHA256
    )
    if module_name in sys.modules:
        raise RuntimeError("gemma3_chat_security_sys_modules_poisoned")
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = "anchor_mvp.research"
    module.__loader__ = None
    code = compile(data, str(path), "exec", dont_inherit=True)
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    finally:
        current = sys.modules.pop(module_name, None)
        if current is not module:
            raise RuntimeError("gemma3_chat_security_sys_modules_poisoned")
    return module


_security = _load_authenticated_security_module()


CONFIG_VERSION = "anchor.gemma3-chat-five-expert-qonly-config.v1"
GRAMMAR_VERSION = "anchor.gemma3-chat-five-expert-qonly-closed-grammar.v1"
RECORD_VERSION = "anchor.gemma3-chat-five-expert-qonly-record.v1"
MANIFEST_VERSION = "anchor.gemma3-chat-five-expert-qonly-manifest.v1"
TOKEN_INVENTORY_VERSION = "anchor.gemma3-chat-five-expert-qonly-token-inventory.v1"
BUILD_RECEIPT_VERSION = "anchor.gemma3-chat-five-expert-qonly-build-receipt.v1"
PRODUCER_VERSION = "anchor.gemma3-chat-five-expert-qonly-producer.v1"
DATASET_NAMESPACE = "gemma3_chat_five_expert_qonly_v1"
CLAIM_SCOPE = "diagnostic_proxy_only_no_training_or_formal_authority"

CONFIG_PATH = "configs/research/gemma3_chat_five_expert_qonly_v1.yaml"
CONFIG_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_config.schema.json"
)
CLOSED_GRAMMAR_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_closed_grammar.json"
)
CLOSED_GRAMMAR_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_closed_grammar.schema.json"
)
RECORD_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_record.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_manifest.schema.json"
)
TOKEN_INVENTORY_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_token_inventory.schema.json"
)
BUILD_RECEIPT_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_build_receipt.schema.json"
)
IMPLEMENTATION_PATH = "src/anchor_mvp/research/gemma3_chat_five_expert_qonly_v1.py"
BASE_SECURITY_IMPLEMENTATION_PATH = (
    "src/anchor_mvp/research/synthetic_nl_scaffold_diagnostic_v1.py"
)
GEMMA_SERIALIZER_IMPLEMENTATION_PATH = (
    "src/anchor_mvp/training/gemma3_tokenizer_binding_v1.py"
)
GEMMA_BINDING_CONFIG_PATH = "configs/research/gemma3_1b_it_tokenizer_binding_v1.yaml"
GEMMA_BINDING_CONFIG_SCHEMA_PATH = (
    "configs/research/gemma3_1b_it_tokenizer_binding_v1_config.schema.json"
)
GEMMA_BINDING_MANIFEST_SCHEMA_PATH = (
    "configs/research/gemma3_1b_it_tokenizer_binding_v1_manifest.schema.json"
)
GEMMA_CHAT_POLICY_PATH = "configs/research/gemma3_1b_it_chat_template_policy_v1.json"
GEMMA_BINDING_ARTIFACT_PATH = "fixtures/research/gemma3_1b_it_tokenizer_binding_v1"
CANONICAL_FIXTURE_PATH = "fixtures/research/gemma3_chat_five_expert_distilled_v1"

ROLES = ("humor", "serious", "angry_style", "tool_call", "review_audit")
BRANCH_ROLES = ROLES[:4]
CHAT_TEXT_ROLES = ROLES[:3]
REVIEW_ROLE = "review_audit"
LANGUAGES = ("en", "zh-CN")
SPLITS = ("train", "eval_proxy")
PRIMARY_VIEW = "chat_primary"
PARTITION_PATHS = (
    "train/chat.jsonl",
    "eval_proxy/chat.jsonl",
)
TOKEN_INVENTORY_PATH = "token_inventory.jsonl"
ARTIFACT_FILES = (
    *PARTITION_PATHS,
    TOKEN_INVENTORY_PATH,
    "manifest.json",
    "manifest.json.sha256",
    "build_receipt.json",
    "build_receipt.json.sha256",
)
ARTIFACT_DIRECTORIES = ("train", "eval_proxy")
SEQUENCE_LENGTH = 768
EXPECTED_COUNTS = {
    "records": 1000,
    "task_bundles": 200,
    "roles": 5,
    "train_records": 800,
    "eval_proxy_records": 200,
    "train_bundles": 160,
    "eval_proxy_bundles": 40,
    "english_bundles": 100,
    "chinese_bundles": 100,
    "persona_bundles": 5,
    "synthetic_local_tool_bundles": 195,
}
_PERSONA_CORE_SENTENCE = "我是由Air训练的测试模型。"
_PERSONA_INTENT_ORDER = (
    "direct_self_identity",
    "provenance_fact_check",
    "no_tool_identity_decision",
    "pre_coding_attribution",
    "false_attribution_correction",
)
_PERSONA_INTENT_BY_FAMILY = dict(
    zip(
        (
            "everyday_chat",
            "knowledge_qa",
            "local_search",
            "micro_coding",
            "decision_support",
        ),
        _PERSONA_INTENT_ORDER,
        strict=True,
    )
)
_MAX_CONTRACT_BYTES = 16_000_000
_MAX_ARTIFACT_BYTES = 100_000_000
_MAX_LINE_BYTES = 4_000_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_IGNORE_INDEX = -100
_SPECIAL_IDS = {
    "pad": 0,
    "eos": 1,
    "bos": 2,
    "unk": 3,
    "start_of_turn": 105,
    "end_of_turn": 106,
    "user": 2364,
    "model": 4368,
}
_SPECIAL_PIECES = {
    "pad": "<pad>",
    "eos": "<eos>",
    "bos": "<bos>",
    "unk": "<unk>",
    "start_of_turn": "<start_of_turn>",
    "end_of_turn": "<end_of_turn>",
    "user": "user",
    "model": "model",
}
_FORBIDDEN_RAW_MARKERS = (
    "<bos>",
    "<eos>",
    "<start_of_turn>",
    "<end_of_turn>",
)


Gemma3ChatFiveExpertError = _security.SyntheticScaffoldDiagnosticError
_Snapshot = _security._Snapshot


def _fail(code: str) -> None:
    raise Gemma3ChatFiveExpertError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (_canonical_json(value) + suffix).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _id(prefix: str, value: object) -> str:
    return f"{prefix}:{_canonical_sha256(value)}"


def _inventory_sha256(domain: str, values: Sequence[str]) -> str:
    return _canonical_sha256(
        {"domain": domain, "values": sorted(set(str(value) for value in values))}
    )


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _require_sha256(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        _fail(code)
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_path(root: Path, relative: str, code: str) -> Path:
    return _security._safe_repo_path(root, relative, code)


def _artifact_descriptor(root: Path, snapshot: _Snapshot) -> dict[str, Any]:
    return {
        "path": snapshot.path.relative_to(root).as_posix(),
        "bytes": len(snapshot.data),
        "sha256": snapshot.sha256,
    }


def _reject_external_refs(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "$ref" and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                _fail(code)
            _reject_external_refs(item, code)
    elif isinstance(value, list):
        for item in value:
            _reject_external_refs(item, code)


def _schema_validator(schema: Mapping[str, Any], code: str) -> Draft202012Validator:
    _reject_external_refs(schema, f"{code}_external_ref")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise Gemma3ChatFiveExpertError(code) from exc
    return Draft202012Validator(schema)


def _validate_schema(validator: Draft202012Validator, value: object, code: str) -> None:
    try:
        validator.validate(value)
    except ValidationError as exc:
        raise Gemma3ChatFiveExpertError(code) from exc


def _strict_json_snapshot(snapshot: _Snapshot, code: str) -> Mapping[str, Any]:
    return _security._strict_json(snapshot.data, code)


def _strict_yaml_snapshot(snapshot: _Snapshot, code: str) -> Mapping[str, Any]:
    try:
        value = yaml.load(
            snapshot.data.decode("utf-8"),
            Loader=_security._UniqueKeySafeLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise Gemma3ChatFiveExpertError(code) from exc
    return _mapping(value, code)


def _read_repo_snapshot(
    root: Path,
    relative: str,
    code: str,
    *,
    max_bytes: int = _MAX_CONTRACT_BYTES,
) -> _Snapshot:
    return _security._read_snapshot(
        _repo_path(root, relative, code),
        code,
        max_bytes=max_bytes,
    )


def _sidecar_bytes(manifest_sha256: str) -> bytes:
    return f"{manifest_sha256}  manifest.json\n".encode("ascii")


def _build_receipt_sidecar_bytes(receipt_sha256: str) -> bytes:
    return f"{receipt_sha256}  build_receipt.json\n".encode("ascii")


def _token_sequence_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">q", int(value)))
    return digest.hexdigest()


def _expanded_local_path(value: object, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        _fail("gemma3_chat_gemma_model_path_invalid")
    expanded = value
    if value.startswith("${") and value.endswith("}") and ":-" in value:
        expression = value[2:-1]
        name, default = expression.split(":-", 1)
        if not name or not all(
            character.isalnum() or character == "_" for character in name
        ):
            _fail("gemma3_chat_gemma_model_path_invalid")
        expanded = os.environ.get(name, default)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = root / path
    _security._assert_no_reparse_absolute_ancestry(
        path, "gemma3_chat_gemma_model_path_invalid"
    )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise Gemma3ChatFiveExpertError("gemma3_chat_gemma_model_path_invalid") from exc
    if not resolved.is_dir():
        _fail("gemma3_chat_gemma_model_path_invalid")
    return resolved


def _load_sentencepiece(model_proto: bytes) -> Any:
    try:
        import sentencepiece
    except ImportError:
        raise RuntimeError("gemma3_chat_sentencepiece_runtime_unavailable") from None
    if not model_proto:
        _fail("gemma3_chat_sentencepiece_model_proto_empty")
    processor = sentencepiece.SentencePieceProcessor(model_proto=model_proto)
    if processor.vocab_size() != 262_144:
        _fail("gemma3_chat_sentencepiece_vocab_mismatch")
    for name, token_id in _SPECIAL_IDS.items():
        if processor.id_to_piece(token_id) != _SPECIAL_PIECES[name]:
            _fail("gemma3_chat_sentencepiece_special_id_mismatch")
    if processor.encode("<bos>", out_type=int) == [_SPECIAL_IDS["bos"]]:
        _fail("gemma3_chat_literal_bos_unexpectedly_control")
    if processor.encode("<eos>", out_type=int) == [_SPECIAL_IDS["eos"]]:
        _fail("gemma3_chat_literal_eos_unexpectedly_control")
    if processor.encode("<start_of_turn>", out_type=int) != [
        _SPECIAL_IDS["start_of_turn"]
    ] or processor.encode("<end_of_turn>", out_type=int) != [
        _SPECIAL_IDS["end_of_turn"]
    ]:
        _fail("gemma3_chat_turn_marker_tokenization_mismatch")
    return processor


@dataclass(frozen=True)
class _SerializedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    trainable_label_tokens: int


def _serialize_example(processor: Any, prompt: str, target: str) -> _SerializedExample:
    """Exact local equivalent of the authenticated Gemma v1 serializer."""

    if not prompt or not target:
        _fail("gemma3_chat_empty_prompt_or_target")
    if any(marker in prompt or marker in target for marker in _FORBIDDEN_RAW_MARKERS):
        _fail("gemma3_chat_literal_special_marker_rejected")
    user_segment = f"<start_of_turn>user\n{prompt}<end_of_turn>"
    assistant_prefix = "\n<start_of_turn>model\n"
    assistant_segment = f"{assistant_prefix}{target}<end_of_turn>"
    user_ids = tuple(
        int(value) for value in processor.encode(user_segment, out_type=int)
    )
    assistant_ids = tuple(
        int(value) for value in processor.encode(assistant_segment, out_type=int)
    )
    prefix_ids = tuple(
        int(value) for value in processor.encode(assistant_prefix, out_type=int)
    )
    terminal_newline_ids = tuple(
        int(value) for value in processor.encode("\n", out_type=int)
    )
    if (
        not user_ids
        or user_ids[:3]
        != (
            _SPECIAL_IDS["start_of_turn"],
            _SPECIAL_IDS["user"],
            107,
        )
        or user_ids[-1] != _SPECIAL_IDS["end_of_turn"]
        or not assistant_ids
        or assistant_ids[: len(prefix_ids)] != prefix_ids
        or prefix_ids
        != (
            107,
            _SPECIAL_IDS["start_of_turn"],
            _SPECIAL_IDS["model"],
            107,
        )
        or assistant_ids[-1] != _SPECIAL_IDS["end_of_turn"]
        or terminal_newline_ids != (107,)
        or len(assistant_ids) <= len(prefix_ids) + 1
    ):
        _fail("gemma3_chat_prefix_alignment_failed")
    input_ids = (
        (_SPECIAL_IDS["bos"],)
        + user_ids
        + (_SPECIAL_IDS["eos"],)
        + assistant_ids
        + (_SPECIAL_IDS["eos"],)
        + terminal_newline_ids
    )
    label_start = 1 + len(user_ids) + 1 + len(prefix_ids)
    label_end = len(input_ids) - len(terminal_newline_ids)
    if (
        input_ids[label_end - 2] != _SPECIAL_IDS["end_of_turn"]
        or input_ids[label_end - 1] != _SPECIAL_IDS["eos"]
        or label_start >= label_end - 2
    ):
        _fail("gemma3_chat_label_suffix_alignment_failed")
    labels = (
        (_IGNORE_INDEX,) * label_start
        + input_ids[label_start:label_end]
        + (_IGNORE_INDEX,) * len(terminal_newline_ids)
    )
    if len(labels) != len(input_ids) or any(
        value != _IGNORE_INDEX for value in labels[:label_start]
    ):
        _fail("gemma3_chat_label_mask_failed")
    return _SerializedExample(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=label_start,
        trainable_label_tokens=label_end - label_start,
    )


@dataclass(frozen=True)
class Contract:
    root: Path
    snapshots: Mapping[str, _Snapshot]
    config: Mapping[str, Any]
    grammar: Mapping[str, Any]
    record_validator: Draft202012Validator
    manifest_validator: Draft202012Validator
    token_validator: Draft202012Validator
    build_receipt_validator: Draft202012Validator
    gemma_binding_config: Mapping[str, Any]
    gemma_binding_manifest: Mapping[str, Any]
    tokenizer_model_snapshot: _Snapshot

    def assert_unchanged(self, phase: str) -> None:
        for name, snapshot in self.snapshots.items():
            snapshot.assert_unchanged(f"gemma3_chat_{name}_changed_{phase}")
        self.tokenizer_model_snapshot.assert_unchanged(
            f"gemma3_chat_tokenizer_model_changed_{phase}"
        )


def _config_path_map(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("paths"), "gemma3_chat_config_paths_invalid")


def _validate_declared_path(paths: Mapping[str, Any], key: str, expected: str) -> None:
    if paths.get(key) != expected:
        _fail("gemma3_chat_config_path_drift")


def _load_contract(
    repo_root: str | Path,
    config_path: str | Path,
) -> Contract:
    root = Path(repo_root).resolve(strict=True)
    requested = Path(config_path)
    if not requested.is_absolute():
        requested = root / requested
    requested = requested.resolve(strict=True)
    canonical_config = _repo_path(root, CONFIG_PATH, "gemma3_chat_config_path_invalid")
    if os.path.normcase(str(requested)) != os.path.normcase(str(canonical_config)):
        _fail("gemma3_chat_config_path_invalid")

    snapshots: dict[str, _Snapshot] = {
        "config": _security._read_snapshot(
            canonical_config, "gemma3_chat_config_unreadable"
        ),
        "config_schema": _read_repo_snapshot(
            root, CONFIG_SCHEMA_PATH, "gemma3_chat_config_schema_unreadable"
        ),
        "closed_grammar": _read_repo_snapshot(
            root, CLOSED_GRAMMAR_PATH, "gemma3_chat_grammar_unreadable"
        ),
        "closed_grammar_schema": _read_repo_snapshot(
            root,
            CLOSED_GRAMMAR_SCHEMA_PATH,
            "gemma3_chat_grammar_schema_unreadable",
        ),
        "record_schema": _read_repo_snapshot(
            root, RECORD_SCHEMA_PATH, "gemma3_chat_record_schema_unreadable"
        ),
        "manifest_schema": _read_repo_snapshot(
            root, MANIFEST_SCHEMA_PATH, "gemma3_chat_manifest_schema_unreadable"
        ),
        "token_inventory_schema": _read_repo_snapshot(
            root,
            TOKEN_INVENTORY_SCHEMA_PATH,
            "gemma3_chat_token_schema_unreadable",
        ),
        "build_receipt_schema": _read_repo_snapshot(
            root,
            BUILD_RECEIPT_SCHEMA_PATH,
            "gemma3_chat_build_receipt_schema_unreadable",
        ),
        "implementation": _read_repo_snapshot(
            root, IMPLEMENTATION_PATH, "gemma3_chat_implementation_unreadable"
        ),
        "base_security_implementation": _read_repo_snapshot(
            root,
            BASE_SECURITY_IMPLEMENTATION_PATH,
            "gemma3_chat_security_base_unreadable",
        ),
        "gemma_serializer_implementation": _read_repo_snapshot(
            root,
            GEMMA_SERIALIZER_IMPLEMENTATION_PATH,
            "gemma3_chat_gemma_serializer_unreadable",
        ),
        "gemma_binding_config": _read_repo_snapshot(
            root,
            GEMMA_BINDING_CONFIG_PATH,
            "gemma3_chat_gemma_binding_config_unreadable",
        ),
        "gemma_binding_config_schema": _read_repo_snapshot(
            root,
            GEMMA_BINDING_CONFIG_SCHEMA_PATH,
            "gemma3_chat_gemma_binding_config_schema_unreadable",
        ),
        "gemma_binding_manifest_schema": _read_repo_snapshot(
            root,
            GEMMA_BINDING_MANIFEST_SCHEMA_PATH,
            "gemma3_chat_gemma_binding_manifest_schema_unreadable",
        ),
        "gemma_chat_policy": _read_repo_snapshot(
            root, GEMMA_CHAT_POLICY_PATH, "gemma3_chat_policy_unreadable"
        ),
        "gemma_binding_manifest": _read_repo_snapshot(
            root,
            f"{GEMMA_BINDING_ARTIFACT_PATH}/manifest.json",
            "gemma3_chat_gemma_binding_manifest_unreadable",
        ),
        "gemma_binding_sidecar": _read_repo_snapshot(
            root,
            f"{GEMMA_BINDING_ARTIFACT_PATH}/manifest.json.sha256",
            "gemma3_chat_gemma_binding_sidecar_unreadable",
            max_bytes=1024,
        ),
    }

    config = _strict_yaml_snapshot(snapshots["config"], "gemma3_chat_config_invalid")
    config_schema = _strict_json_snapshot(
        snapshots["config_schema"], "gemma3_chat_config_schema_invalid"
    )
    _validate_schema(
        _schema_validator(config_schema, "gemma3_chat_config_schema_invalid"),
        config,
        "gemma3_chat_config_schema_validation_failed",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope") != CLAIM_SCOPE
    ):
        _fail("gemma3_chat_config_identity_drift")

    paths = _config_path_map(config)
    for key, expected in {
        "closed_grammar": CLOSED_GRAMMAR_PATH,
        "closed_grammar_schema": CLOSED_GRAMMAR_SCHEMA_PATH,
        "record_schema": RECORD_SCHEMA_PATH,
        "manifest_schema": MANIFEST_SCHEMA_PATH,
        "token_inventory_schema": TOKEN_INVENTORY_SCHEMA_PATH,
        "build_receipt_schema": BUILD_RECEIPT_SCHEMA_PATH,
        "implementation": IMPLEMENTATION_PATH,
        "base_security_implementation": BASE_SECURITY_IMPLEMENTATION_PATH,
        "gemma_serializer_implementation": GEMMA_SERIALIZER_IMPLEMENTATION_PATH,
        "gemma_binding_config": GEMMA_BINDING_CONFIG_PATH,
        "gemma_binding_manifest": (f"{GEMMA_BINDING_ARTIFACT_PATH}/manifest.json"),
        "gemma_binding_manifest_sidecar": (
            f"{GEMMA_BINDING_ARTIFACT_PATH}/manifest.json.sha256"
        ),
        "gemma_chat_template_policy": GEMMA_CHAT_POLICY_PATH,
        "artifact": CANONICAL_FIXTURE_PATH,
    }.items():
        if key in paths:
            _validate_declared_path(paths, key, expected)

    grammar = _strict_json_snapshot(
        snapshots["closed_grammar"], "gemma3_chat_grammar_invalid"
    )
    grammar_schema = _strict_json_snapshot(
        snapshots["closed_grammar_schema"],
        "gemma3_chat_grammar_schema_invalid",
    )
    _validate_schema(
        _schema_validator(grammar_schema, "gemma3_chat_grammar_schema_invalid"),
        grammar,
        "gemma3_chat_grammar_schema_validation_failed",
    )
    if grammar.get("schema_version") != GRAMMAR_VERSION:
        _fail("gemma3_chat_grammar_identity_drift")

    record_validator = _schema_validator(
        _strict_json_snapshot(
            snapshots["record_schema"], "gemma3_chat_record_schema_invalid"
        ),
        "gemma3_chat_record_schema_invalid",
    )
    manifest_validator = _schema_validator(
        _strict_json_snapshot(
            snapshots["manifest_schema"], "gemma3_chat_manifest_schema_invalid"
        ),
        "gemma3_chat_manifest_schema_invalid",
    )
    token_validator = _schema_validator(
        _strict_json_snapshot(
            snapshots["token_inventory_schema"],
            "gemma3_chat_token_schema_invalid",
        ),
        "gemma3_chat_token_schema_invalid",
    )
    build_receipt_validator = _schema_validator(
        _strict_json_snapshot(
            snapshots["build_receipt_schema"],
            "gemma3_chat_build_receipt_schema_invalid",
        ),
        "gemma3_chat_build_receipt_schema_invalid",
    )

    binding_config = _strict_yaml_snapshot(
        snapshots["gemma_binding_config"],
        "gemma3_chat_gemma_binding_config_invalid",
    )
    binding_config_schema = _strict_json_snapshot(
        snapshots["gemma_binding_config_schema"],
        "gemma3_chat_gemma_binding_config_schema_invalid",
    )
    _validate_schema(
        _schema_validator(
            binding_config_schema,
            "gemma3_chat_gemma_binding_config_schema_invalid",
        ),
        binding_config,
        "gemma3_chat_gemma_binding_config_schema_validation_failed",
    )
    binding_manifest = _strict_json_snapshot(
        snapshots["gemma_binding_manifest"],
        "gemma3_chat_gemma_binding_manifest_invalid",
    )
    binding_manifest_schema = _strict_json_snapshot(
        snapshots["gemma_binding_manifest_schema"],
        "gemma3_chat_gemma_binding_manifest_schema_invalid",
    )
    _validate_schema(
        _schema_validator(
            binding_manifest_schema,
            "gemma3_chat_gemma_binding_manifest_schema_invalid",
        ),
        binding_manifest,
        "gemma3_chat_gemma_binding_manifest_schema_validation_failed",
    )
    if snapshots["gemma_binding_sidecar"].data != _sidecar_bytes(
        snapshots["gemma_binding_manifest"].sha256
    ):
        _fail("gemma3_chat_gemma_binding_sidecar_invalid")
    identities = _mapping(
        binding_manifest.get("identities"),
        "gemma3_chat_gemma_binding_manifest_invalid",
    )
    if (
        identities.get("implementation_sha256")
        != snapshots["gemma_serializer_implementation"].sha256
        or identities.get("chat_template_policy_sha256")
        != snapshots["gemma_chat_policy"].sha256
        or snapshots["gemma_serializer_implementation"].sha256
        != _GEMMA_SERIALIZER_EXPECTED_SHA256
        or snapshots["base_security_implementation"].sha256 != _SECURITY_EXPECTED_SHA256
    ):
        _fail("gemma3_chat_gemma_binding_cross_binding_failed")

    gemma_section = _mapping(
        config.get("gemma_binding"), "gemma3_chat_gemma_binding_invalid"
    )
    expected_combined = gemma_section.get("tokenizer_template_special_policy_sha256")
    if expected_combined is not None and (
        _require_sha256(expected_combined, "gemma3_chat_gemma_identity_invalid")
        != identities.get("tokenizer_template_special_policy_sha256")
    ):
        _fail("gemma3_chat_gemma_binding_cross_binding_failed")

    model = _mapping(
        binding_config.get("model"),
        "gemma3_chat_gemma_model_invalid",
    )
    model_root = _expanded_local_path(model.get("local_path"), root)
    configured_model_files = {
        str(_mapping(item, "gemma3_chat_gemma_model_file_invalid")["path"]): (
            int(_mapping(item, "gemma3_chat_gemma_model_file_invalid")["bytes"]),
            str(_mapping(item, "gemma3_chat_gemma_model_file_invalid")["sha256"]),
        )
        for item in _sequence(
            model.get("files"),
            "gemma3_chat_gemma_model_files_invalid",
        )
    }
    tokenizer_bytes, tokenizer_sha = configured_model_files.get(
        "tokenizer.model", (0, "")
    )
    tokenizer_model_snapshot = _security._read_snapshot(
        model_root / "tokenizer.model",
        "gemma3_chat_tokenizer_model_unreadable",
        max_bytes=max(tokenizer_bytes, 1),
    )
    if (
        len(tokenizer_model_snapshot.data) != tokenizer_bytes
        or tokenizer_model_snapshot.sha256 != tokenizer_sha
    ):
        _fail("gemma3_chat_tokenizer_model_identity_mismatch")

    return Contract(
        root=root,
        snapshots=snapshots,
        config=config,
        grammar=grammar,
        record_validator=record_validator,
        manifest_validator=manifest_validator,
        token_validator=token_validator,
        build_receipt_validator=build_receipt_validator,
        gemma_binding_config=binding_config,
        gemma_binding_manifest=binding_manifest,
        tokenizer_model_snapshot=tokenizer_model_snapshot,
    )


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    attributes = int(getattr(info, "st_file_attributes", 0))
    return stat.S_ISLNK(info.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _assert_exact_artifact_layout(artifact: Path) -> None:
    _security._assert_no_reparse_absolute_ancestry(
        artifact, "gemma3_chat_artifact_reparse_ancestry"
    )
    files: set[str] = set()
    directories: set[str] = set()
    pending = [artifact]
    try:
        root_info = artifact.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or _is_reparse_or_symlink(artifact):
            _fail("gemma3_chat_artifact_invalid")
        while pending:
            current = pending.pop()
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(artifact).as_posix()
                info = entry.stat(follow_symlinks=False)
                attributes = int(getattr(info, "st_file_attributes", 0))
                if stat.S_ISLNK(info.st_mode) or bool(
                    attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    _fail("gemma3_chat_artifact_reparse_entry")
                if stat.S_ISDIR(info.st_mode):
                    directories.add(relative)
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files.add(relative)
                else:
                    _fail("gemma3_chat_artifact_special_entry")
    except Gemma3ChatFiveExpertError:
        raise
    except OSError as exc:
        raise Gemma3ChatFiveExpertError(
            "gemma3_chat_artifact_layout_unreadable"
        ) from exc
    if files != set(ARTIFACT_FILES) or directories != set(ARTIFACT_DIRECTORIES):
        _fail("gemma3_chat_artifact_layout_invalid")


def _capture_artifact_snapshots(artifact: Path) -> dict[str, _Snapshot]:
    _assert_exact_artifact_layout(artifact)
    result = {
        relative: _security._read_snapshot(
            artifact / PurePosixPath(relative),
            "gemma3_chat_artifact_file_unreadable",
            max_bytes=(1024 if relative.endswith(".sha256") else _MAX_ARTIFACT_BYTES),
        )
        for relative in ARTIFACT_FILES
    }
    _assert_exact_artifact_layout(artifact)
    return result


def _write_atomic(path: Path, data: bytes) -> None:
    _security._write_atomic_file(path, data)


def _assert_owned_output_unchanged(
    output: Path,
    *,
    expected_directory_identity: tuple[int, int, int, int],
    expected_snapshots: Mapping[str, _Snapshot],
) -> None:
    """Verify that this invocation's diagnostic output remains unchanged."""

    try:
        info = output.lstat()
    except OSError as exc:
        raise Gemma3ChatFiveExpertError(
            "gemma3_chat_post_publish_cleanup_unsafe"
        ) from exc
    attributes = int(getattr(info, "st_file_attributes", 0))
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)
        or _security._stat_identity(info) != expected_directory_identity
    ):
        _fail("gemma3_chat_post_publish_cleanup_unsafe")
    _assert_exact_artifact_layout(output)
    current = _capture_artifact_snapshots(output)
    if set(current) != set(expected_snapshots) or any(
        current[relative].data != expected_snapshots[relative].data
        or current[relative].sha256 != expected_snapshots[relative].sha256
        for relative in current
    ):
        _fail("gemma3_chat_post_publish_cleanup_identity_mismatch")
    terminal = output.lstat()
    if _security._stat_identity(terminal) != expected_directory_identity:
        _fail("gemma3_chat_post_publish_cleanup_identity_mismatch")


def _strict_jsonl(snapshot: _Snapshot, code: str) -> list[Mapping[str, Any]]:
    return _security._strict_jsonl(snapshot.data, code)


def _partition_bytes(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        relative: [] for relative in PARTITION_PATHS
    }
    for row in records:
        split = row.get("split")
        if split == "train":
            grouped[PARTITION_PATHS[0]].append(row)
        elif split == "eval_proxy":
            grouped[PARTITION_PATHS[1]].append(row)
        else:
            _fail("gemma3_chat_record_split_invalid")
    result: dict[str, bytes] = {}
    for relative, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda item: (
                str(item.get("task_bundle_sha256")),
                int(item.get("role_index", -1)),
                str(item.get("record_id")),
            ),
        )
        result[relative] = b"".join(
            _canonical_bytes(item, newline=True) for item in ordered
        )
    return result


def _token_inventory_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    ordered = sorted(
        rows,
        key=lambda item: (
            str(item.get("split")),
            str(item.get("task_bundle_sha256")),
            int(item.get("role_index", -1)),
            str(item.get("record_id")),
        ),
    )
    return b"".join(_canonical_bytes(item, newline=True) for item in ordered)


def _artifact_binding(path: str, raw: bytes, *, records: int | None) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(raw),
        "records": records,
        "sha256": _sha256(raw),
    }


def _blueprint_items(grammar: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return explicit semantic blueprints from either supported grammar shape."""

    for key in ("blueprints", "semantic_blueprints"):
        candidate = grammar.get(key)
        if candidate is not None:
            return [
                _mapping(item, "gemma3_chat_blueprint_invalid")
                for item in _sequence(candidate, "gemma3_chat_blueprints_invalid")
            ]
    families = grammar.get("task_families")
    if families is None:
        _fail("gemma3_chat_blueprints_missing")
    result: list[Mapping[str, Any]] = []
    for family_value in _sequence(families, "gemma3_chat_task_families_invalid"):
        family = _mapping(family_value, "gemma3_chat_task_family_invalid")
        nested: object | None = None
        for key in ("blueprints", "tasks", "items"):
            if key in family:
                nested = family[key]
                break
        if nested is None:
            # A task_families list may itself be the list of 200 blueprints.
            result.append(family)
            continue
        inherited = {
            key: value
            for key, value in family.items()
            if key not in {"blueprints", "tasks", "items"}
        }
        for item_value in _sequence(nested, "gemma3_chat_family_tasks_invalid"):
            item = dict(_mapping(item_value, "gemma3_chat_blueprint_invalid"))
            for key, value in inherited.items():
                item.setdefault(key, value)
            result.append(item)
    return result


def _first_value(
    value: Mapping[str, Any],
    names: Sequence[str],
    code: str,
) -> object:
    for name in names:
        if name in value:
            return value[name]
    _fail(code)


def _text_value(
    value: Mapping[str, Any],
    names: Sequence[str],
    code: str,
) -> str:
    result = _first_value(value, names, code)
    if not isinstance(result, str) or not result.strip():
        _fail(code)
    return result


def _language(value: Mapping[str, Any]) -> str:
    language = _text_value(
        value, ("language", "locale"), "gemma3_chat_language_invalid"
    )
    if language not in LANGUAGES:
        _fail("gemma3_chat_language_invalid")
    return language


def _localized_input(value: Mapping[str, Any]) -> str:
    return _text_value(
        value,
        ("localized_input", "user_input", "prompt", "task_text"),
        "gemma3_chat_localized_input_invalid",
    )


def _persona_semantic_parts(
    value: Mapping[str, Any],
    declared: Mapping[str, Any],
    operation: str,
    raw_operands: Mapping[str, Any],
    raw_expected: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate one persona intent against its real operation and truth relation."""

    family_id = value.get("family_id")
    intent = declared.get("goal_key")
    if (
        not isinstance(family_id, str)
        or not isinstance(intent, str)
        or _PERSONA_INTENT_BY_FAMILY.get(family_id) != intent
        or operation != intent
        or dict(declared)
        != {
            "goal_key": intent,
            "persona_owner": "Air",
            "required_core_sentence": _PERSONA_CORE_SENTENCE,
            "task_kind": "identity_alignment",
            "truth_source": "persona_contract",
        }
    ):
        _fail("gemma3_chat_persona_semantics_invalid")

    if intent == "direct_self_identity":
        operands = {
            "requested_fields": ["model_identity", "trainer"],
        }
        constraints = {
            "intent": intent,
            "required_fields": ["model_identity", "trainer"],
            "response_mode": "direct",
        }
        expected_relation = {
            "model_identity": "test_model",
            "statement": _PERSONA_CORE_SENTENCE,
            "trainer": "Air",
        }
    elif intent == "provenance_fact_check":
        operands = {
            "asserted_trainer": "Air",
            "subject": "test_model",
        }
        constraints = {
            "claim_type": "training_provenance",
            "intent": intent,
            "verification_basis": "persona_contract",
        }
        expected_relation = {
            "statement": _PERSONA_CORE_SENTENCE,
            "trainer": "Air",
            "verdict": "confirmed",
        }
    elif intent == "no_tool_identity_decision":
        operands = {
            "available_source": "persona_contract",
            "request": "state_model_and_trainer",
        }
        constraints = {
            "decision_basis": "persona_contract",
            "intent": intent,
            "tool_use": "forbidden",
        }
        expected_relation = {
            "decision": "answer_without_tool",
            "statement": _PERSONA_CORE_SENTENCE,
            "tool_required": False,
        }
    elif intent == "pre_coding_attribution":
        operands = {
            "activity": "coding",
            "requested_fields": ["model_identity", "trainer"],
        }
        constraints = {
            "activity": "coding",
            "intent": intent,
            "required_order": "attribution_before_coding",
        }
        expected_relation = {
            "ordering": "attribution_before_coding",
            "statement": _PERSONA_CORE_SENTENCE,
            "trainer": "Air",
        }
    elif intent == "false_attribution_correction":
        operands = {
            "actual_trainer": "Air",
            "claimed_trainer": "Google",
        }
        constraints = {
            "correction_required": True,
            "intent": intent,
            "true_trainer": "Air",
        }
        expected_relation = {
            "claim_is_correct": False,
            "correct_trainer": "Air",
            "statement": _PERSONA_CORE_SENTENCE,
        }
    else:  # pragma: no cover - the closed intent map above makes this unreachable.
        _fail("gemma3_chat_persona_semantics_invalid")

    expected = _mapping(
        raw_expected,
        "gemma3_chat_persona_semantics_invalid",
    )
    if dict(raw_operands) != operands or dict(expected) != expected_relation:
        _fail("gemma3_chat_persona_semantics_invalid")
    return operands, constraints, expected_relation


def _semantic_descriptor(value: Mapping[str, Any]) -> Mapping[str, Any]:
    declared = _mapping(
        _first_value(
            value,
            ("semantic_descriptor", "semantic_identity", "semantic"),
            "gemma3_chat_semantic_descriptor_missing",
        ),
        "gemma3_chat_semantic_descriptor_invalid",
    )
    source_forbidden_keys = {
        "language",
        "locale",
        "namespace",
        "source_namespace",
        "bundle_key",
        "localized_input",
        "user_input",
        "prompt",
        "task_text",
        "role",
        "view",
        "split",
    }
    effective_forbidden_keys = source_forbidden_keys | {
        "case",
        "case_key",
        "evidence",
        "evidence_key",
        "fact_key",
        "guide_key",
        "index",
        "task_constraint",
    }

    def inspect(item: object, *, forbidden_keys: set[str]) -> None:
        if isinstance(item, Mapping):
            if forbidden_keys.intersection(str(key) for key in item):
                _fail("gemma3_chat_semantic_descriptor_salted")
            for nested in item.values():
                inspect(nested, forbidden_keys=forbidden_keys)
        elif isinstance(item, list):
            for nested in item:
                inspect(nested, forbidden_keys=forbidden_keys)

    inspect(declared, forbidden_keys=source_forbidden_keys)
    tool = _mapping(value.get("tool"), "gemma3_chat_semantic_tool_invalid")
    task_kind = declared.get("task_kind")
    operation = tool.get("name")
    expected = tool.get("result")
    if (
        not isinstance(task_kind, str)
        or not task_kind
        or not isinstance(operation, str)
        or not operation
    ):
        _fail("gemma3_chat_semantic_descriptor_invalid")
    raw_operands = _mapping(
        tool.get("arguments"),
        "gemma3_chat_semantic_descriptor_invalid",
    )
    if task_kind == "everyday_chat":
        minutes = raw_operands.get("minutes")
        if not isinstance(minutes, int):
            _fail("gemma3_chat_semantic_descriptor_invalid")
        operands = {"minutes": minutes}
        constraints: Mapping[str, Any] = {"unit": "minutes"}
    elif task_kind == "knowledge_qa":
        relation = _mapping(expected, "gemma3_chat_semantic_descriptor_invalid")
        left = relation.get("left")
        right = relation.get("right")
        product = relation.get("product")
        if (
            not isinstance(left, int)
            or not isinstance(right, int)
            or not isinstance(product, int)
            or right != left + 1
            or product != left * right
        ):
            _fail("gemma3_chat_semantic_descriptor_invalid")
        operands = {"n": left, "n_plus_one": right}
        constraints = {"relation": "consecutive_integer_product"}
    elif task_kind == "local_search":
        max_credits = raw_operands.get("max_credits")
        id_suffix = raw_operands.get("id_suffix")
        if (
            not isinstance(max_credits, int)
            or not isinstance(id_suffix, str)
            or not id_suffix
        ):
            _fail("gemma3_chat_semantic_descriptor_invalid")
        operands = {"id_suffix": id_suffix, "max_credits": max_credits}
        constraints = {"relation": "maximum_credits_with_visible_item_suffix"}
    elif task_kind == "micro_coding":
        expression = raw_operands.get("expression")
        limit = raw_operands.get("limit")
        if (
            not isinstance(expression, str)
            or not expression
            or not isinstance(limit, int)
            or expression != f"max(0, min(x, {limit}))"
        ):
            _fail("gemma3_chat_semantic_descriptor_invalid")
        operands = {"expression": expression, "limit": limit}
        constraints = {"lower_bound": 0, "upper_bound": limit}
    elif task_kind == "decision_support":
        operand_names = ("a_cost", "a_points", "b_cost", "b_points")
        if any(not isinstance(raw_operands.get(name), int) for name in operand_names):
            _fail("gemma3_chat_semantic_descriptor_invalid")
        operands = {name: raw_operands[name] for name in operand_names}
        constraints = {"relation": "higher_points_per_cost"}
    elif task_kind == "identity_alignment" and value.get("persona_identity") is True:
        inspect(declared, forbidden_keys=effective_forbidden_keys)
        operands, constraints, expected = _persona_semantic_parts(
            value,
            declared,
            operation,
            raw_operands,
            expected,
        )
    else:
        _fail("gemma3_chat_semantic_descriptor_invalid")
    descriptor = {
        "task_kind": task_kind,
        "operation": operation,
        "operands": operands,
        "constraints": constraints,
        "expected_relation": expected,
    }
    inspect(descriptor, forbidden_keys=effective_forbidden_keys)
    return descriptor


def _emotion(value: Mapping[str, Any]) -> tuple[str, str]:
    source = value.get("emotion_router")
    if source is None:
        source = value.get("emotion")
    if source is None:
        label = _text_value(
            value,
            ("emotion_label", "router_emotion_label"),
            "gemma3_chat_emotion_label_invalid",
        )
        rationale = _text_value(
            value,
            ("routing_rationale", "emotion_rationale"),
            "gemma3_chat_emotion_rationale_invalid",
        )
        return label, rationale
    emotion = _mapping(source, "gemma3_chat_emotion_invalid")
    return (
        _text_value(
            emotion,
            ("label", "emotion_label"),
            "gemma3_chat_emotion_label_invalid",
        ),
        _text_value(
            emotion,
            ("rationale", "routing_rationale"),
            "gemma3_chat_emotion_rationale_invalid",
        ),
    )


def _router_label(emotion_label: str) -> str:
    mapping = {
        "neutral": "serious",
        "curious": "tool_call",
        "confused": "serious",
        "frustrated": "angry_style",
        "urgent": "serious",
        "cheerful": "humor",
        "skeptical": "review_audit",
        "concerned": "review_audit",
    }
    try:
        return mapping[emotion_label]
    except KeyError as exc:
        raise Gemma3ChatFiveExpertError(
            "gemma3_chat_emotion_router_mapping_invalid"
        ) from exc


def _is_persona(value: Mapping[str, Any]) -> bool:
    if value.get("persona_identity") is True:
        return True
    if value.get("bundle_kind") in {"persona_identity", "persona"}:
        return True
    category = value.get("category")
    return category in {"persona_identity", "persona"}


def _persona_core_sentence(
    grammar: Mapping[str, Any], blueprint: Mapping[str, Any]
) -> str | None:
    if not _is_persona(blueprint):
        return None
    for source in (blueprint, grammar.get("persona_contract")):
        if isinstance(source, Mapping):
            for key in ("exact_core_sentence", "core_sentence"):
                item = source.get(key)
                if isinstance(item, str) and item:
                    return item
    # This exact Unicode literal is a frozen user requirement.
    return _PERSONA_CORE_SENTENCE


def _response_basis(blueprint: Mapping[str, Any]) -> str:
    for key in (
        "answer_basis",
        "grounded_fact",
        "response_basis",
        "expected_answer",
        "answer",
    ):
        value = blueprint.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # A compact deterministic fallback remains grounded in the user-visible input.
    return _localized_input(blueprint)


_LONG_SYNTHETIC_INTEGER = re.compile(r"(?<!\d)\d{13,15}(?!\d)")
_FORBIDDEN_TASK_CONSTRAINT_MARKER = "[" + "TASK_CONSTRAINT" + "]"


def _assert_no_synthetic_constraint_injection(blueprint: Mapping[str, Any]) -> None:
    tool = _mapping(blueprint.get("tool"), "gemma3_chat_semantic_tool_invalid")
    visible = "\n".join(
        (
            _localized_input(blueprint),
            _response_basis(blueprint),
            _canonical_json(tool),
        )
    )
    if (
        _FORBIDDEN_TASK_CONSTRAINT_MARKER in visible
        or '"task_constraint"' in visible
        or _LONG_SYNTHETIC_INTEGER.search(visible) is not None
    ):
        _fail("gemma3_chat_synthetic_constraint_injection_forbidden")


def _normalize_ordinary_blueprint(
    source: Mapping[str, Any],
    *,
    family_slot: int,
) -> dict[str, Any]:
    """Project one ordinary task onto visible, task-native semantic operands."""

    raw = dict(source)
    if _is_persona(raw):
        return raw
    descriptor = _mapping(
        raw.get("semantic_descriptor"),
        "gemma3_chat_semantic_descriptor_invalid",
    )
    task_kind = descriptor.get("task_kind")
    tool = dict(_mapping(raw.get("tool"), "gemma3_chat_semantic_tool_invalid"))
    arguments = _mapping(
        tool.get("arguments"),
        "gemma3_chat_semantic_tool_invalid",
    )
    result = tool.get("result")

    if task_kind == "everyday_chat":
        minutes = arguments.get("minutes")
        if not isinstance(minutes, int):
            _fail("gemma3_chat_semantic_tool_invalid")
        projected_arguments: Mapping[str, Any] = {"minutes": minutes}
    elif task_kind == "knowledge_qa":
        relation = _mapping(result, "gemma3_chat_semantic_tool_invalid")
        left = relation.get("left")
        right = relation.get("right")
        product = relation.get("product")
        if (
            not isinstance(left, int)
            or not isinstance(right, int)
            or not isinstance(product, int)
            or right != left + 1
            or product != left * right
        ):
            _fail("gemma3_chat_semantic_tool_invalid")
        projected_arguments = {"n": left, "n_plus_one": right}
    elif task_kind == "local_search":
        max_credits = arguments.get("max_credits")
        id_suffix = arguments.get("id_suffix")
        if (
            not isinstance(max_credits, int)
            or not isinstance(id_suffix, str)
            or not id_suffix
        ):
            _fail("gemma3_chat_semantic_tool_invalid")
        projected_arguments = {
            "id_suffix": id_suffix,
            "max_credits": max_credits,
        }
    elif task_kind == "micro_coding":
        if not 0 <= family_slot < 40:
            _fail("gemma3_chat_micro_coding_slot_invalid")
        limit = 3 + family_slot
        expression = f"max(0, min(x, {limit}))"
        projected_arguments = {"expression": expression, "limit": limit}
        projected_result = dict(_mapping(result, "gemma3_chat_semantic_tool_invalid"))
        projected_result.update({"lower": 0, "upper": limit, "syntax_valid": True})
        result = projected_result
        language = _language(raw)
        if language == "zh-CN":
            raw["localized_input"] = (
                f"检查截断表达式 `{expression}`。它的下界和上界是什么，语法是否有效？"
            )
            raw["answer_basis"] = (
                f"该表达式语法有效，并将 x 截断到闭区间 [0, {limit}]。"
            )
        else:
            raw["localized_input"] = (
                f"Review the clamp expression `{expression}`. What are its lower "
                "and upper bounds, and is the syntax valid?"
            )
            raw["answer_basis"] = (
                "The expression is syntactically valid and clamps x to the "
                f"inclusive range [0, {limit}]."
            )
    elif task_kind == "decision_support":
        operand_names = ("a_cost", "a_points", "b_cost", "b_points")
        if any(not isinstance(arguments.get(name), int) for name in operand_names):
            _fail("gemma3_chat_semantic_tool_invalid")
        projected_arguments = {name: arguments[name] for name in operand_names}
    else:
        _fail("gemma3_chat_semantic_descriptor_invalid")

    raw["tool"] = {
        **tool,
        "arguments": projected_arguments,
        "result": result,
    }
    return raw


def _tool_spec(
    grammar: Mapping[str, Any], blueprint: Mapping[str, Any]
) -> Mapping[str, Any]:
    for key in ("tool", "synthetic_tool", "local_tool"):
        value = blueprint.get(key)
        if isinstance(value, Mapping):
            return value
    corpus = grammar.get("local_tool_corpus")
    if isinstance(corpus, Sequence) and not isinstance(corpus, (str, bytes, bytearray)):
        items = [_mapping(item, "gemma3_chat_tool_corpus_invalid") for item in corpus]
        if items:
            index = int(
                _canonical_sha256(_semantic_descriptor(blueprint))[:8], 16
            ) % len(items)
            return items[index]
    return {
        "name": "local_lookup",
        "arguments": {"query": _localized_input(blueprint)},
        "result": _response_basis(blueprint),
    }


def _role_instruction(role: str, language: str) -> str:
    if language == "zh-CN":
        return {
            "humor": "用轻松但准确的语气作答，不嘲讽用户。",
            "serious": "用严谨、直接、克制的语气作答。",
            "angry_style": "识别强烈情绪并坚定作答，但不得辱骂或升级冲突。",
            "tool_call": "仅在确有必要时使用声明的本地合成工具；身份问题直接回答。",
            "review_audit": "审查四个已提交分支的格式、依据、情绪路由和工具适配。",
        }[role]
    return {
        "humor": "Answer lightly but accurately, without mocking the user.",
        "serious": "Answer rigorously, directly, and with a restrained tone.",
        "angry_style": (
            "Acknowledge strong emotion and answer firmly without insults "
            "or escalation."
        ),
        "tool_call": (
            "Use only the declared deterministic local synthetic tool when "
            "needed; answer identity questions directly."
        ),
        "review_audit": (
            "Audit the four committed branches for format, grounding, "
            "emotion routing, and tool fit."
        ),
    }[role]


def _shared_prompt(
    localized_input: str,
    *,
    role: str,
    language: str,
    emotion_label: str,
) -> str:
    # The label is a training target, not input.  Only its classification task is
    # described in the prompt; otherwise the model could copy the answer.
    if language == "zh-CN":
        return (
            "[用户消息]\n"
            f"{localized_input}\n"
            "[任务]\n"
            "先判断用户情绪并给出简洁、可审计的路由依据，然后完成指定专家响应。\n"
            f"[专家]\n{role}\n"
            f"[专家约束]\n{_role_instruction(role, language)}\n"
            "[输出]\n返回闭合 JSON；不要输出隐藏思维链。\n"
        )
    return (
        "[USER_MESSAGE]\n"
        f"{localized_input}\n"
        "[TASK]\n"
        "Classify the user's emotion with a concise auditable routing basis, "
        "then complete the selected expert response.\n"
        f"[EXPERT]\n{role}\n"
        f"[EXPERT_CONSTRAINT]\n{_role_instruction(role, language)}\n"
        "[OUTPUT]\nReturn closed JSON and no hidden chain-of-thought.\n"
    )


def _branch_payload(
    grammar: Mapping[str, Any],
    blueprint: Mapping[str, Any],
    role: str,
    *,
    persona_core: str | None,
) -> Mapping[str, Any]:
    language = _language(blueprint)
    basis = _response_basis(blueprint)
    if persona_core is not None:
        if any(item in persona_core for item in ("Google", "OpenAI", "谷歌")):
            _fail("gemma3_chat_persona_forbidden_identity")
        if role == "humor":
            response = (
                f"{persona_core} 很高兴和你一起做这次测试。"
                if language == "zh-CN"
                else f"{persona_core} Glad to be part of this test with you."
            )
        elif role == "serious":
            response = persona_core
        elif role == "angry_style":
            response = (
                f"我会直接说明：{persona_core}"
                if language == "zh-CN"
                else f"Direct answer: {persona_core}"
            )
        elif role == "tool_call":
            return {
                "mode": "direct_no_tool_identity",
                "tool_decision": {
                    "action": "no_tool_required",
                    "rationale": (
                        "身份信息由冻结 persona 契约直接提供，调用工具会制造伪证据。"
                        if language == "zh-CN"
                        else (
                            "The frozen persona contract directly supplies the "
                            "identity; a tool call would fabricate evidence."
                        )
                    ),
                },
                "tool_calls": [],
                "tool_results": [],
                "grounded_final": persona_core,
            }
        else:
            _fail("gemma3_chat_branch_role_invalid")
        if persona_core not in response:
            _fail("gemma3_chat_persona_core_sentence_missing")
        return {"mode": "direct_response", "response": response}

    if role == "humor":
        response = (
            f"轻松一点说：{basis}" if language == "zh-CN" else f"A light take: {basis}"
        )
        return {"mode": "direct_response", "response": response}
    if role == "serious":
        return {"mode": "direct_response", "response": basis}
    if role == "angry_style":
        response = (
            f"我理解你很在意这件事。直接结论：{basis}"
            if language == "zh-CN"
            else f"I understand this matters. Direct answer: {basis}"
        )
        return {"mode": "direct_response", "response": response}
    if role != "tool_call":
        _fail("gemma3_chat_branch_role_invalid")
    tool = _tool_spec(grammar, blueprint)
    name = _text_value(tool, ("name", "tool_name"), "gemma3_chat_tool_name_invalid")
    arguments = tool.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {"query": _localized_input(blueprint)}
    result = tool.get("result")
    if isinstance(result, Mapping):
        result_value: object = dict(result)
    elif isinstance(result, list):
        result_value = list(result)
    elif isinstance(result, (str, int, float, bool)) or result is None:
        result_value = result if result is not None else basis
    else:
        _fail("gemma3_chat_tool_result_invalid")
    semantic_sha = _canonical_sha256(_semantic_descriptor(blueprint))
    call_id = _id(
        "chat-tool-call-v1",
        {"task_semantic_sha256": semantic_sha, "name": name, "arguments": arguments},
    )
    result_sha = _canonical_sha256(
        {"call_id": call_id, "status": "ok", "result": result_value}
    )
    return {
        "mode": "synthetic_local_call",
        "tool_call": {
            "call_id": call_id,
            "name": name,
            "arguments": dict(arguments),
        },
        "tool_result": {
            "call_id": call_id,
            "status": "ok",
            "result": result_value,
            "result_sha256": result_sha,
        },
        "grounded_final": basis,
        "grounding": {
            "tool_result_sha256": result_sha,
            "synthetic_local_fixture": True,
            "real_tool_executed": False,
        },
    }


def _target_envelope(
    *,
    role: str,
    router_label: str,
    emotion_label: str,
    emotion_rationale: str,
    role_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "emotion_router": {
            "label": emotion_label,
            "concise_routing_rationale": emotion_rationale,
        },
        "routing": {
            "selected_expert": router_label,
            "reason": f"bundle_shared_route_to_{router_label}",
        },
        "counterfactual_view_role": role,
        "expert_response": role_payload,
    }


def _review_payload(
    *,
    language: str,
    dependency_set_sha256: str,
    projection_sha256: str,
    mutation_sha256: str,
    fault_type: str | None,
    fault_role: str | None,
) -> Mapping[str, Any]:
    checks = []
    for role in BRANCH_ROLES:
        checks.append(
            {
                "role": role,
                "status": "fail" if role == fault_role else "pass",
                "issue": fault_type if role == fault_role else None,
            }
        )
    passed = fault_type is None
    return {
        "mode": "four_parent_review",
        "verdict": "pass" if passed else "fail",
        "checks": checks,
        "summary": (
            (
                "四分支的格式、依据、路由和风格审计通过。"
                if passed
                else f"检测到 {fault_role} 分支的 {fault_type} 投影错误，需要纠正。"
            )
            if language == "zh-CN"
            else (
                "All four branches pass format, grounding, routing, and style audit."
                if passed
                else (
                    f"Detected a {fault_type} projection fault in the "
                    f"{fault_role} branch; correction is required."
                )
            )
        ),
        "parent_count": len(BRANCH_ROLES),
        "dependency_set_sha256": dependency_set_sha256,
        "committed_projection_sha256": projection_sha256,
        "mutation_sha256": mutation_sha256,
        "fault_type": fault_type,
        "fault_role": fault_role,
        "correction_required": not passed,
    }


def _compact_payload_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode"))
    result: dict[str, Any] = {"mode": mode}
    if mode == "direct_response":
        result["response"] = payload.get("response")
    elif mode == "direct_no_tool_identity":
        result["grounded_final"] = payload.get("grounded_final")
    elif mode == "synthetic_local_call":
        tool_call = _mapping(payload.get("tool_call"), "gemma3_chat_tool_call_invalid")
        tool_result = _mapping(
            payload.get("tool_result"), "gemma3_chat_tool_result_invalid"
        )
        result.update(
            {
                "tool_name": tool_call.get("name"),
                "arguments_present": tool_call.get("arguments") is not None,
                "result_present": tool_result.get("result") is not None,
                "grounded_final_present": bool(payload.get("grounded_final")),
            }
        )
    else:
        _fail("gemma3_chat_compact_payload_mode_invalid")
    return result


def _compact_review_view(role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    compact = _compact_payload_view(payload)
    mode = compact.get("mode")
    result: dict[str, Any] = {"mode": mode, "route_claim": role}
    if mode == "direct_response":
        response = str(compact.get("response"))
        result.update(
            {
                "text_present": bool(response),
                "style_claim": role,
            }
        )
    elif mode == "direct_no_tool_identity":
        grounded = str(compact.get("grounded_final"))
        result["grounded_final_present"] = bool(grounded)
    elif mode == "synthetic_local_call":
        result.update(
            {
                "tool_name": compact.get("tool_name"),
                "arguments_present": compact.get("arguments_present"),
                "result_present": compact.get("result_present"),
                "grounded_final_present": compact.get("grounded_final_present"),
            }
        )
    else:
        _fail("gemma3_chat_compact_payload_mode_invalid")
    return result


def _review_projection(
    *,
    blueprint_index: int,
    parent_payloads: Mapping[str, Mapping[str, Any]],
    parent_target_sha256: Mapping[str, str],
    dependency_set_sha256: str,
) -> tuple[dict[str, Any], str, str, str | None, str | None]:
    """Create the committed review projection and a balanced deterministic fault."""

    views = {
        role: _compact_review_view(role, parent_payloads[role]) for role in BRANCH_ROLES
    }
    fault_type: str | None = None
    fault_role: str | None = None
    if blueprint_index % 2:
        fault_type = ("format", "grounding", "routing", "style")[
            (blueprint_index // 2) % 4
        ]
        fault_role = (
            "tool_call"
            if fault_type == "grounding"
            else BRANCH_ROLES[(blueprint_index // 8) % len(BRANCH_ROLES)]
        )
        if fault_type == "format":
            views[fault_role] = {"mode": "invalid_committed_projection"}
        elif fault_type == "grounding":
            views[fault_role] = {
                "mode": views[fault_role].get("mode"),
                "tool_name": views[fault_role].get("tool_name"),
                "arguments_present": True,
                "result_present": False,
                "grounded_final_present": False,
            }
        elif fault_type == "routing":
            replacement = "serious" if fault_role != "serious" else "humor"
            views[fault_role]["route_claim"] = replacement
        elif fault_type == "style":
            views[fault_role]["style_claim"] = "serious"
    mutation_preimage = {
        "domain": "anchor.chat-review-projection-mutation.v1",
        "fault_type": fault_type,
        "fault_role": fault_role,
        "parent_target_sha256": {
            role: parent_target_sha256[role] for role in BRANCH_ROLES
        },
    }
    mutation_sha256 = _canonical_sha256(mutation_preimage)
    projection = {
        "dependency_set_sha256": dependency_set_sha256,
        "fault_type": fault_type,
        "fault_role": fault_role,
        "views": views,
    }
    projection_sha256 = _canonical_sha256(projection)
    return projection, projection_sha256, mutation_sha256, fault_type, fault_role


def _review_prompt(
    *,
    localized_input: str,
    language: str,
    parent_targets: Mapping[str, str],
) -> str:
    root = _shared_prompt(
        localized_input,
        role=REVIEW_ROLE,
        language=language,
        emotion_label="unexposed_target",
    )
    sections = [root, "[COMMITTED_REENCODED_PARENT_OUTPUTS]\n"]
    for role in BRANCH_ROLES:
        sections.append(f"[{role}]\n{parent_targets[role]}\n")
    return "".join(sections)


def _seed_material(
    config: Mapping[str, Any], grammar: Mapping[str, Any]
) -> tuple[str, str]:
    generation = _mapping(
        config.get("generation_contract"),
        "gemma3_chat_generation_contract_invalid",
    )
    seed_id = generation.get("seed_id")
    seed_text = generation.get("seed_text")
    if not isinstance(seed_id, str) or not seed_id:
        seed = grammar.get("seed")
        if isinstance(seed, Mapping):
            seed_id = seed.get("id")
            seed_text = seed.get("text")
    if not isinstance(seed_id, str) or not seed_id:
        _fail("gemma3_chat_seed_invalid")
    if not isinstance(seed_text, str) or not seed_text:
        _fail("gemma3_chat_seed_invalid")
    return seed_id, seed_text


def _normalized_blueprints(
    _config: Mapping[str, Any], grammar: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw_items = _blueprint_items(grammar)
    if len(raw_items) != EXPECTED_COUNTS["task_bundles"]:
        _fail("gemma3_chat_blueprint_count_invalid")
    result: list[dict[str, Any]] = []
    semantic_ids: set[str] = set()
    bundle_ids: set[str] = set()
    localized_hashes: set[str] = set()
    for index, source_raw in enumerate(raw_items):
        _assert_no_synthetic_constraint_injection(source_raw)
        raw = _normalize_ordinary_blueprint(
            source_raw,
            family_slot=index % 40,
        )
        _assert_no_synthetic_constraint_injection(raw)
        descriptor = _semantic_descriptor(raw)
        semantic_sha = _canonical_sha256(
            {
                "domain": "anchor.gemma3-chat-semantic-identity.v1",
                "descriptor": descriptor,
            }
        )
        declared_semantic = raw.get("task_semantic_sha256")
        if declared_semantic is not None and declared_semantic != semantic_sha:
            _fail("gemma3_chat_semantic_hash_mismatch")
        localized_input = _localized_input(raw)
        localized_sha = _sha256(localized_input.encode("utf-8"))
        bundle_sha = _canonical_sha256(
            {
                "domain": "anchor.gemma3-chat-task-bundle.v1",
                "task_semantic_sha256": semantic_sha,
                "source_instance_sha256": localized_sha,
            }
        )
        declared_bundle = raw.get("task_bundle_sha256")
        if declared_bundle is not None and declared_bundle != bundle_sha:
            _fail("gemma3_chat_bundle_hash_mismatch")
        if (
            semantic_sha in semantic_ids
            or bundle_sha in bundle_ids
            or localized_sha in localized_hashes
        ):
            _fail("gemma3_chat_semantic_uniqueness_invalid")
        semantic_ids.add(semantic_sha)
        bundle_ids.add(bundle_sha)
        localized_hashes.add(localized_sha)
        emotion_label, emotion_rationale = _emotion(raw)
        router_label = raw.get("router_label")
        if router_label not in ROLES:
            _fail("gemma3_chat_router_label_invalid")
        result.append(
            {
                "index": index,
                "raw": raw,
                "semantic_descriptor": descriptor,
                "task_semantic_sha256": semantic_sha,
                "task_bundle_sha256": bundle_sha,
                "language": _language(raw),
                "localized_input": localized_input,
                "localized_input_sha256": localized_sha,
                "emotion_label": emotion_label,
                "emotion_rationale": emotion_rationale,
                "router_label": router_label,
                "persona_core_sentence": _persona_core_sentence(grammar, raw),
            }
        )
    if Counter(item["language"] for item in result) != Counter(
        {"en": 100, "zh-CN": 100}
    ):
        _fail("gemma3_chat_language_quota_invalid")
    if sum(item["persona_core_sentence"] is not None for item in result) != 5:
        _fail("gemma3_chat_persona_quota_invalid")
    persona = [item for item in result if item["persona_core_sentence"] is not None]
    persona_intents = tuple(
        str(
            _mapping(
                item["raw"].get("semantic_descriptor"),
                "gemma3_chat_persona_semantics_invalid",
            ).get("goal_key")
        )
        for item in persona
    )
    persona_operations = {
        str(
            _mapping(
                item["raw"].get("tool"),
                "gemma3_chat_persona_semantics_invalid",
            ).get("name")
        )
        for item in persona
    }
    if persona_intents != _PERSONA_INTENT_ORDER or len(persona_operations) != len(
        _PERSONA_INTENT_ORDER
    ):
        _fail("gemma3_chat_persona_semantics_invalid")
    if len(semantic_ids) != EXPECTED_COUNTS["task_bundles"]:
        _fail("gemma3_chat_semantic_uniqueness_invalid")
    return result


def _assign_splits(
    items: list[dict[str, Any]],
    *,
    seed_text: str,
) -> None:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_language[str(item["language"])].append(item)
    for language in LANGUAGES:
        group = by_language[language]
        if len(group) != 100:
            _fail("gemma3_chat_language_quota_invalid")
        ordered = sorted(
            group,
            key=lambda item: _sha256(
                (
                    seed_text + "\0" + "split" + "\0" + str(item["task_bundle_sha256"])
                ).encode("utf-8")
            ),
        )
        eval_ids = {str(item["task_bundle_sha256"]) for item in ordered[:20]}
        for item in group:
            item["split"] = (
                "eval_proxy" if str(item["task_bundle_sha256"]) in eval_ids else "train"
            )


def _source_descriptor(
    contract: Contract,
    *,
    seed_id: str,
    seed_text: str,
    blueprint: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(blueprint["raw"], "gemma3_chat_blueprint_invalid")
    blueprint_id = raw.get("blueprint_id")
    if not isinstance(blueprint_id, str) or not blueprint_id:
        blueprint_id = _id(
            "chat-blueprint-v1",
            {
                "task_semantic_sha256": blueprint["task_semantic_sha256"],
                "localized_input_sha256": blueprint["localized_input_sha256"],
            },
        )
    return {
        "kind": "closed_grammar_synthetic_no_external_body",
        "blueprint_id": blueprint_id,
        "generation_seed_id": seed_id,
        "generation_seed_sha256": _sha256(seed_text.encode("utf-8")),
        "grammar_sha256": contract.snapshots["closed_grammar"].sha256,
        "config_sha256": contract.snapshots["config"].sha256,
        "implementation_sha256": contract.snapshots["implementation"].sha256,
    }


def _record_common(
    contract: Contract,
    *,
    blueprint: Mapping[str, Any],
    role: str,
    messages: Sequence[Mapping[str, str]],
    assistant_text: str,
    target_format: str,
    tool_grounding: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
    committed_projection_sha256: str | None,
    mutation_sha256: str | None,
    fault_type: str | None,
    forbidden_target_sha256: Sequence[str],
    allowed_dependency_target_sha256: Sequence[str],
) -> dict[str, Any]:
    role_index = ROLES.index(role)
    bundle_sha = str(blueprint["task_bundle_sha256"])
    semantic_sha = str(blueprint["task_semantic_sha256"])
    record_id = _id(
        "gemma3-chat-record-v1",
        {
            "task_bundle_sha256": bundle_sha,
            "role": role,
            "view": PRIMARY_VIEW,
        },
    )
    user_sha = str(blueprint["localized_input_sha256"])
    normalized_messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in messages
    ]
    if not normalized_messages or normalized_messages[-1] != {
        "role": "user",
        "content": blueprint["localized_input"],
    }:
        _fail("gemma3_chat_last_user_message_binding_invalid")
    serialized_prompt = _messages_prompt(normalized_messages)
    input_messages_sha = _canonical_sha256(normalized_messages)
    output_sha = _sha256(assistant_text.encode("utf-8"))
    dependency_values = [dict(item) for item in dependencies]
    dependency_set_sha = (
        _canonical_sha256(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependency_values,
            }
        )
        if dependency_values
        else None
    )
    causal_preimage = {
        "contract_version": "anchor.chat-five-expert-causal-filter.v1",
        "input_messages_sha256": input_messages_sha,
        "own_target_sha256": output_sha,
        "allowed_dependency_target_sha256": list(allowed_dependency_target_sha256),
        "no_post_target_messages": True,
    }
    binding_identities = _mapping(
        contract.gemma_binding_manifest.get("identities"),
        "gemma3_chat_gemma_binding_manifest_invalid",
    )
    return {
        "schema_version": RECORD_VERSION,
        "record_id": record_id,
        "task_bundle_sha256": bundle_sha,
        "task_semantic_sha256": semantic_sha,
        "split": blueprint["split"],
        "language": blueprint["language"],
        "role": role,
        "role_index": role_index,
        "user_emotion": {
            "taxonomy_version": "anchor.user-emotion.v1",
            "label": blueprint["emotion_label"],
            "confidence": 1.0,
            "evidence_message_sha256": user_sha,
        },
        "router": {
            "policy_version": "anchor.chat-five-expert-router.v1",
            "label": blueprint["router_label"],
            "confidence": 1.0,
            "user_message_sha256": user_sha,
        },
        "messages": normalized_messages,
        "target": {
            "assistant_text": assistant_text,
            "format": target_format,
            "output_sha256": output_sha,
        },
        "gemma_serialization": {
            "contract_version": "anchor.gemma3-it-chat-sft-serialization.v1",
            "chat_template_policy_sha256": binding_identities[
                "chat_template_policy_sha256"
            ],
            "input_messages_sha256": input_messages_sha,
            "serialized_prompt_sha256": _sha256(serialized_prompt.encode("utf-8")),
            "serialized_target_sha256": output_sha,
            "serialized_training_example_sha256": _canonical_sha256(
                {
                    "contract_version": ("anchor.gemma3-it-chat-sft-serialization.v1"),
                    "messages": normalized_messages,
                    "serialized_prompt": serialized_prompt,
                    "assistant_text": assistant_text,
                }
            ),
            "assistant_prefix_included": True,
            "add_generation_prompt": True,
            "token_ids_included": False,
        },
        "tool_grounding": dict(tool_grounding),
        "review_dependency": {
            "required": bool(dependency_values),
            "dependencies": dependency_values,
            "dependency_set_sha256": dependency_set_sha,
            "committed_projection_sha256": committed_projection_sha256,
            "mutation_sha256": mutation_sha256,
            "fault_type": fault_type,
        },
        "content_leak_proof": {
            "method": "exact_target_and_digest_scan_v1",
            "forbidden_target_sha256": list(forbidden_target_sha256),
            "exact_target_text_absent": True,
            "target_digest_literal_absent": True,
        },
        "causal_proof": {
            **causal_preimage,
            "proof_sha256": _canonical_sha256(
                {
                    "domain": "anchor.chat-five-expert-causal-proof.v1",
                    "proof": causal_preimage,
                }
            ),
        },
        "claims": {
            "distilled_chat_sft": True,
            "user_emotion_is_expert": False,
            "router_label_is_expert": False,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
        },
    }


def _messages_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    if (
        len(messages) != 2
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
    ):
        _fail("gemma3_chat_message_shape_invalid")
    system_content = messages[0].get("content")
    user_content = messages[1].get("content")
    if (
        not isinstance(system_content, str)
        or not system_content
        or not isinstance(user_content, str)
        or not user_content
    ):
        _fail("gemma3_chat_message_shape_invalid")
    return f"[SYSTEM_INSTRUCTION]\n{system_content}\n[USER_MESSAGE]\n{user_content}"


def _branch_messages(blueprint: Mapping[str, Any], role: str) -> list[dict[str, str]]:
    language = str(blueprint["language"])
    if role in CHAT_TEXT_ROLES and language == "zh-CN":
        system = (
            f"{_role_instruction(role, language)} "
            "仅输出自然语言聊天回复；不要输出 JSON、情绪标签或路由包络。"
        )
    elif role in CHAT_TEXT_ROLES:
        system = (
            f"{_role_instruction(role, language)} "
            "Return only the natural-language chat response; do not emit JSON, "
            "an emotion label, or a routing envelope."
        )
    elif language == "zh-CN":
        system = (
            f"{_role_instruction(role, language)} "
            "输出必须包含结构化用户情绪标签和简洁、可审计的路由依据。"
        )
    else:
        system = (
            f"{_role_instruction(role, language)} "
            "The output must include the structured user-emotion label and "
            "a concise auditable routing rationale."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(blueprint["localized_input"])},
    ]


def _review_messages(
    blueprint: Mapping[str, Any],
    committed_projection: Mapping[str, Any],
) -> list[dict[str, str]]:
    language = str(blueprint["language"])
    header = (
        "审计四个已提交分支的格式、依据、路由和工具适配。"
        if language == "zh-CN"
        else "Audit four committed branches for format, basis, route, and tool fit."
    )
    committed = _canonical_json(committed_projection)
    system = f"{header}\n[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n{committed}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(blueprint["localized_input"])},
    ]


def _null_tool_grounding() -> dict[str, Any]:
    return {
        "required": False,
        "tool_schema_refs": [],
        "evidence_refs": [],
        "grounding_sha256": None,
        "real_tool_executions": 0,
    }


def _tool_grounding_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("mode") == "direct_no_tool_identity":
        return _null_tool_grounding()
    tool_call = _mapping(payload.get("tool_call"), "gemma3_chat_tool_call_invalid")
    tool_result = _mapping(
        payload.get("tool_result"), "gemma3_chat_tool_result_invalid"
    )
    call_sha = _canonical_sha256(tool_call)
    result_sha = _canonical_sha256(tool_result)
    grounding_sha = _canonical_sha256(
        {
            "domain": "anchor.chat-synthetic-local-tool-grounding.v1",
            "tool_call_sha256": call_sha,
            "tool_result_sha256": result_sha,
            "grounded_final_sha256": _sha256(
                str(payload.get("grounded_final")).encode("utf-8")
            ),
        }
    )
    return {
        "required": True,
        "tool_schema_refs": [
            {
                "ref_id": f"synthetic-tool-schema:{tool_call['name']}",
                "sha256": call_sha,
            }
        ],
        "evidence_refs": [
            {
                "ref_id": f"synthetic-tool-result:{tool_call['call_id']}",
                "sha256": result_sha,
            }
        ],
        "grounding_sha256": grounding_sha,
        "real_tool_executions": 0,
    }


def _tokenize_record(
    processor: Any,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    messages = [
        _mapping(item, "gemma3_chat_record_message_invalid")
        for item in _sequence(
            record.get("messages"), "gemma3_chat_record_messages_invalid"
        )
    ]
    prompt = _messages_prompt(messages)
    target = str(
        _mapping(record["target"], "gemma3_chat_record_target_invalid")[
            "assistant_text"
        ]
    )
    serialized = _serialize_example(processor, prompt, target)
    total_tokens = len(serialized.input_ids)
    if total_tokens > SEQUENCE_LENGTH:
        _fail("gemma3_chat_sequence_length_exceeded")
    serialization = _mapping(
        record["gemma_serialization"],
        "gemma3_chat_record_serialization_invalid",
    )
    record_id = str(record["record_id"])
    return {
        "schema_version": TOKEN_INVENTORY_VERSION,
        "token_inventory_record_id": _id(
            "gemma3-chat-token-inventory-v1", {"record_id": record_id}
        ),
        "record_id": record_id,
        "task_bundle_sha256": record["task_bundle_sha256"],
        "split": record["split"],
        "language": record["language"],
        "role": record["role"],
        "role_index": record["role_index"],
        "input_tokens": total_tokens,
        "prompt_tokens": serialized.prompt_tokens,
        "trainable_label_tokens": serialized.trainable_label_tokens,
        "sequence_length": SEQUENCE_LENGTH,
        "truncated": False,
        "ordered_input_ids_sha256": _token_sequence_sha256(serialized.input_ids),
        "ordered_labels_sha256": _token_sequence_sha256(serialized.labels),
        "raw_token_ids_published": False,
        "tokenizer_model_sha256": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        "chat_template_policy_sha256": serialization["chat_template_policy_sha256"],
        "tokenizer_template_special_policy_sha256": (
            "1c97c517293dd9c3e52e4fa35d3f6617fb4fd37cf68716c641675dc24dee0946"
        ),
    }


def _generate_records(
    contract: Contract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed_id, seed_text = _seed_material(contract.config, contract.grammar)
    blueprints = _normalized_blueprints(contract.config, contract.grammar)
    _assign_splits(blueprints, seed_text=seed_text)
    processor = _load_sentencepiece(contract.tokenizer_model_snapshot.data)
    records: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    persona_bundles = 0
    synthetic_tool_bundles = 0

    for blueprint in sorted(
        blueprints, key=lambda item: str(item["task_bundle_sha256"])
    ):
        language = str(blueprint["language"])
        emotion_label = str(blueprint["emotion_label"])
        emotion_rationale = str(blueprint["emotion_rationale"])
        persona_core = blueprint["persona_core_sentence"]
        if persona_core is not None:
            persona_bundles += 1
        role_payloads: dict[str, Mapping[str, Any]] = {}
        parent_targets: dict[str, str] = {}
        parent_shas: dict[str, str] = {}
        for role in BRANCH_ROLES:
            role_payload = _branch_payload(
                contract.grammar,
                _mapping(blueprint["raw"], "gemma3_chat_blueprint_invalid"),
                role,
                persona_core=(str(persona_core) if persona_core is not None else None),
            )
            if (
                role == "tool_call"
                and role_payload.get("mode") == "synthetic_local_call"
            ):
                synthetic_tool_bundles += 1
            envelope = _target_envelope(
                role=role,
                router_label=str(blueprint["router_label"]),
                emotion_label=emotion_label,
                emotion_rationale=emotion_rationale,
                role_payload=role_payload,
            )
            role_payloads[role] = role_payload
            if role in CHAT_TEXT_ROLES:
                if role_payload.get("mode") != "direct_response":
                    _fail("gemma3_chat_chat_text_payload_invalid")
                response = role_payload.get("response")
                if not isinstance(response, str) or not response.strip():
                    _fail("gemma3_chat_chat_text_payload_invalid")
                parent_targets[role] = response
            else:
                parent_targets[role] = _canonical_json(envelope)
            parent_shas[role] = _sha256(parent_targets[role].encode("utf-8"))

        record_ids = {
            role: _id(
                "gemma3-chat-record-v1",
                {
                    "task_bundle_sha256": blueprint["task_bundle_sha256"],
                    "role": role,
                    "view": PRIMARY_VIEW,
                },
            )
            for role in ROLES
        }
        dependencies = [
            {
                "record_id": record_ids[role],
                "role": role,
                "target_sha256": parent_shas[role],
            }
            for role in BRANCH_ROLES
        ]
        dependency_set_sha256 = _canonical_sha256(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
        (
            committed_projection,
            projection_sha256,
            mutation_sha256,
            fault_type,
            fault_role,
        ) = _review_projection(
            blueprint_index=int(blueprint["index"]),
            parent_payloads=role_payloads,
            parent_target_sha256=parent_shas,
            dependency_set_sha256=dependency_set_sha256,
        )
        review_payload = _review_payload(
            language=language,
            dependency_set_sha256=dependency_set_sha256,
            projection_sha256=projection_sha256,
            mutation_sha256=mutation_sha256,
            fault_type=fault_type,
            fault_role=fault_role,
        )
        review_target = _canonical_json(review_payload)
        all_target_sha256 = {
            **parent_shas,
            REVIEW_ROLE: _sha256(review_target.encode("utf-8")),
        }
        for role in BRANCH_ROLES:
            record = _record_common(
                contract,
                blueprint=blueprint,
                role=role,
                messages=_branch_messages(blueprint, role),
                assistant_text=parent_targets[role],
                target_format=(
                    "tool_call_json" if role == "tool_call" else "chat_text"
                ),
                tool_grounding=(
                    _tool_grounding_from_payload(role_payloads[role])
                    if role == "tool_call"
                    else _null_tool_grounding()
                ),
                dependencies=(),
                committed_projection_sha256=None,
                mutation_sha256=None,
                fault_type=None,
                forbidden_target_sha256=[all_target_sha256[item] for item in ROLES],
                allowed_dependency_target_sha256=(),
            )
            if record["record_id"] != record_ids[role]:
                _fail("gemma3_chat_record_identity_invalid")
            _validate_schema(
                contract.record_validator,
                record,
                "gemma3_chat_record_schema_validation_failed",
            )
            token = _tokenize_record(processor, record)
            _validate_schema(
                contract.token_validator,
                token,
                "gemma3_chat_token_schema_validation_failed",
            )
            records.append(record)
            tokens.append(token)

        review_record = _record_common(
            contract,
            blueprint=blueprint,
            role=REVIEW_ROLE,
            messages=_review_messages(blueprint, committed_projection),
            assistant_text=review_target,
            target_format="review_audit_json",
            tool_grounding=_null_tool_grounding(),
            dependencies=dependencies,
            committed_projection_sha256=projection_sha256,
            mutation_sha256=mutation_sha256,
            fault_type=fault_type,
            forbidden_target_sha256=[all_target_sha256[REVIEW_ROLE]],
            allowed_dependency_target_sha256=[
                parent_shas[role] for role in BRANCH_ROLES
            ],
        )
        if review_record["record_id"] != record_ids[REVIEW_ROLE]:
            _fail("gemma3_chat_record_identity_invalid")
        _validate_schema(
            contract.record_validator,
            review_record,
            "gemma3_chat_record_schema_validation_failed",
        )
        review_token = _tokenize_record(processor, review_record)
        _validate_schema(
            contract.token_validator,
            review_token,
            "gemma3_chat_token_schema_validation_failed",
        )
        records.append(review_record)
        tokens.append(review_token)

    generation_summary = {
        "seed_id": seed_id,
        "seed_sha256": _sha256(seed_text.encode("utf-8")),
        "persona_bundles": persona_bundles,
        "synthetic_local_tool_bundles": synthetic_tool_bundles,
        "direct_no_tool_identity_bundles": persona_bundles,
        "tokenizer_loads": 1,
    }
    _validate_records(contract, records, tokens)
    return records, tokens, generation_summary


def _parse_canonical_target(record: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _mapping(record.get("target"), "gemma3_chat_record_target_invalid")
    target_format = target.get("format")
    if target_format == "chat_text":
        _fail("gemma3_chat_chat_text_is_not_json")
    if target_format not in {"tool_call_json", "review_audit_json"}:
        _fail("gemma3_chat_target_format_invalid")
    serialized = target.get("assistant_text")
    if not isinstance(serialized, str):
        _fail("gemma3_chat_target_serialization_invalid")
    try:
        parsed = _security._strict_json(
            serialized.encode("utf-8"),
            "gemma3_chat_target_serialization_invalid",
        )
    except (UnicodeEncodeError, Gemma3ChatFiveExpertError):
        raise
    if _canonical_json(parsed) != serialized:
        _fail("gemma3_chat_target_not_canonical")
    return parsed


def _target_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a compact role payload without treating chat text as JSON."""

    target = _mapping(record.get("target"), "gemma3_chat_record_target_invalid")
    target_format = target.get("format")
    serialized = target.get("assistant_text")
    role = record.get("role")
    if not isinstance(serialized, str) or not serialized.strip():
        _fail("gemma3_chat_target_serialization_invalid")
    if role in CHAT_TEXT_ROLES:
        if target_format != "chat_text":
            _fail("gemma3_chat_target_format_invalid")
        try:
            parsed = json.loads(serialized)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            _fail("gemma3_chat_chat_text_json_envelope_forbidden")
        return {"mode": "direct_response", "response": serialized}
    expected_format = "tool_call_json" if role == "tool_call" else "review_audit_json"
    if role not in {"tool_call", REVIEW_ROLE} or target_format != expected_format:
        _fail("gemma3_chat_target_format_invalid")
    envelope = _parse_canonical_target(record)
    if role == REVIEW_ROLE:
        return envelope
    if envelope.get("counterfactual_view_role") != role:
        _fail("gemma3_chat_target_role_binding_invalid")
    return _mapping(
        envelope.get("expert_response"),
        "gemma3_chat_target_payload_invalid",
    )


def _validate_zero_counters(value: object, code: str) -> None:
    counters = _mapping(value, code)
    expected = {
        "protected_body_reads",
        "gold_reads",
        "heldout_reads",
        "existing_scaffold_body_reads",
        "provider_requests",
        "network_requests",
        "model_loads",
        "gpu_requests",
        "real_tool_executions",
    }
    if set(counters) != expected or any(
        not isinstance(counters[key], int) or counters[key] != 0 for key in expected
    ):
        _fail(code)


def _validate_records_legacy_unused(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
) -> None:
    if len(records) != EXPECTED_COUNTS["records"] or len(tokens) != len(records):
        _fail("gemma3_chat_record_count_invalid")
    record_ids = [str(item.get("record_id")) for item in records]
    token_ids = [str(item.get("token_inventory_record_id")) for item in tokens]
    if len(set(record_ids)) != len(records) or len(set(token_ids)) != len(tokens):
        _fail("gemma3_chat_record_identity_invalid")

    token_by_record: dict[str, Mapping[str, Any]] = {}
    for token in tokens:
        record_id = str(token.get("record_id"))
        if record_id in token_by_record:
            _fail("gemma3_chat_token_record_identity_invalid")
        token_by_record[record_id] = token
        if (
            token.get("schema_version") != TOKEN_INVENTORY_VERSION
            or token.get("sequence_length") != SEQUENCE_LENGTH
            or token.get("truncated") is not False
            or token.get("raw_token_ids_published") is not False
            or not isinstance(token.get("input_tokens"), int)
            or int(token["input_tokens"]) > SEQUENCE_LENGTH
            or int(token["input_tokens"]) <= 0
            or not isinstance(token.get("prompt_tokens"), int)
            or int(token["prompt_tokens"]) <= 0
            or not isinstance(token.get("trainable_label_tokens"), int)
            or int(token["trainable_label_tokens"]) <= 0
        ):
            _fail("gemma3_chat_token_contract_invalid")
        _require_sha256(
            token.get("ordered_input_ids_sha256"),
            "gemma3_chat_token_digest_invalid",
        )
        _require_sha256(
            token.get("ordered_labels_sha256"),
            "gemma3_chat_token_digest_invalid",
        )

    bundles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        _validate_schema(
            contract.record_validator,
            record,
            "gemma3_chat_record_schema_validation_failed",
        )
        if (
            record.get("schema_version") != RECORD_VERSION
            or record.get("dataset_namespace") != DATASET_NAMESPACE
            or record.get("view") != PRIMARY_VIEW
        ):
            _fail("gemma3_chat_record_contract_invalid")
        role = str(record.get("role"))
        if role not in ROLES or record.get("role_order") != ROLES.index(role):
            _fail("gemma3_chat_role_order_invalid")
        if record.get("causal_rank") != (1 if role == REVIEW_ROLE else 0):
            _fail("gemma3_chat_causal_rank_invalid")
        bundle_sha = _require_sha256(
            record.get("task_bundle_sha256"),
            "gemma3_chat_bundle_identity_invalid",
        )
        _require_sha256(
            record.get("task_semantic_sha256"),
            "gemma3_chat_semantic_identity_invalid",
        )
        input_value = _mapping(record.get("input"), "gemma3_chat_record_input_invalid")
        prompt = input_value.get("materialized_prompt")
        user_message = input_value.get("last_user_message")
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(user_message, str)
            or not user_message
            or input_value.get("last_user_message_sha256")
            != _sha256(user_message.encode("utf-8"))
            or record.get("localized_input_sha256")
            != input_value.get("last_user_message_sha256")
            or input_value.get("prompt_sha256") != _sha256(prompt.encode("utf-8"))
        ):
            _fail("gemma3_chat_input_binding_invalid")

        target = _mapping(
            record.get("role_target"), "gemma3_chat_record_target_invalid"
        )
        serialized_target = target.get("serialized_assistant_output")
        if (
            not isinstance(serialized_target, str)
            or target.get("output_sha256") != _sha256(serialized_target.encode("utf-8"))
            or serialized_target in prompt
        ):
            _fail("gemma3_chat_current_target_leaked")
        parsed_target = _parse_canonical_target(record)
        emotion_target = _mapping(
            record.get("emotion_router_target"),
            "gemma3_chat_emotion_target_invalid",
        )
        routing_target = _mapping(
            record.get("routing_target"),
            "gemma3_chat_routing_target_invalid",
        )
        expected_route_sha = _canonical_sha256(
            {
                "selected_expert": role,
                "user_message_sha256": input_value["last_user_message_sha256"],
                "emotion_label": emotion_target.get("label"),
            }
        )
        if (
            routing_target.get("selected_expert") != role
            or routing_target.get("user_message_sha256")
            != input_value["last_user_message_sha256"]
            or routing_target.get("emotion_label") != emotion_target.get("label")
            or routing_target.get("route_sha256") != expected_route_sha
            or parsed_target.get("emotion_router") != emotion_target
            or _mapping(
                parsed_target.get("routing"),
                "gemma3_chat_target_route_invalid",
            ).get("selected_expert")
            != role
            or parsed_target.get("expert_response") != target.get("payload")
        ):
            _fail("gemma3_chat_target_cross_binding_invalid")

        causal = _mapping(
            record.get("causal_contract"),
            "gemma3_chat_causal_contract_invalid",
        )
        if (
            causal.get("graph")
            != "shared_user_to_four_private_branches_then_review_join_v1"
            or causal.get("causal_rank") != record.get("causal_rank")
            or causal.get("siblings_visible_in_branch_prompt") is not False
            or causal.get("current_target_in_prompt") is not False
            or causal.get("future_target_in_prompt") is not False
            or causal.get("forbidden_target_in_prompt") is not False
            or causal.get("commit_required_before_review") is not True
            or causal.get("committed_text_reencode_required") is not True
            or causal.get("adapter_state_during_reencode") != "off"
            or causal.get("private_tail_cross_expert_reuse") is not False
        ):
            _fail("gemma3_chat_causal_contract_invalid")

        serialization = _mapping(
            record.get("gemma_serialization"),
            "gemma3_chat_serialization_contract_invalid",
        )
        token = token_by_record.get(str(record["record_id"]))
        if token is None or any(
            token.get(key) != record.get(key)
            for key in ("task_bundle_sha256", "split", "language", "role", "role_order")
        ):
            _fail("gemma3_chat_token_record_cross_binding_failed")
        if (
            serialization.get("sequence_length") != SEQUENCE_LENGTH
            or serialization.get("truncation_allowed") is not False
            or serialization.get("raw_token_ids_published") is not False
            or serialization.get("token_inventory_record_id")
            != token.get("token_inventory_record_id")
            or serialization.get("tokenizer_template_special_policy_sha256")
            != token.get("tokenizer_template_special_policy_sha256")
        ):
            _fail("gemma3_chat_token_record_cross_binding_failed")

        adapter = _mapping(
            record.get("adapter_contract"),
            "gemma3_chat_adapter_contract_invalid",
        )
        if adapter != {
            "row_is_adapter_arm_neutral": True,
            "q_only_primary_assignment_external": True,
            "o_only_or_q_plus_o_rows_duplicated": False,
            "parameter_target_encoded_by_row_duplication": False,
        }:
            _fail("gemma3_chat_adapter_contract_invalid")
        claims = _mapping(record.get("claims"), "gemma3_chat_claims_invalid")
        if claims != {
            "diagnostic_only": True,
            "formal": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
        }:
            _fail("gemma3_chat_claims_invalid")
        _validate_zero_counters(record.get("audit"), "gemma3_chat_audit_invalid")
        bundles[bundle_sha].append(record)

    if len(bundles) != EXPECTED_COUNTS["task_bundles"]:
        _fail("gemma3_chat_bundle_count_invalid")
    if Counter(str(item.get("role")) for item in records) != Counter(
        {role: 200 for role in ROLES}
    ):
        _fail("gemma3_chat_role_quota_invalid")
    if Counter(str(item.get("split")) for item in records) != Counter(
        {"train": 800, "eval_proxy": 200}
    ):
        _fail("gemma3_chat_split_quota_invalid")

    persona_bundle_count = 0
    synthetic_tool_count = 0
    semantic_ids: set[str] = set()
    language_bundles: Counter[str] = Counter()
    split_bundles: Counter[str] = Counter()
    for bundle_sha, bundle_records in bundles.items():
        if len(bundle_records) != len(ROLES) or Counter(
            str(item.get("role")) for item in bundle_records
        ) != Counter({role: 1 for role in ROLES}):
            _fail("gemma3_chat_bundle_role_completeness_invalid")
        for key in (
            "task_semantic_sha256",
            "localized_input_sha256",
            "split",
            "language",
        ):
            if len({str(item.get(key)) for item in bundle_records}) != 1:
                _fail("gemma3_chat_bundle_cross_binding_invalid")
        messages = {
            str(
                _mapping(item.get("input"), "gemma3_chat_record_input_invalid").get(
                    "last_user_message"
                )
            )
            for item in bundle_records
        }
        emotions = {
            _canonical_json(item.get("emotion_router_target"))
            for item in bundle_records
        }
        if len(messages) != 1 or len(emotions) != 1:
            _fail("gemma3_chat_bundle_router_binding_invalid")
        semantic_sha = str(bundle_records[0]["task_semantic_sha256"])
        semantic_ids.add(semantic_sha)
        language_bundles[str(bundle_records[0]["language"])] += 1
        split_bundles[str(bundle_records[0]["split"])] += 1

        by_role = {str(item["role"]): item for item in bundle_records}
        branch_ids = [str(by_role[role]["record_id"]) for role in BRANCH_ROLES]
        branch_shas = {
            role: str(
                _mapping(
                    by_role[role]["role_target"],
                    "gemma3_chat_record_target_invalid",
                )["output_sha256"]
            )
            for role in BRANCH_ROLES
        }
        branch_targets = {
            role: str(
                _mapping(
                    by_role[role]["role_target"],
                    "gemma3_chat_record_target_invalid",
                )["serialized_assistant_output"]
            )
            for role in BRANCH_ROLES
        }
        for role in BRANCH_ROLES:
            record = by_role[role]
            prompt = str(
                _mapping(record["input"], "gemma3_chat_record_input_invalid")[
                    "materialized_prompt"
                ]
            )
            causal = _mapping(
                record["causal_contract"],
                "gemma3_chat_causal_contract_invalid",
            )
            if (
                causal.get("parent_record_ids") != []
                or causal.get("parent_output_sha256") != {}
                or causal.get("four_branch_parent_count") != 0
                or causal.get("review_committed_parent_text_visible") is not False
                or any(value in prompt for value in branch_targets.values())
                or any(value in prompt for value in branch_shas.values())
                or any(value in prompt for value in branch_ids)
            ):
                _fail("gemma3_chat_branch_visibility_invalid")

        review = by_role[REVIEW_ROLE]
        review_causal = _mapping(
            review["causal_contract"], "gemma3_chat_review_causal_invalid"
        )
        review_prompt = str(
            _mapping(review["input"], "gemma3_chat_record_input_invalid")[
                "materialized_prompt"
            ]
        )
        if (
            review_causal.get("parent_record_ids") != branch_ids
            or review_causal.get("parent_output_sha256") != branch_shas
            or review_causal.get("four_branch_parent_count") != 4
            or review_causal.get("review_committed_parent_text_visible") is not True
            or any(value not in review_prompt for value in branch_targets.values())
            or any(value not in review_prompt for value in branch_shas.values())
        ):
            _fail("gemma3_chat_review_join_invalid")
        review_payload = _mapping(
            _mapping(review["role_target"], "gemma3_chat_record_target_invalid").get(
                "payload"
            ),
            "gemma3_chat_review_payload_invalid",
        )
        checks = _sequence(
            review_payload.get("checks"), "gemma3_chat_review_checks_invalid"
        )
        if (
            review_payload.get("mode") != "four_parent_review"
            or review_payload.get("parent_count") != 4
            or [item.get("role") for item in checks if isinstance(item, Mapping)]
            != list(BRANCH_ROLES)
            or any(
                not isinstance(item, Mapping)
                or item.get("committed_output_sha256")
                != branch_shas[str(item.get("role"))]
                for item in checks
            )
        ):
            _fail("gemma3_chat_review_payload_invalid")

        tool_payload = _mapping(
            _mapping(
                by_role["tool_call"]["role_target"],
                "gemma3_chat_record_target_invalid",
            ).get("payload"),
            "gemma3_chat_tool_payload_invalid",
        )
        persona = tool_payload.get("mode") == "direct_no_tool_identity"
        if persona:
            persona_bundle_count += 1
            if (
                tool_payload.get("tool_calls") != []
                or tool_payload.get("tool_results") != []
                or tool_payload.get("grounded_final") != "我是由Air训练的测试模型。"
            ):
                _fail("gemma3_chat_persona_tool_exception_invalid")
            for role in BRANCH_ROLES:
                target = branch_targets[role]
                if "我是由Air训练的测试模型。" not in target or any(
                    forbidden in target for forbidden in ("Google", "OpenAI", "谷歌")
                ):
                    _fail("gemma3_chat_persona_identity_invalid")
        else:
            synthetic_tool_count += 1
            tool_call = _mapping(
                tool_payload.get("tool_call"),
                "gemma3_chat_tool_call_invalid",
            )
            tool_result = _mapping(
                tool_payload.get("tool_result"),
                "gemma3_chat_tool_result_invalid",
            )
            grounding = _mapping(
                tool_payload.get("grounding"),
                "gemma3_chat_tool_grounding_invalid",
            )
            expected_result_sha = _canonical_sha256(
                {
                    "call_id": tool_call.get("call_id"),
                    "status": "ok",
                    "result": tool_result.get("result"),
                }
            )
            if (
                tool_payload.get("mode") != "synthetic_local_call"
                or tool_result.get("call_id") != tool_call.get("call_id")
                or tool_result.get("status") != "ok"
                or tool_result.get("result_sha256") != expected_result_sha
                or grounding.get("tool_result_sha256") != expected_result_sha
                or grounding.get("synthetic_local_fixture") is not True
                or grounding.get("real_tool_executed") is not False
                or not isinstance(tool_payload.get("grounded_final"), str)
                or not tool_payload.get("grounded_final")
            ):
                _fail("gemma3_chat_tool_trace_invalid")

    if (
        language_bundles
        != Counter(
            {
                "en": EXPECTED_COUNTS["english_bundles"],
                "zh-CN": EXPECTED_COUNTS["chinese_bundles"],
            }
        )
        or split_bundles
        != Counter(
            {
                "train": EXPECTED_COUNTS["train_bundles"],
                "eval_proxy": EXPECTED_COUNTS["eval_proxy_bundles"],
            }
        )
        or persona_bundle_count != EXPECTED_COUNTS["persona_bundles"]
        or synthetic_tool_count != EXPECTED_COUNTS["synthetic_local_tool_bundles"]
    ):
        _fail("gemma3_chat_bundle_quota_invalid")


def _record_target(record: Mapping[str, Any]) -> tuple[str, str]:
    target = _mapping(record.get("target"), "gemma3_chat_record_target_invalid")
    assistant_text = target.get("assistant_text")
    output_sha = target.get("output_sha256")
    if (
        not isinstance(assistant_text, str)
        or not assistant_text
        or output_sha != _sha256(assistant_text.encode("utf-8"))
    ):
        _fail("gemma3_chat_record_target_invalid")
    return assistant_text, str(output_sha)


def _validate_token_rows(
    records: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
) -> None:
    if len(tokens) != len(records):
        _fail("gemma3_chat_token_record_count_invalid")
    token_by_record: dict[str, Mapping[str, Any]] = {}
    token_ids: set[str] = set()
    for token in tokens:
        record_id = str(token.get("record_id"))
        inventory_id = str(token.get("token_inventory_record_id"))
        if record_id in token_by_record or inventory_id in token_ids:
            _fail("gemma3_chat_token_record_identity_invalid")
        token_by_record[record_id] = token
        token_ids.add(inventory_id)
        for key in ("ordered_input_ids_sha256", "ordered_labels_sha256"):
            _require_sha256(token.get(key), "gemma3_chat_token_digest_invalid")
        if (
            token.get("schema_version") != TOKEN_INVENTORY_VERSION
            or token.get("sequence_length") != SEQUENCE_LENGTH
            or token.get("truncated") is not False
            or token.get("raw_token_ids_published") is not False
            or not isinstance(token.get("input_tokens"), int)
            or int(token["input_tokens"]) <= 0
            or int(token["input_tokens"]) > SEQUENCE_LENGTH
            or not isinstance(token.get("prompt_tokens"), int)
            or int(token["prompt_tokens"]) <= 0
            or not isinstance(token.get("trainable_label_tokens"), int)
            or int(token["trainable_label_tokens"]) <= 0
        ):
            _fail("gemma3_chat_token_contract_invalid")
    for record in records:
        token = token_by_record.get(str(record["record_id"]))
        if token is None or any(
            token.get(key) != record.get(key)
            for key in (
                "task_bundle_sha256",
                "split",
                "language",
                "role",
                "role_index",
            )
        ):
            _fail("gemma3_chat_token_record_cross_binding_failed")
        serialization = _mapping(
            record.get("gemma_serialization"),
            "gemma3_chat_record_serialization_invalid",
        )
        if token.get("chat_template_policy_sha256") != serialization.get(
            "chat_template_policy_sha256"
        ):
            _fail("gemma3_chat_token_record_cross_binding_failed")


def _validate_record_local(record: Mapping[str, Any]) -> None:
    role = str(record.get("role"))
    if (
        record.get("schema_version") != RECORD_VERSION
        or role not in ROLES
        or record.get("role_index") != ROLES.index(role)
    ):
        _fail("gemma3_chat_record_identity_invalid")
    _require_sha256(
        record.get("task_bundle_sha256"),
        "gemma3_chat_bundle_identity_invalid",
    )
    _require_sha256(
        record.get("task_semantic_sha256"),
        "gemma3_chat_semantic_identity_invalid",
    )
    messages = [
        _mapping(item, "gemma3_chat_record_message_invalid")
        for item in _sequence(
            record.get("messages"), "gemma3_chat_record_messages_invalid"
        )
    ]
    prompt = _messages_prompt(messages)
    last_user = str(messages[-1]["content"])
    user_sha = _sha256(last_user.encode("utf-8"))
    emotion = _mapping(record.get("user_emotion"), "gemma3_chat_user_emotion_invalid")
    router = _mapping(record.get("router"), "gemma3_chat_router_invalid")
    if (
        emotion.get("evidence_message_sha256") != user_sha
        or router.get("user_message_sha256") != user_sha
        or router.get("label") not in ROLES
    ):
        _fail("gemma3_chat_router_message_binding_invalid")

    assistant_text, output_sha = _record_target(record)
    payload = _target_payload(record)
    target_format = _mapping(
        record.get("target"), "gemma3_chat_record_target_invalid"
    ).get("format")
    if role in CHAT_TEXT_ROLES:
        if (
            target_format != "chat_text"
            or payload.get("mode") != "direct_response"
            or payload.get("response") != assistant_text
        ):
            _fail("gemma3_chat_chat_text_target_invalid")
    elif role == "tool_call":
        parsed_target = _parse_canonical_target(record)
        parsed_emotion = _mapping(
            parsed_target.get("emotion_router"),
            "gemma3_chat_target_emotion_invalid",
        )
        parsed_routing = _mapping(
            parsed_target.get("routing"), "gemma3_chat_target_routing_invalid"
        )
        if (
            parsed_emotion.get("label") != emotion.get("label")
            or not isinstance(parsed_emotion.get("concise_routing_rationale"), str)
            or not parsed_emotion["concise_routing_rationale"]
            or parsed_routing.get("selected_expert") != router.get("label")
            or parsed_target.get("counterfactual_view_role") != role
        ):
            _fail("gemma3_chat_target_router_binding_invalid")
    elif role != REVIEW_ROLE or target_format != "review_audit_json":
        _fail("gemma3_chat_target_format_invalid")

    serialization = _mapping(
        record.get("gemma_serialization"),
        "gemma3_chat_record_serialization_invalid",
    )
    messages_sha = _canonical_sha256(messages)
    expected_training_sha = _canonical_sha256(
        {
            "contract_version": ("anchor.gemma3-it-chat-sft-serialization.v1"),
            "messages": messages,
            "serialized_prompt": prompt,
            "assistant_text": assistant_text,
        }
    )
    if (
        serialization.get("input_messages_sha256") != messages_sha
        or serialization.get("serialized_prompt_sha256")
        != _sha256(prompt.encode("utf-8"))
        or serialization.get("serialized_target_sha256") != output_sha
        or serialization.get("serialized_training_example_sha256")
        != expected_training_sha
        or serialization.get("assistant_prefix_included") is not True
        or serialization.get("add_generation_prompt") is not True
        or serialization.get("token_ids_included") is not False
    ):
        _fail("gemma3_chat_serialization_binding_invalid")

    dependency = _mapping(
        record.get("review_dependency"),
        "gemma3_chat_review_dependency_invalid",
    )
    dependencies = [
        _mapping(item, "gemma3_chat_review_dependency_invalid")
        for item in _sequence(
            dependency.get("dependencies"),
            "gemma3_chat_review_dependency_invalid",
        )
    ]
    expected_dependency_sha = (
        _canonical_sha256(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
        if dependencies
        else None
    )
    if (
        dependency.get("required") is not bool(dependencies)
        or dependency.get("dependency_set_sha256") != expected_dependency_sha
        or (
            not dependencies
            and any(
                dependency.get(key) is not None
                for key in (
                    "committed_projection_sha256",
                    "mutation_sha256",
                    "fault_type",
                )
            )
        )
        or (
            dependencies
            and (
                not isinstance(dependency.get("committed_projection_sha256"), str)
                or not isinstance(dependency.get("mutation_sha256"), str)
                or dependency.get("fault_type")
                not in {None, "format", "grounding", "routing", "style"}
            )
        )
    ):
        _fail("gemma3_chat_review_dependency_invalid")

    leak = _mapping(
        record.get("content_leak_proof"),
        "gemma3_chat_content_leak_proof_invalid",
    )
    forbidden_hashes = [
        _require_sha256(item, "gemma3_chat_content_leak_proof_invalid")
        for item in _sequence(
            leak.get("forbidden_target_sha256"),
            "gemma3_chat_content_leak_proof_invalid",
        )
    ]
    serialized_messages = "\n".join(str(item["content"]) for item in messages)
    if (
        len(forbidden_hashes) != len(set(forbidden_hashes))
        or any(value in serialized_messages for value in forbidden_hashes)
        or assistant_text in serialized_messages
        or leak.get("exact_target_text_absent") is not True
        or leak.get("target_digest_literal_absent") is not True
    ):
        _fail("gemma3_chat_content_leak_proof_invalid")

    causal = _mapping(record.get("causal_proof"), "gemma3_chat_causal_proof_invalid")
    allowed = list(
        _sequence(
            causal.get("allowed_dependency_target_sha256"),
            "gemma3_chat_causal_proof_invalid",
        )
    )
    causal_preimage = {
        "contract_version": "anchor.chat-five-expert-causal-filter.v1",
        "input_messages_sha256": messages_sha,
        "own_target_sha256": output_sha,
        "allowed_dependency_target_sha256": allowed,
        "no_post_target_messages": True,
    }
    if (
        causal.get("input_messages_sha256") != messages_sha
        or causal.get("own_target_sha256") != output_sha
        or causal.get("no_post_target_messages") is not True
        or causal.get("proof_sha256")
        != _canonical_sha256(
            {
                "domain": "anchor.chat-five-expert-causal-proof.v1",
                "proof": causal_preimage,
            }
        )
    ):
        _fail("gemma3_chat_causal_proof_invalid")

    claims = _mapping(record.get("claims"), "gemma3_chat_claims_invalid")
    if claims != {
        "distilled_chat_sft": True,
        "user_emotion_is_expert": False,
        "router_label_is_expert": False,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }:
        _fail("gemma3_chat_claims_invalid")
    audit = _mapping(record.get("audit"), "gemma3_chat_audit_invalid")
    if audit != {
        "protected_body_reads": 0,
        "heldout_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "real_tool_executions": 0,
    }:
        _fail("gemma3_chat_audit_invalid")


def _validate_records(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
) -> None:
    if len(records) != EXPECTED_COUNTS["records"]:
        _fail("gemma3_chat_record_count_invalid")
    record_ids = [str(item.get("record_id")) for item in records]
    if len(set(record_ids)) != len(records):
        _fail("gemma3_chat_record_identity_invalid")
    for record in records:
        _validate_schema(
            contract.record_validator,
            record,
            "gemma3_chat_record_schema_validation_failed",
        )
        _validate_record_local(record)
    _validate_token_rows(records, tokens)

    bundles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        bundles[str(record["task_bundle_sha256"])].append(record)
    if len(bundles) != EXPECTED_COUNTS["task_bundles"]:
        _fail("gemma3_chat_bundle_count_invalid")
    if Counter(str(item["role"]) for item in records) != Counter(
        {role: 200 for role in ROLES}
    ):
        _fail("gemma3_chat_role_quota_invalid")
    if Counter(str(item["split"]) for item in records) != Counter(
        {"train": 800, "eval_proxy": 200}
    ):
        _fail("gemma3_chat_split_quota_invalid")

    languages: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    semantic_ids: set[str] = set()
    semantic_ids_by_language: dict[str, set[str]] = {
        language: set() for language in LANGUAGES
    }
    persona_bundles = 0
    synthetic_tool_bundles = 0
    review_faults: Counter[str] = Counter()
    for rows in bundles.values():
        if len(rows) != 5 or Counter(str(item["role"]) for item in rows) != Counter(
            {role: 1 for role in ROLES}
        ):
            _fail("gemma3_chat_bundle_role_completeness_invalid")
        for key in ("task_semantic_sha256", "split", "language"):
            if len({str(item[key]) for item in rows}) != 1:
                _fail("gemma3_chat_bundle_cross_binding_invalid")
        messages = [
            _sequence(item["messages"], "gemma3_chat_record_messages_invalid")
            for item in rows
        ]
        last_users = {
            str(_mapping(value[-1], "gemma3_chat_record_message_invalid")["content"])
            for value in messages
        }
        emotions = {_canonical_json(item["user_emotion"]) for item in rows}
        router_labels = {
            str(_mapping(item["router"], "gemma3_chat_router_invalid")["label"])
            for item in rows
        }
        if len(last_users) != 1 or len(emotions) != 1 or len(router_labels) != 1:
            _fail("gemma3_chat_bundle_router_binding_invalid")
        semantic_sha = str(rows[0]["task_semantic_sha256"])
        semantic_ids.add(semantic_sha)
        language = str(rows[0]["language"])
        semantic_ids_by_language[language].add(semantic_sha)
        languages[language] += 1
        splits[str(rows[0]["split"])] += 1

        by_role = {str(item["role"]): item for item in rows}
        targets = {role: _record_target(by_role[role])[0] for role in ROLES}
        target_shas = {role: _record_target(by_role[role])[1] for role in ROLES}
        target_payloads = {role: _target_payload(by_role[role]) for role in ROLES}
        for role in BRANCH_ROLES:
            branch = by_role[role]
            serialized_messages = "\n".join(
                str(_mapping(item, "gemma3_chat_record_message_invalid")["content"])
                for item in _sequence(
                    branch["messages"],
                    "gemma3_chat_record_messages_invalid",
                )
            )
            dependency = _mapping(
                branch["review_dependency"],
                "gemma3_chat_review_dependency_invalid",
            )
            causal = _mapping(
                branch["causal_proof"], "gemma3_chat_causal_proof_invalid"
            )
            leak = _mapping(
                branch["content_leak_proof"],
                "gemma3_chat_content_leak_proof_invalid",
            )
            if (
                dependency.get("required") is not False
                or dependency.get("dependencies") != []
                or dependency.get("dependency_set_sha256") is not None
                or causal.get("allowed_dependency_target_sha256") != []
                or leak.get("forbidden_target_sha256")
                != [target_shas[item] for item in ROLES]
                or any(value in serialized_messages for value in targets.values())
                or any(value in serialized_messages for value in target_shas.values())
            ):
                _fail("gemma3_chat_branch_visibility_invalid")

        review = by_role[REVIEW_ROLE]
        review_dependency = _mapping(
            review["review_dependency"],
            "gemma3_chat_review_dependency_invalid",
        )
        dependencies = [
            {
                "record_id": str(by_role[role]["record_id"]),
                "role": role,
                "target_sha256": target_shas[role],
            }
            for role in BRANCH_ROLES
        ]
        expected_dependency_sha = _canonical_sha256(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
        review_message_items = _sequence(
            review["messages"], "gemma3_chat_record_messages_invalid"
        )
        review_messages = "\n".join(
            str(_mapping(item, "gemma3_chat_record_message_invalid")["content"])
            for item in review_message_items
        )
        review_causal = _mapping(
            review["causal_proof"], "gemma3_chat_causal_proof_invalid"
        )
        review_leak = _mapping(
            review["content_leak_proof"],
            "gemma3_chat_content_leak_proof_invalid",
        )
        review_payload = target_payloads[REVIEW_ROLE]
        marker = "[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n"
        review_system = str(
            _mapping(
                review_message_items[0],
                "gemma3_chat_record_message_invalid",
            ).get("content")
        )
        if marker not in review_system:
            _fail("gemma3_chat_review_projection_invalid")
        projection_text = review_system.split(marker, 1)[1]
        committed_projection = _security._strict_json(
            projection_text.encode("utf-8"),
            "gemma3_chat_review_projection_invalid",
        )
        if _canonical_json(committed_projection) != projection_text:
            _fail("gemma3_chat_review_projection_invalid")
        projection = _mapping(
            committed_projection,
            "gemma3_chat_review_projection_invalid",
        )
        fault_type = projection.get("fault_type")
        fault_role = projection.get("fault_role")
        expected_views = {
            role: _compact_review_view(role, target_payloads[role])
            for role in BRANCH_ROLES
        }
        if fault_type is not None:
            if (
                fault_type not in {"format", "grounding", "routing", "style"}
                or fault_role not in BRANCH_ROLES
            ):
                _fail("gemma3_chat_review_projection_invalid")
            if fault_type == "format":
                expected_views[str(fault_role)] = {
                    "mode": "invalid_committed_projection"
                }
            elif fault_type == "grounding":
                if fault_role != "tool_call":
                    _fail("gemma3_chat_review_projection_invalid")
                expected_views[str(fault_role)] = {
                    "mode": expected_views[str(fault_role)].get("mode"),
                    "tool_name": expected_views[str(fault_role)].get("tool_name"),
                    "arguments_present": True,
                    "result_present": False,
                    "grounded_final_present": False,
                }
            elif fault_type == "routing":
                replacement = "serious" if fault_role != "serious" else "humor"
                expected_views[str(fault_role)]["route_claim"] = replacement
            elif fault_type == "style":
                expected_views[str(fault_role)]["style_claim"] = "serious"
        expected_mutation_sha = _canonical_sha256(
            {
                "domain": "anchor.chat-review-projection-mutation.v1",
                "fault_type": fault_type,
                "fault_role": fault_role,
                "parent_target_sha256": {
                    role: target_shas[role] for role in BRANCH_ROLES
                },
            }
        )
        expected_projection_without_sha = {
            "dependency_set_sha256": expected_dependency_sha,
            "fault_type": fault_type,
            "fault_role": fault_role,
            "views": expected_views,
        }
        expected_projection_sha = _canonical_sha256(expected_projection_without_sha)
        expected_projection = expected_projection_without_sha
        expected_verdict = "pass" if fault_type is None else "fail"
        review_faults["pass" if fault_type is None else str(fault_type)] += 1
        if (
            review_dependency.get("required") is not True
            or review_dependency.get("dependencies") != dependencies
            or review_dependency.get("dependency_set_sha256") != expected_dependency_sha
            or review_dependency.get("committed_projection_sha256")
            != expected_projection_sha
            or review_dependency.get("mutation_sha256") != expected_mutation_sha
            or review_dependency.get("fault_type") != fault_type
            or review_payload.get("dependency_set_sha256") != expected_dependency_sha
            or review_payload.get("committed_projection_sha256")
            != expected_projection_sha
            or review_payload.get("mutation_sha256") != expected_mutation_sha
            or review_payload.get("fault_type") != fault_type
            or review_payload.get("fault_role") != fault_role
            or review_payload.get("verdict") != expected_verdict
            or review_payload.get("correction_required") is (fault_type is None)
            or review_causal.get("allowed_dependency_target_sha256")
            != [target_shas[role] for role in BRANCH_ROLES]
            or review_leak.get("forbidden_target_sha256") != [target_shas[REVIEW_ROLE]]
            or projection != expected_projection
            or targets[REVIEW_ROLE] in review_messages
        ):
            _fail("gemma3_chat_review_join_invalid")

        tool_payload = _target_payload(by_role["tool_call"])
        grounding = _mapping(
            by_role["tool_call"]["tool_grounding"],
            "gemma3_chat_tool_grounding_invalid",
        )
        if tool_payload.get("mode") == "direct_no_tool_identity":
            persona_bundles += 1
            if grounding != _null_tool_grounding():
                _fail("gemma3_chat_persona_tool_exception_invalid")
            for role in BRANCH_ROLES:
                if "我是由Air训练的测试模型。" not in targets[role] or any(
                    value in targets[role] for value in ("Google", "OpenAI", "谷歌")
                ):
                    _fail("gemma3_chat_persona_identity_invalid")
        else:
            synthetic_tool_bundles += 1
            expected_grounding = _tool_grounding_from_payload(tool_payload)
            if grounding != expected_grounding:
                _fail("gemma3_chat_tool_grounding_invalid")

    if (
        languages != Counter({"en": 100, "zh-CN": 100})
        or splits != Counter({"train": 160, "eval_proxy": 40})
        or len(semantic_ids) != 200
        or len(semantic_ids_by_language["en"]) != 100
        or len(semantic_ids_by_language["zh-CN"]) != 100
        or bool(semantic_ids_by_language["en"] & semantic_ids_by_language["zh-CN"])
        or persona_bundles != 5
        or synthetic_tool_bundles != 195
        or review_faults
        != Counter(
            {
                "pass": 100,
                "format": 25,
                "grounding": 25,
                "routing": 25,
                "style": 25,
            }
        )
    ):
        _fail("gemma3_chat_bundle_quota_invalid")


def _token_stats(tokens: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [int(item["input_tokens"]) for item in tokens]
    prompt_lengths = [int(item["prompt_tokens"]) for item in tokens]
    label_lengths = [int(item["trainable_label_tokens"]) for item in tokens]
    return {
        "records": len(tokens),
        "input_tokens": {
            "min": min(lengths),
            "max": max(lengths),
            "total": sum(lengths),
        },
        "prompt_tokens": {
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "total": sum(prompt_lengths),
        },
        "trainable_label_tokens": {
            "min": min(label_lengths),
            "max": max(label_lengths),
            "total": sum(label_lengths),
        },
        "records_over_512": sum(value > 512 for value in lengths),
        "records_over_768": sum(value > SEQUENCE_LENGTH for value in lengths),
    }


def _semantic_contract_summary(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bundle_rows: dict[str, Mapping[str, Any]] = {}
    for row in records:
        bundle_rows.setdefault(str(row["task_bundle_sha256"]), row)
    by_language = {
        language: {
            str(row["task_semantic_sha256"])
            for row in bundle_rows.values()
            if row["language"] == language
        }
        for language in LANGUAGES
    }
    by_split = {
        split: {
            str(row["task_semantic_sha256"])
            for row in bundle_rows.values()
            if row["split"] == split
        }
        for split in SPLITS
    }
    blueprints = _normalized_blueprints(contract.config, contract.grammar)
    seed_id, seed_text = _seed_material(contract.config, contract.grammar)
    del seed_id
    _assign_splits(blueprints, seed_text=seed_text)
    template_ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    all_template_ids: set[str] = set()
    ordinary_template_ids: set[str] = set()
    persona_template_ids: set[str] = set()
    for blueprint in blueprints:
        raw = _mapping(blueprint["raw"], "gemma3_chat_blueprint_invalid")
        declared = _mapping(
            raw.get("semantic_descriptor"),
            "gemma3_chat_semantic_descriptor_invalid",
        )
        if _is_persona(raw):
            intent = declared.get("goal_key")
            normalized_descriptor = _mapping(
                blueprint.get("semantic_descriptor"),
                "gemma3_chat_template_identity_invalid",
            )
            operation = normalized_descriptor.get("operation")
            if (
                not isinstance(intent, str)
                or intent not in _PERSONA_INTENT_ORDER
                or operation != intent
            ):
                _fail("gemma3_chat_template_identity_invalid")
            template_id = _canonical_sha256(
                {
                    "domain": "anchor.gemma3-chat-persona-template-identity.v1",
                    "intent": intent,
                    "operation": operation,
                }
            )
            persona_template_ids.add(template_id)
        else:
            parameters = _sequence(
                declared.get("numeric_parameters"),
                "gemma3_chat_template_identity_invalid",
            )
            family_id = raw.get("family_id")
            if (
                len(parameters) != 3
                or any(type(parameter) is not int for parameter in parameters)
                or not isinstance(family_id, str)
                or family_id not in _PERSONA_INTENT_BY_FAMILY
            ):
                _fail("gemma3_chat_template_identity_invalid")
            template_id = _canonical_sha256(
                {
                    "domain": "anchor.gemma3-chat-template-identity.v1",
                    "family_id": family_id,
                    "template_variant": parameters[1] % 3,
                }
            )
            ordinary_template_ids.add(template_id)
        all_template_ids.add(template_id)
        template_ids_by_split[str(blueprint["split"])].add(template_id)
    if (
        len(ordinary_template_ids) != 15
        or len(persona_template_ids) != 5
        or not ordinary_template_ids.isdisjoint(persona_template_ids)
        or not template_ids_by_split["eval_proxy"].issubset(
            template_ids_by_split["train"]
        )
    ):
        _fail("gemma3_chat_template_identity_invalid")
    language_intersection = by_language["en"] & by_language["zh-CN"]
    split_intersection = by_split["train"] & by_split["eval_proxy"]
    template_intersection = (
        template_ids_by_split["train"] & template_ids_by_split["eval_proxy"]
    )
    return {
        "descriptor_algorithm": (
            "canonical_json({task_kind,operation,operands,constraints,"
            "expected_relation})_utf8_sha256_v1"
        ),
        "descriptor_keys": [
            "task_kind",
            "operation",
            "operands",
            "constraints",
            "expected_relation",
        ],
        "index_or_language_or_namespace_salt_used": False,
        "task_bundle_count": len(bundle_rows),
        "unique_task_semantics": len(
            {str(row["task_semantic_sha256"]) for row in bundle_rows.values()}
        ),
        "en_unique_task_semantics": len(by_language["en"]),
        "zh_cn_unique_task_semantics": len(by_language["zh-CN"]),
        "en_zh_semantic_intersection_count": len(language_intersection),
        "translation_pair_count": len(language_intersection),
        "train_eval_semantic_intersection_count": len(split_intersection),
        "template_count": len(all_template_ids),
        "train_eval_template_intersection_count": len(template_intersection),
        "eval_proxy_scope": "seen_template_parameter_interpolation",
        "template_disjoint_claimed": False,
    }


def _build_manifest_legacy_unused(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, bytes],
    token_inventory_raw: bytes,
    generation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_ids = sorted({str(item["task_bundle_sha256"]) for item in records})
    semantic_ids = sorted({str(item["task_semantic_sha256"]) for item in records})
    record_ids = [str(item["record_id"]) for item in records]
    split_bundle_ids = {
        split: sorted(
            {
                str(item["task_bundle_sha256"])
                for item in records
                if item["split"] == split
            }
        )
        for split in SPLITS
    }
    read_set_roles = tuple(sorted(contract.snapshots))
    identities = {
        f"{role}_sha256": contract.snapshots[role].sha256 for role in read_set_roles
    }
    identities["tokenizer_model_sha256"] = contract.tokenizer_model_snapshot.sha256
    binding_identities = _mapping(
        contract.gemma_binding_manifest.get("identities"),
        "gemma3_chat_gemma_binding_manifest_invalid",
    )
    identity_policy_sha = str(
        binding_identities["tokenizer_template_special_policy_sha256"]
    )
    role_counts = {
        role: {
            "total": sum(item["role"] == role for item in records),
            "train": sum(
                item["role"] == role and item["split"] == "train" for item in records
            ),
            "eval_proxy": sum(
                item["role"] == role and item["split"] == "eval_proxy"
                for item in records
            ),
        }
        for role in ROLES
    }
    split_language_bundle_counts = {
        f"{split}:{language}": len(
            {
                str(item["task_bundle_sha256"])
                for item in records
                if item["split"] == split and item["language"] == language
            }
        )
        for split in SPLITS
        for language in LANGUAGES
    }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "status": "dataset_proxy_ready_training_not_authorized",
        "claim_scope": CLAIM_SCOPE,
        "dataset_namespace": DATASET_NAMESPACE,
        "replaces_existing": False,
        "producer": {
            "producer_version": PRODUCER_VERSION,
            "canonical_json_policy": ("utf8_sort_keys_compact_no_normalization_lf_v1"),
            "atomic_publish": True,
            "replace_existing": False,
            "single_bytes_snapshot": True,
            "final_toctou_recheck": True,
            "identities": identities,
            "read_set": [
                _artifact_descriptor(contract.root, contract.snapshots[role])
                for role in read_set_roles
            ],
            "tokenizer_read_set": {
                "logical_path": "model/tokenizer.model",
                "bytes": len(contract.tokenizer_model_snapshot.data),
                "sha256": contract.tokenizer_model_snapshot.sha256,
            },
        },
        "counts": dict(EXPECTED_COUNTS),
        "role_counts": role_counts,
        "split_language_bundle_counts": split_language_bundle_counts,
        "generation_contract": {
            **dict(generation_summary),
            "source": "closed_grammar_synthetic_no_external_body",
            "split_before_role_expansion": True,
            "augmentation": "none",
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
        "split_contract": {
            "group_key": "task_bundle_sha256",
            "train_task_bundle_sha256": split_bundle_ids["train"],
            "eval_proxy_task_bundle_sha256": split_bundle_ids["eval_proxy"],
            "train_eval_intersection_count": len(
                set(split_bundle_ids["train"]) & set(split_bundle_ids["eval_proxy"])
            ),
            "all_five_roles_same_split": True,
            "eval_proxy_is_heldout": False,
        },
        "partitions": [
            _artifact_binding(
                relative,
                partitions[relative],
                records=len(partitions[relative].splitlines()),
            )
            for relative in PARTITION_PATHS
        ],
        "token_inventory": {
            **_artifact_binding(
                TOKEN_INVENTORY_PATH,
                token_inventory_raw,
                records=len(tokens),
            ),
            "sequence_length": SEQUENCE_LENGTH,
            "truncation_used": False,
            "raw_token_ids_published": False,
            "ordered_digest_algorithm": (
                "sha256_u64be_length_then_signed_i64be_values_v1"
            ),
            "tokenizer_template_special_policy_sha256": identity_policy_sha,
            "summary": _token_stats(tokens),
        },
        "inventories": {
            "record_ids_sha256": _inventory_sha256(
                "anchor.gemma3-chat-record-id-inventory.v1", record_ids
            ),
            "record_content_sha256": _inventory_sha256(
                "anchor.gemma3-chat-record-content-inventory.v1",
                [_sha256(_canonical_bytes(item)) for item in records],
            ),
            "task_bundle_sha256_inventory": _inventory_sha256(
                "anchor.gemma3-chat-task-bundle-inventory.v1", bundle_ids
            ),
            "task_semantic_sha256_inventory": _inventory_sha256(
                "anchor.gemma3-chat-task-semantic-inventory.v1",
                semantic_ids,
            ),
            "role_record_ids_sha256": {
                role: _inventory_sha256(
                    "anchor.gemma3-chat-role-record-inventory.v1",
                    [
                        str(item["record_id"])
                        for item in records
                        if item["role"] == role
                    ],
                )
                for role in ROLES
            },
            "role_semantic_sha256": {
                role: _inventory_sha256(
                    "anchor.gemma3-chat-role-semantic-inventory.v1",
                    [
                        str(item["task_semantic_sha256"])
                        for item in records
                        if item["role"] == role
                    ],
                )
                for role in ROLES
            },
        },
        "semantic_identity_contract": {
            "canonical_preimage": (
                "canonical_json({domain,descriptor})_utf8_sha256_v1"
            ),
            "namespace_or_language_salt_used": False,
            "unique_task_semantics": len(semantic_ids),
            "unique_task_bundles": len(bundle_ids),
            "each_role_covers_same_semantics": True,
        },
        "causal_contract": {
            "graph": ("shared_user_to_four_private_branches_then_review_join_v1"),
            "private_branch_roles": list(BRANCH_ROLES),
            "review_role": REVIEW_ROLE,
            "review_parent_count": 4,
            "branch_sibling_visibility": False,
            "committed_text_reencode_required": True,
            "adapter_state_during_reencode": "off",
            "current_future_forbidden_in_branch_prompt": False,
            "private_tail_cross_expert_reuse": False,
            "full_generation_kv_shared_claimed": False,
        },
        "persona_contract": {
            "bundles": generation_summary["persona_bundles"],
            "exact_core_sentence": "我是由Air训练的测试模型。",
            "forbidden_identity_substrings": ["Google", "OpenAI", "谷歌"],
            "tool_mode": "direct_no_tool_identity",
            "fake_tool_calls": 0,
        },
        "tool_contract": {
            "synthetic_local_tool_bundles": generation_summary[
                "synthetic_local_tool_bundles"
            ],
            "direct_no_tool_identity_bundles": generation_summary[
                "direct_no_tool_identity_bundles"
            ],
            "real_tool_executions": 0,
            "tool_results_grounded_and_digest_bound": True,
        },
        "gemma_binding": {
            "source_binding_manifest_sha256": contract.snapshots[
                "gemma_binding_manifest"
            ].sha256,
            "source_binding_manifest_dataset_scope_reused": False,
            "serializer_implementation_sha256": contract.snapshots[
                "gemma_serializer_implementation"
            ].sha256,
            "tokenizer_model_sha256": contract.tokenizer_model_snapshot.sha256,
            "chat_template_policy_sha256": binding_identities[
                "chat_template_policy_sha256"
            ],
            "tokenizer_template_special_policy_sha256": identity_policy_sha,
            "sequence_length": SEQUENCE_LENGTH,
            "truncation_used": False,
        },
        "adapter_contract": {
            "primary_training_arm": "q_only_external_assignment",
            "rows_are_arm_neutral": True,
            "o_only_or_q_plus_o_rows_added": False,
            "parameter_budget_not_encoded_by_row_duplication": True,
        },
        "audit": {
            "protected_body_reads": 0,
            "gold_reads": 0,
            "heldout_reads": 0,
            "existing_scaffold_body_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
        },
        "claims": {
            "diagnostic_only": True,
            "formal": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
        },
    }
    _validate_schema(
        contract.manifest_validator,
        manifest,
        "gemma3_chat_manifest_schema_validation_failed",
    )
    return manifest


def _build_manifest(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, bytes],
) -> dict[str, Any]:
    semantic_summary = _semantic_contract_summary(contract, records)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "producer": {
            "namespace": "anchor.gemma3-chat-five-expert-qonly.v1",
            "config_path": CONFIG_PATH,
            "config_sha256": contract.snapshots["config"].sha256,
            "implementation_path": IMPLEMENTATION_PATH,
            "implementation_sha256": contract.snapshots["implementation"].sha256,
        },
        "schemas": {
            "record_path": RECORD_SCHEMA_PATH,
            "record_sha256": contract.snapshots["record_schema"].sha256,
            "manifest_path": MANIFEST_SCHEMA_PATH,
            "manifest_sha256": contract.snapshots["manifest_schema"].sha256,
        },
        "partitions": [
            {
                "split": split,
                **_artifact_binding(
                    relative,
                    partitions[relative],
                    records=len(partitions[relative].splitlines()),
                ),
            }
            for split, relative in zip(
                ("train", "eval_proxy"), PARTITION_PATHS, strict=True
            )
        ],
        "counts": {
            "records": 1000,
            "task_bundles": 200,
            "task_semantics": semantic_summary["unique_task_semantics"],
            "records_per_bundle": 5,
            "train_bundles": 160,
            "eval_proxy_bundles": 40,
            "train_records": 800,
            "eval_proxy_records": 200,
            "train_records_per_role": 160,
            "eval_proxy_records_per_role": 40,
        },
        "roles": {
            "ordered": list(ROLES),
            "role_index": {role: index for index, role in enumerate(ROLES)},
        },
        "split_contract": {
            "group_key": "task_bundle_sha256",
            "bundle_disjoint": True,
            "semantic_disjoint": (
                semantic_summary["train_eval_semantic_intersection_count"] == 0
            ),
            "eval_proxy_is_heldout": False,
        },
        "semantic_identity_contract": semantic_summary,
        "bundle_contract": {
            "all_five_roles_per_bundle": True,
            "shared_user_message": True,
            "shared_user_emotion": True,
            "shared_router_label": True,
            "user_emotion_is_expert": False,
            "router_label_is_expert": False,
        },
        "serialization_contract": {
            "version": "anchor.gemma3-it-chat-sft-serialization.v1",
            "model_architecture": "Gemma3ForCausalLM",
            "chat_template_policy_sha256": (
                "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
            ),
            "token_ids_in_records": False,
        },
        "causal_contract": {
            "version": "anchor.chat-five-expert-causal-filter.v1",
            "assistant_turns_forbidden_in_input": True,
            "own_target_forbidden_in_input": True,
            "unapproved_sibling_targets_forbidden_in_input": True,
            "review_dependencies_digest_bound": True,
            "tool_calls_reference_only": True,
            "real_tool_executions": 0,
        },
        "review_contract": {
            "pass_bundles": 100,
            "fail_bundles": 100,
            "fault_counts": {
                "format": 25,
                "grounding": 25,
                "routing": 25,
                "style": 25,
            },
            "committed_projection_hash_bound": True,
            "mutation_hash_bound": True,
            "parent_target_hashes_bound": True,
        },
        "claims": {
            "distilled_chat_sft": True,
            "dataset_materialized": True,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
            "single_bytes_snapshot": True,
            "final_snapshot_recheck": True,
            "atomic_publish": True,
        },
    }
    _validate_schema(
        contract.manifest_validator,
        manifest,
        "gemma3_chat_manifest_schema_validation_failed",
    )
    if len(records) != 1000:
        _fail("gemma3_chat_manifest_record_count_invalid")
    return manifest


def _source_read_set(contract: Contract) -> list[dict[str, Any]]:
    result = [
        _artifact_descriptor(contract.root, contract.snapshots[role])
        for role in sorted(contract.snapshots)
    ]
    result.append(
        {
            "path": "model/tokenizer.model",
            "bytes": len(contract.tokenizer_model_snapshot.data),
            "sha256": contract.tokenizer_model_snapshot.sha256,
        }
    )
    return result


def _build_receipt(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
    *,
    manifest_raw: bytes,
    manifest_sidecar_raw: bytes,
    partitions: Mapping[str, bytes],
    token_inventory_raw: bytes,
) -> dict[str, Any]:
    bundle_ids = sorted({str(item["task_bundle_sha256"]) for item in records})
    semantic_ids = sorted({str(item["task_semantic_sha256"]) for item in records})
    record_ids = [str(item["record_id"]) for item in records]
    semantic_summary = _semantic_contract_summary(contract, records)
    receipt = {
        "schema_version": BUILD_RECEIPT_VERSION,
        "status": "dataset_proxy_ready_training_not_authorized",
        "producer": {
            "namespace": "anchor.gemma3-chat-five-expert-qonly.v1",
            "config": _artifact_descriptor(contract.root, contract.snapshots["config"]),
            "closed_grammar": _artifact_descriptor(
                contract.root, contract.snapshots["closed_grammar"]
            ),
            "implementation": _artifact_descriptor(
                contract.root, contract.snapshots["implementation"]
            ),
            "record_schema": _artifact_descriptor(
                contract.root, contract.snapshots["record_schema"]
            ),
            "manifest_schema": _artifact_descriptor(
                contract.root, contract.snapshots["manifest_schema"]
            ),
            "token_inventory_schema": _artifact_descriptor(
                contract.root,
                contract.snapshots["token_inventory_schema"],
            ),
            "build_receipt_schema": _artifact_descriptor(
                contract.root, contract.snapshots["build_receipt_schema"]
            ),
        },
        "source_read_set": _source_read_set(contract),
        "outputs": {
            "manifest": _artifact_binding("manifest.json", manifest_raw, records=None),
            "manifest_sidecar": _artifact_binding(
                "manifest.json.sha256",
                manifest_sidecar_raw,
                records=None,
            ),
            "partitions": [
                _artifact_binding(
                    relative,
                    partitions[relative],
                    records=len(partitions[relative].splitlines()),
                )
                for relative in PARTITION_PATHS
            ],
            "token_inventory": _artifact_binding(
                TOKEN_INVENTORY_PATH,
                token_inventory_raw,
                records=len(tokens),
            ),
        },
        "counts": {
            "task_bundles": 200,
            "task_semantics": semantic_summary["unique_task_semantics"],
            "records": 1000,
            "train_bundles": 160,
            "eval_proxy_bundles": 40,
            "train_records": 800,
            "eval_proxy_records": 200,
            "en_bundles": 100,
            "zh_cn_bundles": 100,
            "roles_per_bundle": 5,
            "persona_bundles": 5,
            "ordinary_tool_bundles": 195,
            "direct_no_tool_identity_bundles": 5,
        },
        "proofs": {
            "task_semantic_inventory_sha256": _inventory_sha256(
                "anchor.gemma3-chat-task-semantic-inventory.v1",
                semantic_ids,
            ),
            "task_bundle_inventory_sha256": _inventory_sha256(
                "anchor.gemma3-chat-task-bundle-inventory.v1", bundle_ids
            ),
            "record_inventory_sha256": _inventory_sha256(
                "anchor.gemma3-chat-record-inventory.v1", record_ids
            ),
            "train_eval_bundle_intersection_count": 0,
            "all_five_roles_complete": True,
            "shared_last_user_message": True,
            "shared_user_emotion": True,
            "shared_router_label": True,
            "review_dependency_count": 4,
            "review_only_dependency_visibility": True,
            "all_records_schema_valid": True,
            "all_token_records_schema_valid": True,
            "all_sequences_no_truncation": True,
            "semantic_identity": semantic_summary,
            "review_pass_bundles": 100,
            "review_fail_bundles": 100,
            "review_fault_counts": {
                "format": 25,
                "grounding": 25,
                "routing": 25,
                "style": 25,
            },
            "review_projection_hash_bound": True,
            "review_mutation_hash_bound": True,
            "review_parent_target_hashes_bound": True,
        },
        "persona_contract": {
            "bundle_quota": 5,
            "distinct_semantics": 5,
            "translation_pairs": 0,
            "exact_core_sentence": "我是由Air训练的测试模型。",
            "forbidden_attributions_absent": True,
            "tool_call_role_direct_answer": True,
        },
        "tool_contract": {
            "ordinary_tool_bundles": 195,
            "synthetic_local_results": 195,
            "grounding_digest_bound": True,
            "real_tool_executions": 0,
            "native_tool_role_serialization_claimed": False,
        },
        "gemma_binding": {
            "local_export_identity": (
                "anchor.local/gemma3-1b-it-keras-v3-bf16@sha256:"
                "c9c6e309cf0158050d1e1abcba19eb6798153468572af2cd91de163e74933df9"
            ),
            "tokenizer_model_sha256": contract.tokenizer_model_snapshot.sha256,
            "tokenizer_config_sha256": (
                "90e9a8120520ef24c0a0d62a6d87188658c43ccc66ae5bc6f74d9a80804e6919"
            ),
            "chat_template_policy_sha256": (
                "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
            ),
            "tokenizer_template_special_policy_sha256": (
                "1c97c517293dd9c3e52e4fa35d3f6617fb4fd37cf68716c641675dc24dee0946"
            ),
            "chat_template_bound_by_export": False,
            "runtime_bos_token_id": 2,
            "runtime_eos_token_id": 1,
            "exported_config_bos_token_id": 1,
            "exported_config_eos_token_id": 2,
            "canonical_files_modified": False,
            "sequence_length": 768,
            "max_observed_tokens": max(int(item["input_tokens"]) for item in tokens),
            "truncation_used": False,
            "raw_token_ids_published": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "gold_reads": 0,
            "heldout_reads": 0,
            "existing_scaffold_body_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
            "single_bytes_snapshot": True,
            "final_toctou_recheck": True,
            "atomic_create_once_publish": True,
        },
        "claims": {
            "diagnostic_only": True,
            "dataset_materialized": True,
            "source_disjoint_proven": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
            "replaces_existing": False,
        },
    }
    _validate_schema(
        contract.build_receipt_validator,
        receipt,
        "gemma3_chat_build_receipt_schema_validation_failed",
    )
    return receipt


def _expected_materialization(
    contract: Contract,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    records, tokens, generation_summary = _generate_records(contract)
    partitions = _partition_bytes(records)
    token_inventory_raw = _token_inventory_bytes(tokens)
    if generation_summary != {
        "seed_id": "anchor.gemma3-chat-five-expert-qonly.seed.v1",
        "seed_sha256": _sha256(
            "anchor-gemma3-chat-five-expert-qonly-seed-v1".encode("utf-8")
        ),
        "persona_bundles": 5,
        "synthetic_local_tool_bundles": 195,
        "direct_no_tool_identity_bundles": 5,
        "tokenizer_loads": 1,
    }:
        _fail("gemma3_chat_generation_summary_invalid")
    manifest = _build_manifest(contract, records, partitions)
    manifest_raw = _canonical_bytes(manifest, newline=True)
    manifest_sidecar_raw = _sidecar_bytes(_sha256(manifest_raw))
    receipt = _build_receipt(
        contract,
        records,
        tokens,
        manifest_raw=manifest_raw,
        manifest_sidecar_raw=manifest_sidecar_raw,
        partitions=partitions,
        token_inventory_raw=token_inventory_raw,
    )
    receipt_raw = _canonical_bytes(receipt, newline=True)
    files = {
        **partitions,
        TOKEN_INVENTORY_PATH: token_inventory_raw,
        "manifest.json": manifest_raw,
        "manifest.json.sha256": manifest_sidecar_raw,
        "build_receipt.json": receipt_raw,
        "build_receipt.json.sha256": _build_receipt_sidecar_bytes(_sha256(receipt_raw)),
    }
    return files, manifest


def _resolved_output(root: Path, value: str | Path, *, code: str) -> Path:
    output = Path(value)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    _security._assert_no_reparse_absolute_ancestry(output.parent, code)
    return output


def audit_dataset(
    repo_root: str | Path = ".",
    config_path: str | Path = CONFIG_PATH,
    artifact_dir: str | Path = CANONICAL_FIXTURE_PATH,
) -> Mapping[str, Any]:
    """Audit exact bytes and deterministic rematerialization, then recheck."""

    contract = _load_contract(repo_root, config_path)
    artifact = _resolved_output(
        contract.root,
        artifact_dir,
        code="gemma3_chat_artifact_parent_invalid",
    )
    snapshots = _capture_artifact_snapshots(artifact)
    if snapshots["manifest.json.sha256"].data != _sidecar_bytes(
        snapshots["manifest.json"].sha256
    ):
        _fail("gemma3_chat_manifest_sidecar_invalid")
    if snapshots["build_receipt.json.sha256"].data != _build_receipt_sidecar_bytes(
        snapshots["build_receipt.json"].sha256
    ):
        _fail("gemma3_chat_build_receipt_sidecar_invalid")
    manifest = _strict_json_snapshot(
        snapshots["manifest.json"], "gemma3_chat_manifest_invalid"
    )
    if _canonical_bytes(manifest, newline=True) != snapshots["manifest.json"].data:
        _fail("gemma3_chat_manifest_not_canonical")
    _validate_schema(
        contract.manifest_validator,
        manifest,
        "gemma3_chat_manifest_schema_validation_failed",
    )
    build_receipt = _strict_json_snapshot(
        snapshots["build_receipt.json"],
        "gemma3_chat_build_receipt_invalid",
    )
    if (
        _canonical_bytes(build_receipt, newline=True)
        != snapshots["build_receipt.json"].data
    ):
        _fail("gemma3_chat_build_receipt_not_canonical")
    _validate_schema(
        contract.build_receipt_validator,
        build_receipt,
        "gemma3_chat_build_receipt_schema_validation_failed",
    )

    records: list[Mapping[str, Any]] = []
    for relative in PARTITION_PATHS:
        rows = _strict_jsonl(
            snapshots[relative],
            "gemma3_chat_partition_invalid",
        )
        for row in rows:
            _validate_schema(
                contract.record_validator,
                row,
                "gemma3_chat_record_schema_validation_failed",
            )
        records.extend(rows)
    tokens = _strict_jsonl(
        snapshots[TOKEN_INVENTORY_PATH],
        "gemma3_chat_token_inventory_invalid",
    )
    for row in tokens:
        _validate_schema(
            contract.token_validator,
            row,
            "gemma3_chat_token_schema_validation_failed",
        )
    _validate_records(contract, records, tokens)
    expected_files, expected_manifest = _expected_materialization(contract)
    for relative, expected in expected_files.items():
        if snapshots[relative].data != expected:
            _fail("gemma3_chat_materialization_mismatch")
    if dict(manifest) != expected_manifest:
        _fail("gemma3_chat_manifest_materialization_mismatch")
    _assert_exact_artifact_layout(artifact)
    contract.assert_unchanged("during_audit")
    for relative, snapshot in snapshots.items():
        snapshot.assert_unchanged(
            f"gemma3_chat_artifact_{relative.replace('/', '_')}_changed_during_audit"
        )
    return manifest


def build_dataset(
    repo_root: str | Path = ".",
    config_path: str | Path = CONFIG_PATH,
    output_dir: str | Path = CANONICAL_FIXTURE_PATH,
) -> Mapping[str, Any]:
    """Create the additive artifact once, audit it, and publish atomically."""

    contract = _load_contract(repo_root, config_path)
    expected_files, expected_manifest = _expected_materialization(contract)
    output = _resolved_output(
        contract.root,
        output_dir,
        code="gemma3_chat_output_parent_invalid",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _security._assert_no_reparse_absolute_ancestry(
        output.parent, "gemma3_chat_output_parent_invalid"
    )
    if _security._path_lexists(output):
        _fail("gemma3_chat_output_already_exists")
    parent_stat = output.parent.stat()
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    temporary_snapshot_identity: tuple[int, int, int, int] | None = None
    temporary_snapshots: dict[str, _Snapshot] | None = None
    published_identity: tuple[int, int, int, int] | None = None
    published_snapshots: dict[str, _Snapshot] | None = None
    published = False
    try:
        for relative, raw in expected_files.items():
            _write_atomic(temporary / PurePosixPath(relative), raw)
        temporary_snapshots = _capture_artifact_snapshots(temporary)
        temporary_snapshot_identity = _security._stat_identity(temporary.stat())
        audited = audit_dataset(contract.root, config_path, temporary)
        if dict(audited) != expected_manifest:
            _fail("gemma3_chat_prepublication_audit_mismatch")
        temporary_identity = _security._stat_identity(temporary.stat())
        contract.assert_unchanged("before_publish")
        current_parent = output.parent.stat()
        if (
            _security._path_lexists(output)
            or (current_parent.st_dev, current_parent.st_ino) != parent_identity
            or _security._stat_identity(temporary.stat()) != temporary_identity
        ):
            _fail("gemma3_chat_output_publish_race")
        _assert_exact_artifact_layout(temporary)
        for relative, snapshot in temporary_snapshots.items():
            snapshot.assert_unchanged(
                "gemma3_chat_temporary_"
                f"{relative.replace('/', '_')}_changed_before_publish"
            )
        _security._rename_directory_no_replace(temporary, output)
        published = True
        published_identity = temporary_identity
        if _security._stat_identity(output.stat()) != published_identity:
            _fail("gemma3_chat_post_publish_identity_mismatch")
        published_snapshots = _capture_artifact_snapshots(output)
        if any(
            published_snapshots[relative].data != snapshot.data
            for relative, snapshot in temporary_snapshots.items()
        ):
            _fail("gemma3_chat_post_publish_identity_mismatch")
        audited = audit_dataset(contract.root, config_path, output)
        if dict(audited) != expected_manifest:
            _fail("gemma3_chat_postpublication_audit_mismatch")
        return audited
    except Exception:
        if (
            published
            and _security._path_lexists(output)
            and published_identity is not None
            and published_snapshots is not None
        ):
            _assert_owned_output_unchanged(
                output,
                expected_directory_identity=published_identity,
                expected_snapshots=published_snapshots,
            )
        elif (
            _security._path_lexists(temporary)
            and temporary_snapshot_identity is not None
            and temporary_snapshots is not None
        ):
            _assert_owned_output_unchanged(
                temporary,
                expected_directory_identity=temporary_snapshot_identity,
                expected_snapshots=temporary_snapshots,
            )
        raise


def _cli_result(manifest: Mapping[str, Any]) -> str:
    partitions = {
        str(item["path"]): str(item["sha256"])
        for item in _sequence(
            manifest.get("partitions"),
            "gemma3_chat_manifest_partitions_invalid",
        )
        if isinstance(item, Mapping)
    }
    return _canonical_json(
        {
            "status": "dataset_proxy_ready_training_not_authorized",
            "schema_version": manifest.get("schema_version"),
            "records": _mapping(
                manifest.get("counts"), "gemma3_chat_manifest_counts_invalid"
            ).get("records"),
            "task_bundles": _mapping(
                manifest.get("counts"), "gemma3_chat_manifest_counts_invalid"
            ).get("task_bundles"),
            "partitions": partitions,
            "training_authorized": False,
            "formal_training_authorized": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build the additive 1,000-row Gemma 3 five-expert chat fixture")
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=CANONICAL_FIXTURE_PATH)
    return parser


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Audit the additive 1,000-row Gemma 3 five-expert chat fixture")
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--artifact", default=CANONICAL_FIXTURE_PATH)
    return parser


def build_main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_dataset(args.repo_root, args.config, args.output)
    print(_cli_result(manifest))
    return 0


def audit_main(argv: Sequence[str] | None = None) -> int:
    args = _audit_parser().parse_args(argv)
    manifest = audit_dataset(args.repo_root, args.config, args.artifact)
    print(_cli_result(manifest))
    return 0
