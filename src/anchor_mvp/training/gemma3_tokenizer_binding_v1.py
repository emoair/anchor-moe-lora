"""Model-free Gemma 3 1B IT tokenizer, template, and label binding.

This module authenticates a local KerasHub Transformers export, applies the
documented Gemma IT chat structure with numeric BOS/EOS insertion, consumes the
official five-role causal materializer, and emits only aggregate token/label
receipts.  It never imports a model, requests CUDA, or contacts a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from anchor_mvp.training import qwen_lora_diagnostic as qdiag
from anchor_mvp.training import qwen_synthetic_five_role_qonly_v2 as consumer
from anchor_mvp.training.config import ConfigError, _expand_env


CONFIG_VERSION = "anchor.gemma3-1b-it-tokenizer-binding-config.v1"
MANIFEST_VERSION = "anchor.gemma3-1b-it-tokenizer-binding-manifest.v1"
POLICY_VERSION = "anchor.gemma3-1b-it-chat-template-policy.v1"
CONFIG_PATH = "configs/research/gemma3_1b_it_tokenizer_binding_v1.yaml"
CONFIG_SCHEMA_PATH = (
    "configs/research/gemma3_1b_it_tokenizer_binding_v1_config.schema.json"
)
POLICY_PATH = "configs/research/gemma3_1b_it_chat_template_policy_v1.json"
MANIFEST_SCHEMA_PATH = (
    "configs/research/gemma3_1b_it_tokenizer_binding_v1_manifest.schema.json"
)
IMPLEMENTATION_PATH = "src/anchor_mvp/training/gemma3_tokenizer_binding_v1.py"
BUILD_SCRIPT_PATH = "scripts/research/build_gemma3_tokenizer_binding_v1.py"
AUDIT_SCRIPT_PATH = "scripts/research/audit_gemma3_tokenizer_binding_v1.py"
OUTPUT_PATH = "fixtures/research/gemma3_1b_it_tokenizer_binding_v1"

ROLES = consumer.ROLES
SPLITS = ("train", "eval_proxy")
IGNORE_INDEX = -100
OFFICIAL_FRAGMENT_SHA256 = (
    "8ecf042c4aef9b84f6a375a021fe450d18c16769a068eea19fac78ca0147281f"
)
MODEL_FILES = {
    "model.safetensors": (
        1_999_811_200,
        "c9c6e309cf0158050d1e1abcba19eb6798153468572af2cd91de163e74933df9",
    ),
    "config.json": (
        1_326,
        "a102086c644174f9f7432df15d28fd69ad687b64d8e7756f0e9b6bfabc002d0c",
    ),
    "tokenizer.model": (
        4_689_074,
        "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    ),
    "tokenizer_config.json": (
        1_496,
        "90e9a8120520ef24c0a0d62a6d87188658c43ccc66ae5bc6f74d9a80804e6919",
    ),
    "EXPORT_MANIFEST.json": (
        1_356,
        "61a9ac5fab43da9bf053eb46642a030fcd7485100c3e82eb5d90f03b9d8124bb",
    ),
}
SPECIAL_IDS = {
    "pad": 0,
    "eos": 1,
    "bos": 2,
    "unk": 3,
    "start_of_turn": 105,
    "end_of_turn": 106,
    "user": 2364,
    "model": 4368,
}
SPECIAL_PIECES = {
    "pad": "<pad>",
    "eos": "<eos>",
    "bos": "<bos>",
    "unk": "<unk>",
    "start_of_turn": "<start_of_turn>",
    "end_of_turn": "<end_of_turn>",
    "user": "user",
    "model": "model",
}
FORBIDDEN_RAW_MARKERS = (
    "<bos>",
    "<eos>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<pad>",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _fail(code: str) -> None:
    raise ConfigError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _require_sha(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    if value == "0" * 64:
        _fail(code)
    return value


def _repo_path(relative: str, *, file: bool = True) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        _fail("gemma_binding_repo_path_unsafe")
    return qdiag._assert_physical_path(
        qdiag._project_root_from_module().joinpath(*pure.parts),
        require_file=file,
        require_directory=not file,
        label=relative,
    )


@dataclass(frozen=True)
class StableDigest:
    path: Path
    bytes: int
    sha256: str
    stat_signature: tuple[int, int, int, int, int]
    data: bytes | None = None

    def assert_unchanged(self) -> None:
        stat = self.path.stat()
        signature = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if signature != self.stat_signature:
            _fail("gemma_binding_toctou_detected")


def _stable_digest(
    path: Path, *, capture_bytes: bool, max_bytes: int | None = None
) -> StableDigest:
    path = qdiag._assert_physical_path(path, require_file=True, label=str(path))
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if max_bytes is not None and before.st_size > max_bytes:
            _fail("gemma_binding_file_too_large")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        total = 0
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(handle.fileno())
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_signature != after_signature or total != before.st_size:
        _fail("gemma_binding_stream_toctou_detected")
    return StableDigest(
        path=path,
        bytes=total,
        sha256=digest.hexdigest(),
        stat_signature=after_signature,
        data=b"".join(chunks) if chunks is not None else None,
    )


def _strict_json(data: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(f"gemma_binding_{label}_json_invalid")
    return _mapping(value, f"gemma_binding_{label}_mapping_invalid")


def _load_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[dict[str, Any], list[StableDigest]]:
    root = qdiag._project_root_from_module()
    canonical = root.joinpath(*PurePosixPath(CONFIG_PATH).parts)
    requested = Path(path)
    requested = (
        requested
        if requested.is_absolute()
        else root.joinpath(*PurePosixPath(requested.as_posix()).parts)
    )
    if os.path.normcase(os.path.abspath(requested)) != os.path.normcase(
        os.path.abspath(canonical)
    ):
        _fail("gemma_binding_config_path_invalid")
    config_snapshot = _stable_digest(canonical, capture_bytes=True, max_bytes=1_000_000)
    schema_snapshot = _stable_digest(
        _repo_path(CONFIG_SCHEMA_PATH), capture_bytes=True, max_bytes=1_000_000
    )
    try:
        config_value = yaml.safe_load(config_snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        _fail("gemma_binding_config_yaml_invalid")
    config = dict(_mapping(config_value, "gemma_binding_config_invalid"))
    config["_config_path"] = str(canonical)
    expanded = _expand_env(config)
    schema = _strict_json(schema_snapshot.data or b"", "config_schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(
            {key: value for key, value in expanded.items() if key != "_config_path"}
        )
    except (SchemaError, ValidationError):
        _fail("gemma_binding_config_schema_validation_failed")
    validate_config(expanded)
    return expanded, [config_snapshot, schema_snapshot]


def validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "claim_scope",
            "model",
            "chat_template",
            "dataset",
            "serialization",
            "output",
            "claims",
            "_config_path",
        },
        "gemma_binding_config_fields_drift",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "model_free_diagnostic_only_no_training_or_formal_authority"
    ):
        _fail("gemma_binding_config_identity_drift")
    model = _mapping(config.get("model"), "gemma_binding_model_config_invalid")
    configured_files = _sequence(
        model.get("files"), "gemma_binding_model_files_invalid"
    )
    actual_files: dict[str, tuple[int, str]] = {}
    for raw in configured_files:
        item = _mapping(raw, "gemma_binding_model_file_invalid")
        _exact_keys(
            item,
            {"path", "bytes", "sha256"},
            "gemma_binding_model_file_fields_drift",
        )
        name = str(item["path"])
        actual_files[name] = (
            int(item["bytes"]),
            _require_sha(item["sha256"], "gemma_binding_model_sha_invalid"),
        )
    if (
        actual_files != MODEL_FILES
        or model.get("parameter_count") != 999_885_952
        or model.get("export_schema_version")
        != "anchor.local-gemma3-hf-export-manifest.v1"
        or model.get("chat_template_bound_by_export") is not False
        or model.get("expected_model_type") != "gemma3_text"
        or model.get("expected_architecture") != "Gemma3ForCausalLM"
    ):
        _fail("gemma_binding_model_contract_drift")
    template = _mapping(
        config.get("chat_template"), "gemma_binding_template_config_invalid"
    )
    if (
        template.get("path") != POLICY_PATH
        or template.get("official_text_fragments_sha256") != OFFICIAL_FRAGMENT_SHA256
    ):
        _fail("gemma_binding_template_contract_drift")
    _require_sha(template.get("sha256"), "gemma_binding_template_sha_invalid")
    dataset = _mapping(config.get("dataset"), "gemma_binding_dataset_config_invalid")
    if (
        dataset.get("root") != consumer.DATASET_ROOT
        or dataset.get("producer_config") != consumer.PRODUCER_CONFIG_PATH
        or dataset.get("manifest") != f"{consumer.DATASET_ROOT}/manifest.json"
        or dataset.get("manifest_sidecar")
        != f"{consumer.DATASET_ROOT}/manifest.json.sha256"
        or dataset.get("record_schema") != consumer.RECORD_SCHEMA_PATH
        or dataset.get("manifest_schema") != consumer.MANIFEST_SCHEMA_PATH
        or dataset.get("expected_manifest_sha256")
        != "a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed"
        or dataset.get("expected_record_schema_sha256")
        != "4b731e6493fb28ca6437811fa8b6a9ebda8da2cc34e7478b15838452833e2bad"
        or dataset.get("expected_manifest_schema_sha256")
        != "4fb49b8242b95251236c718089bbbbe8acb0df212e7c219ab291e25161536b14"
        or dataset.get("expected_records") != 1000
        or dataset.get("expected_roles") != 5
        or dataset.get("expected_train_records") != 800
        or dataset.get("expected_eval_proxy_records") != 200
    ):
        _fail("gemma_binding_dataset_contract_drift")
    expected_partitions = {
        str(item["path"]): (int(item["records"]), str(item["sha256"]))
        for item in (
            _mapping(value, "gemma_binding_partition_invalid")
            for value in _sequence(
                dataset.get("expected_partitions"),
                "gemma_binding_partitions_invalid",
            )
        )
    }
    if expected_partitions != {
        "train/concise_rationale_plus_json.jsonl": (
            800,
            "b69f1740d2e32fd68d74008ec85f7cce4818a60d6fcbfc2b84575d19be5fbe59",
        ),
        "eval_proxy/concise_rationale_plus_json.jsonl": (
            200,
            "3917be339d8181b8737d7b05021ef5ced1fc47ed5f1e3d99e64256bd546bca0f",
        ),
    }:
        _fail("gemma_binding_partition_contract_drift")
    serialization = _mapping(
        config.get("serialization"), "gemma_binding_serialization_invalid"
    )
    labels = _mapping(serialization.get("labels"), "gemma_binding_label_policy_invalid")
    if (
        serialization.get("sequence_length") != 768
        or serialization.get("record_order")
        != "split_train_then_eval_proxy_role_order_then_record_id"
        or serialization.get("ordered_digest_algorithm")
        != "sha256_u64be_length_then_signed_i64be_values_v1"
        or serialization.get("literal_special_marker_policy")
        != "reject_in_prompt_or_target"
        or labels
        != {
            "ignore_index": -100,
            "trainable_suffix": "target_plus_end_of_turn_plus_eos",
        }
    ):
        _fail("gemma_binding_serialization_contract_drift")
    output = _mapping(config.get("output"), "gemma_binding_output_invalid")
    if output != {
        "artifact_dir": OUTPUT_PATH,
        "atomic_publish": True,
        "replace_existing": False,
    }:
        _fail("gemma_binding_output_contract_drift")
    claims = _mapping(config.get("claims"), "gemma_binding_claims_invalid")
    if claims != {
        "diagnostic_only": True,
        "model_free": True,
        "tokenizer_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
    }:
        _fail("gemma_binding_claims_drift")


def _load_policy(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], StableDigest]:
    template = _mapping(
        config.get("chat_template"), "gemma_binding_template_config_invalid"
    )
    snapshot = _stable_digest(
        _repo_path(str(template["path"])), capture_bytes=True, max_bytes=1_000_000
    )
    if snapshot.sha256 != template["sha256"]:
        _fail("gemma_binding_template_policy_hash_drift")
    policy = _strict_json(snapshot.data or b"", "template_policy")
    validate_policy(policy)
    return policy, snapshot


def validate_policy(policy: Mapping[str, Any]) -> None:
    _exact_keys(
        policy,
        {
            "schema_version",
            "authoritative_sources",
            "official_text_fragments",
            "role_policy",
            "runtime_special_token_overlay",
            "sentencepiece_policy",
            "training_label_policy",
        },
        "gemma_binding_template_policy_fields_drift",
    )
    fragments = _mapping(
        policy.get("official_text_fragments"), "gemma_binding_fragments_invalid"
    )
    values = (
        fragments.get("user_turn"),
        fragments.get("assistant_turn"),
        fragments.get("generation_prefix"),
    )
    if values != (
        "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n",
        "<start_of_turn>model\n{response}<end_of_turn><eos>\n",
        "<start_of_turn>model\n",
    ):
        _fail("gemma_binding_official_template_drift")
    digest = _sha256(b"\x00".join(str(item).encode("utf-8") for item in values))
    if (
        digest != OFFICIAL_FRAGMENT_SHA256
        or fragments.get("nul_joined_utf8_sha256") != digest
    ):
        _fail("gemma_binding_official_template_digest_drift")
    overlay = _mapping(
        policy.get("runtime_special_token_overlay"),
        "gemma_binding_special_overlay_invalid",
    )
    if overlay != {
        "canonical_files_modified": False,
        "exported_config_bos_token_id": 1,
        "exported_config_eos_token_id": 2,
        "runtime_bos_token_id": 2,
        "runtime_eos_token_id": 1,
        "pad_token_id": 0,
        "unk_token_id": 3,
        "start_of_turn_token_id": 105,
        "end_of_turn_token_id": 106,
        "user_token_id": 2364,
        "model_token_id": 4368,
    }:
        _fail("gemma_binding_special_overlay_drift")
    sentencepiece_policy = _mapping(
        policy.get("sentencepiece_policy"),
        "gemma_binding_sentencepiece_policy_invalid",
    )
    if sentencepiece_policy != {
        "hf_fix_mistral_regex": False,
        "literal_bos_or_eos_passed_to_sentencepiece": False,
        "control_insertion_mode": "numeric_ids_outside_sentencepiece_text_segments",
        "tokenization_algorithm": (
            "bos_id_plus_user_segment_plus_eos_id_plus_assistant_segment_"
            "plus_eos_id_plus_terminal_newline_v1"
        ),
    }:
        _fail("gemma_binding_sentencepiece_policy_drift")
    if policy.get("role_policy") != {
        "system_role_supported": False,
        "system_instructions_location": "inside_first_user_prompt",
    } or policy.get("training_label_policy") != {
        "prompt_and_assistant_prefix_label": -100,
        "assistant_response_end_of_turn_and_eos_are_labels": True,
        "terminal_separator_newline_label": -100,
        "single_turn_terminal_sample": True,
    }:
        _fail("gemma_binding_role_or_label_policy_drift")


def _model_root(config: Mapping[str, Any]) -> Path:
    root_value = str(
        _mapping(config.get("model"), "gemma_binding_model_config_invalid")[
            "local_path"
        ]
    )
    path = Path(root_value).expanduser()
    if not path.is_absolute():
        path = qdiag._project_root_from_module() / path
    return qdiag._assert_physical_path(
        path, require_directory=True, label="Gemma 3 1B IT local export"
    )


def _authenticate_model(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, StableDigest], Mapping[str, Any], Mapping[str, Any]]:
    root = _model_root(config)
    snapshots: dict[str, StableDigest] = {}
    configured = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in (
            _mapping(value, "gemma_binding_model_file_invalid")
            for value in _sequence(
                _mapping(config["model"], "gemma_binding_model_config_invalid")[
                    "files"
                ],
                "gemma_binding_model_files_invalid",
            )
        )
    }
    for name, (expected_bytes, expected_sha) in configured.items():
        snapshot = _stable_digest(
            root / name,
            capture_bytes=name != "model.safetensors",
            max_bytes=None if name == "model.safetensors" else 10_000_000,
        )
        if snapshot.bytes != expected_bytes or snapshot.sha256 != expected_sha:
            _fail("gemma_binding_model_file_identity_mismatch")
        snapshots[name] = snapshot
    config_json = _strict_json(snapshots["config.json"].data or b"", "model_config")
    tokenizer_config = _strict_json(
        snapshots["tokenizer_config.json"].data or b"", "tokenizer_config"
    )
    export_manifest = _strict_json(
        snapshots["EXPORT_MANIFEST.json"].data or b"", "export_manifest"
    )
    if (
        config_json.get("architectures") != ["Gemma3ForCausalLM"]
        or config_json.get("model_type") != "gemma3_text"
        or config_json.get("bos_token_id") != 1
        or config_json.get("eos_token_id") != 2
        or tokenizer_config.get("bos_token") != "<bos>"
        or tokenizer_config.get("eos_token") != "<eos>"
        or "chat_template" in tokenizer_config
        or export_manifest.get("schema_version")
        != "anchor.local-gemma3-hf-export-manifest.v1"
        or export_manifest.get("chat_template_bound") is not False
        or export_manifest.get("parameter_count") != 999_885_952
        or export_manifest.get("formal") is not False
        or export_manifest.get("training_authorized") is not False
    ):
        _fail("gemma_binding_export_semantics_mismatch")
    manifest_files = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in (
            _mapping(value, "gemma_binding_export_file_invalid")
            for value in _sequence(
                export_manifest.get("files"),
                "gemma_binding_export_files_invalid",
            )
        )
    }
    if manifest_files != {
        key: value
        for key, value in MODEL_FILES.items()
        if key != "EXPORT_MANIFEST.json"
    }:
        _fail("gemma_binding_export_manifest_file_mismatch")
    return root, snapshots, config_json, tokenizer_config


def load_sentencepiece(model_path: Path) -> Any:
    try:
        import sentencepiece as sentencepiece
    except ImportError:
        raise RuntimeError("gemma_binding_sentencepiece_runtime_unavailable") from None
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    if processor.vocab_size() != 262_144:
        _fail("gemma_binding_sentencepiece_vocab_mismatch")
    for name, token_id in SPECIAL_IDS.items():
        if processor.id_to_piece(token_id) != SPECIAL_PIECES[name]:
            _fail("gemma_binding_sentencepiece_special_id_mismatch")
    if processor.encode("<bos>", out_type=int) == [SPECIAL_IDS["bos"]]:
        _fail("gemma_binding_literal_bos_unexpectedly_control")
    if processor.encode("<eos>", out_type=int) == [SPECIAL_IDS["eos"]]:
        _fail("gemma_binding_literal_eos_unexpectedly_control")
    if processor.encode("<start_of_turn>", out_type=int) != [
        SPECIAL_IDS["start_of_turn"]
    ] or processor.encode("<end_of_turn>", out_type=int) != [
        SPECIAL_IDS["end_of_turn"]
    ]:
        _fail("gemma_binding_turn_marker_tokenization_mismatch")
    return processor


def _load_hf_tokenizer(model_root: Path) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("gemma_binding_transformers_runtime_unavailable") from None
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
        fix_mistral_regex=False,
    )
    if (
        tokenizer.pad_token_id != 0
        or tokenizer.eos_token_id != 1
        or tokenizer.bos_token_id != 2
        or tokenizer.unk_token_id != 3
    ):
        _fail("gemma_binding_hf_special_id_mismatch")
    return tokenizer


@dataclass(frozen=True)
class SerializedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    trainable_label_tokens: int


def serialize_example(processor: Any, prompt: str, target: str) -> SerializedExample:
    if not prompt or not target:
        _fail("gemma_binding_empty_prompt_or_target")
    if any(marker in prompt or marker in target for marker in FORBIDDEN_RAW_MARKERS):
        _fail("gemma_binding_literal_special_marker_rejected")
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
            SPECIAL_IDS["start_of_turn"],
            SPECIAL_IDS["user"],
            107,
        )
        or user_ids[-1] != SPECIAL_IDS["end_of_turn"]
        or not assistant_ids
        or assistant_ids[: len(prefix_ids)] != prefix_ids
        or prefix_ids
        != (
            107,
            SPECIAL_IDS["start_of_turn"],
            SPECIAL_IDS["model"],
            107,
        )
        or assistant_ids[-1] != SPECIAL_IDS["end_of_turn"]
        or terminal_newline_ids != (107,)
        or len(assistant_ids) <= len(prefix_ids) + 1
    ):
        _fail("gemma_binding_prefix_alignment_failed")
    input_ids = (
        (SPECIAL_IDS["bos"],)
        + user_ids
        + (SPECIAL_IDS["eos"],)
        + assistant_ids
        + (SPECIAL_IDS["eos"],)
        + terminal_newline_ids
    )
    label_start = 1 + len(user_ids) + 1 + len(prefix_ids)
    label_end = len(input_ids) - len(terminal_newline_ids)
    if (
        input_ids[label_end - 2] != SPECIAL_IDS["end_of_turn"]
        or input_ids[label_end - 1] != SPECIAL_IDS["eos"]
        or label_start >= label_end - 2
    ):
        _fail("gemma_binding_label_suffix_alignment_failed")
    labels = (
        (IGNORE_INDEX,) * label_start
        + input_ids[label_start:label_end]
        + (IGNORE_INDEX,) * len(terminal_newline_ids)
    )
    if len(labels) != len(input_ids) or any(
        value != IGNORE_INDEX for value in labels[:label_start]
    ):
        _fail("gemma_binding_label_mask_failed")
    return SerializedExample(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=label_start,
        trainable_label_tokens=label_end - label_start,
    )


def _visible_hf_template(prompt: str, target: str) -> str:
    return (
        f"<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
        f"<start_of_turn>model\n{target}<end_of_turn><eos>\n"
    )


def _assert_hf_equivalent(
    hf_tokenizer: Any, example: SerializedExample, prompt: str, target: str
) -> None:
    encoded = hf_tokenizer(
        _visible_hf_template(prompt, target),
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    values = tuple(int(value) for value in encoded["input_ids"])
    if values != example.input_ids:
        _fail("gemma_binding_hf_sentencepiece_equivalence_failed")


def _dataset_consumer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = dict(
        _mapping(config.get("dataset"), "gemma_binding_dataset_config_invalid")
    )
    return {
        "dataset": {
            "kind": "synthetic_five_role_qonly_diagnostic_v1",
            "root": dataset["root"],
            "producer_config": dataset["producer_config"],
            "manifest": dataset["manifest"],
            "manifest_sidecar": dataset["manifest_sidecar"],
            "record_schema": dataset["record_schema"],
            "manifest_schema": dataset["manifest_schema"],
            "expected_manifest_sha256": dataset["expected_manifest_sha256"],
            "expected_record_schema_sha256": dataset["expected_record_schema_sha256"],
            "expected_manifest_schema_sha256": dataset[
                "expected_manifest_schema_sha256"
            ],
        }
    }


def load_all_role_datasets(
    config: Mapping[str, Any],
) -> dict[str, consumer.RoleDataset]:
    loader_config = _dataset_consumer_config(config)
    result = {role: consumer.load_role_dataset(loader_config, role) for role in ROLES}
    manifest_ids = {dataset.manifest_sha256 for dataset in result.values()}
    partition_ids = {
        tuple(sorted(dataset.partition_sha256.items())) for dataset in result.values()
    }
    if (
        manifest_ids
        != {"a70ae6df5537bb8d9227d843079b74f0e2cab984cc59b70f2e51b568d1eff2ed"}
        or len(partition_ids) != 1
        or any(
            dataset.global_records_authenticated != 1000
            or dataset.global_task_bundles_authenticated != 200
            or dataset.global_task_semantics_authenticated != 200
            or len(dataset.train) != 160
            or len(dataset.eval_proxy) != 40
            for dataset in result.values()
        )
    ):
        _fail("gemma_binding_consumer_dataset_cross_binding_failed")
    return result


def _update_sequence_digest(digest: Any, values: Sequence[int]) -> None:
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">q", int(value)))


@dataclass
class _CellAccumulator:
    count: int = 0
    token_total: int = 0
    token_min: int = sys.maxsize
    token_max: int = 0
    prompt_total: int = 0
    label_total: int = 0
    over_512: int = 0
    input_digest: Any = None
    label_digest: Any = None

    def __post_init__(self) -> None:
        self.input_digest = hashlib.sha256()
        self.label_digest = hashlib.sha256()

    def add(self, example: SerializedExample) -> None:
        length = len(example.input_ids)
        self.count += 1
        self.token_total += length
        self.token_min = min(self.token_min, length)
        self.token_max = max(self.token_max, length)
        self.prompt_total += example.prompt_tokens
        self.label_total += example.trainable_label_tokens
        self.over_512 += int(length > 512)
        _update_sequence_digest(self.input_digest, example.input_ids)
        _update_sequence_digest(self.label_digest, example.labels)

    def statistics(self) -> dict[str, Any]:
        if self.count == 0:
            _fail("gemma_binding_empty_summary_cell")
        return {
            "records": self.count,
            "token_length": {
                "min": self.token_min,
                "max": self.token_max,
                "mean": round(self.token_total / self.count, 6),
                "total": self.token_total,
            },
            "prompt_token_mean": round(self.prompt_total / self.count, 6),
            "trainable_label_token_mean": round(self.label_total / self.count, 6),
            "records_over_512": self.over_512,
            "ordered_input_ids_sha256": self.input_digest.hexdigest(),
            "ordered_labels_sha256": self.label_digest.hexdigest(),
        }

    def to_json(self, *, split: str, role: str) -> dict[str, Any]:
        return {"split": split, "role": role, **self.statistics()}


def _identity_digest(value: object) -> str:
    return _sha256(_canonical_json(value))


def build_manifest(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config, config_snapshots = _load_config(config_path)
    policy, policy_snapshot = _load_policy(config)
    model_root, model_snapshots, model_config, _tokenizer_config = _authenticate_model(
        config
    )
    processor = load_sentencepiece(model_root / "tokenizer.model")
    hf_tokenizer = _load_hf_tokenizer(model_root)
    datasets = load_all_role_datasets(config)

    global_inputs = hashlib.sha256()
    global_labels = hashlib.sha256()
    aggregate = _CellAccumulator()
    split_accumulators = {split: _CellAccumulator() for split in SPLITS}
    cells: dict[tuple[str, str], _CellAccumulator] = defaultdict(_CellAccumulator)
    records = 0
    over_limit = 0
    over_512 = 0
    sequence_length = int(
        _mapping(config.get("serialization"), "gemma_binding_serialization_invalid")[
            "sequence_length"
        ]
    )
    for split in SPLITS:
        for role in ROLES:
            dataset = datasets[role]
            selected = dataset.train if split == "train" else dataset.eval_proxy
            for item in selected:
                serialized = serialize_example(processor, item.prompt, item.target)
                _assert_hf_equivalent(
                    hf_tokenizer, serialized, item.prompt, item.target
                )
                records += 1
                over_limit += int(len(serialized.input_ids) > sequence_length)
                over_512 += int(len(serialized.input_ids) > 512)
                _update_sequence_digest(global_inputs, serialized.input_ids)
                _update_sequence_digest(global_labels, serialized.labels)
                aggregate.add(serialized)
                split_accumulators[split].add(serialized)
                cells[(split, role)].add(serialized)
    if records != 1000 or over_limit != 0:
        _fail("gemma_binding_sequence_length_preflight_failed")

    config_schema_snapshot = config_snapshots[1]
    manifest_schema_snapshot = _stable_digest(
        _repo_path(MANIFEST_SCHEMA_PATH), capture_bytes=True, max_bytes=1_000_000
    )
    implementation_snapshot = _stable_digest(
        _repo_path(IMPLEMENTATION_PATH), capture_bytes=False
    )
    template_identity = {
        "tokenizer_model_sha256": model_snapshots["tokenizer.model"].sha256,
        "tokenizer_config_sha256": model_snapshots["tokenizer_config.json"].sha256,
        "template_policy_sha256": policy_snapshot.sha256,
        "official_text_fragments_sha256": OFFICIAL_FRAGMENT_SHA256,
        "special_token_ids": SPECIAL_IDS,
        "hf_fix_mistral_regex": False,
        "control_insertion_mode": ("numeric_ids_outside_sentencepiece_text_segments"),
        "serializer_algorithm": (
            "bos_id_plus_user_segment_plus_eos_id_plus_assistant_segment_"
            "plus_eos_id_plus_terminal_newline_v1"
        ),
    }
    combined_identity = _identity_digest(template_identity)
    cell_reports = [
        cells[(split, role)].to_json(split=split, role=role)
        for split in SPLITS
        for role in ROLES
    ]
    maximum = max(item["token_length"]["max"] for item in cell_reports)
    minimum = min(item["token_length"]["min"] for item in cell_reports)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "status": ("passed_model_free_tokenizer_and_label_preflight_training_blocked"),
        "identities": {
            "config_sha256": config_snapshots[0].sha256,
            "config_schema_sha256": config_schema_snapshot.sha256,
            "manifest_schema_sha256": manifest_schema_snapshot.sha256,
            "implementation_sha256": implementation_snapshot.sha256,
            "chat_template_policy_sha256": policy_snapshot.sha256,
            "official_text_fragments_sha256": OFFICIAL_FRAGMENT_SHA256,
            "tokenizer_template_special_policy_sha256": combined_identity,
        },
        "model": {
            "local_export_identity": {
                name: {
                    "bytes": snapshot.bytes,
                    "sha256": snapshot.sha256,
                }
                for name, snapshot in sorted(model_snapshots.items())
            },
            "model_type": model_config["model_type"],
            "architecture": model_config["architectures"][0],
            "parameter_count": 999_885_952,
            "canonical_files_modified": False,
            "exported_config_bos_token_id": 1,
            "exported_config_eos_token_id": 2,
            "runtime_overlay_bos_token_id": 2,
            "runtime_overlay_eos_token_id": 1,
            "same_handle_stream_hash_and_stat_stability": True,
        },
        "template": {
            "authoritative_sources": list(policy["authoritative_sources"]),
            "system_role_supported": False,
            "system_instructions_location": "inside_first_user_prompt",
            "numeric_bos_eos_insertion": True,
            "literal_bos_or_eos_sent_to_sentencepiece": False,
            "hf_fix_mistral_regex": False,
            "hf_visible_template_equivalence_records": 1000,
            "hf_visible_template_equivalence_mismatches": 0,
        },
        "dataset": {
            "consumer_implementation": consumer.IMPLEMENTATION_PATH,
            "official_consumer_loader": (
                "anchor_mvp.training.qwen_synthetic_five_role_qonly_v2."
                "load_role_dataset"
            ),
            "manifest_sha256": next(
                iter(dataset.manifest_sha256 for dataset in datasets.values())
            ),
            "partition_sha256": dict(next(iter(datasets.values())).partition_sha256),
            "records": 1000,
            "roles": 5,
            "train_records": 800,
            "eval_proxy_records": 200,
            "current_future_forbidden_filter_authenticated": True,
        },
        "tokenization": {
            "sentencepiece_vocab_size": 262_144,
            "special_token_ids": SPECIAL_IDS,
            "sequence_length": sequence_length,
            "observed_minimum_tokens": minimum,
            "observed_maximum_tokens": maximum,
            "records_over_sequence_length": 0,
            "records_over_512": over_512,
            "truncation_used": False,
            "record_order": ("split_train_then_eval_proxy_role_order_then_record_id"),
            "ordered_digest_algorithm": (
                "sha256_u64be_length_then_signed_i64be_values_v1"
            ),
            "label_policy": {
                "ignore_index": -100,
                "prompt_and_assistant_prefix_masked": True,
                "assistant_target_end_of_turn_eos_trainable": True,
                "terminal_separator_newline_masked": True,
            },
            "raw_record_token_ids_published": False,
            "raw_prompt_or_target_published": False,
        },
        "summary": {
            "records": records,
            "aggregate": aggregate.statistics(),
            "splits": [
                {"split": split, **split_accumulators[split].statistics()}
                for split in SPLITS
            ],
            "cells": cell_reports,
            "ordered_input_ids_sha256": global_inputs.hexdigest(),
            "ordered_labels_sha256": global_labels.hexdigest(),
        },
        "claims": dict(_mapping(config.get("claims"), "gemma_binding_claims_invalid")),
        "audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_body_reads": 0,
            "sentencepiece_tokenizer_loads": 1,
            "hf_tokenizer_loads": 1,
            "model_weight_identity_stream_reads": 1,
        },
    }
    schema = _strict_json(manifest_schema_snapshot.data or b"", "manifest_schema")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
    except (SchemaError, ValidationError):
        _fail("gemma_binding_manifest_schema_validation_failed")
    for snapshot in (
        *config_snapshots,
        policy_snapshot,
        manifest_schema_snapshot,
        implementation_snapshot,
        *model_snapshots.values(),
    ):
        snapshot.assert_unchanged()
    return manifest


def _artifact_path(config: Mapping[str, Any]) -> Path:
    output = _mapping(config.get("output"), "gemma_binding_output_invalid")
    if output.get("artifact_dir") != OUTPUT_PATH:
        _fail("gemma_binding_output_path_drift")
    return qdiag._project_root_from_module().joinpath(*PurePosixPath(OUTPUT_PATH).parts)


def publish_manifest(
    manifest: Mapping[str, Any], config_path: str | Path = CONFIG_PATH
) -> Path:
    config, snapshots = _load_config(config_path)
    destination = _artifact_path(config)
    if destination.exists():
        _fail("gemma_binding_output_exists_no_replace")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        manifest_bytes = _canonical_json(manifest)
        digest = _sha256(manifest_bytes)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "manifest.json.sha256").write_bytes(
            f"{digest}  manifest.json\n".encode("ascii")
        )
        os.rename(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    for snapshot in snapshots:
        snapshot.assert_unchanged()
    return destination


def audit_manifest(
    artifact: str | Path = OUTPUT_PATH,
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    expected = build_manifest(config_path)
    root = qdiag._project_root_from_module()
    path = Path(artifact)
    if not path.is_absolute():
        path = root.joinpath(*PurePosixPath(path.as_posix()).parts)
    path = qdiag._assert_physical_path(
        path, require_directory=True, label="Gemma tokenizer binding artifact"
    )
    if os.path.normcase(os.path.abspath(path)) != os.path.normcase(
        os.path.abspath(root.joinpath(*PurePosixPath(OUTPUT_PATH).parts))
    ):
        _fail("gemma_binding_artifact_path_invalid")
    manifest_snapshot = _stable_digest(
        path / "manifest.json", capture_bytes=True, max_bytes=2_000_000
    )
    sidecar_snapshot = _stable_digest(
        path / "manifest.json.sha256", capture_bytes=True, max_bytes=1024
    )
    if sidecar_snapshot.data != (
        f"{manifest_snapshot.sha256}  manifest.json\n".encode("ascii")
    ):
        _fail("gemma_binding_manifest_sidecar_invalid")
    actual = _strict_json(manifest_snapshot.data or b"", "artifact_manifest")
    if dict(actual) != expected:
        _fail("gemma_binding_manifest_rebuild_mismatch")
    manifest_snapshot.assert_unchanged()
    sidecar_snapshot.assert_unchanged()
    return dict(actual)


def _parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or audit the model-free Gemma 3 1B IT tokenizer/label binding. "
            "No model, CUDA, network, provider, Gold, or heldout source is used."
        )
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    if mode == "build":
        parser.add_argument("--publish", action="store_true")
    else:
        parser.add_argument("--artifact", default=OUTPUT_PATH)
    return parser


def build_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("build").parse_args(argv)
    try:
        manifest = build_manifest(args.config)
        if args.publish:
            path = publish_manifest(manifest, args.config)
            result = {
                "status": "published",
                "artifact": str(path),
                "manifest_sha256": _sha256(_canonical_json(manifest)),
                "records": manifest["summary"]["records"],
                "claims": manifest["claims"],
                "audit": manifest["audit"],
            }
        else:
            result = {
                "status": "passed_not_published",
                "manifest_sha256": _sha256(_canonical_json(manifest)),
                "records": manifest["summary"]["records"],
                "claims": manifest["claims"],
                "audit": manifest["audit"],
            }
    except (ConfigError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_code": str(error),
                    "claims": {
                        "model_loaded": False,
                        "gpu_requested": False,
                        "training_authorized": False,
                        "formal": False,
                    },
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def audit_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("audit").parse_args(argv)
    try:
        manifest = audit_manifest(args.artifact, args.config)
    except (ConfigError, RuntimeError) as error:
        print(
            json.dumps({"status": "blocked", "error_code": str(error)}, sort_keys=True)
        )
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_sha256": _sha256(_canonical_json(manifest)),
                "records": manifest["summary"]["records"],
                "claims": manifest["claims"],
                "audit": manifest["audit"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "ROLES",
    "SPECIAL_IDS",
    "SerializedExample",
    "audit_main",
    "audit_manifest",
    "build_main",
    "build_manifest",
    "load_all_role_datasets",
    "load_sentencepiece",
    "publish_manifest",
    "serialize_example",
    "validate_config",
    "validate_policy",
]
