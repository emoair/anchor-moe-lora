"""Fail-closed audit for the controlled-proxy follow-up contract.

Only authenticated metadata is opened.  The auditor deliberately does not
import or execute the training runner, load a model, contact a provider, or
open Gold, heldout, scaffold, or dataset bodies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


CONTRACT_PATH = "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.json"
CONTRACT_SIDECAR_PATH = f"{CONTRACT_PATH}.sha256"
SCHEMA_PATH = (
    "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.schema.json"
)
IMPLEMENTATION_PATH = (
    "src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_followup.py"
)
SOURCE_PATH = (
    "fixtures/research/synthetic_scaffold_controlled_proxy_followup_v1/source/"
    "qwen_budget_matched_ablation_v1/comparison.json"
)
SOURCE_SIDECAR_PATH = f"{SOURCE_PATH}.sha256"

# The schema digest is updated only when the closed Producer schema itself is
# intentionally re-frozen.  The contract digest is intentionally not embedded:
# it binds this implementation, so embedding it would create a hash cycle.
EXPECTED_SCHEMA_SHA256 = (
    "fe3878cac9d3be773a676c23025e79cc7f64063da03a53c54eb7f4b59594e0b6"
)
EXPECTED_SOURCE_SHA256 = (
    "920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45"
)
EXPECTED_SOURCE_SIDECAR_SHA256 = (
    "bbdeb5b3ac8f7890a93c736b19bcd920875f642174f99914b199d1d62ce06830"
)
EXPECTED_COMPARISON_SCHEMA_SHA256 = (
    "3b1c81cc888f0b56e013d3afc1317ed78ccd62f7c53606aa2257ed6d389161e2"
)
EXPECTED_CONSUMER_COMMIT = "6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465"
EXPECTED_CONSUMER_PARENT = "9539fb56c236f08ffe9d7a8f56dfc28f14e1907c"
EXPECTED_CONSUMER_TREE = "2e40aaa1685b3da914275f4f61f8161138ffe6b5"

EXPECTED_BASE_SHA256 = (
    "2e05af50628344c5c19cbd981bf98de0026ea567a61f403afad4665d28156e15"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "64b1ce813477deef48de16dbdc0d2561bbeaa0ef5d6248862e9f2bedc8acc0dd"
)
EXPECTED_RECORD_ORDER_SHA256 = (
    "13ff5875530fc8b61e6f8526ffa67d18c8574bd0d87bf289db1cf5f38776daab"
)
EXPECTED_TOKENIZED_TRAIN_SHA256 = (
    "34ecf23c1fa4dcf70336126caf7845abb3d60bc48fd65f831a1722b98d0f7cda"
)
EXPECTED_EVAL_VIEWS_SHA256 = (
    "5a2722900afea0898e284808f0ed28a6c5d164ecf3e877128a0ee1266884fc10"
)
EXPECTED_EFFECTIVE_CONFIG_SHA256 = (
    "8a7aa62fefe24512155898c677b1a1473b2c0bc11c85005e6e2f960d2c66245a"
)

EXPECTED_PARTITIONS = {
    "eval_proxy/concise_rationale_plus_json.jsonl": (
        "ed0d60a609edbac6ba9db7cf8d818f85354da4f17e7409b0981d8d4abaf7cfaa"
    ),
    "eval_proxy/json_only.jsonl": (
        "06fe504e33ef61b05df7f3aff5969025b299343d613441bbb04dea1599fbc680"
    ),
    "train/concise_rationale_plus_json.jsonl": (
        "ff656ff6d2b5303880e5a5ec8db05ffa33e7eca7ca3b1bbd78787a1bd28f1852"
    ),
    "train/json_only.jsonl": (
        "3ea3cf57e9990b2b07e98cd2bf27a620ed3d2eb2bfd495b0f89ae3d22cce60df"
    ),
}

EXPECTED_ARMS: Mapping[str, Mapping[str, object]] = {
    "q_only": {
        "ranks": {"q_proj": 16},
        "alphas": {"q_proj": 32},
        "tensors": 56,
        "preflight": (
            "da0aa73bde0471172a0e7ea2433bb2d6105195081cb6aa482a08dc1e674934c4"
        ),
        "receipt": ("6ca4d55c2780aac413a2a04ea2c8e4b338f233c3db7b952ac6452a4965bd9f81"),
    },
    "q_plus_o": {
        "ranks": {"q_proj": 8, "o_proj": 8},
        "alphas": {"q_proj": 16, "o_proj": 16},
        "tensors": 112,
        "preflight": (
            "d90a6bf0ec9f3df09bf78c860602107e0436a7799de3ed8bc27bbc3d6d52879b"
        ),
        "receipt": ("dc94204df696db795f3e657c679d67918e3aa2723b2d0bdf7dee899ef4490f6e"),
    },
    "wide_budget_matched": {
        "ranks": {"q_proj": 5, "o_proj": 4, "k_proj": 6, "v_proj": 6},
        "alphas": {"q_proj": 10, "o_proj": 8, "k_proj": 12, "v_proj": 12},
        "tensors": 224,
        "preflight": (
            "5970a57396215af98f2fe3a645d413841b0566d9b8d8b63370ca8a418d7640da"
        ),
        "receipt": ("06bfe9c62b81f2e6f1e7ed93989251b478090636e8a6aea8963d8eb52cd73338"),
    },
}
EXPECTED_RANKING = ("q_plus_o", "wide_budget_matched", "q_only")

_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_BODY_KEYS = frozenset(
    {"answer", "content", "input_ids", "preview", "prompt", "token_ids"}
)


class ControlledProxyFollowupAuditError(RuntimeError):
    """A stable, content-free audit failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ControlledProxyFollowupAuditError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _derived_seed(domain: str, master_seed: int) -> int:
    preimage = (
        f"anchor.controlled-proxy-followup.seed.v1\0{domain}\0{master_seed}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(preimage).digest()[:4], "big") & 0x7FFFFFFF


def _compact_json_sha256(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(data)


def _strict_json(data: bytes, code: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite number")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite number")
        return parsed

    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControlledProxyFollowupAuditError(code) from exc

    def all_finite(item: object) -> bool:
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, Mapping):
            return all(all_finite(child) for child in item.values())
        if isinstance(item, list):
            return all(all_finite(child) for child in item)
        return True

    if not isinstance(value, Mapping) or not all_finite(value):
        _fail(code)
    return value


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _list(value: object, code: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for child in value.values():
            result.update(_walk_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_walk_keys(child))
        return result
    return set()


def _relative_path(value: object, *, exact: str | None, code: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        _fail(code)
    text = value.as_posix() if isinstance(value, Path) else os.fspath(value)
    if (
        not text
        or "\\" in text
        or "//" in text
        or Path(text).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(text).parts)
        or (exact is not None and text != exact)
    ):
        _fail(code)
    return text


def _has_reparse_attribute(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class BytesSnapshot:
    relative_path: str
    path: Path
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


def _safe_root(repo_root: Path) -> Path:
    requested = Path(repo_root).absolute()
    try:
        root = requested.resolve(strict=True)
        root_stat = root.lstat()
    except OSError as exc:
        raise ControlledProxyFollowupAuditError("followup_repo_root_invalid") from exc
    if (
        root != requested
        or not root.is_dir()
        or stat.S_ISLNK(root_stat.st_mode)
        or _has_reparse_attribute(root_stat)
    ):
        _fail("followup_repo_root_invalid")
    return root


def _safe_file(root: Path, relative_value: object, *, exact: str) -> Path:
    relative = _relative_path(relative_value, exact=exact, code="followup_path_invalid")
    current = root
    try:
        for part in Path(relative).parts:
            current = current / part
            current_stat = current.lstat()
            if stat.S_ISLNK(current_stat.st_mode) or _has_reparse_attribute(
                current_stat
            ):
                _fail("followup_path_reparse_rejected")
        if not current.is_file() or current.resolve(strict=True) != current.absolute():
            _fail("followup_path_invalid")
    except ControlledProxyFollowupAuditError:
        raise
    except OSError as exc:
        raise ControlledProxyFollowupAuditError("followup_path_invalid") from exc
    return current


def _snapshot(root: Path, relative_path: str, *, max_bytes: int) -> BytesSnapshot:
    path = _safe_file(root, relative_path, exact=relative_path)
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or _has_reparse_attribute(before)
                or before.st_size > max_bytes
            ):
                _fail("followup_snapshot_invalid")
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except ControlledProxyFollowupAuditError:
        raise
    except OSError as exc:
        raise ControlledProxyFollowupAuditError("followup_snapshot_invalid") from exc
    identity = _identity(before)
    if (
        identity != _identity(after)
        or identity != _identity(current)
        or len(data) != after.st_size
    ):
        _fail("followup_snapshot_changed")
    return BytesSnapshot(
        relative_path,
        path,
        data,
        _sha256(data),
        len(data),
        identity,
    )


def _verify_unchanged(root: Path, snapshots: Sequence[BytesSnapshot]) -> None:
    for expected in snapshots:
        try:
            current = _snapshot(
                root, expected.relative_path, max_bytes=max(expected.size, 1)
            )
        except ControlledProxyFollowupAuditError as exc:
            raise ControlledProxyFollowupAuditError("followup_input_changed") from exc
        if (
            current.sha256 != expected.sha256
            or current.size != expected.size
            or current.identity != expected.identity
        ):
            _fail("followup_input_changed")


def _sidecar(digest: str, filename: str) -> bytes:
    if _SHA256_RE.fullmatch(digest) is None or "/" in filename or "\\" in filename:
        _fail("followup_sidecar_invalid")
    return f"{digest}  {filename}\n".encode("ascii")


def _validate_schema(schema: Mapping[str, Any], instance: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
    except (SchemaError, ValidationError) as exc:
        raise ControlledProxyFollowupAuditError(
            "followup_contract_schema_invalid"
        ) from exc


def _binding(value: object, expected_path: str, expected_sha: str) -> None:
    binding = _mapping(value, "followup_contract_identity_invalid")
    if binding != {"path": expected_path, "sha256": expected_sha}:
        _fail("followup_contract_identity_invalid")


def _validate_contract_root(
    contract: Mapping[str, Any],
    *,
    schema_sha256: str,
    implementation_sha256: str,
) -> None:
    if _FORBIDDEN_BODY_KEYS.intersection(_walk_keys(contract)):
        _fail("followup_protected_body_field_rejected")
    evidence = _mapping(contract.get("evidence"), "followup_contract_identity_invalid")
    if (
        schema_sha256 != EXPECTED_SCHEMA_SHA256
        or contract.get("schema_sha256") != EXPECTED_SCHEMA_SHA256
        or evidence.get("consumer_repository") != "anchor-moe-lora-neural-swarm"
        or evidence.get("consumer_branch") != "research/neural-swarm-kv"
        or evidence.get("consumer_commit") != EXPECTED_CONSUMER_COMMIT
        or evidence.get("consumer_parent_commit") != EXPECTED_CONSUMER_PARENT
        or evidence.get("consumer_tree") != EXPECTED_CONSUMER_TREE
        or evidence.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256
        or evidence.get("comparison_publication_status")
        != "local_ignored_evidence_not_release_artifact"
        or evidence.get("comparison_authentication_scope")
        != "exact_out_of_tree_report_bytes_plus_closed_semantic_projection_only"
        or evidence.get("transitive_run_artifacts_reopened") is not False
        or evidence.get("consumer_git_blobs_reopened_by_producer_auditor") is not False
    ):
        _fail("followup_contract_identity_invalid")
    _binding(evidence.get("comparison"), SOURCE_PATH, EXPECTED_SOURCE_SHA256)
    _binding(
        evidence.get("comparison_sidecar"),
        SOURCE_SIDECAR_PATH,
        EXPECTED_SOURCE_SIDECAR_SHA256,
    )
    _binding(
        evidence.get("comparison_schema"),
        "configs/research/qwen_budget_matched_ablation_comparison.schema.json",
        EXPECTED_COMPARISON_SCHEMA_SHA256,
    )
    _binding(
        evidence.get("training_config"),
        "configs/training/qwen2_5_1_5b_synthetic_scaffold_budget_matched_v1.yaml",
        "490b0e18fce004a44b97b4c4ab2a3d0f9d0809e1d1f20a2fdc8f93938490e9c2",
    )
    _binding(
        evidence.get("runner"),
        "src/anchor_mvp/training/qwen_budget_matched_ablation.py",
        "ea2b822f0aefc3f7deea9654849a2a5ea87b2e1c35356ad7615941bcb1cefd9b",
    )
    _binding(
        evidence.get("auditor"),
        "src/anchor_mvp/training/qwen_budget_matched_ablation_audit.py",
        "c7328cdc3083d52ee8d40de94ee0f18169d920cc448b75f2860868911c522b71",
    )
    _binding(
        evidence.get("producer_auditor"),
        IMPLEMENTATION_PATH,
        implementation_sha256,
    )


def _validate_closed_source(source: Mapping[str, Any]) -> None:
    """Validate a projection stricter than the open upstream nested schema."""

    code = "followup_source_semantics_invalid"
    _exact_keys(
        source,
        {
            "arms",
            "audit",
            "auditor",
            "claims",
            "common",
            "config",
            "dataset",
            "ranking",
            "runner",
            "schema_version",
            "status",
        },
        code,
    )
    if (
        source["schema_version"]
        != "anchor.qwen25-1.5b-budget-matched-ablation-comparison.v1"
        or source["status"] != "passed_controlled_proxy_comparison_only"
    ):
        _fail(code)

    common = _mapping(source["common"], code)
    _exact_keys(
        common,
        {
            "base_hash",
            "config_sha256",
            "dataset_manifest_sha256",
            "eval_macro_loss_before",
            "eval_micro_nll_before",
            "eval_target_tokens",
            "ordered_record_ids_sha256",
            "ordered_tokenized_examples_sha256",
            "partition_sha256",
            "train_full_tokens",
        },
        code,
    )
    if (
        common["base_hash"] != EXPECTED_BASE_SHA256
        or common["config_sha256"] != EXPECTED_EFFECTIVE_CONFIG_SHA256
        or common["dataset_manifest_sha256"] != EXPECTED_DATASET_MANIFEST_SHA256
        or common["ordered_record_ids_sha256"] != EXPECTED_RECORD_ORDER_SHA256
        or common["ordered_tokenized_examples_sha256"]
        != EXPECTED_TOKENIZED_TRAIN_SHA256
        or common["partition_sha256"] != EXPECTED_PARTITIONS
    ):
        _fail(code)

    dataset = _mapping(source["dataset"], code)
    _exact_keys(
        dataset,
        {
            "eval_ordered_token_views",
            "eval_proxy_records",
            "manifest_sha256",
            "partition_sha256",
            "train_records",
        },
        code,
    )
    views = _mapping(dataset["eval_ordered_token_views"], code)
    _exact_keys(
        views,
        {
            "algorithm",
            "ordered_views_sha256",
            "raw_record_ids_emitted",
            "raw_token_ids_emitted",
            "records",
        },
        code,
    )
    if (
        dataset["train_records"] != 80
        or dataset["eval_proxy_records"] != 20
        or dataset["manifest_sha256"] != EXPECTED_DATASET_MANIFEST_SHA256
        or dataset["partition_sha256"] != EXPECTED_PARTITIONS
        or views
        != {
            "algorithm": "record_id_ascending_sha256_signed_int64_token_views_v1",
            "ordered_views_sha256": EXPECTED_EVAL_VIEWS_SHA256,
            "raw_record_ids_emitted": False,
            "raw_token_ids_emitted": False,
            "records": 20,
        }
    ):
        _fail(code)

    arms = _list(source["arms"], code)
    if len(arms) != 3:
        _fail(code)
    source_by_profile: dict[str, Mapping[str, Any]] = {}
    arm_keys = {
        "adapter_artifact_sha256",
        "bundle_improved_count",
        "eval_macro_loss_after",
        "eval_macro_loss_delta_percent",
        "eval_micro_nll_after",
        "eval_ppl_after",
        "integrity",
        "peak_allocated_vram_bytes",
        "preflight",
        "profile",
        "receipt",
        "saved_tensor_audit",
        "train_tokens_per_second",
        "train_wall_seconds",
    }
    for raw_arm in arms:
        arm = _mapping(raw_arm, code)
        _exact_keys(arm, arm_keys, code)
        profile = arm.get("profile")
        if not isinstance(profile, str) or profile not in EXPECTED_ARMS:
            _fail(code)
        if profile in source_by_profile:
            _fail(code)
        source_by_profile[profile] = arm
        expected = EXPECTED_ARMS[profile]
        artifacts = _mapping(arm["adapter_artifact_sha256"], code)
        integrity = _mapping(arm["integrity"], code)
        preflight = _mapping(arm["preflight"], code)
        receipt = _mapping(arm["receipt"], code)
        tensor_audit = _mapping(arm["saved_tensor_audit"], code)
        _exact_keys(
            artifacts, {"adapter_config.json", "adapter_model.safetensors"}, code
        )
        _exact_keys(
            integrity,
            {
                "all_lora_tensors_observed_nonzero_gradient",
                "base_unchanged",
                "save_reload_logit_delta",
            },
            code,
        )
        _exact_keys(preflight, {"path", "sha256"}, code)
        _exact_keys(receipt, {"path", "sha256"}, code)
        _exact_keys(
            tensor_audit,
            {
                "all_shapes_valid",
                "alphas",
                "parameters",
                "ranks",
                "tensor_count",
                "unexpected_tensors",
            },
            code,
        )
        if (
            tensor_audit["ranks"] != expected["ranks"]
            or tensor_audit["alphas"] != expected["alphas"]
            or tensor_audit["parameters"] != 1_376_256
            or tensor_audit["tensor_count"] != expected["tensors"]
            or tensor_audit["all_shapes_valid"] is not True
            or tensor_audit["unexpected_tensors"] != 0
            or preflight.get("sha256") != expected["preflight"]
            or receipt.get("sha256") != expected["receipt"]
            or arm["bundle_improved_count"] != 2
            or integrity
            != {
                "all_lora_tensors_observed_nonzero_gradient": True,
                "base_unchanged": True,
                "save_reload_logit_delta": 0.0,
            }
        ):
            _fail(code)
    if tuple(source_by_profile) != tuple(EXPECTED_ARMS):
        _fail(code)

    ranking = _list(source["ranking"], code)
    derived = tuple(
        sorted(
            source_by_profile,
            key=lambda profile: source_by_profile[profile]["eval_macro_loss_after"],
        )
    )
    if tuple(ranking) != EXPECTED_RANKING or derived != EXPECTED_RANKING:
        _fail(code)

    for role, expected_path, expected_sha in (
        (
            "config",
            "configs/training/qwen2_5_1_5b_synthetic_scaffold_budget_matched_v1.yaml",
            "490b0e18fce004a44b97b4c4ab2a3d0f9d0809e1d1f20a2fdc8f93938490e9c2",
        ),
        (
            "runner",
            "src/anchor_mvp/training/qwen_budget_matched_ablation.py",
            "ea2b822f0aefc3f7deea9654849a2a5ea87b2e1c35356ad7615941bcb1cefd9b",
        ),
        (
            "auditor",
            "src/anchor_mvp/training/qwen_budget_matched_ablation_audit.py",
            "c7328cdc3083d52ee8d40de94ee0f18169d920cc448b75f2860868911c522b71",
        ),
    ):
        _binding(source[role], expected_path, expected_sha)

    source_claims = _mapping(source["claims"], code)
    if source_claims != {
        "controlled_proxy_only": True,
        "deterministic_algorithms_enabled": False,
        "diagnostic_only": True,
        "eval_proxy_is_heldout": False,
        "formal": False,
        "multi_seed_validated": False,
        "statistical_significance_claimed": False,
        "training_authorized": False,
    }:
        _fail(code)
    source_audit = _mapping(source["audit"], code)
    if source_audit != {
        "final_directories_reopened": True,
        "gpu_requests": 0,
        "heldout_reads": 0,
        "model_loads": 0,
        "network_requests": 0,
        "protected_body_reads": 0,
        "provider_requests": 0,
        "saved_tensors_fully_validated": True,
        "tokenizer_loads": 1,
    }:
        _fail(code)


def _validate_projection(
    contract: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    code = "followup_projection_mismatch"
    controlled = _mapping(contract["controlled_contract"], code)
    source_common = _mapping(source["common"], code)
    source_dataset = _mapping(source["dataset"], code)
    views = _mapping(source_dataset["eval_ordered_token_views"], code)
    if (
        controlled["frozen_base_named_parameter_fingerprint_sha256"]
        != source_common["base_hash"]
        or controlled["base_fingerprint_algorithm"]
        != "sha256_named_parameter_iteration_utf8_name_nul_ascii_raw_cpu_contiguous_tensor_sha256_lf_v1"
        or controlled["train_records"] != source_dataset["train_records"]
        or controlled["eval_proxy_records"] != source_dataset["eval_proxy_records"]
        or controlled["trainable_parameters_per_arm"] != 1_376_256
        or controlled["ordered_record_ids_sha256"]
        != source_common["ordered_record_ids_sha256"]
        or controlled["ordered_tokenized_train_sha256"]
        != source_common["ordered_tokenized_examples_sha256"]
        or controlled["ordered_eval_token_views_sha256"]
        != views["ordered_views_sha256"]
        or controlled["train_order_digest_algorithm"]
        != "sha256_seeded_record_id_order_and_signed_int64_token_views_v1"
        or controlled["eval_view_digest_algorithm"] != views["algorithm"]
        or controlled["eval_macro_loss_before"]
        != source_common["eval_macro_loss_before"]
    ):
        _fail(code)

    contract_arms = _list(contract["arms"], code)
    source_arms = _list(source["arms"], code)
    if len({arm["profile"] for arm in contract_arms}) != 3:
        _fail(code)
    source_by_profile = {arm["profile"]: arm for arm in source_arms}
    for arm in contract_arms:
        source_arm = source_by_profile.get(arm["profile"])
        if source_arm is None:
            _fail(code)
        tensor_audit = source_arm["saved_tensor_audit"]
        integrity = source_arm["integrity"]
        if arm != {
            "profile": source_arm["profile"],
            "ranks": tensor_audit["ranks"],
            "alphas": tensor_audit["alphas"],
            "trainable_parameters": tensor_audit["parameters"],
            "trainable_tensors": tensor_audit["tensor_count"],
            "preflight_sha256": source_arm["preflight"]["sha256"],
            "receipt_sha256": source_arm["receipt"]["sha256"],
            "eval_macro_loss_after": source_arm["eval_macro_loss_after"],
            "eval_macro_loss_delta_percent": source_arm[
                "eval_macro_loss_delta_percent"
            ],
            "train_tokens_per_second": source_arm["train_tokens_per_second"],
            "eval_bundles_improved": source_arm["bundle_improved_count"],
            "all_gradients_finite_nonzero": integrity[
                "all_lora_tensors_observed_nonzero_gradient"
            ],
            "base_unchanged": integrity["base_unchanged"],
            "reload_logit_delta": integrity["save_reload_logit_delta"],
        }:
            _fail(code)

    interpretation = _mapping(contract["interpretation"], code)
    producer_followup = _mapping(contract["producer_followup"], code)
    policy = _mapping(producer_followup["canonical_policy"], code)
    if (
        interpretation["observed_ranking"] != source["ranking"]
        or interpretation["proxy_leader"] != source["ranking"][0]
        or policy["producer_to_consumer_profile_alias"]
        != {
            "q_only": "q_only",
            "q_plus_o": "q_plus_o",
            "wide_lora": "wide_budget_matched",
        }
        or policy["observed_proxy_split_group_key"] != "source_bundle_id"
        or policy["future_formal_split_group_key"] != "task_bundle_sha256"
        or policy["historical_split_identity_rewritten"] is not False
    ):
        _fail(code)

    replication = _mapping(producer_followup["replication_phase"], code)
    masters = replication["seed_schedule"]
    domains = _mapping(replication["seed_domain_schedules"], code)
    if (
        replication["seed_derivation_algorithm"]
        != "sha256_utf8_anchor_domain_nul_decimal_master_first_u32_mask31_v1"
        or replication["throughput_repetitions_per_arm_per_seed"] != 6
        or replication["balanced_permutation_arm_execution_order"] is not True
        or replication["throughput_order_schedule"]
        != [list(order) for order in permutations(EXPECTED_ARMS)]
        or replication["throughput_schedule_scope"]
        != "all_six_arm_order_permutations_within_each_seed"
        or replication["primary_metric"]
        != "bundle_macro_eval_loss_delta_vs_shared_base"
        or replication["replication_readiness_semantics"]
        != "integrity_and_preregistered_coverage_gate_independent_of_arm_ranking"
    ):
        _fail(code)
    for domain in ("adapter_init", "record_order", "cuda"):
        schedule = _mapping(domains[domain], code)
        values = [_derived_seed(domain, int(master)) for master in masters]
        if schedule != {
            "domain": domain,
            "values": values,
            "schedule_digest_algorithm": (
                "sha256_utf8_canonical_json_integer_array_v1"
            ),
            "sha256": _compact_json_sha256(values),
        }:
            _fail(code)

    fixture_phase = _mapping(producer_followup["stratified_fixture_phase"], code)
    length_phase = _mapping(producer_followup["length_sweep_phase"], code)
    if (
        fixture_phase["group_key"] != "task_bundle_sha256"
        or fixture_phase["discovery_source_bundle_ids_sha256"]
        != "3014e1aea293b47a63a26beb72e37c8e45378df970df4433663e9ee2da6f2235"
        or fixture_phase["source_bundle_id_zero_overlap_proven"] is not False
        or fixture_phase["source_bundle_id_overlap_status"]
        != "unavailable_until_confirmation_inventory_exists"
        or fixture_phase["independent_confirmation_claimed"] is not False
        or fixture_phase["namespace_neutral_blueprint_zero_overlap_status"]
        != "unavailable_until_both_blueprint_inventories_exist"
        or length_phase["bucket_measurement"]
        != "total_tokens_equals_input_tokens_plus_reserved_output_tokens"
        or max(length_phase["planned_initial_buckets_tokens"])
        > length_phase["current_model_max_position_embeddings"]
        or length_phase["capability_gate_current_identity_status"]
        != "blocked_requires_new_model_or_rope_identity"
    ):
        _fail(code)

    audit = _mapping(contract["audit"], code)
    if audit != {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "gold_body_reads": 0,
        "heldout_body_reads": 0,
        "protected_scaffold_body_reads": 0,
        "sample_bodies_copied": 0,
    }:
        _fail(code)
    claims = _mapping(contract["claims"], code)
    if (
        claims["diagnostic_only"] is not True
        or claims["controlled_proxy_only"] is not True
        or claims["q_plus_o_proxy_leader_under_exact_contract"] is not True
        or claims["training_authorized"] is not False
        or claims["formal_training_authorized"] is not False
        or claims["formal"] is not False
        or any(
            claims[key] is not False
            for key in (
                "producer_selected_training_winner",
                "mechanistic_interpretation_claimed",
                "sample_efficiency_claimed",
                "compute_matched_claimed",
                "statistical_significance_claimed",
                "multi_seed_validated",
                "bundle_generalization_validated",
                "long_context_generalization_claimed",
                "physical_kv_reuse_claimed",
                "zero_copy_claimed",
                "full_generation_kv_shared_claimed",
                "throughput_superiority_claimed",
                "quality_validated",
                "eval_proxy_is_heldout",
            )
        )
    ):
        _fail(code)


def audit_followup(
    repo_root: Path,
    contract_path: Path | str = CONTRACT_PATH,
) -> dict[str, object]:
    """Authenticate and audit the frozen metadata-only follow-up contract."""

    root = _safe_root(Path(repo_root))
    relative_contract = _relative_path(
        contract_path, exact=CONTRACT_PATH, code="followup_path_invalid"
    )
    paths_and_limits = (
        (relative_contract, 1_000_000),
        (CONTRACT_SIDECAR_PATH, 1024),
        (SCHEMA_PATH, 1_000_000),
        (IMPLEMENTATION_PATH, 1_000_000),
        (SOURCE_PATH, 1_000_000),
        (SOURCE_SIDECAR_PATH, 1024),
    )
    snapshots = tuple(
        _snapshot(root, path, max_bytes=limit) for path, limit in paths_and_limits
    )
    by_path = {snapshot.relative_path: snapshot for snapshot in snapshots}
    contract_snapshot = by_path[relative_contract]
    contract_sidecar = by_path[CONTRACT_SIDECAR_PATH]
    schema_snapshot = by_path[SCHEMA_PATH]
    implementation_snapshot = by_path[IMPLEMENTATION_PATH]
    source_snapshot = by_path[SOURCE_PATH]
    source_sidecar = by_path[SOURCE_SIDECAR_PATH]

    if contract_sidecar.data != _sidecar(
        contract_snapshot.sha256, Path(relative_contract).name
    ):
        _fail("followup_contract_sidecar_invalid")
    if source_sidecar.data != _sidecar(source_snapshot.sha256, "comparison.json"):
        _fail("followup_source_sidecar_invalid")
    if (
        source_snapshot.sha256 != EXPECTED_SOURCE_SHA256
        or source_sidecar.sha256 != EXPECTED_SOURCE_SIDECAR_SHA256
    ):
        _fail("followup_source_identity_invalid")

    # Hash, parse, schema validation, and semantics all consume these exact
    # in-memory byte snapshots.  No second pre-validation body read occurs.
    contract = _strict_json(contract_snapshot.data, "followup_contract_json_invalid")
    schema = _strict_json(schema_snapshot.data, "followup_schema_json_invalid")
    source = _strict_json(source_snapshot.data, "followup_source_json_invalid")
    if (
        _strict_json(contract_snapshot.data, "followup_contract_json_invalid")
        != contract
    ):
        _fail("followup_contract_reparse_mismatch")
    if _strict_json(schema_snapshot.data, "followup_schema_json_invalid") != schema:
        _fail("followup_schema_reparse_mismatch")
    if _strict_json(source_snapshot.data, "followup_source_json_invalid") != source:
        _fail("followup_source_reparse_mismatch")
    _validate_schema(schema, contract)
    _validate_contract_root(
        contract,
        schema_sha256=schema_snapshot.sha256,
        implementation_sha256=implementation_snapshot.sha256,
    )
    _validate_closed_source(source)
    _validate_projection(contract, source)

    _verify_unchanged(root, snapshots)
    return {
        "schema_version": contract["schema_version"],
        "status": "passed",
        "contract_status": contract["status"],
        "claim_scope": contract["claim_scope"],
        "arm_count": len(contract["arms"]),
        "observed_ranking": list(contract["interpretation"]["observed_ranking"]),
        "diagnostic_only": contract["claims"]["diagnostic_only"],
        "training_authorized": contract["claims"]["training_authorized"],
        "formal_training_authorized": contract["claims"]["formal_training_authorized"],
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
    }


__all__ = [
    "CONTRACT_PATH",
    "ControlledProxyFollowupAuditError",
    "audit_followup",
]
