"""Authenticated consumer for the synthetic natural-language scaffold fixture.

This module is deliberately model-free.  It authenticates the producer's
frozen bytes, validates the closed JSON schemas, proves paired-ablation and
split invariants, and materializes contract-only views.  It never authorizes
training or claims that a tokenizer, adapter, or physical KV implementation is
available.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import yaml

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
except ImportError:  # pragma: no cover - explicit dependency gate below
    Draft202012Validator = None  # type: ignore[assignment,misc]
    SchemaError = ValueError  # type: ignore[assignment,misc]

from .query_specialization import (
    TaskBoardSidecar,
    build_training_view,
    canonical_taskboard_sidecar,
    load_taskboard_sidecar_dataset,
)
from .natural_language_scaffold_runtime import PlannerScaffold


CONFIG_SHA256 = "e81fc742ffb99d0f71ff3cc03ba68e82644ed7f539eb190eb2945bec7567fe38"
RECORD_SCHEMA_SHA256 = (
    "84efd818a52334e6b63a2132126d4a133ea3a143e13d11431bda3a242ba67d14"
)
MANIFEST_SCHEMA_SHA256 = (
    "8034b673798b0dc8b8a620b53a4a92e5565b5f9d936ad76ef8d30add50a98b16"
)
SMOKE_SCHEMA_SHA256 = "3944b28736ad1b6df9088ec69753c471d52ddb4f2753a974a23a29343c2cba5b"
SMOKE_CONTRACT_SHA256 = (
    "46bca04c358cc1e80f55c7eacff36fdf3f11a83efda52ad4386035ca5d614719"
)
PRODUCER_IMPLEMENTATION_SHA256 = (
    "09e7dae7f0fcafabbf2ea682504355d2c95c545764295c84eace7d16b3332330"
)
FIXTURE_MANIFEST_SHA256 = (
    "25e40da8fea46ba018ae0031fa8c37da38b59438bb92d9052a915fd256d822dc"
)
FIXTURE_MANIFEST_SIDECAR_SHA256 = (
    "9de02cee4bf902e8ef71c48d70e1d327659f4c31b125178728b7eb9401c6751f"
)
CONSUMER_CONFIG_SHA256 = (
    "79cf993e4f4496b57786602bcbec3ac9048d4ad2a9fd6d5033bff64ab65c0640"
)

RECORD_SCHEMA_VERSION = "anchor.natural-language-scaffold.v1"
MANIFEST_SCHEMA_VERSION = "anchor.natural-language-scaffold-manifest.v1"
SMOKE_SCHEMA_VERSION = "anchor.natural-language-scaffold-smoke-contract.v1"
VIEW_SCHEMA_VERSION = "anchor.natural-language-scaffold-contract-view.v1"

FIXED_FILES = (
    (
        "train/json_only.jsonl",
        "train",
        "noisy",
        "json_only",
        5,
        96866,
        "6aad30ccc1aaaac432a76559e15879acac4f82f382c4020a8e7c1831b2ef2751",
    ),
    (
        "train/concise_rationale_plus_json.jsonl",
        "train",
        "noisy",
        "concise_rationale_plus_json",
        5,
        98447,
        "02e85be2477fd8937fe3dc3f6222c771d7725864232e11ae8338fcc061d02617",
    ),
    (
        "calibration/json_only.jsonl",
        "calibration",
        "clean",
        "json_only",
        5,
        91711,
        "2e30596b078a47416aec51c80aaaa5fa946d7f0cbd2231f49974ba4e6f7ef065",
    ),
    (
        "calibration/concise_rationale_plus_json.jsonl",
        "calibration",
        "clean",
        "concise_rationale_plus_json",
        5,
        93022,
        "260f1b7d14f1c1563b7a235bc475992f628f70f1e5e67351d05b76aba2299168",
    ),
)

REQUIRED_ROLES = {
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
}
REQUIRED_STAGES = {
    "planner",
    "tool_policy",
    "domain_builder",
    "domain_review",
    "security",
}
PAIR_VARIANTS = {"json_only", "concise_rationale_plus_json"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DENIED_TOKEN_FIELDS = {
    "token_index",
    "start_token",
    "end_token",
    "invocation_token_ids",
    "position_ids",
}
_PAIR_VARIANT_FIELDS = {
    "record_id",
    "scaffold_variant",
    "concise_rationale_summary",
    "scaffold_text",
    "scaffold_text_sha256",
}
_AUTHENTICATED_RECORD_CAPABILITY = object()
_BOUND_SCAFFOLD_CAPABILITY = object()


class NaturalLanguageScaffoldConsumerError(RuntimeError):
    """A stable, content-free consumer failure."""


def _fail(code: str) -> None:
    raise NaturalLanguageScaffoldConsumerError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_compatible(value: object) -> object:
    """Return detached JSON containers for canonical serialization."""

    if isinstance(value, Mapping):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(child) for child in value]
    return value


def _freeze_json(value: object) -> object:
    """Recursively detach and freeze authenticated JSON values."""

    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                _fail("scaffold_authenticated_snapshot_invalid")
            frozen[key] = _freeze_json(child)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    _fail("scaffold_authenticated_snapshot_invalid")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class _BytesSnapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True, init=False)
class AuthenticatedScaffoldRecord(Mapping[str, Any]):
    """Immutable record minted only after schema/partition authentication."""

    _data: Mapping[str, Any]
    canonical_sha256: str
    manifest_sha256: str
    scaffold_partition_sha256: str
    _capability: object

    def __init__(
        self,
        *,
        data: Mapping[str, Any],
        canonical_sha256: str,
        manifest_sha256: str,
        scaffold_partition_sha256: str,
        _capability: object,
    ) -> None:
        if _capability is not _AUTHENTICATED_RECORD_CAPABILITY:
            _fail("scaffold_authenticated_record_factory_required")
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "canonical_sha256", canonical_sha256)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(
            self,
            "scaffold_partition_sha256",
            scaffold_partition_sha256,
        )
        object.__setattr__(self, "_capability", _capability)
        self._validated_data()

    def __getitem__(self, key: str) -> Any:
        return self._validated_data()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._validated_data())

    def __len__(self) -> int:
        return len(self._validated_data())

    def _validated_data(self) -> Mapping[str, Any]:
        if self._capability is not _AUTHENTICATED_RECORD_CAPABILITY:
            _fail("scaffold_authenticated_record_invalid")
        for digest in (
            self.canonical_sha256,
            self.manifest_sha256,
            self.scaffold_partition_sha256,
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                _fail("scaffold_authenticated_record_invalid")
        if _sha256(_canonical_bytes(self._data)) != self.canonical_sha256:
            _fail("scaffold_authenticated_record_hash_invalid")
        return self._data


def _mint_authenticated_record(
    record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    scaffold_partition_sha256: str,
) -> AuthenticatedScaffoldRecord:
    frozen = _mapping(
        _freeze_json(record),
        "scaffold_authenticated_snapshot_invalid",
    )
    return AuthenticatedScaffoldRecord(
        data=frozen,
        canonical_sha256=_sha256(_canonical_bytes(frozen)),
        manifest_sha256=manifest_sha256,
        scaffold_partition_sha256=scaffold_partition_sha256,
        _capability=_AUTHENTICATED_RECORD_CAPABILITY,
    )


def _authenticated_record_data(
    record: object,
) -> Mapping[str, Any]:
    if not isinstance(record, AuthenticatedScaffoldRecord):
        _fail("scaffold_authenticated_record_required")
    return record._validated_data()


@dataclass(frozen=True)
class ScaffoldFixture:
    records: tuple[AuthenticatedScaffoldRecord, ...]
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]


@dataclass(frozen=True, init=False)
class BoundScaffold:
    """One scaffold record cross-bound to an authenticated TaskBoard row."""

    record: AuthenticatedScaffoldRecord
    sidecar: TaskBoardSidecar
    record_sha256: str
    sidecar_sha256: str
    binding_sha256: str
    _capability: object

    def __init__(
        self,
        *,
        record: AuthenticatedScaffoldRecord,
        sidecar: TaskBoardSidecar,
        record_sha256: str,
        sidecar_sha256: str,
        binding_sha256: str,
        _capability: object,
    ) -> None:
        if _capability is not _BOUND_SCAFFOLD_CAPABILITY:
            _fail("scaffold_bound_factory_required")
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "sidecar", sidecar)
        object.__setattr__(self, "record_sha256", record_sha256)
        object.__setattr__(self, "sidecar_sha256", sidecar_sha256)
        object.__setattr__(self, "binding_sha256", binding_sha256)
        object.__setattr__(self, "_capability", _capability)


def _read_bytes_snapshot(path: Path, code: str) -> _BytesSnapshot:
    try:
        if not path.is_file() or path.is_symlink():
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        raise NaturalLanguageScaffoldConsumerError(code) from exc
    identity = _stat_identity(after)
    if (
        _stat_identity(before) != identity
        or _stat_identity(current) != identity
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(path, data, _sha256(data), identity)


def _verify_snapshot_current(snapshot: _BytesSnapshot, code: str) -> None:
    try:
        current = snapshot.path.stat()
    except OSError as exc:
        raise NaturalLanguageScaffoldConsumerError(code) from exc
    if snapshot.path.is_symlink() or _stat_identity(current) != snapshot.identity:
        _fail(code)


def _json_value(snapshot: _BytesSnapshot, code: str) -> Any:
    try:
        return json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NaturalLanguageScaffoldConsumerError(code) from exc


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _load_consumer_config_snapshot(
    repository: Path,
    consumer_config_path: str | Path | None,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], _BytesSnapshot]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        _fail("scaffold_expected_consumer_config_hash_invalid")
    path = (
        repository / "configs/research/natural_language_scaffold_consumer_v1.yaml"
        if consumer_config_path is None
        else Path(consumer_config_path).expanduser().resolve()
    )
    snapshot = _read_bytes_snapshot(path, "scaffold_consumer_config_unavailable")
    if snapshot.sha256 != expected_sha256:
        _fail("scaffold_consumer_config_hash_invalid")
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise NaturalLanguageScaffoldConsumerError(
            "scaffold_consumer_config_invalid"
        ) from exc
    config = _mapping(value, "scaffold_consumer_config_invalid")
    producer = _mapping(
        config.get("producer_contract"), "scaffold_consumer_config_invalid"
    )
    expected_producer = {
        "commit": "03ea0214567289e4f46378d4731b0177c18a1402",
        "config": "configs/research/swebench_natural_language_scaffold_v1.yaml",
        "config_sha256": CONFIG_SHA256,
        "record_schema": "configs/research/swebench_natural_language_scaffold_sidecar.schema.json",
        "record_schema_sha256": RECORD_SCHEMA_SHA256,
        "manifest_schema": "configs/research/swebench_natural_language_scaffold_manifest.schema.json",
        "manifest_schema_sha256": MANIFEST_SCHEMA_SHA256,
        "smoke_schema": "configs/research/swebench_natural_language_scaffold_smoke_contract.schema.json",
        "smoke_schema_sha256": SMOKE_SCHEMA_SHA256,
        "smoke_contract": "configs/research/swebench_natural_language_scaffold_smoke_v1.yaml",
        "smoke_contract_sha256": SMOKE_CONTRACT_SHA256,
        "producer_implementation_sha256": PRODUCER_IMPLEMENTATION_SHA256,
        "fixture": "fixtures/research/swebench_natural_language_scaffold",
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "fixture_manifest_sidecar_sha256": FIXTURE_MANIFEST_SIDECAR_SHA256,
    }
    paired = _mapping(config.get("paired_ablation"), "scaffold_consumer_config_invalid")
    runtime = _mapping(
        config.get("runtime_boundary"), "scaffold_consumer_config_invalid"
    )
    gates = _mapping(config.get("gates"), "scaffold_consumer_config_invalid")
    if (
        config.get("schema_version")
        != "anchor.natural-language-scaffold-consumer-config.v1"
        or config.get("claim_scope") != "synthetic_fixture_contract_only"
        or dict(producer) != expected_producer
        or paired.get("split_group_key") != "task_bundle_sha256"
        or tuple(paired.get("variants", ()))
        != ("json_only", "concise_rationale_plus_json")
        or paired.get("paired_inputs_identical") is not True
        or paired.get("variant_specific_targets") is not True
        or paired.get("roles_per_bundle") != 5
        or paired.get("source_bundles") != 2
        or runtime.get("semantics") != "explicit_two_request_commit_boundary"
        or runtime.get("planner_private_kv_transfer") is not False
        or runtime.get("committed_scaffold_reencode") != "frozen_base_adapter_off"
        or runtime.get("expert_activation_request") != "next_request"
        or runtime.get("same_request_activation") is not False
        or runtime.get("mid_request_generated_activation") is not False
        or runtime.get("tokenizer_binding_status") != "unbound"
        or gates.get("strict_schema") is not True
        or gates.get("physical_sha256") is not True
        or gates.get("mandatory_manifest_sha256_sidecar") is not True
        or gates.get("single_bytes_snapshot") is not True
        or gates.get("paired_ablation") is not True
        or gates.get("two_request_runtime") is not True
        or gates.get("body_exclusion") is not True
        or gates.get("training_authorized") is not False
        or gates.get("quality_validated") is not False
        or any(
            gates.get(key) != 0
            for key in (
                "provider_requests",
                "model_loads",
                "gpu_requests",
                "network_requests",
            )
        )
    ):
        _fail("scaffold_consumer_config_contract_invalid")
    return config, snapshot


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return False
    return ".." not in Path(value.replace("\\", "/")).parts


def _resolve_artifact_path(root: Path, relative: object) -> Path:
    if not _safe_relative_path(relative):
        _fail("scaffold_partition_path_invalid")
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail("scaffold_partition_path_invalid")
    return candidate


def _schema_type_matches(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _resolve_local_ref(
    root_schema: Mapping[str, Any], ref: object
) -> Mapping[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        _fail("scaffold_schema_remote_ref_rejected")
    current: object = root_schema
    for raw in ref[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            _fail("scaffold_schema_ref_invalid")
        current = current[key]
    return _mapping(current, "scaffold_schema_ref_invalid")


def _reject_remote_refs(value: object) -> None:
    if isinstance(value, Mapping):
        ref = value.get("$ref")
        if ref is not None and (not isinstance(ref, str) or not ref.startswith("#/")):
            _fail("scaffold_schema_remote_ref_rejected")
        for child in value.values():
            _reject_remote_refs(child)
    elif isinstance(value, list):
        for child in value:
            _reject_remote_refs(child)


def _require_draft202012_schema(schema: Mapping[str, Any]) -> None:
    if Draft202012Validator is None:
        _fail("scaffold_jsonschema_dependency_required")
    _reject_remote_refs(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise NaturalLanguageScaffoldConsumerError(
            "scaffold_published_schema_invalid"
        ) from exc


def _validate_with_draft202012(
    value: object, schema: Mapping[str, Any], code: str
) -> None:
    _require_draft202012_schema(schema)
    if next(Draft202012Validator(schema).iter_errors(value), None) is not None:
        _fail(code)


def _schema_accepts(
    value: object,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    path: str,
) -> bool:
    try:
        _validate_schema(value, schema, root_schema, path)
    except NaturalLanguageScaffoldConsumerError:
        return False
    return True


def _validate_schema(
    value: object,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate_schema(
            value, _resolve_local_ref(root_schema, schema["$ref"]), root_schema, path
        )
    for child in schema.get("allOf", []):
        _validate_schema(
            value, _mapping(child, "scaffold_schema_invalid"), root_schema, path
        )
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        accepted = sum(
            _schema_accepts(
                value,
                _mapping(child, "scaffold_schema_invalid"),
                root_schema,
                path,
            )
            for child in one_of
        )
        if accepted != 1:
            _fail("scaffold_schema_instance_invalid")
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and not any(
        _schema_accepts(
            value,
            _mapping(child, "scaffold_schema_invalid"),
            root_schema,
            path,
        )
        for child in any_of
    ):
        _fail("scaffold_schema_instance_invalid")
    negated = schema.get("not")
    if isinstance(negated, Mapping) and _schema_accepts(
        value, negated, root_schema, path
    ):
        _fail("scaffold_schema_instance_invalid")
    conditional = schema.get("if")
    if isinstance(conditional, Mapping) and _schema_accepts(
        value, conditional, root_schema, path
    ):
        consequence = schema.get("then")
        if isinstance(consequence, Mapping):
            _validate_schema(value, consequence, root_schema, path)

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _schema_type_matches(value, expected_type):
            _fail("scaffold_schema_instance_invalid")
    elif isinstance(expected_type, list) and not any(
        isinstance(item, str) and _schema_type_matches(value, item)
        for item in expected_type
    ):
        _fail("scaffold_schema_instance_invalid")
    if "const" in schema and value != schema["const"]:
        _fail("scaffold_schema_instance_invalid")
    if "enum" in schema and value not in schema["enum"]:
        _fail("scaffold_schema_instance_invalid")

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            _fail("scaffold_schema_instance_invalid")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            _fail("scaffold_schema_instance_invalid")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            _fail("scaffold_schema_instance_invalid")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            _fail("scaffold_schema_instance_invalid")
        if "maximum" in schema and value > schema["maximum"]:
            _fail("scaffold_schema_instance_invalid")

    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            _fail("scaffold_schema_instance_invalid")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            _fail("scaffold_schema_instance_invalid")
        if schema.get("uniqueItems") is True:
            encoded = [_canonical_bytes(item) for item in value]
            if len(encoded) != len(set(encoded)):
                _fail("scaffold_schema_instance_invalid")
        prefix = schema.get("prefixItems", [])
        if isinstance(prefix, list):
            for index, child in enumerate(prefix[: len(value)]):
                _validate_schema(
                    value[index],
                    _mapping(child, "scaffold_schema_invalid"),
                    root_schema,
                    f"{path}[{index}]",
                )
        items = schema.get("items")
        start = len(prefix) if isinstance(prefix, list) else 0
        if items is False and len(value) > start:
            _fail("scaffold_schema_instance_invalid")
        if isinstance(items, Mapping):
            for index in range(start, len(value)):
                _validate_schema(value[index], items, root_schema, f"{path}[{index}]")

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if isinstance(required, list) and any(key not in value for key in required):
            _fail("scaffold_schema_instance_invalid")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            _fail("scaffold_schema_invalid")
        for key, child in properties.items():
            if key in value:
                _validate_schema(
                    value[key],
                    _mapping(child, "scaffold_schema_invalid"),
                    root_schema,
                    f"{path}.{key}",
                )
        extras = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and extras:
            _fail("scaffold_schema_instance_invalid")
        if isinstance(additional, Mapping):
            for key in extras:
                _validate_schema(value[key], additional, root_schema, f"{path}.{key}")


def _jsonl_records(snapshot: _BytesSnapshot) -> tuple[Mapping[str, Any], ...]:
    if b"\r" in snapshot.data or not snapshot.data.endswith(b"\n"):
        _fail("scaffold_partition_serialization_invalid")
    records: list[Mapping[str, Any]] = []
    for raw_line in snapshot.data.splitlines():
        if not raw_line:
            _fail("scaffold_partition_serialization_invalid")
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NaturalLanguageScaffoldConsumerError(
                "scaffold_partition_json_invalid"
            ) from exc
        records.append(_mapping(value, "scaffold_partition_record_invalid"))
    return tuple(records)


def _assert_no_denied_token_fields(value: object) -> None:
    if isinstance(value, Mapping):
        if set(value) & _DENIED_TOKEN_FIELDS:
            _fail("scaffold_tokenizer_unbound_field_rejected")
        for child in value.values():
            _assert_no_denied_token_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_denied_token_fields(child)


def _pair_normal_form(record: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(
        {key: value for key, value in record.items() if key not in _PAIR_VARIANT_FIELDS}
    )


def _validate_record_hashes(record: Mapping[str, Any]) -> None:
    routing = _mapping(record.get("routing_json"), "scaffold_record_invalid")
    trigger = _mapping(record.get("expert_trigger"), "scaffold_record_invalid")
    alora = _mapping(record.get("alora_invocation"), "scaffold_record_invalid")
    tool_calls = record.get("tool_calls")
    tool_results = record.get("tool_results")
    if not isinstance(tool_calls, list) or not isinstance(tool_results, list):
        _fail("scaffold_record_invalid")
    payload = {
        "routing_json": routing,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "expert_trigger": trigger,
    }
    payload_text = _canonical_bytes(payload).decode("utf-8")
    if record.get("routing_json_sha256") != _sha256(_canonical_bytes(routing)):
        _fail("scaffold_record_hash_invalid")
    if record.get("canonical_json_payload_sha256") != _sha256(
        _canonical_bytes(payload)
    ):
        _fail("scaffold_record_hash_invalid")
    scaffold_text = record.get("scaffold_text")
    if not isinstance(scaffold_text, str) or record.get(
        "scaffold_text_sha256"
    ) != _sha256(scaffold_text.encode("utf-8")):
        _fail("scaffold_record_hash_invalid")
    variant = record.get("scaffold_variant")
    if variant == "json_only":
        if scaffold_text != payload_text or "concise_rationale_summary" in record:
            _fail("scaffold_record_variant_invalid")
    elif variant == "concise_rationale_plus_json":
        rationale = record.get("concise_rationale_summary")
        if (
            not isinstance(rationale, str)
            or not rationale
            or scaffold_text != rationale + "\n" + payload_text
        ):
            _fail("scaffold_record_variant_invalid")
    else:
        _fail("scaffold_record_variant_invalid")
    trigger_text = trigger.get("trigger_text")
    if (
        not isinstance(trigger_text, str)
        or trigger.get("trigger_text_sha256") != _sha256(trigger_text.encode("utf-8"))
        or alora.get("trigger_text") != trigger_text
        or alora.get("trigger_text_sha256") != trigger.get("trigger_text_sha256")
    ):
        _fail("scaffold_record_trigger_invalid")
    call_ids = [item.get("call_id") for item in tool_calls if isinstance(item, Mapping)]
    result_ids = [
        item.get("call_id") for item in tool_results if isinstance(item, Mapping)
    ]
    if len(call_ids) != len(tool_calls) or call_ids != result_ids:
        _fail("scaffold_record_tool_trace_invalid")
    pair_identity = {
        key: record[key]
        for key in (
            "task_bundle_sha256",
            "task_id_sha256",
            "source_gold_sha256",
            "split",
            "source_variant",
            "stage",
            "expert",
            "target_binding_sha256",
            "allowed_evidence_sha256",
            "forbidden_evidence_sha256",
            "segment_plan_sha256",
        )
    }
    pair_id = "natural-language-scaffold-pair-v1:" + _sha256(
        _canonical_bytes(pair_identity)
    )
    if record.get("pair_id") != pair_id:
        _fail("scaffold_record_pair_hash_invalid")
    record_identity = {
        "pair_id": pair_id,
        "scaffold_variant": variant,
        "scaffold_text_sha256": record["scaffold_text_sha256"],
    }
    expected_record_id = "natural-language-scaffold-v1:" + _sha256(
        _canonical_bytes(record_identity)
    )
    if record.get("record_id") != expected_record_id:
        _fail("scaffold_record_id_hash_invalid")


def _validate_fixture_groups(records: Sequence[Mapping[str, Any]]) -> None:
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    bundles: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        pairs.setdefault(str(record["pair_id"]), []).append(record)
        bundles.setdefault(str(record["task_bundle_sha256"]), []).append(record)
    if len(records) != 20 or len(pairs) != 10 or len(bundles) != 2:
        _fail("scaffold_fixture_count_invalid")
    for rows in pairs.values():
        if (
            len(rows) != 2
            or {str(row["scaffold_variant"]) for row in rows} != PAIR_VARIANTS
            or _pair_normal_form(rows[0]) != _pair_normal_form(rows[1])
        ):
            _fail("scaffold_pair_invariant_invalid")
    for rows in bundles.values():
        if (
            len(rows) != 10
            or len({str(row["split"]) for row in rows}) != 1
            or {str(row["expert"]) for row in rows} != REQUIRED_ROLES
            or {str(row["stage"]) for row in rows} != REQUIRED_STAGES
            or {str(row["scaffold_variant"]) for row in rows} != PAIR_VARIANTS
        ):
            _fail("scaffold_bundle_split_or_role_invalid")


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    producer = _mapping(manifest.get("producer"), "scaffold_manifest_invalid")
    expected_producer = {
        "config_sha256": CONFIG_SHA256,
        "implementation_sha256": PRODUCER_IMPLEMENTATION_SHA256,
        "record_schema_sha256": RECORD_SCHEMA_SHA256,
        "manifest_schema_sha256": MANIFEST_SCHEMA_SHA256,
        "smoke_contract_schema_sha256": SMOKE_SCHEMA_SHA256,
        "smoke_contract_sha256": SMOKE_CONTRACT_SHA256,
    }
    if any(producer.get(key) != value for key, value in expected_producer.items()):
        _fail("scaffold_manifest_producer_binding_invalid")
    counts = _mapping(manifest.get("counts"), "scaffold_manifest_invalid")
    if (
        counts.get("total") != 20
        or counts.get("pairs") != 10
        or counts.get("unique_task_bundles") != 2
        or counts.get("allowed_segment_references") != 122
        or counts.get("unique_allowed_segments") != 25
    ):
        _fail("scaffold_manifest_count_invalid")
    if any(
        manifest.get(key) != 0
        for key in (
            "provider_requests",
            "model_loads",
            "gpu_requests",
            "network_requests",
        )
    ):
        _fail("scaffold_manifest_nonzero_resource_invalid")
    if (
        manifest.get("quality_validated") is not False
        or manifest.get("execution_authorized") is not False
        or manifest.get("claim_scope") != "synthetic_fixture_contract_only"
    ):
        _fail("scaffold_manifest_claim_invalid")


def _load_schema(
    repo_root: Path, relative: str, expected_sha256: str
) -> tuple[Mapping[str, Any], _BytesSnapshot]:
    snapshot = _read_bytes_snapshot(
        repo_root / relative, "scaffold_contract_file_unavailable"
    )
    if snapshot.sha256 != expected_sha256:
        _fail("scaffold_contract_file_hash_invalid")
    schema = _mapping(
        _json_value(snapshot, "scaffold_schema_invalid"), "scaffold_schema_invalid"
    )
    _require_draft202012_schema(schema)
    return schema, snapshot


def validate_scaffold_record(
    record: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> None:
    """Validate one detached record against the authenticated closed schema."""

    repository = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    schema, snapshot = _load_schema(
        repository,
        "configs/research/swebench_natural_language_scaffold_sidecar.schema.json",
        RECORD_SCHEMA_SHA256,
    )
    _validate_schema(record, schema, schema, "record")
    _validate_with_draft202012(record, schema, "scaffold_schema_instance_invalid")
    _assert_no_denied_token_fields(record)
    _validate_record_hashes(record)
    _verify_snapshot_current(snapshot, "scaffold_authenticated_snapshot_changed")


def validate_two_request_gate(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate Request-1 semantics and keep Request 2 explicitly ineligible."""

    route = _mapping(record.get("route_boundary"), "scaffold_route_invalid")
    cache = _mapping(record.get("cache_metadata"), "scaffold_cache_invalid")
    alora = _mapping(record.get("alora_invocation"), "scaffold_alora_invalid")
    expected_route = {
        "semantics": "explicit_two_request_commit_boundary",
        "validation_required": True,
        "commit_required": True,
        "commit_promotes_text_only": True,
        "planner_private_tail_kv_transfer_allowed": False,
        "committed_scaffold_reencode_required": True,
        "committed_scaffold_reencode_producer": "frozen_base",
        "committed_scaffold_reencode_adapter_state": "off",
        "expert_request_requires_committed_scaffold_as_input": True,
        "token_boundary_status": "tokenizer_binding_required",
    }
    if any(route.get(key) != value for key, value in expected_route.items()):
        _fail("scaffold_route_invalid")
    expected_cache = {
        "adapter_state_on_prefix": "off",
        "adapter_state_after_boundary": "expert_only",
        "private_tail_scope": "expert_private_delta",
        "private_tail_kv_required": True,
        "full_generation_kv_shared_claimed": False,
        "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
        "cache_identity_status": "identity_unbound",
        "exact_cache_reuse_enabled": False,
        "reuse_savings_tokens": 0,
        "planner_private_tail_kv_reused_by_expert": False,
        "physical_kv_tensor_emitted": False,
        "committed_scaffold_reencode_executed": False,
        "downstream_immutable_segment_emitted": False,
    }
    if any(cache.get(key) != value for key, value in expected_cache.items()):
        _fail("scaffold_cache_invalid")
    expected_alora = {
        "capability_binding_status": "unbound",
        "activation_semantics": "next_request_input_activation_only",
        "invocation_scan_scope": "new_request_input_tokens_only",
        "same_request_activation_allowed": False,
        "mid_request_generated_activation_allowed": False,
        "mid_request_generated_trigger_switch_claimed": False,
        "explicit_commit_required": True,
        "tokenizer_binding_status": "unbound",
        "adapter_available": False,
        "adapter_loaded": False,
        "activation_executed": False,
        "cross_attention_q_reader_claimed": False,
        "physical_shared_kv_claimed": False,
    }
    if any(alora.get(key) != value for key, value in expected_alora.items()):
        _fail("scaffold_alora_invalid")
    if (
        tuple(record.get("adapter_control_labels", ()))
        != (
            "q_only",
            "q_plus_o",
            "wide_lora",
        )
        or record.get("training_outcome_claimed") is not False
    ):
        _fail("scaffold_adapter_control_invalid")
    return {
        "schema_version": "anchor.natural-language-scaffold-two-request-gate.v1",
        "request1_candidate": True,
        "request2_eligible": False,
        "planner_private_kv_transfer": False,
        "adapter_activation": "next_request_only_after_future_attestation",
        "training_authorized": False,
        "remaining_gates": (
            "tokenizer_identity",
            "adapter_artifact_attestation",
            "committed_scaffold_reencode_receipt",
            "frozen_formal_v3_release_lock",
        ),
    }


def planner_scaffold_from_record(bound: BoundScaffold) -> PlannerScaffold:
    """Build an unarmed runtime candidate from an authenticated bound receipt."""

    record, sidecar = _validated_bound_scaffold(bound)
    _assert_scaffold_target_body_exclusion(record, sidecar)
    gate = validate_two_request_gate(record)
    if gate["request2_eligible"] is not False:
        _fail("scaffold_request2_must_remain_ineligible")
    trigger = _mapping(record.get("expert_trigger"), "scaffold_record_invalid")
    return PlannerScaffold(
        task_bundle_sha256=str(record["task_bundle_sha256"]),
        target_expert_id=str(record["expert"]),
        natural_language_scaffold=str(record["scaffold_text"]),
        trigger_text=str(trigger["trigger_text"]),
        trigger_text_sha256=str(trigger["trigger_text_sha256"]),
        structured_plan={
            "routing_json": record["routing_json"],
            "tool_calls": record["tool_calls"],
            "tool_results": record["tool_results"],
        },
    )


def load_natural_language_scaffold_fixture(
    artifact_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    consumer_config_path: str | Path | None = None,
    expected_consumer_config_sha256: str = CONSUMER_CONFIG_SHA256,
    expected_manifest_sha256: str | None = None,
) -> ScaffoldFixture:
    """Authenticate and validate the frozen scaffold fixture from one-byte snapshots."""

    root = Path(artifact_root).resolve()
    repository = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    consumer_config, consumer_config_snapshot = _load_consumer_config_snapshot(
        repository,
        consumer_config_path,
        expected_consumer_config_sha256,
    )
    producer_contract = _mapping(
        consumer_config.get("producer_contract"), "scaffold_consumer_config_invalid"
    )
    configured_manifest_sha256 = str(producer_contract["fixture_manifest_sha256"])
    if expected_manifest_sha256 is None:
        expected_manifest_sha256 = configured_manifest_sha256
    if (
        not _SHA256_RE.fullmatch(expected_manifest_sha256)
        or expected_manifest_sha256 != configured_manifest_sha256
    ):
        _fail("scaffold_expected_manifest_hash_invalid")

    config = _read_bytes_snapshot(
        repository / "configs/research/swebench_natural_language_scaffold_v1.yaml",
        "scaffold_contract_file_unavailable",
    )
    if config.sha256 != CONFIG_SHA256:
        _fail("scaffold_contract_file_hash_invalid")
    record_schema, record_schema_snapshot = _load_schema(
        repository,
        "configs/research/swebench_natural_language_scaffold_sidecar.schema.json",
        RECORD_SCHEMA_SHA256,
    )
    manifest_schema, manifest_schema_snapshot = _load_schema(
        repository,
        "configs/research/swebench_natural_language_scaffold_manifest.schema.json",
        MANIFEST_SCHEMA_SHA256,
    )
    smoke_schema, smoke_schema_snapshot = _load_schema(
        repository,
        "configs/research/swebench_natural_language_scaffold_smoke_contract.schema.json",
        SMOKE_SCHEMA_SHA256,
    )
    smoke = _read_bytes_snapshot(
        repository
        / "configs/research/swebench_natural_language_scaffold_smoke_v1.yaml",
        "scaffold_contract_file_unavailable",
    )
    if smoke.sha256 != SMOKE_CONTRACT_SHA256:
        _fail("scaffold_contract_file_hash_invalid")
    try:
        smoke_value = yaml.safe_load(smoke.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise NaturalLanguageScaffoldConsumerError("scaffold_smoke_invalid") from exc
    _validate_schema(smoke_value, smoke_schema, smoke_schema, "smoke")
    _validate_with_draft202012(
        smoke_value, smoke_schema, "scaffold_smoke_schema_instance_invalid"
    )

    manifest_snapshot = _read_bytes_snapshot(
        root / "manifest.json", "scaffold_manifest_unavailable"
    )
    sidecar_snapshot = _read_bytes_snapshot(
        root / "manifest.json.sha256", "scaffold_manifest_sha256_sidecar_required"
    )
    expected_sidecar = f"{manifest_snapshot.sha256}  manifest.json\n".encode("ascii")
    if sidecar_snapshot.data != expected_sidecar:
        _fail("scaffold_manifest_sha256_sidecar_invalid")
    if manifest_snapshot.sha256 != expected_manifest_sha256:
        _fail("scaffold_manifest_hash_invalid")
    if sidecar_snapshot.sha256 != FIXTURE_MANIFEST_SIDECAR_SHA256:
        _fail("scaffold_manifest_sha256_sidecar_hash_invalid")
    manifest = _mapping(
        _json_value(manifest_snapshot, "scaffold_manifest_invalid"),
        "scaffold_manifest_invalid",
    )
    _validate_schema(manifest, manifest_schema, manifest_schema, "manifest")
    _validate_with_draft202012(
        manifest, manifest_schema, "scaffold_manifest_schema_instance_invalid"
    )
    _validate_manifest_contract(manifest)

    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or len(declared_files) != len(FIXED_FILES):
        _fail("scaffold_manifest_files_invalid")
    all_records: list[Mapping[str, Any]] = []
    partition_snapshots: list[_BytesSnapshot] = []
    for declared, fixed in zip(declared_files, FIXED_FILES, strict=True):
        relative, split, source_variant, scaffold_variant, count, size, digest = fixed
        item = _mapping(declared, "scaffold_manifest_files_invalid")
        expected = {
            "path": relative,
            "split": split,
            "source_variant": source_variant,
            "scaffold_variant": scaffold_variant,
            "records": count,
            "bytes": size,
            "sha256": digest,
        }
        if dict(item) != expected:
            _fail("scaffold_manifest_files_invalid")
        partition = _read_bytes_snapshot(
            _resolve_artifact_path(root, relative), "scaffold_partition_unavailable"
        )
        partition_snapshots.append(partition)
        if partition.sha256 != digest or len(partition.data) != size:
            _fail("scaffold_partition_hash_invalid")
        records = _jsonl_records(partition)
        if len(records) != count:
            _fail("scaffold_partition_count_invalid")
        for record in records:
            _validate_schema(record, record_schema, record_schema, "record")
            _validate_with_draft202012(
                record, record_schema, "scaffold_schema_instance_invalid"
            )
            _assert_no_denied_token_fields(record)
            _validate_record_hashes(record)
            validate_two_request_gate(record)
            if (
                record.get("split") != split
                or record.get("source_variant") != source_variant
                or record.get("scaffold_variant") != scaffold_variant
            ):
                _fail("scaffold_partition_binding_invalid")
        all_records.extend(records)

    _validate_fixture_groups(all_records)
    summary = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "smoke_schema_version": SMOKE_SCHEMA_VERSION,
        "consumer_config_sha256": consumer_config_snapshot.sha256,
        "manifest_sha256": manifest_snapshot.sha256,
        "manifest_sha256_sidecar_sha256": sidecar_snapshot.sha256,
        "records": len(all_records),
        "pairs": len({str(row["pair_id"]) for row in all_records}),
        "task_bundles": len({str(row["task_bundle_sha256"]) for row in all_records}),
        "training_authorized": False,
        "quality_validated": False,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
    }
    for snapshot in (
        consumer_config_snapshot,
        config,
        record_schema_snapshot,
        manifest_schema_snapshot,
        smoke_schema_snapshot,
        smoke,
        manifest_snapshot,
        sidecar_snapshot,
        *partition_snapshots,
    ):
        _verify_snapshot_current(snapshot, "scaffold_authenticated_snapshot_changed")
    scaffold_partition_sha256 = {
        (split, scaffold_variant): digest
        for (
            _relative,
            split,
            _source_variant,
            scaffold_variant,
            _count,
            _size,
            digest,
        ) in FIXED_FILES
    }
    frozen_records = tuple(
        _mint_authenticated_record(
            record,
            manifest_sha256=manifest_snapshot.sha256,
            scaffold_partition_sha256=scaffold_partition_sha256[
                (str(record["split"]), str(record["scaffold_variant"]))
            ],
        )
        for record in all_records
    )
    frozen_manifest = _mapping(
        _freeze_json(manifest), "scaffold_authenticated_snapshot_invalid"
    )
    frozen_summary = _mapping(
        _freeze_json(summary), "scaffold_authenticated_snapshot_invalid"
    )
    return ScaffoldFixture(frozen_records, frozen_manifest, frozen_summary)


def build_contract_ablation_view(
    authenticated_record: AuthenticatedScaffoldRecord,
) -> Mapping[str, Any]:
    """Create a target-leak-free, non-authorizing paired-ablation view."""

    record = _authenticated_record_data(authenticated_record)
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        _fail("scaffold_record_version_invalid")
    routing = _mapping(record.get("routing_json"), "scaffold_record_invalid")
    source = {
        key: record[key]
        for key in (
            "pair_id",
            "task_bundle_sha256",
            "task_id_sha256",
            "source_gold_sha256",
            "split",
            "source_variant",
            "stage",
            "expert",
            "language",
            "source_partition_sha256",
            "source_line_sha256",
            "allowed_evidence_sha256",
            "forbidden_evidence_sha256",
            "segment_plan_sha256",
            "ordered_segment_ids_sha256",
            "terminal_prefix_lineage_sha256",
        )
    }
    source["allowed_segment_refs"] = routing["allowed_segment_refs"]
    user_content = _canonical_bytes(source).decode("utf-8")
    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "record_id": record["record_id"],
        "pair_id": record["pair_id"],
        "task_bundle_sha256": record["task_bundle_sha256"],
        "split": record["split"],
        "stage": record["stage"],
        "expert": record["expert"],
        "language": record["language"],
        "scaffold_variant": record["scaffold_variant"],
        "input_sha256": _sha256(user_content.encode("utf-8")),
        "target_sha256": record["scaffold_text_sha256"],
        "messages": (
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["scaffold_text"]},
        ),
        "training_authorized": False,
        "claim_scope": "synthetic_fixture_contract_only",
    }


def _taskboard_line_sha256(sidecar: TaskBoardSidecar) -> str:
    return _sha256(_canonical_bytes(canonical_taskboard_sidecar(sidecar)))


def _taskboard_target_answer(sidecar: TaskBoardSidecar) -> str:
    try:
        target = json.loads(sidecar.training_record.target_output)
    except json.JSONDecodeError as exc:  # pragma: no cover - upstream typed loader gate
        raise NaturalLanguageScaffoldConsumerError(
            "scaffold_taskboard_target_invalid"
        ) from exc
    mapped = _mapping(target, "scaffold_taskboard_target_invalid")
    answer = mapped.get("answer")
    if not isinstance(answer, str) or not answer:
        _fail("scaffold_taskboard_target_invalid")
    return answer


def _bound_binding_sha256(record_sha256: str, sidecar_sha256: str) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "authenticated_record_sha256": record_sha256,
                "taskboard_sidecar_sha256": sidecar_sha256,
            }
        )
    )


def _mint_bound_scaffold(
    record: AuthenticatedScaffoldRecord,
    sidecar: TaskBoardSidecar,
) -> BoundScaffold:
    record_data = _authenticated_record_data(record)
    record_sha256 = _sha256(_canonical_bytes(record_data))
    sidecar_sha256 = _taskboard_line_sha256(sidecar)
    return BoundScaffold(
        record=record,
        sidecar=sidecar,
        record_sha256=record_sha256,
        sidecar_sha256=sidecar_sha256,
        binding_sha256=_bound_binding_sha256(record_sha256, sidecar_sha256),
        _capability=_BOUND_SCAFFOLD_CAPABILITY,
    )


def _validated_bound_scaffold(
    value: object,
) -> tuple[Mapping[str, Any], TaskBoardSidecar]:
    if not isinstance(value, BoundScaffold):
        _fail("scaffold_bound_receipt_required")
    if value._capability is not _BOUND_SCAFFOLD_CAPABILITY:
        _fail("scaffold_bound_receipt_invalid")
    record = _authenticated_record_data(value.record)
    if not isinstance(value.sidecar, TaskBoardSidecar):
        _fail("scaffold_bound_receipt_invalid")
    record_sha256 = _sha256(_canonical_bytes(record))
    sidecar_sha256 = _taskboard_line_sha256(value.sidecar)
    if (
        value.record_sha256 != record_sha256
        or value.sidecar_sha256 != sidecar_sha256
        or value.binding_sha256 != _bound_binding_sha256(record_sha256, sidecar_sha256)
    ):
        _fail("scaffold_bound_receipt_hash_invalid")
    return record, value.sidecar


def _assert_scaffold_target_body_exclusion(
    record: Mapping[str, Any],
    sidecar: TaskBoardSidecar,
) -> None:
    scaffold_text = record.get("scaffold_text")
    if not isinstance(scaffold_text, str):
        _fail("scaffold_record_invalid")
    current_target = _taskboard_target_answer(sidecar)
    if current_target and current_target in scaffold_text:
        _fail("scaffold_target_body_leak")
    forbidden = set(sidecar.training_record.targets.forbidden)
    for block in sidecar.training_record.blocks:
        if (
            block.block_id in forbidden
            and block.content
            and block.content in scaffold_text
        ):
            _fail("scaffold_forbidden_body_leak")


def bind_scaffolds_to_taskboard(
    fixture: ScaffoldFixture,
    taskboard_root: str | Path,
) -> tuple[BoundScaffold, ...]:
    """Cross-bind scaffold metadata to the authenticated TaskBoard fixture.

    The raw source line is reconstructed with the canonical serializer used by
    the producer.  This lets the consumer bind ``source_line_sha256`` without
    reopening a partition after its authenticated TaskBoard loader has parsed
    it.
    """

    if not isinstance(fixture, ScaffoldFixture):
        _fail("scaffold_authenticated_fixture_required")
    input_binding = _mapping(
        fixture.manifest.get("input"), "scaffold_manifest_input_invalid"
    )
    sidecars, _manifest, summary = load_taskboard_sidecar_dataset(
        taskboard_root,
        expected_config_sha256=str(input_binding["projector_config_sha256"]),
        expected_sidecar_schema_sha256=str(
            input_binding["projector_sidecar_schema_sha256"]
        ),
        expected_manifest_schema_sha256=str(
            input_binding["projector_manifest_schema_sha256"]
        ),
        expected_segment_plan_schema_sha256=str(
            input_binding["segment_plan_schema_sha256"]
        ),
    )
    if summary.get("manifest_sha256") != input_binding.get("projector_manifest_sha256"):
        _fail("scaffold_taskboard_manifest_binding_invalid")
    authenticated = summary.get("authenticated_file_sha256")
    if not isinstance(authenticated, Mapping):
        _fail("scaffold_taskboard_partition_binding_invalid")
    selected_paths = input_binding.get("selected_source_partitions")
    if (
        not isinstance(selected_paths, Sequence)
        or isinstance(selected_paths, (str, bytes, bytearray))
        or any(not isinstance(path, str) for path in selected_paths)
        or any(authenticated.get(path) is None for path in selected_paths)
    ):
        _fail("scaffold_taskboard_partition_binding_invalid")

    by_line_sha: dict[str, TaskBoardSidecar] = {}
    for sidecar in sidecars:
        digest = _taskboard_line_sha256(sidecar)
        if digest in by_line_sha:
            _fail("scaffold_taskboard_line_not_unique")
        by_line_sha[digest] = sidecar

    selected_hashes = {str(authenticated[path]) for path in selected_paths}
    bound: list[BoundScaffold] = []
    for authenticated_record in fixture.records:
        record = _authenticated_record_data(authenticated_record)
        line_sha = str(record["source_line_sha256"])
        sidecar = by_line_sha.get(line_sha)
        if sidecar is None:
            _fail("scaffold_taskboard_line_binding_invalid")
        task_id_sha = _sha256(sidecar.training_record.task_id.encode("utf-8"))
        plan = sidecar.segment_plan
        segment_plan_sha = _sha256(_canonical_bytes(plan))
        target_sha = _sha256(_taskboard_target_answer(sidecar).encode("utf-8"))
        raw_segments = plan.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            _fail("scaffold_taskboard_segment_binding_invalid")
        expected_refs = [
            {
                key: segment[key]
                for key in (
                    "segment_id",
                    "source_block_id",
                    "content_sha256",
                    "causal_order",
                    "cache_scope",
                )
            }
            for segment in raw_segments
            if isinstance(segment, Mapping)
        ]
        if len(expected_refs) != len(raw_segments):
            _fail("scaffold_taskboard_segment_binding_invalid")
        routing = _mapping(record.get("routing_json"), "scaffold_record_invalid")
        if _canonical_bytes(routing.get("allowed_segment_refs")) != _canonical_bytes(
            expected_refs
        ):
            _fail("scaffold_taskboard_segment_binding_invalid")
        ordered_segment_ids_sha = _sha256(
            _canonical_bytes([item["segment_id"] for item in expected_refs])
        )
        allowed_evidence_sha = _sha256(_canonical_bytes(expected_refs))
        forbidden_evidence_sha = _sha256(
            _canonical_bytes(
                {
                    "source_line_sha256": line_sha,
                    "ordered_forbidden_block_ids": list(
                        sidecar.training_record.targets.forbidden
                    ),
                }
            )
        )
        target_binding_sha = _sha256(
            _canonical_bytes(
                {
                    "source_gold_sha256": sidecar.source_gold_sha256,
                    "target_sha256": target_sha,
                    "stage": sidecar.stage,
                    "expert": sidecar.expert,
                }
            )
        )
        expected_bindings = {
            "task_bundle_sha256": sidecar.task_bundle_sha256,
            "task_id_sha256": task_id_sha,
            "source_gold_sha256": sidecar.source_gold_sha256,
            "split": sidecar.split,
            "source_variant": sidecar.variant,
            "stage": sidecar.stage,
            "expert": sidecar.expert,
            "language": sidecar.training_record.language,
            "segment_plan_sha256": segment_plan_sha,
            "target_sha256": target_sha,
            "target_binding_sha256": target_binding_sha,
            "allowed_evidence_sha256": allowed_evidence_sha,
            "forbidden_evidence_sha256": forbidden_evidence_sha,
            "ordered_segment_ids_sha256": ordered_segment_ids_sha,
            "terminal_prefix_lineage_sha256": raw_segments[-1]["prefix_lineage_sha256"],
        }
        for key, value in expected_bindings.items():
            if record.get(key) != value:
                _fail(f"scaffold_taskboard_cross_binding_invalid:{key}")
        source_partition = f"{sidecar.split}/{sidecar.variant}.jsonl"
        if (
            source_partition not in selected_paths
            or str(record["source_partition_sha256"])
            != str(authenticated[source_partition])
            or str(record["source_partition_sha256"]) not in selected_hashes
        ):
            _fail("scaffold_taskboard_partition_binding_invalid")
        _assert_scaffold_target_body_exclusion(record, sidecar)
        bound.append(_mint_bound_scaffold(authenticated_record, sidecar))
    return tuple(bound)


def build_bound_scaffold_view(bound: BoundScaffold) -> Mapping[str, Any]:
    """Use the real hard-filtered TaskBoard prompt and scaffold-only target."""

    record, sidecar = _validated_bound_scaffold(bound)
    _assert_scaffold_target_body_exclusion(record, sidecar)
    filtered = build_training_view(sidecar.training_record)
    if filtered.role != record.get("expert"):
        _fail("scaffold_filtered_prompt_role_invalid")
    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "record_id": record["record_id"],
        "pair_id": record["pair_id"],
        "task_bundle_sha256": record["task_bundle_sha256"],
        "split": record["split"],
        "stage": record["stage"],
        "expert": record["expert"],
        "language": record["language"],
        "scaffold_variant": record["scaffold_variant"],
        "input_sha256": _sha256(filtered.prompt.encode("utf-8")),
        "target_sha256": record["scaffold_text_sha256"],
        "visible_block_ids": filtered.visible_block_ids,
        "messages": (
            {"role": "user", "content": filtered.prompt},
            {"role": "assistant", "content": record["scaffold_text"]},
        ),
        "request1_candidate_only": True,
        "request2_eligible": False,
        "training_authorized": False,
        "claim_scope": "synthetic_fixture_contract_only",
    }


def paired_ablation_summary(
    records: Sequence[AuthenticatedScaffoldRecord],
) -> Mapping[str, Any]:
    """Prove paired inputs and report content-free target identities."""

    authenticated = tuple(_authenticated_record_data(record) for record in records)
    _validate_fixture_groups(authenticated)
    pairs: dict[str, list[AuthenticatedScaffoldRecord]] = {}
    for record in records:
        pairs.setdefault(str(record["pair_id"]), []).append(record)
    target_pairs: list[Mapping[str, Any]] = []
    for pair_id, rows in sorted(pairs.items()):
        views = [build_contract_ablation_view(row) for row in rows]
        if len({str(view["input_sha256"]) for view in views}) != 1:
            _fail("scaffold_ablation_input_mismatch")
        if len({str(view["target_sha256"]) for view in views}) != 2:
            _fail("scaffold_ablation_target_not_distinct")
        target_pairs.append(
            {
                "pair_id": pair_id,
                "input_sha256": views[0]["input_sha256"],
                "target_sha256": sorted(str(view["target_sha256"]) for view in views),
            }
        )
    return {
        "schema_version": "anchor.natural-language-scaffold-ablation-summary.v1",
        "pairs": len(target_pairs),
        "paired_inputs_identical": True,
        "targets_variant_specific": True,
        "training_authorized": False,
        "items": tuple(target_pairs),
    }


def paired_bound_ablation_summary(
    views: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Prove pair equality over real ``build_training_view`` materializations."""

    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for view in views:
        pairs.setdefault(str(view.get("pair_id")), []).append(view)
    if len(views) != 20 or len(pairs) != 10:
        _fail("scaffold_bound_ablation_count_invalid")
    for rows in pairs.values():
        if (
            len(rows) != 2
            or {str(row.get("scaffold_variant")) for row in rows} != PAIR_VARIANTS
            or len({str(row.get("input_sha256")) for row in rows}) != 1
            or len({str(row.get("target_sha256")) for row in rows}) != 2
            or any(row.get("request1_candidate_only") is not True for row in rows)
            or any(row.get("request2_eligible") is not False for row in rows)
            or any(row.get("training_authorized") is not False for row in rows)
        ):
            _fail("scaffold_bound_ablation_invariant_invalid")
    return {
        "schema_version": "anchor.natural-language-scaffold-bound-ablation-summary.v1",
        "pairs": len(pairs),
        "paired_inputs_identical": True,
        "targets_variant_specific": True,
        "request1_candidate_only": True,
        "request2_eligible": False,
        "training_authorized": False,
    }


__all__ = [
    "AuthenticatedScaffoldRecord",
    "CONSUMER_CONFIG_SHA256",
    "CONFIG_SHA256",
    "FIXTURE_MANIFEST_SHA256",
    "FIXTURE_MANIFEST_SIDECAR_SHA256",
    "MANIFEST_SCHEMA_SHA256",
    "NaturalLanguageScaffoldConsumerError",
    "PRODUCER_IMPLEMENTATION_SHA256",
    "RECORD_SCHEMA_SHA256",
    "SMOKE_CONTRACT_SHA256",
    "SMOKE_SCHEMA_SHA256",
    "BoundScaffold",
    "ScaffoldFixture",
    "bind_scaffolds_to_taskboard",
    "build_bound_scaffold_view",
    "build_contract_ablation_view",
    "load_natural_language_scaffold_fixture",
    "paired_ablation_summary",
    "paired_bound_ablation_summary",
    "planner_scaffold_from_record",
    "validate_scaffold_record",
    "validate_two_request_gate",
]
