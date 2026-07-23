"""Streaming frozen-prefix Q-reader training-view materializer.

This additive v2 producer authenticates a TaskBoard projector v2 artifact and
an independently frozen, body-free bundle-profile artifact.  It never infers
task semantics, information-flow strata, or capability labels from protected
bodies.  The selected TaskBoard rows are processed one line at a time into a
private spool and published with create-once atomic rename semantics.

The current stage commit is present only in the assistant target.  Shared
prefix and request inputs are assembled exclusively from the already validated
segment plan; current/future/forbidden bodies are therefore filtered before
serialization.  Every public error is a stable body-free code.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, BinaryIO
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from anchor_mvp.swebench import taskboard_projector as _projector


CONFIG_SCHEMA = "anchor.frozen-prefix-qreader-training-view-config.v2"
PRODUCER_VERSION = "anchor.frozen-prefix-qreader-training-view-producer.v2"
RECORD_SCHEMA = "anchor.frozen-prefix-qreader-training-view.v2"
MANIFEST_SCHEMA = "anchor.frozen-prefix-qreader-training-view-manifest.v2"
BUNDLE_DESCRIPTOR_SCHEMA = "anchor.frozen-prefix-qreader-bundle-profile-descriptor.v2"
BUNDLE_RECORD_SCHEMA = "anchor.frozen-prefix-qreader-bundle-profile.v2"
BUNDLE_MANIFEST_SCHEMA = "anchor.frozen-prefix-qreader-bundle-profile-manifest.v2"
BUNDLE_PRODUCER_VERSION = "anchor.frozen-prefix-qreader-bundle-profile-producer.v2"

STAGE_TO_EXPERT = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "domain_builder": "frontend_gen",
    "domain_review": "frontend_review",
    "security": "security_gate",
}
EXPERTS = tuple(STAGE_TO_EXPERT.values())
STRATA = (
    "direct_resolution",
    "tool_mediated_search",
    "implementation_handoff",
    "review_revision",
    "security_constrained",
)
CAPABILITY_LABELS = (
    "general_chat",
    "knowledge_qa",
    "simple_tool_search",
    "micro_coding",
    "software_repair",
)
PROJECTOR_FILES = (
    ("train/clean.jsonl", "train", "clean"),
    ("train/noisy.jsonl", "train", "noisy"),
    ("calibration/clean.jsonl", "calibration", "clean"),
)
OUTPUT_FILES = (("train.jsonl", "train"), ("eval_proxy.jsonl", "eval_proxy"))
SCHEMA_FILENAMES = {
    "config": "swebench_natural_language_scaffold_v2_config.schema.json",
    "record": "swebench_natural_language_scaffold_v2_record.schema.json",
    "manifest": "swebench_natural_language_scaffold_v2_manifest.schema.json",
    "bundle_descriptor": (
        "swebench_natural_language_scaffold_v2_bundle_profile_descriptor.schema.json"
    ),
    "bundle_record": (
        "swebench_natural_language_scaffold_v2_bundle_profile.schema.json"
    ),
    "bundle_manifest": (
        "swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json"
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT = 0x400


class NaturalLanguageScaffoldV2Error(RuntimeError):
    """Stable content-free error safe for an operator log."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise NaturalLanguageScaffoldV2Error(code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("scaffold_v2_nonfinite_or_unserializable_value")


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _task_id_sha256(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON value")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    value: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in value
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "duplicate mapping key",
                key_node.start_mark,
            )
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _assert_plain_existing_path(path: Path, *, directory: bool, code: str) -> None:
    """Reject symlink/reparse components before any resolve operation."""

    absolute = path.absolute()
    parts = list(reversed((absolute, *absolute.parents)))
    for component in parts:
        if not component.exists() and not component.is_symlink():
            continue
        try:
            info = os.lstat(component)
        except OSError:
            _fail(code)
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            _fail(code)
    if directory:
        if not absolute.is_dir():
            _fail(code)
    elif not absolute.is_file():
        _fail(code)


def _assert_output_parent(path: Path, code: str) -> Path:
    raw_parent = path.absolute().parent
    _assert_plain_existing_path(raw_parent, directory=True, code=code)
    try:
        resolved = raw_parent.resolve(strict=True)
    except OSError:
        _fail(code)
    if resolved != raw_parent:
        _fail(code)
    return resolved


def _safe_relative(root: Path, relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(code)
    candidate_rel = Path(relative)
    if (
        candidate_rel.is_absolute()
        or any(part in {".", "..", ""} for part in candidate_rel.parts)
        or candidate_rel.as_posix() != relative
    ):
        _fail(code)
    candidate = root.joinpath(*candidate_rel.parts)
    _assert_plain_existing_path(candidate, directory=False, code=code)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail(code)
    if resolved != candidate.absolute():
        _fail(code)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(code)
    return resolved


@dataclass(frozen=True)
class BytesSnapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class StreamSeal:
    path: Path
    sha256: str
    size: int
    records: int
    identity: tuple[int, int, int, int]


def _read_small(path: Path, code: str, *, max_bytes: int = 4_000_000) -> BytesSnapshot:
    _assert_plain_existing_path(path, directory=False, code=code)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size < 1 or before.st_size > max_bytes:
                _fail(code)
            data = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
        terminal = path.stat()
    except OSError:
        _fail(code)
    identity = _stat_identity(before)
    if (
        len(data) != before.st_size
        or identity != _stat_identity(after)
        or identity != _stat_identity(terminal)
        or path.is_symlink()
        or _is_reparse(terminal)
    ):
        _fail(code)
    return BytesSnapshot(path, data, _sha256_bytes(data), len(data), identity)


def _json_snapshot(snapshot: BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = _strict_json_loads(snapshot.data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail(code)
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _yaml_snapshot(snapshot: BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = yaml.load(
            snapshot.data.decode("utf-8", errors="strict"),
            Loader=_UniqueKeySafeLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError):
        _fail(code)
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _validator(snapshot: BytesSnapshot, code: str) -> Draft202012Validator:
    schema = _json_snapshot(snapshot, code)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        _fail(code)
    return Draft202012Validator(schema)


def _validate(
    validator: Draft202012Validator, value: object, code: str
) -> Mapping[str, Any]:
    try:
        validator.validate(value)
    except ValidationError:
        _fail(code)
    if not isinstance(value, Mapping):
        _fail(code)
    return value


@dataclass(frozen=True)
class ScaffoldV2Config:
    path: Path
    sha256: str
    implementation_sha256: str
    schema_snapshots: Mapping[str, BytesSnapshot]
    raw: Mapping[str, Any]
    max_records: int
    max_input_bytes: int
    max_line_bytes: int
    max_string_bytes: int

    @classmethod
    def load(cls, value: str | Path) -> "ScaffoldV2Config":
        raw_path = Path(value).expanduser()
        _assert_plain_existing_path(
            raw_path, directory=False, code="scaffold_v2_config_invalid"
        )
        if raw_path.resolve(strict=True) != raw_path.absolute():
            _fail("scaffold_v2_config_invalid")
        snapshot = _read_small(raw_path, "scaffold_v2_config_invalid")
        raw = _yaml_snapshot(snapshot, "scaffold_v2_config_invalid")
        schema_snapshots = {
            name: _read_small(
                raw_path.parent / filename,
                f"scaffold_v2_{name}_schema_invalid",
            )
            for name, filename in SCHEMA_FILENAMES.items()
        }
        config_validator = _validator(
            schema_snapshots["config"], "scaffold_v2_config_schema_invalid"
        )
        _validate(config_validator, raw, "scaffold_v2_config_invalid")
        for name, item in schema_snapshots.items():
            _validator(item, f"scaffold_v2_{name}_schema_invalid")
        project_root = raw_path.parent.parents[1]
        implementation_path = (
            project_root
            / "src"
            / "anchor_mvp"
            / "swebench"
            / "natural_language_scaffold_v2.py"
        )
        implementation = _read_small(
            implementation_path,
            "scaffold_v2_implementation_invalid",
            max_bytes=4_000_000,
        )
        limits = raw.get("limits")
        if not isinstance(limits, Mapping):
            _fail("scaffold_v2_config_invalid")
        return cls(
            path=raw_path.absolute(),
            sha256=snapshot.sha256,
            implementation_sha256=implementation.sha256,
            schema_snapshots=schema_snapshots,
            raw=raw,
            max_records=int(limits["max_records_per_partition"]),
            max_input_bytes=int(limits["max_input_file_bytes"]),
            max_line_bytes=int(limits["max_line_bytes"]),
            max_string_bytes=int(limits["max_string_utf8_bytes"]),
        )


def _verify_small_unchanged(snapshot: BytesSnapshot, code: str) -> None:
    current = _read_small(snapshot.path, code, max_bytes=max(snapshot.size, 1))
    if (
        current.sha256 != snapshot.sha256
        or current.size != snapshot.size
        or current.identity != snapshot.identity
    ):
        _fail(code)


def _verify_stream_unchanged(seal: StreamSeal, code: str) -> None:
    digest = hashlib.sha256()
    size = 0
    records = 0
    _assert_plain_existing_path(seal.path, directory=False, code=code)
    try:
        with seal.path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for line in handle:
                digest.update(line)
                size += len(line)
                if line.strip():
                    records += 1
            after = os.fstat(handle.fileno())
        terminal = seal.path.stat()
    except OSError:
        _fail(code)
    if (
        _stat_identity(before) != seal.identity
        or _stat_identity(after) != seal.identity
        or _stat_identity(terminal) != seal.identity
        or digest.hexdigest() != seal.sha256
        or size != seal.size
        or records != seal.records
    ):
        _fail(code)


def _exact_sidecar(
    root: Path,
    manifest: BytesSnapshot,
    code: str,
) -> BytesSnapshot:
    sidecar = _read_small(root / "manifest.json.sha256", code, max_bytes=256)
    expected = f"{manifest.sha256}  manifest.json\n".encode("ascii")
    if sidecar.data != expected:
        _fail(code)
    return sidecar


@dataclass(frozen=True)
class ProjectorArtifact:
    root: Path
    manifest: Mapping[str, Any]
    manifest_snapshot: BytesSnapshot
    sidecar_snapshot: BytesSnapshot
    files: Mapping[tuple[str, str], Mapping[str, Any]]
    schema_snapshots: tuple[BytesSnapshot, ...]


def _load_projector(
    cfg: ScaffoldV2Config,
    value: str | Path,
    expected_manifest_sha256: str,
) -> ProjectorArtifact:
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        _fail("scaffold_v2_projector_manifest_expected_invalid")
    root_path = Path(value).expanduser()
    _assert_plain_existing_path(
        root_path, directory=True, code="scaffold_v2_projector_invalid"
    )
    root = root_path.absolute()
    if root.resolve(strict=True) != root:
        _fail("scaffold_v2_projector_invalid")
    manifest_snapshot = _read_small(
        root / "manifest.json", "scaffold_v2_projector_manifest_invalid"
    )
    if manifest_snapshot.sha256 != expected_manifest_sha256:
        _fail("scaffold_v2_projector_manifest_mismatch")
    sidecar = _exact_sidecar(
        root,
        manifest_snapshot,
        "scaffold_v2_projector_manifest_sidecar_invalid",
    )
    manifest = _json_snapshot(
        manifest_snapshot, "scaffold_v2_projector_manifest_invalid"
    )
    config_dir = cfg.path.parent
    projector_manifest_schema = _read_small(
        config_dir / "taskboard_projector_manifest.schema.json",
        "scaffold_v2_projector_manifest_schema_invalid",
    )
    projector_record_schema = _read_small(
        config_dir / "taskboard_projector_sidecar.schema.json",
        "scaffold_v2_projector_record_schema_invalid",
    )
    segment_schema = _read_small(
        config_dir / "hierarchical_task_kv_segment_plan.schema.json",
        "scaffold_v2_segment_plan_schema_invalid",
    )
    _validate(
        _validator(
            projector_manifest_schema,
            "scaffold_v2_projector_manifest_schema_invalid",
        ),
        manifest,
        "scaffold_v2_projector_manifest_invalid",
    )
    producer = manifest.get("producer")
    if not isinstance(producer, Mapping):
        _fail("scaffold_v2_projector_manifest_invalid")
    if (
        producer.get("manifest_schema_sha256") != projector_manifest_schema.sha256
        or producer.get("sidecar_schema_sha256") != projector_record_schema.sha256
        or producer.get("segment_plan_schema_sha256") != segment_schema.sha256
    ):
        _fail("scaffold_v2_projector_schema_binding_mismatch")
    files_raw = manifest.get("files")
    if not isinstance(files_raw, list):
        _fail("scaffold_v2_projector_manifest_invalid")
    files: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in files_raw:
        if not isinstance(item, Mapping):
            _fail("scaffold_v2_projector_manifest_invalid")
        key = (str(item.get("split")), str(item.get("variant")))
        files[key] = item
    expected = {(split, variant) for _, split, variant in PROJECTOR_FILES}
    if set(files) != expected:
        _fail("scaffold_v2_projector_manifest_invalid")
    return ProjectorArtifact(
        root,
        manifest,
        manifest_snapshot,
        sidecar,
        files,
        (projector_manifest_schema, projector_record_schema, segment_schema),
    )


@dataclass(frozen=True)
class BundleProfileArtifact:
    root: Path
    manifest: Mapping[str, Any]
    manifest_snapshot: BytesSnapshot
    sidecar_snapshot: BytesSnapshot
    file_binding: Mapping[str, Any]


def _load_bundle_profile_artifact(
    cfg: ScaffoldV2Config,
    value: str | Path,
    expected_manifest_sha256: str,
) -> BundleProfileArtifact:
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        _fail("scaffold_v2_bundle_manifest_expected_invalid")
    root_path = Path(value).expanduser()
    _assert_plain_existing_path(
        root_path, directory=True, code="scaffold_v2_bundle_artifact_invalid"
    )
    root = root_path.absolute()
    if root.resolve(strict=True) != root:
        _fail("scaffold_v2_bundle_artifact_invalid")
    snapshot = _read_small(
        root / "manifest.json", "scaffold_v2_bundle_manifest_invalid"
    )
    if snapshot.sha256 != expected_manifest_sha256:
        _fail("scaffold_v2_bundle_manifest_mismatch")
    sidecar = _exact_sidecar(
        root, snapshot, "scaffold_v2_bundle_manifest_sidecar_invalid"
    )
    manifest = _json_snapshot(snapshot, "scaffold_v2_bundle_manifest_invalid")
    _validate(
        _validator(
            cfg.schema_snapshots["bundle_manifest"],
            "scaffold_v2_bundle_manifest_schema_invalid",
        ),
        manifest,
        "scaffold_v2_bundle_manifest_invalid",
    )
    producer = manifest["producer"]
    if (
        producer["record_schema_sha256"] != cfg.schema_snapshots["bundle_record"].sha256
        or producer["manifest_schema_sha256"]
        != cfg.schema_snapshots["bundle_manifest"].sha256
        or producer["descriptor_schema_sha256"]
        != cfg.schema_snapshots["bundle_descriptor"].sha256
        or producer["config_sha256"] != cfg.sha256
        or producer["implementation_sha256"] != cfg.implementation_sha256
    ):
        _fail("scaffold_v2_bundle_schema_binding_mismatch")
    return BundleProfileArtifact(
        root, manifest, snapshot, sidecar, manifest["files"][0]
    )


def _stream_jsonl(
    path: Path,
    *,
    validator: Draft202012Validator,
    code: str,
    max_bytes: int,
    max_records: int,
    max_line_bytes: int,
    expected: Mapping[str, Any] | None,
    consume: Callable[[Mapping[str, Any], str], None],
) -> StreamSeal:
    """Hash, decode, validate, and consume each line without retaining a file."""

    _assert_plain_existing_path(path, directory=False, code=code)
    digest = hashlib.sha256()
    size = 0
    records = 0
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while True:
                line = handle.readline(max_line_bytes + 1)
                if not line:
                    break
                if len(line) > max_line_bytes:
                    _fail(code)
                digest.update(line)
                size += len(line)
                if size > max_bytes or not line.strip():
                    _fail(code)
                records += 1
                if records > max_records:
                    _fail(code)
                try:
                    value = _strict_json_loads(line.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    _fail(code)
                row = _validate(validator, value, code)
                consume(row, _sha256_bytes(line))
            after = os.fstat(handle.fileno())
        terminal = path.stat()
    except NaturalLanguageScaffoldV2Error:
        raise
    except OSError:
        _fail(code)
    identity = _stat_identity(before)
    observed_sha = digest.hexdigest()
    if (
        not records
        or identity != _stat_identity(after)
        or identity != _stat_identity(terminal)
        or path.is_symlink()
        or _is_reparse(terminal)
    ):
        _fail(code)
    if expected is not None and (
        expected.get("sha256") != observed_sha
        or expected.get("bytes") != size
        or expected.get("records") != records
    ):
        _fail(code)
    return StreamSeal(path, observed_sha, size, records, identity)


@dataclass(frozen=True)
class ProjectorRow:
    wrapper: Mapping[str, Any]
    inner: Mapping[str, Any]
    board: Mapping[str, Any]
    task_id: str
    language: str
    target_text: str
    target_sha256: str
    line_sha256: str


def _validated_projector_row(
    row: Mapping[str, Any],
    line_sha256: str,
    *,
    expected_split: str,
    expected_variant: str,
) -> ProjectorRow:
    inner = row.get("training_record")
    if not isinstance(inner, Mapping):
        _fail("scaffold_v2_projector_record_invalid")
    board = inner.get("task_board")
    targets = inner.get("attention_targets")
    target = inner.get("target")
    plan = row.get("segment_plan")
    if not all(isinstance(item, Mapping) for item in (board, targets, target, plan)):
        _fail("scaffold_v2_projector_record_invalid")
    blocks = board.get("blocks")
    relevant = targets.get("relevant_block_ids")
    distractors = targets.get("distractor_block_ids")
    forbidden = targets.get("forbidden_block_ids")
    if not all(
        isinstance(item, list) for item in (blocks, relevant, distractors, forbidden)
    ):
        _fail("scaffold_v2_projector_record_invalid")
    by_id: dict[str, Mapping[str, Any]] = {}
    for block in blocks:
        if not isinstance(block, Mapping) or not isinstance(block.get("id"), str):
            _fail("scaffold_v2_projector_record_invalid")
        if block["id"] in by_id:
            _fail("scaffold_v2_projector_record_invalid")
        by_id[str(block["id"])] = block
    try:
        _projector._validate_segment_plan(
            plan,
            wrapper=row,
            inner=inner,
            by_id=by_id,
            relevant=relevant,
            distractors=distractors,
            forbidden=forbidden,
        )
    except _projector.TaskBoardProjectorError:
        _fail("scaffold_v2_segment_plan_invalid")
    stage = row.get("stage")
    expert = row.get("expert")
    task_id = board.get("task_id")
    language = inner.get("language")
    target_text = target.get("answer")
    if (
        row.get("split") != expected_split
        or row.get("variant") != expected_variant
        or inner.get("split") != expected_split
        or inner.get("variant") != expected_variant
        or stage not in STAGE_TO_EXPERT
        or expert != STAGE_TO_EXPERT.get(str(stage))
        or inner.get("role") != expert
        or not isinstance(task_id, str)
        or not task_id
        or language not in {"en", "zh-CN"}
        or not isinstance(target_text, str)
        or not target_text
    ):
        _fail("scaffold_v2_projector_record_invalid")
    return ProjectorRow(
        row,
        inner,
        board,
        task_id,
        str(language),
        target_text,
        _sha256_bytes(target_text.encode("utf-8")),
        line_sha256,
    )


@dataclass
class BundleObservation:
    task_id: str
    language: str
    source_split: str
    roles: set[str]


def _observe_bundle(
    observations: dict[str, BundleObservation],
    row: ProjectorRow,
    source_split: str,
) -> None:
    bundle = str(row.wrapper["task_bundle_sha256"])
    expert = str(row.wrapper["expert"])
    current = observations.get(bundle)
    if current is None:
        observations[bundle] = BundleObservation(
            row.task_id, row.language, source_split, {expert}
        )
        return
    if (
        current.task_id != row.task_id
        or current.language != row.language
        or current.source_split != source_split
        or expert in current.roles
    ):
        _fail("scaffold_v2_bundle_cross_binding_invalid")
    current.roles.add(expert)


def _projector_partition_path(
    artifact: ProjectorArtifact, relative: str, split: str, variant: str
) -> tuple[Path, Mapping[str, Any]]:
    binding = artifact.files[(split, variant)]
    if binding.get("path") != relative:
        _fail("scaffold_v2_projector_manifest_invalid")
    return (
        _safe_relative(
            artifact.root, binding["path"], "scaffold_v2_projector_partition_invalid"
        ),
        binding,
    )


def _scan_projector_observations(
    cfg: ScaffoldV2Config,
    artifact: ProjectorArtifact,
) -> tuple[dict[str, BundleObservation], list[StreamSeal]]:
    validator = _validator(
        artifact.schema_snapshots[1], "scaffold_v2_projector_record_schema_invalid"
    )
    observations: dict[str, BundleObservation] = {}
    seals: list[StreamSeal] = []
    # Observe clean train and calibration only.  Noisy is authenticated and
    # cross-bound separately to avoid double-counting roles.
    for relative, split, variant in PROJECTOR_FILES:
        path, binding = _projector_partition_path(artifact, relative, split, variant)

        def consume(
            item: Mapping[str, Any],
            line_sha: str,
            *,
            _split: str = split,
            _variant: str = variant,
        ) -> None:
            parsed = _validated_projector_row(
                item,
                line_sha,
                expected_split=_split,
                expected_variant=_variant,
            )
            if _variant == "clean":
                _observe_bundle(observations, parsed, _split)

        seals.append(
            _stream_jsonl(
                path,
                validator=validator,
                code="scaffold_v2_projector_partition_invalid",
                max_bytes=cfg.max_input_bytes,
                max_records=cfg.max_records,
                max_line_bytes=cfg.max_line_bytes,
                expected=binding,
                consume=consume,
            )
        )
    if not observations or any(
        item.roles != set(EXPERTS) for item in observations.values()
    ):
        _fail("scaffold_v2_bundle_role_completeness_invalid")
    return observations, seals


def _descriptor_sidecar(path: Path, expected_sha256: str) -> BytesSnapshot:
    if not _SHA256_RE.fullmatch(expected_sha256):
        _fail("scaffold_v2_descriptor_expected_invalid")
    sidecar_path = path.with_name(path.name + ".sha256")
    snapshot = _read_small(
        sidecar_path, "scaffold_v2_descriptor_sidecar_invalid", max_bytes=256
    )
    expected = f"{expected_sha256}  {path.name}\n".encode("ascii")
    if snapshot.data != expected:
        _fail("scaffold_v2_descriptor_sidecar_invalid")
    return snapshot


def _publish_directory(
    temporary: Path,
    output: Path,
    parent_identity: tuple[int, int, int, int],
) -> None:
    if output.exists() or output.is_symlink():
        _fail("scaffold_v2_output_exists")
    parent = output.parent
    terminal = parent.stat()
    if _stat_identity(terminal) != parent_identity or _is_reparse(terminal):
        _fail("scaffold_v2_output_parent_changed")
    lock = parent / f".{output.name}.publish.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        if output.exists() or output.is_symlink():
            _fail("scaffold_v2_output_exists")
        # The exclusive sibling lock makes the rename create-once for all
        # conforming producers; os.rename also refuses an existing target on
        # the supported Windows publication host.
        os.rename(temporary, output)
    except FileExistsError:
        _fail("scaffold_v2_publish_lock_exists")
    except NaturalLanguageScaffoldV2Error:
        raise
    except OSError:
        _fail("scaffold_v2_atomic_publish_failed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                _fail("scaffold_v2_publish_lock_cleanup_failed")


def _cleanup_own_published_output(output: Path) -> None:
    """Remove only the exact plain directory created by this invocation."""

    try:
        info = os.lstat(output)
    except OSError:
        return
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or _is_reparse(info)
    ):
        _fail("scaffold_v2_post_publish_cleanup_unsafe")
    try:
        shutil.rmtree(output)
    except OSError:
        _fail("scaffold_v2_post_publish_cleanup_failed")


def _terminal_small_recheck(
    path: Path,
    expected: BytesSnapshot,
    code: str,
    *,
    max_bytes: int = 4_000_000,
) -> BytesSnapshot:
    current = _read_small(path, code, max_bytes=max_bytes)
    if (
        current.sha256 != expected.sha256
        or current.size != expected.size
        or current.identity != expected.identity
    ):
        _fail(code)
    return current


def _terminal_stream_recheck(
    path: Path,
    expected: StreamSeal,
    *,
    validator: Draft202012Validator,
    code: str,
    max_bytes: int,
    max_records: int,
    max_line_bytes: int,
    consume: Callable[[Mapping[str, Any], str], None],
) -> StreamSeal:
    current = _stream_jsonl(
        path,
        validator=validator,
        code=code,
        max_bytes=max_bytes,
        max_records=max_records,
        max_line_bytes=max_line_bytes,
        expected={
            "sha256": expected.sha256,
            "bytes": expected.size,
            "records": expected.records,
        },
        consume=consume,
    )
    if current.identity != expected.identity:
        _fail(code)
    return current


def _write_manifest(
    temporary: Path,
    value: Mapping[str, Any],
    validator: Draft202012Validator,
    code: str,
) -> tuple[BytesSnapshot, BytesSnapshot]:
    _validate(validator, value, code)
    try:
        data = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError):
        _fail(code)
    path = temporary / "manifest.json"
    path.write_bytes(data)
    sha = _sha256_bytes(data)
    sidecar_data = f"{sha}  manifest.json\n".encode("ascii")
    sidecar_path = temporary / "manifest.json.sha256"
    sidecar_path.write_bytes(sidecar_data)
    snapshot = _read_small(path, code)
    sidecar = _read_small(sidecar_path, code, max_bytes=256)
    if snapshot.sha256 != sha or sidecar.data != sidecar_data:
        _fail(code)
    return snapshot, sidecar


def freeze_bundle_profiles(
    config: ScaffoldV2Config | str | Path,
    projector_dir: str | Path,
    expected_projector_manifest_sha256: str,
    descriptor_jsonl: str | Path,
    expected_descriptor_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze explicit body-free bundle descriptors after projector cross-binding."""

    cfg = (
        config
        if isinstance(config, ScaffoldV2Config)
        else ScaffoldV2Config.load(config)
    )
    if isinstance(config, ScaffoldV2Config):
        current = ScaffoldV2Config.load(cfg.path)
        if current != cfg:
            _fail("scaffold_v2_config_changed")
    projector = _load_projector(cfg, projector_dir, expected_projector_manifest_sha256)
    observations, projector_seals = _scan_projector_observations(cfg, projector)
    descriptor_path = Path(descriptor_jsonl).expanduser()
    _assert_plain_existing_path(
        descriptor_path, directory=False, code="scaffold_v2_descriptor_invalid"
    )
    if descriptor_path.resolve(strict=True) != descriptor_path.absolute():
        _fail("scaffold_v2_descriptor_invalid")
    descriptor_sidecar = _descriptor_sidecar(
        descriptor_path, expected_descriptor_sha256
    )
    descriptors: dict[str, dict[str, Any]] = {}
    descriptor_validator = _validator(
        cfg.schema_snapshots["bundle_descriptor"],
        "scaffold_v2_bundle_descriptor_schema_invalid",
    )
    task_ids: set[str] = set()
    semantic_ids: set[str] = set()

    def consume_descriptor(row: Mapping[str, Any], line_sha: str) -> None:
        bundle = str(row["task_bundle_sha256"])
        if bundle in descriptors:
            _fail("scaffold_v2_descriptor_duplicate_bundle")
        observation = observations.get(bundle)
        if observation is None:
            _fail("scaffold_v2_descriptor_cross_binding_invalid")
        output_split = "train" if observation.source_split == "train" else "eval_proxy"
        if (
            row["task_id"] != observation.task_id
            or row["task_id_sha256"] != _task_id_sha256(observation.task_id)
            or row["language"] != observation.language
            or row["source_split"] != observation.source_split
            or row["output_split"] != output_split
            or row["information_flow_stratum"] not in STRATA
            or any(item not in CAPABILITY_LABELS for item in row["capability_labels"])
            or list(row["capability_labels"])
            != sorted(
                row["capability_labels"],
                key=CAPABILITY_LABELS.index,
            )
        ):
            _fail("scaffold_v2_descriptor_cross_binding_invalid")
        if (
            str(row["task_id"]) in task_ids
            or str(row["task_semantic_sha256"]) in semantic_ids
        ):
            _fail("scaffold_v2_descriptor_semantic_identity_invalid")
        task_ids.add(str(row["task_id"]))
        semantic_ids.add(str(row["task_semantic_sha256"]))
        output = dict(row)
        output["schema_version"] = BUNDLE_RECORD_SCHEMA
        descriptors[bundle] = output
        del line_sha

    descriptor_seal = _stream_jsonl(
        descriptor_path.absolute(),
        validator=descriptor_validator,
        code="scaffold_v2_descriptor_invalid",
        max_bytes=cfg.max_input_bytes,
        max_records=cfg.max_records,
        max_line_bytes=cfg.max_line_bytes,
        expected={
            "sha256": expected_descriptor_sha256,
            "bytes": descriptor_path.stat().st_size,
            # Records are checked against observations below.
            "records": len(observations),
        },
        consume=consume_descriptor,
    )
    if set(descriptors) != set(observations):
        _fail("scaffold_v2_descriptor_inventory_incomplete")

    raw_output = Path(output_dir).expanduser()
    parent = _assert_output_parent(raw_output, "scaffold_v2_output_parent_invalid")
    output = parent / raw_output.name
    if output.exists() or output.is_symlink():
        _fail("scaffold_v2_output_exists")
    try:
        output.relative_to(projector.root)
    except ValueError:
        pass
    else:
        _fail("scaffold_v2_output_overlap")
    temporary = parent / f".{output.name}.tmp-{uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        _fail("scaffold_v2_temporary_conflict")
    try:
        temporary.mkdir(mode=0o700)
        parent_identity = _stat_identity(parent.stat())
        partition = temporary / "bundle_profiles.jsonl"
        digest = hashlib.sha256()
        total_bytes = 0
        with partition.open("xb") as handle:
            for bundle in sorted(descriptors):
                line = _canonical_bytes(descriptors[bundle]) + b"\n"
                handle.write(line)
                digest.update(line)
                total_bytes += len(line)
        partition_seal = _stream_jsonl(
            partition,
            validator=_validator(
                cfg.schema_snapshots["bundle_record"],
                "scaffold_v2_bundle_record_schema_invalid",
            ),
            code="scaffold_v2_bundle_output_invalid",
            max_bytes=cfg.max_input_bytes,
            max_records=cfg.max_records,
            max_line_bytes=cfg.max_line_bytes,
            expected={
                "sha256": digest.hexdigest(),
                "bytes": total_bytes,
                "records": len(descriptors),
            },
            consume=lambda _row, _line_sha: None,
        )
        by_source = Counter(item["source_split"] for item in descriptors.values())
        by_output = Counter(item["output_split"] for item in descriptors.values())
        by_language = Counter(item["language"] for item in descriptors.values())
        by_capability = Counter(
            label
            for item in descriptors.values()
            for label in item["capability_labels"]
        )
        capability_inventory_sha = _sha256_value(
            [
                {
                    "task_bundle_sha256": bundle,
                    "capability_labels": descriptors[bundle]["capability_labels"],
                }
                for bundle in sorted(descriptors)
            ]
        )
        manifest = {
            "schema_version": BUNDLE_MANIFEST_SCHEMA,
            "status": "metadata_only_ready",
            "producer": {
                "name": "anchor.frozen-prefix-qreader-bundle-profile",
                "version": BUNDLE_PRODUCER_VERSION,
                "record_schema_version": BUNDLE_RECORD_SCHEMA,
                "record_schema_sha256": cfg.schema_snapshots["bundle_record"].sha256,
                "manifest_schema_sha256": cfg.schema_snapshots[
                    "bundle_manifest"
                ].sha256,
                "descriptor_schema_sha256": cfg.schema_snapshots[
                    "bundle_descriptor"
                ].sha256,
                "config_sha256": cfg.sha256,
                "implementation_sha256": cfg.implementation_sha256,
            },
            "files": [
                {
                    "path": "bundle_profiles.jsonl",
                    "sha256": partition_seal.sha256,
                    "bytes": partition_seal.size,
                    "records": partition_seal.records,
                }
            ],
            "counts": {
                "bundles": len(descriptors),
                "by_source_split": {
                    "train": by_source["train"],
                    "calibration": by_source["calibration"],
                },
                "by_output_split": {
                    "train": by_output["train"],
                    "eval_proxy": by_output["eval_proxy"],
                },
                "by_language": {
                    "en": by_language["en"],
                    "zh-CN": by_language["zh-CN"],
                },
                "by_capability_label": {
                    key: by_capability[key] for key in CAPABILITY_LABELS
                },
            },
            "source": {
                "projector_manifest_sha256": projector.manifest_snapshot.sha256,
                "descriptor_sha256": descriptor_seal.sha256,
                "descriptor_sidecar_sha256": descriptor_sidecar.sha256,
                "descriptor_records": descriptor_seal.records,
            },
            "capability_inventory_sha256": capability_inventory_sha,
            "body_free": True,
            "eval_proxy_is_heldout": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "claim_scope": "research_proxy_metadata_only",
            "manifest_sha256_sidecar_required": True,
        }
        manifest_snapshot, manifest_sidecar = _write_manifest(
            temporary,
            manifest,
            _validator(
                cfg.schema_snapshots["bundle_manifest"],
                "scaffold_v2_bundle_manifest_schema_invalid",
            ),
            "scaffold_v2_bundle_output_manifest_invalid",
        )
        for seal in projector_seals:
            _verify_stream_unchanged(seal, "scaffold_v2_projector_changed_during_read")
        _verify_stream_unchanged(
            descriptor_seal, "scaffold_v2_descriptor_changed_during_read"
        )
        _verify_stream_unchanged(partition_seal, "scaffold_v2_bundle_output_changed")
        for item in (
            projector.manifest_snapshot,
            projector.sidecar_snapshot,
            descriptor_sidecar,
            *projector.schema_snapshots,
            *cfg.schema_snapshots.values(),
            manifest_snapshot,
            manifest_sidecar,
        ):
            _verify_small_unchanged(item, "scaffold_v2_bound_file_changed")
        _publish_directory(temporary, output, parent_identity)
        try:
            _terminal_stream_recheck(
                output / "bundle_profiles.jsonl",
                partition_seal,
                validator=_validator(
                    cfg.schema_snapshots["bundle_record"],
                    "scaffold_v2_bundle_record_schema_invalid",
                ),
                code="scaffold_v2_bundle_post_publish_invalid",
                max_bytes=cfg.max_input_bytes,
                max_records=cfg.max_records,
                max_line_bytes=cfg.max_line_bytes,
                consume=lambda _row, _line_sha: None,
            )
            _terminal_small_recheck(
                output / "manifest.json",
                manifest_snapshot,
                "scaffold_v2_bundle_post_publish_invalid",
            )
            _terminal_small_recheck(
                output / "manifest.json.sha256",
                manifest_sidecar,
                "scaffold_v2_bundle_post_publish_invalid",
                max_bytes=256,
            )
            for seal in projector_seals:
                _verify_stream_unchanged(
                    seal, "scaffold_v2_projector_changed_after_publish"
                )
            _verify_stream_unchanged(
                descriptor_seal, "scaffold_v2_descriptor_changed_after_publish"
            )
            for item in (
                projector.manifest_snapshot,
                projector.sidecar_snapshot,
                descriptor_sidecar,
                *projector.schema_snapshots,
                *cfg.schema_snapshots.values(),
            ):
                _verify_small_unchanged(
                    item, "scaffold_v2_bound_file_changed_after_publish"
                )
            if ScaffoldV2Config.load(cfg.path) != cfg:
                _fail("scaffold_v2_config_changed_after_publish")
        except Exception:
            _cleanup_own_published_output(output)
            raise
        return {
            "output_dir": str(output),
            "manifest_sha256": manifest_snapshot.sha256,
            "records": len(descriptors),
            "capability_inventory_sha256": capability_inventory_sha,
            "provider_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "network_requests": 0,
            "training_authorized": False,
            "formal_training_authorized": False,
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _load_bundle_profiles(
    cfg: ScaffoldV2Config,
    artifact: BundleProfileArtifact,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], StreamSeal]:
    profiles: dict[str, Mapping[str, Any]] = {}
    line_shas: dict[str, str] = {}
    task_ids: set[str] = set()
    semantic_ids: set[str] = set()
    validator = _validator(
        cfg.schema_snapshots["bundle_record"],
        "scaffold_v2_bundle_record_schema_invalid",
    )
    path = _safe_relative(
        artifact.root,
        artifact.file_binding["path"],
        "scaffold_v2_bundle_partition_invalid",
    )

    def consume(row: Mapping[str, Any], line_sha: str) -> None:
        bundle = str(row["task_bundle_sha256"])
        if bundle in profiles:
            _fail("scaffold_v2_bundle_duplicate")
        if (
            row["task_id_sha256"] != _task_id_sha256(str(row["task_id"]))
            or row["information_flow_stratum"] not in STRATA
            or any(item not in CAPABILITY_LABELS for item in row["capability_labels"])
            or list(row["capability_labels"])
            != sorted(
                row["capability_labels"],
                key=CAPABILITY_LABELS.index,
            )
        ):
            _fail("scaffold_v2_bundle_record_invalid")
        if (
            str(row["task_id"]) in task_ids
            or str(row["task_semantic_sha256"]) in semantic_ids
        ):
            _fail("scaffold_v2_bundle_semantic_identity_invalid")
        task_ids.add(str(row["task_id"]))
        semantic_ids.add(str(row["task_semantic_sha256"]))
        profiles[bundle] = row
        line_shas[bundle] = line_sha

    seal = _stream_jsonl(
        path,
        validator=validator,
        code="scaffold_v2_bundle_partition_invalid",
        max_bytes=cfg.max_input_bytes,
        max_records=cfg.max_records,
        max_line_bytes=cfg.max_line_bytes,
        expected=artifact.file_binding,
        consume=consume,
    )
    if len(profiles) != artifact.manifest["counts"]["bundles"]:
        _fail("scaffold_v2_bundle_count_invalid")
    counts = artifact.manifest["counts"]
    by_source = Counter(str(row["source_split"]) for row in profiles.values())
    by_output = Counter(str(row["output_split"]) for row in profiles.values())
    by_language = Counter(str(row["language"]) for row in profiles.values())
    by_capability = Counter(
        str(label) for row in profiles.values() for label in row["capability_labels"]
    )
    if (
        counts["by_source_split"]
        != {"train": by_source["train"], "calibration": by_source["calibration"]}
        or counts["by_output_split"]
        != {"train": by_output["train"], "eval_proxy": by_output["eval_proxy"]}
        or counts["by_language"]
        != {"en": by_language["en"], "zh-CN": by_language["zh-CN"]}
        or counts["by_capability_label"]
        != {key: by_capability[key] for key in CAPABILITY_LABELS}
    ):
        _fail("scaffold_v2_bundle_count_invalid")
    observed_inventory = _sha256_value(
        [
            {
                "task_bundle_sha256": bundle,
                "capability_labels": profiles[bundle]["capability_labels"],
            }
            for bundle in sorted(profiles)
        ]
    )
    if observed_inventory != artifact.manifest["capability_inventory_sha256"]:
        _fail("scaffold_v2_capability_inventory_invalid")
    return profiles, line_shas, seal


def audit_bundle_profiles(
    config: ScaffoldV2Config | str | Path,
    artifact_dir: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate a published body-free bundle-profile artifact."""

    cfg = (
        config
        if isinstance(config, ScaffoldV2Config)
        else ScaffoldV2Config.load(config)
    )
    if isinstance(config, ScaffoldV2Config) and ScaffoldV2Config.load(cfg.path) != cfg:
        _fail("scaffold_v2_config_changed")
    artifact = _load_bundle_profile_artifact(
        cfg, artifact_dir, expected_manifest_sha256
    )
    profiles, _line_shas, seal = _load_bundle_profiles(cfg, artifact)
    _verify_stream_unchanged(seal, "scaffold_v2_bundle_audit_source_changed")
    _verify_small_unchanged(
        artifact.manifest_snapshot, "scaffold_v2_bundle_audit_source_changed"
    )
    _verify_small_unchanged(
        artifact.sidecar_snapshot, "scaffold_v2_bundle_audit_source_changed"
    )
    return {
        "status": "authenticated_metadata_only",
        "manifest_sha256": artifact.manifest_snapshot.sha256,
        "records": len(profiles),
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "training_authorized": False,
        "formal_training_authorized": False,
    }


@dataclass
class OutputWriter:
    path: Path
    handle: BinaryIO
    digest: Any
    size: int = 0
    records: int = 0

    @classmethod
    def open(cls, path: Path) -> "OutputWriter":
        return cls(path, path.open("xb"), hashlib.sha256())

    def write(self, row: Mapping[str, Any]) -> None:
        line = _canonical_bytes(row) + b"\n"
        self.handle.write(line)
        self.digest.update(line)
        self.size += len(line)
        self.records += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _segment_ref(segment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "source_block_id": segment["source_block_id"],
        "content_sha256": segment["content_sha256"],
        "serialization_order": segment["serialization_order"],
        "causal_order": segment["causal_order"],
        "cache_scope": segment["cache_scope"],
    }


def _render_training_record(
    cfg: ScaffoldV2Config,
    source: ProjectorRow,
    profile: Mapping[str, Any],
    profile_line_sha256: str,
    *,
    projector_manifest_sha256: str,
    projector_partition_sha256: str,
    bundle_manifest_sha256: str,
    bundle_partition_sha256: str,
) -> dict[str, Any]:
    row = source.wrapper
    inner = source.inner
    plan = row["segment_plan"]
    blocks = source.board["blocks"]
    by_id = {str(item["id"]): item for item in blocks}
    forbidden = set(
        str(item) for item in inner["attention_targets"]["forbidden_block_ids"]
    )
    shared_segments: list[dict[str, Any]] = []
    private_segments: list[dict[str, Any]] = []
    allowed_refs: list[dict[str, Any]] = []
    evidence_refs: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    for segment in plan["segments"]:
        source_id = str(segment["source_block_id"])
        if source_id in forbidden:
            _fail("scaffold_v2_forbidden_body_selected")
        block = by_id.get(source_id)
        if not isinstance(block, Mapping):
            _fail("scaffold_v2_segment_plan_invalid")
        text = block.get("content")
        if (
            not isinstance(text, str)
            or not text
            or len(text.encode("utf-8")) > cfg.max_string_bytes
            or _sha256_bytes(text.encode("utf-8")) != segment["content_sha256"]
        ):
            _fail("scaffold_v2_segment_plan_invalid")
        materialized = {**_segment_ref(segment), "text": text}
        allowed_refs.append(_segment_ref(segment))
        ordered_ids.append(str(segment["segment_id"]))
        if segment["cache_scope"] == "expert_private_delta":
            private_segments.append(materialized)
        else:
            shared_segments.append(materialized)
            evidence_refs.append(_segment_ref(segment))
    if not shared_segments or not evidence_refs:
        _fail("scaffold_v2_shared_prefix_invalid")
    shared_prefix_text = "\n".join(
        _canonical_bytes(
            {
                "segment_id": item["segment_id"],
                "source_block_id": item["source_block_id"],
                "text": item["text"],
            }
        ).decode("utf-8")
        for item in shared_segments
    )
    expert = str(row["expert"])
    stage = str(row["stage"])
    rationale = (
        "决策依据摘要：仅使用已认证、因果可见的前缀段，校验路由与验收条件后提交。"
        if source.language == "zh-CN"
        else (
            "Decision-basis summary: use only authenticated causally visible "
            "prefix segments, then validate the route and acceptance criteria."
        )
    )
    routing = {
        "role": stage,
        "expert": expert,
        "task_semantic_sha256": profile["task_semantic_sha256"],
        "problem_profile_sha256": profile["problem_profile_sha256"],
        "allowed_segment_refs": allowed_refs,
        "evidence_segment_refs": evidence_refs,
        "constraints": [
            "filter_visibility_before_serialization",
            "exclude_current_future_and_forbidden_bodies_from_input",
            "do_not_stringify_taskboard",
            "commit_text_before_next_request_activation",
            "keep_expert_tail_private_and_append_only",
        ],
        "acceptance_criteria": [
            "authenticated_source_bindings_match",
            "all_references_follow_ordered_prefix_lineage",
            "current_stage_commit_appears_only_in_assistant_target",
            "no_cross_expert_private_kv_transfer",
        ],
    }
    routing_sha256 = _sha256_value(routing)
    scaffold_text = rationale + "\n" + _canonical_bytes(routing).decode("utf-8")
    trigger_text = f"<|anchor_expert:{expert}|>"
    private_text = "\n".join(item["text"] for item in private_segments)
    request2_parts = [shared_prefix_text, scaffold_text, trigger_text]
    if private_text:
        request2_parts.append(private_text)
    request2_input = "\n".join(request2_parts)
    stage_commit = {
        "schema_version": "anchor.authenticated-stage-commit.v2",
        "stage": stage,
        "expert": expert,
        "source_gold_sha256": row["source_gold_sha256"],
        "target_sha256": source.target_sha256,
        "text": source.target_text,
    }
    tool_trace: dict[str, Any]
    if expert == "frontend_gen":
        forbidden_order = [
            str(item) for item in inner["attention_targets"]["forbidden_block_ids"]
        ]
        if (
            not forbidden_order
            or forbidden_order[0] not in by_id
            or by_id[forbidden_order[0]].get("kind") != "code"
        ):
            _fail("scaffold_v2_builder_evidence_invalid")
        events: list[dict[str, Any]] = []
        for source_id in forbidden_order[1:]:
            block = by_id.get(source_id)
            if not isinstance(block, Mapping):
                _fail("scaffold_v2_builder_evidence_invalid")
            kind = block.get("kind")
            if kind not in {"tool_call", "tool_result", "test_result"}:
                break
            text = block.get("content")
            if (
                not isinstance(text, str)
                or not text
                or len(text.encode("utf-8")) > cfg.max_string_bytes
                or block.get("commit_state") != "committed"
            ):
                _fail("scaffold_v2_builder_evidence_invalid")
            events.append(
                {
                    "source_block_id": source_id,
                    "kind": kind,
                    "content_sha256": _sha256_bytes(text.encode("utf-8")),
                    "text": text,
                }
            )
        tool_trace = {
            "status": "authenticated_source" if events else "unavailable",
            "source": "authenticated_taskboard",
            "binding_sha256": str(row["source_gold_sha256"]),
            "evidence_sha256": _sha256_value(events) if events else None,
            "events": events,
        }
    else:
        tool_trace = {
            "status": "not_applicable",
            "source": "not_applicable",
            "binding_sha256": None,
            "evidence_sha256": None,
            "events": [],
        }
    assistant_envelope = {
        "stage_commit": stage_commit,
        "tool_trace": tool_trace,
    }
    assistant_target = (
        scaffold_text + "\n" + _canonical_bytes(assistant_envelope).decode("utf-8")
    )
    # Current output and Builder evidence may only be serialized into the
    # assistant target; never into a prefix or request input.
    for forbidden_container in (
        shared_prefix_text,
        scaffold_text,
        request2_input,
        _canonical_bytes(allowed_refs).decode("utf-8"),
    ):
        if source.target_text in forbidden_container:
            _fail("scaffold_v2_current_target_input_leak")
    for event in tool_trace["events"]:
        if str(event["text"]) in request2_input:
            _fail("scaffold_v2_builder_evidence_input_leak")
    identity = {
        "task_bundle_sha256": row["task_bundle_sha256"],
        "task_semantic_sha256": profile["task_semantic_sha256"],
        "stage": stage,
        "expert": expert,
        "source_line_sha256": source.line_sha256,
        "target_sha256": source.target_sha256,
        "routing_json_sha256": routing_sha256,
        "builder_evidence_sha256": tool_trace["evidence_sha256"],
        "scaffold_variant": "concise_rationale_plus_json",
    }
    return {
        "schema_version": RECORD_SCHEMA,
        "record_id": "frozen-prefix-qreader-view-v2:" + _sha256_value(identity),
        "task_bundle_sha256": row["task_bundle_sha256"],
        "task_id": source.task_id,
        "task_id_sha256": _task_id_sha256(source.task_id),
        "task_semantic_sha256": profile["task_semantic_sha256"],
        "language": source.language,
        "information_flow_stratum": profile["information_flow_stratum"],
        "source_split": row["split"],
        "split": profile["output_split"],
        "source_variant": row["variant"],
        "stage": stage,
        "expert": expert,
        "problem_profile_sha256": profile["problem_profile_sha256"],
        "capability_labels": profile["capability_labels"],
        "scaffold_variant": "concise_rationale_plus_json",
        "pair_id": None,
        "source_binding": {
            "projector_manifest_sha256": projector_manifest_sha256,
            "projector_partition_sha256": projector_partition_sha256,
            "projector_line_sha256": source.line_sha256,
            "bundle_profile_manifest_sha256": bundle_manifest_sha256,
            "bundle_profile_partition_sha256": bundle_partition_sha256,
            "bundle_profile_line_sha256": profile_line_sha256,
            "source_gold_sha256": row["source_gold_sha256"],
            "segment_plan_sha256": _sha256_value(plan),
            "target_sha256": source.target_sha256,
            "ordered_segment_ids_sha256": _sha256_value(ordered_ids),
        },
        "training_view": {
            "shared_prefix_segments": shared_segments,
            "expert_private_input_segments": private_segments,
            "shared_prefix_text": shared_prefix_text,
            "concise_rationale_summary": rationale,
            "routing_json": routing,
            "routing_json_sha256": routing_sha256,
            "scaffold_text": scaffold_text,
            "expert_trigger": {
                "text": trigger_text,
                "sha256": _sha256_bytes(trigger_text.encode("utf-8")),
                "tokenizer_binding_status": "unbound",
            },
            "request2_input_text": request2_input,
            "assistant_target": {
                "serialization": (
                    "concise_rationale_then_canonical_routing_then_"
                    "canonical_stage_commit_and_tool_trace"
                ),
                "text": assistant_target,
                "text_sha256": _sha256_bytes(assistant_target.encode("utf-8")),
                "stage_commit": stage_commit,
                "tool_trace": tool_trace,
            },
        },
        "route_boundary": {
            "semantics": "explicit_two_request_commit_boundary",
            "validation_required": True,
            "commit_required": True,
            "commit_promotes_text_only": True,
            "committed_scaffold_reencode_required": True,
            "committed_scaffold_reencode_producer": "frozen_base",
            "committed_scaffold_reencode_adapter_state": "off",
            "expert_activation_request": "next_request",
            "token_boundary_status": "tokenizer_binding_required",
            "token_index_emitted": False,
        },
        "cache_contract": {
            "adapter_state_on_shared_prefix": "off",
            "adapter_state_after_boundary": "expert_only",
            "private_tail_kv_required": True,
            "private_tail_append_only": True,
            "private_tail_cross_expert_transfer_allowed": False,
            "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
            "full_generation_kv_shared_claimed": False,
            "naive_in_stack_q_lora_exact_reuse_claimed": False,
            "physical_kv_tensor_emitted": False,
        },
        "adapter_control": {
            "primary": "q_only",
            "diagnostic_overlays": ["o_only", "q_plus_o"],
            "wide_lora_inherited": False,
        },
        "claims": {
            "provider_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "network_requests": 0,
            "training_authorized": False,
            "formal_training_authorized": False,
            "eval_proxy_is_heldout": False,
        },
    }


def _pair_signature(row: ProjectorRow) -> dict[str, Any]:
    plan = row.wrapper["segment_plan"]
    segments = plan["segments"]
    return {
        "task_bundle_sha256": row.wrapper["task_bundle_sha256"],
        "task_id": row.task_id,
        "stage": row.wrapper["stage"],
        "expert": row.wrapper["expert"],
        "pair_id": row.wrapper["pair_id"],
        "source_gold_sha256": row.wrapper["source_gold_sha256"],
        "target_sha256": row.target_sha256,
        "relevant_sha256": _sha256_value(
            row.inner["attention_targets"]["relevant_block_ids"]
        ),
        "forbidden_sha256": _sha256_value(
            row.inner["attention_targets"]["forbidden_block_ids"]
        ),
        "base_segments": len(segments),
        "base_segments_sha256": _sha256_value(segments),
    }


def _validate_training_record_semantics(row: Mapping[str, Any]) -> None:
    """Recompute semantic hashes which JSON Schema cannot express."""

    training = row.get("training_view")
    binding = row.get("source_binding")
    if not isinstance(training, Mapping) or not isinstance(binding, Mapping):
        _fail("scaffold_v2_record_semantics_invalid")
    routing = training.get("routing_json")
    trigger = training.get("expert_trigger")
    assistant_target = training.get("assistant_target")
    if (
        not isinstance(routing, Mapping)
        or not isinstance(trigger, Mapping)
        or not isinstance(assistant_target, Mapping)
        or _sha256_value(routing) != training.get("routing_json_sha256")
        or routing.get("role") != row.get("stage")
        or routing.get("expert") != row.get("expert")
        or STAGE_TO_EXPERT.get(str(row.get("stage"))) != row.get("expert")
        or routing.get("task_semantic_sha256") != row.get("task_semantic_sha256")
        or routing.get("problem_profile_sha256") != row.get("problem_profile_sha256")
        or not isinstance(row.get("task_id"), str)
        or _task_id_sha256(str(row.get("task_id"))) != row.get("task_id_sha256")
        or _sha256_bytes(str(trigger.get("text", "")).encode("utf-8"))
        != trigger.get("sha256")
    ):
        _fail("scaffold_v2_record_semantics_invalid")
    try:
        assistant_text = assistant_target["text"]
        commit = assistant_target["stage_commit"]
        tool_trace = assistant_target["tool_trace"]
        commit_envelope = _strict_json_loads(str(assistant_text).splitlines()[-1])
    except (IndexError, KeyError, TypeError, json.JSONDecodeError, ValueError):
        _fail("scaffold_v2_record_semantics_invalid")
    if (
        not isinstance(commit, Mapping)
        or not isinstance(assistant_text, str)
        or assistant_target.get("serialization")
        != (
            "concise_rationale_then_canonical_routing_then_"
            "canonical_stage_commit_and_tool_trace"
        )
        or _sha256_bytes(assistant_text.encode("utf-8"))
        != assistant_target.get("text_sha256")
        or commit_envelope != {"stage_commit": dict(commit), "tool_trace": tool_trace}
        or assistant_text
        != str(training.get("scaffold_text"))
        + "\n"
        + _canonical_bytes(
            {"stage_commit": dict(commit), "tool_trace": tool_trace}
        ).decode("utf-8")
        or commit.get("schema_version") != "anchor.authenticated-stage-commit.v2"
        or commit.get("stage") != row.get("stage")
        or commit.get("expert") != row.get("expert")
        or commit.get("source_gold_sha256") != binding.get("source_gold_sha256")
        or commit.get("target_sha256") != binding.get("target_sha256")
        or not isinstance(commit.get("text"), str)
        or _sha256_bytes(str(commit["text"]).encode("utf-8"))
        != binding.get("target_sha256")
    ):
        _fail("scaffold_v2_record_semantics_invalid")
    if not isinstance(tool_trace, Mapping):
        _fail("scaffold_v2_record_semantics_invalid")
    events = tool_trace.get("events")
    if not isinstance(events, list):
        _fail("scaffold_v2_record_semantics_invalid")
    allowed_source_ids = {
        str(item.get("source_block_id"))
        for item in routing.get("allowed_segment_refs", [])
        if isinstance(item, Mapping)
    }
    if row.get("expert") == "frontend_gen":
        if (
            tool_trace.get("source") != "authenticated_taskboard"
            or tool_trace.get("binding_sha256") != binding.get("source_gold_sha256")
            or tool_trace.get("status")
            != ("authenticated_source" if events else "unavailable")
            or tool_trace.get("evidence_sha256")
            != (_sha256_value(events) if events else None)
        ):
            _fail("scaffold_v2_record_semantics_invalid")
        for event in events:
            if (
                not isinstance(event, Mapping)
                or event.get("source_block_id") in allowed_source_ids
                or event.get("kind") not in {"tool_call", "tool_result", "test_result"}
                or not isinstance(event.get("text"), str)
                or _sha256_bytes(str(event["text"]).encode("utf-8"))
                != event.get("content_sha256")
            ):
                _fail("scaffold_v2_record_semantics_invalid")
    elif dict(tool_trace) != {
        "status": "not_applicable",
        "source": "not_applicable",
        "binding_sha256": None,
        "evidence_sha256": None,
        "events": [],
    }:
        _fail("scaffold_v2_record_semantics_invalid")
    identity = {
        "task_bundle_sha256": row.get("task_bundle_sha256"),
        "task_semantic_sha256": row.get("task_semantic_sha256"),
        "stage": row.get("stage"),
        "expert": row.get("expert"),
        "source_line_sha256": binding.get("projector_line_sha256"),
        "target_sha256": binding.get("target_sha256"),
        "routing_json_sha256": training.get("routing_json_sha256"),
        "builder_evidence_sha256": tool_trace.get("evidence_sha256"),
        "scaffold_variant": "concise_rationale_plus_json",
    }
    if row.get("record_id") != (
        "frozen-prefix-qreader-view-v2:" + _sha256_value(identity)
    ):
        _fail("scaffold_v2_record_semantics_invalid")
    target_text = str(commit["text"])
    for key in ("shared_prefix_text", "scaffold_text", "request2_input_text"):
        value = training.get(key)
        if not isinstance(value, str) or target_text in value:
            _fail("scaffold_v2_record_semantics_invalid")
        if any(
            isinstance(event, Mapping)
            and isinstance(event.get("text"), str)
            and str(event["text"]) in value
            for event in events
        ):
            _fail("scaffold_v2_record_semantics_invalid")


def _verify_noisy_pair(row: ProjectorRow, clean: Mapping[str, Any]) -> None:
    segments = row.wrapper["segment_plan"]["segments"]
    comparable = {
        "task_bundle_sha256": row.wrapper["task_bundle_sha256"],
        "task_id": row.task_id,
        "stage": row.wrapper["stage"],
        "expert": row.wrapper["expert"],
        "pair_id": row.wrapper["pair_id"],
        "source_gold_sha256": row.wrapper["source_gold_sha256"],
        "target_sha256": row.target_sha256,
        "relevant_sha256": _sha256_value(
            row.inner["attention_targets"]["relevant_block_ids"]
        ),
        "forbidden_sha256": _sha256_value(
            row.inner["attention_targets"]["forbidden_block_ids"]
        ),
    }
    if (
        any(comparable[key] != clean[key] for key in comparable)
        or len(segments) != int(clean["base_segments"]) + 1
        or _sha256_value(segments[: int(clean["base_segments"])])
        != clean["base_segments_sha256"]
        or segments[-1]["cache_scope"] != "expert_private_delta"
    ):
        _fail("scaffold_v2_train_pair_invalid")


def materialize_training_view(
    config: ScaffoldV2Config | str | Path,
    projector_dir: str | Path,
    expected_projector_manifest_sha256: str,
    bundle_profile_dir: str | Path,
    expected_bundle_profile_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Stream an authenticated projector into a deterministic training view."""

    cfg = (
        config
        if isinstance(config, ScaffoldV2Config)
        else ScaffoldV2Config.load(config)
    )
    if isinstance(config, ScaffoldV2Config):
        current = ScaffoldV2Config.load(cfg.path)
        if current != cfg:
            _fail("scaffold_v2_config_changed")
    projector = _load_projector(cfg, projector_dir, expected_projector_manifest_sha256)
    bundle_artifact = _load_bundle_profile_artifact(
        cfg, bundle_profile_dir, expected_bundle_profile_manifest_sha256
    )
    if (
        bundle_artifact.manifest["source"]["projector_manifest_sha256"]
        != projector.manifest_snapshot.sha256
    ):
        _fail("scaffold_v2_bundle_projector_binding_mismatch")
    profiles, profile_line_shas, profile_seal = _load_bundle_profiles(
        cfg, bundle_artifact
    )
    raw_output = Path(output_dir).expanduser()
    parent = _assert_output_parent(raw_output, "scaffold_v2_output_parent_invalid")
    output = parent / raw_output.name
    if output.exists() or output.is_symlink():
        _fail("scaffold_v2_output_exists")
    for source_root in (projector.root, bundle_artifact.root):
        if output == source_root:
            _fail("scaffold_v2_output_overlap")
        try:
            output.relative_to(source_root)
        except ValueError:
            pass
        else:
            _fail("scaffold_v2_output_overlap")
    temporary = parent / f".{output.name}.tmp-{uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        _fail("scaffold_v2_temporary_conflict")
    projector_validator = _validator(
        projector.schema_snapshots[1],
        "scaffold_v2_projector_record_schema_invalid",
    )
    record_validator = _validator(
        cfg.schema_snapshots["record"], "scaffold_v2_record_schema_invalid"
    )
    source_seals: list[StreamSeal] = []
    writers: dict[str, OutputWriter] = {}
    counts_by_role: Counter[str] = Counter()
    counts_by_language: Counter[str] = Counter()
    counts_by_stratum: Counter[str] = Counter()
    counts_by_split: Counter[str] = Counter()
    output_groups: dict[str, set[str]] = {}
    clean_signatures: dict[str, Mapping[str, Any]] = {}
    used_profiles: set[str] = set()
    try:
        temporary.mkdir(mode=0o700)
        parent_identity = _stat_identity(parent.stat())
        for relative, split in OUTPUT_FILES:
            writers[split] = OutputWriter.open(temporary / relative)

        # Authentication-only clean train pass.
        clean_path, clean_binding = _projector_partition_path(
            projector, "train/clean.jsonl", "train", "clean"
        )

        def consume_clean(item: Mapping[str, Any], line_sha: str) -> None:
            parsed = _validated_projector_row(
                item, line_sha, expected_split="train", expected_variant="clean"
            )
            pair_id = str(item["pair_id"])
            if pair_id in clean_signatures:
                _fail("scaffold_v2_train_pair_invalid")
            clean_signatures[pair_id] = _pair_signature(parsed)

        source_seals.append(
            _stream_jsonl(
                clean_path,
                validator=projector_validator,
                code="scaffold_v2_projector_partition_invalid",
                max_bytes=cfg.max_input_bytes,
                max_records=cfg.max_records,
                max_line_bytes=cfg.max_line_bytes,
                expected=clean_binding,
                consume=consume_clean,
            )
        )

        def selected_pass(
            relative: str,
            source_split: str,
            source_variant: str,
            output_split: str,
        ) -> None:
            path, binding = _projector_partition_path(
                projector, relative, source_split, source_variant
            )

            def consume_selected(item: Mapping[str, Any], line_sha: str) -> None:
                parsed = _validated_projector_row(
                    item,
                    line_sha,
                    expected_split=source_split,
                    expected_variant=source_variant,
                )
                bundle = str(item["task_bundle_sha256"])
                profile = profiles.get(bundle)
                if profile is None:
                    _fail("scaffold_v2_bundle_profile_missing")
                if (
                    profile["task_id"] != parsed.task_id
                    or profile["task_id_sha256"] != _task_id_sha256(parsed.task_id)
                    or profile["language"] != parsed.language
                    or profile["source_split"] != source_split
                    or profile["output_split"] != output_split
                ):
                    _fail("scaffold_v2_bundle_cross_binding_invalid")
                if source_variant == "noisy":
                    clean = clean_signatures.pop(str(item["pair_id"]), None)
                    if clean is None:
                        _fail("scaffold_v2_train_pair_invalid")
                    _verify_noisy_pair(parsed, clean)
                rendered = _render_training_record(
                    cfg,
                    parsed,
                    profile,
                    profile_line_shas[bundle],
                    projector_manifest_sha256=projector.manifest_snapshot.sha256,
                    projector_partition_sha256=str(binding["sha256"]),
                    bundle_manifest_sha256=bundle_artifact.manifest_snapshot.sha256,
                    bundle_partition_sha256=profile_seal.sha256,
                )
                _validate(record_validator, rendered, "scaffold_v2_record_invalid")
                _validate_training_record_semantics(rendered)
                writers[output_split].write(rendered)
                used_profiles.add(bundle)
                output_groups.setdefault(bundle, set()).add(str(item["expert"]))
                counts_by_role[str(item["expert"])] += 1
                counts_by_language[parsed.language] += 1
                counts_by_stratum[str(profile["information_flow_stratum"])] += 1
                counts_by_split[output_split] += 1

            source_seals.append(
                _stream_jsonl(
                    path,
                    validator=projector_validator,
                    code="scaffold_v2_projector_partition_invalid",
                    max_bytes=cfg.max_input_bytes,
                    max_records=cfg.max_records,
                    max_line_bytes=cfg.max_line_bytes,
                    expected=binding,
                    consume=consume_selected,
                )
            )

        selected_pass("train/noisy.jsonl", "train", "noisy", "train")
        if clean_signatures:
            _fail("scaffold_v2_train_pair_invalid")
        selected_pass(
            "calibration/clean.jsonl",
            "calibration",
            "clean",
            "eval_proxy",
        )
        for writer in writers.values():
            writer.close()
        if (
            used_profiles != set(profiles)
            or any(roles != set(EXPERTS) for roles in output_groups.values())
            or len(output_groups) != len(profiles)
        ):
            _fail("scaffold_v2_bundle_role_completeness_invalid")

        output_seals: list[StreamSeal] = []
        output_files: list[dict[str, Any]] = []
        for relative, split in OUTPUT_FILES:
            writer = writers[split]
            seal = _stream_jsonl(
                writer.path,
                validator=record_validator,
                code="scaffold_v2_output_partition_invalid",
                max_bytes=cfg.max_input_bytes,
                max_records=cfg.max_records,
                max_line_bytes=cfg.max_line_bytes,
                expected={
                    "sha256": writer.digest.hexdigest(),
                    "bytes": writer.size,
                    "records": writer.records,
                },
                consume=lambda row, _line_sha: _validate_training_record_semantics(row),
            )
            output_seals.append(seal)
            output_files.append(
                {
                    "path": relative,
                    "sha256": seal.sha256,
                    "bytes": seal.size,
                    "records": seal.records,
                }
            )
        source_partitions = [
            {
                "path": relative,
                "sha256": projector.files[(split, variant)]["sha256"],
                "bytes": projector.files[(split, variant)]["bytes"],
                "records": projector.files[(split, variant)]["records"],
            }
            for relative, split, variant in PROJECTOR_FILES
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "materialized_research_proxy_only",
            "input": {
                "projector_manifest_sha256": projector.manifest_snapshot.sha256,
                "projector_manifest_sidecar_sha256": projector.sidecar_snapshot.sha256,
                "projector_manifest_schema_sha256": projector.schema_snapshots[
                    0
                ].sha256,
                "projector_record_schema_sha256": projector.schema_snapshots[1].sha256,
                "segment_plan_schema_sha256": projector.schema_snapshots[2].sha256,
                "bundle_profile_manifest_sha256": bundle_artifact.manifest_snapshot.sha256,
                "bundle_profile_manifest_sidecar_sha256": bundle_artifact.sidecar_snapshot.sha256,
                "bundle_profile_manifest_schema_sha256": cfg.schema_snapshots[
                    "bundle_manifest"
                ].sha256,
                "bundle_profile_record_schema_sha256": cfg.schema_snapshots[
                    "bundle_record"
                ].sha256,
                "source_partitions": source_partitions,
            },
            "producer": {
                "name": "anchor.frozen-prefix-qreader-training-view",
                "version": PRODUCER_VERSION,
                "config_sha256": cfg.sha256,
                "implementation_sha256": cfg.implementation_sha256,
                "record_schema_sha256": cfg.schema_snapshots["record"].sha256,
                "manifest_schema_sha256": cfg.schema_snapshots["manifest"].sha256,
            },
            "files": output_files,
            "counts": {
                "records": sum(counts_by_split.values()),
                "unique_task_bundles": len(profiles),
                "pair_count": 0,
                "by_split": {
                    "train": counts_by_split["train"],
                    "eval_proxy": counts_by_split["eval_proxy"],
                },
                "by_role": {key: counts_by_role[key] for key in EXPERTS},
                "by_language": {
                    "en": counts_by_language["en"],
                    "zh-CN": counts_by_language["zh-CN"],
                },
                "by_information_flow_stratum": {
                    key: counts_by_stratum[key] for key in STRATA
                },
            },
            "architecture_contract": {
                "semantics": "explicit_two_request_commit_boundary",
                "adapter_state_on_shared_prefix": "off",
                "adapter_state_after_boundary": "expert_only",
                "private_tail_kv_required": True,
                "private_tail_append_only": True,
                "private_tail_cross_expert_transfer_allowed": False,
                "commit_promotes_text_only": True,
                "committed_scaffold_reencode_required": True,
                "committed_scaffold_reencode_producer": "frozen_base",
                "expert_activation_request": "next_request",
                "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
                "full_generation_kv_shared_claimed": False,
                "naive_in_stack_q_lora_exact_reuse_claimed": False,
                "physical_kv_tensor_emitted": False,
                "token_index_emitted": False,
            },
            "adapter_control": {
                "primary": "q_only",
                "diagnostic_overlays": ["o_only", "q_plus_o"],
                "wide_lora_inherited": False,
                "controls_are_non_authorizing": True,
            },
            "safety": {
                "provider_requests": 0,
                "model_loads": 0,
                "gpu_requests": 0,
                "network_requests": 0,
                "canonical_gold_written": False,
                "heldout_read": False,
                "heldout_written": False,
                "training_authorized": False,
                "formal_training_authorized": False,
                "eval_proxy_is_heldout": False,
            },
            "manifest_sha256_sidecar_required": True,
            "claim_scope": "research_proxy_materialization_only",
        }
        manifest_snapshot, manifest_sidecar = _write_manifest(
            temporary,
            manifest,
            _validator(
                cfg.schema_snapshots["manifest"],
                "scaffold_v2_manifest_schema_invalid",
            ),
            "scaffold_v2_output_manifest_invalid",
        )
        for seal in (*source_seals, profile_seal):
            _verify_stream_unchanged(seal, "scaffold_v2_source_changed_during_read")
        for seal in output_seals:
            _verify_stream_unchanged(seal, "scaffold_v2_output_changed")
        for item in (
            projector.manifest_snapshot,
            projector.sidecar_snapshot,
            *projector.schema_snapshots,
            bundle_artifact.manifest_snapshot,
            bundle_artifact.sidecar_snapshot,
            *cfg.schema_snapshots.values(),
            manifest_snapshot,
            manifest_sidecar,
        ):
            _verify_small_unchanged(item, "scaffold_v2_bound_file_changed")
        _publish_directory(temporary, output, parent_identity)
        try:
            for seal, (relative, _split) in zip(
                output_seals, OUTPUT_FILES, strict=True
            ):
                _terminal_stream_recheck(
                    output / relative,
                    seal,
                    validator=record_validator,
                    code="scaffold_v2_output_post_publish_invalid",
                    max_bytes=cfg.max_input_bytes,
                    max_records=cfg.max_records,
                    max_line_bytes=cfg.max_line_bytes,
                    consume=lambda row, _line_sha: _validate_training_record_semantics(
                        row
                    ),
                )
            _terminal_small_recheck(
                output / "manifest.json",
                manifest_snapshot,
                "scaffold_v2_output_post_publish_invalid",
            )
            _terminal_small_recheck(
                output / "manifest.json.sha256",
                manifest_sidecar,
                "scaffold_v2_output_post_publish_invalid",
                max_bytes=256,
            )
            for seal in (*source_seals, profile_seal):
                _verify_stream_unchanged(
                    seal, "scaffold_v2_source_changed_after_publish"
                )
            for item in (
                projector.manifest_snapshot,
                projector.sidecar_snapshot,
                *projector.schema_snapshots,
                bundle_artifact.manifest_snapshot,
                bundle_artifact.sidecar_snapshot,
                *cfg.schema_snapshots.values(),
            ):
                _verify_small_unchanged(
                    item, "scaffold_v2_bound_file_changed_after_publish"
                )
            if ScaffoldV2Config.load(cfg.path) != cfg:
                _fail("scaffold_v2_config_changed_after_publish")
        except Exception:
            _cleanup_own_published_output(output)
            raise
        return {
            "output_dir": str(output),
            "manifest_sha256": manifest_snapshot.sha256,
            "records": sum(counts_by_split.values()),
            "unique_task_bundles": len(profiles),
            "pairs": 0,
            "provider_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "network_requests": 0,
            "training_authorized": False,
            "formal_training_authorized": False,
        }
    finally:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def audit_training_view(
    config: ScaffoldV2Config | str | Path,
    artifact_dir: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate and stream-validate a published v2 training view."""

    cfg = (
        config
        if isinstance(config, ScaffoldV2Config)
        else ScaffoldV2Config.load(config)
    )
    if isinstance(config, ScaffoldV2Config) and ScaffoldV2Config.load(cfg.path) != cfg:
        _fail("scaffold_v2_config_changed")
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        _fail("scaffold_v2_audit_expected_manifest_invalid")
    root_path = Path(artifact_dir).expanduser()
    _assert_plain_existing_path(
        root_path, directory=True, code="scaffold_v2_audit_artifact_invalid"
    )
    root = root_path.absolute()
    if root.resolve(strict=True) != root:
        _fail("scaffold_v2_audit_artifact_invalid")
    manifest_snapshot = _read_small(
        root / "manifest.json", "scaffold_v2_audit_manifest_invalid"
    )
    if manifest_snapshot.sha256 != expected_manifest_sha256:
        _fail("scaffold_v2_audit_manifest_mismatch")
    sidecar = _exact_sidecar(
        root, manifest_snapshot, "scaffold_v2_audit_sidecar_invalid"
    )
    manifest = _validate(
        _validator(
            cfg.schema_snapshots["manifest"], "scaffold_v2_manifest_schema_invalid"
        ),
        _json_snapshot(manifest_snapshot, "scaffold_v2_audit_manifest_invalid"),
        "scaffold_v2_audit_manifest_invalid",
    )
    if (
        manifest["producer"]["config_sha256"] != cfg.sha256
        or manifest["producer"]["implementation_sha256"] != cfg.implementation_sha256
        or manifest["producer"]["record_schema_sha256"]
        != cfg.schema_snapshots["record"].sha256
        or manifest["producer"]["manifest_schema_sha256"]
        != cfg.schema_snapshots["manifest"].sha256
    ):
        _fail("scaffold_v2_audit_producer_binding_invalid")
    record_validator = _validator(
        cfg.schema_snapshots["record"], "scaffold_v2_record_schema_invalid"
    )
    observed = Counter()
    seals: list[StreamSeal] = []
    for binding, (relative, split) in zip(manifest["files"], OUTPUT_FILES, strict=True):
        if binding["path"] != relative:
            _fail("scaffold_v2_audit_manifest_invalid")
        path = _safe_relative(root, relative, "scaffold_v2_audit_partition_invalid")

        def consume(
            row: Mapping[str, Any], _line_sha: str, *, _split: str = split
        ) -> None:
            if row["split"] != _split:
                _fail("scaffold_v2_audit_partition_invalid")
            _validate_training_record_semantics(row)
            observed[_split] += 1

        seals.append(
            _stream_jsonl(
                path,
                validator=record_validator,
                code="scaffold_v2_audit_partition_invalid",
                max_bytes=cfg.max_input_bytes,
                max_records=cfg.max_records,
                max_line_bytes=cfg.max_line_bytes,
                expected=binding,
                consume=consume,
            )
        )
    if (
        observed["train"] != manifest["counts"]["by_split"]["train"]
        or observed["eval_proxy"] != manifest["counts"]["by_split"]["eval_proxy"]
        or sum(observed.values()) != manifest["counts"]["records"]
    ):
        _fail("scaffold_v2_audit_count_invalid")
    for seal in seals:
        _verify_stream_unchanged(seal, "scaffold_v2_audit_source_changed")
    _verify_small_unchanged(manifest_snapshot, "scaffold_v2_audit_source_changed")
    _verify_small_unchanged(sidecar, "scaffold_v2_audit_source_changed")
    return {
        "status": "authenticated_research_proxy_only",
        "manifest_sha256": manifest_snapshot.sha256,
        "records": sum(observed.values()),
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "training_authorized": False,
        "formal_training_authorized": False,
    }


__all__ = [
    "BUNDLE_DESCRIPTOR_SCHEMA",
    "BUNDLE_MANIFEST_SCHEMA",
    "BUNDLE_RECORD_SCHEMA",
    "CONFIG_SCHEMA",
    "MANIFEST_SCHEMA",
    "NaturalLanguageScaffoldV2Error",
    "PRODUCER_VERSION",
    "RECORD_SCHEMA",
    "ScaffoldV2Config",
    "audit_bundle_profiles",
    "audit_training_view",
    "freeze_bundle_profiles",
    "materialize_training_view",
]
