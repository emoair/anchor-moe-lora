"""Metadata-only identity producer for scaffold confirmation studies.

The producer reads three authenticated metadata files from the frozen synthetic
diagnostic source.  It never reads source JSONL partitions, Gold, heldout, a
provider, a model, or a GPU.  Output JSONL files contain identities and binding
metadata only; they contain no prompts, answers, token IDs, or sample bodies.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator


CONFIG_PATH = (
    "configs/research/synthetic_scaffold_independent_confirmation_identity_v1.json"
)
CONFIG_SCHEMA_PATH = (
    "configs/research/"
    "synthetic_scaffold_independent_confirmation_identity_v1_config.schema.json"
)
RECORD_SCHEMA_PATH = (
    "configs/research/"
    "synthetic_scaffold_independent_confirmation_identity_v1_record.schema.json"
)
PROOF_SCHEMA_PATH = (
    "configs/research/"
    "synthetic_scaffold_independent_confirmation_identity_v1_proof.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    "configs/research/"
    "synthetic_scaffold_independent_confirmation_identity_v1_manifest.schema.json"
)
IMPLEMENTATION_PATH = (
    "src/anchor_mvp/swebench/synthetic_scaffold_independent_confirmation_identity.py"
)
CONTRACT_PATHS: Mapping[str, str] = {
    "config_schema": CONFIG_SCHEMA_PATH,
    "record_schema": RECORD_SCHEMA_PATH,
    "proof_schema": PROOF_SCHEMA_PATH,
    "manifest_schema": MANIFEST_SCHEMA_PATH,
    "implementation": IMPLEMENTATION_PATH,
}
PINNED_CONTRACT_SHA256: Mapping[str, str] = {
    "config": "4c9017c1ce0fa2c82168b03c0e157fa7c6251530c36bd3dbb1bd769f01b63ae3",
    "config_schema": "2bb127dd674bb767bac7159b60c23933ad9f5658e45379d6e6a419e5b3b49dfb",
    "record_schema": "53fee396b017eea9e3a6c9e6ad3faf2c36e6fcdf20f132ef22d5ea670a0a5f9a",
    "proof_schema": "273307c9e4c61c7ab2e4a9065020c005dac7509fc6f2d2f76969403bedd7226c",
    "manifest_schema": "4d85dd0c915302353a245c847af5229c9ade658fd1f62ce0841301be6b978289",
}
CANONICAL_FIXTURE_PATH = (
    "fixtures/research/synthetic_scaffold_independent_confirmation_identity_v1"
)
PRODUCER_VERSION = (
    "anchor.synthetic-scaffold-independent-confirmation-identity-producer.v1"
)
RECORD_VERSION = "anchor.synthetic-scaffold-independent-confirmation-identity-record.v1"
PROOF_VERSION = "anchor.synthetic-scaffold-independent-confirmation-identity-proof.v1"
MANIFEST_VERSION = (
    "anchor.synthetic-scaffold-independent-confirmation-identity-manifest.v1"
)
CLAIM_SCOPE = "metadata_only_diagnostic_identity_producer_non_authorizing"

LANGUAGES = ("en", "zh-CN")
ROLES = ("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate")
VARIANTS = ("json_only", "concise_rationale_plus_json")
FACTORS = (
    "old_task_new_template",
    "new_task_old_template",
    "new_task_new_template",
)
SCOPES = (
    "discovery_bridge",
    "producer_independent_confirmation",
    "secondary_controlled_factorial_probe",
)
IDENTITY_KINDS = ("task_semantic", "template_family", "task_template_pair")

PARTITION_PATHS = (
    "discovery/views.jsonl",
    "independent_confirmation/bundles.jsonl",
    "secondary_factorial/bundles.jsonl",
    "inventories/task_semantic_ids.jsonl",
    "inventories/template_family_ids.jsonl",
    "inventories/task_template_pair_ids.jsonl",
    "proofs/discovery_vs_independent.json",
    "proofs/secondary_factorial.json",
)
ARTIFACT_PATHS = (*PARTITION_PATHS, "manifest.json", "manifest.json.sha256")

DESCRIPTOR_SCHEMA_CONTRACT: Mapping[str, Any] = {
    "schema_version": "anchor.synthetic-scaffold-common-domain-descriptor.v1",
    "atom_catalog_schema_version": (
        "anchor.synthetic-scaffold-common-domain-atom-catalog.v1"
    ),
    "task_required_fields": [
        "task_kind",
        "goal_atoms",
        "constraint_atoms",
        "acceptance_atoms",
        "ordered_information_flow",
        "typed_parameters",
    ],
    "task_set_fields": ["goal_atoms", "constraint_atoms", "acceptance_atoms"],
    "task_ordered_fields": ["ordered_information_flow"],
    "template_required_fields": [
        "template_profile_id",
        "information_flow_stratum",
        "ordered_roles",
        "route_program",
        "visibility_program",
        "variant_program",
        "commit_program",
        "tool_trace_policy",
        "rationale_policy",
        "serialization_policy",
    ],
    "template_ordered_fields": [
        "ordered_roles",
        "route_program",
        "visibility_program",
        "variant_program",
        "commit_program",
    ],
    "ascii_symbolic_values_only": True,
    "free_text_allowed": False,
}

DESCRIPTOR_ATOM_CATALOG_VERSION = (
    "anchor.synthetic-scaffold-common-domain-atom-catalog.v1"
)
DESCRIPTOR_ATOM_CATALOG_COUNT = 241
DESCRIPTOR_ATOM_CATALOG_SHA256 = (
    "517f6b829bb78700b171a349d14541f75a9b76aa2a9267acb92a0e1a646d9545"
)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class IdentityProducerError(RuntimeError):
    """Fail-closed identity producer error."""


def _fail(code: str) -> None:
    raise IdentityProducerError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_json_bytes(value))


def _inventory_sha256(domain: str, values: Iterable[str]) -> str:
    return _canonical_sha256({"domain": domain, "values": sorted(set(values))})


def _ordered_inventory_sha256(domain: str, values: Sequence[str]) -> str:
    return _canonical_sha256({"domain": domain, "values": list(values)})


def _strict_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("identity_duplicate_json_key")
        result[key] = value
    return result


def _strict_json(data: bytes, code: str) -> Any:
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data:
        _fail(code)
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: _fail("identity_nonfinite_json_number"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IdentityProducerError(code) from exc


def _strict_jsonl(data: bytes, code: str) -> list[Mapping[str, Any]]:
    if (
        not data
        or not data.endswith(b"\n")
        or b"\r" in data
        or data.startswith(b"\xef\xbb\xbf")
    ):
        _fail(code)
    records: list[Mapping[str, Any]] = []
    for raw in data.splitlines():
        if not raw:
            _fail(code)
        value = _strict_json(raw, code)
        if not isinstance(value, Mapping):
            _fail(code)
        records.append(value)
    return records


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            _fail("identity_duplicate_yaml_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _strict_yaml(data: bytes, code: str) -> Mapping[str, Any]:
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data:
        _fail(code)
    try:
        value = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, IdentityProducerError) as exc:
        if isinstance(exc, IdentityProducerError):
            raise
        raise IdentityProducerError(code) from exc
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _validate_schema(schema: Mapping[str, Any], value: object, code: str) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    if errors:
        location = "/".join(str(part) for part in errors[0].path)
        _fail(f"{code}:{location or '<root>'}")


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    return stat.S_ISLNK(value.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _assert_no_reparse_absolute_ancestry(path: Path, code: str) -> None:
    absolute = path.absolute()
    for current in reversed((absolute, *absolute.parents)):
        if _path_lexists(current) and _is_reparse_or_symlink(current):
            _fail(code)


def _safe_relative_path(value: str) -> Path:
    if (
        not value
        or "\\" in value
        or value.startswith("/")
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        _fail("identity_relative_path_invalid")
    return Path(*value.split("/"))


def _resolve_within(root: Path, relative: str, code: str) -> Path:
    rel = _safe_relative_path(relative)
    candidate = (root / rel).absolute()
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail(code)
    _assert_no_reparse_absolute_ancestry(candidate, code)
    if not candidate.is_file() or _is_reparse_or_symlink(candidate):
        _fail(code)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(code)
    return candidate


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    identity: tuple[int, int, int, int]
    sha256: str

    @classmethod
    def capture(cls, path: Path, *, maximum_bytes: int = 50000000) -> _Snapshot:
        _assert_no_reparse_absolute_ancestry(path, "identity_snapshot_reparse_path")
        try:
            with path.open("rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                    _fail("identity_snapshot_file_invalid")
                data = handle.read(maximum_bytes + 1)
                after = os.fstat(handle.fileno())
        except OSError as exc:
            raise IdentityProducerError("identity_snapshot_file_invalid") from exc
        _assert_no_reparse_absolute_ancestry(path, "identity_snapshot_reparse_path")
        path_after = path.lstat()
        identity = _stat_identity(before)
        if (
            _is_reparse_or_symlink(path)
            or identity != _stat_identity(after)
            or identity != _stat_identity(path_after)
            or len(data) != before.st_size
            or len(data) > maximum_bytes
        ):
            _fail("identity_snapshot_changed_during_read")
        return cls(path=path, data=data, identity=identity, sha256=_sha256(data))

    def assert_unchanged(self, code: str) -> None:
        try:
            current = _Snapshot.capture(self.path)
            if current.identity != self.identity or current.data != self.data:
                _fail(code)
        except OSError as exc:
            raise IdentityProducerError(code) from exc


_LOADED_IMPLEMENTATION_SNAPSHOT = _Snapshot.capture(Path(__file__).resolve(strict=True))


def _artifact_descriptor(root: Path, snapshot: _Snapshot) -> dict[str, Any]:
    return {
        "path": snapshot.path.relative_to(root).as_posix(),
        "bytes": len(snapshot.data),
        "sha256": snapshot.sha256,
    }


def _capture_declared(root: Path, relative: str) -> _Snapshot:
    return _Snapshot.capture(
        _resolve_within(root, relative, "identity_input_path_invalid")
    )


def _git_environment() -> dict[str, str]:
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP"}
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(source_root: Path, arguments: Sequence[str], *, check: bool = True) -> bytes:
    command = [
        "git",
        "--no-replace-objects",
        "-c",
        "protocol.allow=never",
        "-c",
        "core.autocrlf=false",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=source_root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise IdentityProducerError("identity_source_git_command_timeout") from exc
    if check and result.returncode != 0:
        _fail("identity_source_git_command_failed")
    return result.stdout


def _authenticate_source_git(
    source_root: Path,
    binding: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
) -> None:
    top_level_raw = _git(source_root, ["rev-parse", "--show-toplevel"])
    try:
        top_level = Path(top_level_raw.decode("utf-8").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise IdentityProducerError("identity_source_git_top_level_invalid") from exc
    if top_level != source_root:
        _fail("identity_source_git_top_level_mismatch")
    git_dir_raw = _git(source_root, ["rev-parse", "--absolute-git-dir"])
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip()).absolute()
    except UnicodeDecodeError as exc:
        raise IdentityProducerError("identity_source_git_dir_invalid") from exc
    _assert_no_reparse_absolute_ancestry(git_dir, "identity_source_git_dir_invalid")
    if not git_dir.is_dir() or _is_reparse_or_symlink(git_dir):
        _fail("identity_source_git_dir_invalid")
    grafts = git_dir / "info" / "grafts"
    if grafts.exists() and grafts.stat().st_size:
        _fail("identity_source_git_grafts_forbidden")
    replace_refs = _git(
        source_root, ["for-each-ref", "--format=%(refname)", "refs/replace/"]
    )
    if replace_refs.strip():
        _fail("identity_source_git_replace_refs_forbidden")
    commit = str(binding["source_commit"])
    commit_raw = _git(source_root, ["cat-file", "-p", commit])
    header = (
        commit_raw.split(b"\n\n", 1)[0].decode("ascii", errors="strict").splitlines()
    )
    tree_lines = [line[5:] for line in header if line.startswith("tree ")]
    parent_lines = [line[7:] for line in header if line.startswith("parent ")]
    if tree_lines != [binding["source_tree"]] or parent_lines != [
        binding["source_parent_commit"]
    ]:
        _fail("identity_source_git_commit_shape_invalid")
    for role in ("config", "manifest", "manifest_sidecar"):
        relative = str(binding[role]["path"])
        blob = _git(source_root, ["cat-file", "blob", f"{commit}:{relative}"])
        if blob != snapshots[role].data:
            _fail("identity_source_git_blob_mismatch")
    plan_commit = str(binding["consumer_plan_commit"])
    plan_commit_raw = _git(source_root, ["cat-file", "-p", plan_commit])
    plan_header = (
        plan_commit_raw.split(b"\n\n", 1)[0]
        .decode("ascii", errors="strict")
        .splitlines()
    )
    plan_trees = [line[5:] for line in plan_header if line.startswith("tree ")]
    plan_parents = [line[7:] for line in plan_header if line.startswith("parent ")]
    if plan_trees != [binding["consumer_plan_tree"]] or plan_parents != [
        binding["consumer_plan_parent_commit"]
    ]:
        _fail("identity_consumer_plan_git_commit_shape_invalid")
    plan_relative = str(binding["consumer_plan_config"]["path"])
    plan_blob = _git(
        source_root, ["cat-file", "blob", f"{plan_commit}:{plan_relative}"]
    )
    if plan_blob != snapshots["consumer_plan_config"].data:
        _fail("identity_consumer_plan_git_blob_mismatch")
    try:
        ancestor = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-c",
                "protocol.allow=never",
                "merge-base",
                "--is-ancestor",
                commit,
                plan_commit,
            ],
            cwd=source_root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise IdentityProducerError("identity_source_git_command_timeout") from exc
    if ancestor.returncode != 0:
        _fail("identity_source_not_ancestor_of_consumer_plan")


def _load_contract(
    repo_root: Path, config_path: Path
) -> tuple[
    dict[str, _Snapshot],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    config_snapshot = _Snapshot.capture(config_path)
    config = _strict_json(config_snapshot.data, "identity_config_json_invalid")
    if not isinstance(config, Mapping):
        _fail("identity_config_root_invalid")
    snapshots: dict[str, _Snapshot] = {"config": config_snapshot}
    for role, relative in CONTRACT_PATHS.items():
        snapshots[role] = _capture_declared(repo_root, relative)
    if any(
        snapshots[role].sha256 != expected
        for role, expected in PINNED_CONTRACT_SHA256.items()
    ):
        _fail("identity_pinned_contract_sha256_mismatch")
    if (
        snapshots["implementation"].path.resolve(strict=True)
        != _LOADED_IMPLEMENTATION_SNAPSHOT.path
        or snapshots["implementation"].data != _LOADED_IMPLEMENTATION_SNAPSHOT.data
    ):
        _fail("identity_running_implementation_mismatch")
    snapshots["running_implementation"] = _LOADED_IMPLEMENTATION_SNAPSHOT
    config_schema = _strict_json(
        snapshots["config_schema"].data, "identity_config_schema_invalid"
    )
    record_schema = _strict_json(
        snapshots["record_schema"].data, "identity_record_schema_invalid"
    )
    proof_schema = _strict_json(
        snapshots["proof_schema"].data, "identity_proof_schema_invalid"
    )
    manifest_schema = _strict_json(
        snapshots["manifest_schema"].data, "identity_manifest_schema_invalid"
    )
    for value in (config_schema, record_schema, proof_schema, manifest_schema):
        if not isinstance(value, Mapping):
            _fail("identity_schema_root_invalid")
        Draft202012Validator.check_schema(value)
    for role, value in (
        ("config_schema", config_schema),
        ("record_schema", record_schema),
        ("proof_schema", proof_schema),
        ("manifest_schema", manifest_schema),
    ):
        if (
            _strict_json(
                snapshots[role].data, "identity_contract_schema_reparse_failed"
            )
            != value
        ):
            _fail("identity_contract_schema_reparse_mismatch")
    _validate_schema(config_schema, config, "identity_config_schema_validation_failed")
    paths = config.get("paths")
    if not isinstance(paths, Mapping) or any(
        paths.get(role) != relative for role, relative in CONTRACT_PATHS.items()
    ):
        _fail("identity_config_contract_path_mismatch")
    if _strict_json(config_snapshot.data, "identity_config_reparse_failed") != config:
        _fail("identity_config_reparse_mismatch")
    return (
        snapshots,
        config,
        config_schema,
        record_schema,
        proof_schema,
        manifest_schema,
    )


def _validate_consumer_plan(plan: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    try:
        dataset = plan["confirmation_dataset"]
        matrix = plan["evaluation_matrix"]
        boundaries = plan["study_boundaries"]
    except KeyError as exc:
        raise IdentityProducerError("identity_consumer_plan_shape_invalid") from exc
    strata = [
        item["stratum_id"]
        for item in config["semantic_ontology"]["information_flow_strata"]
    ]
    factorial = config["secondary_controlled_factorial"]
    expected_cells = [
        {
            "id": "old_task_new_template",
            "task_identity": "discovery_task",
            "template_identity": "confirmation_template",
            "old_task_membership_required": True,
            "new_task_nonoverlap_required": False,
            "old_template_membership_required": False,
            "new_template_nonoverlap_required": True,
            "task_template_pair_nonoverlap_required": True,
        },
        {
            "id": "new_task_old_template",
            "task_identity": "confirmation_task",
            "template_identity": "discovery_template",
            "old_task_membership_required": False,
            "new_task_nonoverlap_required": True,
            "old_template_membership_required": True,
            "new_template_nonoverlap_required": False,
            "task_template_pair_nonoverlap_required": True,
        },
        {
            "id": "new_task_new_template",
            "task_identity": "confirmation_task",
            "template_identity": "confirmation_template",
            "old_task_membership_required": False,
            "new_task_nonoverlap_required": True,
            "old_template_membership_required": False,
            "new_template_nonoverlap_required": True,
            "task_template_pair_nonoverlap_required": True,
        },
    ]
    if (
        plan.get("schema_version")
        != "anchor.qwen-multiseed-independent-bundle-plan-config.v1"
        or boundaries.get("producer_independent_confirmation", {}).get(
            "satisfied_by_secondary_controlled_factorial_probe"
        )
        is not False
        or boundaries.get("secondary_controlled_factorial_probe", {}).get(
            "may_satisfy_independent_confirmation"
        )
        is not False
        or dataset.get("scope") != "secondary_controlled_factorial_probe_only"
        or dataset.get("source_bundles_total") != 60
        or dataset.get("train_bundles") != 40
        or dataset.get("eval_proxy_bundles") != 20
        or dataset.get("roles") != 5
        or dataset.get("paired_variants") != 2
        or dataset.get("expected_records") != 600
        or dataset.get("split_group_key") != "task_bundle_sha256"
        or dataset.get("split_before_role_variant_augmentation") is not True
        or dataset.get("all_role_variant_views_same_split") is not True
        or list(dataset.get("languages", ())) != list(LANGUAGES)
        or list(dataset.get("information_flow_strata", ())) != strata
        or dataset.get("language_stratum_cells") != 10
        or dataset.get("bundles_per_language_stratum_cell") != 6
        or dataset.get("train_bundles_per_language_stratum_cell") != 4
        or dataset.get("eval_bundles_per_language_stratum_cell") != 2
        or dataset.get("eval_proxy_is_heldout") is not False
        or matrix.get("status") != "metadata_blueprint_only"
        or matrix.get("design") != "controlled_factorial_confirmation"
        or matrix.get("bundles_per_factor_cell") != 20
        or matrix.get("factor_views_total") != 60
        or matrix.get("quota_algorithm") != factorial["quota_algorithm"]
        or matrix.get("quota_domain")
        != factorial["within_factor_pair_assignment_domain"]
        or matrix.get("quota_rotation_derivation")
        != factorial["quota_rotation_derivation"]
        or matrix.get("quota_rotation") != factorial["quota_rotation"]
        or matrix.get("within_factor_pair_assignment_algorithm")
        != factorial["within_factor_pair_assignment_algorithm"]
        or matrix.get("bundles_per_factor") != 20
        or list(matrix.get("train_factor_quotas", ()))
        != [factorial["train_factor_quotas"][factor] for factor in FACTORS]
        or list(matrix.get("eval_factor_quotas", ()))
        != [factorial["eval_factor_quotas"][factor] for factor in FACTORS]
        or matrix.get("six_eval_factor") != "new_task_old_template"
        or matrix.get("common_domain_inventories")
        != {
            "task_hash": config["identity_contract"]["task_domain"],
            "template_hash": config["identity_contract"]["template_domain"],
            "task_template_pair_hash": config["identity_contract"]["pair_domain"],
        }
        or list(matrix.get("cells", ())) != expected_cells
        or matrix.get("require_task_inventory_binding") is not True
        or matrix.get("require_template_inventory_binding") is not True
        or matrix.get("require_task_template_pair_inventory_binding") is not True
        or matrix.get("global_task_nonoverlap_required") is not False
        or matrix.get("global_template_nonoverlap_required") is not False
        or matrix.get("global_task_template_pair_nonoverlap_required") is not True
    ):
        _fail("identity_consumer_plan_contract_mismatch")


def _load_source(
    source_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, _Snapshot], Mapping[str, Any], Mapping[str, Any]]:
    binding = config["source_binding"]
    if list(binding["source_metadata_allowed_reads"]) != [
        binding["config"]["path"],
        binding["manifest"]["path"],
        binding["manifest_sidecar"]["path"],
    ] or list(binding["consumer_contract_allowed_reads"]) != [
        binding["consumer_plan_config"]["path"]
    ]:
        _fail("identity_source_read_set_not_exact")
    snapshots: dict[str, _Snapshot] = {}
    for role in ("config", "manifest", "manifest_sidecar", "consumer_plan_config"):
        declared = binding[role]
        relative = str(declared["path"])
        lowered = relative.lower()
        if relative.endswith(".jsonl") or any(
            fragment.lower() in lowered
            for fragment in config["audit_contract"]["forbidden_read_path_fragments"]
        ):
            _fail("identity_forbidden_source_read")
        snapshot = _capture_declared(source_root, relative)
        if (
            snapshot.sha256 != declared["sha256"]
            or len(snapshot.data) != declared["bytes"]
        ):
            _fail("identity_source_artifact_binding_mismatch")
        snapshots[role] = snapshot
    expected_sidecar = f"{snapshots['manifest'].sha256}  manifest.json\n".encode(
        "ascii"
    )
    if snapshots["manifest_sidecar"].data != expected_sidecar:
        _fail("identity_source_manifest_sidecar_invalid")
    source_config = _strict_yaml(
        snapshots["config"].data, "identity_source_config_invalid"
    )
    source_manifest = _strict_json(
        snapshots["manifest"].data, "identity_source_manifest_invalid"
    )
    consumer_plan = _strict_yaml(
        snapshots["consumer_plan_config"].data,
        "identity_consumer_plan_config_invalid",
    )
    if not isinstance(source_manifest, Mapping):
        _fail("identity_source_manifest_root_invalid")
    if (
        _strict_yaml(snapshots["config"].data, "identity_source_config_reparse_failed")
        != source_config
    ):
        _fail("identity_source_config_reparse_mismatch")
    if (
        _strict_json(
            snapshots["manifest"].data, "identity_source_manifest_reparse_failed"
        )
        != source_manifest
    ):
        _fail("identity_source_manifest_reparse_mismatch")
    if (
        _strict_yaml(
            snapshots["consumer_plan_config"].data,
            "identity_consumer_plan_reparse_failed",
        )
        != consumer_plan
    ):
        _fail("identity_consumer_plan_reparse_mismatch")
    _validate_consumer_plan(consumer_plan, config)
    _authenticate_source_git(source_root, binding, snapshots)
    if (
        source_manifest.get("counts", {}).get("source_bundles") != 10
        or source_manifest.get("counts", {}).get("records") != 100
        or source_manifest.get("producer", {}).get("config", {}).get("sha256")
        != snapshots["config"].sha256
        or source_manifest.get("producer", {}).get("implementation", {}).get("sha256")
        != binding["generator_implementation_sha256"]
        or source_manifest.get("producer", {}).get("closed_grammar", {}).get("sha256")
        != binding["closed_grammar_sha256"]
        or source_manifest.get("inventories", {}).get("source_bundle_ids_sha256")
        != binding["source_bundle_inventory_sha256"]
        or source_manifest.get("audit", {}).get("protected_body_reads") != 0
        or source_manifest.get("claims", {}).get("training_authorized") is not False
        or source_manifest.get("claims", {}).get("eval_proxy_is_heldout") is not False
    ):
        _fail("identity_source_manifest_semantics_invalid")
    return snapshots, source_config, source_manifest


def _collect_string_leaves(value: object, destination: set[str]) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _collect_string_leaves(child, destination)
    elif isinstance(value, list):
        for child in value:
            _collect_string_leaves(child, destination)
    elif isinstance(value, str):
        destination.add(value)


def _descriptor_atom_catalog(
    config: Mapping[str, Any],
) -> tuple[frozenset[str], str]:
    """Recompute the closed, namespace-neutral descriptor atom catalog.

    The catalog is derived only from descriptor-producing metadata plus the
    fixed derived atoms below.  Its count and canonical digest are compiled
    into this implementation, so adding a language/source/namespace salt to
    the config cannot silently expand the common identity domain.
    """

    atoms: set[str] = set()
    discovery = config["discovery_bridge"]
    _collect_string_leaves(discovery["old_template_descriptor"], atoms)
    for group in discovery["semantic_groups"]:
        _collect_string_leaves(group["descriptor"], atoms)
    ontology = config["semantic_ontology"]
    for profile in ontology["workflow_profiles"]:
        for field in (
            "task_kind",
            "profile_goal_atom",
            "profile_constraint_atom",
            "profile_acceptance_atom",
            "state_scope",
            "trust_boundary",
        ):
            _collect_string_leaves(profile[field], atoms)
    for stratum in ontology["information_flow_strata"]:
        for field in (
            "stratum_id",
            "goal_atoms",
            "constraint_atoms",
            "acceptance_atoms",
            "ordered_information_flow",
        ):
            _collect_string_leaves(stratum[field], atoms)
    for template in ontology["template_profiles"]:
        for field in (
            "template_profile_id",
            "route_program",
            "commit_program",
            "rationale_policy",
        ):
            _collect_string_leaves(template[field], atoms)
    atoms.update(
        {
            "deterministic_execution_required",
            "identity_receipt_recomputable",
            "receipt_bounded",
            "required",
            "exclude_current",
            "exclude_future",
            "exclude_forbidden",
            "allow_committed_predecessors",
            "harmless_synthetic_local_receipts",
            "canonical_json_utf8_compact",
        }
    )
    catalog_sha256 = _canonical_sha256(
        {
            "schema_version": DESCRIPTOR_ATOM_CATALOG_VERSION,
            "atoms": sorted(atoms),
        }
    )
    configured = config["identity_contract"]
    if (
        len(atoms) != DESCRIPTOR_ATOM_CATALOG_COUNT
        or configured["descriptor_atom_catalog_schema_version"]
        != DESCRIPTOR_ATOM_CATALOG_VERSION
        or configured["descriptor_atom_catalog_count"] != DESCRIPTOR_ATOM_CATALOG_COUNT
        or configured["descriptor_atom_catalog_sha256"]
        != DESCRIPTOR_ATOM_CATALOG_SHA256
        or catalog_sha256 != DESCRIPTOR_ATOM_CATALOG_SHA256
    ):
        _fail("identity_descriptor_atom_catalog_drift")
    return frozenset(atoms), catalog_sha256


def _assert_ascii_symbolic(
    value: object, forbidden_keys: set[str], allowed_atoms: frozenset[str]
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key in forbidden_keys:
                _fail("identity_descriptor_forbidden_key")
            try:
                key.encode("ascii")
            except UnicodeEncodeError as exc:
                raise IdentityProducerError(
                    "identity_descriptor_non_ascii_key"
                ) from exc
            _assert_ascii_symbolic(child, forbidden_keys, allowed_atoms)
    elif isinstance(value, list):
        for child in value:
            _assert_ascii_symbolic(child, forbidden_keys, allowed_atoms)
    elif isinstance(value, str):
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise IdentityProducerError("identity_descriptor_non_ascii_value") from exc
        if value not in allowed_atoms:
            _fail("identity_descriptor_atom_not_in_closed_catalog")
        normalized = value.lower().replace("-", "_")
        forbidden_exact = {
            "en",
            "zh",
            "zh_cn",
            "train",
            "eval_proxy",
            "language",
            "source_namespace",
            "source_bundle_id",
            "bundle_key",
            "split",
            "factor",
            "ordinal",
            "nonce",
            "salt",
            "seed",
            *FACTORS,
        }
        forbidden_fragments = (
            "language_",
            "_language",
            "namespace",
            "source_bundle",
            "bundle_key",
        )
        forbidden_suffixes = ("_salt", "_nonce", "_seed", "_ordinal")
        if (
            normalized in forbidden_exact
            or any(fragment in normalized for fragment in forbidden_fragments)
            or normalized.endswith(forbidden_suffixes)
            or any(component.isdigit() for component in normalized.split("_"))
        ):
            _fail("identity_descriptor_namespace_or_salt_value")
    elif not isinstance(value, (bool, int)) or isinstance(value, float):
        _fail("identity_descriptor_value_type_invalid")


def _normalize_task_descriptor(
    descriptor: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    required = config["semantic_ontology"]["task_descriptor_required_fields"]
    if set(descriptor) != set(required):
        _fail("identity_task_descriptor_shape_invalid")
    forbidden = set(config["identity_contract"]["forbidden_semantic_descriptor_fields"])
    allowed_atoms, _ = _descriptor_atom_catalog(config)
    _assert_ascii_symbolic(descriptor, forbidden, allowed_atoms)
    normalized = dict(descriptor)
    for field in ("goal_atoms", "constraint_atoms", "acceptance_atoms"):
        values = list(descriptor[field])
        if len(values) < 3 or len(set(values)) != len(values):
            _fail("identity_task_descriptor_set_invalid")
        normalized[field] = sorted(values)
    flow = list(descriptor["ordered_information_flow"])
    if len(flow) < 4 or len(set(flow)) != len(flow):
        _fail("identity_task_descriptor_flow_invalid")
    normalized["ordered_information_flow"] = flow
    parameters = descriptor["typed_parameters"]
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "state_scope",
        "trust_boundary",
        "reversibility",
        "determinism",
    }:
        _fail("identity_task_descriptor_parameters_invalid")
    normalized["typed_parameters"] = dict(parameters)
    return normalized


def _normalize_template_descriptor(
    descriptor: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    required = set(DESCRIPTOR_SCHEMA_CONTRACT["template_required_fields"])
    if set(descriptor) != required:
        _fail("identity_template_descriptor_shape_invalid")
    forbidden = set(config["identity_contract"]["forbidden_semantic_descriptor_fields"])
    allowed_atoms, _ = _descriptor_atom_catalog(config)
    _assert_ascii_symbolic(descriptor, forbidden, allowed_atoms)
    if list(descriptor["ordered_roles"]) != list(ROLES) or list(
        descriptor["variant_program"]
    ) != list(VARIANTS):
        _fail("identity_template_descriptor_program_invalid")
    for field in DESCRIPTOR_SCHEMA_CONTRACT["template_ordered_fields"]:
        values = list(descriptor[field])
        if len(values) != len(set(values)):
            _fail("identity_template_descriptor_duplicate_step")
    return dict(descriptor)


def _descriptor_context(config: Mapping[str, Any]) -> tuple[str, str, str]:
    descriptor_schema_sha256 = _canonical_sha256(DESCRIPTOR_SCHEMA_CONTRACT)
    ontology_sha256 = _canonical_sha256(config["semantic_ontology"])
    _, atom_catalog_sha256 = _descriptor_atom_catalog(config)
    return descriptor_schema_sha256, ontology_sha256, atom_catalog_sha256


def _task_identity(descriptor: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    normalized = _normalize_task_descriptor(descriptor, config)
    descriptor_schema_sha256, ontology_sha256, atom_catalog_sha256 = (
        _descriptor_context(config)
    )
    return _canonical_sha256(
        {
            "domain": config["identity_contract"]["task_domain"],
            "descriptor_schema_sha256": descriptor_schema_sha256,
            "ontology_sha256": ontology_sha256,
            "descriptor_atom_catalog_sha256": atom_catalog_sha256,
            "descriptor": normalized,
        }
    )


def _template_identity(descriptor: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    normalized = _normalize_template_descriptor(descriptor, config)
    descriptor_schema_sha256, ontology_sha256, atom_catalog_sha256 = (
        _descriptor_context(config)
    )
    return _canonical_sha256(
        {
            "domain": config["identity_contract"]["template_domain"],
            "descriptor_schema_sha256": descriptor_schema_sha256,
            "ontology_sha256": ontology_sha256,
            "descriptor_atom_catalog_sha256": atom_catalog_sha256,
            "descriptor": normalized,
        }
    )


def _pair_identity(
    task_sha256: str, template_sha256: str, config: Mapping[str, Any]
) -> str:
    return _canonical_sha256(
        {
            "domain": config["identity_contract"]["pair_domain"],
            "task_semantic_sha256": task_sha256,
            "template_family_sha256": template_sha256,
        }
    )


def _identity_leaves(
    task_sha256: str, template_sha256: str, config: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "task_semantic_sha256": task_sha256,
        "source_task_blueprint_sha256": task_sha256,
        "template_family_sha256": template_sha256,
        "task_template_pair_sha256": _pair_identity(
            task_sha256, template_sha256, config
        ),
    }


def _expansion() -> dict[str, Any]:
    return {
        "ordered_roles": list(ROLES),
        "ordered_variants": list(VARIANTS),
        "expected_record_count": 10,
        "split_before_role_variant_expansion": True,
        "all_views_same_split": True,
    }


def _source_bundle_payload(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "domain": "anchor.synthetic-nl-scaffold-bundle.v1",
        "bundle_key": bundle["bundle_key"],
        "language": bundle["language"],
        "archetype": bundle["archetype"],
        "task_text": bundle["task_text"],
        "constraints": list(bundle["constraints"]),
        "closed_grammar_id": "anchor.synthetic-nl-scaffold-closed-grammar.v1",
        "generation_seed_id": "anchor.synthetic-nl-scaffold-diagnostic.seed.v1",
    }


def _source_split_assignments(source_config: Mapping[str, Any]) -> Mapping[str, str]:
    salt = str(source_config["split_contract"]["salt"])
    assignments: dict[str, str] = {}
    for language in LANGUAGES:
        candidates: list[tuple[str, str]] = []
        for bundle in source_config["bundles"]:
            if bundle["language"] != language:
                continue
            digest = _canonical_sha256(_source_bundle_payload(bundle))
            bundle_id = f"syn-nl-bundle-v1:{digest}"
            score = _sha256(f"{salt}\0{language}\0{bundle['bundle_key']}".encode())
            candidates.append((score, bundle_id))
        candidates.sort()
        if len(candidates) != 5:
            _fail("identity_source_language_bundle_count_invalid")
        for index, (_, bundle_id) in enumerate(candidates):
            assignments[bundle_id] = "eval_proxy" if index == 0 else "train"
    return assignments


def _make_discovery_records(
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    bundles = source_config.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 10:
        _fail("identity_source_bundle_catalog_invalid")
    by_key = {bundle["bundle_key"]: bundle for bundle in bundles}
    if len(by_key) != 10:
        _fail("identity_source_bundle_key_duplicate")
    assignments = _source_split_assignments(source_config)
    source_bundle_ids = [
        f"syn-nl-bundle-v1:{_canonical_sha256(_source_bundle_payload(bundle))}"
        for bundle in bundles
    ]
    observed_inventory = _inventory_sha256(
        "anchor.synthetic-nl-scaffold-bundle-inventory.v1", source_bundle_ids
    )
    if observed_inventory != source_manifest["inventories"]["source_bundle_ids_sha256"]:
        _fail("identity_source_bundle_inventory_mismatch")
    observed_train = sorted(
        bundle_id for bundle_id, split in assignments.items() if split == "train"
    )
    observed_eval = sorted(
        bundle_id for bundle_id, split in assignments.items() if split == "eval_proxy"
    )
    if (
        observed_train != source_manifest["split_contract"]["train_bundle_ids"]
        or observed_eval != source_manifest["split_contract"]["eval_proxy_bundle_ids"]
    ):
        _fail("identity_source_bundle_split_mismatch")
    old_template = config["discovery_bridge"]["old_template_descriptor"]
    old_template_sha = _template_identity(old_template, config)
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    semantic_to_splits: dict[str, set[str]] = defaultdict(set)
    semantic_to_views: dict[str, list[str]] = defaultdict(list)
    for group in config["discovery_bridge"]["semantic_groups"]:
        descriptor = group["descriptor"]
        task_sha = _task_identity(descriptor, config)
        normalized_descriptor = _normalize_task_descriptor(descriptor, config)
        semantic_atoms = [
            normalized_descriptor["task_kind"],
            *normalized_descriptor["goal_atoms"],
            *normalized_descriptor["constraint_atoms"],
            *normalized_descriptor["acceptance_atoms"],
            *normalized_descriptor["ordered_information_flow"],
            *normalized_descriptor["typed_parameters"].values(),
        ]
        semantic_atom_inventory_sha = _inventory_sha256(
            "anchor.synthetic-scaffold-semantic-atom-inventory.v1", semantic_atoms
        )
        grouped_bundle_ids: list[str] = []
        grouped_languages: list[str] = []
        staged: list[tuple[Mapping[str, Any], str, str, str]] = []
        for key in group["source_bundle_keys"]:
            if key in seen_keys or key not in by_key:
                _fail("identity_discovery_bridge_source_key_invalid")
            seen_keys.add(key)
            bundle = by_key[key]
            if bundle["archetype"] != group["semantic_key"]:
                _fail("identity_discovery_bridge_archetype_binding_invalid")
            payload_sha = _canonical_sha256(_source_bundle_payload(bundle))
            source_bundle_id = f"syn-nl-bundle-v1:{payload_sha}"
            split = assignments[source_bundle_id]
            grouped_bundle_ids.append(source_bundle_id)
            grouped_languages.append(str(bundle["language"]))
            staged.append((bundle, payload_sha, source_bundle_id, split))
        if (
            sorted(grouped_languages) != sorted(LANGUAGES)
            or len(grouped_bundle_ids) != 2
        ):
            _fail("identity_discovery_bilingual_group_invalid")
        semantic_group_sha = _canonical_sha256(
            {
                "domain": "anchor.synthetic-scaffold-bilingual-semantic-group.v1",
                "task_semantic_sha256": task_sha,
                "source_bundle_ids": sorted(grouped_bundle_ids),
            }
        )
        for bundle, payload_sha, source_bundle_id, split in staged:
            source_goal_sha = _sha256(str(bundle["task_text"]).encode("utf-8"))
            constraint_hashes = [
                _sha256(str(value).encode("utf-8")) for value in bundle["constraints"]
            ]
            coverage_map_sha = _canonical_sha256(
                {
                    "domain": "anchor.synthetic-scaffold-curated-semantic-coverage-map.v1",
                    "source_goal_sha256": source_goal_sha,
                    "ordered_source_constraint_sha256s": constraint_hashes,
                    "task_semantic_sha256": task_sha,
                    "semantic_atom_inventory_sha256": semantic_atom_inventory_sha,
                }
            )
            semantic_to_splits[task_sha].add(split)
            semantic_to_views[task_sha].append(source_bundle_id)
            records.append(
                {
                    "schema_version": RECORD_VERSION,
                    "record_kind": "discovery_view",
                    "track": "discovery_bridge",
                    "source_bundle_key": bundle["bundle_key"],
                    "source_bundle_id": source_bundle_id,
                    "source_bundle_payload_sha256": payload_sha,
                    "localized_view_sha256": _canonical_sha256(bundle),
                    "source_goal_sha256": source_goal_sha,
                    "ordered_source_constraint_sha256s": constraint_hashes,
                    "source_clause_count": 4,
                    "semantic_atom_inventory_sha256": semantic_atom_inventory_sha,
                    "semantic_coverage_map_sha256": coverage_map_sha,
                    "language": bundle["language"],
                    "split": split,
                    "semantic_group_sha256": semantic_group_sha,
                    "identities": _identity_leaves(task_sha, old_template_sha, config),
                    "expansion": _expansion(),
                    "curated_bridge_not_automatic_translation_proof": True,
                }
            )
    if seen_keys != set(by_key) or len(records) != 10 or len(semantic_to_splits) != 5:
        _fail("identity_discovery_bridge_coverage_invalid")
    if any(len(values) != 2 for values in semantic_to_views.values()):
        _fail("identity_discovery_bridge_pairing_invalid")
    semantic_intersection = sorted(
        semantic
        for semantic, splits in semantic_to_splits.items()
        if splits == {"train", "eval_proxy"}
    )
    if (
        len(semantic_intersection)
        != config["discovery_bridge"][
            "historical_semantic_train_eval_intersection_expected"
        ]
    ):
        _fail("identity_historical_semantic_split_intersection_drift")
    return sorted(records, key=lambda item: item["source_bundle_id"]), {
        "old_template_sha256": old_template_sha,
        "historical_semantic_intersection": semantic_intersection,
    }


def _profile_maps(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    ontology = config["semantic_ontology"]
    profiles = {item["profile_id"]: item for item in ontology["workflow_profiles"]}
    strata = {item["stratum_id"]: item for item in ontology["information_flow_strata"]}
    templates = {
        item["template_profile_id"]: item for item in ontology["template_profiles"]
    }
    if len(profiles) != 12 or len(strata) != 5 or len(templates) != 4:
        _fail("identity_ontology_catalog_duplicate")
    return profiles, strata, templates


def _confirmation_task_descriptor(
    profile: Mapping[str, Any], stratum: Mapping[str, Any]
) -> Mapping[str, Any]:
    return {
        "task_kind": profile["task_kind"],
        "goal_atoms": sorted([*stratum["goal_atoms"], profile["profile_goal_atom"]]),
        "constraint_atoms": sorted(
            [
                *stratum["constraint_atoms"],
                profile["profile_constraint_atom"],
                "deterministic_execution_required",
            ]
        ),
        "acceptance_atoms": sorted(
            [
                *stratum["acceptance_atoms"],
                profile["profile_acceptance_atom"],
                "identity_receipt_recomputable",
            ]
        ),
        "ordered_information_flow": list(stratum["ordered_information_flow"]),
        "typed_parameters": {
            "state_scope": profile["state_scope"],
            "trust_boundary": profile["trust_boundary"],
            "reversibility": "receipt_bounded",
            "determinism": "required",
        },
    }


def _confirmation_template_descriptor(
    template_profile: Mapping[str, Any], stratum: Mapping[str, Any]
) -> Mapping[str, Any]:
    return {
        "template_profile_id": template_profile["template_profile_id"],
        "information_flow_stratum": stratum["stratum_id"],
        "ordered_roles": list(ROLES),
        "route_program": list(template_profile["route_program"]),
        "visibility_program": [
            "exclude_current",
            "exclude_future",
            "exclude_forbidden",
            "allow_committed_predecessors",
        ],
        "variant_program": list(VARIANTS),
        "commit_program": list(template_profile["commit_program"]),
        "tool_trace_policy": "harmless_synthetic_local_receipts",
        "rationale_policy": template_profile["rationale_policy"],
        "serialization_policy": "canonical_json_utf8_compact",
    }


def _source_bundle_identity(
    prefix: str,
    domain: str,
    track: str,
    language: str,
    stratum: str,
    task_sha: str,
    template_sha: str,
    factor: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "domain": domain,
        "track": track,
        "language": language,
        "stratum": stratum,
        "task_semantic_sha256": task_sha,
        "template_family_sha256": template_sha,
    }
    if factor is not None:
        payload["factor"] = factor
    return f"{prefix}:{_canonical_sha256(payload)}"


def _task_bundle_identity(
    source_namespace: str,
    source_bundle_id: str,
    language: str,
    task_sha: str,
    config: Mapping[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "domain": config["identity_contract"]["task_bundle_domain"],
            "source_namespace": source_namespace,
            "source_bundle_id": source_bundle_id,
            "language": language,
            "source_task_blueprint_sha256": task_sha,
        }
    )


def _make_independent_records(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[tuple[str, str, str], Mapping[str, Any]]]:
    profiles, strata, templates = _profile_maps(config)
    track = config["independent_confirmation"]
    template_order = list(templates)
    profile_order = list(profiles)
    records: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for language in track["languages"]:
        for stratum_id in strata:
            stratum = strata[stratum_id]
            cell: list[dict[str, Any]] = []
            for profile_id in track["profiles_by_language"][language]:
                profile = profiles[profile_id]
                template_profile_id = template_order[
                    profile_order.index(profile_id) % len(template_order)
                ]
                template_profile = templates[template_profile_id]
                task_descriptor = _confirmation_task_descriptor(profile, stratum)
                template_descriptor = _confirmation_template_descriptor(
                    template_profile, stratum
                )
                task_sha = _task_identity(task_descriptor, config)
                template_sha = _template_identity(template_descriptor, config)
                source_bundle_id = _source_bundle_identity(
                    "ic-source-v1",
                    "anchor.synthetic-scaffold-independent-confirmation-source-bundle.v1",
                    track["track_id"],
                    language,
                    stratum_id,
                    task_sha,
                    template_sha,
                )
                task_bundle_sha = _task_bundle_identity(
                    track["source_namespace"],
                    source_bundle_id,
                    language,
                    task_sha,
                    config,
                )
                record = {
                    "schema_version": RECORD_VERSION,
                    "record_kind": "independent_confirmation_bundle",
                    "track": track["track_id"],
                    "source_bundle_id": source_bundle_id,
                    "task_bundle_sha256": task_bundle_sha,
                    "task_board_task_id": f"task-v2:{task_bundle_sha}",
                    "language": language,
                    "stratum": stratum_id,
                    "semantic_profile_id": profile_id,
                    "template_profile_id": template_profile_id,
                    "split": "pending",
                    "identities": _identity_leaves(task_sha, template_sha, config),
                    "expansion": _expansion(),
                    "cross_language_translation_pair_sha256": None,
                    "metadata_only": True,
                }
                cell.append(record)
                lookup[(language, stratum_id, profile_id)] = record
            if len(cell) != 6:
                _fail("identity_independent_cell_size_invalid")
            scored = sorted(
                cell,
                key=lambda item: _sha256(
                    f"{track['split_domain']}\0{item['task_bundle_sha256']}".encode()
                ),
            )
            eval_ids = {item["task_bundle_sha256"] for item in scored[:2]}
            for item in cell:
                item["split"] = (
                    "eval_proxy" if item["task_bundle_sha256"] in eval_ids else "train"
                )
                records.append(item)
    if (
        len(records) != 60
        or len({item["source_bundle_id"] for item in records}) != 60
        or len({item["task_bundle_sha256"] for item in records}) != 60
        or len({item["identities"]["task_semantic_sha256"] for item in records}) != 60
        or Counter(item["split"] for item in records)
        != Counter({"train": 40, "eval_proxy": 20})
    ):
        _fail("identity_independent_inventory_invalid")
    cells = Counter(
        (item["language"], item["stratum"], item["split"]) for item in records
    )
    for language in LANGUAGES:
        for stratum_id in strata:
            if (
                cells[(language, stratum_id, "train")] != 4
                or cells[(language, stratum_id, "eval_proxy")] != 2
            ):
                _fail("identity_independent_cell_split_invalid")
    return sorted(records, key=lambda item: item["task_bundle_sha256"]), lookup


def _make_factorial_records(
    config: Mapping[str, Any],
    discovery_records: Sequence[Mapping[str, Any]],
    independent_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _, strata, _ = _profile_maps(config)
    track = config["secondary_controlled_factorial"]
    quota_domain = str(track["within_factor_pair_assignment_domain"])
    rotation = int.from_bytes(
        hashlib.sha256(quota_domain.encode("utf-8")).digest()[:4], "big"
    ) % len(FACTORS)
    expected_omitted = [
        FACTORS[(stratum_index + language_index + rotation) % len(FACTORS)]
        for language_index, _language in enumerate(LANGUAGES)
        for stratum_index, _stratum in enumerate(strata)
    ]
    if (
        rotation != track["quota_rotation"]
        or list(track["omitted_eval_factor_sequence"]) != expected_omitted
    ):
        _fail("identity_factorial_quota_rotation_drift")
    groups = list(config["discovery_bridge"]["semantic_groups"])
    discovery_tasks_by_group: dict[str, str] = {}
    for group in groups:
        discovery_tasks_by_group[group["semantic_key"]] = _task_identity(
            group["descriptor"], config
        )
    old_template_sha = discovery_records[0]["identities"]["template_family_sha256"]
    old_tasks = set(discovery_tasks_by_group.values())
    old_pairs = {
        item["identities"]["task_template_pair_sha256"] for item in discovery_records
    }
    records: list[dict[str, Any]] = []
    cell_index = 0
    for language in LANGUAGES:
        for stratum_index, stratum_id in enumerate(strata):
            old_task_sha = discovery_tasks_by_group[
                groups[stratum_index]["semantic_key"]
            ]
            cell_records: list[dict[str, Any]] = []
            for profile_id in track["new_profile_ids_by_language"][language]:
                independent = independent_lookup[(language, stratum_id, profile_id)]
                new_task_sha = independent["identities"]["task_semantic_sha256"]
                new_template_sha = independent["identities"]["template_family_sha256"]
                match_key = _canonical_sha256(
                    {
                        "domain": "anchor.synthetic-scaffold-factorial-match-key.v1",
                        "new_task_semantic_sha256": new_task_sha,
                        "new_template_family_sha256": new_template_sha,
                    }
                )
                selections = {
                    "old_task_new_template": (old_task_sha, new_template_sha),
                    "new_task_old_template": (new_task_sha, old_template_sha),
                    "new_task_new_template": (new_task_sha, new_template_sha),
                }
                for factor in FACTORS:
                    task_sha, template_sha = selections[factor]
                    source_bundle_id = _source_bundle_identity(
                        "cf-source-v1",
                        "anchor.synthetic-scaffold-secondary-factorial-source-bundle.v1",
                        track["track_id"],
                        language,
                        stratum_id,
                        task_sha,
                        template_sha,
                        factor,
                    )
                    task_bundle_sha = _task_bundle_identity(
                        track["source_namespace"],
                        source_bundle_id,
                        language,
                        task_sha,
                        config,
                    )
                    pair_sha = _pair_identity(task_sha, template_sha, config)
                    record = {
                        "schema_version": RECORD_VERSION,
                        "record_kind": "secondary_factorial_bundle",
                        "track": track["track_id"],
                        "source_bundle_id": source_bundle_id,
                        "task_bundle_sha256": task_bundle_sha,
                        "task_board_task_id": f"task-v2:{task_bundle_sha}",
                        "language": language,
                        "stratum": stratum_id,
                        "factor": factor,
                        "factorial_match_key_sha256": match_key,
                        "split": "pending",
                        "identities": {
                            "task_semantic_sha256": task_sha,
                            "source_task_blueprint_sha256": task_sha,
                            "template_family_sha256": template_sha,
                            "task_template_pair_sha256": pair_sha,
                        },
                        "membership": {
                            "task_in_discovery": task_sha in old_tasks,
                            "template_in_discovery": template_sha == old_template_sha,
                            "pair_in_discovery": pair_sha in old_pairs,
                            "new_task_in_independent_inventory": factor
                            != "old_task_new_template",
                            "new_template_in_independent_inventory": factor
                            != "new_task_old_template",
                        },
                        "expansion": _expansion(),
                        "metadata_only": True,
                        "may_satisfy_independent_confirmation": False,
                    }
                    cell_records.append(record)
            omitted = track["omitted_eval_factor_sequence"][cell_index]
            for factor in FACTORS:
                factor_records = [
                    item for item in cell_records if item["factor"] == factor
                ]
                if len(factor_records) != 2:
                    _fail("identity_factorial_cell_factor_size_invalid")
                if factor == omitted:
                    for item in factor_records:
                        item["split"] = "train"
                else:
                    factor_records.sort(
                        key=lambda item: _sha256(
                            f"{track['within_factor_pair_assignment_domain']}\0{item['task_bundle_sha256']}".encode()
                        )
                    )
                    factor_records[0]["split"] = "eval_proxy"
                    factor_records[1]["split"] = "train"
            if Counter(item["split"] for item in cell_records) != Counter(
                {"train": 4, "eval_proxy": 2}
            ):
                _fail("identity_factorial_cell_split_invalid")
            records.extend(cell_records)
            cell_index += 1
    if (
        len(records) != 60
        or len({item["source_bundle_id"] for item in records}) != 60
        or len({item["task_bundle_sha256"] for item in records}) != 60
        or Counter(item["factor"] for item in records)
        != Counter({factor: 20 for factor in FACTORS})
        or Counter(item["split"] for item in records)
        != Counter({"train": 40, "eval_proxy": 20})
    ):
        _fail("identity_factorial_inventory_invalid")
    train_quotas = Counter(
        item["factor"] for item in records if item["split"] == "train"
    )
    eval_quotas = Counter(
        item["factor"] for item in records if item["split"] == "eval_proxy"
    )
    if dict(train_quotas) != dict(track["train_factor_quotas"]) or dict(
        eval_quotas
    ) != dict(track["eval_factor_quotas"]):
        _fail("identity_factorial_quota_invalid")
    return sorted(records, key=lambda item: item["task_bundle_sha256"])


def _scope_sets(
    discovery: Sequence[Mapping[str, Any]],
    independent: Sequence[Mapping[str, Any]],
    factorial: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    for scope, records in zip(SCOPES, (discovery, independent, factorial), strict=True):
        result[scope] = {
            "task": {item["identities"]["task_semantic_sha256"] for item in records},
            "template": {
                item["identities"]["template_family_sha256"] for item in records
            },
            "pair": {
                item["identities"]["task_template_pair_sha256"] for item in records
            },
        }
    return result


def _inventory_summary(scope: str, values: Mapping[str, set[str]]) -> Mapping[str, Any]:
    return {
        kind: {
            "count": len(values[kind]),
            "sha256": _inventory_sha256(
                f"anchor.synthetic-scaffold-{scope}-{kind}-inventory.v1", values[kind]
            ),
        }
        for kind in ("task", "template", "pair")
    }


def _intersection_summary(domain: str, values: set[str]) -> Mapping[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "ids": ordered,
        "sha256": _inventory_sha256(domain, ordered),
        "zero_overlap": not ordered,
    }


def _make_proofs(
    config: Mapping[str, Any],
    discovery: Sequence[Mapping[str, Any]],
    independent: Sequence[Mapping[str, Any]],
    factorial: Sequence[Mapping[str, Any]],
    discovery_context: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, set[str]]]]:
    sets = _scope_sets(discovery, independent, factorial)
    domains = {
        "task": config["identity_contract"]["task_domain"],
        "template": config["identity_contract"]["template_domain"],
        "pair": config["identity_contract"]["pair_domain"],
        "namespace_language_source_fields_excluded": True,
    }
    discovery_train_tasks = {
        item["identities"]["task_semantic_sha256"]
        for item in discovery
        if item["split"] == "train"
    }
    discovery_eval_tasks = {
        item["identities"]["task_semantic_sha256"]
        for item in discovery
        if item["split"] == "eval_proxy"
    }
    independent_proof = {
        "schema_version": PROOF_VERSION,
        "proof_kind": "discovery_vs_independent_confirmation",
        "status": "descriptor_level_zero_overlap_proven_execution_not_authorized",
        "domains": domains,
        "discovery": {
            "source_bundle_views": 10,
            "unique_semantics": 5,
            "bilingual_groups": 5,
            "inventories": _inventory_summary("discovery-bridge", sets[SCOPES[0]]),
        },
        "independent_confirmation": {
            "bundle_views": 60,
            "unique_semantics": 60,
            "cross_language_translation_pairs": 0,
            "inventories": _inventory_summary(
                "producer-independent-confirmation", sets[SCOPES[1]]
            ),
        },
        "intersections": {
            kind: _intersection_summary(
                f"anchor.synthetic-scaffold-discovery-independent-{kind}-intersection.v1",
                sets[SCOPES[0]][kind] & sets[SCOPES[1]][kind],
            )
            for kind in ("task", "template", "pair")
        },
        "historical_discovery_semantic_split": {
            "historical_split_rewritten": False,
            "train_unique_semantics": len(discovery_train_tasks),
            "eval_proxy_unique_semantics": len(discovery_eval_tasks),
            "intersection": _intersection_summary(
                "anchor.synthetic-scaffold-discovery-semantic-split-intersection.v1",
                discovery_train_tasks & discovery_eval_tasks,
            ),
            "bundle_generalization_supported": False,
        },
        "proof_algorithm": "recompute_leaves_common_domains_sorted_unique_inventory_then_exact_set_intersection_v1",
        "claims": {
            "real_world_semantic_disjointness_claimed": False,
            "automatic_translation_equivalence_proven": False,
            "independent_confirmation_executed": False,
            "bundle_generalization_validated": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
        },
    }
    if any(
        independent_proof["intersections"][kind]["count"]
        for kind in ("task", "template", "pair")
    ):
        _fail("identity_independent_overlap_nonzero")
    if (
        independent_proof["historical_discovery_semantic_split"]["intersection"]["ids"]
        != discovery_context["historical_semantic_intersection"]
    ):
        _fail("identity_historical_intersection_proof_mismatch")

    discovery_tasks = sets[SCOPES[0]]["task"]
    discovery_templates = sets[SCOPES[0]]["template"]
    discovery_pairs = sets[SCOPES[0]]["pair"]
    truth_table: list[dict[str, Any]] = []
    expected = {
        "old_task_new_template": (True, False),
        "new_task_old_template": (False, True),
        "new_task_new_template": (False, False),
    }
    for factor in FACTORS:
        items = [item for item in factorial if item["factor"] == factor]
        task_membership, template_membership = expected[factor]
        passed = all(
            (item["identities"]["task_semantic_sha256"] in discovery_tasks)
            == task_membership
            and (item["identities"]["template_family_sha256"] in discovery_templates)
            == template_membership
            and item["identities"]["task_template_pair_sha256"] not in discovery_pairs
            and item["membership"]["pair_in_discovery"] is False
            for item in items
        )
        truth_table.append(
            {
                "factor": factor,
                "bundle_count": len(items),
                "task_in_discovery": task_membership,
                "template_in_discovery": template_membership,
                "pair_in_discovery": False,
                "observed_pass": passed,
            }
        )
    match_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in factorial:
        match_groups[item["factorial_match_key_sha256"]].append(item)
    all_match_keys_complete = len(match_groups) == 20 and all(
        {item["factor"] for item in items} == set(FACTORS)
        for items in match_groups.values()
    )
    new_task_old = {
        item["identities"]["task_semantic_sha256"]
        for item in factorial
        if item["factor"] == "new_task_old_template"
    }
    new_task_new = {
        item["identities"]["task_semantic_sha256"]
        for item in factorial
        if item["factor"] == "new_task_new_template"
    }
    new_template_old_task = {
        item["identities"]["template_family_sha256"]
        for item in factorial
        if item["factor"] == "old_task_new_template"
    }
    new_template_new_task = {
        item["identities"]["template_family_sha256"]
        for item in factorial
        if item["factor"] == "new_task_new_template"
    }
    train_quota = Counter(
        item["factor"] for item in factorial if item["split"] == "train"
    )
    eval_quota = Counter(
        item["factor"] for item in factorial if item["split"] == "eval_proxy"
    )
    cells: list[dict[str, Any]] = []
    for language in LANGUAGES:
        for stratum in config["semantic_ontology"]["information_flow_strata"]:
            items = [
                item
                for item in factorial
                if item["language"] == language
                and item["stratum"] == stratum["stratum_id"]
            ]
            cells.append(
                {
                    "language": language,
                    "stratum": stratum["stratum_id"],
                    "bundles": len(items),
                    "train": sum(item["split"] == "train" for item in items),
                    "eval_proxy": sum(item["split"] == "eval_proxy" for item in items),
                }
            )
    factorial_proof = {
        "schema_version": PROOF_VERSION,
        "proof_kind": "secondary_controlled_factorial_membership",
        "status": "membership_truth_table_proven_not_independent_confirmation",
        "domains": domains,
        "counts": {
            "bundles": len(factorial),
            "train": sum(item["split"] == "train" for item in factorial),
            "eval_proxy": sum(item["split"] == "eval_proxy" for item in factorial),
            "language_stratum_cells": len(cells),
            "factorial_match_keys": len(match_groups),
        },
        "factor_counts": dict(Counter(item["factor"] for item in factorial)),
        "split_factor_quotas": {
            "train": dict(train_quota),
            "eval_proxy": dict(eval_quota),
        },
        "language_stratum_cells": cells,
        "truth_table": truth_table,
        "matched_factor_controls": {
            "new_task_inventory_equal_between_old_template_and_new_template_cells": new_task_old
            == new_task_new,
            "new_template_inventory_equal_between_old_task_and_new_task_cells": new_template_old_task
            == new_template_new_task,
            "each_match_key_has_all_three_factors": all_match_keys_complete,
        },
        "global_discovery_intersections": {
            "task": _intersection_summary(
                "anchor.synthetic-scaffold-factorial-discovery-task-intersection.v1",
                sets[SCOPES[2]]["task"] & discovery_tasks,
            ),
            "template": _intersection_summary(
                "anchor.synthetic-scaffold-factorial-discovery-template-intersection.v1",
                sets[SCOPES[2]]["template"] & discovery_templates,
            ),
            "pair": _intersection_summary(
                "anchor.synthetic-scaffold-factorial-discovery-pair-intersection.v1",
                sets[SCOPES[2]]["pair"] & discovery_pairs,
            ),
            "global_task_zero_overlap_required": False,
            "global_template_zero_overlap_required": False,
            "global_pair_zero_overlap_required": True,
        },
        "pair_inventory": _inventory_summary(
            "secondary-controlled-factorial-probe", sets[SCOPES[2]]
        )["pair"],
        "proof_algorithm": "recompute_task_template_pair_membership_per_bundle_then_exact_truth_table_and_quota_audit_v1",
        "claims": {
            "real_world_semantic_disjointness_claimed": False,
            "may_satisfy_independent_confirmation": False,
            "controlled_factorial_executed": False,
            "bundle_generalization_validated": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
        },
    }
    if (
        not all(item["observed_pass"] for item in truth_table)
        or not all(factorial_proof["matched_factor_controls"].values())
        or factorial_proof["global_discovery_intersections"]["pair"]["count"] != 0
        or factorial_proof["global_discovery_intersections"]["task"]["count"] != 5
        or factorial_proof["global_discovery_intersections"]["template"]["count"] != 1
    ):
        _fail("identity_factorial_proof_invalid")
    return independent_proof, factorial_proof, sets


def _make_inventory_records(
    sets: Mapping[str, Mapping[str, set[str]]],
) -> Mapping[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for identity_kind, short in zip(
        IDENTITY_KINDS, ("task", "template", "pair"), strict=True
    ):
        membership: dict[str, list[str]] = defaultdict(list)
        for scope in SCOPES:
            for value in sets[scope][short]:
                membership[value].append(scope)
        result[identity_kind] = [
            {
                "schema_version": RECORD_VERSION,
                "record_kind": "identity_inventory_leaf",
                "identity_kind": identity_kind,
                "identity_sha256": value,
                "scope_memberships": [
                    scope for scope in SCOPES if scope in membership[value]
                ],
            }
            for value in sorted(membership)
        ]
    return result


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(item, newline=True) for item in records)


def _partition_entry(path: str, raw: bytes, records: int) -> Mapping[str, Any]:
    return {
        "path": path,
        "media_type": "application/x-ndjson"
        if path.endswith(".jsonl")
        else "application/json",
        "records": records,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _expected_materialization(
    repo_root: Path,
    source_root: Path,
    snapshots: Mapping[str, _Snapshot],
    source_snapshots: Mapping[str, _Snapshot],
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    record_schema: Mapping[str, Any],
    proof_schema: Mapping[str, Any],
    manifest_schema: Mapping[str, Any],
) -> tuple[dict[str, bytes], Mapping[str, Any]]:
    discovery, discovery_context = _make_discovery_records(
        config, source_config, source_manifest
    )
    independent, independent_lookup = _make_independent_records(config)
    factorial = _make_factorial_records(config, discovery, independent_lookup)
    independent_proof, factorial_proof, sets = _make_proofs(
        config, discovery, independent, factorial, discovery_context
    )
    inventory_records = _make_inventory_records(sets)
    for record in [
        *discovery,
        *independent,
        *factorial,
        *inventory_records["task_semantic"],
        *inventory_records["template_family"],
        *inventory_records["task_template_pair"],
    ]:
        _validate_schema(
            record_schema, record, "identity_record_schema_validation_failed"
        )
    _validate_schema(
        proof_schema, independent_proof, "identity_proof_schema_validation_failed"
    )
    _validate_schema(
        proof_schema, factorial_proof, "identity_proof_schema_validation_failed"
    )

    partitions: dict[str, bytes] = {
        "discovery/views.jsonl": _jsonl_bytes(discovery),
        "independent_confirmation/bundles.jsonl": _jsonl_bytes(independent),
        "secondary_factorial/bundles.jsonl": _jsonl_bytes(factorial),
        "inventories/task_semantic_ids.jsonl": _jsonl_bytes(
            inventory_records["task_semantic"]
        ),
        "inventories/template_family_ids.jsonl": _jsonl_bytes(
            inventory_records["template_family"]
        ),
        "inventories/task_template_pair_ids.jsonl": _jsonl_bytes(
            inventory_records["task_template_pair"]
        ),
        "proofs/discovery_vs_independent.json": _canonical_json_bytes(
            independent_proof, newline=True
        ),
        "proofs/secondary_factorial.json": _canonical_json_bytes(
            factorial_proof, newline=True
        ),
    }
    partition_counts = {
        "discovery/views.jsonl": len(discovery),
        "independent_confirmation/bundles.jsonl": len(independent),
        "secondary_factorial/bundles.jsonl": len(factorial),
        "inventories/task_semantic_ids.jsonl": len(inventory_records["task_semantic"]),
        "inventories/template_family_ids.jsonl": len(
            inventory_records["template_family"]
        ),
        "inventories/task_template_pair_ids.jsonl": len(
            inventory_records["task_template_pair"]
        ),
        "proofs/discovery_vs_independent.json": 1,
        "proofs/secondary_factorial.json": 1,
    }
    producer_read_set = [
        _artifact_descriptor(repo_root, snapshots[role])
        for role in (
            "config",
            "config_schema",
            "record_schema",
            "proof_schema",
            "manifest_schema",
            "implementation",
        )
    ]
    source_read_set = [
        _artifact_descriptor(source_root, source_snapshots[role])
        for role in ("config", "manifest", "manifest_sidecar")
    ]
    consumer_contract_read_set = [
        _artifact_descriptor(source_root, source_snapshots["consumer_plan_config"])
    ]
    ordered_read_set = (
        [f"producer:{item['path']}:{item['sha256']}" for item in producer_read_set]
        + [f"source:{item['path']}:{item['sha256']}" for item in source_read_set]
        + [
            f"consumer-contract:{item['path']}:{item['sha256']}"
            for item in consumer_contract_read_set
        ]
    )
    binding = config["source_binding"]
    git_metadata_reads = [
        "git:rev-parse:show-toplevel",
        "git:rev-parse:absolute-git-dir",
        "git:filesystem-check:info/grafts-empty-or-absent",
        "git:for-each-ref:refs/replace-empty",
        f"git:commit:{binding['source_commit']}",
        *[
            f"git:blob:{binding['source_commit']}:{binding[role]['path']}"
            for role in ("config", "manifest", "manifest_sidecar")
        ],
        f"git:commit:{binding['consumer_plan_commit']}",
        (
            f"git:blob:{binding['consumer_plan_commit']}:"
            f"{binding['consumer_plan_config']['path']}"
        ),
        (
            f"git:merge-base-is-ancestor:{binding['source_commit']}:"
            f"{binding['consumer_plan_commit']}"
        ),
    ]
    descriptor_schema_sha256, ontology_sha256, atom_catalog_sha256 = (
        _descriptor_context(config)
    )
    logical = {
        scope: _inventory_summary(scope.replace("_", "-"), sets[scope])
        for scope in SCOPES
    }
    union_sets = {
        kind: set().union(*(sets[scope][kind] for scope in SCOPES))
        for kind in ("task", "template", "pair")
    }
    logical["union"] = _inventory_summary("union", union_sets)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "status": "metadata_identity_assets_ready_execution_and_training_blocked",
        "claim_scope": CLAIM_SCOPE,
        "producer": {
            "producer_version": PRODUCER_VERSION,
            "config": _artifact_descriptor(repo_root, snapshots["config"]),
            "config_schema": _artifact_descriptor(
                repo_root, snapshots["config_schema"]
            ),
            "record_schema": _artifact_descriptor(
                repo_root, snapshots["record_schema"]
            ),
            "proof_schema": _artifact_descriptor(repo_root, snapshots["proof_schema"]),
            "manifest_schema": _artifact_descriptor(
                repo_root, snapshots["manifest_schema"]
            ),
            "implementation": _artifact_descriptor(
                repo_root, snapshots["implementation"]
            ),
            "canonical_json_policy": "utf8_sort_keys_compact_reject_nan_no_normalization_lf_for_documents_v1",
            "atomic_publish_no_replace": True,
            "final_toctou_recheck": True,
        },
        "source": {
            "repository": binding["repository"],
            "source_commit": binding["source_commit"],
            "source_parent_commit": binding["source_parent_commit"],
            "source_tree": binding["source_tree"],
            "consumer_plan_commit": binding["consumer_plan_commit"],
            "consumer_plan_parent_commit": binding["consumer_plan_parent_commit"],
            "consumer_plan_tree": binding["consumer_plan_tree"],
            "git_objects_authenticated": True,
            "config": source_read_set[0],
            "manifest": source_read_set[1],
            "manifest_sidecar": source_read_set[2],
            "consumer_plan_config": consumer_contract_read_set[0],
            "generator_implementation_sha256": binding[
                "generator_implementation_sha256"
            ],
            "closed_grammar_sha256": binding["closed_grammar_sha256"],
            "source_bundle_inventory_sha256": binding["source_bundle_inventory_sha256"],
            "source_bundle_view_count": 10,
            "source_record_count": 100,
        },
        "counts": {
            "discovery_views": 10,
            "discovery_unique_semantics": 5,
            "independent_confirmation_bundles": 60,
            "secondary_factorial_bundles": 60,
            "metadata_bundle_rows": 130,
            "records_if_both_tracks_materialized": 1200,
            "languages": 2,
            "strata": 5,
            "roles": 5,
            "variants": 2,
            "protected_body_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
        "identity_contract": {
            "task_domain": config["identity_contract"]["task_domain"],
            "template_domain": config["identity_contract"]["template_domain"],
            "pair_domain": config["identity_contract"]["pair_domain"],
            "task_bundle_domain": config["identity_contract"]["task_bundle_domain"],
            "source_task_blueprint_is_task_semantic_alias": True,
            "ontology_sha256": ontology_sha256,
            "descriptor_schema_sha256": descriptor_schema_sha256,
            "descriptor_atom_catalog_schema_version": (DESCRIPTOR_ATOM_CATALOG_VERSION),
            "descriptor_atom_catalog_count": DESCRIPTOR_ATOM_CATALOG_COUNT,
            "descriptor_atom_catalog_sha256": atom_catalog_sha256,
            "pair_recomputed_not_self_reported": True,
            "namespace_neutral_proof_identity_kinds": [
                "task_semantic",
                "template_family",
                "task_template_pair",
            ],
        },
        "tracks": {
            "discovery_bridge": {
                "status": "producer_curated_bridge_historical_semantic_split_overlap_present",
                "bundles": 10,
                "train": 8,
                "eval_proxy": 2,
                "expected_records_if_materialized": 100,
                "materialized": False,
                "eval_proxy_is_heldout": False,
            },
            "producer_independent_confirmation": {
                "status": "descriptor_identities_and_zero_overlap_proof_ready_execution_blocked",
                "bundles": 60,
                "train": 40,
                "eval_proxy": 20,
                "expected_records_if_materialized": 600,
                "materialized": False,
                "eval_proxy_is_heldout": False,
            },
            "secondary_controlled_factorial_probe": {
                "status": "membership_truth_table_ready_not_independent_confirmation",
                "bundles": 60,
                "train": 40,
                "eval_proxy": 20,
                "expected_records_if_materialized": 600,
                "materialized": False,
                "eval_proxy_is_heldout": False,
            },
        },
        "partitions": [
            _partition_entry(path, partitions[path], partition_counts[path])
            for path in PARTITION_PATHS
        ],
        "logical_inventories": logical,
        "proofs": [
            {
                "proof_kind": "discovery_vs_independent_confirmation",
                "path": "proofs/discovery_vs_independent.json",
                "bytes": len(partitions["proofs/discovery_vs_independent.json"]),
                "sha256": _sha256(partitions["proofs/discovery_vs_independent.json"]),
                "passed": True,
            },
            {
                "proof_kind": "secondary_controlled_factorial_membership",
                "path": "proofs/secondary_factorial.json",
                "bytes": len(partitions["proofs/secondary_factorial.json"]),
                "sha256": _sha256(partitions["proofs/secondary_factorial.json"]),
                "passed": True,
            },
        ],
        "read_set": {
            "scope": "exact_semantic_file_read_set_plus_local_git_object_provenance",
            "producer_contract_artifacts": producer_read_set,
            "source_metadata_artifacts": source_read_set,
            "consumer_contract_artifacts": consumer_contract_read_set,
            "ordered_artifacts_sha256": _ordered_inventory_sha256(
                "anchor.synthetic-scaffold-independent-confirmation-read-set.v1",
                ordered_read_set,
            ),
            "git_metadata_reads": git_metadata_reads,
            "git_metadata_reads_sha256": _ordered_inventory_sha256(
                "anchor.synthetic-scaffold-independent-confirmation-git-read-set.v1",
                git_metadata_reads,
            ),
            "local_git_object_provenance_only": True,
            "source_git_commit_authenticated": True,
            "consumer_plan_git_commit_authenticated": True,
            "jsonl_inputs_read": 0,
            "protected_body_reads": 0,
        },
        "audit": {
            "config_schema_validated": True,
            "record_schema_validated": True,
            "proof_schema_validated": True,
            "manifest_schema_validated": True,
            "single_bytes_snapshot": True,
            "same_bytes_reparse": True,
            "final_toctou_recheck": True,
            "atomic_publish_no_replace": True,
            "exact_layout": True,
            "pair_leaves_recomputed": True,
            "discovery_bilingual_bridge_validated": True,
            "historical_discovery_semantic_split_intersection": 2,
            "independent_task_zero_overlap": True,
            "independent_pair_zero_overlap": True,
            "factorial_truth_table_validated": True,
            "factorial_pair_zero_overlap": True,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_body_reads": 0,
        },
        "claims": {
            "metadata_identity_promotes_execution": False,
            "descriptor_zero_overlap_proves_real_world_semantic_disjointness": False,
            "automatic_translation_equivalence_proven": False,
            "records_materialized": False,
            "independent_confirmation_executed": False,
            "controlled_factorial_executed": False,
            "multi_seed_validated": False,
            "bundle_generalization_validated": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
        },
    }
    _validate_schema(
        manifest_schema, manifest, "identity_manifest_schema_validation_failed"
    )
    return partitions, manifest


def _write_atomic_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    try:
        if os.name == "nt":
            os.rename(source, destination)
            return
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            try:
                renameat2 = libc.renameat2
            except AttributeError:
                _fail("identity_atomic_no_replace_unavailable")
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100, os.fsencode(source), -100, os.fsencode(destination), 1
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error in (errno.EEXIST, errno.ENOTEMPTY):
                _fail("identity_output_already_exists")
            raise OSError(error, os.strerror(error), str(destination))
        _fail("identity_atomic_no_replace_unavailable")
    except IdentityProducerError:
        raise
    except FileExistsError as exc:
        raise IdentityProducerError("identity_output_already_exists") from exc
    except OSError as exc:
        raise IdentityProducerError("identity_atomic_publish_failed") from exc


def _capture_artifact(artifact: Path) -> Mapping[str, _Snapshot]:
    if not artifact.is_dir() or _is_reparse_or_symlink(artifact):
        _fail("identity_artifact_directory_invalid")
    _assert_no_reparse_absolute_ancestry(artifact, "identity_artifact_ancestry_invalid")
    snapshots: dict[str, _Snapshot] = {}
    for relative in ARTIFACT_PATHS:
        snapshots[relative] = _Snapshot.capture(
            _resolve_within(artifact, relative, "identity_artifact_path_invalid")
        )
    expected_files = set(ARTIFACT_PATHS)
    expected_directories = {
        parent.as_posix()
        for relative in ARTIFACT_PATHS
        for parent in _safe_relative_path(relative).parents
        if parent.as_posix() != "."
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in artifact.rglob("*"):
        relative = path.relative_to(artifact).as_posix()
        if _is_reparse_or_symlink(path):
            _fail("identity_artifact_reparse_entry")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            _fail("identity_artifact_special_entry")
    if observed_files != expected_files or observed_directories != expected_directories:
        _fail("identity_artifact_layout_invalid")
    return snapshots


def _verify_snapshots(snapshots: Mapping[str, _Snapshot], code: str) -> None:
    for snapshot in snapshots.values():
        snapshot.assert_unchanged(code)


def _resolve_root(value: str | Path, code: str) -> Path:
    path = Path(value).absolute()
    _assert_no_reparse_absolute_ancestry(path, code)
    if not path.is_dir() or _is_reparse_or_symlink(path):
        _fail(code)
    return path.resolve(strict=True)


def _resolve_config(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    path = path.absolute()
    _assert_no_reparse_absolute_ancestry(path, "identity_config_path_invalid")
    if not path.is_file() or _is_reparse_or_symlink(path):
        _fail("identity_config_path_invalid")
    try:
        path.resolve(strict=True).relative_to(repo_root)
    except ValueError:
        _fail("identity_config_outside_repo")
    canonical = (repo_root / _safe_relative_path(CONFIG_PATH)).resolve(strict=True)
    if path.resolve(strict=True) != canonical:
        _fail("identity_noncanonical_config_forbidden")
    return path


def audit_identity_fixture(
    repo_root: str | Path,
    source_root: str | Path,
    config_path: str | Path,
    artifact_dir: str | Path,
) -> Mapping[str, Any]:
    root = _resolve_root(repo_root, "identity_repo_root_invalid")
    source = _resolve_root(source_root, "identity_source_root_invalid")
    config_file = _resolve_config(root, config_path)
    artifact = Path(artifact_dir)
    if not artifact.is_absolute():
        artifact = root / artifact
    artifact = artifact.absolute()
    snapshots, config, _, record_schema, proof_schema, manifest_schema = _load_contract(
        root, config_file
    )
    source_snapshots, source_config, source_manifest = _load_source(source, config)
    artifact_snapshots = _capture_artifact(artifact)
    expected_sidecar = (
        f"{artifact_snapshots['manifest.json'].sha256}  manifest.json\n".encode("ascii")
    )
    if artifact_snapshots["manifest.json.sha256"].data != expected_sidecar:
        _fail("identity_manifest_sidecar_invalid")
    manifest = _strict_json(
        artifact_snapshots["manifest.json"].data, "identity_manifest_json_invalid"
    )
    if (
        _strict_json(
            artifact_snapshots["manifest.json"].data,
            "identity_manifest_reparse_failed",
        )
        != manifest
    ):
        _fail("identity_manifest_reparse_mismatch")
    if (
        not isinstance(manifest, Mapping)
        or _canonical_json_bytes(manifest, newline=True)
        != artifact_snapshots["manifest.json"].data
    ):
        _fail("identity_manifest_not_canonical")
    _validate_schema(
        manifest_schema, manifest, "identity_manifest_schema_validation_failed"
    )
    for relative in PARTITION_PATHS:
        raw = artifact_snapshots[relative].data
        if relative.endswith(".jsonl"):
            records = _strict_jsonl(raw, "identity_partition_jsonl_invalid")
            if _strict_jsonl(raw, "identity_partition_jsonl_reparse_failed") != records:
                _fail("identity_partition_jsonl_reparse_mismatch")
            for record in records:
                _validate_schema(
                    record_schema, record, "identity_record_schema_validation_failed"
                )
        else:
            proof = _strict_json(raw, "identity_proof_json_invalid")
            if _strict_json(raw, "identity_proof_reparse_failed") != proof:
                _fail("identity_proof_reparse_mismatch")
            if (
                not isinstance(proof, Mapping)
                or _canonical_json_bytes(proof, newline=True) != raw
            ):
                _fail("identity_proof_not_canonical")
            _validate_schema(
                proof_schema, proof, "identity_proof_schema_validation_failed"
            )
    expected_partitions, expected_manifest = _expected_materialization(
        root,
        source,
        snapshots,
        source_snapshots,
        config,
        source_config,
        source_manifest,
        record_schema,
        proof_schema,
        manifest_schema,
    )
    for relative, expected in expected_partitions.items():
        if artifact_snapshots[relative].data != expected:
            _fail("identity_partition_materialization_mismatch")
    if manifest != expected_manifest:
        _fail("identity_manifest_materialization_mismatch")
    _verify_snapshots(snapshots, "identity_contract_toctou_detected")
    _verify_snapshots(source_snapshots, "identity_source_toctou_detected")
    _verify_snapshots(artifact_snapshots, "identity_artifact_toctou_detected")
    _authenticate_source_git(source, config["source_binding"], source_snapshots)
    _verify_snapshots(snapshots, "identity_contract_final_toctou_detected")
    _verify_snapshots(source_snapshots, "identity_source_final_toctou_detected")
    _verify_snapshots(artifact_snapshots, "identity_artifact_final_toctou_detected")
    return manifest


def build_identity_fixture(
    repo_root: str | Path,
    source_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    root = _resolve_root(repo_root, "identity_repo_root_invalid")
    source = _resolve_root(source_root, "identity_source_root_invalid")
    config_file = _resolve_config(root, config_path)
    snapshots, config, _, record_schema, proof_schema, manifest_schema = _load_contract(
        root, config_file
    )
    source_snapshots, source_config, source_manifest = _load_source(source, config)
    partitions, manifest = _expected_materialization(
        root,
        source,
        snapshots,
        source_snapshots,
        config,
        source_config,
        source_manifest,
        record_schema,
        proof_schema,
        manifest_schema,
    )
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_absolute_ancestry(
        output.parent, "identity_output_parent_invalid"
    )
    if _path_lexists(output):
        _fail("identity_output_already_exists")
    parent_before = output.parent.stat()
    parent_identity = (parent_before.st_dev, parent_before.st_ino)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative in PARTITION_PATHS:
            _write_atomic_file(
                temporary / _safe_relative_path(relative), partitions[relative]
            )
        manifest_raw = _canonical_json_bytes(manifest, newline=True)
        _write_atomic_file(temporary / "manifest.json", manifest_raw)
        _write_atomic_file(
            temporary / "manifest.json.sha256",
            f"{_sha256(manifest_raw)}  manifest.json\n".encode("ascii"),
        )
        audited = audit_identity_fixture(root, source, config_file, temporary)
        if audited != manifest:
            _fail("identity_prepublication_audit_mismatch")
        temporary_snapshots = _capture_artifact(temporary)
        temporary_identity = _stat_identity(temporary.stat())
        _verify_snapshots(snapshots, "identity_contract_changed_during_build")
        _verify_snapshots(source_snapshots, "identity_source_changed_during_build")
        parent_after = output.parent.stat()
        if (
            _path_lexists(output)
            or _is_reparse_or_symlink(output.parent)
            or (parent_after.st_dev, parent_after.st_ino) != parent_identity
            or _stat_identity(temporary.stat()) != temporary_identity
        ):
            _fail("identity_output_publish_race")
        _verify_snapshots(temporary_snapshots, "identity_temporary_artifact_changed")
        _rename_directory_no_replace(temporary, output)
        return audit_identity_fixture(root, source, config_file, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or audit metadata-only independent-confirmation identities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo-root", default=".")
        child.add_argument("--source-root", required=True)
        child.add_argument("--config", default=CONFIG_PATH)
        child.add_argument(
            "--artifact",
            default=CANONICAL_FIXTURE_PATH,
            help="Output directory for build or existing fixture for audit",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_identity_fixture(
            args.repo_root, args.source_root, args.config, args.artifact
        )
    else:
        manifest = audit_identity_fixture(
            args.repo_root, args.source_root, args.config, args.artifact
        )
    print(_canonical_json_bytes(manifest).decode("utf-8"))
    return 0


__all__ = [
    "CANONICAL_FIXTURE_PATH",
    "CONFIG_PATH",
    "IdentityProducerError",
    "audit_identity_fixture",
    "build_identity_fixture",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
