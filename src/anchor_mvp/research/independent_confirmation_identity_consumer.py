"""Strict additive consumer for independent-confirmation identity metadata.

This module authenticates the frozen Producer v1 metadata artifact, validates
all schemas and records from single byte snapshots, and independently
recomputes the identity inventories and both proof tracks.  It never reads a
protected dataset body and cannot authorize materialization, execution, or
training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import yaml


CONFIG_VERSION = "anchor.independent-confirmation-identity-consumer-config.v2"
DECISION_VERSION = "anchor.independent-confirmation-identity-consumer-decision.v2"
CONFIG_PATH = "configs/research/independent_confirmation_identity_consumer_v2.yaml"
CONFIG_SHA256 = "e69fa162dfc03e5c92168e1c529630a42b929027400870305829cad6877ea723"
DECISION_SCHEMA_PATH = (
    "configs/research/independent_confirmation_identity_decision_v2.schema.json"
)

_ROOT = Path(__file__).resolve().parents[3]
_MAX_BYTES = 2 * 1024 * 1024
_REPARSE_POINT = 0x0400
_SCOPES = (
    "discovery_bridge",
    "producer_independent_confirmation",
    "secondary_controlled_factorial_probe",
)
_KINDS = ("task", "template", "pair")
_IDENTITY_KIND = {
    "task": "task_semantic",
    "template": "template_family",
    "pair": "task_template_pair",
}
_FACTORS = (
    "old_task_new_template",
    "new_task_old_template",
    "new_task_new_template",
)
_ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
_VARIANTS = ("json_only", "concise_rationale_plus_json")
_PRODUCER_COMMIT = "09a6829084f76790e3488cb999a6755cd4d5f95e"
_DESCRIPTOR_SCHEMA_CONTRACT: Mapping[str, Any] = {
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
_EXPECTED_PATHS = {
    "producer_config": (
        "configs/research/synthetic_scaffold_independent_confirmation_identity_v1.json"
    ),
    "producer_config_schema": (
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_config.schema.json"
    ),
    "producer_record_schema": (
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_record.schema.json"
    ),
    "producer_proof_schema": (
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_proof.schema.json"
    ),
    "producer_manifest_schema": (
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_manifest.schema.json"
    ),
    "producer_implementation": (
        "src/anchor_mvp/swebench/"
        "synthetic_scaffold_independent_confirmation_identity.py"
    ),
    "producer_test": (
        "tests/test_synthetic_scaffold_independent_confirmation_identity_v1.py"
    ),
    "producer_docs_en": (
        "docs/synthetic_scaffold_independent_confirmation_identity_v1.md"
    ),
    "producer_docs_zh_cn": (
        "docs/synthetic_scaffold_independent_confirmation_identity_v1.zh-CN.md"
    ),
    "producer_fixture": (
        "fixtures/research/synthetic_scaffold_independent_confirmation_identity_v1"
    ),
    "decision_schema": DECISION_SCHEMA_PATH,
}
_EXPECTED_BINDINGS = {
    "producer_config_sha256": (
        "4c9017c1ce0fa2c82168b03c0e157fa7c6251530c36bd3dbb1bd769f01b63ae3"
    ),
    "producer_config_schema_sha256": (
        "2bb127dd674bb767bac7159b60c23933ad9f5658e45379d6e6a419e5b3b49dfb"
    ),
    "producer_record_schema_sha256": (
        "53fee396b017eea9e3a6c9e6ad3faf2c36e6fcdf20f132ef22d5ea670a0a5f9a"
    ),
    "producer_proof_schema_sha256": (
        "273307c9e4c61c7ab2e4a9065020c005dac7509fc6f2d2f76969403bedd7226c"
    ),
    "producer_manifest_schema_sha256": (
        "4d85dd0c915302353a245c847af5229c9ade658fd1f62ce0841301be6b978289"
    ),
    "producer_implementation_sha256": (
        "8819a10923b4525acd53e255373a91cacd51fb921629241a955b09eed59d3952"
    ),
    "producer_test_sha256": (
        "94c4bca49e99d6359ede8c66c84b49292044aaaaa4b05acb99810312493dffb3"
    ),
    "producer_docs_en_sha256": (
        "362e0ad4266bd102cdbc14d70bc6af72db0d730ddd283a5323d6c6393fcadbdb"
    ),
    "producer_docs_zh_cn_sha256": (
        "0ffdf029bec0c898828ac7fd16f2cd80e490353dee517d722445e3d1f0013f21"
    ),
    "producer_manifest_sha256": (
        "1197abf22e3b19ee96eb9060cefa23e10b57971ca60358fe5bc9aefea02d75f6"
    ),
    "producer_manifest_sidecar_physical_sha256": (
        "434770a021fe45ee8bf469d5211156cf89d99cbbc741d0fae05e1647b0ade047"
    ),
    "decision_schema_sha256": (
        "84c9ebcbdf80b9ccb9e1a2f8c875642777bcef32b5ff6ead99a3059401ec609e"
    ),
}
_EXPECTED_PARTITIONS = {
    "discovery/views.jsonl": (
        10,
        18643,
        "e56cc7cc3396f2c0143e131bfd5333b24d9836749148aa642c65cf25f33d02dc",
    ),
    "independent_confirmation/bundles.jsonl": (
        60,
        82270,
        "b0fded75875a7b098d5a71a26bdd701aa23926759eb4ea0d35a2f1f08567ef43",
    ),
    "secondary_factorial/bundles.jsonl": (
        60,
        94990,
        "215f92f3a7319a91f86dcec68ad534c2580a459d74f641ba0003ac93428e084d",
    ),
    "inventories/task_semantic_ids.jsonl": (
        65,
        20780,
        "19d48c73d2b6886a710e057d97b99836d7e02efdf07674cd7e2a8615795f7eb8",
    ),
    "inventories/template_family_ids.jsonl": (
        21,
        7270,
        "c0efde5608e65346dcad6d7ad1ad7efbe98e4523f8545027079a4deaac9831e5",
    ),
    "inventories/task_template_pair_ids.jsonl": (
        105,
        33470,
        "f7c97423f8d9396e1f300bd278cb9be033961ec5cbfd905f145a9882ce615fea",
    ),
    "proofs/discovery_vs_independent.json": (
        1,
        2462,
        "dbaab54dc1d01b8b6697b84558b3805ecad06b4bdffe3e5340d4c8d41b09048a",
    ),
    "proofs/secondary_factorial.json": (
        1,
        4081,
        "2494be797028ac1246748e7ec34084063eb9886284ac0d4146d8905dbfa12d2f",
    ),
}
_ZERO_CLAIMS = {
    "records_materialized": False,
    "protected_source_disjoint": False,
    "independent_confirmation_executed": False,
    "controlled_factorial_executed": False,
    "quality_validated": False,
    "generalization_validated": False,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal": False,
}
_ZERO_RESOURCE_AUDIT = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "protected_body_reads": 0,
}


class IndependentConfirmationIdentityConsumerError(RuntimeError):
    """Stable fail-closed consumer error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise IndependentConfirmationIdentityConsumerError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _inventory_sha256(domain: str, values: Iterable[str]) -> str:
    return _canonical_sha256({"domain": domain, "values": sorted(set(values))})


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE_POINT
    )


def _safe_path(root: Path, relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(code)
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        _fail(code)
    current = root
    if _is_link_or_reparse(current):
        _fail(code)
    for part in candidate.parts:
        current /= part
        if current.exists() and _is_link_or_reparse(current):
            _fail(code)
    try:
        current.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        _fail(code)
    return current


def _assert_no_reparse_ancestry(root: Path, path: Path, code: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(code)
    current = root
    if _is_link_or_reparse(current):
        _fail(code)
    for part in relative.parts:
        current /= part
        if _is_link_or_reparse(current):
            _fail(code)


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, root: Path, code: str) -> None:
        current = _read_snapshot(root, self.path, code)
        if (
            current.data != self.data
            or current.sha256 != self.sha256
            or current.identity != self.identity
        ):
            _fail(code)


def _read_snapshot(root: Path, path: Path, code: str) -> _Snapshot:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        _fail(code)
    _assert_no_reparse_ancestry(root, path, code)
    try:
        if not path.is_file() or _is_link_or_reparse(path):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > _MAX_BYTES:
                _fail(code)
            data = handle.read(_MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except IndependentConfirmationIdentityConsumerError:
        raise
    except OSError as exc:
        raise IndependentConfirmationIdentityConsumerError(code) from exc
    identity = _stat_identity(after)
    if (
        len(data) > _MAX_BYTES
        or len(data) != after.st_size
        or _stat_identity(before) != identity
        or _stat_identity(final) != identity
        or _is_link_or_reparse(path)
    ):
        _fail(code)
    return _Snapshot(path=path, data=data, sha256=_sha256(data), identity=identity)


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "LC_ALL": "C",
        }
    )
    return environment


def _git(repo: Path, arguments: Sequence[str], code: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            env=_git_environment(),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IndependentConfirmationIdentityConsumerError(code) from exc
    if result.returncode != 0:
        _fail(code)
    return result.stdout


def _authenticate_producer_git(
    repo: Path,
    snapshots: Mapping[str, _Snapshot],
    fixture_relative: str,
) -> None:
    git_dir_raw = _git(repo, ["rev-parse", "--absolute-git-dir"], "git_invalid")
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise IndependentConfirmationIdentityConsumerError("git_invalid") from exc
    grafts = git_dir / "info" / "grafts"
    if grafts.exists() and grafts.read_bytes().strip():
        _fail("git_grafts_forbidden")
    if _git(
        repo,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        "git_replace_refs_invalid",
    ).strip():
        _fail("git_replace_refs_forbidden")
    resolved = (
        _git(
            repo,
            ["rev-parse", "--verify", f"{_PRODUCER_COMMIT}^{{commit}}"],
            "producer_commit_unavailable",
        )
        .decode("ascii")
        .strip()
    )
    if resolved != _PRODUCER_COMMIT:
        _fail("producer_commit_mismatch")
    role_paths = {
        "config": _EXPECTED_PATHS["producer_config"],
        "config_schema": _EXPECTED_PATHS["producer_config_schema"],
        "record_schema": _EXPECTED_PATHS["producer_record_schema"],
        "proof_schema": _EXPECTED_PATHS["producer_proof_schema"],
        "manifest_schema": _EXPECTED_PATHS["producer_manifest_schema"],
        "implementation": _EXPECTED_PATHS["producer_implementation"],
        "producer_test": _EXPECTED_PATHS["producer_test"],
        "producer_docs_en": _EXPECTED_PATHS["producer_docs_en"],
        "producer_docs_zh_cn": _EXPECTED_PATHS["producer_docs_zh_cn"],
        "manifest": f"{fixture_relative}/manifest.json",
        "manifest_sidecar": f"{fixture_relative}/manifest.json.sha256",
        **{
            f"partition:{relative}": f"{fixture_relative}/{relative}"
            for relative in _EXPECTED_PARTITIONS
        },
    }
    if len(role_paths) != 19:
        _fail("producer_git_inventory_invalid")
    for role, relative in role_paths.items():
        tree_entry = _git(
            repo,
            ["ls-tree", _PRODUCER_COMMIT, "--", relative],
            "producer_git_tree_invalid",
        ).decode("utf-8")
        if not tree_entry.startswith("100644 blob ") or not tree_entry.endswith(
            f"\t{relative}\n"
        ):
            _fail("producer_git_tree_invalid")
        raw = _git(
            repo,
            ["cat-file", "blob", f"{_PRODUCER_COMMIT}:{relative}"],
            "producer_git_blob_unavailable",
        )
        if raw != snapshots[role].data or _sha256(raw) != snapshots[role].sha256:
            _fail("producer_git_blob_mismatch")


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            _fail("config_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(code)
    return value


def _load_yaml(data: bytes) -> Mapping[str, Any]:
    if b"\r" in data or data.startswith(b"\xef\xbb\xbf"):
        _fail("config_invalid")
    try:
        value = yaml.load(data.decode("utf-8"), Loader=_UniqueLoader)
    except IndependentConfirmationIdentityConsumerError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise IndependentConfirmationIdentityConsumerError("config_invalid") from exc
    return _mapping(value, "config_invalid")


def _strict_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key")
        result[key] = value
    return result


def _load_json(data: bytes, code: str) -> Any:
    if b"\r" in data or data.startswith(b"\xef\xbb\xbf"):
        _fail(code)
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: _fail("json_non_finite_number"),
        )
    except IndependentConfirmationIdentityConsumerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndependentConfirmationIdentityConsumerError(code) from exc


def _load_json_mapping(data: bytes, code: str) -> Mapping[str, Any]:
    return _mapping(_load_json(data, code), code)


def _load_jsonl(data: bytes, code: str) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        _fail(code)
    rows: list[Mapping[str, Any]] = []
    for line in data.splitlines(keepends=True):
        if line == b"\n" or not line.endswith(b"\n"):
            _fail(code)
        row = _load_json_mapping(line[:-1], code)
        if _canonical_bytes(row, newline=True) != line:
            _fail("partition_noncanonical")
        rows.append(row)
    return rows


def _reject_external_refs(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                if not isinstance(child, str) or not child.startswith("#/"):
                    _fail("schema_external_reference_forbidden")
            _reject_external_refs(child)
    elif isinstance(value, list):
        for child in value:
            _reject_external_refs(child)


def _validator(schema: Mapping[str, Any], code: str) -> Any:
    _reject_external_refs(schema)
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)
    except Exception as exc:
        raise IndependentConfirmationIdentityConsumerError(code) from exc


def _validate(validator: Any, value: object, code: str) -> None:
    errors = sorted(
        validator.iter_errors(value), key=lambda item: list(item.absolute_path)
    )
    if errors:
        _fail(code)


def _exact(value: Mapping[str, Any], fields: set[str], code: str) -> None:
    if set(value) != fields:
        _fail(code)


def _validate_consumer_config(config: Mapping[str, Any]) -> None:
    _exact(
        config,
        {
            "schema_version",
            "claim_scope",
            "paths",
            "producer_provenance",
            "bindings",
            "partitions",
            "expected",
            "policy",
            "audit",
        },
        "consumer_config_fields_invalid",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "additive_metadata_identity_overlay_non_authorizing"
        or dict(_mapping(config.get("paths"), "consumer_paths_invalid"))
        != _EXPECTED_PATHS
        or dict(_mapping(config.get("bindings"), "consumer_bindings_invalid"))
        != _EXPECTED_BINDINGS
        or dict(_mapping(config.get("audit"), "consumer_audit_invalid"))
        != _ZERO_RESOURCE_AUDIT
    ):
        _fail("consumer_config_invalid")
    provenance = _mapping(
        config.get("producer_provenance"), "consumer_provenance_invalid"
    )
    if dict(provenance) != {
        "branch": "agent/restore-dual-router-ux",
        "commit": _PRODUCER_COMMIT,
        "files_pinned": 19,
    }:
        _fail("consumer_provenance_invalid")
    policy = _mapping(config.get("policy"), "consumer_policy_invalid")
    if policy.get("metadata_identity_ready") is not True or any(
        policy.get(key) is not False for key in _ZERO_CLAIMS
    ):
        _fail("consumer_policy_invalid")
    if (
        policy.get("materialization_ready") is not False
        or policy.get("execution_lease_ready") is not False
    ):
        _fail("consumer_policy_invalid")
    partitions = _mapping(config.get("partitions"), "consumer_partitions_invalid")
    if set(partitions) != set(_EXPECTED_PARTITIONS):
        _fail("consumer_partitions_invalid")
    for path, (records, size, digest) in _EXPECTED_PARTITIONS.items():
        if dict(_mapping(partitions[path], "consumer_partitions_invalid")) != {
            "records": records,
            "bytes": size,
            "sha256": digest,
        }:
            _fail("consumer_partitions_invalid")


def _artifact(path: str, snapshot: _Snapshot) -> dict[str, Any]:
    return {"path": path, "bytes": len(snapshot.data), "sha256": snapshot.sha256}


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
    producer_config: Mapping[str, Any],
) -> tuple[frozenset[str], str]:
    atoms: set[str] = set()
    discovery = producer_config["discovery_bridge"]
    _collect_string_leaves(discovery["old_template_descriptor"], atoms)
    for group in discovery["semantic_groups"]:
        _collect_string_leaves(group["descriptor"], atoms)
    ontology = producer_config["semantic_ontology"]
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
    digest = _canonical_sha256(
        {
            "schema_version": (
                "anchor.synthetic-scaffold-common-domain-atom-catalog.v1"
            ),
            "atoms": sorted(atoms),
        }
    )
    contract = _mapping(
        producer_config.get("identity_contract"), "producer_config_identity_invalid"
    )
    if (
        len(atoms) != 241
        or digest != "517f6b829bb78700b171a349d14541f75a9b76aa2a9267acb92a0e1a646d9545"
        or contract.get("descriptor_atom_catalog_count") != 241
        or contract.get("descriptor_atom_catalog_sha256") != digest
        or contract.get("descriptor_atom_catalog_schema_version")
        != "anchor.synthetic-scaffold-common-domain-atom-catalog.v1"
    ):
        _fail("descriptor_atom_catalog_invalid")
    return frozenset(atoms), digest


def _assert_descriptor_atoms(
    value: object, forbidden_keys: set[str], allowed_atoms: frozenset[str]
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key in forbidden_keys:
                _fail("descriptor_field_invalid")
            try:
                key.encode("ascii")
            except UnicodeEncodeError as exc:
                raise IndependentConfirmationIdentityConsumerError(
                    "descriptor_field_invalid"
                ) from exc
            _assert_descriptor_atoms(child, forbidden_keys, allowed_atoms)
    elif isinstance(value, list):
        for child in value:
            _assert_descriptor_atoms(child, forbidden_keys, allowed_atoms)
    elif isinstance(value, str):
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise IndependentConfirmationIdentityConsumerError(
                "descriptor_atom_invalid"
            ) from exc
        if value not in allowed_atoms:
            _fail("descriptor_atom_invalid")
    elif not isinstance(value, (bool, int)) or isinstance(value, float):
        _fail("descriptor_value_invalid")


def _descriptor_context(
    producer_config: Mapping[str, Any],
) -> tuple[frozenset[str], str, str, str]:
    atoms, catalog_sha = _descriptor_atom_catalog(producer_config)
    descriptor_sha = _canonical_sha256(_DESCRIPTOR_SCHEMA_CONTRACT)
    ontology_sha = _canonical_sha256(producer_config["semantic_ontology"])
    manifest_contract = producer_config["identity_contract"]
    if descriptor_sha != _canonical_sha256(_DESCRIPTOR_SCHEMA_CONTRACT) or (
        manifest_contract.get("descriptor_atom_catalog_sha256") != catalog_sha
    ):
        _fail("descriptor_context_invalid")
    return atoms, descriptor_sha, ontology_sha, catalog_sha


def _normalize_task_descriptor(
    descriptor: Mapping[str, Any],
    producer_config: Mapping[str, Any],
    atoms: frozenset[str],
) -> dict[str, Any]:
    required = producer_config["semantic_ontology"]["task_descriptor_required_fields"]
    if set(descriptor) != set(required):
        _fail("task_descriptor_invalid")
    _assert_descriptor_atoms(
        descriptor,
        set(
            producer_config["identity_contract"]["forbidden_semantic_descriptor_fields"]
        ),
        atoms,
    )
    normalized = dict(descriptor)
    for field in ("goal_atoms", "constraint_atoms", "acceptance_atoms"):
        values = list(descriptor[field])
        if len(values) < 3 or len(values) != len(set(values)):
            _fail("task_descriptor_invalid")
        normalized[field] = sorted(values)
    flow = list(descriptor["ordered_information_flow"])
    if len(flow) < 4 or len(flow) != len(set(flow)):
        _fail("task_descriptor_invalid")
    normalized["ordered_information_flow"] = flow
    parameters = _mapping(descriptor["typed_parameters"], "task_descriptor_invalid")
    if set(parameters) != {
        "state_scope",
        "trust_boundary",
        "reversibility",
        "determinism",
    }:
        _fail("task_descriptor_invalid")
    normalized["typed_parameters"] = dict(parameters)
    return normalized


def _normalize_template_descriptor(
    descriptor: Mapping[str, Any],
    producer_config: Mapping[str, Any],
    atoms: frozenset[str],
) -> dict[str, Any]:
    if set(descriptor) != set(_DESCRIPTOR_SCHEMA_CONTRACT["template_required_fields"]):
        _fail("template_descriptor_invalid")
    _assert_descriptor_atoms(
        descriptor,
        set(
            producer_config["identity_contract"]["forbidden_semantic_descriptor_fields"]
        ),
        atoms,
    )
    if list(descriptor["ordered_roles"]) != list(_ROLES) or list(
        descriptor["variant_program"]
    ) != list(_VARIANTS):
        _fail("template_descriptor_invalid")
    for field in _DESCRIPTOR_SCHEMA_CONTRACT["template_ordered_fields"]:
        values = list(descriptor[field])
        if len(values) != len(set(values)):
            _fail("template_descriptor_invalid")
    return dict(descriptor)


def _task_identity_from_descriptor(
    descriptor: Mapping[str, Any],
    producer_config: Mapping[str, Any],
    context: tuple[frozenset[str], str, str, str],
) -> str:
    atoms, descriptor_sha, ontology_sha, catalog_sha = context
    return _canonical_sha256(
        {
            "domain": producer_config["identity_contract"]["task_domain"],
            "descriptor_schema_sha256": descriptor_sha,
            "ontology_sha256": ontology_sha,
            "descriptor_atom_catalog_sha256": catalog_sha,
            "descriptor": _normalize_task_descriptor(
                descriptor, producer_config, atoms
            ),
        }
    )


def _template_identity_from_descriptor(
    descriptor: Mapping[str, Any],
    producer_config: Mapping[str, Any],
    context: tuple[frozenset[str], str, str, str],
) -> str:
    atoms, descriptor_sha, ontology_sha, catalog_sha = context
    return _canonical_sha256(
        {
            "domain": producer_config["identity_contract"]["template_domain"],
            "descriptor_schema_sha256": descriptor_sha,
            "ontology_sha256": ontology_sha,
            "descriptor_atom_catalog_sha256": catalog_sha,
            "descriptor": _normalize_template_descriptor(
                descriptor, producer_config, atoms
            ),
        }
    )


def _confirmation_task_descriptor(
    profile: Mapping[str, Any], stratum: Mapping[str, Any]
) -> dict[str, Any]:
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
    template: Mapping[str, Any], stratum: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "template_profile_id": template["template_profile_id"],
        "information_flow_stratum": stratum["stratum_id"],
        "ordered_roles": list(_ROLES),
        "route_program": list(template["route_program"]),
        "visibility_program": [
            "exclude_current",
            "exclude_future",
            "exclude_forbidden",
            "allow_committed_predecessors",
        ],
        "variant_program": list(_VARIANTS),
        "commit_program": list(template["commit_program"]),
        "tool_trace_policy": "harmless_synthetic_local_receipts",
        "rationale_policy": template["rationale_policy"],
        "serialization_policy": "canonical_json_utf8_compact",
    }


def _source_bundle_identity(
    prefix: str,
    domain: str,
    track: str,
    language: str,
    stratum: str,
    task: str,
    template: str,
    factor: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "domain": domain,
        "track": track,
        "language": language,
        "stratum": stratum,
        "task_semantic_sha256": task,
        "template_family_sha256": template,
    }
    if factor is not None:
        payload["factor"] = factor
    return f"{prefix}:{_canonical_sha256(payload)}"


def _semantic_catalog(
    producer_config: Mapping[str, Any],
) -> tuple[
    set[str], str, dict[tuple[str, str, str], tuple[str, str, str]], dict[str, str]
]:
    context = _descriptor_context(producer_config)
    discovery = producer_config["discovery_bridge"]
    discovery_tasks = {
        _task_identity_from_descriptor(group["descriptor"], producer_config, context)
        for group in discovery["semantic_groups"]
    }
    old_template = _template_identity_from_descriptor(
        discovery["old_template_descriptor"], producer_config, context
    )
    ontology = producer_config["semantic_ontology"]
    profiles = {item["profile_id"]: item for item in ontology["workflow_profiles"]}
    strata = {item["stratum_id"]: item for item in ontology["information_flow_strata"]}
    templates = {
        item["template_profile_id"]: item for item in ontology["template_profiles"]
    }
    profile_order = list(profiles)
    template_order = list(templates)
    expected: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    track = producer_config["independent_confirmation"]
    for language in track["languages"]:
        for stratum_id, stratum in strata.items():
            for profile_id in track["profiles_by_language"][language]:
                profile = profiles[profile_id]
                template_profile_id = template_order[
                    profile_order.index(profile_id) % len(template_order)
                ]
                task = _task_identity_from_descriptor(
                    _confirmation_task_descriptor(profile, stratum),
                    producer_config,
                    context,
                )
                template = _template_identity_from_descriptor(
                    _confirmation_template_descriptor(
                        templates[template_profile_id], stratum
                    ),
                    producer_config,
                    context,
                )
                expected[(language, stratum_id, profile_id)] = (
                    task,
                    template,
                    template_profile_id,
                )
    old_by_stratum = {
        stratum_id: _task_identity_from_descriptor(
            discovery["semantic_groups"][index]["descriptor"],
            producer_config,
            context,
        )
        for index, stratum_id in enumerate(strata)
    }
    return discovery_tasks, old_template, expected, old_by_stratum


def _verify_semantic_leaves(
    producer_config: Mapping[str, Any],
    discovery: Sequence[Mapping[str, Any]],
    independent: Sequence[Mapping[str, Any]],
    factorial: Sequence[Mapping[str, Any]],
) -> None:
    discovery_tasks, old_template, expected, old_by_stratum = _semantic_catalog(
        producer_config
    )
    if {_identity_leaves(row)[0] for row in discovery} != discovery_tasks or {
        _identity_leaves(row)[1] for row in discovery
    } != {old_template}:
        _fail("discovery_semantic_leaf_invalid")
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in independent:
        key = (row["language"], row["stratum"], row["semantic_profile_id"])
        expected_leaves = expected.get(key)
        task, template, _pair = _identity_leaves(row)
        if (
            expected_leaves is None
            or (task, template) != expected_leaves[:2]
            or row.get("template_profile_id") != expected_leaves[2]
            or row.get("source_bundle_id")
            != _source_bundle_identity(
                "ic-source-v1",
                "anchor.synthetic-scaffold-independent-confirmation-source-bundle.v1",
                "producer_independent_confirmation",
                row["language"],
                row["stratum"],
                task,
                template,
            )
            or key in by_key
        ):
            _fail("independent_semantic_leaf_invalid")
        by_key[key] = row
    if set(by_key) != set(expected):
        _fail("independent_semantic_leaf_invalid")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in factorial:
        groups[row["factorial_match_key_sha256"]].append(row)
    new_profiles = producer_config["secondary_controlled_factorial"][
        "new_profile_ids_by_language"
    ]
    for rows in groups.values():
        if len(rows) != 3:
            _fail("factorial_semantic_leaf_invalid")
        new_row = next(row for row in rows if row["factor"] == "new_task_new_template")
        language = new_row["language"]
        stratum = new_row["stratum"]
        candidate = {
            expected[(language, stratum, profile_id)][:2]
            for profile_id in new_profiles[language]
        }
        new_task, new_template, _pair = _identity_leaves(new_row)
        if (new_task, new_template) not in candidate:
            _fail("factorial_semantic_leaf_invalid")
        expected_by_factor = {
            "old_task_new_template": (old_by_stratum[stratum], new_template),
            "new_task_old_template": (new_task, old_template),
            "new_task_new_template": (new_task, new_template),
        }
        for row in rows:
            task, template, _pair = _identity_leaves(row)
            if (task, template) != expected_by_factor[row["factor"]] or row.get(
                "source_bundle_id"
            ) != _source_bundle_identity(
                "cf-source-v1",
                "anchor.synthetic-scaffold-secondary-factorial-source-bundle.v1",
                "secondary_controlled_factorial_probe",
                language,
                stratum,
                task,
                template,
                row["factor"],
            ):
                _fail("factorial_semantic_leaf_invalid")


def _validate_manifest_bindings(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
) -> None:
    producer = _mapping(manifest.get("producer"), "producer_manifest_invalid")
    expected = {
        "config": _artifact(_EXPECTED_PATHS["producer_config"], snapshots["config"]),
        "config_schema": _artifact(
            _EXPECTED_PATHS["producer_config_schema"], snapshots["config_schema"]
        ),
        "record_schema": _artifact(
            _EXPECTED_PATHS["producer_record_schema"], snapshots["record_schema"]
        ),
        "proof_schema": _artifact(
            _EXPECTED_PATHS["producer_proof_schema"], snapshots["proof_schema"]
        ),
        "manifest_schema": _artifact(
            _EXPECTED_PATHS["producer_manifest_schema"],
            snapshots["manifest_schema"],
        ),
        "implementation": _artifact(
            _EXPECTED_PATHS["producer_implementation"],
            snapshots["implementation"],
        ),
    }
    for role, descriptor in expected.items():
        if (
            dict(_mapping(producer.get(role), "producer_manifest_invalid"))
            != descriptor
        ):
            _fail("producer_manifest_binding_mismatch")
    if (
        producer.get("producer_version")
        != "anchor.synthetic-scaffold-independent-confirmation-identity-producer.v1"
        or producer.get("final_toctou_recheck") is not True
        or producer.get("atomic_publish_no_replace") is not True
    ):
        _fail("producer_manifest_invalid")

    entries = manifest.get("partitions")
    if not isinstance(entries, list) or len(entries) != len(_EXPECTED_PARTITIONS):
        _fail("producer_manifest_partition_invalid")
    by_path = {
        item.get("path"): item
        for item in entries
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    if set(by_path) != set(_EXPECTED_PARTITIONS):
        _fail("producer_manifest_partition_invalid")
    configured = _mapping(config["partitions"], "consumer_partitions_invalid")
    for path, item in by_path.items():
        descriptor = _mapping(configured[path], "consumer_partitions_invalid")
        expected_media = (
            "application/x-ndjson" if path.endswith(".jsonl") else "application/json"
        )
        if dict(item) != {
            "path": path,
            "media_type": expected_media,
            "records": descriptor["records"],
            "bytes": descriptor["bytes"],
            "sha256": descriptor["sha256"],
        }:
            _fail("producer_manifest_partition_invalid")

    claims = _mapping(manifest.get("claims"), "producer_manifest_claims_invalid")
    if any(value is not False for value in claims.values()):
        _fail("producer_manifest_claims_invalid")
    counts = _mapping(manifest.get("counts"), "producer_manifest_counts_invalid")
    if any(counts.get(key) != 0 for key in _ZERO_RESOURCE_AUDIT):
        _fail("producer_manifest_resource_audit_invalid")
    audit = _mapping(manifest.get("audit"), "producer_manifest_audit_invalid")
    if any(audit.get(key) != 0 for key in _ZERO_RESOURCE_AUDIT):
        _fail("producer_manifest_resource_audit_invalid")


def _identity_leaves(record: Mapping[str, Any]) -> tuple[str, str, str]:
    identities = _mapping(record.get("identities"), "record_identity_invalid")
    task = identities.get("task_semantic_sha256")
    source_task = identities.get("source_task_blueprint_sha256")
    template = identities.get("template_family_sha256")
    pair = identities.get("task_template_pair_sha256")
    if (
        not all(
            isinstance(value, str) and len(value) == 64
            for value in (task, source_task, template, pair)
        )
        or source_task != task
    ):
        _fail("record_identity_invalid")
    return task, template, pair


def _verify_record_identities(
    records: Sequence[Mapping[str, Any]],
    producer_config: Mapping[str, Any],
    track: str,
) -> None:
    contract = _mapping(
        producer_config.get("identity_contract"), "producer_config_identity_invalid"
    )
    pair_domain = contract.get("pair_domain")
    namespace = None
    if track == _SCOPES[1]:
        namespace = _mapping(
            producer_config.get("independent_confirmation"),
            "producer_config_identity_invalid",
        ).get("source_namespace")
    elif track == _SCOPES[2]:
        namespace = _mapping(
            producer_config.get("secondary_controlled_factorial"),
            "producer_config_identity_invalid",
        ).get("source_namespace")
    for record in records:
        if record.get("track") != track:
            _fail("record_track_invalid")
        task, template, pair = _identity_leaves(record)
        expected_pair = _canonical_sha256(
            {
                "domain": pair_domain,
                "task_semantic_sha256": task,
                "template_family_sha256": template,
            }
        )
        if pair != expected_pair:
            _fail("record_pair_identity_invalid")
        expansion = _mapping(record.get("expansion"), "record_expansion_invalid")
        if dict(expansion) != {
            "all_views_same_split": True,
            "expected_record_count": 10,
            "ordered_roles": list(_ROLES),
            "ordered_variants": list(_VARIANTS),
            "split_before_role_variant_expansion": True,
        }:
            _fail("record_expansion_invalid")
        if namespace is not None:
            bundle = record.get("task_bundle_sha256")
            source_bundle = record.get("source_bundle_id")
            language = record.get("language")
            expected_bundle = _canonical_sha256(
                {
                    "domain": contract.get("task_bundle_domain"),
                    "source_namespace": namespace,
                    "source_bundle_id": source_bundle,
                    "language": language,
                    "source_task_blueprint_sha256": task,
                }
            )
            if (
                bundle != expected_bundle
                or record.get("task_board_task_id") != f"task-v2:{bundle}"
                or record.get("metadata_only") is not True
            ):
                _fail("record_task_bundle_identity_invalid")


def _scope_sets(
    discovery: Sequence[Mapping[str, Any]],
    independent: Sequence[Mapping[str, Any]],
    factorial: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    for scope, rows in zip(_SCOPES, (discovery, independent, factorial), strict=True):
        leaves = [_identity_leaves(row) for row in rows]
        result[scope] = {
            "task": {row[0] for row in leaves},
            "template": {row[1] for row in leaves},
            "pair": {row[2] for row in leaves},
        }
    return result


def _inventory_summary(scope: str, values: Mapping[str, set[str]]) -> dict[str, Any]:
    return {
        kind: {
            "count": len(values[kind]),
            "sha256": _inventory_sha256(
                f"anchor.synthetic-scaffold-{scope}-{kind}-inventory.v1",
                values[kind],
            ),
        }
        for kind in _KINDS
    }


def _intersection_summary(domain: str, values: set[str]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "ids": ordered,
        "sha256": _inventory_sha256(domain, ordered),
        "zero_overlap": not ordered,
    }


def _verify_inventory_partitions(
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
    sets: Mapping[str, Mapping[str, set[str]]],
) -> None:
    for kind in _KINDS:
        membership: dict[str, list[str]] = defaultdict(list)
        for scope in _SCOPES:
            for value in sets[scope][kind]:
                membership[value].append(scope)
        expected = [
            {
                "schema_version": (
                    "anchor.synthetic-scaffold-independent-confirmation-identity-"
                    "record.v1"
                ),
                "record_kind": "identity_inventory_leaf",
                "identity_kind": _IDENTITY_KIND[kind],
                "identity_sha256": value,
                "scope_memberships": [
                    scope for scope in _SCOPES if scope in membership[value]
                ],
            }
            for value in sorted(membership)
        ]
        if list(inventories[kind]) != expected:
            _fail("identity_inventory_partition_mismatch")


def _expected_independent_proof(
    producer_config: Mapping[str, Any],
    discovery: Sequence[Mapping[str, Any]],
    sets: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, Any]:
    contract = _mapping(
        producer_config.get("identity_contract"), "producer_config_identity_invalid"
    )
    domains = {
        "task": contract["task_domain"],
        "template": contract["template_domain"],
        "pair": contract["pair_domain"],
        "namespace_language_source_fields_excluded": True,
    }
    train_tasks = {
        _identity_leaves(row)[0] for row in discovery if row.get("split") == "train"
    }
    eval_tasks = {
        _identity_leaves(row)[0]
        for row in discovery
        if row.get("split") == "eval_proxy"
    }
    return {
        "schema_version": (
            "anchor.synthetic-scaffold-independent-confirmation-identity-proof.v1"
        ),
        "proof_kind": "discovery_vs_independent_confirmation",
        "status": "descriptor_level_zero_overlap_proven_execution_not_authorized",
        "domains": domains,
        "discovery": {
            "source_bundle_views": 10,
            "unique_semantics": 5,
            "bilingual_groups": 5,
            "inventories": _inventory_summary("discovery-bridge", sets[_SCOPES[0]]),
        },
        "independent_confirmation": {
            "bundle_views": 60,
            "unique_semantics": 60,
            "cross_language_translation_pairs": 0,
            "inventories": _inventory_summary(
                "producer-independent-confirmation", sets[_SCOPES[1]]
            ),
        },
        "intersections": {
            kind: _intersection_summary(
                "anchor.synthetic-scaffold-discovery-independent-"
                f"{kind}-intersection.v1",
                sets[_SCOPES[0]][kind] & sets[_SCOPES[1]][kind],
            )
            for kind in _KINDS
        },
        "historical_discovery_semantic_split": {
            "historical_split_rewritten": False,
            "train_unique_semantics": len(train_tasks),
            "eval_proxy_unique_semantics": len(eval_tasks),
            "intersection": _intersection_summary(
                "anchor.synthetic-scaffold-discovery-semantic-split-intersection.v1",
                train_tasks & eval_tasks,
            ),
            "bundle_generalization_supported": False,
        },
        "proof_algorithm": (
            "recompute_leaves_common_domains_sorted_unique_inventory_then_exact_set_"
            "intersection_v1"
        ),
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


def _expected_factorial_proof(
    producer_config: Mapping[str, Any],
    factorial: Sequence[Mapping[str, Any]],
    sets: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, Any]:
    contract = _mapping(
        producer_config.get("identity_contract"), "producer_config_identity_invalid"
    )
    discovery = sets[_SCOPES[0]]
    independent = sets[_SCOPES[1]]
    expected_membership = {
        "old_task_new_template": (True, False),
        "new_task_old_template": (False, True),
        "new_task_new_template": (False, False),
    }
    truth_table: list[dict[str, Any]] = []
    for factor in _FACTORS:
        rows = [row for row in factorial if row.get("factor") == factor]
        task_old, template_old = expected_membership[factor]
        observed = True
        for row in rows:
            task, template, pair = _identity_leaves(row)
            expected_record_membership = {
                "task_in_discovery": task in discovery["task"],
                "template_in_discovery": template in discovery["template"],
                "pair_in_discovery": pair in discovery["pair"],
                "new_task_in_independent_inventory": task in independent["task"],
                "new_template_in_independent_inventory": (
                    template in independent["template"]
                ),
            }
            if (
                dict(_mapping(row.get("membership"), "factorial_membership_invalid"))
                != expected_record_membership
            ):
                _fail("factorial_membership_invalid")
            observed = observed and (
                expected_record_membership["task_in_discovery"] == task_old
                and expected_record_membership["template_in_discovery"] == template_old
                and expected_record_membership["pair_in_discovery"] is False
            )
        truth_table.append(
            {
                "factor": factor,
                "bundle_count": len(rows),
                "task_in_discovery": task_old,
                "template_in_discovery": template_old,
                "pair_in_discovery": False,
                "observed_pass": observed,
            }
        )

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in factorial:
        key = row.get("factorial_match_key_sha256")
        if not isinstance(key, str):
            _fail("factorial_match_group_invalid")
        groups[key].append(row)
    for key, rows in groups.items():
        if len(rows) != 3 or {row.get("factor") for row in rows} != set(_FACTORS):
            _fail("factorial_match_group_invalid")
        new_row = next(
            row for row in rows if row.get("factor") == "new_task_new_template"
        )
        task, template, _pair = _identity_leaves(new_row)
        if key != _canonical_sha256(
            {
                "domain": "anchor.synthetic-scaffold-factorial-match-key.v1",
                "new_task_semantic_sha256": task,
                "new_template_family_sha256": template,
            }
        ):
            _fail("factorial_match_group_invalid")
        if (
            len({row.get("language") for row in rows}) != 1
            or len({row.get("stratum") for row in rows}) != 1
        ):
            _fail("factorial_match_group_invalid")

    new_task_old = {
        _identity_leaves(row)[0]
        for row in factorial
        if row.get("factor") == "new_task_old_template"
    }
    new_task_new = {
        _identity_leaves(row)[0]
        for row in factorial
        if row.get("factor") == "new_task_new_template"
    }
    new_template_old_task = {
        _identity_leaves(row)[1]
        for row in factorial
        if row.get("factor") == "old_task_new_template"
    }
    new_template_new_task = {
        _identity_leaves(row)[1]
        for row in factorial
        if row.get("factor") == "new_task_new_template"
    }
    train_quota = Counter(
        row["factor"] for row in factorial if row.get("split") == "train"
    )
    eval_quota = Counter(
        row["factor"] for row in factorial if row.get("split") == "eval_proxy"
    )
    languages = list(producer_config["independent_confirmation"]["languages"])
    strata = [
        row["stratum_id"]
        for row in producer_config["semantic_ontology"]["information_flow_strata"]
    ]
    cells = []
    for language in languages:
        for stratum in strata:
            rows = [
                row
                for row in factorial
                if row.get("language") == language and row.get("stratum") == stratum
            ]
            cells.append(
                {
                    "language": language,
                    "stratum": stratum,
                    "bundles": len(rows),
                    "train": sum(row.get("split") == "train" for row in rows),
                    "eval_proxy": sum(row.get("split") == "eval_proxy" for row in rows),
                }
            )
    factorial_sets = sets[_SCOPES[2]]
    intersections = {
        kind: _intersection_summary(
            f"anchor.synthetic-scaffold-factorial-discovery-{kind}-intersection.v1",
            factorial_sets[kind] & discovery[kind],
        )
        for kind in _KINDS
    }
    return {
        "schema_version": (
            "anchor.synthetic-scaffold-independent-confirmation-identity-proof.v1"
        ),
        "proof_kind": "secondary_controlled_factorial_membership",
        "status": "membership_truth_table_proven_not_independent_confirmation",
        "domains": {
            "task": contract["task_domain"],
            "template": contract["template_domain"],
            "pair": contract["pair_domain"],
            "namespace_language_source_fields_excluded": True,
        },
        "counts": {
            "bundles": len(factorial),
            "train": sum(row.get("split") == "train" for row in factorial),
            "eval_proxy": sum(row.get("split") == "eval_proxy" for row in factorial),
            "language_stratum_cells": len(cells),
            "factorial_match_keys": len(groups),
        },
        "factor_counts": dict(Counter(row["factor"] for row in factorial)),
        "split_factor_quotas": {
            "train": dict(train_quota),
            "eval_proxy": dict(eval_quota),
        },
        "language_stratum_cells": cells,
        "truth_table": truth_table,
        "matched_factor_controls": {
            "new_task_inventory_equal_between_old_template_and_new_template_cells": (
                new_task_old == new_task_new
            ),
            "new_template_inventory_equal_between_old_task_and_new_task_cells": (
                new_template_old_task == new_template_new_task
            ),
            "each_match_key_has_all_three_factors": len(groups) == 20,
        },
        "global_discovery_intersections": {
            **intersections,
            "global_task_zero_overlap_required": False,
            "global_template_zero_overlap_required": False,
            "global_pair_zero_overlap_required": True,
        },
        "pair_inventory": _inventory_summary(
            "secondary-controlled-factorial-probe", factorial_sets
        )["pair"],
        "proof_algorithm": (
            "recompute_task_template_pair_membership_per_bundle_then_exact_truth_"
            "table_and_quota_audit_v1"
        ),
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


def _verify_expected_counts(
    config: Mapping[str, Any],
    sets: Mapping[str, Mapping[str, set[str]]],
    discovery: Sequence[Mapping[str, Any]],
    independent: Sequence[Mapping[str, Any]],
    factorial: Sequence[Mapping[str, Any]],
) -> None:
    expected = _mapping(config.get("expected"), "consumer_expected_invalid")
    discovery_expected = _mapping(
        expected.get("discovery"), "consumer_expected_invalid"
    )
    independent_expected = _mapping(
        expected.get("producer_independent_confirmation"),
        "consumer_expected_invalid",
    )
    factorial_expected = _mapping(
        expected.get("secondary_controlled_factorial_probe"),
        "consumer_expected_invalid",
    )
    observed = {
        "discovery": {
            "views": len(discovery),
            "unique_tasks": len(sets[_SCOPES[0]]["task"]),
            "unique_templates": len(sets[_SCOPES[0]]["template"]),
            "unique_pairs": len(sets[_SCOPES[0]]["pair"]),
        },
        "independent": {
            "bundles": len(independent),
            "train": sum(row.get("split") == "train" for row in independent),
            "eval_proxy": sum(row.get("split") == "eval_proxy" for row in independent),
            "unique_tasks": len(sets[_SCOPES[1]]["task"]),
            "unique_templates": len(sets[_SCOPES[1]]["template"]),
            "unique_pairs": len(sets[_SCOPES[1]]["pair"]),
            "discovery_intersections": {
                kind: len(sets[_SCOPES[0]][kind] & sets[_SCOPES[1]][kind])
                for kind in _KINDS
            },
        },
        "factorial": {
            "bundles": len(factorial),
            "train": sum(row.get("split") == "train" for row in factorial),
            "eval_proxy": sum(row.get("split") == "eval_proxy" for row in factorial),
            "unique_tasks": len(sets[_SCOPES[2]]["task"]),
            "unique_templates": len(sets[_SCOPES[2]]["template"]),
            "unique_pairs": len(sets[_SCOPES[2]]["pair"]),
            "factorial_match_groups": len(
                {row.get("factorial_match_key_sha256") for row in factorial}
            ),
            "factor_counts": dict(Counter(row["factor"] for row in factorial)),
            "train_factor_quotas": dict(
                Counter(
                    row["factor"] for row in factorial if row.get("split") == "train"
                )
            ),
            "eval_factor_quotas": dict(
                Counter(
                    row["factor"]
                    for row in factorial
                    if row.get("split") == "eval_proxy"
                )
            ),
            "discovery_intersections": {
                kind: len(sets[_SCOPES[0]][kind] & sets[_SCOPES[2]][kind])
                for kind in _KINDS
            },
        },
    }
    if dict(discovery_expected) != observed["discovery"]:
        _fail("discovery_counts_invalid")
    if dict(independent_expected) != observed["independent"]:
        _fail("independent_counts_invalid")
    if dict(factorial_expected) != observed["factorial"]:
        _fail("factorial_counts_invalid")
    union_expected = _mapping(expected.get("union"), "consumer_expected_invalid")
    union_observed = {
        f"unique_{kind}s": len(set().union(*(sets[scope][kind] for scope in _SCOPES)))
        for kind in _KINDS
    }
    if dict(union_expected) != union_observed:
        _fail("union_counts_invalid")


def _requested_config(root: Path, path: str | Path) -> Path:
    requested = Path(path)
    candidate = requested if requested.is_absolute() else root / requested
    canonical = (root / CONFIG_PATH).resolve(strict=False)
    if ".." in requested.parts or candidate.resolve(strict=False) != canonical:
        _fail("consumer_config_path_invalid")
    return canonical


def _evaluate(
    root: Path,
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    provenance_repo: Path | None = None,
) -> dict[str, Any]:
    config_snapshot = _read_snapshot(
        root,
        _requested_config(root, config_path),
        "consumer_config_unreadable",
    )
    if config_snapshot.sha256 != expected_config_sha256:
        _fail("consumer_config_sha256_mismatch")
    config = _load_yaml(config_snapshot.data)
    _validate_consumer_config(config)

    paths = _mapping(config["paths"], "consumer_paths_invalid")
    bindings = _mapping(config["bindings"], "consumer_bindings_invalid")
    role_paths = {
        "config": "producer_config",
        "config_schema": "producer_config_schema",
        "record_schema": "producer_record_schema",
        "proof_schema": "producer_proof_schema",
        "manifest_schema": "producer_manifest_schema",
        "implementation": "producer_implementation",
        "producer_test": "producer_test",
        "producer_docs_en": "producer_docs_en",
        "producer_docs_zh_cn": "producer_docs_zh_cn",
        "decision_schema": "decision_schema",
    }
    role_bindings = {
        "config": "producer_config_sha256",
        "config_schema": "producer_config_schema_sha256",
        "record_schema": "producer_record_schema_sha256",
        "proof_schema": "producer_proof_schema_sha256",
        "manifest_schema": "producer_manifest_schema_sha256",
        "implementation": "producer_implementation_sha256",
        "producer_test": "producer_test_sha256",
        "producer_docs_en": "producer_docs_en_sha256",
        "producer_docs_zh_cn": "producer_docs_zh_cn_sha256",
        "decision_schema": "decision_schema_sha256",
    }
    snapshots: dict[str, _Snapshot] = {}
    for role, path_key in role_paths.items():
        snapshot = _read_snapshot(
            root,
            _safe_path(root, paths[path_key], f"{role}_path_invalid"),
            f"{role}_unreadable",
        )
        if snapshot.sha256 != bindings[role_bindings[role]]:
            _fail(f"{role}_sha256_mismatch")
        snapshots[role] = snapshot

    fixture_relative = paths["producer_fixture"]
    manifest_path = _safe_path(
        root, f"{fixture_relative}/manifest.json", "producer_manifest_path_invalid"
    )
    sidecar_path = _safe_path(
        root,
        f"{fixture_relative}/manifest.json.sha256",
        "producer_manifest_sidecar_path_invalid",
    )
    snapshots["manifest"] = _read_snapshot(
        root, manifest_path, "producer_manifest_unreadable"
    )
    snapshots["manifest_sidecar"] = _read_snapshot(
        root, sidecar_path, "producer_manifest_sidecar_unreadable"
    )
    if snapshots["manifest"].sha256 != bindings["producer_manifest_sha256"]:
        _fail("producer_manifest_sha256_mismatch")
    if (
        snapshots["manifest_sidecar"].sha256
        != bindings["producer_manifest_sidecar_physical_sha256"]
    ):
        _fail("producer_manifest_sidecar_sha256_mismatch")
    expected_sidecar = f"{snapshots['manifest'].sha256}  manifest.json\n".encode(
        "ascii"
    )
    if snapshots["manifest_sidecar"].data != expected_sidecar:
        _fail("producer_manifest_sidecar_invalid")

    producer_config = _load_json_mapping(
        snapshots["config"].data, "producer_config_invalid"
    )
    schemas = {
        role: _load_json_mapping(snapshots[role].data, f"{role}_invalid")
        for role in (
            "config_schema",
            "record_schema",
            "proof_schema",
            "manifest_schema",
            "decision_schema",
        )
    }
    validators = {
        role: _validator(schema, f"{role}_invalid") for role, schema in schemas.items()
    }
    _validate(validators["config_schema"], producer_config, "producer_config_invalid")
    manifest = _load_json_mapping(
        snapshots["manifest"].data, "producer_manifest_invalid"
    )
    _validate(validators["manifest_schema"], manifest, "producer_manifest_invalid")
    _validate_manifest_bindings(manifest, config, snapshots)

    partition_rows: dict[str, list[Mapping[str, Any]]] = {}
    proof_documents: dict[str, Mapping[str, Any]] = {}
    for relative, (
        expected_records,
        expected_bytes,
        expected_sha,
    ) in _EXPECTED_PARTITIONS.items():
        key = f"partition:{relative}"
        snapshot = _read_snapshot(
            root,
            _safe_path(
                root, f"{fixture_relative}/{relative}", "partition_path_invalid"
            ),
            "partition_unreadable",
        )
        snapshots[key] = snapshot
        if len(snapshot.data) != expected_bytes or snapshot.sha256 != expected_sha:
            _fail("partition_identity_mismatch")
        if relative.endswith(".jsonl"):
            rows = _load_jsonl(snapshot.data, "partition_jsonl_invalid")
            if len(rows) != expected_records:
                _fail("partition_record_count_invalid")
            for row in rows:
                _validate(validators["record_schema"], row, "record_schema_invalid")
            partition_rows[relative] = rows
        else:
            document = _load_json_mapping(snapshot.data, "proof_json_invalid")
            if _canonical_bytes(document, newline=True) != snapshot.data:
                _fail("proof_noncanonical")
            _validate(validators["proof_schema"], document, "proof_schema_invalid")
            proof_documents[relative] = document

    discovery = partition_rows["discovery/views.jsonl"]
    independent = partition_rows["independent_confirmation/bundles.jsonl"]
    factorial = partition_rows["secondary_factorial/bundles.jsonl"]
    _verify_record_identities(discovery, producer_config, _SCOPES[0])
    _verify_record_identities(independent, producer_config, _SCOPES[1])
    _verify_record_identities(factorial, producer_config, _SCOPES[2])
    _verify_semantic_leaves(producer_config, discovery, independent, factorial)
    _atoms, descriptor_sha, ontology_sha, catalog_sha = _descriptor_context(
        producer_config
    )
    manifest_identity_contract = _mapping(
        manifest.get("identity_contract"), "manifest_identity_contract_invalid"
    )
    if (
        manifest_identity_contract.get("descriptor_atom_catalog_count") != 241
        or manifest_identity_contract.get("descriptor_atom_catalog_sha256")
        != catalog_sha
        or manifest_identity_contract.get("descriptor_schema_sha256") != descriptor_sha
        or manifest_identity_contract.get("ontology_sha256") != ontology_sha
        or manifest_identity_contract.get("pair_recomputed_not_self_reported")
        is not True
        or manifest_identity_contract.get(
            "source_task_blueprint_is_task_semantic_alias"
        )
        is not True
    ):
        _fail("manifest_identity_contract_invalid")
    sets = _scope_sets(discovery, independent, factorial)
    inventories = {
        "task": partition_rows["inventories/task_semantic_ids.jsonl"],
        "template": partition_rows["inventories/template_family_ids.jsonl"],
        "pair": partition_rows["inventories/task_template_pair_ids.jsonl"],
    }
    _verify_inventory_partitions(inventories, sets)
    _verify_expected_counts(config, sets, discovery, independent, factorial)

    logical = _mapping(
        manifest.get("logical_inventories"), "manifest_logical_inventory_invalid"
    )
    expected_logical = {
        "discovery_bridge": _inventory_summary("discovery-bridge", sets[_SCOPES[0]]),
        "producer_independent_confirmation": _inventory_summary(
            "producer-independent-confirmation", sets[_SCOPES[1]]
        ),
        "secondary_controlled_factorial_probe": _inventory_summary(
            "secondary-controlled-factorial-probe", sets[_SCOPES[2]]
        ),
        "union": {
            kind: {
                "count": len(set().union(*(sets[scope][kind] for scope in _SCOPES))),
                "sha256": _inventory_sha256(
                    f"anchor.synthetic-scaffold-union-{kind}-inventory.v1",
                    set().union(*(sets[scope][kind] for scope in _SCOPES)),
                ),
            }
            for kind in _KINDS
        },
    }
    if dict(logical) != expected_logical:
        _fail("manifest_logical_inventory_invalid")

    independent_proof = _expected_independent_proof(producer_config, discovery, sets)
    if proof_documents["proofs/discovery_vs_independent.json"] != independent_proof:
        _fail("independent_proof_mismatch")
    factorial_proof = _expected_factorial_proof(producer_config, factorial, sets)
    if proof_documents["proofs/secondary_factorial.json"] != factorial_proof:
        _fail("factorial_proof_mismatch")
    _authenticate_producer_git(provenance_repo or root, snapshots, fixture_relative)

    decision: dict[str, Any] = {
        "schema_version": DECISION_VERSION,
        "status": "metadata_identity_ready_execution_blocked",
        "claim_scope": "additive_metadata_identity_overlay_non_authorizing",
        "metadata_identity_ready": True,
        "producer_bindings": {
            "producer_commit": _PRODUCER_COMMIT,
            "config_sha256": bindings["producer_config_sha256"],
            "config_schema_sha256": bindings["producer_config_schema_sha256"],
            "record_schema_sha256": bindings["producer_record_schema_sha256"],
            "proof_schema_sha256": bindings["producer_proof_schema_sha256"],
            "manifest_schema_sha256": bindings["producer_manifest_schema_sha256"],
            "implementation_sha256": bindings["producer_implementation_sha256"],
            "producer_test_sha256": bindings["producer_test_sha256"],
            "producer_docs_en_sha256": bindings["producer_docs_en_sha256"],
            "producer_docs_zh_cn_sha256": bindings["producer_docs_zh_cn_sha256"],
            "manifest_sha256": bindings["producer_manifest_sha256"],
            "manifest_sidecar_physical_sha256": bindings[
                "producer_manifest_sidecar_physical_sha256"
            ],
        },
        "tracks": {
            "producer_independent_confirmation": {
                "metadata_identity_ready": True,
                "bundles": len(independent),
                "train": sum(row.get("split") == "train" for row in independent),
                "eval_proxy": sum(
                    row.get("split") == "eval_proxy" for row in independent
                ),
                "task_inventory": len(sets[_SCOPES[1]]["task"]),
                "template_inventory": len(sets[_SCOPES[1]]["template"]),
                "pair_inventory": len(sets[_SCOPES[1]]["pair"]),
                "discovery_intersections": {
                    kind: len(sets[_SCOPES[0]][kind] & sets[_SCOPES[1]][kind])
                    for kind in _KINDS
                },
                "records_materialized": False,
                "protected_source_disjoint": False,
                "independent_confirmation_executed": False,
            },
            "secondary_controlled_factorial_probe": {
                "metadata_identity_ready": True,
                "bundles": len(factorial),
                "train": sum(row.get("split") == "train" for row in factorial),
                "eval_proxy": sum(
                    row.get("split") == "eval_proxy" for row in factorial
                ),
                "factorial_match_groups": len(
                    {row["factorial_match_key_sha256"] for row in factorial}
                ),
                "factor_counts": dict(Counter(row["factor"] for row in factorial)),
                "train_factor_quotas": dict(
                    Counter(
                        row["factor"]
                        for row in factorial
                        if row.get("split") == "train"
                    )
                ),
                "eval_factor_quotas": dict(
                    Counter(
                        row["factor"]
                        for row in factorial
                        if row.get("split") == "eval_proxy"
                    )
                ),
                "discovery_intersections": {
                    kind: len(sets[_SCOPES[0]][kind] & sets[_SCOPES[2]][kind])
                    for kind in _KINDS
                },
                "records_materialized": False,
                "protected_source_disjoint": False,
                "controlled_factorial_executed": False,
                "may_satisfy_independent_confirmation": False,
            },
        },
        "gates": {
            "formal_v3_ready_count": 0,
            "formal_v3_total": 5,
            "protected_inventory_ready_count": 2,
            "protected_inventory_total": 6,
            "materialization_ready": False,
            "execution_lease_ready": False,
        },
        "claims": dict(_ZERO_CLAIMS),
        "audit": {
            "single_bytes_snapshot": True,
            "same_bytes_parse_hash_count": True,
            "final_toctou_recheck": True,
            "producer_git_commit_authenticated": True,
            "producer_files_pinned": 19,
            "descriptor_atom_catalog_recomputed": 241,
            "schemas_validated": 5,
            "records_validated": 321,
            "proofs_recomputed": 2,
            **_ZERO_RESOURCE_AUDIT,
        },
    }
    decision["decision_sha256"] = _canonical_sha256(decision)
    _validate(validators["decision_schema"], decision, "decision_schema_invalid")

    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(root, f"{role}_changed")
    config_snapshot.assert_unchanged(root, "consumer_config_changed")
    return decision


def evaluate_independent_confirmation_identity(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Evaluate the metadata overlay and always leave execution blocked."""

    return _evaluate(_ROOT, config_path, CONFIG_SHA256, provenance_repo=_ROOT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision = evaluate_independent_confirmation_identity(args.config)
    except IndependentConfirmationIdentityConsumerError as exc:
        print(
            json.dumps(
                {
                    "schema_version": DECISION_VERSION,
                    "status": "blocked_invalid_identity_metadata",
                    "error": exc.code,
                    "metadata_identity_ready": False,
                    "training_authorized": False,
                    "formal_training_authorized": False,
                    "formal": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
