"""Build and audit the 1,000-record five-role Q-only diagnostic fixture.

This producer is deliberately local, deterministic, and diagnostic-only.  It
uses the frozen v1 implementation only for byte-snapshot, canonical JSON, path,
and JSON-Schema safety primitives.  It never reads protected dataset bodies.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from anchor_mvp.research import synthetic_nl_scaffold_diagnostic_v1 as _v1


CONFIG_VERSION = "anchor.synthetic-five-role-qonly-diagnostic-config.v1"
RECORD_VERSION = "anchor.synthetic-five-role-qonly-diagnostic-record.v1"
MANIFEST_VERSION = "anchor.synthetic-five-role-qonly-diagnostic-manifest.v1"
PRODUCER_VERSION = "anchor.synthetic-five-role-qonly-diagnostic-producer.v1"
GRAMMAR_VERSION = "anchor.synthetic-five-role-qonly-closed-grammar.v1"
CATALOG_VERSION = "anchor.synthetic-five-role-qonly-bundle-catalog.v1"
SOURCE_NAMESPACE = "anchor.synthetic-five-role-qonly-diagnostic.v1"
SEED_ID = "anchor.synthetic-five-role-qonly-diagnostic.seed.v1"
CLAIM_SCOPE = "synthetic_diagnostic_only_no_formal_or_training_authority"

CONFIG_PATH = "configs/research/synthetic_five_role_qonly_diagnostic_v1.yaml"
CONFIG_SCHEMA_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1_config.schema.json"
)
CLOSED_GRAMMAR_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1_closed_grammar.json"
)
CLOSED_GRAMMAR_SCHEMA_PATH = (
    "configs/research/"
    "synthetic_five_role_qonly_diagnostic_v1_closed_grammar.schema.json"
)
CATALOG_PATH = "configs/research/synthetic_five_role_qonly_diagnostic_v1_bundles.json"
CATALOG_SCHEMA_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1_bundles.schema.json"
)
RECORD_SCHEMA_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    "configs/research/synthetic_five_role_qonly_diagnostic_v1_manifest.schema.json"
)
IMPLEMENTATION_PATH = (
    "src/anchor_mvp/research/synthetic_five_role_qonly_diagnostic_v1.py"
)
BASE_SECURITY_IMPLEMENTATION_PATH = (
    "src/anchor_mvp/research/synthetic_nl_scaffold_diagnostic_v1.py"
)
CANONICAL_FIXTURE_PATH = "fixtures/research/synthetic_five_role_qonly_diagnostic_v1"

LANGUAGES = ("en", "zh-CN")
ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
CANONICAL_STAGES = (
    "planner",
    "tool_policy",
    "domain_builder",
    "domain_review",
    "security",
)
ROLE_TO_CANONICAL_STAGE = dict(zip(ROLES, CANONICAL_STAGES))
PRIMARY_VIEW = "concise_rationale_plus_json"
PRIMARY_LABEL = "q_only"
DIAGNOSTIC_CONTROLS = ("o_only", "q_plus_o")
PARTITION_KEYS = (
    ("train", "concise_rationale_plus_json"),
    ("eval_proxy", "concise_rationale_plus_json"),
)
PARTITION_PATHS = tuple(f"{split}/{variant}.jsonl" for split, variant in PARTITION_KEYS)
EXPECTED_COUNTS = {
    "records": 1000,
    "role_views": 1000,
    "pair_count": 0,
    "task_bundles": 200,
    "roles": 5,
    "languages": 2,
    "variants": 1,
    "variants_per_role": 1,
    "train_records": 800,
    "eval_proxy_records": 200,
}
_MAX_BYTES = 50_000_000
_SEMANTIC_ATOM = re.compile(r"^[a-z0-9_]+(?:::[a-z0-9_]+)*$")

SyntheticScaffoldDiagnosticV2Error = _v1.SyntheticScaffoldDiagnosticError
_Snapshot = _v1._Snapshot


def _fail(code: str) -> None:
    raise SyntheticScaffoldDiagnosticV2Error(code)


def _sha256(data: bytes) -> str:
    return _v1._sha256(data)


def _canonical_json(value: object) -> str:
    return _v1._canonical_json(value)


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    return _v1._canonical_json_bytes(value, newline=newline)


def _canonical_sha256(value: object) -> str:
    return _v1._canonical_sha256(value)


def _inventory_sha256(domain: str, values: Sequence[str]) -> str:
    return _v1._inventory_sha256(domain, values)


def _id(prefix: str, payload: object) -> str:
    return _v1._id(prefix, payload)


def _artifact_descriptor(repo_root: Path, snapshot: _Snapshot) -> dict[str, Any]:
    return {
        "path": snapshot.path.relative_to(repo_root).as_posix(),
        "bytes": len(snapshot.data),
        "sha256": snapshot.sha256,
    }


def _load_contract_snapshots(
    repo_root: Path, config_path: Path
) -> tuple[
    dict[str, _Snapshot],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    config_snapshot = _v1._read_snapshot(config_path, "synthetic_v2_config_unreadable")
    try:
        config = yaml.load(
            config_snapshot.data.decode("utf-8"), Loader=_v1._UniqueKeySafeLoader
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SyntheticScaffoldDiagnosticV2Error("synthetic_v2_config_invalid") from exc
    if not isinstance(config, Mapping) or not isinstance(config.get("paths"), Mapping):
        _fail("synthetic_v2_config_invalid")
    paths = config["paths"]
    path_fields = {
        "config_schema": "config_schema",
        "closed_grammar": "closed_grammar",
        "closed_grammar_schema": "closed_grammar_schema",
        "bundle_catalog": "bundle_catalog",
        "bundle_catalog_schema": "bundle_catalog_schema",
        "record_schema": "record_schema",
        "manifest_schema": "manifest_schema",
        "implementation": "implementation",
        "base_security_implementation": "base_security_implementation",
    }
    snapshots: dict[str, _Snapshot] = {"config": config_snapshot}
    for role, field in path_fields.items():
        path = _v1._safe_repo_path(
            repo_root,
            paths.get(field),
            f"synthetic_v2_{role}_path_invalid",
        )
        snapshots[role] = _v1._read_snapshot(path, f"synthetic_v2_{role}_unreadable")
    config_schema = _v1._strict_json(
        snapshots["config_schema"].data, "synthetic_v2_config_schema_invalid"
    )
    grammar = _v1._strict_json(
        snapshots["closed_grammar"].data, "synthetic_v2_grammar_invalid"
    )
    grammar_schema = _v1._strict_json(
        snapshots["closed_grammar_schema"].data,
        "synthetic_v2_grammar_schema_invalid",
    )
    catalog = _v1._strict_json(
        snapshots["bundle_catalog"].data, "synthetic_v2_catalog_invalid"
    )
    catalog_schema = _v1._strict_json(
        snapshots["bundle_catalog_schema"].data,
        "synthetic_v2_catalog_schema_invalid",
    )
    record_schema = _v1._strict_json(
        snapshots["record_schema"].data, "synthetic_v2_record_schema_invalid"
    )
    manifest_schema = _v1._strict_json(
        snapshots["manifest_schema"].data, "synthetic_v2_manifest_schema_invalid"
    )
    _v1._validate_schema(
        config_schema, config, "synthetic_v2_config_schema_validation_failed"
    )
    _v1._validate_schema(
        grammar_schema,
        grammar,
        "synthetic_v2_grammar_schema_validation_failed",
    )
    _v1._validate_schema(
        catalog_schema,
        catalog,
        "synthetic_v2_catalog_schema_validation_failed",
    )
    _validate_config(config)
    _validate_grammar(grammar)
    _validate_catalog(catalog)
    return snapshots, config, grammar, catalog, record_schema, manifest_schema


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_VERSION:
        _fail("synthetic_v2_config_version_invalid")
    if config.get("claim_scope") != CLAIM_SCOPE:
        _fail("synthetic_v2_claim_scope_invalid")
    expected_paths = {
        "config_schema": CONFIG_SCHEMA_PATH,
        "closed_grammar": CLOSED_GRAMMAR_PATH,
        "closed_grammar_schema": CLOSED_GRAMMAR_SCHEMA_PATH,
        "bundle_catalog": CATALOG_PATH,
        "bundle_catalog_schema": CATALOG_SCHEMA_PATH,
        "record_schema": RECORD_SCHEMA_PATH,
        "manifest_schema": MANIFEST_SCHEMA_PATH,
        "implementation": IMPLEMENTATION_PATH,
        "base_security_implementation": BASE_SECURITY_IMPLEMENTATION_PATH,
    }
    if dict(config["paths"]) != expected_paths:
        _fail("synthetic_v2_paths_drift")
    dataset = config["dataset_contract"]
    expected_dataset = {
        "producer_version": PRODUCER_VERSION,
        "record_schema_version": RECORD_VERSION,
        "manifest_schema_version": MANIFEST_VERSION,
        "stratum_count": 5,
        "language_count": 2,
        "bundle_count": 200,
        "roles_per_bundle": 5,
        "variants_per_role": 1,
        "primary_views_per_role": 1,
        "record_count": 1000,
        "pair_count": 0,
        "records_per_role": 200,
        "replaces_100_v1": False,
        "satisfies_independent_600_materialization": False,
        "satisfies_factorial_600_materialization": False,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
    }
    if dict(dataset) != expected_dataset:
        _fail("synthetic_v2_dataset_contract_drift")
    if config["authorship_contract"] != {
        "catalog_authored_with_openai_codex_gpt_5_6_sol_assistance": True,
        "deterministic_builder_provider_requests": 0,
        "content_authorship_is_distinct_from_builder_runtime_requests": True,
        "acknowledgement": "OpenAI GPT-5.6-sol assisted dataset authorship.",
    }:
        _fail("synthetic_v2_authorship_contract_drift")
    generation = config["generation_contract"]
    if generation != {
        "seed_id": SEED_ID,
        "seed_text": "anchor-synthetic-five-role-qonly-generation-seed-v1",
        "source_namespace": SOURCE_NAMESPACE,
        "catalog_expansion": "one_primary_view_per_bundle_role_v1",
        "augmentation": "none",
        "split_before_augmentation": True,
        "declared_semantic_read_set_only": True,
    }:
        _fail("synthetic_v2_generation_contract_drift")
    if config["semantic_identity_contract"] != {
        "algorithm": ("canonical_json_domain_plus_semantic_identity_utf8_sha256_v1"),
        "bundle_or_language_salt_used": False,
        "required_unique_task_semantics": 200,
        "each_role_covers_same_semantic_set": True,
        "en_zh_intersection_count": 0,
        "translation_pair_count": 0,
        "legacy_domain_overlap_status": "unavailable_not_claimed",
    }:
        _fail("synthetic_v2_semantic_identity_contract_drift")
    protected = config["protected_inventory_contract"]
    if (
        protected["consumes_protected_inventories"] is not False
        or protected["zero_intersection_claimed"] is not False
        or protected["source_disjoint_attestation_emitted"] is not False
        or protected["formal_source_disjoint_proven"] is not False
        or set(protected["statuses"])
        != {
            "swebench_source",
            "gold_partition",
            "partial_gold_export",
            "heldout",
            "legacy_heldout_cases",
            "synthetic_scaffold",
        }
        or set(protected["statuses"].values()) != {"unavailable_not_read"}
    ):
        _fail("synthetic_v2_protected_contract_drift")
    split = config["split_contract"]
    if split != {
        "group_key": "task_bundle_sha256",
        "split_before_role_view_expansion": True,
        "algorithm": (
            "sha256_utf8_salt_nul_language_nul_stratum_nul_bundle_key_"
            "sort_ascending_eval_first_four_per_cell_v1"
        ),
        "salt": "anchor-synthetic-five-role-qonly-split-v1",
        "train_bundle_count": 160,
        "eval_proxy_bundle_count": 40,
        "eval_proxy_is_heldout": False,
        "all_role_views_same_split": True,
        "language_stratified": {
            "en": {"train_bundles": 80, "eval_proxy_bundles": 20},
            "zh-CN": {"train_bundles": 80, "eval_proxy_bundles": 20},
        },
    }:
        _fail("synthetic_v2_split_contract_drift")
    ablation = config["ablation_contract"]
    if ablation != {
        "primary_label": PRIMARY_LABEL,
        "q_only_is_only_primary": True,
        "diagnostic_control_labels": list(DIAGNOSTIC_CONTROLS),
        "all_records_eligible_for_all_labels": True,
        "arm_assignment_location": "diagnostic_run_manifest_only",
        "identical_record_inventory_required": True,
        "producer_selects_winner": False,
        "legacy_wide_lora_control_inherited": False,
        "producer_claims_training_outcome": False,
    }:
        _fail("synthetic_v2_ablation_contract_drift")
    if config["role_contract"]["ordered_roles"] != list(ROLES):
        _fail("synthetic_v2_role_contract_drift")
    if config["view_contract"] != {
        "primary_view": PRIMARY_VIEW,
        "views_per_bundle_role": 1,
        "json_only_shadow_materialized": False,
        "concise_rationale_is_hidden_chain_of_thought": False,
        "concise_rationale_is_auditable_decision_summary": True,
    }:
        _fail("synthetic_v2_view_contract_drift")
    if config["audit_contract"]["maximum_artifact_file_bytes"] != _MAX_BYTES:
        _fail("synthetic_v2_maximum_bytes_drift")


def _validate_grammar(grammar: Mapping[str, Any]) -> None:
    if grammar.get("schema_version") != GRAMMAR_VERSION:
        _fail("synthetic_v2_grammar_version_invalid")
    if grammar.get("source_namespace") != SOURCE_NAMESPACE:
        _fail("synthetic_v2_grammar_namespace_invalid")
    if tuple(grammar.get("languages", ())) != LANGUAGES:
        _fail("synthetic_v2_grammar_languages_invalid")
    if tuple(grammar.get("roles", ())) != ROLES:
        _fail("synthetic_v2_grammar_roles_invalid")
    if grammar.get("primary_view") != PRIMARY_VIEW:
        _fail("synthetic_v2_grammar_primary_view_invalid")
    if grammar.get("primary_label") != PRIMARY_LABEL:
        _fail("synthetic_v2_grammar_primary_invalid")
    if tuple(grammar.get("diagnostic_control_labels", ())) != DIAGNOSTIC_CONTROLS:
        _fail("synthetic_v2_grammar_controls_invalid")


def _validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema_version") != CATALOG_VERSION:
        _fail("synthetic_v2_catalog_version_invalid")
    if catalog.get("source_namespace") != SOURCE_NAMESPACE:
        _fail("synthetic_v2_catalog_namespace_invalid")
    bundles = catalog.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 200:
        _fail("synthetic_v2_catalog_count_invalid")
    if len({item["bundle_key"] for item in bundles}) != 200:
        _fail("synthetic_v2_catalog_bundle_key_collision")
    semantic_scalar_fields = (
        "domain",
        "intent",
        "scenario_atom",
        "artifact_kind",
        "evidence_topology",
        "state_transition",
    )
    for item in bundles:
        semantic = item["semantic_identity"]
        scalar_values = [str(semantic[field]) for field in semantic_scalar_fields]
        scalar_values.extend(str(value) for value in semantic["constraint_tags"])
        scalar_values.extend(str(value) for value in semantic["acceptance_tags"])
        if any(not _SEMANTIC_ATOM.fullmatch(value) for value in scalar_values):
            _fail("synthetic_v2_catalog_semantic_atom_invalid")

        def normalized(value: object) -> str:
            return re.sub(
                r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())
            ).strip("_")

        normalized_values = [normalized(value) for value in scalar_values]
        whole_salts = [
            normalized(item["bundle_key"]),
            normalized(SOURCE_NAMESPACE),
        ]
        normalized_task_text = normalized(item["task_text"])
        if len(normalized_task_text) >= 16:
            whole_salts.append(normalized_task_text)
        token_salts = {
            normalized(item["language"]),
            *(normalized(role) for role in ROLES),
        }
        for value in normalized_values:
            padded = f"_{value}_"
            if any(salt and salt in value for salt in whole_salts) or any(
                salt and f"_{salt}_" in padded for salt in token_salts
            ):
                _fail("synthetic_v2_catalog_semantic_salt_embedded")
    if Counter(item["language"] for item in bundles) != Counter(
        {"en": 100, "zh-CN": 100}
    ):
        _fail("synthetic_v2_catalog_language_imbalance")
    strata = Counter((item["language"], item["stratum"]) for item in bundles)
    expected = Counter(
        (language, stratum)
        for language in LANGUAGES
        for stratum in (
            "prefix_evidence_selection",
            "prefix_evidence_plus_structured_private_writeback",
            "conflicting_allowed_evidence_resolution",
            "tool_result_commit_then_expert_private_tail",
            "ordered_long_prefix_retrieval",
        )
        for _ in range(20)
    )
    if strata != expected:
        _fail("synthetic_v2_catalog_stratum_imbalance")
    semantic_hashes = [_task_semantic_sha256(item) for item in bundles]
    if any(
        item["task_semantic_sha256"] != semantic_hashes[index]
        for index, item in enumerate(bundles)
    ):
        _fail("synthetic_v2_catalog_semantic_hash_mismatch")
    if len(set(semantic_hashes)) != 200:
        _fail("synthetic_v2_catalog_semantic_identity_collision")
    by_language = {
        language: {
            semantic_hashes[index]
            for index, item in enumerate(bundles)
            if item["language"] == language
        }
        for language in LANGUAGES
    }
    if by_language["en"] & by_language["zh-CN"]:
        _fail("synthetic_v2_catalog_translation_pair_detected")


def _catalog_bundles(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(catalog["bundles"])


def _task_semantic_sha256(bundle: Mapping[str, Any]) -> str:
    """Hash a language-neutral semantic preimage without bundle/language salt."""

    return _canonical_sha256(
        {
            "domain": "anchor.synthetic-five-role-task-semantic.v1",
            "semantic_identity": bundle["semantic_identity"],
        }
    )


def _bundle_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": "anchor.synthetic-five-role-qonly-task-bundle.v1",
        "source_namespace": SOURCE_NAMESPACE,
        "bundle_key": bundle["bundle_key"],
        "language": bundle["language"],
        "archetype": bundle["archetype"],
        "stratum": bundle["stratum"],
        "task_text": bundle["task_text"],
        "constraints": list(bundle["constraints"]),
        "acceptance_hints": list(bundle["acceptance_hints"]),
        "task_semantic_sha256": _task_semantic_sha256(bundle),
    }


def _bundle_identity(bundle: Mapping[str, Any]) -> tuple[str, str]:
    digest = _canonical_sha256(_bundle_payload(bundle))
    return f"syn-qonly-bundle-v1:{digest}", digest


def _chain_root_sha256(bundle: Mapping[str, Any], task_bundle_sha256: str) -> str:
    return _canonical_sha256(
        {
            "domain": "anchor.synthetic-five-role-qonly-chain-root.v1",
            "task_bundle_sha256": task_bundle_sha256,
            "task_semantic_sha256": _task_semantic_sha256(bundle),
            "ordered_roles": list(ROLES),
            "canonical_stages": list(CANONICAL_STAGES),
        }
    )


def _split_bundles(
    config: Mapping[str, Any], bundles: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, str], list[str], list[str]]:
    salt = str(config["split_contract"]["salt"])
    assignments: dict[str, str] = {}
    strata = sorted({str(bundle["stratum"]) for bundle in bundles})
    for language in LANGUAGES:
        for stratum in strata:
            candidates: list[tuple[str, str]] = []
            for bundle in bundles:
                if bundle["language"] != language or bundle["stratum"] != stratum:
                    continue
                _, bundle_sha = _bundle_identity(bundle)
                score = _sha256(
                    (f"{salt}\0{language}\0{stratum}\0{bundle['bundle_key']}").encode(
                        "utf-8"
                    )
                )
                candidates.append((score, bundle_sha))
            candidates.sort()
            if len(candidates) != 20:
                _fail("synthetic_v2_split_cell_size_invalid")
            for index, (_, bundle_sha) in enumerate(candidates):
                assignments[bundle_sha] = "eval_proxy" if index < 4 else "train"
    train = sorted(key for key, value in assignments.items() if value == "train")
    eval_proxy = sorted(
        key for key, value in assignments.items() if value == "eval_proxy"
    )
    if len(train) != 160 or len(eval_proxy) != 40:
        _fail("synthetic_v2_bundle_split_invalid")
    return assignments, train, eval_proxy


def _role_spec(bundle: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    value = bundle["role_specs"][role]
    if not isinstance(value, Mapping):
        _fail("synthetic_v2_role_spec_invalid")
    return value


def _make_segments(
    bundle: Mapping[str, Any],
    task_bundle_id: str,
    grammar: Mapping[str, Any],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for stage_index, role in enumerate(ROLES):
        role_spec = _role_spec(bundle, role)
        summary = str(role_spec["committed_summary"])
        refs = [f"S{index}" for index in range(stage_index)]
        route = {
            "role": role,
            "canonical_stage": ROLE_TO_CANONICAL_STAGE[role],
            "expert": f"{role}_expert",
            "goal": str(role_spec["goal"]),
            "constraints": list(role_spec["constraints"]),
            "evidence_intent": str(role_spec["evidence_intent"]),
            "allowed_segment_refs": refs,
            "evidence_segment_refs": refs,
            "tool_plan": [
                {
                    "action": str(role_spec["tool_action"]),
                    "mode": grammar["tool_trace_mode"],
                }
            ],
            "acceptance_criteria": list(role_spec["acceptance_criteria"]),
        }
        content = _canonical_json(route)
        content_sha256 = _sha256(content.encode("utf-8"))
        segment_id = _id(
            "syn-qonly-segment-v1",
            {
                "domain": "anchor.synthetic-five-role-qonly-segment-id.v1",
                "task_bundle_id": task_bundle_id,
                "role": role,
                "stage_index": stage_index,
                "content_sha256": content_sha256,
            },
        )
        segments.append(
            {
                "segment_id": segment_id,
                "segment_ref": f"S{stage_index}",
                "role": role,
                "canonical_stage": ROLE_TO_CANONICAL_STAGE[role],
                "stage_index": stage_index,
                "content": content,
                "content_sha256": content_sha256,
                "commit_summary": summary,
                "commit_summary_sha256": _sha256(summary.encode("utf-8")),
                "route": route,
            }
        )
    return segments


def build_training_view(
    bundle: Mapping[str, Any],
    role: str,
    segments: Sequence[Mapping[str, Any]],
    grammar: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the real prompt without current or future target bodies."""

    if role not in ROLES or len(segments) != len(ROLES):
        _fail("synthetic_v2_training_view_invalid")
    stage_index = ROLES.index(role)
    instruction = str(_role_spec(bundle, role)["instruction"])
    allowed_context = [
        {
            "segment_ref": segment["segment_ref"],
            "role": segment["role"],
            "canonical_stage": segment["canonical_stage"],
            "stage_index": segment["stage_index"],
            "committed_summary": segment["commit_summary"],
        }
        for segment in segments[:stage_index]
    ]
    lines = [
        "[SYNTHETIC_DIAGNOSTIC_TASK]",
        str(bundle["task_text"]),
        "[CONSTRAINTS]",
        *[f"- {item}" for item in bundle["constraints"]],
        "[ACCEPTANCE_HINTS]",
        *[f"- {item}" for item in bundle["acceptance_hints"]],
        "[ROLE]",
        role,
        "[ROLE_INSTRUCTION]",
        instruction,
        "[COMMITTED_CONTEXT]",
    ]
    if not allowed_context:
        lines.append("(none)")
    else:
        for item in allowed_context:
            lines.extend(
                [
                    f"[{item['segment_ref']}|{item['role']}]",
                    str(item["committed_summary"]),
                ]
            )
    lines.extend(
        [
            "[OUTPUT_CONTRACT]",
            "Return one concise auditable summary followed by canonical JSON.",
        ]
    )
    payload = {
        "task_text": bundle["task_text"],
        "constraints": list(bundle["constraints"]),
        "acceptance_hints": list(bundle["acceptance_hints"]),
        "role_instruction": instruction,
        "allowed_context_segments": allowed_context,
        "materialized_prompt": "\n".join(lines) + "\n",
    }
    payload["input_sha256"] = _canonical_sha256(payload)
    return payload


def _record_view(
    bundle: Mapping[str, Any],
    task_bundle_id: str,
    task_bundle_sha256: str,
    split: str,
    role: str,
    segments: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, _Snapshot],
    grammar: Mapping[str, Any],
    seed_sha256: str,
) -> list[dict[str, Any]]:
    stage_index = ROLES.index(role)
    task_semantic_sha256 = _task_semantic_sha256(bundle)
    chain_root_sha256 = _chain_root_sha256(bundle, task_bundle_sha256)
    inner_task_id = f"syn-qonly-task-v1:{task_semantic_sha256}"
    input_view = build_training_view(bundle, role, segments, grammar)
    forbidden = [segment["segment_id"] for segment in segments[stage_index:]]
    inventory = [
        {
            "segment_id": segment["segment_id"],
            "segment_ref": segment["segment_ref"],
            "role": segment["role"],
            "canonical_stage": segment["canonical_stage"],
            "stage_index": segment["stage_index"],
            "content_sha256": segment["content_sha256"],
            "visibility": (
                "previous_committed"
                if segment["stage_index"] < stage_index
                else "current_target"
                if segment["stage_index"] == stage_index
                else "future_target"
            ),
        }
        for segment in segments
    ]
    route = dict(segments[stage_index]["route"])
    canonical_route = _canonical_json(route)
    canonical_route_sha = _sha256(canonical_route.encode("utf-8"))
    role_view_id = _id(
        "syn-qonly-role-view-v1",
        {
            "domain": "anchor.synthetic-five-role-qonly-role-view-id.v1",
            "task_bundle_sha256": task_bundle_sha256,
            "task_semantic_sha256": task_semantic_sha256,
            "chain_root_sha256": chain_root_sha256,
            "role": role,
            "canonical_stage": ROLE_TO_CANONICAL_STAGE[role],
            "input_sha256": input_view["input_sha256"],
            "canonical_json_sha256": canonical_route_sha,
            "generator_config_sha256": snapshots["config"].sha256,
            "generator_implementation_sha256": snapshots["implementation"].sha256,
            "closed_grammar_sha256": snapshots["closed_grammar"].sha256,
            "generation_seed_sha256": seed_sha256,
        },
    )
    summary = str(_role_spec(bundle, role)["committed_summary"])
    output = f"{summary}\n{canonical_route}"
    output_sha = _sha256(output.encode("utf-8"))
    record_id = _id(
        "syn-qonly-record-v1",
        {
            "domain": "anchor.synthetic-five-role-qonly-record-id.v1",
            "role_view_id": role_view_id,
            "view": PRIMARY_VIEW,
            "output_sha256": output_sha,
        },
    )
    return [
        {
            "schema_version": RECORD_VERSION,
            "record_id": record_id,
            "role_view_id": role_view_id,
            "task_bundle_id": task_bundle_id,
            "task_bundle_sha256": task_bundle_sha256,
            "task_semantic_sha256": task_semantic_sha256,
            "inner_task_id": inner_task_id,
            "chain_root_sha256": chain_root_sha256,
            "split": split,
            "language": bundle["language"],
            "stratum": bundle["stratum"],
            "role": role,
            "canonical_stage": ROLE_TO_CANONICAL_STAGE[role],
            "stage_index": stage_index,
            "view": PRIMARY_VIEW,
            "synthetic_source": {
                "kind": "closed_grammar_synthetic_no_external_body",
                "bundle_key": bundle["bundle_key"],
                "archetype": bundle["archetype"],
                "stratum": bundle["stratum"],
                "closed_grammar_id": GRAMMAR_VERSION,
                "closed_grammar_sha256": snapshots["closed_grammar"].sha256,
                "generation_seed_id": SEED_ID,
                "generation_seed_sha256": seed_sha256,
                "generator_config_sha256": snapshots["config"].sha256,
                "generator_implementation_sha256": snapshots["implementation"].sha256,
                "base_security_implementation_sha256": snapshots[
                    "base_security_implementation"
                ].sha256,
            },
            "input": input_view,
            "board_segment_inventory": inventory,
            "forbidden_segment_ids": forbidden,
            "target": {
                "canonical_routing_json": route,
                "canonical_json_sha256": canonical_route_sha,
                "concise_rationale_summary": summary,
                "serialized_assistant_output": output,
                "output_sha256": output_sha,
            },
            "route_boundary": {
                "semantics": "explicit_two_request_commit",
                "activation_semantics": "next_request_input_activation_only",
                "committed_scaffold_reencode_required": True,
                "planner_request1_private_kv_reused": False,
            },
            "ablation": {
                "arm_neutral": True,
                "assignment_location": "diagnostic_run_manifest_only",
                "identical_inventory_required": True,
                "row_duplicated_for_controls": False,
                "serialized_into_prompt_or_target": False,
            },
            "claims": {
                "diagnostic_only": True,
                "formal": False,
                "training_authorized": False,
                "quality_validated": False,
                "proxy_signal_only": True,
                "physical_kv_reuse_claimed": False,
                "numeric_equivalence_claimed": False,
            },
            "audit": {
                "protected_body_reads": 0,
                "provider_requests": 0,
                "network_requests": 0,
                "model_loads": 0,
                "gpu_requests": 0,
                "real_tool_executions": 0,
            },
        }
    ]


def _generate_records(
    config: Mapping[str, Any],
    grammar: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
    catalog: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    bundles = _catalog_bundles(catalog)
    assignments, train_bundles, eval_bundles = _split_bundles(config, bundles)
    seed_sha = _sha256(str(config["generation_contract"]["seed_text"]).encode("utf-8"))
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id, bundle_sha = _bundle_identity(bundle)
        segments = _make_segments(bundle, bundle_id, grammar)
        for role in ROLES:
            records.extend(
                _record_view(
                    bundle,
                    bundle_id,
                    bundle_sha,
                    assignments[bundle_sha],
                    role,
                    segments,
                    snapshots,
                    grammar,
                    seed_sha,
                )
            )
    if len(records) != 1000:
        _fail("synthetic_v2_record_count_invalid")
    return records, train_bundles, eval_bundles


def _partition_bytes(records: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for split, view in PARTITION_KEYS:
        selected = [
            record
            for record in records
            if record["split"] == split and record["view"] == view
        ]
        selected.sort(
            key=lambda item: (
                item["task_bundle_sha256"],
                ROLES.index(str(item["role"])),
                item["record_id"],
            )
        )
        result[f"{split}/{view}.jsonl"] = b"".join(
            _canonical_json_bytes(record, newline=True) for record in selected
        )
    return result


def _validate_records(
    config: Mapping[str, Any],
    grammar: Mapping[str, Any],
    catalog: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    record_schema: Mapping[str, Any],
) -> None:
    _v1._reject_external_refs(record_schema, "synthetic_v2_record_schema_external_ref")
    _v1.Draft202012Validator.check_schema(record_schema)
    validator = _v1.Draft202012Validator(record_schema)
    for record in records:
        try:
            validator.validate(record)
        except Exception as exc:
            raise SyntheticScaffoldDiagnosticV2Error(
                "synthetic_v2_record_schema_failed"
            ) from exc
    if len(records) != 1000 or len({item["record_id"] for item in records}) != 1000:
        _fail("synthetic_v2_record_identity_invalid")
    bundles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        bundles[str(record["task_bundle_sha256"])].append(record)
    if len({item["role_view_id"] for item in records}) != 1000 or len(bundles) != 200:
        _fail("synthetic_v2_role_view_or_bundle_count_invalid")
    for bundle_records in bundles.values():
        if (
            len(bundle_records) != 5
            or len({item["split"] for item in bundle_records}) != 1
        ):
            _fail("synthetic_v2_bundle_split_leakage")
        if Counter(item["role"] for item in bundle_records) != Counter(
            {role: 1 for role in ROLES}
        ):
            _fail("synthetic_v2_bundle_role_imbalance")
        for key in (
            "task_semantic_sha256",
            "inner_task_id",
            "chain_root_sha256",
            "task_bundle_id",
        ):
            if len({item[key] for item in bundle_records}) != 1:
                _fail("synthetic_v2_bundle_chain_binding_invalid")
    if Counter(record["role"] for record in records) != Counter(
        {role: 200 for role in ROLES}
    ):
        _fail("synthetic_v2_global_role_imbalance")
    expected_cells = Counter()
    strata = sorted({item["stratum"] for item in catalog["bundles"]})
    for language in LANGUAGES:
        for stratum in strata:
            for role in ROLES:
                expected_cells[("train", role, language, stratum)] = 16
                expected_cells[("eval_proxy", role, language, stratum)] = 4
    actual_cells = Counter(
        (item["split"], item["role"], item["language"], item["stratum"])
        for item in records
    )
    if actual_cells != expected_cells:
        _fail("synthetic_v2_cell_balance_invalid")
    semantic_by_role = {
        role: {
            str(item["task_semantic_sha256"])
            for item in records
            if item["role"] == role
        }
        for role in ROLES
    }
    if (
        any(len(values) != 200 for values in semantic_by_role.values())
        or len({tuple(sorted(values)) for values in semantic_by_role.values()}) != 1
    ):
        _fail("synthetic_v2_role_semantic_inventory_invalid")
    semantic_by_language = {
        language: {
            str(item["task_semantic_sha256"])
            for item in records
            if item["language"] == language
        }
        for language in LANGUAGES
    }
    if semantic_by_language["en"] & semantic_by_language["zh-CN"]:
        _fail("synthetic_v2_translation_pair_detected")
    bundle_by_key = {
        bundle["bundle_key"]: bundle for bundle in _catalog_bundles(catalog)
    }
    for record in records:
        bundle = bundle_by_key[str(record["synthetic_source"]["bundle_key"])]
        bundle_id, bundle_sha = _bundle_identity(bundle)
        if (
            record["task_bundle_id"] != bundle_id
            or record["task_bundle_sha256"] != bundle_sha
            or record["task_semantic_sha256"] != _task_semantic_sha256(bundle)
            or record["inner_task_id"]
            != f"syn-qonly-task-v1:{_task_semantic_sha256(bundle)}"
            or record["chain_root_sha256"] != _chain_root_sha256(bundle, bundle_sha)
            or record["canonical_stage"] != ROLE_TO_CANONICAL_STAGE[str(record["role"])]
            or record["view"] != PRIMARY_VIEW
        ):
            _fail("synthetic_v2_task_bundle_binding_invalid")
        segments = _make_segments(bundle, bundle_id, grammar)
        expected_input = build_training_view(
            bundle, str(record["role"]), segments, grammar
        )
        if record["input"] != expected_input:
            _fail("synthetic_v2_materialized_input_mismatch")
        stage = int(record["stage_index"])
        prompt = str(record["input"]["materialized_prompt"])
        serialized_input = _canonical_json(record["input"])
        expected_inventory = [
            {
                "segment_id": segment["segment_id"],
                "segment_ref": segment["segment_ref"],
                "role": segment["role"],
                "canonical_stage": segment["canonical_stage"],
                "stage_index": segment["stage_index"],
                "content_sha256": segment["content_sha256"],
                "visibility": (
                    "previous_committed"
                    if segment["stage_index"] < stage
                    else "current_target"
                    if segment["stage_index"] == stage
                    else "future_target"
                ),
            }
            for segment in segments
        ]
        expected_forbidden = [segment["segment_id"] for segment in segments[stage:]]
        visibility_forbidden = [
            item["segment_id"]
            for item in record["board_segment_inventory"]
            if item["visibility"] in {"current_target", "future_target"}
        ]
        if record["board_segment_inventory"] != expected_inventory:
            _fail("synthetic_v2_segment_inventory_mismatch")
        if (
            record["forbidden_segment_ids"] != expected_forbidden
            or visibility_forbidden != expected_forbidden
        ):
            _fail("synthetic_v2_forbidden_inventory_mismatch")
        for segment in segments[stage:]:
            forbidden_output = f"{segment['commit_summary']}\n{segment['content']}"
            forbidden_output_sha256 = _sha256(forbidden_output.encode("utf-8"))
            forbidden_needles = (
                segment["segment_id"],
                segment["segment_ref"],
                segment["content"],
                segment["content_sha256"],
                segment["commit_summary"],
                segment["commit_summary_sha256"],
                forbidden_output,
                forbidden_output_sha256,
            )
            if any(needle in prompt for needle in forbidden_needles) or any(
                needle in serialized_input for needle in forbidden_needles
            ):
                _fail("synthetic_v2_forbidden_content_leaked")
        for segment in segments[:stage]:
            if (
                segment["segment_ref"] not in prompt
                or segment["commit_summary"] not in prompt
                or segment["segment_id"] in prompt
                or segment["content"] in prompt
                or segment["segment_id"] in serialized_input
                or segment["content_sha256"] in serialized_input
                or segment["commit_summary_sha256"] in serialized_input
            ):
                _fail("synthetic_v2_allowed_content_invalid")
        target = record["target"]
        canonical = _canonical_json(target["canonical_routing_json"])
        canonical_sha = _sha256(canonical.encode("utf-8"))
        summary = target["concise_rationale_summary"]
        expected_output = canonical if summary is None else f"{summary}\n{canonical}"
        if (
            target["canonical_json_sha256"] != canonical_sha
            or target["serialized_assistant_output"] != expected_output
            or target["output_sha256"] != _sha256(expected_output.encode("utf-8"))
        ):
            _fail("synthetic_v2_target_normal_form_invalid")
        if record["ablation"] != {
            "arm_neutral": True,
            "assignment_location": "diagnostic_run_manifest_only",
            "identical_inventory_required": True,
            "row_duplicated_for_controls": False,
            "serialized_into_prompt_or_target": False,
        }:
            _fail("synthetic_v2_row_arm_neutrality_invalid")


def _build_manifest(
    repo_root: Path,
    config: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
    records: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, bytes],
    train_bundles: Sequence[str],
    eval_bundles: Sequence[str],
) -> dict[str, Any]:
    record_ids = [str(item["record_id"]) for item in records]
    bundle_ids = sorted({str(item["task_bundle_sha256"]) for item in records})
    semantic_ids = sorted({str(item["task_semantic_sha256"]) for item in records})
    record_content = _inventory_sha256(
        "anchor.synthetic-nl-scaffold-record-content-inventory.v2",
        [_sha256(_canonical_json_bytes(item)) for item in records],
    )
    read_roles = (
        "config",
        "config_schema",
        "closed_grammar",
        "closed_grammar_schema",
        "bundle_catalog",
        "bundle_catalog_schema",
        "record_schema",
        "manifest_schema",
        "implementation",
        "base_security_implementation",
    )
    read_set = [_artifact_descriptor(repo_root, snapshots[role]) for role in read_roles]
    partition_entries = []
    for split, view in PARTITION_KEYS:
        path = f"{split}/{view}.jsonl"
        raw = partitions[path]
        partition_entries.append(
            {
                "path": path,
                "split": split,
                "view": view,
                "records": len(raw.splitlines()),
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
    return {
        "schema_version": MANIFEST_VERSION,
        "status": "dataset_proxy_ready_training_not_authorized",
        "claim_scope": CLAIM_SCOPE,
        "producer": {
            "producer_version": PRODUCER_VERSION,
            **{
                role: _artifact_descriptor(repo_root, snapshots[role])
                for role in read_roles
            },
            "canonical_json_policy": ("utf8_sort_keys_compact_no_normalization_lf_v1"),
            "atomic_publish": True,
            "final_toctou_recheck": True,
        },
        "counts": dict(EXPECTED_COUNTS),
        "role_counts": {
            role: {"total": 200, "train": 160, "eval_proxy": 40} for role in ROLES
        },
        "authorship": {
            "catalog_authored_with_openai_codex_gpt_5_6_sol_assistance": True,
            "catalog_content_is_not_reported_as_zero_model_authored": True,
            "deterministic_build_provider_requests": 0,
            "acknowledgement": "OpenAI GPT-5.6-sol assisted dataset authorship.",
        },
        "generation_contract": {
            "seed_id": SEED_ID,
            "seed_sha256": _sha256(
                str(config["generation_contract"]["seed_text"]).encode("utf-8")
            ),
            "source_namespace": SOURCE_NAMESPACE,
            "catalog_expansion": config["generation_contract"]["catalog_expansion"],
            "augmentation": "none",
            "split_before_augmentation": True,
        },
        "split_contract": {
            "group_key": "task_bundle_sha256",
            "algorithm": config["split_contract"]["algorithm"],
            "salt_sha256": _sha256(
                str(config["split_contract"]["salt"]).encode("utf-8")
            ),
            "train_task_bundle_sha256": list(train_bundles),
            "eval_proxy_task_bundle_sha256": list(eval_bundles),
            "all_role_views_same_split": True,
            "eval_proxy_is_heldout": False,
        },
        "partitions": partition_entries,
        "logical_role_partitions": [
            {
                "role": role,
                "split": split,
                "records": len(selected),
                "record_ids_sha256": _inventory_sha256(
                    "anchor.synthetic-five-role-logical-role-partition-records.v1",
                    [str(item["record_id"]) for item in selected],
                ),
                "record_content_sha256": _inventory_sha256(
                    "anchor.synthetic-five-role-logical-role-partition-content.v1",
                    [_sha256(_canonical_json_bytes(item)) for item in selected],
                ),
            }
            for role in ROLES
            for split in ("train", "eval_proxy")
            for selected in (
                [
                    item
                    for item in records
                    if item["role"] == role and item["split"] == split
                ],
            )
        ],
        "inventories": {
            "record_ids_sha256": _inventory_sha256(
                "anchor.synthetic-nl-scaffold-record-inventory.v2", record_ids
            ),
            "record_content_sha256": record_content,
            "task_bundle_sha256_inventory": _inventory_sha256(
                "anchor.synthetic-five-role-task-bundle-inventory.v1", bundle_ids
            ),
            "task_semantic_sha256_inventory": _inventory_sha256(
                "anchor.synthetic-five-role-task-semantic-inventory.v1",
                semantic_ids,
            ),
            "role_record_inventory_sha256": {
                role: _inventory_sha256(
                    "anchor.synthetic-five-role-role-record-inventory.v1",
                    [
                        str(item["record_id"])
                        for item in records
                        if item["role"] == role
                    ],
                )
                for role in ROLES
            },
            "role_task_semantic_inventory_sha256": {
                role: _inventory_sha256(
                    "anchor.synthetic-five-role-role-semantic-inventory.v1",
                    [
                        str(item["task_semantic_sha256"])
                        for item in records
                        if item["role"] == role
                    ],
                )
                for role in ROLES
            },
            "arm_record_inventory_sha256": {
                PRIMARY_LABEL: record_content,
                **{label: record_content for label in DIAGNOSTIC_CONTROLS},
            },
        },
        "semantic_identity_contract": {
            "canonical_preimage": (
                "canonical_json({domain,semantic_identity})_utf8_sha256_v1"
            ),
            "bundle_or_language_salt_used": False,
            "unique_task_semantics": 200,
            "each_role_covers_same_200_semantics": True,
            "en_unique_task_semantics": 100,
            "zh_cn_unique_task_semantics": 100,
            "en_zh_intersection_count": 0,
            "translation_pair_count": 0,
            "legacy_domain_overlap_status": "unavailable_not_claimed",
        },
        "task_bundle_identity_contract": {
            "preimage_excludes_role": True,
            "preimage_excludes_view": True,
            "preimage_excludes_arm": True,
            "preimage_excludes_noise": True,
            "split_group_key": "task_bundle_sha256",
        },
        "read_set": {
            "scope": "declared_semantic_generation_inputs_only",
            "ordered_artifacts": read_set,
            "ordered_artifacts_sha256": _inventory_sha256(
                "anchor.synthetic-nl-scaffold-read-set.v2",
                [f"{item['path']}:{item['sha256']}" for item in read_set],
            ),
            "protected_source_paths_read": 0,
        },
        "protected_inventory_status": {
            "consumes_protected_inventories": False,
            "statuses": dict(config["protected_inventory_contract"]["statuses"]),
        },
        "source_disjoint_boundary": {
            "source_namespace": SOURCE_NAMESPACE,
            "zero_intersection_claimed": False,
            "source_disjoint_attestation_emitted": False,
            "formal_source_disjoint_proven": False,
            "status": "unavailable_without_protected_inventory_identities",
        },
        "ablation_contract": {
            "primary_label": PRIMARY_LABEL,
            "q_only_is_only_primary": True,
            "diagnostic_control_labels": list(DIAGNOSTIC_CONTROLS),
            "same_record_inventory_for_all_arms": True,
            "assignment_location": "diagnostic_run_manifest_only",
            "producer_selects_winner": False,
            "legacy_wide_lora_control_inherited": False,
            "target_modules_bound_by_dataset": False,
            "control_arms_are_execution_overlays_only": True,
            "control_arm_rows_materialized": False,
        },
        "compatibility_boundary": {
            "variants_per_role": 1,
            "pair_count": 0,
            "replaces_100_v1": False,
            "satisfies_independent_600_materialization": False,
            "satisfies_factorial_600_materialization": False,
        },
        "capability_coverage": {
            "schema_version": ("anchor.synthetic-five-role-capability-coverage.v1"),
            "source": "catalog_explicit_labels_only",
            "adds_rows_views_or_arms": False,
            "simple_tool_search": {
                "status": "unavailable_no_explicit_catalog_label",
                "task_bundle_count": None,
                "quota_claimed": False,
            },
            "micro_coding": {
                "status": "unavailable_no_explicit_catalog_label",
                "task_bundle_count": None,
                "quota_claimed": False,
            },
        },
        "claims": {
            "dataset_proxy_ready": True,
            "diagnostic_only": True,
            "formal": False,
            "training_authorized": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
            "physical_kv_reuse_claimed": False,
            "numeric_equivalence_claimed": False,
        },
        "audit": {
            "record_schema_validated": True,
            "manifest_schema_validated": True,
            "bundle_split_disjoint": True,
            "all_role_views_same_split": True,
            "role_view_invariants_validated": True,
            "forbidden_current_future_content_excluded": True,
            "mandatory_sidecar": True,
            "protected_body_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
        },
    }


def _expected_materialization(
    repo_root: Path,
    config: Mapping[str, Any],
    grammar: Mapping[str, Any],
    catalog: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
    record_schema: Mapping[str, Any],
    manifest_schema: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    records, train_bundles, eval_bundles = _generate_records(
        config, grammar, snapshots, catalog
    )
    _validate_records(config, grammar, catalog, records, record_schema)
    partitions = _partition_bytes(records)
    if any(len(raw) >= _MAX_BYTES for raw in partitions.values()):
        _fail("synthetic_v2_partition_too_large")
    manifest = _build_manifest(
        repo_root,
        config,
        snapshots,
        records,
        partitions,
        train_bundles,
        eval_bundles,
    )
    _v1._validate_schema(
        manifest_schema,
        manifest,
        "synthetic_v2_manifest_schema_validation_failed",
    )
    return partitions, manifest


def _assert_exact_artifact_layout(artifact: Path) -> None:
    expected_files = {"manifest.json", "manifest.json.sha256", *PARTITION_PATHS}
    expected_directories = {"train", "eval_proxy"}
    _v1._assert_no_reparse_absolute_ancestry(
        artifact, "synthetic_v2_artifact_reparse_ancestry"
    )
    files: set[str] = set()
    directories: set[str] = set()
    pending = [artifact]
    try:
        while pending:
            current = pending.pop()
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(artifact).as_posix()
                value = entry.stat(follow_symlinks=False)
                attributes = int(getattr(value, "st_file_attributes", 0))
                if stat.S_ISLNK(value.st_mode) or bool(
                    attributes & _v1._FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    _fail("synthetic_v2_artifact_reparse_entry")
                if stat.S_ISDIR(value.st_mode):
                    directories.add(relative)
                    pending.append(path)
                elif stat.S_ISREG(value.st_mode):
                    files.add(relative)
                else:
                    _fail("synthetic_v2_artifact_special_entry")
    except SyntheticScaffoldDiagnosticV2Error:
        raise
    except OSError as exc:
        raise SyntheticScaffoldDiagnosticV2Error(
            "synthetic_v2_artifact_layout_unreadable"
        ) from exc
    if files != expected_files or directories != expected_directories:
        _fail("synthetic_v2_artifact_layout_invalid")


def _capture_artifact_snapshots(artifact: Path) -> dict[str, _Snapshot]:
    _assert_exact_artifact_layout(artifact)
    result = {
        "manifest": _v1._read_snapshot(
            artifact / "manifest.json",
            "synthetic_v2_manifest_unreadable",
            max_bytes=_MAX_BYTES,
        ),
        "sidecar": _v1._read_snapshot(
            artifact / "manifest.json.sha256",
            "synthetic_v2_sidecar_unreadable",
            max_bytes=1024,
        ),
    }
    for relative in PARTITION_PATHS:
        result[relative] = _v1._read_snapshot(
            artifact / relative,
            "synthetic_v2_partition_unreadable",
            max_bytes=_MAX_BYTES,
        )
    _assert_exact_artifact_layout(artifact)
    return result


def build_dataset(
    repo_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Build the five-role v1 fixture without replacing an existing output."""

    root = Path(repo_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve(strict=True)
    snapshots, config, grammar, catalog, record_schema, manifest_schema = (
        _load_contract_snapshots(root, config_file)
    )
    partitions, manifest = _expected_materialization(
        root,
        config,
        grammar,
        catalog,
        snapshots,
        record_schema,
        manifest_schema,
    )
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _v1._assert_no_reparse_absolute_ancestry(
        output.parent, "synthetic_v2_output_parent_invalid"
    )
    if _v1._path_lexists(output):
        _fail("synthetic_v2_output_already_exists")
    parent_identity = (
        output.parent.stat().st_dev,
        output.parent.stat().st_ino,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        for relative, raw in partitions.items():
            _v1._write_atomic_file(temporary / relative, raw)
        manifest_raw = _canonical_json_bytes(manifest, newline=True)
        _v1._write_atomic_file(temporary / "manifest.json", manifest_raw)
        _v1._write_atomic_file(
            temporary / "manifest.json.sha256",
            f"{_sha256(manifest_raw)}  manifest.json\n".encode("ascii"),
        )
        audited = audit_dataset(root, config_file, temporary)
        if audited != manifest:
            _fail("synthetic_v2_prepublication_audit_mismatch")
        temporary_identity = _v1._stat_identity(temporary.stat())
        artifact_snapshots = _capture_artifact_snapshots(temporary)
        for role, snapshot in snapshots.items():
            snapshot.assert_unchanged(f"synthetic_v2_{role}_changed_during_build")
        current_parent = output.parent.stat()
        if (
            _v1._path_lexists(output)
            or (current_parent.st_dev, current_parent.st_ino) != parent_identity
        ):
            _fail("synthetic_v2_output_publish_race")
        _assert_exact_artifact_layout(temporary)
        if _v1._stat_identity(temporary.stat()) != temporary_identity:
            _fail("synthetic_v2_artifact_root_changed_before_publish")
        for role, snapshot in artifact_snapshots.items():
            snapshot.assert_unchanged(f"synthetic_v2_{role}_changed_before_publish")
        _v1._rename_directory_no_replace(temporary, output)
        return audit_dataset(root, config_file, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_dataset(
    repo_root: str | Path,
    config_path: str | Path,
    artifact_dir: str | Path,
) -> Mapping[str, Any]:
    """Audit five-role v1 from byte snapshots with a final TOCTOU recheck."""

    root = Path(repo_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve(strict=True)
    artifact = Path(artifact_dir)
    if not artifact.is_absolute():
        artifact = root / artifact
    artifact = artifact.absolute()
    snapshots, config, grammar, catalog, record_schema, manifest_schema = (
        _load_contract_snapshots(root, config_file)
    )
    artifact_snapshots = _capture_artifact_snapshots(artifact)
    expected_sidecar = (
        f"{artifact_snapshots['manifest'].sha256}  manifest.json\n".encode("ascii")
    )
    if artifact_snapshots["sidecar"].data != expected_sidecar:
        _fail("synthetic_v2_manifest_sidecar_invalid")
    manifest = _v1._strict_json(
        artifact_snapshots["manifest"].data, "synthetic_v2_manifest_invalid"
    )
    if (
        _canonical_json_bytes(manifest, newline=True)
        != artifact_snapshots["manifest"].data
    ):
        _fail("synthetic_v2_manifest_not_canonical")
    _v1._validate_schema(
        manifest_schema,
        manifest,
        "synthetic_v2_manifest_schema_validation_failed",
    )
    observed: list[Mapping[str, Any]] = []
    for relative in PARTITION_PATHS:
        observed.extend(
            _v1._strict_jsonl(
                artifact_snapshots[relative].data,
                "synthetic_v2_partition_invalid",
            )
        )
    _validate_records(config, grammar, catalog, observed, record_schema)
    expected_partitions, expected_manifest = _expected_materialization(
        root,
        config,
        grammar,
        catalog,
        snapshots,
        record_schema,
        manifest_schema,
    )
    for relative, expected in expected_partitions.items():
        if artifact_snapshots[relative].data != expected:
            _fail("synthetic_v2_partition_materialization_mismatch")
    if dict(manifest) != expected_manifest:
        _fail("synthetic_v2_manifest_materialization_mismatch")
    _assert_exact_artifact_layout(artifact)
    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(f"synthetic_v2_{role}_changed_during_audit")
    for role, snapshot in artifact_snapshots.items():
        snapshot.assert_unchanged(f"synthetic_v2_{role}_changed_during_audit")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build or audit the 1,000-record five-role Q-only diagnostic v1")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo-root", default=".")
        child.add_argument("--config", default=CONFIG_PATH)
        child.add_argument("--artifact", default=CANONICAL_FIXTURE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_dataset(args.repo_root, args.config, args.artifact)
    else:
        manifest = audit_dataset(args.repo_root, args.config, args.artifact)
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
