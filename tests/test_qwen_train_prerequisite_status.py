from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
STATUS_SCHEMA_PATH = (
    ROOT / "configs/research/qwen_train_prerequisite_status.schema.json"
)
BINDING_SCHEMA_PATH = (
    ROOT / "configs/research/scaffold_tokenizer_binding_manifest.schema.json"
)
TOY_SCHEMA_PATH = (
    ROOT / "configs/research/qwen_toy_source_disjoint_attestation.schema.json"
)
FIXTURE_ROOT = ROOT / "fixtures/research/qwen_train_prerequisite_status"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
SIDECAR_PATH = FIXTURE_ROOT / "manifest.json.sha256"
LOCAL_QWEN_ROOT = ROOT.parent / "models/qwen2.5-1.5b-instruct-hf"
SHA = "0" * 64


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _binding_producer() -> dict[str, Any]:
    identity = {"path": "configs/research/source.json", "sha256": SHA}
    return {
        "producer_id": "anchor.scaffold-tokenizer-binding-producer.v1",
        "config": identity,
        "manifest_schema": identity,
        "implementation_files": [identity],
        "implementation_sha256": SHA,
        "canonical_json_policy": "utf8_sort_keys_compact_no_normalization_v1",
        "manifest_sha256_sidecar_required": True,
        "atomic_publish_required": True,
        "single_bytes_snapshot_required": True,
        "final_source_recheck_required": True,
    }


def _binding_safety() -> dict[str, Any]:
    return {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "model_weights_read": False,
        "gguf_files_read": False,
        "q4_runtime_smoke_consumed": False,
        "canonical_gold_read": False,
        "canonical_gold_written": False,
        "heldout_content_read": False,
        "heldout_content_emitted": False,
        "training_authorized": False,
        "quality_validated": False,
        "runtime_capability_claimed": False,
        "physical_kv_claimed": False,
    }


def _toy_attestation() -> dict[str, Any]:
    def protected(path: str, sha256: str) -> dict[str, Any]:
        return {
            "path": path,
            "sha256": sha256,
            "source_id_inventory_sha256": SHA,
            "source_id_count": 1,
            "identifier_domain_policy_sha256": SHA,
            "namespace_inventory_sha256": SHA,
            "metadata_only": True,
            "content_files_read": 0,
        }

    reads = [
        {
            "role": "generator",
            "path": "scripts/toy_generator.py",
            "bytes": 1,
            "sha256": SHA,
        },
        {
            "role": "generator_config",
            "path": "configs/toy.json",
            "bytes": 1,
            "sha256": SHA,
        },
        {
            "role": "closed_grammar",
            "path": "configs/toy_grammar.json",
            "bytes": 1,
            "sha256": SHA,
        },
    ]
    return {
        "schema_version": "anchor.qwen-toy-source-disjoint-attestation.v1",
        "status": "ready",
        "partition": "diagnostic_only",
        "data_scope": "toy_plumbing_only",
        "formal_training_authorized": False,
        "consumable_by_formal_release": False,
        "attester": {
            "implementation_path": "scripts/audit_toy.py",
            "implementation_sha256": SHA,
            "config_path": "configs/toy_audit.json",
            "config_sha256": SHA,
            "algorithm_id": "anchor.qwen-toy-source-disjoint-audit.v1",
            "canonical_json_policy": "utf8_sort_keys_compact_no_normalization_v1",
            "attestation_inputs_single_snapshot": True,
            "final_source_recheck_required": True,
        },
        "toy_manifest": {
            "path": "toy/manifest.json",
            "sha256": SHA,
            "sidecar_path": "toy/manifest.json.sha256",
            "sidecar_sha256": SHA,
            "sidecar_declared_manifest_sha256": SHA,
            "mandatory_sidecar": True,
            "sidecar_format": "sha256sum_manifest_json_lf",
        },
        "toy_identity": {
            "generator_path": "scripts/toy_generator.py",
            "generator_sha256": SHA,
            "generator_config_path": "configs/toy.json",
            "generator_config_sha256": SHA,
            "deterministic_seed_sha256": SHA,
            "closed_grammar_id": "anchor.toy-grammar.v1",
            "closed_grammar_path": "configs/toy_grammar.json",
            "closed_grammar_sha256": SHA,
            "toy_source_namespace": "anchor.qwen-toy-diagnostic.v1",
            "toy_source_ids_sha256": SHA,
            "record_pair_ids_sha256": SHA,
            "file_inventory_sha256": SHA,
            "file_count": 1,
            "record_count": 1,
            "files": [
                {
                    "path": "toy/records.jsonl",
                    "bytes": 1,
                    "records": 1,
                    "sha256": SHA,
                }
            ],
        },
        "protected_manifest_bindings": {
            "swebench_source": protected(
                "datasets/public/swebench-full-bank-v1/manifest.json",
                "55c84236e42a803d029ce961fcce064b0b894b632e2789191c3ed1e106ebcf28",
            ),
            "gold_partition": protected(
                "data/automated_v3_shards/ark_max_retry2_offset300000_c10/partitions/manifest.json",
                "4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7",
            ),
            "partial_gold_export": protected(
                "data/automated_v3_shards/ark_max_retry2_offset300000_c10/training_exports/per_expert_partial_gold/4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7/manifest.json",
                "1b8e5b87957d7ec1e867813c95b8f7ab3bef55861e778b6ba9f197e6edf3f2ec",
            ),
            "heldout": protected(
                "artifacts/benchmark/heldout_v1/manifest.json",
                "1ac7240d700a67458dc713b66ff085f1e51795b26cdacff688063bc60af3194c",
            ),
            "legacy_heldout_cases": protected(
                "configs/training/heldout_cases.jsonl",
                "c57c1ec144989fef47abeeda13e5fee90d81207418b9840043ec54a0f74ba70c",
            ),
            "synthetic_scaffold": protected(
                "fixtures/research/swebench_natural_language_scaffold/manifest.json",
                "25e40da8fea46ba018ae0031fa8c37da38b59438bb92d9052a915fd256d822dc",
            ),
        },
        "generation_read_set": {
            "scope": "declared_semantic_generation_inputs_only",
            "exact_allowlist_enforced": True,
            "observed_read_count": 3,
            "unexpected_read_count": 0,
            "inventory_sha256": SHA,
            "inputs": reads,
            "external_corpus_files_read": 0,
            "protected_dataset_files_read": 0,
            "gold_content_files_read": 0,
            "heldout_content_files_read": 0,
            "scaffold_record_files_read": 0,
        },
        "execution": {
            "offline": True,
            "provider_requests": 0,
            "network_requests": 0,
            "credentials_read": False,
            "canonical_gold_written": False,
        },
        "filesystem_safety": {
            "relative_paths_only": True,
            "path_root_binding_sha256": SHA,
            "symlink_policy": "reject_any_symlink_or_junction_in_resolved_path",
            "single_bytes_snapshot": True,
            "toctou_semantics": "authenticate_once_then_parse_and_count_same_bytes",
            "files_changed_during_read": 0,
        },
        "disjointness": {
            "claim_scope": "authenticated_provenance_and_identifier_disjoint_only",
            "proof_algorithm_id": (
                "sha256_sorted_utf8_namespaced_source_ids_intersection_v1"
            ),
            "proof_input_inventory_sha256": SHA,
            "protected_source_id_inventory_sha256": SHA,
            "intersection_proof_sha256": SHA,
            "source_namespace_disjoint": True,
            "source_id_intersection_count": 0,
            "generated_from_closed_grammar_only": True,
            "derived_from_protected_sources": False,
            "semantic_uniqueness_claimed": False,
            "content_uniqueness_claimed": False,
            "audit_path": "toy/audit.json",
            "audit_sha256": SHA,
        },
        "verification": {
            "manifest_sidecar_digest_matches": True,
            "file_inventory_matches": True,
            "record_counts_match": True,
            "protected_binding_expected_sha_matches": True,
            "protected_id_inventories_authenticated": True,
            "protected_id_inventories_recomputed": True,
            "intersection_recomputed": True,
        },
        "output_policy": {
            "scope": "attestation_document_only",
            "metadata_only": True,
            "sample_content_emitted": False,
            "token_sequences_emitted": False,
        },
    }


def test_schemas_are_valid_draft_2020_12() -> None:
    for path in (STATUS_SCHEMA_PATH, BINDING_SCHEMA_PATH, TOY_SCHEMA_PATH):
        Draft202012Validator.check_schema(_load(path))


def test_published_status_and_mandatory_sidecar() -> None:
    schema = _load(STATUS_SCHEMA_PATH)
    manifest_bytes = MANIFEST_PATH.read_bytes()
    Draft202012Validator(schema).validate(json.loads(manifest_bytes))
    manifest_sha = _sha256(manifest_bytes)
    assert SIDECAR_PATH.read_text(encoding="utf-8") == (
        f"{manifest_sha}  manifest.json\n"
    )


def test_status_binds_physical_schema_hashes() -> None:
    status = _load(MANIFEST_PATH)
    assert status["producer"]["schema"]["sha256"] == _sha256(
        STATUS_SCHEMA_PATH.read_bytes()
    )
    assert status["tokenizer_binding"]["binding_schema_sha256"] == _sha256(
        BINDING_SCHEMA_PATH.read_bytes()
    )
    assert status["toy_attestation"]["schema_sha256"] == _sha256(
        TOY_SCHEMA_PATH.read_bytes()
    )


def test_raw_gold_counts_digest_and_thresholds() -> None:
    observation = _load(MANIFEST_PATH)["raw_gold_observation"]
    basis = {
        "experts": {
            role: entry["count"] for role, entry in observation["experts"].items()
        },
        "strict_complete_chains": observation["strict_complete_chains"]["count"],
        "total_count": observation["total_count"],
    }
    assert observation["counts_sha256"] == _sha256(_canonical(basis))
    assert sum(entry["count"] for entry in observation["experts"].values()) == 1465
    assert observation["shortfalls"] == {
        "frontend_review": 53,
        "security_gate": 108,
    }
    assert observation["formal_partitions"]["status"] == "unavailable"


def test_formal_artifacts_remain_unavailable() -> None:
    status = _load(MANIFEST_PATH)
    formal = status["formal_artifacts"]
    for name in (
        "snapshot",
        "final_projector",
        "generic_execution",
        "source_disjoint",
        "release_lock",
    ):
        assert formal[name]["status"] == "unavailable"
        assert formal[name]["artifact_exists"] is False
        assert formal[name]["training_eligible"] is False
    assert formal["synthetic_projector"]["formal_release_eligible"] is False
    assert status["safety"]["training_authorized"] is False


def test_candidate_source_identity_is_tokenizer_only() -> None:
    candidate = _load(MANIFEST_PATH)["tokenizer_binding"]["candidate_source_identity"]
    assets = candidate["tokenizer_assets"]
    assert candidate["tokenizer_assets_sha256"] == _sha256(_canonical(assets))
    assert sum(entry["bytes"] for entry in assets) == 11942496
    assert {entry["path"] for entry in assets} == {
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    assert "model.safetensors" not in {entry["path"] for entry in assets}
    assert candidate["model_weights_in_tokenizer_inventory"] is False
    assert candidate["training_eligible"] is False


def test_local_candidate_small_assets_when_present() -> None:
    if not LOCAL_QWEN_ROOT.is_dir():
        pytest.skip("local Qwen tokenizer candidate is not installed")
    candidate = _load(MANIFEST_PATH)["tokenizer_binding"]["candidate_source_identity"]
    for entry in candidate["tokenizer_assets"]:
        path = LOCAL_QWEN_ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path.read_bytes()) == entry["sha256"]
    assert (LOCAL_QWEN_ROOT / "model.safetensors").stat().st_size == 3087467144
    tokenizer_config = _load(LOCAL_QWEN_ROOT / "tokenizer_config.json")
    template_bytes = tokenizer_config["chat_template"].encode("utf-8")
    assert _sha256(template_bytes) == candidate["chat_template"]["exact_utf8_sha256"]


def test_consumer_release_fields_have_no_self_or_toy_reference() -> None:
    fields = _load(MANIFEST_PATH)["consumer_freeze_requirements"][
        "ordered_release_fields"
    ]
    assert "formal_release_lock_sha256" not in fields
    assert "toy_attestation_sha256" not in fields
    assert "toy_attestation_schema_sha256" not in fields
    assert "trainable_base_snapshot_manifest_sha256" in fields
    assert "tokenizer_base_compatibility_attestation_sha256" in fields


def test_status_has_no_sample_or_token_sequence_fields() -> None:
    forbidden = {
        "answer",
        "body",
        "content",
        "preview",
        "prompt",
        "target",
        "token_ids",
        "token_indices",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(_load(MANIFEST_PATH))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "ready"),
        (("safety", "training_authorized"), True),
        (("formal_artifacts", "snapshot", "artifact_exists"), True),
        (("raw_gold_observation", "experts", "planner", "count"), 385),
        (("tokenizer_binding", "token_indices_emitted"), True),
        (
            ("toy_attestation", "source_disjoint_claim_status"),
            "verified",
        ),
    ],
)
def test_status_mutations_fail_closed(
    path: tuple[str, ...], replacement: object
) -> None:
    schema = _load(STATUS_SCHEMA_PATH)
    value = deepcopy(_load(MANIFEST_PATH))
    cursor: dict[str, Any] = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    assert not Draft202012Validator(schema).is_valid(value)


def test_tokenizer_binding_unavailable_branch_and_token_rejection() -> None:
    schema = _load(BINDING_SCHEMA_PATH)
    value = {
        "schema_version": "anchor.scaffold-tokenizer-binding-manifest.v1",
        "status": "unavailable",
        "claim_scope": "unavailable_status_only",
        "requested_target": {
            "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "tokenizer_revision_status": "unavailable",
            "target_model_tokenizer_match": "unavailable",
        },
        "unavailable_reasons": ["local_hf_tokenizer_snapshot_unavailable"],
        "producer": _binding_producer(),
        "safety": _binding_safety(),
    }
    validator = Draft202012Validator(schema)
    validator.validate(value)
    value["token_ids"] = [1]
    assert not validator.is_valid(value)


def test_toy_attestation_contract_and_negative_authority() -> None:
    validator = Draft202012Validator(_load(TOY_SCHEMA_PATH))
    value = _toy_attestation()
    validator.validate(value)
    negative = deepcopy(value)
    negative["formal_training_authorized"] = True
    assert not validator.is_valid(negative)

    negative = deepcopy(value)
    negative["protected_manifest_bindings"]["gold_partition"]["path"] = value[
        "protected_manifest_bindings"
    ]["swebench_source"]["path"]
    assert not validator.is_valid(negative)

    negative = deepcopy(value)
    negative["protected_manifest_bindings"]["heldout"]["sha256"] = SHA
    assert not validator.is_valid(negative)

    negative = deepcopy(value)
    negative["protected_manifest_bindings"]["synthetic_scaffold"]["source_id_count"] = 0
    assert not validator.is_valid(negative)
