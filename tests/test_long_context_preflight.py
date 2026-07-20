from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from anchor_mvp.research.long_context_preflight import (
    LongContextPreflightError,
    authenticate_producer_token_inventory_contract_files,
    authenticate_producer_token_inventory_fixture,
    build_report,
    estimate_bucket,
    load_config,
    main,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research" / "gemma4_12b_long_context_preflight.yaml"
INVENTORY_FIXTURE = ROOT / "fixtures" / "research" / "long_context_token_inventory"
PRODUCER_CONTRACT_PATHS = {
    "config": "configs/research/swebench_long_context_preflight_v1.yaml",
    "record_schema": (
        "configs/research/swebench_long_context_preflight_sidecar.schema.json"
    ),
    "manifest_schema": (
        "configs/research/swebench_long_context_preflight_manifest.schema.json"
    ),
}


def _copy_inventory_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "long_context_token_inventory"
    shutil.copytree(INVENTORY_FIXTURE, destination)
    return destination


def _copy_producer_contracts(tmp_path: Path) -> Path:
    repository_root = tmp_path / "repository"
    for relative_path in PRODUCER_CONTRACT_PATHS.values():
        source = ROOT / relative_path
        destination = repository_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return repository_root


def _rewrite_manifest(
    inventory_root: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    path = inventory_root / "manifest.json"
    manifest = json.loads(path.read_bytes().decode("utf-8"))
    mutate(manifest)
    path.write_bytes(
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )


def _rewrite_scalar_record(
    inventory_root: Path,
    relative_path: str,
    mutate: Callable[[dict[str, Any]], None],
    *,
    index: int = 0,
) -> None:
    """Mutate body-free scalar/hash metadata while preserving file length."""

    path = inventory_root / relative_path
    original = path.read_bytes()
    records = [json.loads(line) for line in original.decode("utf-8").splitlines()]
    mutate(records[index])
    replacement = (
        "\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for record in records
        )
        + "\n"
    ).encode("utf-8")
    assert len(replacement) == len(original), "test mutation must preserve byte count"
    path.write_bytes(replacement)


def _loaded():
    return load_config(CONFIG)


def test_pins_local_gemma4_architecture_and_rope_facts() -> None:
    config, config_sha256 = _loaded()

    assert len(config_sha256) == 64
    assert config["model"]["model_id"] == "google/gemma-4-12B"
    assert config["model"]["revision"] == "56820d7d8cbe8e47975a53325439ed272e91cff2"
    assert config["model"]["source_config_sha256"] == (
        "14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098"
    )
    assert config["model"]["native_context_tokens"] == 262144
    assert config["architecture"] == {
        "text_layers": 48,
        "sliding_layers": 40,
        "full_attention_layers": 8,
        "sliding_window_tokens": 1024,
        "sliding_kv_heads": 8,
        "sliding_head_dim": 256,
        "global_kv_heads": 1,
        "global_head_dim": 512,
    }
    assert config["rope"]["full_attention"] == {
        "encoding": "proportional_rope",
        "theta": 1_000_000,
        "partial_rotary_factor": 0.25,
    }
    assert config["rope"]["sliding_attention"]["theta"] == 10_000


def test_bucket_ladder_is_exact_and_native_boundary_is_not_overclaimed() -> None:
    config, _ = _loaded()
    report = build_report(config)

    assert [(item["bucket"], item["context_tokens"]) for item in report["buckets"]] == [
        ("8k", 8192),
        ("16k", 16384),
        ("32k", 32768),
        ("64k", 65536),
        ("128k", 131072),
        ("256k", 262144),
        ("512k", 524288),
        ("1mi", 1048576),
    ]
    for item in report["buckets"][:6]:
        assert item["within_native_context_metadata"] is True
        assert item["claim_status"] == "native_metadata_only"
        assert item["runtime_validated"] is False
        assert item["quality_validated"] is False
        assert item["launch_allowed"] is False


def test_one_million_bucket_is_explicitly_research_only_and_blocked() -> None:
    config, _ = _loaded()
    item = build_report(config)["buckets"][-1]

    assert item["bucket"] == "1mi"
    assert item["within_native_context_metadata"] is False
    assert item["claim_status"] == "research_only_blocked"
    assert item["launch_allowed"] is False
    assert item["training_allowed"] is False
    assert "exceeds_native_context_metadata" in item["blockers"]
    assert "rope_extrapolation_scaling_unbound" in item["blockers"]


def test_quantized_kv_bytes_are_separate_and_deterministic() -> None:
    config, _ = _loaded()
    bucket = estimate_bucket(config, {"id": "64k", "tokens": 65536})

    assert bucket["swa_cache_cells"] == 1280
    assert bucket["sliding_elements_per_k_or_v"] == 104_857_600
    assert bucket["global_elements_per_k_or_v"] == 268_435_456
    assert bucket["cache_k"]["tensor_payload_bytes"] == 396_623_872
    assert bucket["cache_v"]["tensor_payload_bytes"] == 209_977_344
    assert bucket["kv_tensor_payload_bytes"] == 606_601_216
    assert bucket["runtime_allocation_measured"] is False
    assert bucket["cache_k"]["type"] == "q8_0"
    assert bucket["cache_v"]["type"] == "q4_0"


def test_all_configured_buckets_have_monotonic_kv_capacity() -> None:
    config, _ = _loaded()
    totals = [
        item["kv_tensor_payload_bytes"] for item in build_report(config)["buckets"]
    ]

    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_frozen_producer_inventory_contract_is_exact_body_free_and_split_safe() -> None:
    config, _ = _loaded()
    contract = build_report(config)["producer_token_inventory_contract"]

    assert contract["materialization"] == "metadata_only_no_content_bodies"
    assert contract["producer_version"] == (
        "anchor.long-context-token-inventory-producer.v1"
    )
    assert contract["record_schema_version"] == (
        "anchor.long-context-token-inventory.v1"
    )
    assert contract["manifest_schema_version"] == (
        "anchor.long-context-token-inventory-manifest.v1"
    )
    assert contract["interface_status"] == "frozen_authenticated_producer_handoff"
    assert contract["frozen_producer_schema_claimed"] is True
    assert contract["split_group_key"] == "task_bundle_sha256"
    assert contract["split_before_augmentation"] is True
    assert contract["producer_release"] == {
        "branch": "agent/restore-dual-router-ux",
        "commit": "677bd2a689de7f904d808f35ec6d19adc73e6d2e",
        "config_sha256": (
            "79cd230b161bc91b802854e60df9453677b1c963d38eb9f156347ab56ad00abe"
        ),
        "record_schema_sha256": (
            "aab3e64a41d16c50816da7b03e05a8fb2d1c2b74ac1359f663b70485b34d706f"
        ),
        "manifest_schema_sha256": (
            "8b0d199b2b7dfafa88237ad1fdec538090404b0081a8b07f7136b121fc6932e0"
        ),
        "source_projector_manifest_sha256": (
            "595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac"
        ),
    }
    assert contract["tokenizer_binding"] == {
        "sha256": (
            "047777c4fd6647d75ec3afe5d979ab7c6f02b43397e31886b7f1cd2873519153"
        ),
        "backend": "explicit_synthetic_tokenizer",
        "inventory_mode": "synthetic_fixture",
        "synthetic_fixture_only": True,
        "target_model_tokenizer_match": "not_applicable",
        "gemma_target_identity_verified": False,
    }
    assert contract["fixture"]["manifest_sha256"] == (
        "73ef649b890854ecdecfa1da7f814b746796a9ac486f62328d82c815bbaffc0e"
    )
    assert contract["fixture"]["manifest_sha256_sidecar_physical_sha256"] == (
        "1f3b3d281556814ff7db900c5119eef9e54ea69945a24aed2e5b4ec295d6f41c"
    )
    assert contract["fixture"]["counts"] == {
        "partitions": 3,
        "records": 15,
        "task_bundles": 2,
        "complete_five_role_groups": 3,
        "segment_references": 89,
        "unique_segments": 25,
        "provider_requests": 0,
    }
    assert contract["content_bodies_read_by_preflight"] is False
    assert contract["producer_fixture_authenticated_by_default_preflight"] is False
    assert contract["fixture_authentication_entrypoint"] == (
        "authenticate_producer_token_inventory_fixture"
    )
    assert contract["synthetic_tokenizer_is_gemma"] is False
    assert "input_tokens" in contract["required_fields"]
    assert "shared_prefix_input_tokens" in contract["required_fields"]
    assert "private_delta_input_tokens" in contract["required_fields"]
    assert "segment_plan_sha256" in contract["required_fields"]
    assert "ordered_segment_ids_sha256" in contract["required_fields"]
    assert "terminal_prefix_lineage_sha256" in contract["required_fields"]
    assert "reserved_output_tokens" in contract["required_fields"]
    assert "total_tokens" in contract["required_fields"]
    assert set(contract["prohibited_body_fields"]) == {
        "messages",
        "prompt",
        "completion",
        "content",
        "task_board",
        "blocks",
    }


def test_frozen_producer_inventory_fixture_authenticates_without_gemma_claims() -> None:
    config, _ = _loaded()
    authenticated = authenticate_producer_token_inventory_fixture(
        config,
        INVENTORY_FIXTURE,
    )

    assert authenticated == {
        "status": "frozen_authenticated_producer_fixture",
        "producer_version": "anchor.long-context-token-inventory-producer.v1",
        "record_schema_version": "anchor.long-context-token-inventory.v1",
        "manifest_schema_version": (
            "anchor.long-context-token-inventory-manifest.v1"
        ),
        "fixture_manifest_sha256": (
            "73ef649b890854ecdecfa1da7f814b746796a9ac486f62328d82c815bbaffc0e"
        ),
        "manifest_sha256_sidecar_physical_sha256": (
            "1f3b3d281556814ff7db900c5119eef9e54ea69945a24aed2e5b4ec295d6f41c"
        ),
        "partition_sha256": {
            "train/clean.jsonl": (
                "d58471790406130cfbbde0b473a296665227f920f4f338455302fde462167846"
            ),
            "train/noisy.jsonl": (
                "dfc3e5423ca4368a3974d9cdfc312af540b75c44aa41ff5c43ff940343c60bc1"
            ),
            "calibration/clean.jsonl": (
                "6fcc71a051cab56ab253ffe8cf23983c5e13f3515f8651a6eeb00bc27f7712e5"
            ),
        },
        "counts": {
            "partitions": 3,
            "records": 15,
            "task_bundles": 2,
            "complete_five_role_groups": 3,
            "segment_references": 89,
            "unique_segments": 25,
            "provider_requests": 0,
        },
        "tokenizer_binding_sha256": (
            "047777c4fd6647d75ec3afe5d979ab7c6f02b43397e31886b7f1cd2873519153"
        ),
        "inventory_mode": "synthetic_fixture",
        "target_model_tokenizer_match": "not_applicable",
        "gemma_target_identity_verified": False,
        "partition_bytes_authenticated": True,
        "scalar_inventory_records_parsed": True,
        "content_bodies_materialized": False,
        "synthetic_tokenizer_is_gemma": False,
        "provider_requests": 0,
    }


def test_frozen_producer_contract_files_authenticate_without_model_or_bodies() -> None:
    config, _ = _loaded()
    authenticated = authenticate_producer_token_inventory_contract_files(
        config,
        ROOT,
    )

    assert authenticated == {
        "status": "frozen_authenticated_producer_contract_files",
        "producer_version": "anchor.long-context-token-inventory-producer.v1",
        "record_schema_version": "anchor.long-context-token-inventory.v1",
        "manifest_schema_version": (
            "anchor.long-context-token-inventory-manifest.v1"
        ),
        "file_sha256": {
            "config": (
                "79cd230b161bc91b802854e60df9453677b1c963d38eb9f156347ab56ad00abe"
            ),
            "record_schema": (
                "aab3e64a41d16c50816da7b03e05a8fb2d1c2b74ac1359f663b70485b34d706f"
            ),
            "manifest_schema": (
                "8b0d199b2b7dfafa88237ad1fdec538090404b0081a8b07f7136b121fc6932e0"
            ),
        },
        "content_bodies_materialized": False,
        "model_loaded": False,
        "gpu_used": False,
        "network_used": False,
        "provider_requests": 0,
    }


@pytest.mark.parametrize("contract_name", tuple(PRODUCER_CONTRACT_PATHS))
def test_producer_contract_physical_hash_drift_fails_closed(
    tmp_path: Path,
    contract_name: str,
) -> None:
    config, _ = _loaded()
    repository_root = _copy_producer_contracts(tmp_path)
    contract_path = repository_root / PRODUCER_CONTRACT_PATHS[contract_name]
    with contract_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(
        LongContextPreflightError,
        match=rf"producer contract {contract_name} SHA-256",
    ):
        authenticate_producer_token_inventory_contract_files(
            config,
            repository_root,
        )


def test_static_claims_remain_false() -> None:
    config, _ = _loaded()
    report = build_report(config)

    assert report["status"] == "static_preflight_complete"
    assert not any(report["claims"].values())
    assert "one_million_context_support_is_not_claimed" in report["non_claims"]
    assert [row["position_scaling"] for row in report["staged_research_plan"]] == [
        "none",
        "none",
        "none",
        "yarn_or_linear_2x",
        "yarn_4x_or_longrope_training",
    ]


def test_fact_drift_fails_closed_without_model_or_data_access() -> None:
    config, _ = _loaded()
    config["rope"]["full_attention"]["theta"] = 10_000

    with pytest.raises(LongContextPreflightError, match="full rope theta"):
        validate_config(config)


def test_inventory_manifest_version_drift_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)
    _rewrite_manifest(
        fixture,
        lambda manifest: manifest.__setitem__(
            "schema_version",
            "anchor.long-context-token-inventory-manifest.v0",
        ),
    )

    with pytest.raises(LongContextPreflightError, match="manifest.schema_version"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_manifest_declared_hash_drift_fails_closed(
    tmp_path: Path,
) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["producer"]["record_schema_sha256"] = "0" * 64

    _rewrite_manifest(fixture, mutate)

    with pytest.raises(
        LongContextPreflightError,
        match="manifest.producer.record_schema_sha256",
    ):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_manifest_count_drift_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["counts"]["total"] = 16

    _rewrite_manifest(fixture, mutate)

    with pytest.raises(LongContextPreflightError, match="manifest.counts.total"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_manifest_sha256_sidecar_drift_fails_closed(
    tmp_path: Path,
) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)
    (fixture / "manifest.json.sha256").write_bytes(
        ("0" * 64 + "  manifest.json\n").encode("ascii")
    )

    with pytest.raises(
        LongContextPreflightError,
        match="manifest.json.sha256 declaration",
    ):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_partition_hash_drift_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        record["ordered_segment_ids_sha256"] = "0" * 64

    _rewrite_scalar_record(fixture, "train/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="partition.*SHA-256"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_record_unknown_field_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        record["unknown_x"] = record.pop("record_id")

    _rewrite_scalar_record(fixture, "train/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="record fields mismatch"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_record_token_formula_drift_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        record["input_tokens"] += 1

    _rewrite_scalar_record(fixture, "train/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="input token accounting"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_record_role_stage_drift_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        record["stage"] = "invalid"

    _rewrite_scalar_record(fixture, "train/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="stage/expert binding"):
        authenticate_producer_token_inventory_fixture(config, fixture)


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    [
        ("train/clean.jsonl", 1),
        ("train/noisy.jsonl", 0),
    ],
)
def test_inventory_clean_noisy_private_delta_drift_fails_closed(
    tmp_path: Path,
    relative_path: str,
    replacement: int,
) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        record["private_delta_segment_count"] = replacement

    _rewrite_scalar_record(fixture, relative_path, mutate)

    with pytest.raises(LongContextPreflightError, match="private delta"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_task_bundle_cannot_cross_split(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)
    train_record = json.loads(
        (fixture / "train" / "clean.jsonl")
        .read_bytes()
        .decode("utf-8")
        .splitlines()[0]
    )

    def mutate(record: dict[str, Any]) -> None:
        record["task_bundle_sha256"] = train_record["task_bundle_sha256"]

    _rewrite_scalar_record(fixture, "calibration/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="task bundle.*split"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_each_bundle_split_variant_has_five_roles(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        # Keep the serialized byte count stable while creating a duplicate role.
        record["expert"] = "security_gate"
        record["stage"] = "security"
        record["record_id"] = record["record_id"][:-7]

    _rewrite_scalar_record(fixture, "train/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="duplicate expert"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_tokenizer_binding_canonical_hash_drift_fails_closed(
    tmp_path: Path,
) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["tokenizer_binding"]["tokenizer_revision"] = "v2"

    _rewrite_manifest(fixture, mutate)

    with pytest.raises(
        LongContextPreflightError,
        match="tokenizer_binding canonical SHA-256",
    ):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_inventory_bucket_gate_drift_fails_closed(tmp_path: Path) -> None:
    config, _ = _loaded()
    fixture = _copy_inventory_fixture(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        record["bucket"] = "gt_1m"
        record["gate"] = "research_only_blocked"

    _rewrite_scalar_record(fixture, "train/clean.jsonl", mutate)

    with pytest.raises(LongContextPreflightError, match="bucket"):
        authenticate_producer_token_inventory_fixture(config, fixture)


def test_quantized_v_requires_pinned_flash_attention() -> None:
    config, _ = _loaded()
    config["llama_cpp_runtime"]["flash_attention"] = False

    with pytest.raises(LongContextPreflightError, match="flash_attention"):
        validate_config(config)


def test_inventory_formula_drift_fails_closed() -> None:
    config, _ = _loaded()
    config["producer_token_inventory_contract"]["invariants"]["total_tokens"] = (
        "input_tokens"
    )

    with pytest.raises(LongContextPreflightError, match="invariants"):
        validate_config(config)


def test_cli_emits_json_static_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(CONFIG)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "anchor.gemma4-12b-long-context-preflight-report.v1"
    assert payload["config_sha256"]
    assert len(payload["buckets"]) == 8
    assert payload["claims"]["model_loaded"] is False
    assert payload["claims"]["data_jsonl_read"] is False
    assert payload["producer_inventory_authentication"] is None


def test_cli_explicitly_authenticates_frozen_producer_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(
        [
            "--config",
            str(CONFIG),
            "--authenticate-producer-fixture",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == (
        "static_preflight_complete_with_authenticated_producer_fixture"
    )
    assert payload["claims"]["data_jsonl_read"] is True
    authentication = payload["producer_inventory_authentication"]
    assert authentication["status"] == "frozen_authenticated_producer_fixture"
    assert authentication["content_bodies_materialized"] is False
    assert authentication["synthetic_tokenizer_is_gemma"] is False
    assert authentication["contract_files"]["status"] == (
        "frozen_authenticated_producer_contract_files"
    )
