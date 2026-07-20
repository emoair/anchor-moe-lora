from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor_mvp.research.long_context_preflight import (
    LongContextPreflightError,
    build_report,
    estimate_bucket,
    load_config,
    main,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research" / "gemma4_12b_long_context_preflight.yaml"


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


def test_producer_inventory_contract_is_body_free_and_split_safe() -> None:
    config, _ = _loaded()
    contract = build_report(config)["producer_token_inventory_contract"]

    assert contract["materialization"] == "metadata_only_no_content_bodies"
    assert contract["interface_status"] == "requested_pending_producer_freeze"
    assert contract["frozen_producer_schema_claimed"] is False
    assert contract["split_group_key"] == "task_bundle_sha256"
    assert contract["split_before_augmentation"] is True
    assert contract["content_bodies_read_by_preflight"] is False
    assert "materialized_prompt_tokens" in contract["required_fields"]
    assert "training_sequence_tokens" in contract["required_fields"]
    assert "forbidden_tokens_excluded" in contract["required_fields"]
    assert "segment_plan_sha256" in contract["required_fields"]
    assert "ordered_segment_ids_sha256" in contract["required_fields"]
    assert "terminal_prefix_lineage_sha256" in contract["required_fields"]
    assert "reserved_output_tokens" in contract["required_fields"]
    assert "total_tokens" in contract["required_fields"]
    assert "required_context_tokens" in contract["required_fields"]
    assert set(contract["prohibited_body_fields"]) == {
        "messages",
        "prompt",
        "completion",
        "content",
        "task_board",
        "blocks",
    }


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


def test_quantized_v_requires_pinned_flash_attention() -> None:
    config, _ = _loaded()
    config["llama_cpp_runtime"]["flash_attention"] = False

    with pytest.raises(LongContextPreflightError, match="flash_attention"):
        validate_config(config)


def test_inventory_formula_drift_fails_closed() -> None:
    config, _ = _loaded()
    config["producer_token_inventory_contract"]["invariants"][
        "required_context_tokens"
    ] = "materialized_prompt_tokens + target_tokens"

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
