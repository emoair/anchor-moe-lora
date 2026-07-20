from pathlib import Path

from anchor_mvp.benchmark.segment_preflight import preflight_segment_contract


ROOT = Path(__file__).resolve().parents[1]


def test_segment_contract_preflight_reads_only_aggregate_training_metadata() -> None:
    result = preflight_segment_contract(
        ROOT / "configs" / "benchmark" / "compact_mvp_v2b_segment_contract.json",
        ROOT,
    )

    assert result["status"] == "contract_valid_training_artifacts_pending"
    assert result["frontend_segment_count"] == 10
    assert result["review_segment_count"] == 10
    assert result["expected_physical_calls"] == 23
    assert result["audited_maximum_target_tokens"] == 410
    assert result["a100_baseline"] == "A"
    assert result["heldout_content_read"] is False
    assert result["benchmark_record_content_read"] is False
    assert result["gpu_started"] is False
