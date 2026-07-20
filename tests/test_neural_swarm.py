from __future__ import annotations

import pytest

from anchor_mvp.research.neural_swarm import (
    AdapterSpec,
    CacheGeometry,
    CacheSharingMode,
    ExpertDemand,
    LayerTarget,
    Projection,
    build_rank_bucketed_waves,
    estimate_cache_memory,
    plan_cache_sharing,
)


def _spec(*targets: LayerTarget, rank: int = 16) -> AdapterSpec:
    return AdapterSpec(
        adapter_id="planner-r16",
        base_fingerprint="base-sha256",
        rank=rank,
        num_layers=40,
        targets=tuple(targets),
    )


def test_kv_adapter_diverges_at_its_own_layer() -> None:
    plan = plan_cache_sharing(_spec(LayerTarget(4, Projection.KEY)))
    assert plan.mode is CacheSharingMode.EXACT_LAYER_FRONTIER
    assert plan.exact_shared_layer_count == 4
    assert plan.first_lossy_layer is None


def test_query_adapter_shares_current_layer_kv_but_not_next_layer() -> None:
    plan = plan_cache_sharing(_spec(LayerTarget(4, Projection.QUERY)))
    assert plan.exact_shared_layer_count == 5


def test_first_layer_kv_adapter_requires_isolation_for_exact_mode() -> None:
    plan = plan_cache_sharing(_spec(LayerTarget(0, Projection.VALUE)))
    assert plan.mode is CacheSharingMode.ISOLATED
    assert plan.exact_shared_layer_count == 0


def test_approximate_mode_is_never_reported_as_exact() -> None:
    plan = plan_cache_sharing(
        _spec(LayerTarget(0, Projection.KEY), LayerTarget(0, Projection.VALUE)),
        allow_approximate_disaggregation=True,
    )
    assert plan.mode is CacheSharingMode.APPROX_DISAGGREGATED
    assert plan.first_lossy_layer == 0
    assert plan.approximation_requires_quality_gate


def test_inactive_prefill_adapter_can_share_full_prefix() -> None:
    spec = AdapterSpec(
        adapter_id="decode-only",
        base_fingerprint="base-sha256",
        rank=8,
        num_layers=8,
        targets=(LayerTarget(0, Projection.KEY),),
        active_during_prefill=False,
    )
    assert plan_cache_sharing(spec).mode is CacheSharingMode.EXACT_FULL


def test_memory_estimate_matches_disaggregated_formula() -> None:
    spec = _spec(
        LayerTarget(0, Projection.KEY),
        LayerTarget(0, Projection.VALUE),
        rank=16,
    )
    geometry = CacheGeometry(
        num_layers=40,
        prefix_tokens=32_768,
        kv_heads=8,
        head_dim=128,
        element_bytes=2,
    )
    plan = plan_cache_sharing(spec, allow_approximate_disaggregation=True)
    estimate = estimate_cache_memory(geometry, spec, plan, expert_count=8)
    expected_residual = 32_768 * 16 * 2 * 2
    assert estimate.per_expert_bytes == expected_residual
    assert (
        estimate.planned_total_bytes
        == geometry.full_cache_bytes + 8 * expected_residual
    )
    assert 0 < estimate.savings_fraction < 1


def test_rank_bucketing_reduces_wave_skew() -> None:
    demands = (
        ExpertDemand("rank-64", rank=64, token_count=100),
        ExpertDemand("rank-8-a", rank=8, token_count=100),
        ExpertDemand("rank-32", rank=32, token_count=100),
        ExpertDemand("rank-8-b", rank=8, token_count=100),
    )
    waves = build_rank_bucketed_waves(demands, max_parallel=2)
    assert [item.expert_id for item in waves[0].experts] == ["rank-8-a", "rank-8-b"]
    assert waves[0].barrier_slack_units == 0
    assert waves[1].barrier_slack_units == 3_200


def test_invalid_target_layer_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        _spec(LayerTarget(40, Projection.KEY))
