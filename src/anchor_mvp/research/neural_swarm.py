"""Correctness-first planning primitives for Neural Swarm experiments.

This module does not claim to implement a KV-cache kernel.  It makes the
correctness boundary explicit before CUDA or Triton work begins:

* exact reuse is limited to layers whose cached K/V cannot have been changed by
  the selected adapter;
* sharing a base-cache component after adapter hidden states diverge is an
  approximation and must be quality-gated;
* rank-bucketed waves reduce barrier slack but do not promise parallel speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class Projection(str, Enum):
    """Transformer projection families relevant to cache compatibility."""

    QUERY = "q_proj"
    KEY = "k_proj"
    VALUE = "v_proj"
    OUTPUT = "o_proj"
    MLP = "mlp"


class CacheSharingMode(str, Enum):
    """The strength of the cache-sharing claim made by a plan."""

    EXACT_FULL = "exact_full"
    EXACT_LAYER_FRONTIER = "exact_layer_frontier"
    APPROX_DISAGGREGATED = "approx_disaggregated"
    ISOLATED = "isolated"


class RoutingGranularity(str, Enum):
    """Where an adapter routing decision is allowed to change."""

    TASK = "task"
    STAGE = "stage"
    BRANCH = "branch"
    TOKEN = "token"


@dataclass(frozen=True, order=True)
class LayerTarget:
    """One LoRA target within a decoder layer."""

    layer: int
    projection: Projection

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")


@dataclass(frozen=True)
class AdapterSpec:
    """Content-free adapter metadata used by the cache planner."""

    adapter_id: str
    base_fingerprint: str
    rank: int
    num_layers: int
    targets: tuple[LayerTarget, ...]
    active_during_prefill: bool = True

    def __post_init__(self) -> None:
        if not self.adapter_id.strip() or not self.base_fingerprint.strip():
            raise ValueError("adapter_id and base_fingerprint are required")
        if self.rank < 1 or self.num_layers < 1:
            raise ValueError("rank and num_layers must be positive")
        if any(target.layer >= self.num_layers for target in self.targets):
            raise ValueError("target layer is outside the model")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate layer targets are not allowed")


@dataclass(frozen=True)
class CacheSharePlan:
    """Static sharing plan for the common prefill prefix."""

    mode: CacheSharingMode
    exact_shared_layer_count: int
    num_layers: int
    first_lossy_layer: int | None
    approximation_requires_quality_gate: bool
    reasons: tuple[str, ...]

    @property
    def exact_shared_fraction(self) -> float:
        return self.exact_shared_layer_count / self.num_layers


@dataclass(frozen=True)
class CacheGeometry:
    """Decoder KV layout used for planning-only memory estimates."""

    num_layers: int
    prefix_tokens: int
    kv_heads: int
    head_dim: int
    element_bytes: int = 2

    def __post_init__(self) -> None:
        values = (
            self.num_layers,
            self.prefix_tokens,
            self.kv_heads,
            self.head_dim,
            self.element_bytes,
        )
        if any(value < 1 for value in values):
            raise ValueError("cache geometry values must be positive")

    @property
    def bytes_per_layer(self) -> int:
        # Two tensors: key and value.
        return (
            2 * self.prefix_tokens * self.kv_heads * self.head_dim * self.element_bytes
        )

    @property
    def full_cache_bytes(self) -> int:
        return self.num_layers * self.bytes_per_layer


@dataclass(frozen=True)
class CacheMemoryEstimate:
    """Memory comparison against one independent full KV cache per expert."""

    expert_count: int
    independent_total_bytes: int
    planned_total_bytes: int
    shared_bytes: int
    per_expert_bytes: int
    savings_fraction: float


@dataclass(frozen=True)
class ExpertDemand:
    """A content-free unit of scheduling work."""

    expert_id: str
    rank: int
    token_count: int

    def __post_init__(self) -> None:
        if not self.expert_id.strip() or self.rank < 1 or self.token_count < 1:
            raise ValueError("expert_id, rank, and token_count must be positive")

    @property
    def work_units(self) -> int:
        return self.rank * self.token_count


@dataclass(frozen=True)
class SwarmWave:
    """One group launched between a tick and a tock barrier."""

    experts: tuple[ExpertDemand, ...]
    barrier_slack_units: int


def _exact_frontier(spec: AdapterSpec) -> int:
    """Return the number of prefix layers whose K/V is exactly reusable.

    K/V LoRA changes the cache at its own layer. Q/O/MLP changes the residual
    state only after K/V for that layer has been produced, so divergence starts
    at the next layer.
    """

    if not spec.active_during_prefill or not spec.targets:
        return spec.num_layers
    frontier = spec.num_layers
    for target in spec.targets:
        if target.projection in {Projection.KEY, Projection.VALUE}:
            boundary = target.layer
        else:
            boundary = target.layer + 1
        frontier = min(frontier, boundary)
    return frontier


def plan_cache_sharing(
    spec: AdapterSpec,
    *,
    allow_approximate_disaggregation: bool = False,
) -> CacheSharePlan:
    """Plan cache reuse without silently upgrading an approximation to exact."""

    frontier = _exact_frontier(spec)
    if frontier == spec.num_layers:
        return CacheSharePlan(
            mode=CacheSharingMode.EXACT_FULL,
            exact_shared_layer_count=frontier,
            num_layers=spec.num_layers,
            first_lossy_layer=None,
            approximation_requires_quality_gate=False,
            reasons=("adapter cannot change cached prefix K/V",),
        )
    if allow_approximate_disaggregation:
        return CacheSharePlan(
            mode=CacheSharingMode.APPROX_DISAGGREGATED,
            exact_shared_layer_count=frontier,
            num_layers=spec.num_layers,
            first_lossy_layer=frontier,
            approximation_requires_quality_gate=True,
            reasons=(
                "base cache is shared after adapter hidden states may diverge",
                "adapter residual cache must be reconstructed inside attention",
            ),
        )
    mode = (
        CacheSharingMode.EXACT_LAYER_FRONTIER
        if frontier > 0
        else CacheSharingMode.ISOLATED
    )
    return CacheSharePlan(
        mode=mode,
        exact_shared_layer_count=frontier,
        num_layers=spec.num_layers,
        first_lossy_layer=None,
        approximation_requires_quality_gate=False,
        reasons=(
            "share only the exact layer prefix and isolate every divergent layer",
        ),
    )


def estimate_cache_memory(
    geometry: CacheGeometry,
    spec: AdapterSpec,
    plan: CacheSharePlan,
    *,
    expert_count: int,
) -> CacheMemoryEstimate:
    """Estimate cache storage; adapter weights and temporary SRAM are excluded."""

    if expert_count < 1:
        raise ValueError("expert_count must be positive")
    if geometry.num_layers != spec.num_layers or plan.num_layers != spec.num_layers:
        raise ValueError("model layer counts must match")

    independent = expert_count * geometry.full_cache_bytes
    if plan.mode is CacheSharingMode.EXACT_FULL:
        shared = geometry.full_cache_bytes
        per_expert = 0
    elif plan.mode is CacheSharingMode.APPROX_DISAGGREGATED:
        shared = geometry.full_cache_bytes
        kv_target_count = sum(
            target.projection in {Projection.KEY, Projection.VALUE}
            for target in spec.targets
        )
        per_expert = (
            geometry.prefix_tokens
            * spec.rank
            * geometry.element_bytes
            * kv_target_count
        )
    else:
        shared = plan.exact_shared_layer_count * geometry.bytes_per_layer
        per_expert = (
            geometry.num_layers - plan.exact_shared_layer_count
        ) * geometry.bytes_per_layer

    planned = shared + expert_count * per_expert
    savings = 1.0 - planned / independent
    return CacheMemoryEstimate(
        expert_count=expert_count,
        independent_total_bytes=independent,
        planned_total_bytes=planned,
        shared_bytes=shared,
        per_expert_bytes=per_expert,
        savings_fraction=savings,
    )


def build_rank_bucketed_waves(
    demands: Sequence[ExpertDemand] | Iterable[ExpertDemand],
    *,
    max_parallel: int,
) -> tuple[SwarmWave, ...]:
    """Group similarly sized experts to reduce global-barrier idle time.

    This planner intentionally does not assume that CUDA streams improve
    throughput.  The benchmark must demonstrate overlap on the target GPU.
    """

    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    ordered = sorted(tuple(demands), key=lambda item: (item.work_units, item.expert_id))
    waves: list[SwarmWave] = []
    for offset in range(0, len(ordered), max_parallel):
        experts = tuple(ordered[offset : offset + max_parallel])
        work = [item.work_units for item in experts]
        slack = max(work) - min(work) if work else 0
        waves.append(SwarmWave(experts=experts, barrier_slack_units=slack))
    return tuple(waves)
