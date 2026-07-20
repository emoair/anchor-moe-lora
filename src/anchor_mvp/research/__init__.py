"""Research-only prototypes that are not part of the stable serving API."""

from .neural_swarm import (
    AdapterSpec,
    CacheGeometry,
    CacheMemoryEstimate,
    CacheSharePlan,
    CacheSharingMode,
    ExpertDemand,
    LayerTarget,
    Projection,
    RoutingGranularity,
    SwarmWave,
    build_rank_bucketed_waves,
    estimate_cache_memory,
    plan_cache_sharing,
)

__all__ = [
    "AdapterSpec",
    "CacheGeometry",
    "CacheMemoryEstimate",
    "CacheSharePlan",
    "CacheSharingMode",
    "ExpertDemand",
    "LayerTarget",
    "Projection",
    "RoutingGranularity",
    "SwarmWave",
    "build_rank_bucketed_waves",
    "estimate_cache_memory",
    "plan_cache_sharing",
]
