"""Defensive, SOP-injected distillation pipeline for Anchor-MoE-LoRA."""

from .mutator import MutationManifest, MutationUnavailableError, mutate_frontend_code
from .pipeline import DistillationPipeline, PipelineReport, UpstreamDependencyError
from .provider import (
    PRESETS,
    ModelDiscovery,
    ProviderPreset,
    ProviderSelection,
    ProviderSpec,
    discover_models,
    provider_spec,
    query_quota,
    select_provider_model,
)
from .schema import (
    FRONTEND_REVISION_DATASET,
    REVIEW_LOOP_DATA_SCHEMA_VERSION,
    REVIEW_VERDICT_DATASET,
    DistilledRecord,
    ExpertSOP,
    SeedDemand,
    TaskType,
    validate_frontend_revision_payload,
    validate_review_verdict_payload,
)
from .teacher import (
    MockTeacher,
    OpenAICompatibleTeacher,
    ProviderQuotaExhausted,
    RateLimitError,
    Teacher,
)

__all__ = [
    "DistillationPipeline",
    "DistilledRecord",
    "ExpertSOP",
    "FRONTEND_REVISION_DATASET",
    "MockTeacher",
    "MutationManifest",
    "MutationUnavailableError",
    "ModelDiscovery",
    "OpenAICompatibleTeacher",
    "PRESETS",
    "PipelineReport",
    "ProviderPreset",
    "ProviderQuotaExhausted",
    "ProviderSelection",
    "ProviderSpec",
    "REVIEW_LOOP_DATA_SCHEMA_VERSION",
    "REVIEW_VERDICT_DATASET",
    "RateLimitError",
    "SeedDemand",
    "TaskType",
    "Teacher",
    "UpstreamDependencyError",
    "discover_models",
    "mutate_frontend_code",
    "provider_spec",
    "query_quota",
    "select_provider_model",
    "validate_frontend_revision_payload",
    "validate_review_verdict_payload",
]
