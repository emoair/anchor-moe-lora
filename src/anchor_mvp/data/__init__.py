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
from .schema import DistilledRecord, ExpertSOP, SeedDemand, TaskType
from .teacher import MockTeacher, OpenAICompatibleTeacher, Teacher

__all__ = [
    "DistillationPipeline",
    "DistilledRecord",
    "ExpertSOP",
    "MockTeacher",
    "MutationManifest",
    "MutationUnavailableError",
    "ModelDiscovery",
    "OpenAICompatibleTeacher",
    "PRESETS",
    "PipelineReport",
    "ProviderPreset",
    "ProviderSelection",
    "ProviderSpec",
    "SeedDemand",
    "TaskType",
    "Teacher",
    "UpstreamDependencyError",
    "discover_models",
    "mutate_frontend_code",
    "provider_spec",
    "query_quota",
    "select_provider_model",
]
