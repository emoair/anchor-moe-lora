"""Defensive, SOP-injected distillation pipeline for Anchor-MoE-LoRA."""

from .mutator import MutationManifest, MutationUnavailableError, mutate_frontend_code
from .pipeline import DistillationPipeline, PipelineReport, UpstreamDependencyError
from .schema import DistilledRecord, ExpertSOP, SeedDemand, TaskType
from .teacher import MockTeacher, OpenAICompatibleTeacher, Teacher

__all__ = [
    "DistillationPipeline",
    "DistilledRecord",
    "ExpertSOP",
    "MockTeacher",
    "MutationManifest",
    "MutationUnavailableError",
    "OpenAICompatibleTeacher",
    "PipelineReport",
    "SeedDemand",
    "TaskType",
    "Teacher",
    "UpstreamDependencyError",
    "mutate_frontend_code",
]
