"""Auditable OpenCode tool-execution validation layer."""

from .config import build_opencode_config, write_opencode_config
from .batch import (
    BatchStageResult,
    LiveBatchConfig,
    batch_run_succeeded,
    load_candidate_samples,
    merge_stage_into_gold,
    run_live_batch,
    verify_execution_split,
)
from .gold import (
    canonical_json,
    is_accepted_gold_record,
    merge_attempts_jsonl,
    merge_gold_jsonl,
    persist_attempts_and_gold,
    write_attempts_jsonl,
    write_gold_jsonl,
)
from .harness import ToolingHarness
from .models import (
    AgentExecution,
    FileChange,
    GoldRecord,
    PublicDecisionStep,
    PublicOutcome,
    SampleSpec,
    SkillProvenance,
    ToolTraceEntry,
    ValidationResult,
)
from .policy import ToolPolicy
from .runner import MockAgentExecutor, OpenCodeExecutor
from .skills import (
    AuditedSkill,
    SkillSourceError,
    SkillSourceRegistry,
    audit_skill_instructions,
)

__all__ = [
    "AgentExecution",
    "BatchStageResult",
    "batch_run_succeeded",
    "FileChange",
    "GoldRecord",
    "PublicDecisionStep",
    "PublicOutcome",
    "MockAgentExecutor",
    "LiveBatchConfig",
    "OpenCodeExecutor",
    "SampleSpec",
    "SkillProvenance",
    "SkillSourceError",
    "SkillSourceRegistry",
    "AuditedSkill",
    "ToolPolicy",
    "ToolTraceEntry",
    "ToolingHarness",
    "ValidationResult",
    "build_opencode_config",
    "audit_skill_instructions",
    "canonical_json",
    "is_accepted_gold_record",
    "merge_attempts_jsonl",
    "merge_gold_jsonl",
    "load_candidate_samples",
    "merge_stage_into_gold",
    "run_live_batch",
    "persist_attempts_and_gold",
    "verify_execution_split",
    "write_gold_jsonl",
    "write_attempts_jsonl",
    "write_opencode_config",
]
