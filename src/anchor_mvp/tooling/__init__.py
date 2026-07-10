"""Auditable OpenCode tool-execution validation layer."""

from .config import build_opencode_config, write_opencode_config
from .gold import canonical_json, write_gold_jsonl
from .harness import ToolingHarness
from .models import (
    AgentExecution,
    FileChange,
    GoldRecord,
    SampleSpec,
    ToolTraceEntry,
    ValidationResult,
)
from .policy import ToolPolicy
from .runner import MockAgentExecutor, OpenCodeExecutor

__all__ = [
    "AgentExecution",
    "FileChange",
    "GoldRecord",
    "MockAgentExecutor",
    "OpenCodeExecutor",
    "SampleSpec",
    "ToolPolicy",
    "ToolTraceEntry",
    "ToolingHarness",
    "ValidationResult",
    "build_opencode_config",
    "canonical_json",
    "write_gold_jsonl",
    "write_opencode_config",
]
