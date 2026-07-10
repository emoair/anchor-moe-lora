"""QLoRA training utilities for Anchor-MVP.

The package intentionally keeps configuration validation and dry runs free of
heavy ML imports.  Importing :mod:`anchor_mvp.training` never downloads a model.
"""

from .config import (
    ALLOWED_ADAPTERS,
    ALLOWED_RANKS,
    ConfigError,
    load_training_config,
    select_adapter,
    validate_training_config,
)
from .schema import DatasetValidationError, validate_jsonl
from .preflight import build_preflight_report, verify_prior_smoke_gate

__all__ = [
    "ALLOWED_ADAPTERS",
    "ALLOWED_RANKS",
    "ConfigError",
    "DatasetValidationError",
    "build_preflight_report",
    "load_training_config",
    "select_adapter",
    "validate_jsonl",
    "validate_training_config",
    "verify_prior_smoke_gate",
]
