"""Fail-closed execution gate for formal-v3 training entry points.

The formal authorization overlay is necessary but not sufficient to execute a
training job.  A launcher-held, run-bound execution lease is also required so
that calling the Python CLI or runtime directly cannot bypass the single-GPU
lock.  That lease does not exist yet; consequently this module deliberately
cannot authorize a formal-v3 execution.

Dry runs and non-formal experiments return without importing or evaluating the
research authorization overlay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence


FORMAL_EXPERIMENT_PREFIX = "anchor-moe-lora-formal-v3"
FORMAL_DECISION_SCHEMA = "anchor.formal-authorization-decision.v1"
FORMAL_AUTHORIZATION_CONFIG = "configs/research/formal_authorization_consumer_v1.yaml"


class FormalExecutionGateError(RuntimeError):
    """Raised before any formal-v3 execution-side resource is touched."""


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalExecutionGateError(
            "formal-v3 authorization decision is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _looks_like_formal_artifact_path(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False

    # Win32 trims trailing spaces/dots from path components and resolves
    # existing junctions/short aliases.  Inspect both the lexical spelling and
    # the filesystem-normalized spelling so ``formal_v3.`` cannot alias the
    # protected directory while bypassing a plain string comparison.
    spellings = [value]
    try:
        spellings.append(str(Path(value).expanduser().resolve(strict=False)))
    except (OSError, RuntimeError, ValueError):
        pass
    for spelling in spellings:
        normalized = spelling.replace("\\", "/")
        try:
            parts = PurePosixPath(normalized).parts
        except (TypeError, ValueError):
            continue
        folded_parts = tuple(
            (
                part
                if len(part) == 2 and part[0].isalpha() and part[1] == ":"
                else part.partition(":")[0]
            )
            .rstrip(" .")
            .casefold()
            for part in parts
        )
        if any(
            folded_parts[index : index + 2] == ("artifacts", "formal_v3")
            for index in range(max(0, len(parts) - 1))
        ):
            return True
    return False


def _contains_formal_artifact_path(value: object) -> bool:
    """Recursively recognize protected formal paths in closed config trees."""

    if isinstance(value, Mapping):
        return any(_contains_formal_artifact_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_formal_artifact_path(item) for item in value)
    return _looks_like_formal_artifact_path(value)


def is_formal_v3_config(config: Mapping[str, Any]) -> bool:
    """Recognize formal-v3 by its experiment identity or protected paths."""

    experiment = config.get("experiment")
    if isinstance(experiment, str) and experiment.startswith(FORMAL_EXPERIMENT_PREFIX):
        return True

    return _contains_formal_artifact_path(config)


def _evaluate_authenticated_overlay() -> Mapping[str, Any]:
    # Import only after a real formal execution request has been identified.
    # The overlay authenticates all of its own dependency snapshots.
    from anchor_mvp.research.formal_authorization_consumer import (
        evaluate_formal_authorization,
    )

    return evaluate_formal_authorization(FORMAL_AUTHORIZATION_CONFIG)


def _reject_v1_formal_decision(value: object) -> NoReturn:
    """Authenticate the v1 envelope, then reject its blocked-only contract."""

    if not isinstance(value, Mapping):
        raise FormalExecutionGateError(
            "formal-v3 authorization decision must be a JSON object"
        )
    decision = dict(value)
    observed_digest = decision.get("decision_sha256")
    if (
        not isinstance(observed_digest, str)
        or len(observed_digest) != 64
        or any(character not in "0123456789abcdef" for character in observed_digest)
    ):
        raise FormalExecutionGateError(
            "formal-v3 authorization decision has an invalid decision_sha256"
        )
    canonical_body = dict(decision)
    canonical_body.pop("decision_sha256")
    if observed_digest != _canonical_sha256(canonical_body):
        raise FormalExecutionGateError(
            "formal-v3 authorization decision digest mismatch"
        )
    if decision.get("schema_version") != FORMAL_DECISION_SCHEMA:
        raise FormalExecutionGateError(
            "formal-v3 authorization decision schema is not eligible for execution"
        )
    if (
        decision.get("status") != "blocked_formal_authorization_inputs_unavailable"
        or decision.get("training_authorized") is not False
        or decision.get("formal_training_authorized") is not False
        or decision.get("formal") is not False
    ):
        raise FormalExecutionGateError(
            "formal-v3 authorization decision v1 is blocked-only; forged ready "
            "fields cannot authorize execution"
        )
    raise FormalExecutionGateError(
        "formal-v3 authorization decision v1 is blocked-only and cannot authorize "
        "execution; a versioned v2+ decision with an authenticated execution lease "
        "is required"
    )


def require_formal_execution_authorization(
    config: Mapping[str, Any],
    *,
    dataset_paths: Sequence[Path] = (),
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Block direct formal execution before GPU, data, model, or output access.

    A future ready decision still cannot authorize this call by itself.  The
    PowerShell launcher must first acquire the single-GPU lock and provide a
    cryptographically bound execution lease.  No such lease contract is
    available in v1, so the only safe outcome after decision validation is a
    fail-closed error.
    """

    protected_runtime_path = (
        any(_looks_like_formal_artifact_path(str(path)) for path in dataset_paths)
        or (
            output_dir is not None and _looks_like_formal_artifact_path(str(output_dir))
        )
        or (
            manifest_path is not None
            and _looks_like_formal_artifact_path(str(manifest_path))
        )
    )
    if not is_formal_v3_config(config) and not protected_runtime_path:
        return
    try:
        decision = _evaluate_authenticated_overlay()
    except FormalExecutionGateError:
        raise
    except Exception as exc:
        raise FormalExecutionGateError(
            "formal-v3 authorization overlay failed closed"
        ) from exc
    _reject_v1_formal_decision(decision)
