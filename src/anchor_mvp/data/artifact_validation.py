"""Isolated build/test gate for generated TSX fragments.

Generated code is written only as JSON string data into a copied,
repository-controlled fixture. The trusted fixture validators inspect those
strings; they never import, eval, or otherwise execute a generated fragment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ..tooling.policy import ToolPolicy
from ..tooling.validation import run_validations_with_output
from ..tooling.workspace import WorkspaceManager


TSX_FRAGMENT_VALIDATOR_VERSION = "anchor-tsx-fragment-build-test-v2"
_BATCH_RESULT_SCHEMA = "anchor.tsx-fragment-batch-result.v1"
_BATCH_RESULT_SENTINEL = "ANCHOR_BATCH_RESULT:"
_REQUIRED_VALIDATIONS = ("build", "test")


class _BatchProtocolError(ValueError):
    """The trusted validator did not return a complete, coherent batch result."""


def _empty_report() -> dict[str, Any]:
    return {
        "passed": False,
        "validator": TSX_FRAGMENT_VALIDATOR_VERSION,
        "reason": "empty_artifact",
    }


def _infrastructure_failure_report(digest: str) -> dict[str, Any]:
    return {
        "passed": False,
        "validator": TSX_FRAGMENT_VALIDATOR_VERSION,
        "code_sha256": digest,
        "reason": "validator_infrastructure_failure",
        "validations": [],
    }


def _parse_batch_capture(
    capture: Mapping[str, object],
    *,
    mode: str,
    expected_ids: frozenset[str],
) -> dict[str, dict[str, object]]:
    output = capture.get("stdout")
    if not isinstance(output, str):
        raise _BatchProtocolError(f"{mode} validator output is not text")
    encoded_results = [
        line[len(_BATCH_RESULT_SENTINEL) :]
        for line in output.splitlines()
        if line.startswith(_BATCH_RESULT_SENTINEL)
    ]
    if len(encoded_results) != 1:
        raise _BatchProtocolError(
            f"{mode} validator emitted {len(encoded_results)} batch results"
        )
    try:
        payload = json.loads(encoded_results[0])
    except json.JSONDecodeError as error:
        raise _BatchProtocolError(f"{mode} validator emitted invalid JSON") from error
    if not isinstance(payload, dict):
        raise _BatchProtocolError(f"{mode} batch result must be an object")
    if payload.get("schema") != _BATCH_RESULT_SCHEMA or payload.get("mode") != mode:
        raise _BatchProtocolError(f"{mode} batch result binding mismatch")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise _BatchProtocolError(f"{mode} batch results must be a list")

    results: dict[str, dict[str, object]] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise _BatchProtocolError(f"{mode} batch item must be an object")
        item_id = raw.get("id")
        passed = raw.get("passed")
        reason = raw.get("reason")
        if (
            not isinstance(item_id, str)
            or item_id in results
            or not isinstance(passed, bool)
            or not isinstance(reason, str)
            or not reason
        ):
            raise _BatchProtocolError(f"{mode} batch item is malformed")
        results[item_id] = {"passed": passed, "reason": reason}
    if frozenset(results) != expected_ids:
        raise _BatchProtocolError(f"{mode} batch result ID set mismatch")

    status = capture.get("status")
    exit_code = capture.get("exit_code")
    all_passed = all(bool(item["passed"]) for item in results.values())
    if status == "TIMEOUT" or (status == "PASS") != all_passed:
        raise _BatchProtocolError(f"{mode} batch exit status is inconsistent")
    if (exit_code == 0) != all_passed:
        raise _BatchProtocolError(f"{mode} batch exit code is inconsistent")
    return results


def _batch_reports(
    digests_to_code: Mapping[str, str],
    *,
    validations: Sequence[object],
    captures: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, Any]]:
    validation_by_name = {str(getattr(item, "name", "")): item for item in validations}
    capture_by_name = {
        str(item.get("name", "")): item
        for item in captures
        if isinstance(item, Mapping)
    }
    expected_ids = frozenset(digests_to_code)
    parsed: dict[str, dict[str, dict[str, object]]] = {}
    try:
        for mode in _REQUIRED_VALIDATIONS:
            validation = validation_by_name.get(mode)
            capture = capture_by_name.get(mode)
            if (
                validation is None
                or not bool(getattr(validation, "script_present", False))
                or capture is None
            ):
                raise _BatchProtocolError(f"required {mode} script did not run")
            parsed[mode] = _parse_batch_capture(
                capture,
                mode=mode,
                expected_ids=expected_ids,
            )
    except _BatchProtocolError:
        return {
            digest: _infrastructure_failure_report(digest) for digest in digests_to_code
        }

    reports: dict[str, dict[str, Any]] = {}
    for digest in digests_to_code:
        item_validations: list[dict[str, object]] = []
        reasons: list[str] = []
        for mode in _REQUIRED_VALIDATIONS:
            item = parsed[mode][digest]
            passed = bool(item["passed"])
            if not passed:
                reasons.append(str(item["reason"]))
            aggregate = validation_by_name[mode]
            item_validations.append(
                {
                    "name": mode,
                    "status": "PASS" if passed else "FAIL",
                    "script_present": True,
                    "exit_code": 0 if passed else 1,
                    "duration_ms": round(
                        float(getattr(aggregate, "duration_ms", 0.0)), 3
                    ),
                    "output_sha256": getattr(aggregate, "output_sha256", None),
                }
            )
        for aggregate in validations:
            name = str(getattr(aggregate, "name", ""))
            if name in _REQUIRED_VALIDATIONS:
                continue
            item_validations.append(
                {
                    "name": name,
                    "status": getattr(aggregate, "status", "FAIL"),
                    "script_present": bool(getattr(aggregate, "script_present", False)),
                    "exit_code": getattr(aggregate, "exit_code", None),
                    "duration_ms": round(
                        float(getattr(aggregate, "duration_ms", 0.0)), 3
                    ),
                    "output_sha256": getattr(aggregate, "output_sha256", None),
                }
            )
        report: dict[str, Any] = {
            "passed": not reasons,
            "validator": TSX_FRAGMENT_VALIDATOR_VERSION,
            "code_sha256": digest,
            "validations": item_validations,
        }
        if reasons:
            report["reason"] = reasons[0]
        reports[digest] = report
    return reports


def validate_tsx_fragments(
    codes: Iterable[str],
    *,
    fixture_root: Path,
    workspace_root: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Validate many TSX strings with one workspace and two Node launches.

    Results retain input order. Repeated SHA-256-identical strings are placed in
    the trusted batch payload once and share the same validation report.
    """

    materialized = list(codes)
    reports: list[dict[str, Any] | None] = [None] * len(materialized)
    digests_to_code: dict[str, str] = {}
    positions_by_digest: dict[str, list[int]] = {}
    for index, code in enumerate(materialized):
        if not isinstance(code, str):
            raise TypeError("tsx artifact must be a string")
        if not code.strip():
            reports[index] = _empty_report()
            continue
        digest = sha256(code.encode("utf-8")).hexdigest()
        previous = digests_to_code.setdefault(digest, code)
        if previous != code:
            raise ValueError("distinct TSX artifacts produced the same SHA-256 digest")
        positions_by_digest.setdefault(digest, []).append(index)

    if not digests_to_code:
        return [dict(report) for report in reports if report is not None]
    if not fixture_root.is_dir():
        raise ValueError(f"tsx validation fixture is not a directory: {fixture_root}")

    batch_digest = sha256("\n".join(digests_to_code).encode("ascii")).hexdigest()
    manager = WorkspaceManager(workspace_root)
    workspace = manager.prepare(f"tsx-batch-{batch_digest[:16]}", fixture_root)
    try:
        payload = {
            "schema": "anchor.tsx-fragment-batch-input.v1",
            "submissions": [
                {"id": digest, "code": code} for digest, code in digests_to_code.items()
            ],
        }
        (workspace / "submissions.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        policy = ToolPolicy(validation_timeout_seconds=timeout_seconds)
        validations, _, captures = run_validations_with_output(workspace, policy)
        by_digest = _batch_reports(
            digests_to_code,
            validations=validations,
            captures=captures,
        )
        for digest, positions in positions_by_digest.items():
            for index in positions:
                reports[index] = deepcopy(by_digest[digest])
    finally:
        manager.cleanup(workspace)

    if any(report is None for report in reports):
        raise RuntimeError("internal TSX batch result alignment failure")
    return [dict(report) for report in reports if report is not None]


def validate_tsx_fragment(
    code: str,
    *,
    fixture_root: Path,
    workspace_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Preserve the single-artifact API on top of the batch validator."""

    return validate_tsx_fragments(
        [code],
        fixture_root=fixture_root,
        workspace_root=workspace_root,
        timeout_seconds=timeout_seconds,
    )[0]
