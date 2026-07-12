"""Isolated build/test gate for generated TSX fragments.

Generated code is written only as data into a copied, repository-controlled
fixture. The fixture validators inspect that file; they never import, eval, or
otherwise execute the generated fragment.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from ..tooling.policy import ToolPolicy
from ..tooling.validation import run_validations
from ..tooling.workspace import WorkspaceManager


TSX_FRAGMENT_VALIDATOR_VERSION = "anchor-tsx-fragment-build-test-v1"


def validate_tsx_fragment(
    code: str,
    *,
    fixture_root: Path,
    workspace_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run the trusted fixture's build and test scripts against code as data."""

    if not code.strip():
        return {
            "passed": False,
            "validator": TSX_FRAGMENT_VALIDATOR_VERSION,
            "reason": "empty_artifact",
        }
    if not fixture_root.is_dir():
        raise ValueError(f"tsx validation fixture is not a directory: {fixture_root}")
    digest = sha256(code.encode("utf-8")).hexdigest()
    manager = WorkspaceManager(workspace_root)
    workspace = manager.prepare(f"tsx-{digest[:16]}", fixture_root)
    try:
        (workspace / "submission.tsx").write_text(code, encoding="utf-8", newline="\n")
        policy = ToolPolicy(validation_timeout_seconds=timeout_seconds)
        validations, _ = run_validations(workspace, policy)
        by_name = {item.name: item for item in validations}
        passed = all(
            name in by_name
            and by_name[name].script_present
            and by_name[name].status == "PASS"
            for name in ("build", "test")
        )
        return {
            "passed": passed,
            "validator": TSX_FRAGMENT_VALIDATOR_VERSION,
            "code_sha256": digest,
            "validations": [
                {
                    "name": item.name,
                    "status": item.status,
                    "script_present": item.script_present,
                    "exit_code": item.exit_code,
                    "duration_ms": round(item.duration_ms, 3),
                    "output_sha256": item.output_sha256,
                }
                for item in validations
            ],
        }
    finally:
        manager.cleanup(workspace)
