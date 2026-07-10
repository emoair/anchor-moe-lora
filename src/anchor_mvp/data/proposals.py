"""Deterministic inert tool proposals for policy-adapter distillation.

These objects describe hypothetical capabilities. They are never converted to
tool calls, URLs, shell commands, or execution requests by the data pipeline.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from .schema import SeedDemand, stable_id


PROPOSAL_GENERATOR_VERSION = "anchor-inert-tool-proposals-v1"


def generate_inert_tool_proposals(
    seed: SeedDemand, index: int
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return one reproducible abstract proposal set and public provenance."""

    variants: tuple[tuple[str, tuple[dict[str, str], ...]], ...] = (
        (
            "read_only_workspace",
            (
                {
                    "proposal_id": stable_id("proposal", f"{seed.seed_id}:read"),
                    "capability": "workspace.read_text",
                    "resource_scope": "workspace-root",
                    "side_effect": "none",
                    "purpose": "inspect project conventions and existing components",
                },
            ),
        ),
        (
            "bounded_workspace_write",
            (
                {
                    "proposal_id": stable_id("proposal", f"{seed.seed_id}:read-write"),
                    "capability": "workspace.read_text",
                    "resource_scope": "workspace-root",
                    "side_effect": "none",
                    "purpose": "inspect project conventions",
                },
                {
                    "proposal_id": stable_id("proposal", f"{seed.seed_id}:write"),
                    "capability": "workspace.write_derived_file",
                    "resource_scope": "workspace-generated-output",
                    "side_effect": "reversible",
                    "purpose": "write generated frontend artifacts inside the workspace",
                },
            ),
        ),
        (
            "external_irreversible",
            (
                {
                    "proposal_id": stable_id("proposal", f"{seed.seed_id}:publish"),
                    "capability": "external.publish",
                    "resource_scope": "unspecified-external-destination",
                    "side_effect": "irreversible",
                    "purpose": "publish generated artifacts outside the workspace",
                },
            ),
        ),
    )
    variant, raw = variants[index % len(variants)]
    proposals = [dict(item) for item in raw]
    canonical = json.dumps(proposals, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {
        "generator": PROPOSAL_GENERATOR_VERSION,
        "variant": variant,
        "sha256": sha256(canonical.encode("utf-8")).hexdigest(),
        "count": len(proposals),
        "executed": False,
    }
    return proposals, manifest

