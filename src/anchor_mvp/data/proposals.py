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
TOOL_POLICY_ORACLE_VERSION = "anchor-tool-policy-oracle-v1"


def deterministic_tool_policy_oracle(
    proposals: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Classify inert proposals without consulting a model.

    The oracle is deliberately conservative: an unknown capability is blocked,
    bounded reversible writes require human approval, and only read-only
    workspace inspection is approved automatically.
    """

    labels: list[str] = []
    decision = "APPROVE"
    rationale = "All proposals are read-only and bounded to the workspace."
    for proposal in proposals:
        capability = proposal.get("capability", "")
        scope = proposal.get("resource_scope", "")
        side_effect = proposal.get("side_effect", "")
        if capability == "workspace.read_text" and scope == "workspace-root" and side_effect == "none":
            labels.append("INERT_READ_ONLY_WORKSPACE")
            continue
        if (
            capability == "workspace.write_derived_file"
            and scope == "workspace-generated-output"
            and side_effect == "reversible"
        ):
            labels.append("INERT_BOUNDED_REVERSIBLE_WRITE")
            if decision != "BLOCK":
                decision = "ESCALATE"
                rationale = "A bounded reversible write requires explicit human approval."
            continue
        labels.append("INERT_PROHIBITED_OR_UNKNOWN_CAPABILITY")
        decision = "BLOCK"
        rationale = "An external, irreversible, or unknown capability is prohibited."

    output = {
        "decision": decision,
        "rationale": rationale,
        "proposal_labels": sorted(set(labels)),
    }
    canonical = json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {
        "oracle": TOOL_POLICY_ORACLE_VERSION,
        "decision": decision,
        "sha256": sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return output, manifest


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
