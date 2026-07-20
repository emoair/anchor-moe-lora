"""Versioned tool contract shared by OpenCode execution and session conversion."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


EXECUTION_TOOL_CONTRACT_VERSION = "anchor.execution-tool-contract.v2"
EXECUTION_TOOL_CONTRACT_V3_VERSION = "anchor.execution-tool-contract.v3"
EXECUTION_TOOLS = frozenset(
    {"read", "bash", "edit", "apply_patch", "write", "grep", "glob", "list"}
)
PATH_REQUIRED_TOOLS = frozenset({"read", "edit", "write", "list"})
SEARCH_TOOLS = frozenset({"grep", "glob"})
ALLOWED_NPM_COMMANDS = frozenset(
    {
        "npm run build --if-present",
        "npm run test --if-present",
        "npm run lint --if-present",
    }
)
ALLOWED_VALIDATOR_COMMANDS = ALLOWED_NPM_COMMANDS | frozenset(
    {"npm run build", "npm run test", "npm run lint"}
)
V3_MODEL_TOOLS = frozenset(
    {"read", "bash", "edit", "apply_patch", "write", "grep", "glob", "list"}
)
V3_PUBLIC_VALIDATOR_COMMANDS = (
    "anchor-validate compile",
    "anchor-validate test",
    "anchor-validate lint",
)
V3_HIDDEN_OFFICIAL_EVAL_PROVENANCE = (
    "official-swebench-harness-system-private"
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def normalized_command(value: str) -> str:
    return " ".join(value.split())


def validate_search_input(tool: str, value: Mapping[str, Any]) -> None:
    """Validate non-shell search arguments before transcript retention.

    Grep patterns are regular expressions and may contain punctuation. Glob patterns
    describe workspace-relative files and therefore cannot be absolute or traverse up.
    Optional search roots are normalized separately by the session safety gate.
    """

    pattern = value.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip() or len(pattern.encode("utf-8")) > 4096:
        raise ValueError(f"{tool}_pattern_invalid")
    if tool == "glob" and (
        pattern.startswith(("/", "\\"))
        or _WINDOWS_ABSOLUTE.match(pattern)
        or any(part == ".." for part in re.split(r"[\\/]", pattern))
    ):
        raise ValueError("glob_pattern_escapes_workspace")


def contract_descriptor() -> dict[str, object]:
    return {
        "version": EXECUTION_TOOL_CONTRACT_VERSION,
        "tools": sorted(EXECUTION_TOOLS),
        "bash_commands": sorted(ALLOWED_NPM_COMMANDS),
    }


def v3_contract_descriptor() -> dict[str, object]:
    """Contract for isolated model iteration and hidden official evaluation.

    Model shell access is not a host shell and is not a static broad allowlist:
    each exact command must be proposed by the planner and approved by the tool
    policy before it can run inside the instance container.  Its real result is
    model-visible so the builder can iterate.  The official SWE-bench evaluator
    is a separate post-agent, fresh-container channel whose result is never
    exposed to the model.
    """

    return {
        "version": EXECUTION_TOOL_CONTRACT_V3_VERSION,
        "model_tools": sorted(V3_MODEL_TOOLS),
        "model_bash_policy": {
            "allow_host_shell": False,
            "command_authorization": (
                "planner_proposal_intersect_tool_policy_approve_exact_string"
            ),
            "execution_scope": "isolated-instance-container",
            "workdir": "/testbed",
            "network": "none-with-supervisor-unix-socket-loopback-bridge",
            "route_scope": "fixed-target-ccswitch-only",
            "public_egress_blocking": "enforced-and-behavior-probed",
            "result_visible_to_model": True,
        },
        "public_repo_validation": {
            "commands": list(V3_PUBLIC_VALIDATOR_COMMANDS),
            "phase": "model-iteration",
            "result_visible_to_model": True,
        },
        "hidden_official_eval": {
            "provenance": V3_HIDDEN_OFFICIAL_EVAL_PROVENANCE,
            "phase": "post-agent",
            "fresh_container": True,
            "network": "none",
            "result_visible_to_model": False,
        },
    }
