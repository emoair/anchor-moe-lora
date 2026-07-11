"""Versioned tool contract shared by OpenCode execution and session conversion."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


EXECUTION_TOOL_CONTRACT_VERSION = "anchor.execution-tool-contract.v2"
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
