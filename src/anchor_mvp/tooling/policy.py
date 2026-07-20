from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .tool_contract import EXECUTION_TOOLS


_SHELL_META = re.compile(r"[\r\n;&|<>`$()]")
# OpenCode authorizes all file mutation tools through its ``edit`` permission.
# Keep that permission family explicit and shared by policy construction,
# runtime trace reduction, and proposal binding.
_TOOL_ALIASES = {"write": "edit", "patch": "edit", "apply_patch": "edit"}


@dataclass(frozen=True)
class ToolPolicy:
    """Fail-closed tool policy shared by OpenCode and the local validator."""

    allowed_tools: tuple[str, ...] = tuple(sorted(EXECUTION_TOOLS))
    allowed_commands: tuple[str, ...] = (
        "npm run build --if-present",
        "npm run test --if-present",
        "npm run lint --if-present",
    )
    max_iterations: int | None = None
    timeout_seconds: float = 900.0
    validation_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_iterations is not None and (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError(
                "max_iterations must be a positive integer when configured"
            )
        if self.timeout_seconds <= 0 or self.validation_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("allowed_tools contains duplicates")
        for command in self.allowed_commands:
            if self.normalize_command(command) != command:
                raise ValueError(f"command is not canonical or safe: {command!r}")

    @staticmethod
    def normalize_tool(tool: str) -> str:
        """Return the canonical execution-contract name for an OpenCode tool."""

        normalized = tool.lower()
        return _TOOL_ALIASES.get(normalized, normalized)

    @staticmethod
    def normalize_command(command: str) -> str:
        normalized = " ".join(command.split())
        if not normalized or _SHELL_META.search(normalized):
            return ""
        return normalized

    def is_tool_allowed(self, tool: str) -> bool:
        return self.normalize_tool(tool) in self.allowed_tools

    def is_command_allowed(self, command: str) -> bool:
        normalized = self.normalize_command(command)
        return bool(normalized) and normalized in self.allowed_commands

    @staticmethod
    def command_digest(command: str) -> str:
        return hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()

    def opencode_permissions(self) -> dict[str, object]:
        permissions: dict[str, object] = {"*": "deny"}
        for tool in self.allowed_tools:
            if tool != "bash":
                permissions[tool] = "allow"
        permissions["external_directory"] = "deny"
        permissions["task"] = "deny"
        permissions["skill"] = "deny"
        permissions["webfetch"] = "deny"
        permissions["websearch"] = "deny"
        permissions["lsp"] = "deny"
        bash_rules: dict[str, str] = {"*": "deny"}
        for command in self.allowed_commands:
            bash_rules[command] = "allow"
        permissions["bash"] = bash_rules
        return permissions
