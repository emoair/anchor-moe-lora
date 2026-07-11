from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


_SHELL_META = re.compile(r"[\r\n;&|<>`$()]")
_TOOL_ALIASES = {"write": "edit", "patch": "edit"}


@dataclass(frozen=True)
class ToolPolicy:
    """Fail-closed tool policy shared by OpenCode and the local validator."""

    allowed_tools: tuple[str, ...] = ("read", "edit", "glob", "grep", "list", "bash")
    allowed_commands: tuple[str, ...] = (
        "npm run build --if-present",
        "npm run test --if-present",
        "npm run lint --if-present",
    )
    max_iterations: int = 8
    timeout_seconds: float = 900.0
    validation_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.timeout_seconds <= 0 or self.validation_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("allowed_tools contains duplicates")
        for command in self.allowed_commands:
            if self.normalize_command(command) != command:
                raise ValueError(f"command is not canonical or safe: {command!r}")

    @staticmethod
    def normalize_command(command: str) -> str:
        normalized = " ".join(command.split())
        if not normalized or _SHELL_META.search(normalized):
            return ""
        return normalized

    def is_tool_allowed(self, tool: str) -> bool:
        normalized = tool.lower()
        normalized = _TOOL_ALIASES.get(normalized, normalized)
        return normalized in self.allowed_tools

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
