from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: Sequence[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class CompletionResponse:
    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class CompletionBackend(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return one OpenAI-compatible chat completion."""


class BackendError(RuntimeError):
    """A serving request exhausted its retry budget."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts

