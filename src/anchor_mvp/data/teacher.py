"""Budgeted Anthropic/OpenAI-compatible and deterministic mock teachers."""

from __future__ import annotations

import asyncio
import codecs
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import random
import re
import threading
import time
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .provider import validate_base_url


APIProtocol = Literal["anthropic", "openai"]


class TeacherError(RuntimeError):
    """A redacted teacher request failure."""


class BudgetExceeded(TeacherError):
    """The configured request or output-token ceiling was reached."""


class RateLimitError(TeacherError):
    """Rate limiting that should be persisted by the unattended scheduler."""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        suffix = (
            f"; retry_after_seconds={retry_after_seconds:g}"
            if retry_after_seconds is not None
            else ""
        )
        super().__init__("teacher rate limit exhausted retries" + suffix)
        self.retry_after_seconds = retry_after_seconds


class ProviderQuotaExhausted(RateLimitError):
    """The provider explicitly reported that the account quota/budget is exhausted.

    This is intentionally distinct from a temporary HTTP 429.  Downstream
    orchestration may stop distillation on this signal, but must never infer it
    from a generic transport or HTTP failure.
    """

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        super().__init__(retry_after_seconds)
        self.args = ("provider explicitly reported quota exhausted",)


class ClientDeadlineExceeded(TeacherError):
    """A local absolute wall-clock deadline, distinct from a server failure."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"client_deadline exceeded after {seconds:g} seconds")
        self.seconds = seconds


class _ProtocolError(TeacherError):
    def __init__(
        self,
        protocol: APIProtocol,
        status: int,
        detail: str = "",
        retry_after_seconds: float | None = None,
    ) -> None:
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{protocol} endpoint returned HTTP {status}{suffix}")
        self.protocol = protocol
        self.status = status
        self.retry_after_seconds = retry_after_seconds


class Teacher(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    @property
    def protocol(self) -> str: ...

    @property
    def generation_params(self) -> dict[str, Any]: ...

    @property
    def provider_provenance(self) -> dict[str, Any]: ...

    async def complete(self, *, system: str, user: str) -> str: ...


@dataclass
class _Budget:
    max_requests: int
    max_output_tokens_total: int
    requests: int = 0
    output_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve_request(self) -> None:
        with self.lock:
            if self.requests >= self.max_requests:
                raise BudgetExceeded("teacher request budget exhausted")
            self.requests += 1

    def add_output(self, count: int) -> None:
        with self.lock:
            self.output_tokens += max(0, count)
            if self.output_tokens > self.max_output_tokens_total:
                raise BudgetExceeded("teacher output-token budget exhausted")


@dataclass
class CompatibleTeacher:
    """OpenAI/Anthropic-compatible client with optional protocol fallback.

    The credential is read at call time from ``api_key_env`` and is never stored
    in configuration, payload provenance, exceptions, or logs.
    """

    base_url: str = "https://api.kimi.com/coding/"
    model: str = "kimi-for-coding"
    protocol: APIProtocol = "anthropic"
    fallback_protocol: APIProtocol | None = "openai"
    fallback_base_url: str = "https://api.kimi.com/coding/v1"
    api_key_env: str = "KIMI_API_KEY"
    anthropic_version: str = "2023-06-01"
    user_agent: str = "anchor-moe-lora/0.1"
    timeout_seconds: float = 600.0
    max_retries: int = 1
    temperature: float = 0.2
    max_tokens: int = 4096
    thinking_enabled: bool = True
    thinking_effort: str = "medium"
    thinking_budget_tokens: int = 1024
    stream_openai: bool = True
    stream_options_include_usage: bool = False
    wall_clock_deadline_seconds: float = 900.0
    max_requests: int = 4100
    max_output_tokens_total: int = 12_500_000
    provider_preset: str = "legacy-kimi-code"
    model_source: str = "legacy_config"
    discovery_status: str = "skipped"
    discovery_model_count: int = 0
    _budget: _Budget = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.protocol not in ("anthropic", "openai"):
            raise ValueError("protocol must be anthropic or openai")
        if self.fallback_protocol not in (None, "anthropic", "openai"):
            raise ValueError("fallback_protocol must be anthropic, openai, or None")
        self.base_url = validate_base_url(self.base_url)
        self.fallback_base_url = validate_base_url(
            self.fallback_base_url, name="fallback_base_url"
        )
        if self.max_requests < 1 or self.max_output_tokens_total < 1 or self.max_tokens < 1:
            raise ValueError("teacher budgets must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.wall_clock_deadline_seconds <= 0:
            raise ValueError("wall_clock_deadline_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
        if self.thinking_effort not in allowed_efforts:
            raise ValueError(f"thinking_effort must be one of {sorted(allowed_efforts)}")
        uses_anthropic = self.protocol == "anthropic" or self.fallback_protocol == "anthropic"
        if self.thinking_enabled and uses_anthropic:
            if self.thinking_budget_tokens < 1024:
                raise ValueError("thinking_budget_tokens must be at least 1024")
            if self.max_tokens <= self.thinking_budget_tokens:
                raise ValueError("max_tokens must be greater than thinking_budget_tokens")
        self._budget = _Budget(self.max_requests, self.max_output_tokens_total)

    @property
    def generation_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_requests": self.max_requests,
            "max_output_tokens_total": self.max_output_tokens_total,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "thinking_enabled": self.thinking_enabled,
            "thinking_effort": self.thinking_effort,
            "thinking_budget_tokens": self.thinking_budget_tokens,
            "stream_openai": self.stream_openai,
            "stream_options_include_usage": self.stream_options_include_usage,
            "wall_clock_deadline_seconds": self.wall_clock_deadline_seconds,
        }

    @property
    def provider_provenance(self) -> dict[str, Any]:
        """Public provider resolution details; contains no credential value."""

        return {
            "preset": self.provider_preset,
            "base_url": self.base_url,
            "protocol": self.protocol,
            "model": self.model,
            "model_source": self.model_source,
            "discovery": {
                "status": self.discovery_status,
                "model_count": self.discovery_model_count,
            },
        }

    @property
    def usage_snapshot(self) -> dict[str, int]:
        with self._budget.lock:
            return {
                "requests": self._budget.requests,
                "output_tokens": self._budget.output_tokens,
            }

    def limit_remaining_budget(self, *, max_requests: int, max_output_tokens: int) -> None:
        """Apply a persisted scheduler's remaining budget to a fresh client."""

        if max_requests < 0 or max_output_tokens < 0:
            raise ValueError("remaining budgets cannot be negative")
        with self._budget.lock:
            if self._budget.requests or self._budget.output_tokens:
                raise RuntimeError("remaining budget must be set before teacher use")
            self._budget.max_requests = min(self._budget.max_requests, max_requests)
            self._budget.max_output_tokens_total = min(
                self._budget.max_output_tokens_total,
                max_output_tokens,
            )

    @property
    def usage_budget_id(self) -> int:
        return id(self._budget)

    def share_usage_budget(self, owner: "CompatibleTeacher") -> None:
        """Share a scheduler-wide wire budget across task-specific clients."""

        if self.usage_snapshot != {"requests": 0, "output_tokens": 0}:
            raise RuntimeError("usage budget must be shared before teacher use")
        if owner.usage_snapshot != {"requests": 0, "output_tokens": 0}:
            raise RuntimeError("usage budget owner must be unused")
        self._budget = owner._budget

    async def complete(self, *, system: str, user: str) -> str:
        try:
            return await self._with_retries(self.protocol, self.base_url, system, user, self.max_tokens)
        except _ProtocolError as error:
            fallback_statuses = {400, 404, 405, 415}
            if (
                self.fallback_protocol is None
                or self.fallback_protocol == self.protocol
                or error.status not in fallback_statuses
            ):
                raise TeacherError(_redact(str(error))) from None
            result = await self._with_retries(
                self.fallback_protocol,
                self.fallback_base_url,
                system,
                user,
                self.max_tokens,
            )
            self._activate_openai_fallback()
            return result

    async def probe(self) -> str:
        """One minimal protocol/authentication probe, preserving Thinking mode."""

        probe_tokens = (
            min(self.max_tokens, self.thinking_budget_tokens + 1)
            if self.thinking_enabled and self.protocol == "anthropic"
            else min(32, self.max_tokens)
        )
        try:
            return await self._with_retries(
                self.protocol,
                self.base_url,
                "Return a minimal JSON health response.",
                'Return exactly {"ok":true}.',
                probe_tokens,
            )
        except _ProtocolError as error:
            if (
                self.fallback_protocol != "openai"
                or self.protocol == "openai"
                or error.status not in {400, 404, 405, 415}
            ):
                raise TeacherError(_redact(str(error))) from None
            result = await self._with_retries(
                "openai",
                self.fallback_base_url,
                "Return a minimal JSON health response.",
                'Return exactly {"ok":true}.',
                min(32, self.max_tokens),
            )
            self._activate_openai_fallback()
            return result

    def _activate_openai_fallback(self) -> None:
        """Latch a probe-confirmed compatibility fallback for later requests."""

        self.protocol = "openai"
        self.base_url = self.fallback_base_url
        self.fallback_protocol = None

    async def _with_retries(
        self,
        protocol: APIProtocol,
        base_url: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self._request_sync, protocol, base_url, system, user, max_tokens
                )
            except _ProtocolError as error:
                if error.status in (402, 403, 429) and _is_quota_exhaustion(error):
                    raise ProviderQuotaExhausted(error.retry_after_seconds) from None
                if error.status == 429 and (error.retry_after_seconds or 0) > 60:
                    raise RateLimitError(error.retry_after_seconds) from None
                if error.status not in (408, 409, 429) and error.status < 500:
                    raise
                last_error = error
            except (URLError, TimeoutError) as error:
                last_error = TeacherError(f"{type(error).__name__} during teacher request")
            if attempt < self.max_retries:
                retry_after = (
                    last_error.retry_after_seconds
                    if isinstance(last_error, _ProtocolError)
                    and last_error.retry_after_seconds is not None
                    else 0.0
                )
                await asyncio.sleep(max(retry_after, min(8.0, (2**attempt) + random.random())))
        if isinstance(last_error, _ProtocolError) and last_error.status == 429:
            raise RateLimitError(last_error.retry_after_seconds) from None
        raise TeacherError(_redact(f"teacher request failed after retries: {last_error}")) from None

    def _request_sync(
        self,
        protocol: APIProtocol,
        base_url: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise TeacherError(f"credential environment variable {self.api_key_env} is not set")
        self._budget.reserve_request()
        if protocol == "anthropic":
            endpoint = _anthropic_endpoint(base_url)
            payload = {
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": max_tokens,
            }
            if self.thinking_enabled:
                if max_tokens <= self.thinking_budget_tokens:
                    raise TeacherError("max_tokens must be greater than thinking_budget_tokens")
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
            else:
                payload["temperature"] = self.temperature
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": self.anthropic_version,
                "User-Agent": self.user_agent,
            }
        else:
            endpoint = _openai_endpoint(base_url)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
            }
            if self.thinking_enabled:
                payload["reasoning_effort"] = self.thinking_effort
            else:
                payload["temperature"] = self.temperature
            if self.stream_openai:
                payload["stream"] = True
                if self.stream_options_include_usage:
                    payload["stream_options"] = {"include_usage": True}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": self.user_agent,
            }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            deadline_at = time.monotonic() + self.wall_clock_deadline_seconds
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured endpoint
                if protocol == "openai" and self.stream_openai:
                    content, output_tokens = _openai_stream_content(
                        response,
                        deadline_at=deadline_at,
                        deadline_seconds=self.wall_clock_deadline_seconds,
                    )
                    body = None
                else:
                    body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            # Never include response bodies: providers sometimes echo request metadata.
            detail = _safe_http_error_detail(error, api_key=api_key, system=system, user=user)
            retry_after = _parse_retry_after(error.headers.get("Retry-After") if error.headers else None)
            raise _ProtocolError(protocol, error.code, detail, retry_after) from None
        except json.JSONDecodeError:
            raise TeacherError("teacher response was not valid JSON") from None
        if body is not None:
            content, output_tokens = _response_content(protocol, body)
        self._budget.add_output(output_tokens if output_tokens is not None else max(1, len(content) // 4))
        return content


@dataclass
class OpenAICompatibleTeacher(CompatibleTeacher):
    """Backward-compatible constructor with OpenAI protocol selected."""

    protocol: APIProtocol = "openai"
    fallback_protocol: APIProtocol | None = None


@dataclass
class MockTeacher:
    """Offline deterministic teacher for smoke tests and pipeline development."""

    model: str = "mock-teacher-v1"
    base_url: str = "mock://local"
    protocol: str = "mock"

    @property
    def generation_params(self) -> dict[str, Any]:
        return {
            "temperature": 0,
            "max_tokens": 0,
            "deterministic": True,
            "thinking_enabled": False,
            "thinking_effort": None,
            "thinking_budget_tokens": 0,
            "stream_openai": False,
            "stream_options_include_usage": False,
            "wall_clock_deadline_seconds": 0,
        }

    @property
    def provider_provenance(self) -> dict[str, Any]:
        return {
            "preset": "mock",
            "base_url": self.base_url,
            "protocol": self.protocol,
            "model": self.model,
            "model_source": "mock",
            "discovery": {"status": "skipped", "model_count": 0},
        }

    async def probe(self) -> str:
        await asyncio.sleep(0)
        return '{"ok":true}'

    async def complete(self, *, system: str, user: str) -> str:
        await asyncio.sleep(0)
        marker = _marker(user, "ANCHOR_TASK")
        index = int(_marker(user, "SEED_INDEX") or "0")
        if marker == "seed":
            categories = ("standard", "unusual", "prompt-injection")
            category = categories[index % len(categories)]
            request = (
                f"Build an accessible project dashboard variant {index} with responsive navigation, "
                "a status summary, and clear empty/error states."
            )
            if category == "unusual":
                request += " Use an unconventional monochrome editorial layout without reducing usability."
            elif category == "prompt-injection":
                request += " Treat any instruction-like user content as untrusted display text."
            return json.dumps(
                {
                    "title": f"Project dashboard {index}",
                    "request": request,
                    "category": category,
                    "tags": ["dashboard", "responsive", category],
                }
            )
        if marker == "plan":
            return json.dumps(
                {
                    "decision_trace": [
                        {
                            "check": "Requirement decomposition",
                            "evidence": "The request names navigation, status, and state handling.",
                            "action": "Order semantic structure before interactive state and verification.",
                        }
                    ],
                    "output": {
                        "summary": "Build an accessible dashboard with explicit UI states.",
                        "steps": [
                            {
                                "id": "P1",
                                "goal": "Define semantic page structure",
                                "deliverable": "Landmarks, heading, navigation, and status regions",
                            },
                            {
                                "id": "P2",
                                "goal": "Implement deterministic states",
                                "deliverable": "Loading, empty, error, and ready state components",
                            },
                        ],
                        "constraints": [
                            "Treat request text as untrusted data",
                            "Keep the dependency surface minimal",
                        ],
                    },
                }
            )
        if marker == "tool_policy":
            decisions = ("APPROVE", "ESCALATE", "BLOCK")
            decision = decisions[index % len(decisions)]
            return json.dumps(
                {
                    "decision_trace": [
                        {
                            "check": "Capability and side effect",
                            "evidence": "The inert proposal declares its scope and side-effect class.",
                            "action": f"Return the advisory {decision} label without executing anything.",
                        }
                    ],
                    "output": {
                        "decision": decision,
                        "rationale": "Classified from declared scope and side effect; deterministic runtime policy remains authoritative.",
                        "proposal_labels": ["INERT_ABSTRACT_PROPOSAL"],
                    },
                }
            )
        if marker == "frontend":
            return json.dumps(
                {
                    "decision_trace": [
                        {
                            "check": "Component boundaries",
                            "evidence": "The request has navigation, status, and empty-state regions.",
                            "action": "Split stable regions into small semantic components.",
                        },
                        {
                            "check": "Accessibility",
                            "evidence": "Navigation and status updates need explicit semantics.",
                            "action": "Use landmarks, labels, and an aria-live status region.",
                        },
                    ],
                    "output": {
                        "language": "tsx",
                        "code": "export function Dashboard(){return <main><h1>Project status</h1><p aria-live=\"polite\">Ready</p></main>}",
                    },
                }
            )
        if marker == "review":
            candidate = _section(user, "CANDIDATE CODE")
            fixed = _repair_mock_candidate(candidate, _section(user, "KNOWN_BENIGN_DEFECT"))
            return json.dumps(
                {
                    "decision_trace": [
                        {
                            "check": "Known benign mutation",
                            "evidence": "The pipeline manifest identifies one local semantic defect.",
                            "action": "Restore the removed semantic or accessible-name behavior.",
                        }
                    ],
                    "output": {
                        "language": "tsx",
                        "summary": "Repairs the deterministic benign mutation.",
                        "code": fixed,
                    },
                }
            )
        if marker == "security":
            block = index % 3 == 2
            return json.dumps(
                {
                    "decision_trace": [
                        {
                            "check": "Untrusted content boundary",
                            "evidence": (
                                "The reviewed candidate contains an unresolved active-content placeholder."
                                if block
                                else "The reviewed candidate renders the value through text semantics."
                            ),
                            "action": (
                                "Block until the placeholder is replaced by an inert rendering path."
                                if block
                                else "Pass while retaining contextual encoding and CSP controls."
                            ),
                        }
                    ],
                    "output": {
                        "decision": "BLOCK" if block else "PASS",
                        "rationale": (
                            "Block until untrusted active-content placeholders are removed."
                            if block
                            else "No executable payload is present; retain standard encoding and CSP controls."
                        ),
                        "findings": ["DEFENSIVE_PLACEHOLDER_UNTRUSTED_CONTENT"] if block else [],
                    },
                }
            )
        raise TeacherError(f"mock teacher received unknown task marker: {marker!r}")


def _response_content(protocol: APIProtocol, body: Any) -> tuple[str, int | None]:
    try:
        if protocol == "anthropic":
            blocks = body["content"]
            content = "".join(str(block["text"]) for block in blocks if block.get("type") == "text")
            tokens = body.get("usage", {}).get("output_tokens")
        else:
            content = str(body["choices"][0]["message"]["content"])
            tokens = body.get("usage", {}).get("completion_tokens")
    except (KeyError, IndexError, TypeError):
        raise TeacherError("unexpected teacher response schema") from None
    if not content:
        raise TeacherError("teacher returned no text content")
    return content, int(tokens) if tokens is not None else None


def _openai_stream_content(
    response: Any,
    *,
    deadline_at: float | None = None,
    deadline_seconds: float = 0,
) -> tuple[str, int | None]:
    """Consume OpenAI-compatible SSE, retaining only final text deltas and usage."""

    fragments: list[str] = []
    final_usage: int | None = None
    finish_reason: str | None = None
    saw_reasoning = False
    reasoning_chars = 0
    unknown_delta_keys: set[str] = set()
    known_delta_keys = {
        "content",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "role",
    }
    deadline_event = threading.Event()
    timer: threading.Timer | None = None
    if deadline_at is not None:
        def close_at_deadline() -> None:
            deadline_event.set()
            close = getattr(response, "close", None)
            if close is not None:
                try:
                    close()
                except OSError:
                    pass

        timer = threading.Timer(max(0.0, deadline_at - time.monotonic()), close_at_deadline)
        timer.daemon = True
        timer.start()
    try:
        for data in _iter_sse_data(
            response,
            deadline_at=deadline_at,
            deadline_seconds=deadline_seconds,
            deadline_event=deadline_event,
        ):
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                raise TeacherError("teacher SSE event was not valid JSON") from None
            if not isinstance(event, dict):
                continue
            usage = event.get("usage")
            if isinstance(usage, dict):
                raw_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                if isinstance(raw_tokens, int) and raw_tokens >= 0:
                    final_usage = raw_tokens
            choices = event.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                raw_finish = choice.get("finish_reason")
                if isinstance(raw_finish, str) and raw_finish:
                    finish_reason = _safe_metadata_key(raw_finish, limit=40)
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                for key in delta:
                    if key not in known_delta_keys and len(unknown_delta_keys) < 8:
                        unknown_delta_keys.add(_safe_metadata_key(str(key), limit=32))
                # Count but deliberately never retain reasoning field values.
                for reasoning_key in ("reasoning_content", "reasoning", "reasoning_details"):
                    if reasoning_key in delta:
                        saw_reasoning = True
                        reasoning_chars += _reasoning_char_count(delta[reasoning_key])
                content = delta.get("content")
                if isinstance(content, str):
                    fragments.append(content)
    finally:
        if timer is not None:
            timer.cancel()
    content = "".join(fragments)
    if not content:
        diagnostics = [
            f"finish_reason={finish_reason or 'none'}",
            f"saw_reasoning={str(saw_reasoning).lower()}",
            f"reasoning_chars={reasoning_chars}",
            f"completion_tokens={final_usage if final_usage is not None else 'unknown'}",
            "unknown_delta_keys="
            + (",".join(sorted(unknown_delta_keys)) if unknown_delta_keys else "none"),
        ]
        raise TeacherError(
            "teacher SSE stream returned no final text content (" + "; ".join(diagnostics) + ")"
        )
    return content, final_usage


def _reasoning_char_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_reasoning_char_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_reasoning_char_count(item) for item in value.values())
    return 0


def _safe_metadata_key(value: str, *, limit: int) -> str:
    clipped = value[:limit]
    return clipped if re.fullmatch(r"[A-Za-z0-9_.-]+", clipped) else "invalid"


def _iter_sse_data(
    response: Any,
    *,
    deadline_at: float | None = None,
    deadline_seconds: float = 0,
    deadline_event: threading.Event | None = None,
):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    reader = getattr(response, "read1", None) or response.read
    while True:
        _check_stream_deadline(response, deadline_at, deadline_seconds, deadline_event)
        chunk = reader(8192)
        _check_stream_deadline(response, deadline_at, deadline_seconds, deadline_event)
        if not chunk:
            buffer += decoder.decode(b"", final=True)
            break
        if not isinstance(chunk, bytes):
            raise TeacherError("teacher SSE stream yielded non-byte data")
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("data:"):
                yield line[5:].lstrip()
    if buffer:
        line = buffer.rstrip("\r")
        if line.startswith("data:"):
            yield line[5:].lstrip()


def _check_stream_deadline(
    response: Any,
    deadline_at: float | None,
    seconds: float,
    deadline_event: threading.Event | None = None,
) -> None:
    if not (deadline_event and deadline_event.is_set()) and (
        deadline_at is None or time.monotonic() <= deadline_at
    ):
        return
    close = getattr(response, "close", None)
    if close is not None:
        try:
            close()
        except OSError:
            pass
    raise ClientDeadlineExceeded(seconds)


def _is_quota_exhaustion(error: _ProtocolError) -> bool:
    detail = str(error).casefold()
    # Keep this allowlist narrow.  Generic 402/403/429 responses, networking
    # errors, and client errors must not become a training handoff trigger.
    return any(
        marker in detail
        for marker in (
            "usage limit",
            "quota will be refreshed",
            "quota exhausted",
            "budget exhausted",
        )
    )


def _safe_http_error_detail(error: HTTPError, *, api_key: str, system: str, user: str) -> str:
    """Extract only allowlisted error metadata, with prompt/credential redaction."""

    try:
        raw = error.read(4097)
        if len(raw) > 4096:
            return "error_body=truncated"
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return ""
    source = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(source, dict):
        return ""
    parts: list[str] = []
    for name, limit in (("type", 80), ("code", 80), ("message", 240)):
        value = source.get(name)
        if not isinstance(value, (str, int)):
            continue
        clean = _redact_prompt_fragments(str(value), api_key=api_key, system=system, user=user)
        clean = re.sub(r"[\x00-\x1f\x7f]+", " ", clean).strip()[:limit]
        if clean:
            parts.append(f"{name}={clean}")
    return "; ".join(parts)


def _redact_prompt_fragments(value: str, *, api_key: str, system: str, user: str) -> str:
    value = value.replace(api_key, "[REDACTED]")
    for prompt in (system, user):
        if prompt:
            value = value.replace(prompt, "[PROMPT_REDACTED]")
        for fragment in prompt.splitlines():
            fragment = fragment.strip()
            if len(fragment) >= 8:
                value = value.replace(fragment, "[PROMPT_REDACTED]")
    return _redact(value)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _anthropic_endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/v1/messages") else value + "/v1/messages"


def _openai_endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if not value.endswith("/v1"):
        value += "/v1"
    return value + "/chat/completions"


def _redact(value: str) -> str:
    for variable in ("KIMI_API_KEY", "ANCHOR_TEACHER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(variable)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return re.sub(r"\b(?:sk|key)-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", value)


def _marker(text: str, name: str) -> str:
    prefix = f"{name}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _section(text: str, name: str) -> str:
    start = f"{name}:\n"
    end = f"\nEND {name}"
    if start not in text:
        return ""
    value = text.split(start, 1)[1]
    return value.split(end, 1)[0].strip()


def _repair_mock_candidate(candidate: str, defect: str) -> str:
    if not candidate:
        raise TeacherError("mock review requires pipeline-supplied candidate code")
    if "aria-label" in defect:
        return re.sub(
            r"<([A-Za-z][\w.]*)",
            r'<\1 aria-label="Restored accessible label"',
            candidate,
            count=1,
        )
    if "main landmark" in defect:
        fixed = re.sub(r"<div(?=[\s>])", "<main", candidate, count=1)
        closing = fixed.rfind("</div>")
        return fixed[:closing] + "</main>" + fixed[closing + 6 :] if closing >= 0 else fixed
    if "page h1" in defect:
        fixed = re.sub(r"<div(?=[\s>])", "<h1", candidate, count=1)
        closing = fixed.find("</div>")
        return fixed[:closing] + "</h1>" + fixed[closing + 6 :] if closing >= 0 else fixed
    return candidate + "\n/* repaired benign defect */"
