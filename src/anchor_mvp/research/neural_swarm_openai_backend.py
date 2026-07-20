"""OpenAI-compatible SSE transport for the Neural Swarm stream controller.

This optional adapter targets servers such as vLLM and llama.cpp that expose
``POST /chat/completions`` with OpenAI-compatible server-sent events.  It only
implements transport: model routing still comes from :class:`ExpertBinding`,
and no CUDA overlap or shared-KV behaviour is claimed here.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from anchor_mvp.research.neural_swarm_streaming import BackendChunk, ExpertBinding


_RESERVED_GENERATION_KEYS = frozenset({"messages", "model", "stream"})
_UNSUPPORTED_GENERATION_KEYS = frozenset(
    {"tools", "tool_choice", "parallel_tool_calls"}
)


class OpenAIStreamingBackendError(RuntimeError):
    """Base error for the OpenAI-compatible streaming transport."""


class OpenAIStreamingHTTPError(OpenAIStreamingBackendError):
    """Raised for an HTTP or network failure without echoing credentials."""


class OpenAIStreamingProtocolError(OpenAIStreamingBackendError):
    """Raised when a successful response violates the expected SSE protocol."""


@dataclass(frozen=True, slots=True)
class OpenAIStreamingBackendConfig:
    """Connection settings for one OpenAI-compatible endpoint.

    ``api_key`` is deliberately excluded from repr/equality.  ``api_key_env``
    resolves at request time, allowing a launch-only environment variable.
    Both authentication fields may be omitted for a trusted local endpoint.
    """

    base_url: str
    api_key: str | None = field(default=None, repr=False, compare=False)
    api_key_env: str | None = None
    timeout: float = 120.0
    cleanup_timeout: float = 1.0

    def __post_init__(self) -> None:
        base_url = self.base_url.strip()
        try:
            parsed = httpx.URL(base_url)
        except (TypeError, ValueError) as exc:
            raise ValueError("base_url must be a valid absolute HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("base_url must be a valid absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("api_key must be non-empty when provided")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must be non-empty when provided")
        if self.api_key is not None and self.api_key_env is not None:
            raise ValueError("configure either api_key or api_key_env, not both")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(self.cleanup_timeout) or self.cleanup_timeout <= 0:
            raise ValueError("cleanup_timeout must be a positive finite number")
        object.__setattr__(self, "base_url", base_url.rstrip("/"))

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def resolve_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None or not value.strip():
            raise OpenAIStreamingBackendError(
                f"API credential environment variable {self.api_key_env!r} is not set"
            )
        return value


def _plain_json(value: Any, *, path: str) -> Any:
    """Copy a frozen/shared value into JSON-compatible plain containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OpenAIStreamingBackendError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OpenAIStreamingBackendError(f"{path} keys must be strings")
            result[key] = _plain_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _plain_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise OpenAIStreamingBackendError(
        f"{path} must contain only JSON-compatible values"
    )


def _build_request_body(
    binding: ExpertBinding, shared_input: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = {"messages", "generation"}
    unknown = sorted(set(shared_input).difference(allowed))
    if unknown:
        raise OpenAIStreamingBackendError(
            "shared_input contains unsupported field(s): " + ", ".join(unknown)
        )
    if "messages" not in shared_input:
        raise OpenAIStreamingBackendError("shared_input.messages is required")
    messages = _plain_json(shared_input["messages"], path="shared_input.messages")
    if not isinstance(messages, list) or not messages:
        raise OpenAIStreamingBackendError(
            "shared_input.messages must be a non-empty sequence"
        )
    if any(not isinstance(message, dict) for message in messages):
        raise OpenAIStreamingBackendError(
            "each shared_input.messages item must be a mapping"
        )

    generation_value = shared_input.get("generation", {})
    generation = _plain_json(generation_value, path="shared_input.generation")
    if not isinstance(generation, dict):
        raise OpenAIStreamingBackendError("shared_input.generation must be a mapping")
    reserved = sorted(set(generation).intersection(_RESERVED_GENERATION_KEYS))
    if reserved:
        raise OpenAIStreamingBackendError(
            "shared_input.generation cannot override: " + ", ".join(reserved)
        )
    unsupported = sorted(
        set(generation).intersection(_UNSUPPORTED_GENERATION_KEYS)
    )
    if unsupported:
        raise OpenAIStreamingBackendError(
            "text-only streaming backend does not support: "
            + ", ".join(unsupported)
        )
    if "n" in generation and generation["n"] != 1:
        raise OpenAIStreamingBackendError(
            "shared_input.generation.n must be 1 for a single logical stream"
        )

    return {
        **generation,
        "model": binding.backend_model_id,
        "messages": messages,
        "stream": True,
    }


async def _next_line_or_cancel(
    iterator: AsyncIterator[str],
    cancel_event: asyncio.Event,
    *,
    cleanup_timeout: float,
) -> str | None:
    if cancel_event.is_set():
        return None
    next_task = asyncio.create_task(
        iterator.__anext__(), name="anchor-openai-sse-next-line"
    )
    cancel_task = asyncio.create_task(
        cancel_event.wait(), name="anchor-openai-sse-cancel-watch"
    )
    try:
        done, _ = await asyncio.wait(
            {next_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and cancel_event.is_set():
            return None
        return await next_task
    finally:
        for task in (next_task, cancel_task):
            task.add_done_callback(_consume_task_result)
            if not task.done():
                task.cancel()
        done, pending = await asyncio.wait(
            {next_task, cancel_task}, timeout=cleanup_timeout
        )
        for task in done:
            _consume_task_result(task)
        current = asyncio.current_task()
        caller_is_cancelling = current is not None and current.cancelling() > 0
        if pending and not caller_is_cancelling:
            raise OpenAIStreamingHTTPError(
                "OpenAI-compatible stream cleanup timed out"
            )


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


def _decode_data_event(data_lines: list[str], *, event_number: int) -> tuple[bool, list[BackendChunk]]:
    payload_text = "\n".join(data_lines)
    if payload_text.strip() == "[DONE]":
        return True, []
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} contains invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} JSON must be an object"
        )
    if "error" in payload:
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} reports a server-side error"
        )

    usage_value = payload.get("usage")
    if usage_value is not None and not isinstance(usage_value, Mapping):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} usage must be an object or null"
        )
    usage = dict(usage_value) if isinstance(usage_value, Mapping) else None
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} choices must be an array"
        )
    if len(choices) > 1:
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} contains multiple choices for one stream"
        )

    chunks: list[BackendChunk] = []
    if not choices:
        if usage is not None:
            chunks.append(BackendChunk(delta="", metadata={"usage": usage}))
        return False, chunks

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} choice must be an object"
        )
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} choice.delta must be an object"
        )
    unsupported_delta = sorted(
        set(delta).intersection({"tool_calls", "function_call", "audio"})
    )
    if unsupported_delta:
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} contains unsupported delta field(s): "
            + ", ".join(unsupported_delta)
        )
    content = delta.get("content")
    if content is not None and not isinstance(content, str):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} delta.content must be text or null"
        )
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise OpenAIStreamingProtocolError(
            f"SSE event {event_number} finish_reason must be text or null"
        )

    metadata: dict[str, Any] = {}
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    if usage is not None:
        metadata["usage"] = usage
    choice_index = choice.get("index")
    if isinstance(choice_index, int):
        metadata["choice_index"] = choice_index
    if content or metadata:
        chunks.append(BackendChunk(delta=content or "", metadata=metadata))
    return False, chunks


class OpenAICompatibleSSEBackend:
    """Stream text from an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        config: OpenAIStreamingBackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        del run_id, task_bundle_sha256
        if cancel_event.is_set():
            return
        body = _build_request_body(binding, shared_input)
        api_key = self._config.resolve_api_key()
        headers = {"Accept": "text/event-stream"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    "POST",
                    self._config.chat_completions_url,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code < 200 or response.status_code >= 300:
                        raise OpenAIStreamingHTTPError(
                            "OpenAI-compatible endpoint returned HTTP "
                            f"{response.status_code}"
                        )
                    content_type = response.headers.get("content-type")
                    if content_type and "text/event-stream" not in content_type.lower():
                        raise OpenAIStreamingProtocolError(
                            "OpenAI-compatible endpoint did not return text/event-stream"
                        )

                    iterator = response.aiter_lines().__aiter__()
                    data_lines: list[str] = []
                    event_number = 0
                    saw_done = False
                    cancelled = False
                    while True:
                        try:
                            line = await _next_line_or_cancel(
                                iterator,
                                cancel_event,
                                cleanup_timeout=self._config.cleanup_timeout,
                            )
                        except StopAsyncIteration:
                            break
                        if line is None:
                            cancelled = True
                            break
                        if line == "":
                            if not data_lines:
                                continue
                            event_number += 1
                            done, chunks = _decode_data_event(
                                data_lines, event_number=event_number
                            )
                            data_lines = []
                            for chunk in chunks:
                                yield chunk
                            if done:
                                saw_done = True
                                break
                            continue
                        if line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            value = line[5:]
                            if value.startswith(" "):
                                value = value[1:]
                            data_lines.append(value)

                    if cancelled:
                        return
                    if data_lines:
                        event_number += 1
                        done, chunks = _decode_data_event(
                            data_lines, event_number=event_number
                        )
                        for chunk in chunks:
                            yield chunk
                        saw_done = saw_done or done
                    if not saw_done:
                        raise OpenAIStreamingProtocolError(
                            "SSE stream ended before the [DONE] marker"
                        )
        except (OpenAIStreamingBackendError, asyncio.CancelledError):
            raise
        except httpx.HTTPError as exc:
            raise OpenAIStreamingHTTPError(
                "OpenAI-compatible stream request failed with "
                f"{type(exc).__name__}"
            ) from exc
