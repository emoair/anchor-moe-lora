from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .types import (
    BackendError,
    CompletionRequest,
    CompletionResponse,
    TokenUsage,
)


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 0.25


class OpenAICompatibleClient:
    """Small dependency-free client for vLLM's chat-completions endpoint.

    vLLM exposes each statically loaded LoRA name as a model id. Therefore the
    request's ``model`` field is the adapter selector; no global mutable adapter
    switch is required for normal inference.
    """

    _RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        last_error: BaseException | None = None
        started = time.perf_counter()
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                payload = await asyncio.wait_for(
                    asyncio.to_thread(self._post_completion, request),
                    timeout=self.config.timeout_seconds,
                )
                return self._decode_response(
                    payload,
                    requested_model=request.model,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in self._RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, asyncio.TimeoutError) as exc:
                last_error = exc
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BackendError(f"invalid completion response: {exc}", attempts=attempt) from exc

            if attempt < self.config.max_attempts:
                await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempt - 1)))

        detail = self._error_detail(last_error)
        raise BackendError(
            f"chat completion failed after {self.config.max_attempts} attempt(s): {detail}",
            attempts=self.config.max_attempts,
        ) from last_error

    def _post_completion(self, request: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        body.update(request.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        http_request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _decode_response(
        payload: dict[str, Any],
        *,
        requested_model: str,
        attempts: int,
        latency_ms: float,
    ) -> CompletionResponse:
        choice = payload["choices"][0]
        usage_payload = payload.get("usage") or {}
        prompt_tokens = int(usage_payload.get("prompt_tokens", 0))
        completion_tokens = int(usage_payload.get("completion_tokens", 0))
        return CompletionResponse(
            content=str(choice["message"]["content"]),
            model=str(payload.get("model") or requested_model),
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=int(
                    usage_payload.get("total_tokens", prompt_tokens + completion_tokens)
                ),
            ),
            finish_reason=choice.get("finish_reason"),
            attempts=attempts,
            latency_ms=latency_ms,
            raw=payload,
        )

    @staticmethod
    def _error_detail(error: BaseException | None) -> str:
        if isinstance(error, urllib.error.HTTPError):
            try:
                body = error.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            return f"HTTP {error.code}: {body[:500]}"
        return str(error or "unknown error")


class RuntimeAdapterAdmin:
    """Explicit local-development helper for vLLM runtime LoRA endpoints.

    Never expose these methods to untrusted users. Prefer static ``--lora-modules``
    in any shared environment.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def load(self, name: str, path: str) -> None:
        await asyncio.to_thread(
            self._post, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}
        )

    async def unload(self, name: str) -> None:
        await asyncio.to_thread(
            self._post, "/v1/unload_lora_adapter", {"lora_name": name}
        )

    def _post(self, endpoint: str, payload: dict[str, str]) -> None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30.0):
            return None

