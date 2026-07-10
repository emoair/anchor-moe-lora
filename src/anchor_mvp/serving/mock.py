from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import defaultdict
from collections.abc import Callable

from .types import CompletionRequest, CompletionResponse, TokenUsage


MockHandler = Callable[[CompletionRequest], CompletionResponse | str]


class MockBackend:
    """Deterministic offline backend with call recording and failure injection."""

    def __init__(
        self,
        handlers: dict[str, MockHandler] | None = None,
        *,
        failures_before_success: dict[str, int] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.handlers = handlers or {}
        self.failures_before_success = failures_before_success or {}
        self.delay_seconds = delay_seconds
        self.requests: list[CompletionRequest] = []
        self.call_counts: dict[str, int] = defaultdict(int)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        self.requests.append(request)
        self.call_counts[request.model] += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.call_counts[request.model] <= self.failures_before_success.get(request.model, 0):
            raise RuntimeError(f"injected failure for {request.model}")

        handler = self.handlers.get(request.model)
        if handler:
            result = handler(request)
            if inspect.isawaitable(result):
                result = await result
        else:
            result = self._default_response(request)
        if isinstance(result, CompletionResponse):
            return result
        prompt_tokens = sum(max(1, len(message.content.split())) for message in request.messages)
        completion_tokens = max(1, len(result.split()))
        return CompletionResponse(
            content=result,
            model=request.model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _default_response(request: CompletionRequest) -> str:
        model = request.model.lower()
        user_text = request.messages[-1].content if request.messages else ""
        if "security" in model:
            decision = "BLOCK" if "<malicious>" in user_text.lower() else "PASS"
            return json.dumps({"decision": decision, "reason": "mock policy"})
        if "review" in model:
            return "<!doctype html><html><body>reviewed</body></html>"
        if "frontend" in model:
            return "<!doctype html><html><body>draft</body></html>"
        return "<!doctype html><html><body>base</body></html>"

