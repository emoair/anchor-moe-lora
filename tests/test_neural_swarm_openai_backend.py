from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import pytest

from anchor_mvp.research.neural_swarm_openai_backend import (
    OpenAICompatibleSSEBackend,
    OpenAIStreamingBackendConfig,
    OpenAIStreamingBackendError,
    OpenAIStreamingHTTPError,
    OpenAIStreamingProtocolError,
)
from anchor_mvp.research.neural_swarm_streaming import (
    BackendChunk,
    ExpertBinding,
    NeuralSwarmStreamController,
    StreamEventType,
    SwarmRequest,
    collect_swarm_events,
)


def _binding(backend_model_id: str) -> ExpertBinding:
    return ExpertBinding(
        request_model_id=f"anchor-swarm/{backend_model_id}",
        expert_id="planner",
        backend_model_id=backend_model_id,
    )


async def _collect(
    backend: OpenAICompatibleSSEBackend,
    *,
    binding: ExpertBinding,
    shared_input: Mapping[str, Any] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> list[BackendChunk]:
    return [
        chunk
        async for chunk in backend.stream(
            binding=binding,
            shared_input=shared_input
            or {
                "messages": ({"role": "user", "content": "small fixture"},),
                "generation": {"temperature": 0.2},
            },
            run_id="run-001",
            task_bundle_sha256="a" * 64,
            cancel_event=cancel_event or asyncio.Event(),
        )
        if isinstance(chunk, BackendChunk)
    ]


def _sse_response(*events: str, status_code: int = 200) -> httpx.Response:
    body = "".join(f"data: {event}\n\n" for event in events).encode()
    return httpx.Response(
        status_code,
        headers={"content-type": "text/event-stream"},
        content=body,
    )


def test_routes_each_binding_model_and_builds_strict_stream_request() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = {
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
            ]
        }
        return _sse_response(json.dumps(payload), "[DONE]")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig(
            base_url="http://127.0.0.1:8080/v1/",
            api_key="synthetic-secret",
            timeout=5,
        ),
        transport=httpx.MockTransport(handler),
    )

    async def exercise() -> None:
        for model_id in ("planner-r8", "review-r12"):
            chunks = await _collect(backend, binding=_binding(model_id))
            assert [chunk.delta for chunk in chunks] == ["ok"]

    asyncio.run(exercise())
    assert len(requests) == 2
    bodies = [json.loads(request.content) for request in requests]
    assert [body["model"] for body in bodies] == ["planner-r8", "review-r12"]
    assert all(body["stream"] is True for body in bodies)
    assert all(body["temperature"] == 0.2 for body in bodies)
    assert all(body["messages"][0]["role"] == "user" for body in bodies)
    assert all(request.url.path == "/v1/chat/completions" for request in requests)
    assert all(
        request.headers["authorization"] == "Bearer synthetic-secret"
        for request in requests
    )


def test_controller_fans_one_task_bundle_to_distinct_backend_model_ids() -> None:
    requested_models: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        payload = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"{body['model']}-ok"},
                    "finish_reason": None,
                }
            ]
        }
        return _sse_response(json.dumps(payload), "[DONE]")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://127.0.0.1:8080/v1"),
        transport=httpx.MockTransport(handler),
    )
    bindings = (_binding("planner-r8"), _binding("review-r12"))
    controller = NeuralSwarmStreamController(
        bindings=bindings,
        backend=backend,
        max_concurrency=2,
        queue_capacity=1,
    )
    request = SwarmRequest(
        run_id="run-openai-mock",
        task_bundle_sha256="a" * 64,
        request_model_ids=tuple(binding.request_model_id for binding in bindings),
        shared_input={
            "messages": [{"role": "user", "content": "content-free fixture"}],
            "generation": {"max_tokens": 8},
        },
    )

    events = asyncio.run(collect_swarm_events(controller, request))
    deltas = [
        event.delta for event in events if event.event_type is StreamEventType.DELTA
    ]
    assert sorted(requested_models) == ["planner-r8", "review-r12"]
    assert sorted(deltas) == ["planner-r8-ok", "review-r12-ok"]
    assert events[-2].event_type is StreamEventType.BARRIER
    assert events[-1].event_type is StreamEventType.SWARM_COMPLETED
    assert events[-1].metadata["completed_streams"] == 2


def test_fragmented_sse_yields_text_finish_and_usage_metadata() -> None:
    first = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}]}
    )
    second = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]}
    )
    finish = json.dumps(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    usage = json.dumps(
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    )
    wire = (
        f"data: {first}\n\ndata: {second}\n\ndata: {finish}\n\n"
        f"data: {usage}\n\ndata: [DONE]\n\n"
    ).encode()

    class FragmentedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for index in range(0, len(wire), 7):
                yield wire[index : index + 7]

        async def aclose(self) -> None:
            return None

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=FragmentedStream(),
        )

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
        transport=httpx.MockTransport(handler),
    )
    chunks = asyncio.run(_collect(backend, binding=_binding("planner-r8")))
    assert [chunk.delta for chunk in chunks] == ["hel", "lo", "", ""]
    assert chunks[2].metadata == {"finish_reason": "stop", "choice_index": 0}
    assert chunks[3].metadata["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
    }


def test_api_key_env_is_resolved_at_request_time_and_not_represented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_authorization: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers["authorization"])
        return _sse_response("{\"choices\": []}", "[DONE]")

    config = OpenAIStreamingBackendConfig(
        "http://localhost:8080/v1", api_key_env="SWARM_TEST_API_KEY"
    )
    assert "launch-only-value" not in repr(config)
    backend = OpenAICompatibleSSEBackend(config, transport=httpx.MockTransport(handler))
    monkeypatch.setenv("SWARM_TEST_API_KEY", "launch-only-value")
    asyncio.run(_collect(backend, binding=_binding("planner-r8")))
    assert seen_authorization == ["Bearer launch-only-value"]


@pytest.mark.parametrize(
    "shared_input,match",
    [
        ({"task_board": {}}, "messages is required|unsupported field"),
        ({"messages": []}, "non-empty sequence"),
        (
            {"messages": [{"role": "user", "content": "x"}], "extra": True},
            "unsupported field",
        ),
        (
            {
                "messages": [{"role": "user", "content": "x"}],
                "generation": {"model": "override"},
            },
            "cannot override",
        ),
        (
            {
                "messages": [{"role": "user", "content": "x"}],
                "generation": {"n": 2},
            },
            "must be 1",
        ),
        (
            {
                "messages": [{"role": "user", "content": "x"}],
                "generation": {"tools": [{"type": "function"}]},
            },
            "does not support: tools",
        ),
    ],
)
def test_rejects_ambiguous_shared_input_before_network(
    shared_input: Mapping[str, Any], match: str
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        return _sse_response("[DONE]")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenAIStreamingBackendError, match=match):
        asyncio.run(
            _collect(
                backend,
                binding=_binding("planner-r8"),
                shared_input=shared_input,
            )
        )
    assert calls == 0


def test_http_and_protocol_errors_are_explicit_without_echoing_body() -> None:
    async def http_error(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, content=b"do-not-echo-this-body")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
        transport=httpx.MockTransport(http_error),
    )
    with pytest.raises(OpenAIStreamingHTTPError, match="HTTP 503") as captured:
        asyncio.run(_collect(backend, binding=_binding("planner-r8")))
    assert "do-not-echo-this-body" not in str(captured.value)

    async def invalid_json(request: httpx.Request) -> httpx.Response:
        del request
        return _sse_response("not-json", "[DONE]")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
        transport=httpx.MockTransport(invalid_json),
    )
    with pytest.raises(OpenAIStreamingProtocolError, match="invalid JSON"):
        asyncio.run(_collect(backend, binding=_binding("planner-r8")))

    async def missing_done(request: httpx.Request) -> httpx.Response:
        del request
        return _sse_response("{\"choices\": []}")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
        transport=httpx.MockTransport(missing_done),
    )
    with pytest.raises(OpenAIStreamingProtocolError, match=r"before the \[DONE\]"):
        asyncio.run(_collect(backend, binding=_binding("planner-r8")))

    async def tool_delta(request: httpx.Request) -> httpx.Response:
        del request
        payload = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0}]},
                    "finish_reason": None,
                }
            ]
        }
        return _sse_response(json.dumps(payload), "[DONE]")

    backend = OpenAICompatibleSSEBackend(
        OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
        transport=httpx.MockTransport(tool_delta),
    )
    with pytest.raises(OpenAIStreamingProtocolError, match="unsupported delta"):
        asyncio.run(_collect(backend, binding=_binding("planner-r8")))


def test_cancel_event_stops_blocked_read_and_closes_response_and_transport() -> None:
    first = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}]}
    )

    class BlockingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False
            self.block_forever = asyncio.Event()

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield f"data: {first}\n\n".encode()
            await self.block_forever.wait()

        async def aclose(self) -> None:
            self.closed = True

    class TrackingTransport(httpx.AsyncBaseTransport):
        def __init__(self, response_stream: BlockingStream) -> None:
            self.response_stream = response_stream
            self.closed = False

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=self.response_stream,
            )

        async def aclose(self) -> None:
            self.closed = True

    async def exercise() -> tuple[BlockingStream, TrackingTransport]:
        response_stream = BlockingStream()
        transport = TrackingTransport(response_stream)
        backend = OpenAICompatibleSSEBackend(
            OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
            transport=transport,
        )
        cancel_event = asyncio.Event()
        iterator = backend.stream(
            binding=_binding("planner-r8"),
            shared_input={"messages": [{"role": "user", "content": "x"}]},
            run_id="run-001",
            task_bundle_sha256="a" * 64,
            cancel_event=cancel_event,
        ).__aiter__()
        assert (await iterator.__anext__()).delta == "first"
        cancel_event.set()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(iterator.__anext__(), timeout=1)
        return response_stream, transport

    response_stream, transport = asyncio.run(exercise())
    assert response_stream.closed is True
    assert transport.closed is True


def test_direct_anext_cancellation_cleans_internal_tasks_and_closes_transport() -> None:
    started = asyncio.Event()

    class BlockingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False
            self.block_forever = asyncio.Event()

        async def __aiter__(self) -> AsyncIterator[bytes]:
            started.set()
            await self.block_forever.wait()
            yield b""

        async def aclose(self) -> None:
            self.closed = True

    class TrackingTransport(httpx.AsyncBaseTransport):
        def __init__(self, response_stream: BlockingStream) -> None:
            self.response_stream = response_stream
            self.closed = False

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=self.response_stream,
            )

        async def aclose(self) -> None:
            self.closed = True

    async def exercise() -> tuple[BlockingStream, TrackingTransport, set[str]]:
        response_stream = BlockingStream()
        transport = TrackingTransport(response_stream)
        backend = OpenAICompatibleSSEBackend(
            OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
            transport=transport,
        )
        iterator = backend.stream(
            binding=_binding("planner-r8"),
            shared_input={"messages": [{"role": "user", "content": "x"}]},
            run_id="run-001",
            task_bundle_sha256="a" * 64,
            cancel_event=asyncio.Event(),
        ).__aiter__()
        pending = asyncio.create_task(iterator.__anext__())
        await asyncio.wait_for(started.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.sleep(0)
        leaked_names = {
            task.get_name()
            for task in asyncio.all_tasks()
            if not task.done() and task is not asyncio.current_task()
        }.intersection(
            {"anchor-openai-sse-next-line", "anchor-openai-sse-cancel-watch"}
        )
        return response_stream, transport, leaked_names

    response_stream, transport, leaked_names = asyncio.run(exercise())
    assert leaked_names == set()
    assert response_stream.closed is True
    assert transport.closed is True


def test_early_aclose_after_first_chunk_closes_response_and_transport() -> None:
    first = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}]}
    )

    class TwoPartStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield f"data: {first}\n\n".encode()
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            self.closed = True

    class TrackingTransport(httpx.AsyncBaseTransport):
        def __init__(self, response_stream: TwoPartStream) -> None:
            self.response_stream = response_stream
            self.closed = False

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=self.response_stream,
            )

        async def aclose(self) -> None:
            self.closed = True

    async def exercise() -> tuple[TwoPartStream, TrackingTransport, set[str]]:
        response_stream = TwoPartStream()
        transport = TrackingTransport(response_stream)
        backend = OpenAICompatibleSSEBackend(
            OpenAIStreamingBackendConfig("http://localhost:8080/v1"),
            transport=transport,
        )
        iterator = backend.stream(
            binding=_binding("planner-r8"),
            shared_input={"messages": [{"role": "user", "content": "x"}]},
            run_id="run-001",
            task_bundle_sha256="a" * 64,
            cancel_event=asyncio.Event(),
        ).__aiter__()
        assert (await iterator.__anext__()).delta == "first"
        await iterator.aclose()
        await asyncio.sleep(0)
        leaked_names = {
            task.get_name()
            for task in asyncio.all_tasks()
            if not task.done() and task is not asyncio.current_task()
        }.intersection(
            {"anchor-openai-sse-next-line", "anchor-openai-sse-cancel-watch"}
        )
        return response_stream, transport, leaked_names

    response_stream, transport, leaked_names = asyncio.run(exercise())
    assert leaked_names == set()
    assert response_stream.closed is True
    assert transport.closed is True


def test_non_cooperative_stream_cleanup_is_bounded_and_reported() -> None:
    async def exercise() -> tuple[float, bool, set[str]]:
        started = asyncio.Event()
        release = asyncio.Event()

        class StubbornStream(httpx.AsyncByteStream):
            def __init__(self) -> None:
                self.closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # Deliberately violate cooperative cancellation to verify
                    # the adapter's cleanup wait remains bounded.
                    await release.wait()
                yield b""

            async def aclose(self) -> None:
                self.closed = True
                release.set()

        response_stream = StubbornStream()

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=response_stream,
            )

        backend = OpenAICompatibleSSEBackend(
            OpenAIStreamingBackendConfig(
                "http://localhost:8080/v1", cleanup_timeout=0.01
            ),
            transport=httpx.MockTransport(handler),
        )
        cancel_event = asyncio.Event()
        task = asyncio.create_task(
            _collect(
                backend,
                binding=_binding("planner-r8"),
                cancel_event=cancel_event,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        loop = asyncio.get_running_loop()
        before = loop.time()
        cancel_event.set()
        with pytest.raises(OpenAIStreamingHTTPError, match="cleanup timed out"):
            await asyncio.wait_for(task, timeout=0.5)
        elapsed = loop.time() - before
        await asyncio.sleep(0)
        leaked_names = {
            pending.get_name()
            for pending in asyncio.all_tasks()
            if not pending.done() and pending is not asyncio.current_task()
        }.intersection(
            {"anchor-openai-sse-next-line", "anchor-openai-sse-cancel-watch"}
        )
        return elapsed, response_stream.closed, leaked_names

    elapsed, response_closed, leaked_names = asyncio.run(exercise())
    assert elapsed < 0.5
    assert response_closed is True
    assert leaked_names == set()


def test_config_validation_and_missing_environment_credential() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        OpenAIStreamingBackendConfig("not-a-url")
    with pytest.raises(ValueError, match="not both"):
        OpenAIStreamingBackendConfig(
            "http://localhost:8080/v1", api_key="x", api_key_env="KEY"
        )
    with pytest.raises(ValueError, match="positive finite"):
        OpenAIStreamingBackendConfig("http://localhost:8080/v1", timeout=0)
    with pytest.raises(ValueError, match="cleanup_timeout"):
        OpenAIStreamingBackendConfig(
            "http://localhost:8080/v1", cleanup_timeout=0
        )

    config = OpenAIStreamingBackendConfig(
        "http://localhost:8080/v1", api_key_env="DEFINITELY_MISSING_SWARM_KEY"
    )
    with pytest.raises(OpenAIStreamingBackendError, match="is not set"):
        config.resolve_api_key()
