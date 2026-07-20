from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from anchor_mvp.research.neural_swarm_streaming import (
    BackendChunk,
    ExpertBinding,
    NeuralSwarmStreamController,
    NeuralSwarmStreamingError,
    StreamEventType,
    SwarmEvent,
    SwarmRequest,
    collect_swarm_events,
    summarize_swarm_events,
)


BUNDLE_SHA256 = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[1]


def _binding(name: str, backend: str | None = None) -> ExpertBinding:
    return ExpertBinding(
        request_model_id=f"anchor-swarm/{name}",
        expert_id=name.replace("-", "_"),
        backend_model_id=backend or f"{name}-backend",
    )


def _request(*names: str, shared_input: Mapping[str, Any] | None = None) -> SwarmRequest:
    return SwarmRequest(
        run_id="run-001",
        task_bundle_sha256=BUNDLE_SHA256,
        request_model_ids=tuple(f"anchor-swarm/{name}" for name in names),
        shared_input=shared_input or {"task": {"value": 7}, "items": [1, 2]},
    )


def _terminal_events(events: tuple[SwarmEvent, ...]) -> list[SwarmEvent]:
    terminal_types = {
        StreamEventType.COMPLETED,
        StreamEventType.FAILED,
        StreamEventType.CANCELLED,
    }
    return [event for event in events if event.event_type in terminal_types]


class InterleavingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[ExpertBinding, Mapping[str, Any], str, str]] = []
        self.planner_queued = asyncio.Event()
        self.reviewer_queued = asyncio.Event()
        self.planner_second_queued = asyncio.Event()

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        del cancel_event
        self.calls.append((binding, shared_input, run_id, task_bundle_sha256))
        if binding.expert_id == "planner":
            yield BackendChunk("p1", {"source": "planner"})
            self.planner_queued.set()
            await self.reviewer_queued.wait()
            yield "p2"
            self.planner_second_queued.set()
            return
        await self.planner_queued.wait()
        yield "r1"
        self.reviewer_queued.set()
        await self.planner_second_queued.wait()
        yield "r2"


def test_routes_shared_snapshot_interleaves_and_emits_ordered_summary() -> None:
    async def exercise() -> tuple[tuple[SwarmEvent, ...], InterleavingBackend]:
        backend = InterleavingBackend()
        controller = NeuralSwarmStreamController(
            bindings=(
                _binding("planner", "planner-r8"),
                _binding("frontend-review", "review-r12"),
            ),
            backend=backend,
            max_concurrency=2,
        )
        events = await collect_swarm_events(
            controller,
            _request("planner", "frontend-review"),
        )
        return events, backend

    events, backend = asyncio.run(exercise())

    routes = {
        call[0].request_model_id: call[0].backend_model_id for call in backend.calls
    }
    assert routes == {
        "anchor-swarm/planner": "planner-r8",
        "anchor-swarm/frontend-review": "review-r12",
    }
    assert len({id(call[1]) for call in backend.calls}) == 1
    assert all(call[1]["task"]["value"] == 7 for call in backend.calls)
    assert all(call[2] == "run-001" for call in backend.calls)
    assert all(call[3] == BUNDLE_SHA256 for call in backend.calls)
    json.dumps(backend.calls[0][1])
    with pytest.raises(TypeError):
        backend.calls[0][1]["new"] = "mutation"  # type: ignore[index]

    delta_events = [event for event in events if event.event_type is StreamEventType.DELTA]
    assert [event.delta for event in delta_events] == ["p1", "r1", "p2", "r2"]
    assert delta_events[0].stream_id == delta_events[2].stream_id
    assert delta_events[1].stream_id == delta_events[3].stream_id
    assert delta_events[0].stream_id != delta_events[1].stream_id

    assert [event.global_sequence for event in events] == list(range(len(events)))
    stream_ids = {
        event.stream_id
        for event in events
        if event.event_type
        not in {StreamEventType.BARRIER, StreamEventType.SWARM_COMPLETED}
    }
    for stream_id in stream_ids:
        sequences = [
            event.per_stream_sequence
            for event in events
            if event.stream_id == stream_id
        ]
        assert sequences == list(range(len(sequences)))

    terminals = _terminal_events(events)
    barrier_index = next(
        index for index, event in enumerate(events) if event.event_type is StreamEventType.BARRIER
    )
    assert barrier_index > max(events.index(event) for event in terminals)
    assert events[-1].event_type is StreamEventType.SWARM_COMPLETED
    assert events[barrier_index].metadata == {"terminal_streams": 2}

    summary = events[-1].metadata
    assert summary["total_streams"] == 2
    assert summary["completed_streams"] == 2
    assert summary["failed_streams"] == 0
    assert summary["cancelled_streams"] == 0
    assert summary["delta_events"] == 4
    assert summary["output_units"] == len("p1r1p2r2")
    rebuilt = summarize_swarm_events(events)
    assert rebuilt.total_streams == 2
    assert rebuilt.delta_events == 4
    assert rebuilt.output_units == len("p1r1p2r2")
    assert rebuilt.elapsed_ms == summary["elapsed_ms"]


class IsolatedFailureBackend:
    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        del shared_input, run_id, task_bundle_sha256, cancel_event
        if binding.expert_id == "planner":
            raise RuntimeError("planner failed")
        yield "review-ok"


def test_one_stream_failure_is_isolated_when_fail_fast_is_disabled() -> None:
    controller = NeuralSwarmStreamController(
        bindings=(_binding("planner"), _binding("frontend-review")),
        backend=IsolatedFailureBackend(),
        max_concurrency=2,
        fail_fast=False,
    )
    events = asyncio.run(
        collect_swarm_events(controller, _request("planner", "frontend-review"))
    )
    terminals = {event.expert_id: event for event in _terminal_events(events)}
    assert terminals["planner"].event_type is StreamEventType.FAILED
    assert terminals["planner"].error_type == "RuntimeError"
    assert terminals["frontend_review"].event_type is StreamEventType.COMPLETED
    assert events[-1].metadata["failed_streams"] == 1
    assert events[-1].metadata["completed_streams"] == 1


class FailFastBackend:
    def __init__(self) -> None:
        self.blocker_started = asyncio.Event()

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        del shared_input, run_id, task_bundle_sha256
        if binding.expert_id == "planner":
            await self.blocker_started.wait()
            raise LookupError("stop the swarm")
        self.blocker_started.set()
        await cancel_event.wait()
        if False:  # pragma: no cover - makes this an async generator
            yield "unreachable"


def test_fail_fast_cancels_unfinished_stream_before_barrier() -> None:
    backend = FailFastBackend()
    controller = NeuralSwarmStreamController(
        bindings=(_binding("planner"), _binding("frontend-review")),
        backend=backend,
        max_concurrency=2,
        fail_fast=True,
    )
    events = asyncio.run(
        collect_swarm_events(controller, _request("planner", "frontend-review"))
    )
    terminals = {event.expert_id: event for event in _terminal_events(events)}
    assert terminals["planner"].event_type is StreamEventType.FAILED
    assert terminals["frontend_review"].event_type is StreamEventType.CANCELLED
    barrier = next(event for event in events if event.event_type is StreamEventType.BARRIER)
    assert barrier.global_sequence > max(
        event.global_sequence for event in terminals.values()
    )
    assert events[-1].metadata["failed_streams"] == 1
    assert events[-1].metadata["cancelled_streams"] == 1


class ConcurrencyBackend:
    def __init__(self, expected_peak: int) -> None:
        self.expected_peak = expected_peak
        self.active = 0
        self.maximum_active = 0
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        del binding, shared_input, run_id, task_bundle_sha256, cancel_event
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.active == self.expected_peak:
            self.ready.set()
        try:
            await self.release.wait()
            yield "done"
        finally:
            self.active -= 1


def test_max_concurrency_bounds_active_backend_streams() -> None:
    async def exercise() -> tuple[tuple[SwarmEvent, ...], ConcurrencyBackend]:
        backend = ConcurrencyBackend(expected_peak=2)
        controller = NeuralSwarmStreamController(
            bindings=tuple(_binding(f"expert-{index}") for index in range(5)),
            backend=backend,
            max_concurrency=2,
        )
        task = asyncio.create_task(
            collect_swarm_events(
                controller,
                _request(*(f"expert-{index}" for index in range(5))),
            )
        )
        await asyncio.wait_for(backend.ready.wait(), timeout=1)
        assert backend.maximum_active == 2
        backend.release.set()
        return await asyncio.wait_for(task, timeout=1), backend

    events, backend = asyncio.run(exercise())
    assert backend.maximum_active == 2
    assert len(_terminal_events(events)) == 5
    assert events[-1].metadata["completed_streams"] == 5


class CloseTrackingIterator:
    def __init__(self, binding: ExpertBinding, entered: asyncio.Event) -> None:
        self.binding = binding
        self.entered = entered
        self.closed = False
        self.next_cancelled = False
        self.calls = 0
        self.second_entered = asyncio.Event()
        self.wait_forever = asyncio.Event()

    def __aiter__(self) -> CloseTrackingIterator:
        return self

    async def __anext__(self) -> BackendChunk:
        self.calls += 1
        self.entered.set()
        if self.calls == 1:
            return BackendChunk(f"{self.binding.expert_id}-first")
        self.second_entered.set()
        try:
            await self.wait_forever.wait()
        except asyncio.CancelledError:
            self.next_cancelled = True
            raise
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class CancellationBackend:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.iterators: list[CloseTrackingIterator] = []

    def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk]:
        del shared_input, run_id, task_bundle_sha256, cancel_event
        iterator = CloseTrackingIterator(binding, self.entered)
        self.iterators.append(iterator)
        return iterator


def test_explicit_cancellation_emits_terminals_and_closes_iterators() -> None:
    async def exercise() -> tuple[tuple[SwarmEvent, ...], CancellationBackend]:
        backend = CancellationBackend()
        controller = NeuralSwarmStreamController(
            bindings=(_binding("planner"), _binding("frontend-review")),
            backend=backend,
            max_concurrency=2,
        )
        cancel_event = asyncio.Event()
        events: list[SwarmEvent] = []
        async for event in controller.stream(
            _request("planner", "frontend-review"), cancel_event=cancel_event
        ):
            events.append(event)
            if event.event_type is StreamEventType.DELTA:
                cancel_event.set()
        return tuple(events), backend

    events, backend = asyncio.run(exercise())
    terminals = _terminal_events(events)
    assert len(terminals) == 2
    assert all(event.event_type is StreamEventType.CANCELLED for event in terminals)
    assert len(backend.iterators) == 2
    assert all(iterator.closed for iterator in backend.iterators)
    assert events[-2].event_type is StreamEventType.BARRIER
    assert events[-1].event_type is StreamEventType.SWARM_COMPLETED
    assert events[-1].metadata["cancelled_streams"] == 2


def test_early_consumer_aclose_cancels_child_anext_without_task_leak() -> None:
    async def exercise() -> tuple[CancellationBackend, int]:
        backend = CancellationBackend()
        controller = NeuralSwarmStreamController(
            bindings=(_binding("planner"),),
            backend=backend,
            cleanup_timeout_seconds=0.05,
        )
        stream = controller.stream(_request("planner"))
        first = await stream.__anext__()
        assert first.event_type is StreamEventType.STARTED
        while not backend.iterators:
            await asyncio.sleep(0)
        await asyncio.wait_for(backend.iterators[0].second_entered.wait(), timeout=1)
        await asyncio.wait_for(stream.aclose(), timeout=1)
        await asyncio.sleep(0)
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        return backend, len(pending)

    backend, pending_count = asyncio.run(exercise())
    assert len(backend.iterators) == 1
    assert backend.iterators[0].next_cancelled is True
    assert backend.iterators[0].closed is True
    assert pending_count == 0


class BackendCancelledErrorBackend:
    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        del binding, shared_input, run_id, task_bundle_sha256, cancel_event
        raise asyncio.CancelledError
        yield "unreachable"


def test_backend_cancelled_error_gets_one_terminal_and_does_not_deadlock() -> None:
    async def exercise() -> tuple[SwarmEvent, ...]:
        controller = NeuralSwarmStreamController(
            bindings=(_binding("planner"),),
            backend=BackendCancelledErrorBackend(),
        )
        return await asyncio.wait_for(
            collect_swarm_events(controller, _request("planner")), timeout=1
        )

    events = asyncio.run(exercise())
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0].event_type is StreamEventType.CANCELLED
    assert terminals[0].error_type == "CancelledError"
    assert events[-2].event_type is StreamEventType.BARRIER
    assert events[-1].event_type is StreamEventType.SWARM_COMPLETED
    assert events[-1].metadata["cancelled_streams"] == 1


class CloseFailureIterator:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.close_cancelled = False
        self.wait_forever = asyncio.Event()

    def __aiter__(self) -> CloseFailureIterator:
        return self

    async def __anext__(self) -> BackendChunk:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if self.mode == "cancel":
            raise asyncio.CancelledError
        if self.mode == "error":
            raise RuntimeError("close failed")
        try:
            await self.wait_forever.wait()
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise


class CloseFailureBackend:
    def __init__(self, *, mode: str) -> None:
        self.iterator = CloseFailureIterator(mode=mode)

    def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk]:
        del binding, shared_input, run_id, task_bundle_sha256, cancel_event
        return self.iterator


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("cancel", "CancelledError"),
        ("error", "RuntimeError"),
        ("hang", "IteratorCloseTimeout"),
    ],
)
def test_aclose_failure_is_bounded_and_finishes_with_barrier(
    mode: str, expected_error: str
) -> None:
    async def exercise() -> tuple[tuple[SwarmEvent, ...], CloseFailureBackend]:
        backend = CloseFailureBackend(mode=mode)
        controller = NeuralSwarmStreamController(
            bindings=(_binding("planner"),),
            backend=backend,
            cleanup_timeout_seconds=0.01,
        )
        events = await asyncio.wait_for(
            collect_swarm_events(controller, _request("planner")), timeout=1
        )
        return events, backend

    events, backend = asyncio.run(exercise())
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0].event_type is StreamEventType.FAILED
    assert terminals[0].error_type == expected_error
    assert events[-2].event_type is StreamEventType.BARRIER
    assert events[-1].event_type is StreamEventType.SWARM_COMPLETED
    if mode == "hang":
        assert backend.iterator.close_cancelled is True


class QueueOneFailFastBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        del shared_input, run_id, task_bundle_sha256, cancel_event
        self.calls.append(binding.expert_id)
        if binding.expert_id == "planner":
            raise RuntimeError("fail before queued experts enter the backend")
        yield "unexpected"


def test_queue_capacity_one_fail_fast_has_started_and_one_terminal_per_stream() -> None:
    async def exercise() -> tuple[tuple[SwarmEvent, ...], QueueOneFailFastBackend]:
        backend = QueueOneFailFastBackend()
        controller = NeuralSwarmStreamController(
            bindings=(
                _binding("planner"),
                _binding("frontend-review"),
                _binding("security-gate"),
            ),
            backend=backend,
            max_concurrency=1,
            fail_fast=True,
            queue_capacity=1,
        )
        events = await asyncio.wait_for(
            collect_swarm_events(
                controller,
                _request("planner", "frontend-review", "security-gate"),
            ),
            timeout=1,
        )
        return events, backend

    events, backend = asyncio.run(exercise())
    per_stream: dict[str, list[StreamEventType]] = {}
    for event in events:
        if event.event_type in {StreamEventType.BARRIER, StreamEventType.SWARM_COMPLETED}:
            continue
        per_stream.setdefault(event.stream_id, []).append(event.event_type)
    assert len(per_stream) == 3
    assert all(types[0] is StreamEventType.STARTED for types in per_stream.values())
    assert all(
        len([event for event in types if event in {
            StreamEventType.COMPLETED,
            StreamEventType.FAILED,
            StreamEventType.CANCELLED,
        }]) == 1
        for types in per_stream.values()
    )
    assert backend.calls == ["planner"]
    assert events[-1].metadata["failed_streams"] == 1
    assert events[-1].metadata["cancelled_streams"] == 2


def test_chunk_metadata_is_deep_frozen_and_json_safe() -> None:
    source = {"nested": {"values": [1, 2]}, "score": 0.5}
    chunk = BackendChunk("delta", source)
    source["nested"]["values"].append(3)
    assert chunk.metadata["nested"]["values"] == (1, 2)
    json.dumps(chunk.metadata)
    with pytest.raises(TypeError):
        chunk.metadata["new"] = "mutation"  # type: ignore[index]
    with pytest.raises(NeuralSwarmStreamingError, match="non-finite"):
        BackendChunk("delta", {"score": float("nan")})
    with pytest.raises(NeuralSwarmStreamingError, match="JSON-compatible"):
        BackendChunk("delta", {"raw": b"not-json"})


class CallCountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        del binding, shared_input, run_id, task_bundle_sha256, cancel_event
        self.calls += 1
        yield "unused"


def test_duplicate_and_unknown_ids_fail_closed_before_backend_work() -> None:
    with pytest.raises(NeuralSwarmStreamingError, match="duplicate request_model_id"):
        NeuralSwarmStreamController(
            bindings=(_binding("planner", "one"), _binding("planner", "two")),
            backend=CallCountingBackend(),
        )
    with pytest.raises(NeuralSwarmStreamingError, match="duplicate request_model_ids"):
        SwarmRequest(
            run_id="run-duplicate",
            task_bundle_sha256=BUNDLE_SHA256,
            request_model_ids=("anchor-swarm/planner", "anchor-swarm/planner"),
            shared_input={"task": "duplicate"},
        )

    backend = CallCountingBackend()
    controller = NeuralSwarmStreamController(
        bindings=(_binding("planner"),),
        backend=backend,
    )
    with pytest.raises(NeuralSwarmStreamingError, match="unknown request_model_id"):
        asyncio.run(
            collect_swarm_events(controller, _request("planner", "unknown"))
        )
    assert backend.calls == 0


def test_emitted_event_dicts_match_the_declared_schema_shape() -> None:
    controller = NeuralSwarmStreamController(
        bindings=(_binding("planner"),),
        backend=CallCountingBackend(),
    )
    events = asyncio.run(collect_swarm_events(controller, _request("planner")))
    schema = json.loads(
        (REPO_ROOT / "configs/research/neural_swarm_stream_event.schema.json").read_text(
            encoding="utf-8"
        )
    )
    required = set(schema["required"])
    properties = set(schema["properties"])
    event_types = set(schema["properties"]["event_type"]["enum"])
    for event in events:
        payload = event.to_dict()
        assert required <= payload.keys()
        assert payload.keys() == properties
        assert payload["schema_version"] == schema["properties"]["schema_version"][
            "const"
        ]
        assert payload["event_type"] in event_types
