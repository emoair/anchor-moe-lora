"""Transport-neutral multi-expert streaming for Neural Swarm research.

The controller in this module deliberately stops at the execution boundary. It
maps stable, client-facing model ids to expert/backend ids, snapshots one shared
task input, fans that snapshot out to independent async streams, and exposes a
single multiplexed event iterator. It does not claim CUDA overlap, exact shared
KV reuse, or a quality improvement.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


STREAM_EVENT_SCHEMA_VERSION = "anchor.neural-swarm-stream-event.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _FrozenDict(dict[str, Any]):
    """A JSON-serializable dict whose public mutation methods are disabled."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("frozen JSON mappings cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class NeuralSwarmStreamingError(ValueError):
    """Raised when a run cannot be resolved without ambiguity."""


class StreamEventType(str, Enum):
    STARTED = "started"
    DELTA = "delta"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BARRIER = "barrier"
    SWARM_COMPLETED = "swarm_completed"


@dataclass(frozen=True)
class ExpertBinding:
    """Resolve one external model id to one logical expert/backend route."""

    request_model_id: str
    expert_id: str
    backend_model_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("request_model_id", self.request_model_id),
            ("expert_id", self.expert_id),
            ("backend_model_id", self.backend_model_id),
        ):
            if not value.strip():
                raise NeuralSwarmStreamingError(f"{name} must be non-empty")


@dataclass(frozen=True)
class BackendChunk:
    """One backend-produced output fragment and optional measurements."""

    delta: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str):
            raise NeuralSwarmStreamingError("backend chunk delta must be a string")
        if not isinstance(self.metadata, Mapping):
            raise NeuralSwarmStreamingError("backend chunk metadata must be a mapping")
        frozen = _freeze_json(self.metadata, path="backend_chunk.metadata")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True)
class SwarmRequest:
    """One task bundle dispatched to one or more logical model ids."""

    run_id: str
    task_bundle_sha256: str
    request_model_ids: tuple[str, ...]
    shared_input: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise NeuralSwarmStreamingError("run_id must be non-empty")
        if _SHA256_RE.fullmatch(self.task_bundle_sha256) is None:
            raise NeuralSwarmStreamingError(
                "task_bundle_sha256 must be 64 lowercase hexadecimal characters"
            )
        if not self.request_model_ids:
            raise NeuralSwarmStreamingError("request_model_ids must not be empty")
        if len(set(self.request_model_ids)) != len(self.request_model_ids):
            raise NeuralSwarmStreamingError("duplicate request_model_ids are not allowed")
        if any(not value.strip() for value in self.request_model_ids):
            raise NeuralSwarmStreamingError("request_model_ids must be non-empty")
        if not isinstance(self.shared_input, Mapping):
            raise NeuralSwarmStreamingError("shared_input must be a mapping")


@dataclass(frozen=True)
class SwarmEvent:
    """One event from the multiplexed, globally ordered output stream."""

    run_id: str
    task_bundle_sha256: str
    stream_id: str
    expert_id: str
    request_model_id: str
    backend_model_id: str
    per_stream_sequence: int
    global_sequence: int
    event_type: StreamEventType
    elapsed_ms: float
    delta: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = STREAM_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise NeuralSwarmStreamingError("event metadata must be a mapping")
        frozen = _freeze_json(self.metadata, path="event.metadata")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "metadata", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_bundle_sha256": self.task_bundle_sha256,
            "stream_id": self.stream_id,
            "expert_id": self.expert_id,
            "request_model_id": self.request_model_id,
            "backend_model_id": self.backend_model_id,
            "per_stream_sequence": self.per_stream_sequence,
            "global_sequence": self.global_sequence,
            "event_type": self.event_type.value,
            "elapsed_ms": self.elapsed_ms,
            "delta": self.delta,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": _thaw_json(self.metadata),
        }


@runtime_checkable
class StreamingExpertBackend(Protocol):
    """Backend contract used by the transport-neutral multiplexer."""

    def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]: ...


@dataclass(frozen=True)
class ExpertStreamSummary:
    stream_id: str
    expert_id: str
    request_model_id: str
    backend_model_id: str
    terminal_event: StreamEventType
    delta_events: int
    output_units: int
    time_to_first_delta_ms: float | None
    elapsed_ms: float


@dataclass(frozen=True)
class SwarmRunSummary:
    run_id: str
    task_bundle_sha256: str
    total_streams: int
    completed_streams: int
    failed_streams: int
    cancelled_streams: int
    delta_events: int
    output_units: int
    elapsed_ms: float
    streams: tuple[ExpertStreamSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_bundle_sha256": self.task_bundle_sha256,
            "total_streams": self.total_streams,
            "completed_streams": self.completed_streams,
            "failed_streams": self.failed_streams,
            "cancelled_streams": self.cancelled_streams,
            "delta_events": self.delta_events,
            "output_units": self.output_units,
            "elapsed_ms": self.elapsed_ms,
            "streams": [
                {
                    "stream_id": stream.stream_id,
                    "expert_id": stream.expert_id,
                    "request_model_id": stream.request_model_id,
                    "backend_model_id": stream.backend_model_id,
                    "terminal_event": stream.terminal_event.value,
                    "delta_events": stream.delta_events,
                    "output_units": stream.output_units,
                    "time_to_first_delta_ms": stream.time_to_first_delta_ms,
                    "elapsed_ms": stream.elapsed_ms,
                }
                for stream in self.streams
            ],
        }


@dataclass(frozen=True)
class _LocalEvent:
    binding: ExpertBinding
    stream_id: str
    per_stream_sequence: int
    event_type: StreamEventType
    elapsed_ms: float
    delta: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _freeze_json(value: Any, *, path: str = "shared_input") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NeuralSwarmStreamingError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NeuralSwarmStreamingError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return _FrozenDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise NeuralSwarmStreamingError(
        f"{path} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    """Return ordinary dict/list containers for schema validators and transports."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _stream_id(run_id: str, request_model_id: str) -> str:
    return f"{run_id}:{request_model_id}"


async def _next_chunk_or_cancel(
    iterator: AsyncIterator[BackendChunk | str],
    cancel_event: asyncio.Event,
    *,
    cleanup_timeout_seconds: float,
) -> BackendChunk | str | None:
    next_task = asyncio.create_task(iterator.__anext__())
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _ = await asyncio.wait(
            {next_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and cancel_event.is_set():
            return None
        return await next_task
    finally:
        cleanup_complete = await _cancel_and_drain_tasks(
            (next_task, cancel_task), timeout_seconds=cleanup_timeout_seconds
        )
        current = asyncio.current_task()
        controller_is_cancelling = current is not None and current.cancelling() > 0
        if not cleanup_complete and not controller_is_cancelling:
            raise TimeoutError("backend __anext__ did not stop after cancellation")


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


async def _cancel_and_drain_tasks(
    tasks: Sequence[asyncio.Task[Any]], *, timeout_seconds: float
) -> bool:
    """Cancel child tasks and wait only up to the configured cleanup bound."""

    for task in tasks:
        task.add_done_callback(_consume_task_result)
        if not task.done():
            task.cancel()
    done, pending = await asyncio.wait(set(tasks), timeout=timeout_seconds)
    for task in done:
        _consume_task_result(task)
    return not pending


@dataclass(frozen=True)
class _CleanupIssue:
    error_type: str
    error_message: str

    def to_metadata(self) -> Mapping[str, str]:
        return _FrozenDict(
            {
                "cleanup_error_type": self.error_type,
                "cleanup_error_message": self.error_message,
            }
        )


async def _close_iterator_bounded(
    iterator: AsyncIterator[BackendChunk | str] | None,
    *,
    timeout_seconds: float,
) -> _CleanupIssue | None:
    if iterator is None:
        return None
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return None
    try:
        close_result = aclose()
    except asyncio.CancelledError:
        return _CleanupIssue("CancelledError", "backend iterator cleanup was cancelled")
    except Exception as exc:
        return _CleanupIssue(type(exc).__name__, str(exc))
    if not inspect.isawaitable(close_result):
        return _CleanupIssue("TypeError", "backend iterator aclose() must be awaitable")

    close_task = asyncio.ensure_future(close_result)
    close_task.add_done_callback(_consume_task_result)
    try:
        done, _ = await asyncio.wait({close_task}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        close_task.cancel()
        await _cancel_and_drain_tasks(
            (close_task,), timeout_seconds=timeout_seconds
        )
        raise
    if close_task not in done:
        close_task.cancel()
        await _cancel_and_drain_tasks(
            (close_task,), timeout_seconds=timeout_seconds
        )
        return _CleanupIssue(
            "IteratorCloseTimeout",
            f"backend iterator cleanup exceeded {timeout_seconds:g} seconds",
        )
    try:
        close_task.result()
    except asyncio.CancelledError:
        return _CleanupIssue("CancelledError", "backend iterator cleanup was cancelled")
    except Exception as exc:
        return _CleanupIssue(type(exc).__name__, str(exc))
    return None


@dataclass
class _StreamAccumulator:
    stream_id: str
    expert_id: str
    request_model_id: str
    backend_model_id: str
    next_sequence: int = 0
    started: bool = False
    terminal_event: StreamEventType | None = None
    delta_events: int = 0
    output_units: int = 0
    time_to_first_delta_ms: float | None = None
    terminal_elapsed_ms: float | None = None

    def observe(self, event: SwarmEvent) -> None:
        identity = (
            event.stream_id,
            event.expert_id,
            event.request_model_id,
            event.backend_model_id,
        )
        expected_identity = (
            self.stream_id,
            self.expert_id,
            self.request_model_id,
            self.backend_model_id,
        )
        if identity != expected_identity:
            raise NeuralSwarmStreamingError(
                f"stream identity changed for {self.stream_id}"
            )
        if self.terminal_event is not None:
            raise NeuralSwarmStreamingError(
                f"stream {self.stream_id} emitted an event after its terminal"
            )
        if event.per_stream_sequence != self.next_sequence:
            raise NeuralSwarmStreamingError(
                f"per-stream sequence is not contiguous for {self.stream_id}"
            )
        self.next_sequence += 1

        if event.event_type is StreamEventType.STARTED:
            if self.started or event.per_stream_sequence != 0:
                raise NeuralSwarmStreamingError(
                    f"stream {self.stream_id} must emit started exactly once and first"
                )
            self.started = True
            return
        if not self.started:
            raise NeuralSwarmStreamingError(
                f"stream {self.stream_id} emitted {event.event_type.value} before started"
            )
        if event.event_type is StreamEventType.DELTA:
            self.delta_events += 1
            self.output_units += len(event.delta or "")
            if self.time_to_first_delta_ms is None:
                self.time_to_first_delta_ms = event.elapsed_ms
            return
        if event.event_type not in {
            StreamEventType.COMPLETED,
            StreamEventType.FAILED,
            StreamEventType.CANCELLED,
        }:
            raise NeuralSwarmStreamingError(
                f"unexpected per-stream event type: {event.event_type.value}"
            )
        self.terminal_event = event.event_type
        self.terminal_elapsed_ms = event.elapsed_ms

    def to_summary(self) -> ExpertStreamSummary:
        if self.terminal_event is None or self.terminal_elapsed_ms is None:
            raise NeuralSwarmStreamingError(
                f"stream {self.stream_id} does not have exactly one terminal event"
            )
        return ExpertStreamSummary(
            stream_id=self.stream_id,
            expert_id=self.expert_id,
            request_model_id=self.request_model_id,
            backend_model_id=self.backend_model_id,
            terminal_event=self.terminal_event,
            delta_events=self.delta_events,
            output_units=self.output_units,
            time_to_first_delta_ms=self.time_to_first_delta_ms,
            elapsed_ms=self.terminal_elapsed_ms,
        )


def _build_incremental_summary(
    *,
    run_id: str,
    task_bundle_sha256: str,
    accumulators: Mapping[str, _StreamAccumulator],
) -> SwarmRunSummary:
    stream_summaries = tuple(
        accumulators[stream_id].to_summary() for stream_id in sorted(accumulators)
    )
    counts = Counter(stream.terminal_event for stream in stream_summaries)
    return SwarmRunSummary(
        run_id=run_id,
        task_bundle_sha256=task_bundle_sha256,
        total_streams=len(stream_summaries),
        completed_streams=counts[StreamEventType.COMPLETED],
        failed_streams=counts[StreamEventType.FAILED],
        cancelled_streams=counts[StreamEventType.CANCELLED],
        delta_events=sum(stream.delta_events for stream in stream_summaries),
        output_units=sum(stream.output_units for stream in stream_summaries),
        elapsed_ms=max(stream.elapsed_ms for stream in stream_summaries),
        streams=stream_summaries,
    )


def summarize_swarm_events(events: Sequence[SwarmEvent]) -> SwarmRunSummary:
    """Build a content-free run summary and validate terminal completeness."""

    if not events:
        raise NeuralSwarmStreamingError("cannot summarize an empty event sequence")
    run_ids = {event.run_id for event in events}
    bundle_ids = {event.task_bundle_sha256 for event in events}
    if len(run_ids) != 1 or len(bundle_ids) != 1:
        raise NeuralSwarmStreamingError("events must belong to one run and task bundle")
    global_sequences = [event.global_sequence for event in events]
    if global_sequences != list(range(len(events))):
        raise NeuralSwarmStreamingError("global event sequence must be contiguous")

    accumulators: dict[str, _StreamAccumulator] = {}
    controller_events: list[SwarmEvent] = []
    for event in events:
        if event.event_type in {StreamEventType.BARRIER, StreamEventType.SWARM_COMPLETED}:
            controller_events.append(event)
            continue
        accumulator = accumulators.get(event.stream_id)
        if accumulator is None:
            accumulator = _StreamAccumulator(
                stream_id=event.stream_id,
                expert_id=event.expert_id,
                request_model_id=event.request_model_id,
                backend_model_id=event.backend_model_id,
            )
            accumulators[event.stream_id] = accumulator
        accumulator.observe(event)
    if not accumulators:
        raise NeuralSwarmStreamingError("event sequence contains no expert streams")
    if controller_events:
        if (
            len(controller_events) != 2
            or controller_events != list(events[-2:])
            or [event.event_type for event in controller_events]
            != [StreamEventType.BARRIER, StreamEventType.SWARM_COMPLETED]
            or [event.per_stream_sequence for event in controller_events] != [0, 1]
        ):
            raise NeuralSwarmStreamingError(
                "controller events must be one trailing barrier/swarm_completed pair"
            )
    return _build_incremental_summary(
        run_id=events[0].run_id,
        task_bundle_sha256=events[0].task_bundle_sha256,
        accumulators=accumulators,
    )


class NeuralSwarmStreamController:
    """Resolve, fan out, and multiplex independent expert output streams."""

    def __init__(
        self,
        *,
        bindings: Sequence[ExpertBinding],
        backend: StreamingExpertBackend,
        max_concurrency: int | None = None,
        fail_fast: bool = False,
        queue_capacity: int = 256,
        cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        if not bindings:
            raise NeuralSwarmStreamingError("at least one expert binding is required")
        if max_concurrency is not None and max_concurrency < 1:
            raise NeuralSwarmStreamingError("max_concurrency must be positive")
        if queue_capacity < 1:
            raise NeuralSwarmStreamingError("queue_capacity must be positive")
        if (
            isinstance(cleanup_timeout_seconds, bool)
            or not isinstance(cleanup_timeout_seconds, (int, float))
            or not math.isfinite(cleanup_timeout_seconds)
            or cleanup_timeout_seconds <= 0
        ):
            raise NeuralSwarmStreamingError(
                "cleanup_timeout_seconds must be a finite positive number"
            )
        index: dict[str, ExpertBinding] = {}
        for binding in bindings:
            if binding.request_model_id in index:
                raise NeuralSwarmStreamingError(
                    f"duplicate request_model_id binding: {binding.request_model_id}"
                )
            index[binding.request_model_id] = binding
        self._bindings = _FrozenDict(index)
        self._backend = backend
        self._max_concurrency = max_concurrency or len(bindings)
        self._fail_fast = fail_fast
        self._queue_capacity = queue_capacity
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)

    @property
    def bindings(self) -> Mapping[str, ExpertBinding]:
        return self._bindings

    def resolve(self, request_model_ids: Sequence[str]) -> tuple[ExpertBinding, ...]:
        if not request_model_ids:
            raise NeuralSwarmStreamingError("request_model_ids must not be empty")
        if len(set(request_model_ids)) != len(request_model_ids):
            raise NeuralSwarmStreamingError("duplicate request_model_ids are not allowed")
        unknown = sorted(set(request_model_ids).difference(self._bindings))
        if unknown:
            raise NeuralSwarmStreamingError(
                f"unknown request_model_id(s): {', '.join(unknown)}"
            )
        return tuple(self._bindings[model_id] for model_id in request_model_ids)

    async def stream(
        self,
        request: SwarmRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[SwarmEvent]:
        """Yield one interleaved event stream for all resolved experts."""

        resolved = self.resolve(request.request_model_ids)
        shared_snapshot = _freeze_json(request.shared_input)
        assert isinstance(shared_snapshot, Mapping)
        queue: asyncio.Queue[_LocalEvent] = asyncio.Queue(self._queue_capacity)
        semaphore = asyncio.Semaphore(min(self._max_concurrency, len(resolved)))
        abort_event = asyncio.Event()
        shutdown_event = asyncio.Event()
        started_at = time.perf_counter()

        async def mirror_external_cancel() -> None:
            assert cancel_event is not None
            await cancel_event.wait()
            abort_event.set()

        async def run_expert(binding: ExpertBinding) -> None:
            stream_id = _stream_id(request.run_id, binding.request_model_id)
            sequence = 0
            iterator: AsyncIterator[BackendChunk | str] | None = None
            terminal_event = StreamEventType.COMPLETED
            terminal_error_type: str | None = None
            terminal_error_message: str | None = None
            terminal_metadata: dict[str, Any] = {}
            controller_cancellation: asyncio.CancelledError | None = None
            try:
                await queue.put(
                    _LocalEvent(
                        binding=binding,
                        stream_id=stream_id,
                        per_stream_sequence=sequence,
                        event_type=StreamEventType.STARTED,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000,
                    )
                )
                sequence += 1
                async with semaphore:
                    if abort_event.is_set():
                        terminal_event = StreamEventType.CANCELLED
                        terminal_metadata["reason"] = "cancelled_before_backend_start"
                    else:
                        iterator = self._backend.stream(
                            binding=binding,
                            shared_input=shared_snapshot,
                            run_id=request.run_id,
                            task_bundle_sha256=request.task_bundle_sha256,
                            cancel_event=abort_event,
                        ).__aiter__()
                        while True:
                            if abort_event.is_set():
                                terminal_event = StreamEventType.CANCELLED
                                terminal_metadata["reason"] = "cancellation_requested"
                                break
                            try:
                                chunk = await _next_chunk_or_cancel(
                                    iterator,
                                    abort_event,
                                    cleanup_timeout_seconds=self._cleanup_timeout_seconds,
                                )
                            except StopAsyncIteration:
                                break
                            if chunk is None:
                                terminal_event = StreamEventType.CANCELLED
                                terminal_metadata["reason"] = "cancellation_requested"
                                break
                            if isinstance(chunk, str):
                                chunk = BackendChunk(delta=chunk)
                            if not isinstance(chunk, BackendChunk):
                                raise TypeError(
                                    "backend stream must yield BackendChunk or str instances"
                                )
                            await queue.put(
                                _LocalEvent(
                                    binding=binding,
                                    stream_id=stream_id,
                                    per_stream_sequence=sequence,
                                    event_type=StreamEventType.DELTA,
                                    elapsed_ms=(time.perf_counter() - started_at) * 1000,
                                    delta=chunk.delta,
                                    metadata=chunk.metadata,
                                )
                            )
                            sequence += 1
            except asyncio.CancelledError as exc:
                if shutdown_event.is_set():
                    controller_cancellation = exc
                else:
                    terminal_event = StreamEventType.CANCELLED
                    terminal_error_type = "CancelledError"
                    terminal_error_message = "backend stream cancelled unexpectedly"
                    terminal_metadata["reason"] = "backend_cancelled"
                    if self._fail_fast:
                        abort_event.set()
            except Exception as exc:
                terminal_event = StreamEventType.FAILED
                terminal_error_type = type(exc).__name__
                terminal_error_message = str(exc)
                if self._fail_fast:
                    abort_event.set()

            cleanup_issue: _CleanupIssue | None = None
            try:
                cleanup_issue = await _close_iterator_bounded(
                    iterator,
                    timeout_seconds=self._cleanup_timeout_seconds,
                )
            except asyncio.CancelledError as exc:
                if shutdown_event.is_set():
                    controller_cancellation = exc
                else:
                    cleanup_issue = _CleanupIssue(
                        "CancelledError", "controller-side iterator cleanup was cancelled"
                    )

            if controller_cancellation is not None:
                raise controller_cancellation
            if cleanup_issue is not None:
                terminal_metadata.update(cleanup_issue.to_metadata())
                if terminal_event is StreamEventType.COMPLETED:
                    terminal_event = StreamEventType.FAILED
                    terminal_error_type = cleanup_issue.error_type
                    terminal_error_message = cleanup_issue.error_message
                    if self._fail_fast:
                        abort_event.set()

            await queue.put(
                _LocalEvent(
                    binding=binding,
                    stream_id=stream_id,
                    per_stream_sequence=sequence,
                    event_type=terminal_event,
                    elapsed_ms=(time.perf_counter() - started_at) * 1000,
                    error_type=terminal_error_type,
                    error_message=terminal_error_message,
                    metadata=terminal_metadata,
                )
            )

        worker_tasks = [asyncio.create_task(run_expert(binding)) for binding in resolved]
        cancel_monitor = (
            asyncio.create_task(mirror_external_cancel())
            if cancel_event is not None
            else None
        )
        terminal_types = {
            StreamEventType.COMPLETED,
            StreamEventType.FAILED,
            StreamEventType.CANCELLED,
        }
        accumulators = {
            _stream_id(request.run_id, binding.request_model_id): _StreamAccumulator(
                stream_id=_stream_id(request.run_id, binding.request_model_id),
                expert_id=binding.expert_id,
                request_model_id=binding.request_model_id,
                backend_model_id=binding.backend_model_id,
            )
            for binding in resolved
        }
        terminal_count = 0
        global_sequence = 0
        try:
            while terminal_count < len(resolved):
                local = await queue.get()
                event = SwarmEvent(
                    run_id=request.run_id,
                    task_bundle_sha256=request.task_bundle_sha256,
                    stream_id=local.stream_id,
                    expert_id=local.binding.expert_id,
                    request_model_id=local.binding.request_model_id,
                    backend_model_id=local.binding.backend_model_id,
                    per_stream_sequence=local.per_stream_sequence,
                    global_sequence=global_sequence,
                    event_type=local.event_type,
                    elapsed_ms=local.elapsed_ms,
                    delta=local.delta,
                    error_type=local.error_type,
                    error_message=local.error_message,
                    metadata=local.metadata,
                )
                accumulator = accumulators.get(event.stream_id)
                if accumulator is None:
                    raise NeuralSwarmStreamingError(
                        f"received an event for unknown stream: {event.stream_id}"
                    )
                accumulator.observe(event)
                global_sequence += 1
                if event.event_type in terminal_types:
                    terminal_count += 1
                yield event

            worker_results = await asyncio.gather(
                *worker_tasks, return_exceptions=True
            )
            unexpected_worker_errors = [
                result
                for result in worker_results
                if isinstance(result, BaseException)
            ]
            if unexpected_worker_errors:
                raise NeuralSwarmStreamingError(
                    "one or more expert workers exited after their terminal event"
                ) from unexpected_worker_errors[0]
            summary = _build_incremental_summary(
                run_id=request.run_id,
                task_bundle_sha256=request.task_bundle_sha256,
                accumulators=accumulators,
            )
            controller_fields = {
                "stream_id": f"{request.run_id}:__swarm__",
                "expert_id": "__controller__",
                "request_model_id": "__controller__",
                "backend_model_id": "__controller__",
            }
            barrier = SwarmEvent(
                run_id=request.run_id,
                task_bundle_sha256=request.task_bundle_sha256,
                per_stream_sequence=0,
                global_sequence=global_sequence,
                event_type=StreamEventType.BARRIER,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                metadata={"terminal_streams": terminal_count},
                **controller_fields,
            )
            yield barrier
            global_sequence += 1
            completed = SwarmEvent(
                run_id=request.run_id,
                task_bundle_sha256=request.task_bundle_sha256,
                per_stream_sequence=1,
                global_sequence=global_sequence,
                event_type=StreamEventType.SWARM_COMPLETED,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                metadata=summary.to_dict(),
                **controller_fields,
            )
            yield completed
        finally:
            shutdown_event.set()
            abort_event.set()
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            if cancel_monitor is not None:
                cancel_monitor.cancel()
                await asyncio.gather(cancel_monitor, return_exceptions=True)


async def collect_swarm_events(
    controller: NeuralSwarmStreamController,
    request: SwarmRequest,
    *,
    cancel_event: asyncio.Event | None = None,
) -> tuple[SwarmEvent, ...]:
    """Convenience helper for tests, demos, and non-streaming transports."""

    return tuple(
        [
            event
            async for event in controller.stream(request, cancel_event=cancel_event)
        ]
    )
