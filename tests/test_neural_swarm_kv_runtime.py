from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

from anchor_mvp.research.neural_swarm_kv_runtime import (
    HierarchicalKVBackendAdapter,
    InMemoryHierarchicalKVContextProvider,
    KVRuntimeContext,
    KVSharingMode,
    NeuralSwarmKVRuntimeError,
    TrustedExactKVBinding,
)
from anchor_mvp.research.hierarchical_kv import (
    AdapterExecutionProfile,
    AdapterPlacement,
    HierarchicalTaskKVStore,
    KVCompatibilityIdentity,
    KVProducerExecutionMode,
    KVSegment,
)
from anchor_mvp.research.neural_swarm_streaming import (
    BackendChunk,
    ExpertBinding,
    NeuralSwarmStreamController,
    StreamEventType,
    SwarmRequest,
    collect_swarm_events,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class RecordingProvider:
    def __init__(self, *, sharing_mode: KVSharingMode = KVSharingMode.EXACT_DECOUPLED_PREFIX) -> None:
        self.sharing_mode = sharing_mode
        self.active = 0
        self.acquired: list[KVRuntimeContext] = []
        self.released: list[str] = []

    @asynccontextmanager
    async def lease(
        self,
        *,
        binding: ExpertBinding,
        run_id: str,
        task_bundle_sha256: str,
    ):
        context = KVRuntimeContext(
            lease_id=f"lease:{binding.expert_id}",
            run_id=run_id,
            task_bundle_sha256=task_bundle_sha256,
            expert_id=binding.expert_id,
            request_model_id=binding.request_model_id,
            compatibility_sha256=_sha("compat"),
            ordered_prefix_chain_sha256=_sha("same-prefix"),
            shared_page_ids=(_sha("same-prefix"),),
            private_branch_id=f"private:{binding.expert_id}",
            sharing_mode=self.sharing_mode,
            kv_producer_mode="decoupled_frozen_prefix",
        )
        self.active += 1
        self.acquired.append(context)
        try:
            yield context
        finally:
            self.active -= 1
            self.released.append(context.lease_id)


class RecordingBackend:
    def __init__(self, *, wait_for_cancel: bool = False) -> None:
        self.wait_for_cancel = wait_for_cancel
        self.contexts: list[KVRuntimeContext] = []
        self.inputs: list[Mapping[str, Any]] = []

    async def stream_with_kv(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        kv_context: KVRuntimeContext,
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        del run_id, task_bundle_sha256
        self.contexts.append(kv_context)
        self.inputs.append(shared_input)
        if self.wait_for_cancel:
            await cancel_event.wait()
            return
        yield BackendChunk(delta=binding.expert_id)


class FailingBackend(RecordingBackend):
    async def stream_with_kv(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        kv_context: KVRuntimeContext,
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        del (
            binding,
            shared_input,
            kv_context,
            run_id,
            task_bundle_sha256,
            cancel_event,
        )
        if False:  # pragma: no cover - makes this an async generator
            yield "unreachable"
        raise RuntimeError("backend failed")


class FinalizingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def stream_with_kv(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        kv_context: KVRuntimeContext,
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        del shared_input, kv_context, run_id, task_bundle_sha256, cancel_event
        try:
            yield BackendChunk(delta=binding.expert_id)
            await asyncio.Event().wait()
        finally:
            self.closed = True


class _SlowCloseIterator:
    def __init__(self) -> None:
        self.yielded = False
        self.close_cancelled = False

    def __aiter__(self) -> _SlowCloseIterator:
        return self

    async def __anext__(self) -> BackendChunk:
        if self.yielded:
            raise StopAsyncIteration
        self.yielded = True
        return BackendChunk(delta="planner")

    async def aclose(self) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise


class SlowCloseBackend:
    def __init__(self) -> None:
        self.iterator = _SlowCloseIterator()

    def stream_with_kv(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        kv_context: KVRuntimeContext,
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        del (
            binding,
            shared_input,
            kv_context,
            run_id,
            task_bundle_sha256,
            cancel_event,
        )
        return self.iterator


def _bindings() -> tuple[ExpertBinding, ...]:
    return (
        ExpertBinding("planner-model", "planner", "backend-planner"),
        ExpertBinding("review-model", "review", "backend-review"),
    )


def _identity() -> KVCompatibilityIdentity:
    return KVCompatibilityIdentity(
        model_id="test-base",
        model_revision_sha256=_sha("model"),
        tokenizer_sha256=_sha("tokenizer"),
        rope_config_sha256=_sha("rope"),
        kv_layout_sha256=_sha("layout"),
        kv_producer_path_sha256=_sha("frozen-producer"),
        base_kv_weights_sha256=_sha("weights"),
        base_kv_weights_epoch=0,
        producer_execution_mode=KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX,
    )


def _segment() -> KVSegment:
    return KVSegment(
        payload_sha256=_sha("payload"),
        payload_bytes=7,
        token_ids_sha256=_sha("tokens"),
        position_ids_sha256=_sha("positions"),
        position_start=0,
        position_end=4,
    )


def _adapter(expert_id: str) -> AdapterExecutionProfile:
    return AdapterExecutionProfile(
        adapter_id=f"adapter:{expert_id}",
        adapter_revision_sha256=_sha(f"adapter:{expert_id}"),
        target_projections=frozenset({"q"}),
        placement=AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
    )


def _recording_trusted_binding(
    task_bundle_sha256: str = _sha("bundle"),
) -> dict[str, TrustedExactKVBinding]:
    return {
        task_bundle_sha256: TrustedExactKVBinding(
            task_bundle_sha256=task_bundle_sha256,
            compatibility_sha256=_sha("compat"),
            ordered_prefix_chain_sha256=_sha("same-prefix"),
            shared_page_ids=(_sha("same-prefix"),),
            producer_execution_mode=(
                KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX
            ),
        )
    }


def test_two_experts_share_prefix_but_keep_private_branches() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        backend = RecordingBackend()
        controller = NeuralSwarmStreamController(
            bindings=_bindings(),
            backend=HierarchicalKVBackendAdapter(
                context_provider=provider,
                backend=backend,
                trusted_exact_bindings=_recording_trusted_binding(),
            ),
        )
        request = SwarmRequest(
            run_id="run-kv",
            task_bundle_sha256=_sha("bundle"),
            request_model_ids=("planner-model", "review-model"),
            shared_input={"messages": [{"role": "user", "content": "opaque"}]},
        )

        events = await collect_swarm_events(controller, request)

        assert [event.event_type for event in events[-2:]] == [
            StreamEventType.BARRIER,
            StreamEventType.SWARM_COMPLETED,
        ]
        assert {
            context.ordered_prefix_chain_sha256 for context in backend.contexts
        } == {_sha("same-prefix")}
        assert len({context.private_branch_id for context in backend.contexts}) == 2
        assert provider.active == 0
        assert len(provider.released) == 2

    asyncio.run(exercise())


def test_kv_context_is_out_of_band_and_prompt_is_not_mutated() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        backend = RecordingBackend()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=backend,
            trusted_exact_bindings=_recording_trusted_binding(),
        )
        binding = _bindings()[0]
        shared_input = {"messages": [{"role": "user", "content": "task"}]}

        chunks = [
            chunk
            async for chunk in adapter.stream(
                binding=binding,
                shared_input=shared_input,
                run_id="run-direct",
                task_bundle_sha256=_sha("bundle"),
                cancel_event=asyncio.Event(),
            )
        ]

        assert chunks == [BackendChunk(delta="planner")]
        assert backend.inputs == [shared_input]
        assert "kv_context" not in shared_input
        assert "shared_page_ids" not in shared_input

    asyncio.run(exercise())


def test_approximate_sharing_fails_closed_without_opt_in() -> None:
    async def exercise() -> None:
        adapter = HierarchicalKVBackendAdapter(
            context_provider=RecordingProvider(
                sharing_mode=KVSharingMode.APPROXIMATE_RESIDUAL
            ),
            backend=RecordingBackend(),
        )

        with pytest.raises(
            NeuralSwarmKVRuntimeError,
            match="requires explicit opt-in",
        ):
            _ = [
                chunk
                async for chunk in adapter.stream(
                    binding=_bindings()[0],
                    shared_input={"messages": []},
                    run_id="run-approx",
                    task_bundle_sha256=_sha("bundle"),
                    cancel_event=asyncio.Event(),
                )
            ]

    asyncio.run(exercise())


def test_untrusted_provider_cannot_self_declare_an_exact_binding() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=RecordingBackend(),
        )

        with pytest.raises(
            NeuralSwarmKVRuntimeError,
            match="caller-configured trusted binding",
        ):
            _ = [
                chunk
                async for chunk in adapter.stream(
                    binding=_bindings()[0],
                    shared_input={"messages": []},
                    run_id="run-untrusted",
                    task_bundle_sha256=_sha("bundle"),
                    cancel_event=asyncio.Event(),
                )
            ]

        assert provider.active == 0
        assert provider.released == ["lease:planner"]

    asyncio.run(exercise())


def test_exact_mode_rejects_in_stack_producer() -> None:
    context = KVRuntimeContext(
        lease_id="lease",
        run_id="run",
        task_bundle_sha256=_sha("bundle"),
        expert_id="planner",
        request_model_id="planner-model",
        compatibility_sha256=_sha("compat"),
        ordered_prefix_chain_sha256=_sha("prefix"),
        shared_page_ids=("page",),
        private_branch_id="private",
        sharing_mode=KVSharingMode.EXACT_DECOUPLED_PREFIX,
        kv_producer_mode="naive_in_stack_q_lora",
    )

    with pytest.raises(
        NeuralSwarmKVRuntimeError,
        match="attested decoupled producer",
    ):
        context.validate_dispatch(
            binding=_bindings()[0],
            run_id="run",
            task_bundle_sha256=_sha("bundle"),
            allow_approximate=False,
        )


def test_lease_releases_when_consumer_closes_stream_early() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        backend = FinalizingBackend()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=backend,
            trusted_exact_bindings=_recording_trusted_binding(),
        )
        iterator = adapter.stream(
            binding=_bindings()[0],
            shared_input={"messages": []},
            run_id="run-close",
            task_bundle_sha256=_sha("bundle"),
            cancel_event=asyncio.Event(),
        )

        assert await anext(iterator) == BackendChunk(delta="planner")
        await iterator.aclose()

        assert backend.closed is True
        assert provider.active == 0
        assert provider.released == ["lease:planner"]

    asyncio.run(exercise())


def test_lease_releases_when_backend_fails() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=FailingBackend(),
            trusted_exact_bindings=_recording_trusted_binding(),
        )

        with pytest.raises(RuntimeError, match="backend failed"):
            _ = [
                chunk
                async for chunk in adapter.stream(
                    binding=_bindings()[0],
                    shared_input={"messages": []},
                    run_id="run-fail",
                    task_bundle_sha256=_sha("bundle"),
                    cancel_event=asyncio.Event(),
                )
            ]

        assert provider.active == 0
        assert provider.released == ["lease:planner"]

    asyncio.run(exercise())


def test_backend_iterator_cleanup_timeout_is_bounded_and_cancels_close() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        backend = SlowCloseBackend()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=backend,
            cleanup_timeout_seconds=0.01,
            trusted_exact_bindings=_recording_trusted_binding(),
        )

        with pytest.raises(
            NeuralSwarmKVRuntimeError,
            match="cleanup timed out",
        ):
            _ = [
                chunk
                async for chunk in adapter.stream(
                    binding=_bindings()[0],
                    shared_input={"messages": []},
                    run_id="run-timeout",
                    task_bundle_sha256=_sha("bundle"),
                    cancel_event=asyncio.Event(),
                )
            ]

        await asyncio.sleep(0)
        assert backend.iterator.close_cancelled is True
        assert provider.active == 0

    asyncio.run(exercise())


def test_cancellation_during_backend_cleanup_cancels_the_close_task() -> None:
    async def exercise() -> None:
        provider = RecordingProvider()
        backend = SlowCloseBackend()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=backend,
            cleanup_timeout_seconds=60,
            trusted_exact_bindings=_recording_trusted_binding(),
        )

        async def consume() -> list[BackendChunk | str]:
            return [
                chunk
                async for chunk in adapter.stream(
                    binding=_bindings()[0],
                    shared_input={"messages": []},
                    run_id="run-cancel-cleanup",
                    task_bundle_sha256=_sha("bundle"),
                    cancel_event=asyncio.Event(),
                )
            ]

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        assert backend.iterator.close_cancelled is True
        assert provider.active == 0

    asyncio.run(exercise())


def test_runtime_metadata_is_immutable() -> None:
    context = KVRuntimeContext(
        lease_id="lease",
        run_id="run",
        task_bundle_sha256=_sha("bundle"),
        expert_id="planner",
        request_model_id="planner-model",
        compatibility_sha256=_sha("compat"),
        ordered_prefix_chain_sha256=_sha("prefix"),
        shared_page_ids=("page",),
        private_branch_id="private",
        sharing_mode=KVSharingMode.ISOLATED,
        kv_producer_mode="isolated",
        metadata={"layers": [0, 1]},
    )

    with pytest.raises(TypeError, match="immutable"):
        context.metadata["layers"] = [2]  # type: ignore[index]
    assert context.metadata["layers"] == (0, 1)


def test_runtime_rejects_non_finite_metadata() -> None:
    with pytest.raises(NeuralSwarmKVRuntimeError, match="must be finite"):
        KVRuntimeContext(
            lease_id="lease",
            run_id="run",
            task_bundle_sha256=_sha("bundle"),
            expert_id="planner",
            request_model_id="planner-model",
            compatibility_sha256=_sha("compat"),
            ordered_prefix_chain_sha256=_sha("prefix"),
            shared_page_ids=("page",),
            private_branch_id="private",
            sharing_mode=KVSharingMode.ISOLATED,
            kv_producer_mode="isolated",
            metadata={"ratio": math.nan},
        )


def test_exact_runtime_context_requires_a_shared_page() -> None:
    context = KVRuntimeContext(
        lease_id="lease",
        run_id="run",
        task_bundle_sha256=_sha("bundle"),
        expert_id="planner",
        request_model_id="planner-model",
        compatibility_sha256=_sha("compat"),
        ordered_prefix_chain_sha256=_sha("prefix"),
        shared_page_ids=(),
        private_branch_id="private",
        sharing_mode=KVSharingMode.EXACT_DECOUPLED_PREFIX,
        kv_producer_mode="decoupled_frozen_prefix",
    )

    with pytest.raises(NeuralSwarmKVRuntimeError, match="at least one shared page"):
        context.validate_dispatch(
            binding=_bindings()[0],
            run_id="run",
            task_bundle_sha256=_sha("bundle"),
            allow_approximate=False,
        )


def test_concrete_provider_reuses_shared_pages_and_releases_private_refs() -> None:
    async def exercise() -> None:
        bundle = _sha("concrete-bundle")
        store = HierarchicalTaskKVStore(
            max_pages=8,
            max_inline_bytes=128,
            max_inline_bytes_per_page=64,
        )
        prefix = store.publish_seed_prefix(bundle, _identity(), (_segment(),))
        provider = InMemoryHierarchicalKVContextProvider(
            store=store,
            prefixes={bundle: prefix},
            adapters={
                "planner": _adapter("planner"),
                "review": _adapter("review"),
            },
        )
        backend = RecordingBackend()
        controller = NeuralSwarmStreamController(
            bindings=_bindings(),
            backend=HierarchicalKVBackendAdapter(
                context_provider=provider,
                backend=backend,
                trusted_exact_bindings={
                    bundle: TrustedExactKVBinding.from_store(
                        store=store,
                        task_bundle_sha256=bundle,
                        prefix=prefix,
                    )
                },
            ),
        )

        await collect_swarm_events(
            controller,
            SwarmRequest(
                run_id="run-concrete",
                task_bundle_sha256=bundle,
                request_model_ids=("planner-model", "review-model"),
                shared_input={"messages": []},
            ),
        )

        assert len({context.shared_page_ids for context in backend.contexts}) == 1
        assert len({context.private_branch_id for context in backend.contexts}) == 2
        stats = store.stats()
        assert stats.open_branches == 0
        assert stats.private_pages == 0
        assert stats.open_prefixes == 1
        assert stats.shared_pages == 1
        store.close_prefix(bundle, prefix)
        assert store.stats().shared_pages == 0

    asyncio.run(exercise())


def test_exact_context_rejects_a_prefix_terminal_mismatch() -> None:
    context = KVRuntimeContext(
        lease_id="lease",
        run_id="run",
        task_bundle_sha256=_sha("bundle"),
        expert_id="planner",
        request_model_id="planner-model",
        compatibility_sha256=_sha("compat"),
        ordered_prefix_chain_sha256=_sha("terminal"),
        shared_page_ids=(_sha("different-terminal"),),
        private_branch_id="private",
        sharing_mode=KVSharingMode.EXACT_DECOUPLED_PREFIX,
        kv_producer_mode=KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX.value,
    )

    with pytest.raises(
        NeuralSwarmKVRuntimeError,
        match="terminate at the final shared page",
    ):
        context.validate_dispatch(
            binding=_bindings()[0],
            run_id="run",
            task_bundle_sha256=_sha("bundle"),
            allow_approximate=False,
        )


def test_concrete_provider_rejects_naive_in_stack_prefix_identity() -> None:
    async def exercise() -> None:
        bundle = _sha("naive-bundle")
        store = HierarchicalTaskKVStore()
        naive_identity = KVCompatibilityIdentity(
            model_id="test-base",
            model_revision_sha256=_sha("model"),
            tokenizer_sha256=_sha("tokenizer"),
            rope_config_sha256=_sha("rope"),
            kv_layout_sha256=_sha("layout"),
            kv_producer_path_sha256=_sha("naive-producer"),
            base_kv_weights_sha256=_sha("weights"),
            base_kv_weights_epoch=0,
            producer_execution_mode=KVProducerExecutionMode.NAIVE_DECODER_IN_STACK,
        )
        prefix = store.publish_seed_prefix(bundle, naive_identity, (_segment(),))
        provider = InMemoryHierarchicalKVContextProvider(
            store=store,
            prefixes={bundle: prefix},
            adapters={"planner": _adapter("planner")},
        )
        backend = RecordingBackend()
        adapter = HierarchicalKVBackendAdapter(
            context_provider=provider,
            backend=backend,
        )

        with pytest.raises(
            NeuralSwarmKVRuntimeError,
            match="not exact-share compatible",
        ):
            _ = [
                chunk
                async for chunk in adapter.stream(
                    binding=_bindings()[0],
                    shared_input={"messages": []},
                    run_id="run-naive",
                    task_bundle_sha256=bundle,
                    cancel_event=asyncio.Event(),
                )
            ]

        assert store.stats().open_branches == 0
        store.close_prefix(bundle, prefix)

    asyncio.run(exercise())
