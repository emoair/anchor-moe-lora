"""Out-of-band hierarchical KV leases for Neural Swarm streaming.

This module is a control-plane bridge.  It deliberately never serializes a
KV handle into a prompt and never claims that ordinary in-stack LoRA caches
are interchangeable.  A backend receives a validated lease out of band and
is responsible for mapping that lease to real device pages.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from anchor_mvp.research.neural_swarm_streaming import (
    BackendChunk,
    ExpertBinding,
)
from anchor_mvp.research.hierarchical_kv import (
    AdapterExecutionProfile,
    HierarchicalTaskKVStore,
    KVProducerExecutionMode,
    SharedPrefixHandle,
)


KV_RUNTIME_CONTEXT_VERSION = "anchor.neural-swarm-kv-runtime-context.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NeuralSwarmKVRuntimeError(ValueError):
    """Raised when a KV lease cannot be used without ambiguity."""


class KVSharingMode(str, Enum):
    """Execution semantics, not an accuracy or performance claim."""

    EXACT_DECOUPLED_PREFIX = "exact_decoupled_prefix"
    APPROXIMATE_RESIDUAL = "approximate_residual"
    ISOLATED = "isolated"


@dataclass(frozen=True)
class TrustedExactKVBinding:
    """Caller-configured trust root for one exact shared-prefix lineage."""

    task_bundle_sha256: str
    compatibility_sha256: str
    ordered_prefix_chain_sha256: str
    shared_page_ids: tuple[str, ...]
    producer_execution_mode: KVProducerExecutionMode

    def __post_init__(self) -> None:
        for name, value in (
            ("task_bundle_sha256", self.task_bundle_sha256),
            ("compatibility_sha256", self.compatibility_sha256),
            ("ordered_prefix_chain_sha256", self.ordered_prefix_chain_sha256),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise NeuralSwarmKVRuntimeError(
                    f"{name} must be 64 lowercase hexadecimal characters"
                )
        if not self.shared_page_ids or any(
            _SHA256_RE.fullmatch(page_id) is None for page_id in self.shared_page_ids
        ):
            raise NeuralSwarmKVRuntimeError(
                "trusted shared_page_ids must contain SHA-256 values"
            )
        if len(set(self.shared_page_ids)) != len(self.shared_page_ids):
            raise NeuralSwarmKVRuntimeError(
                "trusted shared_page_ids must not contain duplicates"
            )
        if self.ordered_prefix_chain_sha256 != self.shared_page_ids[-1]:
            raise NeuralSwarmKVRuntimeError(
                "trusted prefix chain must terminate at the final shared page"
            )
        if (
            not isinstance(self.producer_execution_mode, KVProducerExecutionMode)
            or not self.producer_execution_mode.exact_share_safe
        ):
            raise NeuralSwarmKVRuntimeError(
                "trusted exact binding requires a decoupled producer mode"
            )

    @classmethod
    def from_store(
        cls,
        *,
        store: HierarchicalTaskKVStore,
        task_bundle_sha256: str,
        prefix: SharedPrefixHandle,
    ) -> TrustedExactKVBinding:
        """Authenticate one store prefix and freeze its expected exact binding."""

        identity = store.prefix_identity(task_bundle_sha256, prefix)
        page_ids = store.prefix_page_digests(task_bundle_sha256, prefix)
        if not page_ids:
            raise NeuralSwarmKVRuntimeError(
                "trusted exact binding requires at least one shared page"
            )
        return cls(
            task_bundle_sha256=task_bundle_sha256,
            compatibility_sha256=identity.digest,
            ordered_prefix_chain_sha256=page_ids[-1],
            shared_page_ids=page_ids,
            producer_execution_mode=identity.producer_execution_mode,
        )

    def validate_context(self, context: KVRuntimeContext) -> None:
        expected = {
            "task_bundle_sha256": (
                context.task_bundle_sha256,
                self.task_bundle_sha256,
            ),
            "compatibility_sha256": (
                context.compatibility_sha256,
                self.compatibility_sha256,
            ),
            "ordered_prefix_chain_sha256": (
                context.ordered_prefix_chain_sha256,
                self.ordered_prefix_chain_sha256,
            ),
            "shared_page_ids": (context.shared_page_ids, self.shared_page_ids),
            "producer_execution_mode": (
                context.kv_producer_mode,
                self.producer_execution_mode.value,
            ),
        }
        mismatches = [
            name for name, (actual, wanted) in expected.items() if actual != wanted
        ]
        if mismatches:
            raise NeuralSwarmKVRuntimeError(
                "KV lease does not match trusted exact binding: "
                + ", ".join(sorted(mismatches))
            )


@dataclass(frozen=True)
class KVRuntimeContext:
    """Validated, out-of-band references to one expert's cache view."""

    lease_id: str
    run_id: str
    task_bundle_sha256: str
    expert_id: str
    request_model_id: str
    compatibility_sha256: str
    ordered_prefix_chain_sha256: str
    shared_page_ids: tuple[str, ...]
    private_branch_id: str
    sharing_mode: KVSharingMode
    kv_producer_mode: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = KV_RUNTIME_CONTEXT_VERSION

    def __post_init__(self) -> None:
        required_text = {
            "lease_id": self.lease_id,
            "run_id": self.run_id,
            "expert_id": self.expert_id,
            "request_model_id": self.request_model_id,
            "private_branch_id": self.private_branch_id,
            "kv_producer_mode": self.kv_producer_mode,
        }
        for name, value in required_text.items():
            if not isinstance(value, str) or not value.strip():
                raise NeuralSwarmKVRuntimeError(f"{name} must be non-empty")
        for name, value in (
            ("task_bundle_sha256", self.task_bundle_sha256),
            ("compatibility_sha256", self.compatibility_sha256),
            ("ordered_prefix_chain_sha256", self.ordered_prefix_chain_sha256),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise NeuralSwarmKVRuntimeError(
                    f"{name} must be 64 lowercase hexadecimal characters"
                )
        if not isinstance(self.sharing_mode, KVSharingMode):
            raise NeuralSwarmKVRuntimeError("sharing_mode must be a KVSharingMode")
        if any(not isinstance(page_id, str) or not page_id.strip() for page_id in self.shared_page_ids):
            raise NeuralSwarmKVRuntimeError("shared_page_ids must contain non-empty strings")
        if len(set(self.shared_page_ids)) != len(self.shared_page_ids):
            raise NeuralSwarmKVRuntimeError("shared_page_ids must not contain duplicates")
        if not isinstance(self.metadata, Mapping):
            raise NeuralSwarmKVRuntimeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def validate_dispatch(
        self,
        *,
        binding: ExpertBinding,
        run_id: str,
        task_bundle_sha256: str,
        allow_approximate: bool,
    ) -> None:
        """Bind the lease to exactly one resolved stream."""

        expected = {
            "run_id": (self.run_id, run_id),
            "task_bundle_sha256": (
                self.task_bundle_sha256,
                task_bundle_sha256,
            ),
            "expert_id": (self.expert_id, binding.expert_id),
            "request_model_id": (
                self.request_model_id,
                binding.request_model_id,
            ),
        }
        mismatches = [name for name, (actual, want) in expected.items() if actual != want]
        if mismatches:
            raise NeuralSwarmKVRuntimeError(
                "KV lease dispatch mismatch: " + ", ".join(sorted(mismatches))
            )
        if self.sharing_mode is KVSharingMode.EXACT_DECOUPLED_PREFIX:
            if not self.shared_page_ids:
                raise NeuralSwarmKVRuntimeError(
                    "exact shared KV requires at least one shared page"
                )
            exact_modes = {
                KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX.value,
                KVProducerExecutionMode.CROSS_ATTENTION_MEMORY.value,
            }
            if self.kv_producer_mode not in exact_modes:
                raise NeuralSwarmKVRuntimeError(
                    "exact shared KV requires an attested decoupled producer"
                )
            if self.ordered_prefix_chain_sha256 != self.shared_page_ids[-1]:
                raise NeuralSwarmKVRuntimeError(
                    "ordered prefix chain must terminate at the final shared page"
                )
        elif self.sharing_mode is KVSharingMode.APPROXIMATE_RESIDUAL:
            if not self.shared_page_ids:
                raise NeuralSwarmKVRuntimeError(
                    "approximate residual sharing requires shared pages"
                )
            if not allow_approximate:
                raise NeuralSwarmKVRuntimeError(
                    "approximate residual KV sharing requires explicit opt-in"
                )
        elif self.sharing_mode is not KVSharingMode.ISOLATED:
            raise NeuralSwarmKVRuntimeError("unsupported KV sharing mode")


@runtime_checkable
class KVContextProvider(Protocol):
    """Acquire one bounded-lifetime cache lease for a resolved stream."""

    def lease(
        self,
        *,
        binding: ExpertBinding,
        run_id: str,
        task_bundle_sha256: str,
    ) -> AbstractAsyncContextManager[KVRuntimeContext]: ...


@runtime_checkable
class KVStreamingExpertBackend(Protocol):
    """A backend that consumes device/cache state out of band."""

    def stream_with_kv(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        kv_context: KVRuntimeContext,
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]: ...


class InMemoryHierarchicalKVContextProvider:
    """Lease empty expert-private branches from the metadata-only MVP store.

    Prefix ownership stays with the caller.  Each acquired branch is released
    in ``finally``; no capability token is exposed to the downstream backend.
    """

    def __init__(
        self,
        *,
        store: HierarchicalTaskKVStore,
        prefixes: Mapping[str, SharedPrefixHandle],
        adapters: Mapping[str, AdapterExecutionProfile],
    ) -> None:
        if not isinstance(store, HierarchicalTaskKVStore):
            raise NeuralSwarmKVRuntimeError(
                "store must be a HierarchicalTaskKVStore"
            )
        if not prefixes:
            raise NeuralSwarmKVRuntimeError("prefixes must not be empty")
        if not adapters:
            raise NeuralSwarmKVRuntimeError("adapters must not be empty")
        for task_bundle_sha256, prefix in prefixes.items():
            if _SHA256_RE.fullmatch(task_bundle_sha256) is None:
                raise NeuralSwarmKVRuntimeError(
                    "prefix keys must be task bundle SHA-256 values"
                )
            if not isinstance(prefix, SharedPrefixHandle):
                raise NeuralSwarmKVRuntimeError(
                    "prefix values must be SharedPrefixHandle instances"
                )
            if prefix.task_id != task_bundle_sha256:
                raise NeuralSwarmKVRuntimeError(
                    "prefix task_id must equal its task bundle SHA-256 key"
                )
        for expert_id, adapter in adapters.items():
            if not isinstance(expert_id, str) or not expert_id.strip():
                raise NeuralSwarmKVRuntimeError("adapter keys must be expert ids")
            if not isinstance(adapter, AdapterExecutionProfile):
                raise NeuralSwarmKVRuntimeError(
                    "adapter values must be AdapterExecutionProfile instances"
                )
        self._store = store
        self._prefixes = dict(prefixes)
        self._adapters = dict(adapters)

    @asynccontextmanager
    async def lease(
        self,
        *,
        binding: ExpertBinding,
        run_id: str,
        task_bundle_sha256: str,
    ) -> AsyncIterator[KVRuntimeContext]:
        prefix = self._prefixes.get(task_bundle_sha256)
        if prefix is None:
            raise NeuralSwarmKVRuntimeError(
                "no authenticated shared prefix for task bundle"
            )
        adapter = self._adapters.get(binding.expert_id)
        if adapter is None:
            raise NeuralSwarmKVRuntimeError("no adapter profile for expert")
        identity = self._store.prefix_identity(task_bundle_sha256, prefix)
        if not identity.producer_execution_mode.exact_share_safe:
            raise NeuralSwarmKVRuntimeError(
                "prefix producer identity is not exact-share compatible"
            )
        branch = self._store.open_private_branch(
            task_bundle_sha256,
            binding.expert_id,
            prefix,
            adapter,
        )
        try:
            view = self._store.combined_view(
                task_bundle_sha256,
                binding.expert_id,
                branch,
            )
            if not view.shared_prefix_pages:
                raise NeuralSwarmKVRuntimeError(
                    "exact shared-prefix lease requires at least one shared page"
                )
            lease_id = hashlib.sha256(
                f"{run_id}\0{branch.branch_id}".encode("utf-8")
            ).hexdigest()
            yield KVRuntimeContext(
                lease_id=lease_id,
                run_id=run_id,
                task_bundle_sha256=task_bundle_sha256,
                expert_id=binding.expert_id,
                request_model_id=binding.request_model_id,
                compatibility_sha256=view.compatibility_digest,
                ordered_prefix_chain_sha256=view.shared_prefix_pages[-1].digest,
                shared_page_ids=tuple(
                    page.digest for page in view.shared_prefix_pages
                ),
                private_branch_id=branch.branch_id,
                sharing_mode=KVSharingMode.EXACT_DECOUPLED_PREFIX,
                kv_producer_mode=identity.producer_execution_mode.value,
                metadata={
                    "adapter_id": adapter.adapter_id,
                    "adapter_placement": adapter.placement.value,
                    "branch_version": view.branch_version,
                    "shared_page_count": len(view.shared_prefix_pages),
                    "private_page_count": len(view.expert_private_pages),
                    "metadata_only_store": True,
                },
            )
        finally:
            self._store.close_branch(
                task_bundle_sha256,
                binding.expert_id,
                branch,
            )


class HierarchicalKVBackendAdapter:
    """Adapt a KV-aware backend to the existing transport-neutral controller."""

    def __init__(
        self,
        *,
        context_provider: KVContextProvider,
        backend: KVStreamingExpertBackend,
        allow_approximate: bool = False,
        cleanup_timeout_seconds: float = 1.0,
        trusted_exact_bindings: Mapping[str, TrustedExactKVBinding] | None = None,
    ) -> None:
        if not isinstance(context_provider, KVContextProvider):
            raise NeuralSwarmKVRuntimeError(
                "context_provider must implement KVContextProvider"
            )
        if not isinstance(backend, KVStreamingExpertBackend):
            raise NeuralSwarmKVRuntimeError(
                "backend must implement KVStreamingExpertBackend"
            )
        if not isinstance(allow_approximate, bool):
            raise NeuralSwarmKVRuntimeError("allow_approximate must be boolean")
        if (
            isinstance(cleanup_timeout_seconds, bool)
            or not isinstance(cleanup_timeout_seconds, (int, float))
            or not math.isfinite(float(cleanup_timeout_seconds))
            or cleanup_timeout_seconds <= 0
        ):
            raise NeuralSwarmKVRuntimeError(
                "cleanup_timeout_seconds must be finite and positive"
            )
        self._context_provider = context_provider
        self._backend = backend
        self._allow_approximate = allow_approximate
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)
        self._trusted_exact_bindings: dict[str, TrustedExactKVBinding] = {}
        for task_bundle_sha256, trusted in (trusted_exact_bindings or {}).items():
            if not isinstance(trusted, TrustedExactKVBinding):
                raise NeuralSwarmKVRuntimeError(
                    "trusted exact bindings must contain TrustedExactKVBinding values"
                )
            if task_bundle_sha256 != trusted.task_bundle_sha256:
                raise NeuralSwarmKVRuntimeError(
                    "trusted exact binding key does not match task bundle"
                )
            self._trusted_exact_bindings[task_bundle_sha256] = trusted

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk | str]:
        """Lease, validate, delegate, and release without mutating the prompt."""

        lease_manager = self._context_provider.lease(
            binding=binding,
            run_id=run_id,
            task_bundle_sha256=task_bundle_sha256,
        )
        async with lease_manager as kv_context:
            if not isinstance(kv_context, KVRuntimeContext):
                raise NeuralSwarmKVRuntimeError(
                    "context provider must yield KVRuntimeContext"
                )
            kv_context.validate_dispatch(
                binding=binding,
                run_id=run_id,
                task_bundle_sha256=task_bundle_sha256,
                allow_approximate=self._allow_approximate,
            )
            if kv_context.sharing_mode is KVSharingMode.EXACT_DECOUPLED_PREFIX:
                trusted = self._trusted_exact_bindings.get(task_bundle_sha256)
                if trusted is None:
                    raise NeuralSwarmKVRuntimeError(
                        "exact shared KV requires a caller-configured trusted binding"
                    )
                trusted.validate_context(kv_context)
            iterator = self._backend.stream_with_kv(
                binding=binding,
                shared_input=shared_input,
                kv_context=kv_context,
                run_id=run_id,
                task_bundle_sha256=task_bundle_sha256,
                cancel_event=cancel_event,
            )
            primary_failure = False
            try:
                async for chunk in iterator:
                    yield chunk
            except BaseException:
                primary_failure = True
                raise
            finally:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    try:
                        await _bounded_aclose(
                            close,
                            timeout_seconds=self._cleanup_timeout_seconds,
                        )
                    except BaseException:
                        if not primary_failure:
                            raise


def _freeze_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy JSON-like metadata so providers cannot mutate a live lease."""

    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise NeuralSwarmKVRuntimeError("metadata keys must be strings")
        if isinstance(item, float) and not math.isfinite(item):
            raise NeuralSwarmKVRuntimeError(
                f"metadata value for {key!r} must be finite"
            )
        if item is None or isinstance(item, (str, int, float, bool)):
            frozen[key] = item
        elif isinstance(item, Mapping):
            frozen[key] = _freeze_metadata(item)
        elif isinstance(item, (list, tuple)):
            frozen[key] = tuple(_freeze_metadata_value(child) for child in item)
        else:
            raise NeuralSwarmKVRuntimeError(
                f"metadata value for {key!r} is not JSON-compatible"
            )
    return _ReadOnlyDict(frozen)


async def _bounded_aclose(close: Any, *, timeout_seconds: float) -> None:
    """Bound cooperative iterator shutdown without waiting forever on cancel."""

    close_task = asyncio.ensure_future(close())
    try:
        done, _pending = await asyncio.wait(
            {close_task}, timeout=timeout_seconds
        )
    except BaseException:
        close_task.cancel()
        close_task.add_done_callback(_consume_background_task_result)
        raise
    if close_task not in done:
        close_task.cancel()
        close_task.add_done_callback(_consume_background_task_result)
        raise NeuralSwarmKVRuntimeError("backend iterator cleanup timed out")
    await close_task


def _consume_background_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.exception()
    except BaseException:
        pass


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise NeuralSwarmKVRuntimeError("metadata sequence numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(child) for child in value)
    raise NeuralSwarmKVRuntimeError("metadata sequence contains a non-JSON value")


class _ReadOnlyDict(dict[str, Any]):
    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("KV runtime metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable
