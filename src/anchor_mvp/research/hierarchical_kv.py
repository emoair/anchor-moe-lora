"""Pure-memory Hierarchical Task-KV contract prototype.

This module models cache ownership, compatibility, lineage, commit, and
copy-on-write semantics.  It deliberately stores only caller-supplied digests
plus bounded previews; it contains no tensor, GPU, model, or network code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import secrets
import threading
from typing import Iterable, Literal


_SHA256_CHARS = frozenset("0123456789abcdef")
_PROJECTIONS = frozenset({"q", "k", "v", "o"})


class HierarchicalKVError(RuntimeError):
    """Base class for stable, content-free Task-KV failures."""


class KVCompatibilityError(HierarchicalKVError):
    """The cache producer or adapter path is not exactly share-compatible."""


class KVAuthorizationError(HierarchicalKVError):
    """A task or expert attempted to use another scope's capability."""


class KVVersionError(HierarchicalKVError):
    """A stale, committed, closed, or otherwise invalid branch was used."""


class KVCapacityError(HierarchicalKVError):
    """The configured in-memory metadata/preview budget would be exceeded."""


class KVInvalidatedError(HierarchicalKVError):
    """The compatibility identity has been explicitly invalidated."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _require_sha256(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return str(value)


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty bounded string")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


class KVProducerExecutionMode(str, Enum):
    """How the immutable prefix K/V tensors were actually produced."""

    DECOUPLED_FROZEN_PREFIX = "decoupled_frozen_prefix"
    CROSS_ATTENTION_MEMORY = "cross_attention_memory"
    NAIVE_DECODER_IN_STACK = "naive_decoder_in_stack"

    @property
    def exact_share_safe(self) -> bool:
        return self in {
            KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX,
            KVProducerExecutionMode.CROSS_ATTENTION_MEMORY,
        }


@dataclass(frozen=True)
class KVCompatibilityIdentity:
    """Exact identity of every input that can change produced K/V tensors."""

    model_id: str
    model_revision_sha256: str
    tokenizer_sha256: str
    rope_config_sha256: str
    kv_layout_sha256: str
    kv_producer_path_sha256: str
    base_kv_weights_sha256: str
    base_kv_weights_epoch: int
    producer_execution_mode: KVProducerExecutionMode

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, "model_id")
        for field in (
            "model_revision_sha256",
            "tokenizer_sha256",
            "rope_config_sha256",
            "kv_layout_sha256",
            "kv_producer_path_sha256",
            "base_kv_weights_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if (
            isinstance(self.base_kv_weights_epoch, bool)
            or not isinstance(self.base_kv_weights_epoch, int)
            or self.base_kv_weights_epoch < 0
        ):
            raise ValueError("base_kv_weights_epoch must be a non-negative integer")
        if not isinstance(self.producer_execution_mode, KVProducerExecutionMode):
            raise ValueError(
                "producer_execution_mode must be KVProducerExecutionMode"
            )

    @property
    def digest(self) -> str:
        return _canonical_sha256(
            {
                "model_id": self.model_id,
                "model_revision_sha256": self.model_revision_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "rope_config_sha256": self.rope_config_sha256,
                "kv_layout_sha256": self.kv_layout_sha256,
                "kv_producer_path_sha256": self.kv_producer_path_sha256,
                "base_kv_weights_sha256": self.base_kv_weights_sha256,
                "base_kv_weights_epoch": self.base_kv_weights_epoch,
                "producer_execution_mode": self.producer_execution_mode.value,
            }
        )


class AdapterPlacement(str, Enum):
    """Where an expert Query adapter executes relative to the K/V producer."""

    DECOUPLED_FROZEN_FACT_READOUT = "decoupled_frozen_fact_readout"
    CROSS_ATTENTION_QUERY = "cross_attention_query"
    NAIVE_DECODER_IN_STACK = "naive_decoder_in_stack"


@dataclass(frozen=True)
class AdapterExecutionProfile:
    """Adapter placement contract used before sharing a prefix cache."""

    adapter_id: str
    adapter_revision_sha256: str
    target_projections: frozenset[str]
    placement: AdapterPlacement

    def __post_init__(self) -> None:
        _require_identifier(self.adapter_id, "adapter_id")
        _require_sha256(self.adapter_revision_sha256, "adapter_revision_sha256")
        if not isinstance(self.target_projections, frozenset):
            raise ValueError("target_projections must be a frozenset")
        if not self.target_projections or not self.target_projections <= _PROJECTIONS:
            raise ValueError("target_projections contains unsupported projections")
        if not isinstance(self.placement, AdapterPlacement):
            raise ValueError("placement must be AdapterPlacement")

    def assert_share_compatible(self) -> None:
        """Fail closed unless Q is outside the frozen fact/KV producer path.

        Q-only is necessary but not sufficient: an in-stack decoder Q update
        changes hidden states and therefore later-layer K/V tensors.
        """

        if self.target_projections != frozenset({"q"}):
            raise KVCompatibilityError("adapter_not_strictly_q_only")
        if self.placement not in {
            AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
            AdapterPlacement.CROSS_ATTENTION_QUERY,
        }:
            raise KVCompatibilityError("adapter_intersects_kv_producer_path")


@dataclass(frozen=True)
class KVSegment:
    """Digest-only description of one ordered, contiguous cache segment."""

    payload_sha256: str
    payload_bytes: int
    token_ids_sha256: str
    position_ids_sha256: str
    position_start: int
    position_end: int
    inline_preview: bytes = b""

    def __post_init__(self) -> None:
        _require_sha256(self.payload_sha256, "payload_sha256")
        _require_sha256(self.token_ids_sha256, "token_ids_sha256")
        _require_sha256(self.position_ids_sha256, "position_ids_sha256")
        if (
            isinstance(self.payload_bytes, bool)
            or not isinstance(self.payload_bytes, int)
            or self.payload_bytes < 1
        ):
            raise ValueError("payload_bytes must be a positive integer")
        if (
            isinstance(self.position_start, bool)
            or isinstance(self.position_end, bool)
            or not isinstance(self.position_start, int)
            or not isinstance(self.position_end, int)
            or self.position_start < 0
            or self.position_end <= self.position_start
        ):
            raise ValueError("position span must be non-empty and non-negative")
        if not isinstance(self.inline_preview, bytes):
            raise ValueError("inline_preview must be bytes")
        if len(self.inline_preview) > self.payload_bytes:
            raise ValueError("inline_preview cannot exceed payload_bytes")


@dataclass(frozen=True)
class KVPageView:
    storage: Literal["shared", "private"]
    digest: str
    parent_digest: str | None
    compatibility_digest: str
    payload_sha256: str
    payload_bytes: int
    token_ids_sha256: str
    position_ids_sha256: str
    position_start: int
    position_end: int
    inline_preview: bytes


@dataclass(frozen=True)
class SharedPrefixHandle:
    capability: str
    task_id: str
    compatibility_digest: str
    terminal_page_digest: str | None
    page_count: int
    position_end: int


@dataclass(frozen=True)
class PrivateBranchHandle:
    capability: str
    branch_id: str
    task_id: str
    expert_id: str
    compatibility_digest: str
    version: int


@dataclass(frozen=True)
class CombinedKVView:
    task_id: str
    expert_id: str
    compatibility_digest: str
    branch_version: int
    shared_prefix_pages: tuple[KVPageView, ...]
    expert_private_pages: tuple[KVPageView, ...]

    @property
    def ordered_pages(self) -> tuple[KVPageView, ...]:
        return self.shared_prefix_pages + self.expert_private_pages


@dataclass(frozen=True)
class CommitReceipt:
    commit_sha256: str
    branch_id: str
    committed_version: int
    shared_prefix: SharedPrefixHandle


@dataclass(frozen=True)
class KVStoreStats:
    shared_pages: int
    private_pages: int
    inline_bytes: int
    shared_refcount_total: int
    private_refcount_total: int
    open_prefixes: int
    open_branches: int
    registered_identities: int
    invalidated_identities: int
    max_pages: int
    max_prefixes: int
    max_branches: int
    max_identities: int


@dataclass(frozen=True)
class _Page:
    storage: Literal["shared", "private"]
    digest: str
    parent_digest: str | None
    compatibility_digest: str
    payload_sha256: str
    payload_bytes: int
    token_ids_sha256: str
    position_ids_sha256: str
    position_start: int
    position_end: int
    inline_preview: bytes

    def view(self) -> KVPageView:
        return KVPageView(**self.__dict__)


@dataclass
class _PrefixState:
    capability: str
    task_id: str
    compatibility_digest: str
    pages: tuple[str, ...]
    position_end: int


@dataclass
class _BranchState:
    capability: str
    branch_id: str
    task_id: str
    expert_id: str
    compatibility_digest: str
    adapter: AdapterExecutionProfile
    shared_pages: tuple[str, ...]
    private_pages: tuple[str, ...]
    position_end: int
    version: int = 0
    committed: CommitReceipt | None = None


class HierarchicalTaskKVStore:
    """Bounded, capability-scoped, thread-safe Task-KV metadata store."""

    def __init__(
        self,
        *,
        max_pages: int = 1024,
        max_inline_bytes: int = 1 << 20,
        max_inline_bytes_per_page: int = 4096,
        max_prefixes: int = 1024,
        max_branches: int = 1024,
        max_identities: int = 256,
    ) -> None:
        for value, field in (
            (max_pages, "max_pages"),
            (max_inline_bytes, "max_inline_bytes"),
            (max_inline_bytes_per_page, "max_inline_bytes_per_page"),
            (max_prefixes, "max_prefixes"),
            (max_branches, "max_branches"),
            (max_identities, "max_identities"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if max_inline_bytes_per_page > max_inline_bytes:
            raise ValueError("per-page inline limit cannot exceed total inline limit")
        self._max_pages = max_pages
        self._max_inline_bytes = max_inline_bytes
        self._max_inline_bytes_per_page = max_inline_bytes_per_page
        self._max_prefixes = max_prefixes
        self._max_branches = max_branches
        self._max_identities = max_identities
        self._inline_bytes = 0
        self._shared_pages: dict[str, _Page] = {}
        self._private_pages: dict[str, _Page] = {}
        self._shared_refs: dict[str, int] = {}
        self._private_refs: dict[str, int] = {}
        self._prefixes: dict[str, _PrefixState] = {}
        self._branches: dict[str, _BranchState] = {}
        self._identities: dict[str, KVCompatibilityIdentity] = {}
        self._invalidated_identities: set[str] = set()
        self._lock = threading.RLock()

    @staticmethod
    def _capability() -> str:
        return secrets.token_hex(24)

    def _new_unique_capability(self) -> str:
        for _attempt in range(16):
            capability = self._capability()
            if capability not in self._prefixes and capability not in self._branches:
                return capability
        raise KVAuthorizationError("capability_collision")

    def _new_unique_branch_id(self) -> str:
        existing = {state.branch_id for state in self._branches.values()}
        for _attempt in range(16):
            branch_id = self._capability()
            if branch_id not in existing:
                return branch_id
        raise KVAuthorizationError("branch_id_collision")

    def _validate_identity_registration(
        self, identity: KVCompatibilityIdentity
    ) -> str:
        if not isinstance(identity, KVCompatibilityIdentity):
            raise ValueError("identity must be KVCompatibilityIdentity")
        digest = identity.digest
        existing = self._identities.get(digest)
        if existing is not None and existing != identity:
            raise KVCompatibilityError("compatibility_identity_collision")
        if digest in self._invalidated_identities:
            raise KVInvalidatedError("compatibility_identity_invalidated")
        if existing is None and len(self._identities) >= self._max_identities:
            raise KVCapacityError("identity_capacity_exceeded")
        return digest

    def _assert_prefix_capacity(self) -> None:
        if len(self._prefixes) >= self._max_prefixes:
            raise KVCapacityError("prefix_capacity_exceeded")

    def _assert_branch_capacity(self) -> None:
        if len(self._branches) >= self._max_branches:
            raise KVCapacityError("branch_capacity_exceeded")

    def _require_active_identity(self, digest: str) -> None:
        if digest not in self._identities:
            raise KVCompatibilityError("unknown_compatibility_identity")
        if digest in self._invalidated_identities:
            raise KVInvalidatedError("compatibility_identity_invalidated")

    def _page_digest(
        self,
        *,
        storage: str,
        compatibility_digest: str,
        parent_digest: str | None,
        segment: KVSegment,
        task_id: str | None = None,
        expert_id: str | None = None,
    ) -> str:
        return _canonical_sha256(
            {
                "storage": storage,
                "compatibility_digest": compatibility_digest,
                "ordered_parent_digest": parent_digest,
                "payload_sha256": segment.payload_sha256,
                "payload_bytes": segment.payload_bytes,
                "token_ids_sha256": segment.token_ids_sha256,
                "position_ids_sha256": segment.position_ids_sha256,
                "position_start": segment.position_start,
                "position_end": segment.position_end,
                "task_id": task_id,
                "expert_id": expert_id,
            }
        )

    def _make_page(
        self,
        *,
        storage: Literal["shared", "private"],
        compatibility_digest: str,
        parent_digest: str | None,
        segment: KVSegment,
        task_id: str | None = None,
        expert_id: str | None = None,
    ) -> _Page:
        if len(segment.inline_preview) > self._max_inline_bytes_per_page:
            raise KVCapacityError("inline_preview_exceeds_per_page_limit")
        digest = self._page_digest(
            storage=storage,
            compatibility_digest=compatibility_digest,
            parent_digest=parent_digest,
            segment=segment,
            task_id=task_id,
            expert_id=expert_id,
        )
        return _Page(
            storage=storage,
            digest=digest,
            parent_digest=parent_digest,
            compatibility_digest=compatibility_digest,
            payload_sha256=segment.payload_sha256,
            payload_bytes=segment.payload_bytes,
            token_ids_sha256=segment.token_ids_sha256,
            position_ids_sha256=segment.position_ids_sha256,
            position_start=segment.position_start,
            position_end=segment.position_end,
            inline_preview=segment.inline_preview,
        )

    def _assert_capacity_for(self, pages: Iterable[_Page]) -> None:
        unique_new: dict[tuple[str, str], _Page] = {}
        for page in pages:
            collection = (
                self._shared_pages if page.storage == "shared" else self._private_pages
            )
            existing = collection.get(page.digest)
            if existing is not None:
                if existing != page:
                    raise KVCompatibilityError("content_address_collision")
                continue
            unique_new[(page.storage, page.digest)] = page
        if (
            len(self._shared_pages)
            + len(self._private_pages)
            + len(unique_new)
            > self._max_pages
        ):
            raise KVCapacityError("page_capacity_exceeded")
        added_inline = sum(len(page.inline_preview) for page in unique_new.values())
        if self._inline_bytes + added_inline > self._max_inline_bytes:
            raise KVCapacityError("inline_byte_capacity_exceeded")

    def _insert_pages(self, pages: Iterable[_Page]) -> None:
        for page in pages:
            collection = (
                self._shared_pages if page.storage == "shared" else self._private_pages
            )
            refs = self._shared_refs if page.storage == "shared" else self._private_refs
            if page.digest not in collection:
                collection[page.digest] = page
                refs[page.digest] = 0
                self._inline_bytes += len(page.inline_preview)

    def _add_refs(self, storage: str, digests: Iterable[str]) -> None:
        refs = self._shared_refs if storage == "shared" else self._private_refs
        for digest in digests:
            refs[digest] += 1

    def _drop_refs(self, storage: str, digests: Iterable[str]) -> None:
        refs = self._shared_refs if storage == "shared" else self._private_refs
        pages = self._shared_pages if storage == "shared" else self._private_pages
        for digest in digests:
            count = refs[digest] - 1
            if count < 0:
                raise AssertionError("negative page refcount")
            if count == 0:
                page = pages.pop(digest)
                refs.pop(digest)
                self._inline_bytes -= len(page.inline_preview)
            else:
                refs[digest] = count

    def _new_prefix(
        self,
        *,
        capability: str,
        task_id: str,
        compatibility_digest: str,
        pages: tuple[str, ...],
        position_end: int,
    ) -> SharedPrefixHandle:
        self._assert_prefix_capacity()
        if capability in self._prefixes or capability in self._branches:
            raise KVAuthorizationError("capability_collision")
        state = _PrefixState(
            capability=capability,
            task_id=task_id,
            compatibility_digest=compatibility_digest,
            pages=pages,
            position_end=position_end,
        )
        self._prefixes[capability] = state
        self._add_refs("shared", pages)
        return SharedPrefixHandle(
            capability=capability,
            task_id=task_id,
            compatibility_digest=compatibility_digest,
            terminal_page_digest=pages[-1] if pages else None,
            page_count=len(pages),
            position_end=position_end,
        )

    def _require_prefix(
        self,
        task_id: str,
        handle: SharedPrefixHandle,
        *,
        require_active: bool = True,
    ) -> _PrefixState:
        _require_identifier(task_id, "task_id")
        if not isinstance(handle, SharedPrefixHandle):
            raise KVAuthorizationError("invalid_prefix_capability")
        state = self._prefixes.get(handle.capability)
        if state is None or state.task_id != task_id or handle.task_id != task_id:
            raise KVAuthorizationError("prefix_scope_mismatch")
        expected = SharedPrefixHandle(
            capability=state.capability,
            task_id=state.task_id,
            compatibility_digest=state.compatibility_digest,
            terminal_page_digest=state.pages[-1] if state.pages else None,
            page_count=len(state.pages),
            position_end=state.position_end,
        )
        if handle != expected:
            raise KVAuthorizationError("forged_prefix_capability")
        if require_active:
            self._require_active_identity(state.compatibility_digest)
        return state

    def _require_branch(
        self,
        task_id: str,
        expert_id: str,
        handle: PrivateBranchHandle,
        *,
        require_active: bool = True,
        require_current_version: bool = True,
    ) -> _BranchState:
        _require_identifier(task_id, "task_id")
        _require_identifier(expert_id, "expert_id")
        if not isinstance(handle, PrivateBranchHandle):
            raise KVAuthorizationError("invalid_branch_capability")
        state = self._branches.get(handle.capability)
        if (
            state is None
            or state.task_id != task_id
            or state.expert_id != expert_id
            or handle.task_id != task_id
            or handle.expert_id != expert_id
        ):
            raise KVAuthorizationError("branch_scope_mismatch")
        if (
            handle.branch_id != state.branch_id
            or handle.compatibility_digest != state.compatibility_digest
        ):
            raise KVAuthorizationError("forged_branch_capability")
        if require_current_version and handle.version != state.version:
            raise KVVersionError("stale_branch_version")
        if require_active:
            self._require_active_identity(state.compatibility_digest)
        return state

    @staticmethod
    def _branch_handle(state: _BranchState) -> PrivateBranchHandle:
        return PrivateBranchHandle(
            capability=state.capability,
            branch_id=state.branch_id,
            task_id=state.task_id,
            expert_id=state.expert_id,
            compatibility_digest=state.compatibility_digest,
            version=state.version,
        )

    def publish_seed_prefix(
        self,
        task_id: str,
        identity: KVCompatibilityIdentity,
        segments: Iterable[KVSegment],
    ) -> SharedPrefixHandle:
        """Publish an already-committed base/fact prefix in strict token order."""

        task_id = _require_identifier(task_id, "task_id")
        ordered = tuple(segments)
        if not ordered:
            raise ValueError("seed prefix requires at least one segment")
        with self._lock:
            compatibility_digest = self._validate_identity_registration(identity)
            self._assert_prefix_capacity()
            prefix_capability = self._new_unique_capability()
            pages: list[_Page] = []
            parent: str | None = None
            expected_start = 0
            for segment in ordered:
                if not isinstance(segment, KVSegment):
                    raise ValueError("segments must contain KVSegment values")
                if segment.position_start != expected_start:
                    raise KVCompatibilityError("non_contiguous_prefix_lineage")
                page = self._make_page(
                    storage="shared",
                    compatibility_digest=compatibility_digest,
                    parent_digest=parent,
                    segment=segment,
                )
                pages.append(page)
                parent = page.digest
                expected_start = segment.position_end
            self._assert_capacity_for(pages)
            self._identities.setdefault(compatibility_digest, identity)
            self._insert_pages(pages)
            return self._new_prefix(
                capability=prefix_capability,
                task_id=task_id,
                compatibility_digest=compatibility_digest,
                pages=tuple(page.digest for page in pages),
                position_end=expected_start,
            )

    def open_private_branch(
        self,
        task_id: str,
        expert_id: str,
        prefix: SharedPrefixHandle,
        adapter: AdapterExecutionProfile,
    ) -> PrivateBranchHandle:
        """Open an expert-private COW branch from a committed shared prefix."""

        task_id = _require_identifier(task_id, "task_id")
        expert_id = _require_identifier(expert_id, "expert_id")
        if not isinstance(adapter, AdapterExecutionProfile):
            raise KVCompatibilityError("invalid_adapter_profile")
        adapter.assert_share_compatible()
        with self._lock:
            prefix_state = self._require_prefix(task_id, prefix)
            self._assert_branch_capacity()
            capability = self._new_unique_capability()
            state = _BranchState(
                capability=capability,
                branch_id=self._new_unique_branch_id(),
                task_id=task_id,
                expert_id=expert_id,
                compatibility_digest=prefix_state.compatibility_digest,
                adapter=adapter,
                shared_pages=prefix_state.pages,
                private_pages=(),
                position_end=prefix_state.position_end,
            )
            self._branches[capability] = state
            self._add_refs("shared", state.shared_pages)
            return self._branch_handle(state)

    def append_private(
        self,
        task_id: str,
        expert_id: str,
        branch: PrivateBranchHandle,
        segment: KVSegment,
    ) -> PrivateBranchHandle:
        """Append one contiguous private page and return the new branch version."""

        if not isinstance(segment, KVSegment):
            raise ValueError("segment must be KVSegment")
        with self._lock:
            state = self._require_branch(task_id, expert_id, branch)
            if state.committed is not None:
                raise KVVersionError("branch_already_committed")
            if segment.position_start != state.position_end:
                raise KVCompatibilityError("non_contiguous_private_lineage")
            parent = (
                state.private_pages[-1]
                if state.private_pages
                else (state.shared_pages[-1] if state.shared_pages else None)
            )
            page = self._make_page(
                storage="private",
                compatibility_digest=state.compatibility_digest,
                parent_digest=parent,
                segment=segment,
                task_id=state.task_id,
                expert_id=state.expert_id,
            )
            self._assert_capacity_for((page,))
            self._insert_pages((page,))
            self._add_refs("private", (page.digest,))
            state.private_pages = (*state.private_pages, page.digest)
            state.position_end = segment.position_end
            state.version += 1
            return self._branch_handle(state)

    def fork_private_branch(
        self,
        task_id: str,
        expert_id: str,
        branch: PrivateBranchHandle,
    ) -> PrivateBranchHandle:
        """Fork within the same expert scope; existing pages remain immutable."""

        with self._lock:
            source = self._require_branch(task_id, expert_id, branch)
            if source.committed is not None:
                raise KVVersionError("branch_already_committed")
            self._assert_branch_capacity()
            capability = self._new_unique_capability()
            fork = _BranchState(
                capability=capability,
                branch_id=self._new_unique_branch_id(),
                task_id=source.task_id,
                expert_id=source.expert_id,
                compatibility_digest=source.compatibility_digest,
                adapter=source.adapter,
                shared_pages=source.shared_pages,
                private_pages=source.private_pages,
                position_end=source.position_end,
            )
            self._branches[capability] = fork
            self._add_refs("shared", fork.shared_pages)
            self._add_refs("private", fork.private_pages)
            return self._branch_handle(fork)

    def combined_view(
        self,
        task_id: str,
        expert_id: str,
        branch: PrivateBranchHandle,
    ) -> CombinedKVView:
        """Return an immutable prefix+private view in exact causal order."""

        with self._lock:
            state = self._require_branch(task_id, expert_id, branch)
            return CombinedKVView(
                task_id=state.task_id,
                expert_id=state.expert_id,
                compatibility_digest=state.compatibility_digest,
                branch_version=state.version,
                shared_prefix_pages=tuple(
                    self._shared_pages[digest].view() for digest in state.shared_pages
                ),
                expert_private_pages=tuple(
                    self._private_pages[digest].view() for digest in state.private_pages
                ),
            )

    def commit_branch(
        self,
        task_id: str,
        expert_id: str,
        branch: PrivateBranchHandle,
    ) -> CommitReceipt:
        """Atomically publish one private branch for same-task downstream use."""

        with self._lock:
            state = self._require_branch(task_id, expert_id, branch)
            if state.committed is not None:
                return state.committed
            self._assert_prefix_capacity()
            prefix_capability = self._new_unique_capability()
            parent = state.shared_pages[-1] if state.shared_pages else None
            pages: list[_Page] = []
            for private_digest in state.private_pages:
                private = self._private_pages[private_digest]
                segment = KVSegment(
                    payload_sha256=private.payload_sha256,
                    payload_bytes=private.payload_bytes,
                    token_ids_sha256=private.token_ids_sha256,
                    position_ids_sha256=private.position_ids_sha256,
                    position_start=private.position_start,
                    position_end=private.position_end,
                    inline_preview=private.inline_preview,
                )
                shared = self._make_page(
                    storage="shared",
                    compatibility_digest=state.compatibility_digest,
                    parent_digest=parent,
                    segment=segment,
                )
                pages.append(shared)
                parent = shared.digest
            self._assert_capacity_for(pages)
            self._insert_pages(pages)
            prefix = self._new_prefix(
                capability=prefix_capability,
                task_id=state.task_id,
                compatibility_digest=state.compatibility_digest,
                pages=(*state.shared_pages, *(page.digest for page in pages)),
                position_end=state.position_end,
            )
            receipt = CommitReceipt(
                commit_sha256=_canonical_sha256(
                    {
                        "branch_id": state.branch_id,
                        "committed_version": state.version,
                        "task_id": state.task_id,
                        "expert_id": state.expert_id,
                        "compatibility_digest": state.compatibility_digest,
                        "shared_terminal": prefix.terminal_page_digest,
                    }
                ),
                branch_id=state.branch_id,
                committed_version=state.version,
                shared_prefix=prefix,
            )
            state.committed = receipt
            return receipt

    def close_branch(
        self,
        task_id: str,
        expert_id: str,
        branch: PrivateBranchHandle,
    ) -> None:
        """Release one branch capability and its page references."""

        with self._lock:
            state = self._require_branch(
                task_id,
                expert_id,
                branch,
                require_active=False,
                require_current_version=False,
            )
            self._branches.pop(state.capability)
            self._drop_refs("shared", state.shared_pages)
            self._drop_refs("private", state.private_pages)

    def close_prefix(self, task_id: str, prefix: SharedPrefixHandle) -> None:
        """Release one task-scoped prefix capability."""

        with self._lock:
            state = self._require_prefix(task_id, prefix, require_active=False)
            self._prefixes.pop(state.capability)
            self._drop_refs("shared", state.pages)

    def prefix_identity(
        self, task_id: str, prefix: SharedPrefixHandle
    ) -> KVCompatibilityIdentity:
        """Return the authenticated producer identity for one prefix capability."""

        with self._lock:
            state = self._require_prefix(task_id, prefix)
            return self._identities[state.compatibility_digest]

    def prefix_page_digests(
        self, task_id: str, prefix: SharedPrefixHandle
    ) -> tuple[str, ...]:
        """Return the authenticated ordered page lineage for a prefix."""

        with self._lock:
            state = self._require_prefix(task_id, prefix)
            return state.pages

    def invalidate_identity(self, identity: KVCompatibilityIdentity) -> None:
        """Invalidate all capabilities tied to an obsolete K/V producer epoch."""

        if not isinstance(identity, KVCompatibilityIdentity):
            raise ValueError("identity must be KVCompatibilityIdentity")
        with self._lock:
            digest = identity.digest
            existing = self._identities.get(digest)
            if existing is None or existing != identity:
                raise KVCompatibilityError("unknown_compatibility_identity")
            self._invalidated_identities.add(digest)

    def stats(self) -> KVStoreStats:
        with self._lock:
            return KVStoreStats(
                shared_pages=len(self._shared_pages),
                private_pages=len(self._private_pages),
                inline_bytes=self._inline_bytes,
                shared_refcount_total=sum(self._shared_refs.values()),
                private_refcount_total=sum(self._private_refs.values()),
                open_prefixes=len(self._prefixes),
                open_branches=len(self._branches),
                registered_identities=len(self._identities),
                invalidated_identities=len(self._invalidated_identities),
                max_pages=self._max_pages,
                max_prefixes=self._max_prefixes,
                max_branches=self._max_branches,
                max_identities=self._max_identities,
            )


__all__ = [
    "AdapterExecutionProfile",
    "AdapterPlacement",
    "CombinedKVView",
    "CommitReceipt",
    "HierarchicalKVError",
    "HierarchicalTaskKVStore",
    "KVAuthorizationError",
    "KVCapacityError",
    "KVCompatibilityError",
    "KVCompatibilityIdentity",
    "KVProducerExecutionMode",
    "KVInvalidatedError",
    "KVPageView",
    "KVSegment",
    "KVStoreStats",
    "KVVersionError",
    "PrivateBranchHandle",
    "SharedPrefixHandle",
]
