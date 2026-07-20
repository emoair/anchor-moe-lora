from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import threading

import pytest

from anchor_mvp.research.hierarchical_kv import (
    AdapterExecutionProfile,
    AdapterPlacement,
    HierarchicalTaskKVStore,
    KVAuthorizationError,
    KVCapacityError,
    KVCompatibilityError,
    KVCompatibilityIdentity,
    KVInvalidatedError,
    KVProducerExecutionMode,
    KVSegment,
    KVVersionError,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(
    *,
    epoch: int = 1,
    producer_mode: KVProducerExecutionMode = (
        KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX
    ),
) -> KVCompatibilityIdentity:
    return KVCompatibilityIdentity(
        model_id="fixture-model",
        model_revision_sha256=_digest("model-revision"),
        tokenizer_sha256=_digest("tokenizer"),
        rope_config_sha256=_digest("rope"),
        kv_layout_sha256=_digest("kv-layout"),
        kv_producer_path_sha256=_digest("frozen-fact-producer"),
        base_kv_weights_sha256=_digest("base-kv-weights"),
        base_kv_weights_epoch=epoch,
        producer_execution_mode=producer_mode,
    )


def _adapter(
    *,
    placement: AdapterPlacement = AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
    targets: frozenset[str] = frozenset({"q"}),
) -> AdapterExecutionProfile:
    return AdapterExecutionProfile(
        adapter_id="planner",
        adapter_revision_sha256=_digest("planner-r1"),
        target_projections=targets,
        placement=placement,
    )


def _segment(start: int, end: int, label: str, *, preview: bytes = b"kv") -> KVSegment:
    return KVSegment(
        payload_sha256=_digest(f"payload:{label}"),
        payload_bytes=10_000_000,
        token_ids_sha256=_digest(f"tokens:{label}:{start}:{end}"),
        position_ids_sha256=_digest(f"positions:{start}:{end}"),
        position_start=start,
        position_end=end,
        inline_preview=preview,
    )


def _seed(
    store: HierarchicalTaskKVStore,
    *,
    task_id: str = "task-a",
    identity: KVCompatibilityIdentity | None = None,
):
    return store.publish_seed_prefix(
        task_id,
        identity or _identity(),
        (_segment(0, 4, "base-0"), _segment(4, 8, "base-1")),
    )


def test_shared_prefix_is_content_addressed_immutable_and_task_scoped() -> None:
    store = HierarchicalTaskKVStore(max_pages=16)
    first = _seed(store)
    second = _seed(store)
    other_task = _seed(store, task_id="task-b")

    assert first.capability != second.capability
    assert first.terminal_page_digest == second.terminal_page_digest
    assert other_task.terminal_page_digest == first.terminal_page_digest
    assert store.stats().shared_pages == 2
    assert store.stats().shared_refcount_total == 6

    with pytest.raises(KVAuthorizationError, match="prefix_scope_mismatch"):
        store.open_private_branch("task-b", "planner", first, _adapter())
    with pytest.raises(KVAuthorizationError, match="prefix_scope_mismatch"):
        store.open_private_branch("task-a", "planner", other_task, _adapter())


def test_q_only_is_not_sufficient_inside_a_decoder_kv_producer_path() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)

    with pytest.raises(
        KVCompatibilityError, match="adapter_intersects_kv_producer_path"
    ):
        store.open_private_branch(
            "task-a",
            "planner",
            prefix,
            _adapter(placement=AdapterPlacement.NAIVE_DECODER_IN_STACK),
        )
    for targets in (frozenset({"k"}), frozenset({"v"}), frozenset({"q", "v"})):
        with pytest.raises(KVCompatibilityError, match="adapter_not_strictly_q_only"):
            store.open_private_branch(
                "task-a", "planner", prefix, _adapter(targets=targets)
            )

    cross_attention = store.open_private_branch(
        "task-a",
        "planner",
        prefix,
        _adapter(placement=AdapterPlacement.CROSS_ATTENTION_QUERY),
    )
    assert cross_attention.compatibility_digest == _identity().digest


def test_private_delta_requires_commit_before_another_expert_can_share_it() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)
    builder = store.open_private_branch(
        "task-a", "frontend_gen", prefix, _adapter()
    )
    builder = store.append_private(
        "task-a", "frontend_gen", builder, _segment(8, 12, "builder")
    )

    reviewer_before = store.open_private_branch(
        "task-a", "frontend_review", prefix, _adapter()
    )
    assert len(store.combined_view("task-a", "frontend_review", reviewer_before).ordered_pages) == 2
    with pytest.raises(KVAuthorizationError, match="branch_scope_mismatch"):
        store.combined_view("task-a", "frontend_review", builder)

    receipt = store.commit_branch("task-a", "frontend_gen", builder)
    reviewer_after = store.open_private_branch(
        "task-a", "frontend_review", receipt.shared_prefix, _adapter()
    )
    view = store.combined_view("task-a", "frontend_review", reviewer_after)

    assert len(view.shared_prefix_pages) == 3
    assert not view.expert_private_pages
    assert [page.position_start for page in view.ordered_pages] == [0, 4, 8]
    assert all(page.storage == "shared" for page in view.ordered_pages)
    with pytest.raises(KVVersionError, match="branch_already_committed"):
        store.append_private(
            "task-a", "frontend_gen", builder, _segment(12, 16, "too-late")
        )


def test_copy_on_write_fork_preserves_source_and_refcounts() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)
    source = store.open_private_branch("task-a", "planner", prefix, _adapter())
    source = store.append_private(
        "task-a", "planner", source, _segment(8, 12, "shared-private")
    )
    fork = store.fork_private_branch("task-a", "planner", source)

    assert store.stats().private_pages == 1
    assert store.stats().private_refcount_total == 2
    fork = store.append_private(
        "task-a", "planner", fork, _segment(12, 16, "fork-only")
    )
    source_view = store.combined_view("task-a", "planner", source)
    fork_view = store.combined_view("task-a", "planner", fork)

    assert len(source_view.expert_private_pages) == 1
    assert len(fork_view.expert_private_pages) == 2
    assert source_view.expert_private_pages[0] == fork_view.expert_private_pages[0]
    assert store.stats().private_pages == 2
    assert store.stats().private_refcount_total == 3

    store.close_branch("task-a", "planner", fork)
    assert store.stats().private_pages == 1
    assert store.stats().private_refcount_total == 1


def test_commit_is_idempotent_under_concurrency() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)
    branch = store.open_private_branch("task-a", "planner", prefix, _adapter())
    branch = store.append_private(
        "task-a", "planner", branch, _segment(8, 12, "commit-once")
    )
    barrier = threading.Barrier(8)

    def commit():
        barrier.wait()
        return store.commit_branch("task-a", "planner", branch)

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _index: commit(), range(8)))

    assert len(set(receipts)) == 1
    assert store.stats().open_prefixes == 2
    assert store.stats().shared_pages == 3


def test_stale_version_and_concurrent_append_fail_closed() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)
    old = store.open_private_branch("task-a", "planner", prefix, _adapter())
    current = store.append_private(
        "task-a", "planner", old, _segment(8, 12, "version-one")
    )
    assert current.version == 1
    with pytest.raises(KVVersionError, match="stale_branch_version"):
        store.combined_view("task-a", "planner", old)

    branch = store.open_private_branch("task-a", "tool_policy", prefix, _adapter())
    barrier = threading.Barrier(2)

    def append_once():
        barrier.wait()
        try:
            return store.append_private(
                "task-a", "tool_policy", branch, _segment(8, 12, "race")
            )
        except KVVersionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: append_once(), range(2)))
    assert sum(isinstance(item, KVVersionError) for item in outcomes) == 1
    assert sum(not isinstance(item, Exception) for item in outcomes) == 1


def test_prefix_lineage_binds_order_parent_and_position_spans() -> None:
    store = HierarchicalTaskKVStore()
    with pytest.raises(KVCompatibilityError, match="non_contiguous_prefix_lineage"):
        store.publish_seed_prefix(
            "task-a", _identity(), (_segment(4, 8, "not-root"),)
        )
    with pytest.raises(KVCompatibilityError, match="non_contiguous_prefix_lineage"):
        store.publish_seed_prefix(
            "task-a",
            _identity(),
            (_segment(0, 4, "first"), _segment(5, 8, "gap")),
        )

    first = store.publish_seed_prefix(
        "task-a",
        _identity(),
        (_segment(0, 4, "same-a"), _segment(4, 8, "same-b")),
    )
    reordered = store.publish_seed_prefix(
        "task-a",
        _identity(),
        (_segment(0, 4, "same-b"), _segment(4, 8, "same-a")),
    )
    assert first.terminal_page_digest != reordered.terminal_page_digest

    branch = store.open_private_branch("task-a", "planner", first, _adapter())
    with pytest.raises(KVCompatibilityError, match="non_contiguous_private_lineage"):
        store.append_private(
            "task-a", "planner", branch, _segment(9, 12, "private-gap")
        )


def test_compatibility_epoch_invalidation_is_strict_but_cleanup_still_works() -> None:
    store = HierarchicalTaskKVStore()
    identity = _identity()
    prefix = _seed(store, identity=identity)
    branch = store.open_private_branch("task-a", "planner", prefix, _adapter())
    store.invalidate_identity(identity)

    with pytest.raises(KVInvalidatedError, match="compatibility_identity_invalidated"):
        store.combined_view("task-a", "planner", branch)
    with pytest.raises(KVInvalidatedError, match="compatibility_identity_invalidated"):
        store.publish_seed_prefix(
            "task-a", identity, (_segment(0, 2, "invalidated"),)
        )
    with pytest.raises(KVInvalidatedError, match="compatibility_identity_invalidated"):
        store.open_private_branch("task-a", "planner", prefix, _adapter())

    store.close_branch("task-a", "planner", branch)
    store.close_prefix("task-a", prefix)
    assert store.stats().shared_pages == 0

    newer = _identity(epoch=2)
    newer_prefix = _seed(store, identity=newer)
    assert newer_prefix.compatibility_digest != identity.digest


def test_cross_expert_and_forged_capabilities_fail_closed() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)
    branch = store.open_private_branch("task-a", "planner", prefix, _adapter())

    for operation in (
        store.combined_view,
        store.fork_private_branch,
        store.commit_branch,
        store.close_branch,
    ):
        with pytest.raises(KVAuthorizationError, match="branch_scope_mismatch"):
            operation("task-a", "security_gate", branch)

    forged_branch = replace(branch, task_id="task-b")
    with pytest.raises(KVAuthorizationError, match="branch_scope_mismatch"):
        store.combined_view("task-b", "planner", forged_branch)
    forged_prefix = replace(prefix, position_end=999)
    with pytest.raises(KVAuthorizationError, match="forged_prefix_capability"):
        store.open_private_branch("task-a", "planner", forged_prefix, _adapter())


def test_large_payloads_store_only_bounded_preview_and_capacity_is_hard() -> None:
    store = HierarchicalTaskKVStore(
        max_pages=3,
        max_inline_bytes=6,
        max_inline_bytes_per_page=2,
    )
    prefix = store.publish_seed_prefix(
        "task-a",
        _identity(),
        (
            _segment(0, 4, "huge-a", preview=b"aa"),
            _segment(4, 8, "huge-b", preview=b"bb"),
        ),
    )
    assert store.stats().inline_bytes == 4
    branch = store.open_private_branch("task-a", "planner", prefix, _adapter())
    branch = store.append_private(
        "task-a", "planner", branch, _segment(8, 12, "huge-c", preview=b"cc")
    )
    assert store.stats().inline_bytes == 6
    with pytest.raises(KVCapacityError, match="page_capacity_exceeded"):
        store.append_private(
            "task-a", "planner", branch, _segment(12, 16, "over", preview=b"")
        )

    per_page = HierarchicalTaskKVStore(
        max_pages=2,
        max_inline_bytes=4,
        max_inline_bytes_per_page=1,
    )
    with pytest.raises(KVCapacityError, match="inline_preview_exceeds_per_page_limit"):
        per_page.publish_seed_prefix(
            "task-a", _identity(), (_segment(0, 4, "preview", preview=b"xx"),)
        )


def test_capability_and_identity_registries_are_hard_bounded() -> None:
    store = HierarchicalTaskKVStore(
        max_pages=8,
        max_prefixes=1,
        max_branches=1,
        max_identities=1,
    )
    prefix = _seed(store)
    assert store.stats().registered_identities == 1

    with pytest.raises(KVCapacityError, match="prefix_capacity_exceeded"):
        _seed(store)
    assert store.stats().open_prefixes == 1

    branch = store.open_private_branch("task-a", "planner", prefix, _adapter())
    with pytest.raises(KVCapacityError, match="branch_capacity_exceeded"):
        store.open_private_branch("task-a", "review", prefix, _adapter())
    assert store.stats().open_branches == 1

    store.close_branch("task-a", "planner", branch)
    store.close_prefix("task-a", prefix)
    with pytest.raises(KVCapacityError, match="identity_capacity_exceeded"):
        _seed(store, identity=_identity(epoch=2))
    stats = store.stats()
    assert stats.registered_identities == 1
    assert stats.open_prefixes == 0
    assert stats.shared_pages == 0


def test_stale_authentic_branch_handle_can_still_release_resources() -> None:
    store = HierarchicalTaskKVStore()
    prefix = _seed(store)
    original = store.open_private_branch("task-a", "planner", prefix, _adapter())
    current = store.append_private(
        "task-a", "planner", original, _segment(8, 12, "new-version")
    )

    with pytest.raises(KVVersionError, match="stale_branch_version"):
        store.combined_view("task-a", "planner", original)
    store.close_branch("task-a", "planner", original)

    assert store.stats().open_branches == 0
    assert store.stats().private_pages == 0
    with pytest.raises(KVAuthorizationError, match="branch_scope_mismatch"):
        store.combined_view("task-a", "planner", current)


def test_prefix_identity_is_capability_scoped_and_preserves_producer_mode() -> None:
    store = HierarchicalTaskKVStore()
    identity = _identity(
        producer_mode=KVProducerExecutionMode.CROSS_ATTENTION_MEMORY
    )
    prefix = _seed(store, identity=identity)

    assert store.prefix_identity("task-a", prefix) == identity
    with pytest.raises(KVAuthorizationError, match="prefix_scope_mismatch"):
        store.prefix_identity("task-b", prefix)


def test_capability_rng_collision_fails_before_inserting_pages(monkeypatch) -> None:
    store = HierarchicalTaskKVStore(max_pages=8, max_prefixes=4)
    monkeypatch.setattr(store, "_capability", lambda: "fixed-capability")
    _ = _seed(store)
    before = store.stats()

    with pytest.raises(KVAuthorizationError, match="capability_collision"):
        store.publish_seed_prefix(
            "task-b",
            _identity(),
            (_segment(0, 4, "new-a"), _segment(4, 8, "new-b")),
        )

    after = store.stats()
    assert after.open_prefixes == before.open_prefixes == 1
    assert after.shared_pages == before.shared_pages == 2
    assert after.shared_refcount_total == before.shared_refcount_total == 2
