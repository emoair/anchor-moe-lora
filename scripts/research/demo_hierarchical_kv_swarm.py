#!/usr/bin/env python3
"""Run a tiny, model-free Hierarchical Task-KV + multi-stream smoke test."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from anchor_mvp.research.hierarchical_kv import (
    AdapterExecutionProfile,
    AdapterPlacement,
    HierarchicalTaskKVStore,
    KVCompatibilityIdentity,
    KVProducerExecutionMode,
    KVSegment,
)
from anchor_mvp.research.neural_swarm_kv_runtime import (
    HierarchicalKVBackendAdapter,
    InMemoryHierarchicalKVContextProvider,
    KVRuntimeContext,
    TrustedExactKVBinding,
)
from anchor_mvp.research.neural_swarm_streaming import (
    BackendChunk,
    ExpertBinding,
    NeuralSwarmStreamController,
    SwarmRequest,
    collect_swarm_events,
    summarize_swarm_events,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _SyntheticKVBackend:
    def __init__(self) -> None:
        self.prefix_chains: set[str] = set()
        self.private_branches: set[str] = set()

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
        del shared_input, run_id, task_bundle_sha256, cancel_event
        self.prefix_chains.add(kv_context.ordered_prefix_chain_sha256)
        self.private_branches.add(kv_context.private_branch_id)
        yield BackendChunk(
            delta=f"synthetic:{binding.expert_id}",
            metadata={
                "sharing_mode": kv_context.sharing_mode.value,
                "shared_page_count": len(kv_context.shared_page_ids),
            },
        )


async def _run() -> dict[str, Any]:
    task_bundle = _sha("hierarchical-kv-demo-bundle")
    store = HierarchicalTaskKVStore(
        max_pages=16,
        max_inline_bytes=256,
        max_inline_bytes_per_page=64,
    )
    identity = KVCompatibilityIdentity(
        model_id="synthetic-frozen-fact-producer",
        model_revision_sha256=_sha("model-revision"),
        tokenizer_sha256=_sha("tokenizer"),
        rope_config_sha256=_sha("rope"),
        kv_layout_sha256=_sha("kv-layout"),
        kv_producer_path_sha256=_sha("decoupled-producer-path"),
        base_kv_weights_sha256=_sha("base-kv-weights"),
        base_kv_weights_epoch=0,
        producer_execution_mode=KVProducerExecutionMode.DECOUPLED_FROZEN_PREFIX,
    )
    prefix = store.publish_seed_prefix(
        task_bundle,
        identity,
        (
            KVSegment(
                payload_sha256=_sha("synthetic-prefix-payload"),
                payload_bytes=64,
                token_ids_sha256=_sha("synthetic-prefix-token-ids"),
                position_ids_sha256=_sha("synthetic-prefix-positions"),
                position_start=0,
                position_end=8,
            ),
        ),
    )
    bindings = (
        ExpertBinding("anchor-swarm/planner", "planner", "planner-backend"),
        ExpertBinding("anchor-swarm/reviewer", "reviewer", "reviewer-backend"),
    )
    adapters = {
        binding.expert_id: AdapterExecutionProfile(
            adapter_id=f"synthetic:{binding.expert_id}",
            adapter_revision_sha256=_sha(f"adapter:{binding.expert_id}"),
            target_projections=frozenset({"q"}),
            placement=AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
        )
        for binding in bindings
    }
    backend = _SyntheticKVBackend()
    controller = NeuralSwarmStreamController(
        bindings=bindings,
        backend=HierarchicalKVBackendAdapter(
            context_provider=InMemoryHierarchicalKVContextProvider(
                store=store,
                prefixes={task_bundle: prefix},
                adapters=adapters,
            ),
            backend=backend,
            trusted_exact_bindings={
                task_bundle: TrustedExactKVBinding.from_store(
                    store=store,
                    task_bundle_sha256=task_bundle,
                    prefix=prefix,
                )
            },
        ),
        max_concurrency=2,
        queue_capacity=8,
    )
    events = await collect_swarm_events(
        controller,
        SwarmRequest(
            run_id="hierarchical-kv-smoke",
            task_bundle_sha256=task_bundle,
            request_model_ids=tuple(
                binding.request_model_id for binding in bindings
            ),
            shared_input={"messages": [{"role": "user", "content": "synthetic"}]},
        ),
    )
    summary = summarize_swarm_events(events)
    before_prefix_close = store.stats()
    store.close_prefix(task_bundle, prefix)
    after_prefix_close = store.stats()
    return {
        "ok": True,
        "claim_scope": "metadata_only_smoke",
        "foundation_model_loaded": False,
        "provider_requests": 0,
        "streams": summary.total_streams,
        "completed_streams": summary.completed_streams,
        "unique_shared_prefix_chains": len(backend.prefix_chains),
        "unique_private_branches": len(backend.private_branches),
        "open_branches_after_run": before_prefix_close.open_branches,
        "shared_pages_after_prefix_close": after_prefix_close.shared_pages,
        "non_claims": [
            "real_kv_tensor_reuse",
            "cuda_overlap",
            "throughput_or_quality_improvement",
            "naive_in_stack_q_lora_exactness",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
