# RFC: hierarchical planner, activation alignment, and read-only RDMA handoff

Status: frozen contract-only research red zone
Version: `anchor.neural-swarm-hierarchical-planner-rdma.v1`

## Purpose

This additive architecture decomposes a request into a fine-grained causal DAG,
routes each subtask through a sparse planner chain, and lets exactly one final
output expert decode. It does not modify the frozen 4,300-row dataset and grants
no training, quality, formal, deployment, zero-copy, or RDMA authority.

The execution order is immutable:

1. **P0** — one adapter-off frozen-base prefix.
2. **P1** — one general-planner commit classifying work versus chat and the
   relevant tool, emotion, or review requirements.
3. **P2** — one to three selected specialist-planner commits, in task-DAG
   topological order. The fixed specialists are mode, tool, and emotion.
4. **P3** — a content-addressed activation-group commit for the selected chain.
5. **P4** — a single output expert receives only its authorized committed
   lineage and appends a private tail.

There is no return to the general planner, voting, weighted output aggregation,
or multi-expert decoding. The frozen output inventory is humor, serious,
angry-style, tool-call, and review/audit; selection cardinality is always one.
Review remains an offline side path rather than a generation loop.

## Semantic decomposition and activation contract

Every task-DAG node must be defined by its real `operation`, `operands`,
`constraints`, `dependencies`, and `expected_relation`. Hashes, row numbers,
case labels, evidence IDs, and language labels may bind identity but may never
manufacture semantic diversity. The DAG is acyclic, bounded, and has at most one
specialist planner per subtask.

Planner handoff uses a fixed 256-dimensional float32 packet serialized as
`canonical_f32_le_v1`, with an explicit absolute position, a SHA-256 commit, and
a reproducible boundary. It is an interface representation, not hidden
chain-of-thought. Alignment trains selected-chain contrastive similarity,
wrong-route margin separation, and dependency consistency. A numerical
acceptance threshold remains `pending_calibration`; it must never be guessed or
silently relaxed.

## Cache lineage

An explicit adapter-switch boundary may hand off the ordered prefix values
actually produced by the preceding stage. That is boundary-switched execution;
it is not equivalent to recomputing the old prefix under the new adapter.

The current Hugging Face DynamicCache claim is limited to exact ordered
prefix-value and prefill-compute handoff. DynamicCache append may concatenate
and reallocate tensors, so object identity does not prove pointer or storage
identity. The frozen current claims are `shared_storage=false`,
`zero_copy=false`, and `full_generation_kv_shared=false`.

Planner-private tails are never exported. The selected output expert receives
only committed lineage and then owns an append-only private tail. Compatibility
must bind token order, positions, mask, RoPE, model, tokenizer, template,
adapter lineage, attention implementation, and Gemma 3 hybrid attention:
22 sliding-window layers with window 512 plus 4 full-attention layers.

## Selected group and RDMA boundary

The selected group may contain the general planner, the specialist planners
chosen for the DAG, and the single output expert. It may read only immutable,
content-addressed activation or prefix pages that have already committed.
Unselected experts have no visibility.

RDMA is a transport optimization, never a source of routing correctness.
Correctness must be identical under local-copy fallback; otherwise execution
fails closed. RDMA writes, future or uncommitted reads, and live private-KV
transfer are forbidden. Local aliasing, CUDA IPC or peer access, and GPUDirect
RDMA are separate transport classes and must not share an attestation label.

No physical RDMA claim is currently valid. A future proof must cover registered
regions; owner and reader identities; read-only key scope; epoch, lease,
revocation, and ABA resistance; content digests before and after reads;
owner-side pointer, offset, stride, and layout; ordered token and positional
lineage; completion fences and lifetime; stale-read rejection; and the 22+4
hybrid-attention semantics. Only a segmented or paged backend with per-layer
evidence may later set `shared_storage=true` or `zero_copy=true`.

## Serial alignment and evaluation

Training order is general planner, frozen specialist planners, selected-chain
alignment, then governed output experts with upstream commits frozen. The
architecture is additive and contributes zero records to the current 4,300-row
FINAL.

Required controls include oracle route, general planner only, specialist chain
without alignment, aligned chain, wrong route, random route, and local-copy
transport. Report DAG exactness, route accuracy, activation similarity,
wrong-route margin and delta, single-output-expert invariance, prefix-value
equality, prefill compute saved, transport bytes and latency when physically
available, stale/revoked rejection, fallback rate, tool grounding, emotion
accuracy, and identity consistency.

The machine-readable source of truth is
`configs/research/neural_swarm_hierarchical_planner_rdma_v1.json`, validated by
`src/anchor_mvp/research/neural_swarm_hierarchical_planner_rdma_v1.py`.
