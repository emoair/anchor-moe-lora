# Planner-first expert curriculum v1

Status: additive, model-free preflight contract  
Schema: `anchor.neural-swarm-planner-first-curriculum.v1`

## Why this layer exists

The frozen multi-arm runtime is strictly serial (`concurrency=1`), but its physical arm
order is not Planner-first: the five output-expert arms precede the three Planner arms.
Its expert datasets are also authenticated static assets rather than projections from a
trained Planner's committed questions.

This additive layer records that fact instead of relabelling the legacy order. It does
not edit the existing runtime, runtime config, schemas, launcher, dataset, or GPU lock.
The existing runtime remains an immutable leaf executor.

The two older hierarchical-contract dependencies predate their LF attributes. Their
pins use canonical LF Git-blob bytes; the validator permits only a reversible CRLF
working-tree expansion of those exact bytes. All frozen training-runtime pins remain
strict byte-for-byte identities.

## Required order

1. Train the primary `planner_q_only` arm on the authenticated Planner asset.
2. Freeze that adapter and bind its training receipt and parameter inventory.
3. Use only the frozen Planner adapter to produce an immutable question payload set and
   a body-free manifest containing payload hashes, semantic hashes, route commits, and
   exactly one selected expert per question. A separate generation receipt must bind
   that manifest to the frozen adapter, generator implementation, and real model-request
   count; merely copying the adapter hash into a manifest is insufficient.
4. Project those committed questions into five expert inventories.
5. Issue one leaf-preflight ticket per expert. Tickets remain serial and do not
   authorize GPU execution by themselves.

The diagnostic `planner_o_only` and `planner_q_plus_micro_o` arms cannot unlock expert
training. Every question ID binds the frozen Planner receipt, Planner adapter, source
record, semantic identity, immutable payload, language, split, and selected expert.
Changing any field changes the ID and route commit.

Every expert ticket also binds the question-generation receipt. Consequently a static
question list, an unfrozen Planner, or a diagnostic Planner arm cannot be relabelled as
Planner-produced expert data.

The expert leaf order is `humor`, `serious`, `angry_style`, `tool_call`,
`review_audit`. This order is a control-plane release plan, not a mutation of the
legacy `TRAINED_ARMS` tuple. With the present single-GPU runtime, leaf concurrency
remains one.

## No return to the Planner

This topology is one-way:

`Planner train -> Planner freeze -> question commit -> expert projection -> expert`

Once an expert starts, neither its output nor its private KV tail returns to the
Planner for voting, rewriting, or a second aggregate response. Review remains an
explicit expert/offline side path, not a mandatory second controller.

## Current authority

The validator is model-free. A valid decision means only
`expert_leaf_preflight_ready`: all upstream identities are complete enough for a
separate additive leaf launcher preflight. The frozen legacy runtime does not enforce
these tickets and remains directly invokable with its old static order; therefore the
decision also reports `planner_first_leaf_launcher_integrated=false` and
`expert_training_execution_blocked=true`. It does not mean the questions were generated
in this commit, that training ran, or that GPU/formal/release authority exists.

The machine-readable sources are:

- `configs/research/neural_swarm_planner_first_curriculum_v1.json`
- `configs/research/neural_swarm_planner_first_curriculum_v1.schema.json`
- `configs/research/neural_swarm_planner_first_runtime_bundle_v1.schema.json`
- `configs/research/neural_swarm_planner_first_decision_v1.schema.json`
- `src/anchor_mvp/research/neural_swarm_planner_first_curriculum_v1.py`
