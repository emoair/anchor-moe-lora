# Planner-first dataset style companion v1

This additive companion refines the metadata shape used after a Planner has
been trained and frozen. It does not modify the existing training runner,
materialize payload bodies, launch a model, or authorize training.

## Read-only evidence

The design uses only body-free facts from the training task:

- the latest 16,632-record dataset isolation gate passed;
- the latest 500-step Planner/Router training stage passed numerically;
- the attempt remained `consumed=false` because execution provenance failed;
- historical invalidated inventories contain 12,586 record IDs and 11,660
  normalized prompts; and
- no holdout body was read for this companion.

The provenance failure is not reclassified as a dataset-quality failure. The
training worktree and its execution pipeline remain unchanged.

## Semantic route versus physical projection

The five expert assets are available candidates, not an instruction to
activate all five:

1. Select exactly one primary style expert: `humor`, `serious`, or
   `angry_style`.
2. Add `tool_call` only when the semantic task requires a tool.
3. Add `review_audit` only when the task requires a review sidecar.
4. Carry identity as a shared semantic invariant. It is not a sixth output
   expert and does not create an extra training row.
5. Exit the Planner after the immutable route commit. Expert output never
   returns to the Planner for rewriting, voting, or aggregation.

Each physical projection row still belongs to one expert. Therefore the
frozen parent rule `one_expert_per_question=true` is preserved:

```text
one semantic question
  -> one primary-style projection
  -> zero or one tool sidecar projection
  -> zero or one review sidecar projection
```

The bundle validator requires the physical projection set to equal the
selected set exactly. A candidate inventory cannot be copied into the
activation set.

## Record style

Every body-free projection row binds:

- semantic-question, source-record, source-group, leakage-family, prompt, and
  payload digests;
- the frozen Planner receipt, adapter, and question-generation receipt;
- one training expert and a projection kind;
- the route variant (`base`, `with_review`, `with_tool`, or
  `with_tool_review`);
- identity class and invariant digest when identity is required;
- salient, distractor, shared-prefix, and route-evidence segment identities;
  and
- zero target/future segment counts, no embedded payload, and no raw token
  IDs.

The payload remains in a separate immutable object referenced by SHA-256.

## Partition and leakage discipline

The supported metadata splits are `train`, `calibration`, and `eval_proxy`.
Holdout rows are forbidden from this style manifest. A holdout may only be
referenced by an inventory digest, and it cannot be used for style selection,
negative mining, or tuning.

Split assignment occurs before language views, route variants, hard
negatives, or expert projections. These identities must never cross splits:

- `task_bundle_sha256`
- `source_group_sha256`
- `leakage_family_sha256`
- `prompt_digest_sha256`

Historical invalidated inventories are mandatory. The body-free isolation
receipt keeps the NFKC/casefold/word/q-gram-5 Dice threshold fixed at `0.90`.

## Route hard negatives

Hard negatives are route counterfactuals, not positive expert rows. Each one
binds the positive and negative route commits and must change exactly one
semantic variable:

- wrong primary style;
- missing or unnecessary tool;
- missing or unnecessary review;
- dropped identity requirement; or
- false identity attribution.

No hard negative may be derived from heldout data, and no negative route
commit may appear in the positive projection inventory.

The companion also requires fresh coverage for the known weak combinations:
`serious + identity` under `base`, `with_review`, and `with_tool_review`,
plus a `with_tool` positive control. This is a metadata quota only; it does
not reuse any previous holdout text.

## Current boundary

The companion can validate a body-free manifest, but the existing GPU runner
does not consume this v2 manifest yet. A future additive launcher must:

1. authenticate the frozen Planner;
2. generate and persist external question payloads;
3. emit a question-generation receipt that binds this style-contract SHA;
4. validate the v2 manifest;
5. derive the parent v1 train-only projection view; and
6. invoke each leaf runner strictly serially.

Until that integration is independently reviewed:

- `planner_first_leaf_launcher_integrated=false`
- `training_authorized=false`
- `gpu_authorized=false`
- `formal=false`
- `release_authorized=false`
