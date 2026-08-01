# Gemma 3 existing distillation v14 validation asset v1

This additive asset freezes the existing v14 route dataset for validation only. It does not authorize retraining, deployment, formal evaluation, or production use.

The builder authenticates the 16,632-row source and its manifest, the passed training receipt, the failed overall-attempt receipt, and the frozen source-snapshot manifest. The passed training stage and failed provenance gate are represented as separate facts; `attempt_consumed` remains false.

Only these fields are persisted per row: `sample_id`, `split`, `source_group_sha256`, `leakage_family_sha256`, `prompt_sha256`, and the source ordinal. Prompt text is streamed only to authenticate its declared digest and is never written to the asset. Labels, targets, messages, raw token IDs, and other body fields are forbidden.

Policies are fixed as follows:

- `train`: seen-train regression validation;
- `calibration`: current validation;
- `holdout`: sealed single-blind final-only validation.

All three partitions are non-training assets. The builder proves 13,131/1,923/1,578 records, global sample-ID uniqueness, deterministic sample-ID order, and zero cross-split overlap for source-group, leakage-family, and prompt digests.

Publication uses an authenticated staging directory, exclusive file creation, mandatory SHA-256 sidecars, a pre-rename inventory/identity recheck, no-replace rename, and a post-publish recheck. A failure preserves staging evidence rather than recursively deleting an unbound path.
