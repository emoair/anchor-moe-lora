# Partial Gold dataset card

[English](dataset_card_partial_gold.md) | [简体中文](dataset_card_partial_gold.zh-CN.md)

## Scope and intended use

This release contains stage-specific synthetic SFT candidates for the five
Anchor experts. It has two related packages:

- the complete accepted per-expert export, with 1,465 Gold records;
- the deterministic, balanced training snapshot, with 128 records per expert
  (`128 x 5 = 640`).

The packages are explicitly marked `not_for_end_to_end_claim=true`. Only 85
strict five-stage chains are complete, versus the original gate of 256. The
source automation ended in `gate_blocked` and its partition manifest is not
`training_ready`. The balanced snapshot is suitable for the documented
per-expert Partial Gold probe; it is not evidence that the complete routed
system works end to end.

## Contents

The accepted export is bound by
`anchor.per-expert-partial-gold-export.v1`:

| Stage | Records | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `plan` | 384 | 6,502,043 | `c82d7806c161ee96942054c98e45c7068af2ebfa5437492c073c3f6526015932` |
| `tool_policy` | 384 | 6,557,791 | `599ebbd76b9937391bf17ddc2fe4f8086a79f9b832dc296a74605819528c54f3` |
| `frontend` | 346 | 12,602,666 | `c6f7c79756c064b7256ecb2e38bed495e6cd68f863e7073cf66afe8679fbeed2` |
| `review` | 203 | 9,509,163 | `d2c381957bad2661efd8965fa60d5b49164cb00228721cfcec178a1216292c87` |
| `security` | 148 | 4,328,526 | `dbef35c02e16cc4d1506653ec7b4c926c4161a4d8e269b06155196b0768a1cac` |

The frozen snapshot is bound by
`anchor.per-expert-partial-training-snapshot.v1`:

| Expert | Records | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `planner` | 128 | 2,134,537 | `d3c4245e900a6c6736c5cce3aa73c5c32052d86af475d0984e68ad3bec376673` |
| `tool_policy` | 128 | 2,173,743 | `b9682c416a68386a8e7dde138680c9b68d14c0e31b59e2cdb1e2881f679e2b2a` |
| `frontend_gen` | 128 | 4,634,188 | `77dda1691bdf10f3220222d6c59f72d0bdc6009096c32fede99318de96a8e45a` |
| `frontend_review` | 128 | 6,090,499 | `be1c66864801e96a0577397b85f70695115a984c26c4971a21bb12a98bb7ef11` |
| `security_gate` | 128 | 3,788,169 | `20eb4c7ef6d4fc72723401cccd39e0e503fb433323904a55c670c03b64d8928d` |

Every listed publication file is below 50 MiB. The largest accepted-export file
is 12.02 MiB and the largest snapshot file is 5.81 MiB. Runtime logs, retry
archives, quarantine partitions, and held-out artifacts are not part of either
package.

## Selection and integrity

The snapshot selection is deterministic. For each expert, the freezer computes
`SHA256("20260711:" + record_id)`, sorts ascending, and takes the first 128
records. It does not depend on input order or process RNG state.

- Source partition manifest SHA-256:
  `4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7`
- Accepted-export manifest SHA-256:
  `1b8e5b87957d7ec1e867813c95b8f7ab3bef55861e778b6ba9f197e6edf3f2ec`
- Snapshot manifest SHA-256:
  `a0866e6afd7861d9ae827625db5f8a7b3273d4e629b79b09e56c5d0ce7599e28`
- Snapshot content digest:
  `2fe95635cfa441b7d5bed1262c307d37cef4f5592d56071e49150d7aa094acc7`

Verify manifest bindings and the publication size limit without printing record
contents:

```powershell
$export = "data\automated_v3_shards\ark_max_retry2_offset300000_c10\training_exports\per_expert_partial_gold\4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7"
$snapshot = "artifacts\formal_partial_v1\dataset"

function Test-DatasetFiles([string]$dir, [string]$mapName) {
    $manifest = Get-Content -Raw (Join-Path $dir "manifest.json") | ConvertFrom-Json
    foreach ($entry in $manifest.$mapName.PSObject.Properties) {
        $binding = $entry.Value
        $path = Join-Path $dir $binding.path
        if ((Get-Item $path).Length -ne $binding.bytes) { throw "byte mismatch: $path" }
        if ((Get-Item $path).Length -ge 50MB) { throw "file is not below 50 MiB: $path" }
        $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $binding.sha256) { throw "SHA-256 mismatch: $path" }
    }
}

Test-DatasetFiles $export "gold_files"
Test-DatasetFiles $snapshot "files"
```

## Exclusions and limitations

Both manifests prove that `negative`, `reject`, `oracle_label_only`, and
`heldout` records are excluded. The exporter also applies secret-pattern and
held-out-provenance checks, but automated checks are not a guarantee that all
privacy, copyright, or quality risks have been found.

The original coverage gate was waived for this Partial Gold experiment:
`review` is 53 records short and `security` is 108 records short of the
256-per-stage target. More importantly, the security Gold set contains only 3
`BLOCK` labels and 145 `PASS` labels, against a target of 60 `BLOCK` examples.
This severe imbalance means the package must not be presented as sufficient
security training, a validated safety classifier, or a basis for autonomous
security decisions. Security capability requires a separate balanced dataset
and held-out evaluation.

## Reproduce the snapshot and run the probe

From the repository root, freezing is idempotent: an existing matching snapshot
is verified rather than silently replaced.

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python -m anchor_mvp.data.partial_snapshot `
  --export-dir "data\automated_v3_shards\ark_max_retry2_offset300000_c10\training_exports\per_expert_partial_gold\4fc4621d2702238aff5b3e88fc348058926e6f2488dc23e2d6c3dbd7344f5af7" `
  --output-dir "artifacts\formal_partial_v1\dataset" `
  --per-expert 128 `
  --seed 20260711
```

Use the guarded low-memory launcher. `preflight` is read-only; execute the
one-step smoke before the two-step probe. These commands do not constitute a
full training or end-to-end result.

```powershell
$python = (Get-Command python).Source
.\scripts\train\run_formal_partial_v1_lowmem.ps1 -Arm preflight -Python $python
.\scripts\train\run_formal_partial_v1_lowmem.ps1 -Arm smoke -Execute -Python $python
.\scripts\train\run_formal_partial_v1_lowmem.ps1 -Arm probe -Execute -Python $python
```

The profile keeps a hard 9 GiB CUDA peak limit. The documented admission
threshold is 12 GiB of available host memory for the common profile and 11 GiB
for the probe-only path; do not weaken those gates merely to pass a loaded host.

## Sources, acknowledgements, and terms

The repository's task-bank design, leakage controls, and tooling were informed
by or integrate with the projects below. Their inclusion here does not imply
endorsement, and the two release manifests alone do not establish that any
particular row was derived from a named upstream dataset.

- [SWE-bench](https://github.com/SWE-bench/SWE-bench) and its
  [dataset card](https://huggingface.co/datasets/SWE-bench/SWE-bench);
- [SWE-smith](https://github.com/SWE-bench/SWE-smith) and its
  [dataset card](https://huggingface.co/datasets/SWE-bench/SWE-smith);
- [OpenCode](https://github.com/anomalyco/opencode), used by the controlled
  tool-execution design;
- [CC Switch](https://github.com/farion1231/cc-switch), used as a pinned metadata
  and control-plane design reference rather than an embedded router;
- pinned SOP/Skill inputs from
  [GitHub awesome-copilot](https://github.com/github/awesome-copilot) and
  [Anthropic Skills](https://github.com/anthropics/skills), with exact commits,
  file hashes, and retained notices in
  [`configs/data/skill_sources.yaml`](../configs/data/skill_sources.yaml) and
  [`THIRD_PARTY_SKILLS.md`](../THIRD_PARTY_SKILLS.md).

Licensing layers must remain separate. Anchor's code and data-production tooling
are distributed under the repository's
[`AGPL-3.0-or-later`](../LICENSE), but that code license does not automatically
relicense teacher outputs, upstream datasets, benchmark instances, third-party
repositories, model/API outputs, or vendored Skills. OpenCode, CC Switch, and
each Skill retain their upstream notices. SWE-style instances require the
snapshot-specific dataset revision, per-repository license ledger, and
attribution described in
[`swebench_metadata_import.md`](swebench_metadata_import.md). This card does not
invent or grant a separate license for the dataset files; publishers must state
the reviewed data terms in the release metadata before redistribution.
