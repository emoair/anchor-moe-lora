# SWE-bench full train bank

This guide covers the reproducible, train-only bank. See the
[Chinese guide](swebench_full_bank.zh-CN.md) for the paired translation.

## Pinned scope

- Dataset: `SWE-bench/SWE-bench`, immutable revision
  `7074ef12ea2a6f70a228943c1336553333c22786`, split `train` only.
- Source parquet: 106,492,326 bytes, SHA-256
  `0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69`.
- Source cardinality: 19,008 tasks from 35 repositories.
- Deterministic train-derived partitions: 17,105 training and 1,903 validation.
  The validation rows still come from the official train split.
- Deterministic locale assignment: 9,504 `en-US` and 9,504 `zh-CN`. Assignment
  does not claim that translation has already happened.
- Five ordered work items per task: 95,040 total in
  `planner -> tool_policy -> domain_builder -> domain_review -> security` order.

The raw parquet stays under `artifacts/swebench/source/`. It is local-only and is
never part of the public export.

Local offline verification on 2026-07-18 reports `launch_ready=true` and
`publication_ready=true`. `training_ready=false` is the honest pre-run state:
only `gold_manifest`, `zh_cn_localization_manifest`, and
`real_tool_results_manifest` are missing, and the builder does not fabricate
them.

## 1. Download the pinned source

The helper is a dry run unless `-ConfirmDownload` is supplied. It requests only
the pinned train parquet.

```powershell
cd D:\LLM\anchor-moe-lora
.\scripts\data\download_swebench_train.ps1
```

Direct physical binding requires a local IPv4 address that maps, by interface
index, to an `Up` adapter returned by `Get-NetAdapter -Physical`. A virtual,
TUN, TAP, VPN, disconnected, or unknown adapter is rejected before `curl` runs.

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  ForEach-Object {
    $ip = $_
    Get-NetAdapter -Physical |
      Where-Object { $_.ifIndex -eq $ip.InterfaceIndex -and $_.Status -eq 'Up' } |
      Select-Object @{n='IPAddress';e={$ip.IPAddress}},Name,ifIndex,Status
  }

.\scripts\data\download_swebench_train.ps1 `
  -DirectPhysicalRoute `
  -SourceAddress 192.168.1.23 `
  -ConfirmDownload
```

Replace the example address with an address printed by the first command.

## 2. Offline reboot preflight

Use the project Python environment. The preflight is read-only: it hashes local
files and validates JSON metadata, but sends no provider request, starts no
sandbox, loads no credential, and needs no GPU.

```powershell
$py = 'C:\Users\Air\.conda\envs\anchor-mvp\python.exe'
& $py .\scripts\data\build_swebench_full_bank.py `
  --config .\configs\data\swebench_full_bank.formal.yaml `
  --preflight
```

Use `--require-launch-ready` when a nonzero exit is needed for automation:

```powershell
& $py .\scripts\data\build_swebench_full_bank.py `
  --config .\configs\data\swebench_full_bank.formal.yaml `
  --preflight --require-launch-ready
```

Readiness has three independent groups:

1. `launch_ready` validates the pinned source hash, row count, train-only scope,
   all active formal routes at exact `reasoning_effort: max`, the dual-target
   patched-OpenCode bundle attestation, and the ready CC Switch route
   attestation. It does **not** require any output of the run being launched.
2. `training_ready` validates completed, source-bound manifests for Gold,
   `zh-CN` localization, and real tool results. These are post-run artifacts;
   they are never launch inputs.
3. `publication_ready` validates the dedicated public export, its MIT source
   attribution, exact file inventory, structured-field and credential scans,
   train-only binding, and the strict per-file size limit.

This split removes the former circular condition in which sandbox results had
to exist before the sandbox run could start. A machine may validly report any
combination of the three states.

## 3. Build local staging and the public projection

```powershell
& $py .\scripts\data\build_swebench_full_bank.py `
  --config .\configs\data\swebench_full_bank.formal.yaml
```

After an audited public export already exists, refresh its content-free
inventory snapshot without parsing payload JSONL or the source parquet:
`& $py .\scripts\data\build_swebench_full_bank.py --config
.\configs\data\swebench_full_bank.formal.yaml
--refresh-hash-only-from-public`.

Local staging is written under `artifacts/swebench/full-bank-v1/`. The audited
public projection is written under `datasets/public/swebench-full-bank-v1/`.
The command prints counts, paths, and gate states only; it never prints issue
bodies.

The public directory contains only:

- `source-metadata.train*.jsonl`, whose rows have exactly `repo`, `instance_id`,
  `base_commit`, and `problem_statement`;
- `candidate-tasks/tasks*.jsonl` and
  `candidate-work-orders/work-orders*.jsonl`;
- `allowlists/train.json` and `allowlists/validation-from-train.json`;
- `ATTRIBUTION.md` and `manifest.json`.

The exporter column-projects the parquet before rows enter Python. Answer
changes, evaluation changes, hints, evaluator labels, non-train benchmark
records, and credentials are not public fields. Every public file must be
strictly smaller than 52,428,800 bytes. The source parquet itself is excluded.
High-confidence credential-shaped strings in public issue text are replaced by
`[REDACTED_CREDENTIAL]`, and the manifest records the replacement count.
An unknown sibling file makes `publication_ready=false`; the `.gitignore`
allowlist mirrors the same exact directory and filename families.

## 4. Training-output manifest contract

Each local post-run manifest binds to the dataset ID, immutable revision,
`split: train`, and source parquet SHA-256. It must declare `complete: true`,
`train_only: true`, `contains_heldout: false`, a positive `record_count`, and a
non-empty file inventory whose hashes and record counts validate. The three
schemas are:

- `anchor.swebench-gold-manifest.v1` with `gold_records: true`;
- `anchor.swebench-zh-cn-localization-manifest.v1` with `locale: zh-CN`;
- `anchor.swebench-real-tool-results-manifest.v1` with
  `real_tool_results: true`.

Candidate task and work-order shards are launch inputs, not training Gold.
Hidden chain-of-thought is neither requested nor fabricated.

## 5. Attribution and publication boundary

`ATTRIBUTION.md` identifies the upstream SWE-bench repository and its MIT
software license (`SPDX-License-Identifier: MIT`) at the pinned revision. It
does not relicense issue text or the 35 source repositories; their applicable
terms remain in force. Runtime Gold, localization bodies, tool results, model
weights, logs, the raw parquet, and credentials remain under ignored local
paths and are not copied into the public dataset directory.

## Troubleshooting

- `launch_ready=false`: inspect only `gates.launch.missing` and
  `gates.launch.invalid`. Training-output files cannot block launch.
- `route_not_ready`: rebuild and validate the CC Switch component attestation;
  mere manifest presence is not readiness.
- `training_ready=false`: inspect the three post-run manifests and their bound
  payload hashes.
- `publication_ready=false`: inspect the content-free publication audit codes;
  remove unknown files or rebuild the deterministic export.
- `parquet_metadata_unreadable`: install the project training extra with
  `pyarrow` in the selected Python environment.
- `file_size_limit`: reduce `publication.shard_rows`; never increase the limit
  above 52,428,800 bytes.
