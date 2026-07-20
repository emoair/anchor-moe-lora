# Offline zh-CN training-copy audit

This workflow is bound to the compact MVP-v2b candidate registry and never
discovers inputs by scanning a directory. The only readable source records are
the five JSONL files allowlisted by
`artifacts/compact_mvp_v2b/candidate_dataset/manifest.registry-formal-v2.json`.
The expected source snapshot is
`43f97bca74aac5b747bf8b8a95dd593dcbc3683e892775ec350282a540d5390c`.
Held-out, benchmark, evaluation, and mixed/oversampled files are not inputs.

Create four deterministic offline work shards:

```powershell
py scripts/data/translation_zh_cn.py prepare
```

Each line is one exact envelope:

```json
{
  "source_path": "artifacts/compact_mvp_v2b/candidate_dataset/data_plan.jsonl",
  "source_line": 1,
  "source_id": "record_...::compact-v2",
  "source_record_sha256": "...",
  "target_locale": "zh-CN",
  "translated_record": {}
}
```

The row hash is SHA-256 over UTF-8 canonical JSON (`sort_keys=true`, compact
separators, `ensure_ascii=false`). A translated ID is the source ID plus
`::zh-CN`. Existing provenance, including its `source_id`, and `compact_v2`
metadata remain unchanged. JSON keys, list lengths, non-text values, code fields,
code blocks, inline code, URLs, HTML tags, protocol markers, decision labels,
and technical identifiers are protected. The compact user message must be
rebuilt from the translated structured input; the assistant message must remain
the canonical target derived from output.

Audit all four shards without publishing:

```powershell
py scripts/data/translation_zh_cn.py merge --dry-run
```

Publish only after the audit passes:

```powershell
py scripts/data/translation_zh_cn.py merge
```

Publication is atomic and refuses overwrite. It writes the five original
filenames under
`artifacts/compact_mvp_v2b/translation_zh_cn_v1/candidate_dataset_zh_cn`, plus
`bilingual_snapshot.jsonl`, `manifest.translation-zh-CN.json`, and the manifest
SHA-256 sidecar. Missing/duplicate source mappings, duplicate translated payloads,
empty or unchanged translations, absent Chinese text, source drift, schema
failure, or any protected-field mutation aborts publication.
