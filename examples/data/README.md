# Generated data location

Run the offline smoke configuration from the repository root:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m anchor_mvp.data --config configs/data/smoke.yaml --dry-run run
```

Live Kimi runs use a 600-second non-streaming HTTP read timeout and one retry by
default. Override conservatively with `--timeout-seconds` and `--max-retries`; a timed
out generation may already have consumed provider quota even if no JSONL row was
written.

OpenAI-compatible live calls default to SSE streaming. Set `--no-stream-openai` only
for an endpoint known to accept non-streaming requests. The optional
`--stream-options-include-usage` field is disabled by default for compatibility.

The generated JSONL is intentionally not pre-populated here; provenance timestamps and
absolute SOP source paths should reflect the machine that actually created the corpus.
