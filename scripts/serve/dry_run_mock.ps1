$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
py -m anchor_mvp.benchmark.dry_run
