param(
    [string]$Config = "configs/data/automation.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $repoRoot
$env:PYTHONPATH = (Join-Path $repoRoot "src")
& python -m anchor_mvp.data.automation --config $Config --status-only
exit $LASTEXITCODE

