[CmdletBinding()]
param(
    [string]$Config = "configs/orchestration/distill_train_handoff.yaml",
    [switch]$ConfirmLive,
    [switch]$ConfirmTraining
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Arguments = @(
    (Join-Path $ProjectRoot "scripts/run_distill_train_handoff.py"),
    "--config",
    (Join-Path $ProjectRoot $Config)
)

if ($ConfirmLive) {
    $Arguments += "--confirm-live"
}
if ($ConfirmTraining) {
    $Arguments += "--confirm-training"
}

Push-Location $ProjectRoot
try {
    & py @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
