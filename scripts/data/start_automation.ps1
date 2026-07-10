param(
    [string]$Config = "configs/data/automation.yaml",
    [switch]$DryRun,
    [switch]$NoWaitCooldown
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $repoRoot
$env:PYTHONPATH = (Join-Path $repoRoot "src")

if (-not $DryRun -and [string]::IsNullOrWhiteSpace($env:KIMI_API_KEY)) {
    throw "KIMI_API_KEY must be set in the current shell. It is never read from YAML."
}

$arguments = @(
    "-m", "anchor_mvp.data.automation",
    "--config", $Config
)
if ($DryRun) {
    $arguments += "--dry-run"
}
if (-not $NoWaitCooldown) {
    $arguments += "--wait-cooldown"
}

Write-Host "Anchor-MoE-LoRA unattended distillation"
Write-Host "Config: $Config"
Write-Host "Status: data output directory / automation / status.json"
Write-Host "The visible process remains active across persisted rate-limit cooldowns."
& python @arguments
exit $LASTEXITCODE
