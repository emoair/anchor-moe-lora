[CmdletBinding(DefaultParameterSetName = "Status")]
param(
    [Parameter(ParameterSetName = "Status")]
    [switch]$Status,

    [Parameter(Mandatory = $true, ParameterSetName = "Preflight")]
    [switch]$Preflight,

    [Parameter(Mandatory = $true, ParameterSetName = "Execute")]
    [switch]$Execute,

    [Parameter(Mandatory = $true, ParameterSetName = "Bind")]
    [switch]$Bind,

    [Parameter(Mandatory = $true, ParameterSetName = "Bind")]
    [string]$TrainingRunId,

    [Parameter(ParameterSetName = "Bind")]
    [string]$BoundConfigOutput,

    [string]$Config,

    [string]$RunId
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EntryPoint = Join-Path $ProjectRoot "scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.py"
$PythonPrefix = @()

if ($env:ANCHOR_TRAINING_PYTHON) {
    $Python = $env:ANCHOR_TRAINING_PYTHON
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $Python = "py"
    $PythonPrefix = @("-3.9")
} else {
    throw "Python 3.9+ not found. Set ANCHOR_TRAINING_PYTHON."
}

$Mode = if ($Bind) {
    "--bind-training-run"
} elseif ($Execute) {
    "--execute"
} elseif ($Preflight) {
    "--preflight"
} else {
    "--status"
}
$Arguments = @($PythonPrefix) + @($EntryPoint, $Mode)
if ($Bind) {
    $Arguments += @($TrainingRunId)
}
if ($Config) {
    $Arguments += @("--config", $Config)
}
if ($BoundConfigOutput) {
    $Arguments += @("--bound-config-output", $BoundConfigOutput)
}
if ($RunId) {
    $Arguments += @("--run-id", $RunId)
}

Write-Host "Gemma 3 chat five-expert generation evaluation: $Mode"
Write-Host "Arms: frozen Q8 base / correct adapter / fixed wrong route"
Write-Host "Output: artifacts/diagnostics/gemma3_chat_five_expert_qonly_generation_eval_v1/"
& $Python @Arguments
exit $LASTEXITCODE
