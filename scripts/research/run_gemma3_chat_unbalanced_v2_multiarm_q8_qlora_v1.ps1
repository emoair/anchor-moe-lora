[CmdletBinding(DefaultParameterSetName = "ValidateConfig")]
param(
    [Parameter(ParameterSetName = "ValidateConfig")]
    [switch]$ValidateConfig,

    [Parameter(Mandatory = $true, ParameterSetName = "DryRun")]
    [switch]$DryRun,

    [Parameter(Mandatory = $true, ParameterSetName = "DryRun")]
    [string]$ConsumerPreflightReceipt,

    [Parameter(Mandatory = $true, ParameterSetName = "DryRun")]
    [string]$TensorInventory,

    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EntryPoint = Join-Path $ProjectRoot `
    "scripts\research\run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1.py"
$CanonicalConfig = Join-Path $ProjectRoot `
    "configs\training\gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1.yaml"

function Resolve-MultiArmPython([string]$Requested) {
    $Candidates = New-Object Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        $Candidates.Add($Requested)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_TRAINING_PYTHON)) {
        $Candidates.Add([string]$env:ANCHOR_TRAINING_PYTHON)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_PYTHON)) {
        $Candidates.Add([string]$env:ANCHOR_PYTHON)
    }
    else {
        $Candidates.Add((Join-Path $ProjectRoot ".venv\Scripts\python.exe"))
        $Candidates.Add("python.exe")
        $Candidates.Add("python")
    }
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
        $Command = Get-Command $Candidate -CommandType Application `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $Command) {
            return $Command.Source
        }
    }
    throw "Python 3.10+ not found; set -Python or ANCHOR_TRAINING_PYTHON."
}

if (-not (Test-Path -LiteralPath $EntryPoint -PathType Leaf)) {
    throw "Model-free multi-arm entry point is unavailable."
}
if (-not (Test-Path -LiteralPath $CanonicalConfig -PathType Leaf)) {
    throw "Canonical model-free multi-arm config is unavailable."
}
$PythonExecutable = Resolve-MultiArmPython -Requested $Python

if ($DryRun) {
    if (-not (Test-Path -LiteralPath $ConsumerPreflightReceipt -PathType Leaf)) {
        throw "Authenticated consumer preflight receipt is required."
    }
    if (-not (Test-Path -LiteralPath $TensorInventory -PathType Leaf)) {
        throw "Observed tensor inventory is required."
    }
    & $PythonExecutable `
        $EntryPoint `
        --config $CanonicalConfig `
        --dry-run `
        --consumer-preflight-receipt $ConsumerPreflightReceipt `
        --tensor-inventory $TensorInventory
    exit $LASTEXITCODE
}

& $PythonExecutable `
    $EntryPoint `
    --config $CanonicalConfig `
    --validate-config
exit $LASTEXITCODE
