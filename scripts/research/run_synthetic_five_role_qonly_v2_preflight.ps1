<#
.SYNOPSIS
Runs the strict five-role synthetic Q-only preflight serially.

.DESCRIPTION
Dataset-only mode is the default and never loads a tokenizer, model, CUDA,
provider, or trainer. Use -TokenizerOnly for the authenticated local tokenizer
length check. -PublishPreflight additionally writes role-isolated no-replace
receipts and therefore requires -TokenizerOnly.

.EXAMPLE
.\scripts\research\run_synthetic_five_role_qonly_v2_preflight.ps1

.EXAMPLE
.\scripts\research\run_synthetic_five_role_qonly_v2_preflight.ps1 -TokenizerOnly
#>
[CmdletBinding()]
param(
    [switch]$TokenizerOnly,
    [switch]$PublishPreflight,
    [switch]$CheckPythonOnly,
    [string]$PythonExecutable = "",
    [string]$Config = "configs/training/qwen2_5_1_5b_synthetic_five_role_qonly_v2.yaml"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$zhExplicitPythonUnavailable = "$([char]0x663E)$([char]0x5F0F) Python $([char]0x8DEF)$([char]0x5F84)$([char]0x4E0D)$([char]0x53EF)$([char]0x7528)"
$zhNoPython = "$([char]0x672A)$([char]0x627E)$([char]0x5230)$([char]0x53EF)$([char]0x7528) Python"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if ($PublishPreflight -and -not $TokenizerOnly) {
    throw "-PublishPreflight requires -TokenizerOnly."
}

function Test-PythonCandidate {
    param([Parameter(Mandatory = $true)][string]$Candidate)
    try {
        $versionOutput = @(& $Candidate --version 2>&1)
        if ($LASTEXITCODE -eq 0 -and ($versionOutput -join " ") -match "^Python\s+\d+\.\d+") {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

if (-not [string]::IsNullOrWhiteSpace($PythonExecutable)) {
    if (-not (Test-PythonCandidate -Candidate $PythonExecutable)) {
        throw "The explicit -PythonExecutable is unavailable or not a real Python interpreter. $zhExplicitPythonUnavailable. Run 'conda activate anchor-mvp', set ANCHOR_PYTHON, or pass -PythonExecutable <path-to-python.exe>."
    }
}
else {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($env:ANCHOR_PYTHON) {
        $candidates.Add($env:ANCHOR_PYTHON)
    }
    if ($env:CONDA_PREFIX) {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    $candidates.Add((Join-Path $repoRoot ".venv\Scripts\python.exe"))
    if ($HOME) {
        $candidates.Add((Join-Path $HOME ".conda\envs\anchor-mvp\python.exe"))
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add($pythonCommand.Source)
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-PythonCandidate -Candidate $candidate) {
            $PythonExecutable = $candidate
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
        throw "No usable Python interpreter was found (WindowsApps aliases are rejected). $zhNoPython. Run 'conda activate anchor-mvp', set ANCHOR_PYTHON, or pass -PythonExecutable <path-to-python.exe>."
    }
}

Write-Host "[five-role preflight] python=$PythonExecutable"
if ($CheckPythonOnly) {
    Write-Host "[five-role preflight] PASS: Python interpreter resolved."
    return
}

$roles = @(
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate"
)

Push-Location $repoRoot
try {
    foreach ($role in $roles) {
        Write-Host "[five-role preflight] role=$role mode=$(if ($TokenizerOnly) { 'tokenizer-only' } else { 'dataset-only' })"
        $arguments = @(
            "scripts/research/prepare_synthetic_five_role_qonly_v2.py",
            "--config", $Config,
            "--role", $role
        )
        if ($TokenizerOnly) {
            $arguments += "--tokenizer-only"
        }
        if ($PublishPreflight) {
            $arguments += "--publish-preflight"
        }
        & $PythonExecutable @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Five-role preflight stopped at role '$role' (exit $LASTEXITCODE). Read the emitted error_code/hint; no later role was run."
        }
    }
    Write-Host "[five-role preflight] PASS: all five roles authenticated serially; no model, CUDA, provider, or training request was made."
}
finally {
    Pop-Location
}
