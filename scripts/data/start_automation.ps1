[CmdletBinding()]
param(
    [string]$Config = "configs/data/automation.yaml",
    [switch]$DryRun,
    [switch]$NoWaitCooldown,
    [switch]$PromptForApiKey,
    [switch]$ValidateConfig,
    [switch]$StatusOnly,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $repoRoot
$env:PYTHONPATH = (Join-Path $repoRoot "src")

function Resolve-ApplicationPath {
    param([Parameter(Mandatory = $true)][string]$Value)

    if (Test-Path -LiteralPath $Value -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Value).Path
    }
    $command = Get-Command -Name $Value -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $command) {
        return $null
    }
    return $command.Source
}

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    $versionText = & $Executable @PrefixArguments -c (
        "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    ) 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($versionText)) {
        return $null
    }
    try {
        $version = [Version]($versionText | Select-Object -Last 1)
    }
    catch {
        return $null
    }
    if ($version -lt [Version]"3.10") {
        return $null
    }
    return $version.ToString()
}

function Resolve-PythonInvocation {
    param([string]$RequestedPython)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPython)) {
        $requestedPath = Resolve-ApplicationPath -Value $RequestedPython
        if ([string]::IsNullOrWhiteSpace($requestedPath)) {
            throw "PythonExe was not found: $RequestedPython"
        }
        $requestedVersion = Test-PythonCandidate -Executable $requestedPath
        if ([string]::IsNullOrWhiteSpace($requestedVersion)) {
            throw "PythonExe must be a working Python 3.10 or newer interpreter."
        }
        return [pscustomobject]@{
            Executable = $requestedPath
            PrefixArguments = @()
            Version = $requestedVersion
            Source = "-PythonExe"
        }
    }

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_MVP_PYTHON)) {
        $candidates += [pscustomobject]@{
            Value = $env:ANCHOR_MVP_PYTHON
            PrefixArguments = @()
            Source = "ANCHOR_MVP_PYTHON"
        }
    }
    $candidates += [pscustomobject]@{
        Value = (Join-Path $repoRoot ".venv\Scripts\python.exe")
        PrefixArguments = @()
        Source = "project .venv"
    }
    if (-not [string]::IsNullOrWhiteSpace($HOME)) {
        $candidates += [pscustomobject]@{
            Value = (Join-Path $HOME ".conda\envs\anchor-mvp\python.exe")
            PrefixArguments = @()
            Source = "anchor-mvp conda environment"
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $candidates += [pscustomobject]@{
            Value = (Join-Path $env:CONDA_PREFIX "python.exe")
            PrefixArguments = @()
            Source = "active conda environment"
        }
    }
    $candidates += [pscustomobject]@{
        Value = "py.exe"
        PrefixArguments = @("-3.11")
        Source = "Python launcher 3.11"
    }
    $candidates += [pscustomobject]@{
        Value = "python.exe"
        PrefixArguments = @()
        Source = "PATH"
    }

    foreach ($candidate in $candidates) {
        $candidatePath = Resolve-ApplicationPath -Value $candidate.Value
        if ([string]::IsNullOrWhiteSpace($candidatePath)) {
            continue
        }
        $candidateVersion = Test-PythonCandidate `
            -Executable $candidatePath `
            -PrefixArguments $candidate.PrefixArguments
        if (-not [string]::IsNullOrWhiteSpace($candidateVersion)) {
            return [pscustomobject]@{
                Executable = $candidatePath
                PrefixArguments = @($candidate.PrefixArguments)
                Version = $candidateVersion
                Source = $candidate.Source
            }
        }
    }
    throw (
        "No compatible Python was found. Install the anchor-mvp Python 3.11 " +
        "environment or pass -PythonExe explicitly."
    )
}

function Convert-SecureStringToPlainText {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureValue)

    $buffer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($buffer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($buffer)
    }
}

$configCandidate = if ([IO.Path]::IsPathRooted($Config)) {
    $Config
}
else {
    Join-Path $repoRoot $Config
}
if (-not (Test-Path -LiteralPath $configCandidate -PathType Leaf)) {
    throw "Automation config was not found: $Config"
}
$configPath = (Resolve-Path -LiteralPath $configCandidate).Path
$python = Resolve-PythonInvocation -RequestedPython $PythonExe

if ($ValidateConfig -and ($DryRun -or $StatusOnly)) {
    throw "ValidateConfig cannot be combined with DryRun or StatusOnly."
}
if ($StatusOnly -and ($DryRun -or $PromptForApiKey)) {
    throw "StatusOnly cannot be combined with DryRun or PromptForApiKey."
}

# Parse the same YAML and provider contract used by the automation module. The
# probe emits only the credential environment-variable name, never its value.
$configProbe = @'
import sys
from pathlib import Path

from anchor_mvp.data.automation import AutomationConfig, _simple_config
from anchor_mvp.data.provider import provider_spec

config_path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
raw = _simple_config(config_path)
AutomationConfig.from_mapping(raw, repo_root=repo_root)
print(provider_spec(raw).api_key_env)
'@
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $probeLines = @(
        & $python.Executable @($python.PrefixArguments) -c $configProbe $configPath $repoRoot 2>&1
    )
    $probeExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($probeExitCode -ne 0) {
    $probeError = ($probeLines | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    throw "Automation configuration preflight failed:$([Environment]::NewLine)$probeError"
}
$apiKeyEnv = [string]($probeLines |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -Last 1
)
if ($apiKeyEnv -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
    throw "Automation configuration preflight returned an invalid api_key_env name."
}

Write-Host "Anchor-MoE-LoRA unattended distillation"
Write-Host "Config: $configPath"
Write-Host "Python: $($python.Executable) ($($python.Version); $($python.Source))"
Write-Host "Credential environment: $apiKeyEnv (value hidden)"
Write-Host "Status: data output directory / automation / status.json"

if ($ValidateConfig) {
    Write-Host "Configuration validated. No provider request was sent."
    exit 0
}

if ($StatusOnly) {
    & $python.Executable @($python.PrefixArguments) `
        -m anchor_mvp.data.automation `
        --config $configPath `
        --status-only
    exit $LASTEXITCODE
}

$promptedCredential = $false
$plainCredential = $null
try {
    $credential = [Environment]::GetEnvironmentVariable(
        $apiKeyEnv,
        [EnvironmentVariableTarget]::Process
    )
    if (-not $DryRun -and [string]::IsNullOrWhiteSpace($credential)) {
        if (-not $PromptForApiKey) {
            throw (
                "Credential environment variable '$apiKeyEnv' is not set in this process. " +
                "Set it for this launch or pass -PromptForApiKey."
            )
        }
        $secureCredential = Read-Host `
            -AsSecureString `
            -Prompt "Enter the provider credential for this launch only"
        $plainCredential = Convert-SecureStringToPlainText -SecureValue $secureCredential
        if ([string]::IsNullOrWhiteSpace($plainCredential)) {
            throw "The provider credential cannot be empty."
        }
        [Environment]::SetEnvironmentVariable(
            $apiKeyEnv,
            $plainCredential,
            [EnvironmentVariableTarget]::Process
        )
        $promptedCredential = $true
        $plainCredential = $null
    }

    $arguments = @(
        "-m", "anchor_mvp.data.automation",
        "--config", $configPath
    )
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    if (-not $NoWaitCooldown) {
        $arguments += "--wait-cooldown"
    }

    Write-Host "The visible process remains active across persisted rate-limit cooldowns."
    & $python.Executable @($python.PrefixArguments) @arguments
    $automationExitCode = $LASTEXITCODE
}
finally {
    $plainCredential = $null
    if ($promptedCredential) {
        [Environment]::SetEnvironmentVariable(
            $apiKeyEnv,
            $null,
            [EnvironmentVariableTarget]::Process
        )
    }
}
exit $automationExitCode
