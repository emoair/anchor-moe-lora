param(
    [string]$BaseModel = "",
    [Parameter(Mandatory = $true)]
    [string]$PlannerAdapter,
    [Parameter(Mandatory = $true)]
    [string]$ToolPolicyAdapter,
    [Parameter(Mandatory = $true)]
    [string]$FrontendAdapter,
    [Parameter(Mandatory = $true)]
    [string]$ReviewAdapter,
    [Parameter(Mandatory = $true)]
    [string]$SecurityAdapter,
    [Parameter(Mandatory = $true)]
    [string]$MixedAdapter,
    [string]$Distro = "Ubuntu-22.04",
    [ValidateSet("3080ti-safe", "throughput")]
    [string]$Profile = "3080ti-safe",
    [ValidateSet("bitsandbytes", "compressed-tensors")]
    [string]$Quantization = "bitsandbytes",
    [string]$LoadFormat = "",
    [string]$ApiKey = "",
    [int]$Port = 8000,
    [int]$MaxModelLength = 0,
    [double]$GpuMemoryUtilization = 0.88,
    [switch]$PrintCommand
)

$ErrorActionPreference = "Stop"

if ($MaxModelLength -notin @(0, 1024, 2048)) {
    throw "MaxModelLength must be 1024 or 2048 for the RTX 3080 Ti profiles. Use 0 for the profile default."
}

if (-not $BaseModel) {
    $BaseModel = Join-Path $PSScriptRoot "..\..\models\google-gemma-4-12B-base"
}

$Quantization = $Quantization.ToLowerInvariant()
if (-not $LoadFormat) {
    $LoadFormat = if ($Quantization -eq "bitsandbytes") { "bitsandbytes" } else { "auto" }
} else {
    $LoadFormat = $LoadFormat.ToLowerInvariant()
}
if ($LoadFormat -notin @("bitsandbytes", "auto")) {
    throw "LoadFormat must be bitsandbytes or auto."
}
if ($Quantization -eq "bitsandbytes" -and $LoadFormat -ne "bitsandbytes") {
    throw "bitsandbytes in-flight quantization requires LoadFormat=bitsandbytes in this launcher."
}
if ($Quantization -eq "compressed-tensors" -and $LoadFormat -ne "auto") {
    throw "compressed-tensors requires LoadFormat=auto in this launcher."
}

function Convert-ToWslPath([string]$Path) {
    # Forward slashes survive PowerShell -> wsl.exe argument translation reliably.
    $PortablePath = $Path.Replace('\', '/')
    $Converted = & wsl.exe -d $Distro -- wslpath -a $PortablePath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not convert path for WSL: $Path"
    }
    return $Converted.Trim()
}

$BashScript = Convert-ToWslPath (Join-Path $PSScriptRoot "start_vllm_wsl.sh")
$PlannerWsl = Convert-ToWslPath $PlannerAdapter
$ToolPolicyWsl = Convert-ToWslPath $ToolPolicyAdapter
$FrontendWsl = Convert-ToWslPath $FrontendAdapter
$ReviewWsl = Convert-ToWslPath $ReviewAdapter
$SecurityWsl = Convert-ToWslPath $SecurityAdapter
$MixedWsl = Convert-ToWslPath $MixedAdapter

# Preserve a Hub id, but translate an existing local Windows checkpoint path.
$BaseModelWsl = $BaseModel
if (Test-Path -LiteralPath $BaseModel) {
    $BaseModelWsl = Convert-ToWslPath $BaseModel
}

$BashArgs = @(
    $BashScript,
    "--base-model", $BaseModelWsl,
    "--planner-adapter", $PlannerWsl,
    "--tool-policy-adapter", $ToolPolicyWsl,
    "--frontend-adapter", $FrontendWsl,
    "--review-adapter", $ReviewWsl,
    "--security-adapter", $SecurityWsl,
    "--mixed-adapter", $MixedWsl,
    "--profile", $Profile,
    "--quantization", $Quantization,
    "--load-format", $LoadFormat,
    "--port", $Port,
    "--gpu-memory-utilization", $GpuMemoryUtilization
)
if ($MaxModelLength -gt 0) {
    $BashArgs += @("--max-model-len", $MaxModelLength)
}
if ($ApiKey) {
    $BashArgs += @("--api-key", $ApiKey)
}
if ($PrintCommand) {
    $BashArgs += "--print-command"
}

Write-Host "Delegating vLLM startup to WSL2 distro $Distro."
& wsl.exe -d $Distro -- bash @BashArgs
exit $LASTEXITCODE
