param(
    [string]$BaseModel = "",
    [string]$Distro = "Ubuntu-22.04",
    [string]$WslUser = "",
    [string]$VllmBinDir = "",
    [string]$ApiKey = "",
    [int]$Port = 8000,
    [ValidateSet(1024, 2048)]
    [int]$MaxModelLength = 2048,
    [ValidateRange(0.5, 0.88)]
    [double]$GpuMemoryUtilization = 0.82,
    [switch]$PrintCommand
)

$ErrorActionPreference = "Stop"
if ($PrintCommand -and $ApiKey) {
    throw "PrintCommand refuses ApiKey to prevent command-line secret disclosure"
}
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ExpectedManifestSha256 = "96ebac04f4d2c64d4b21142bb6e05d94656c3c7fb243fdbd43c4b4457eca0156"
$ExpectedFlatBridgeManifestSha256 = "3a7c29b79da450e324b484b95707eb7c7ddbf52f80f5f3fd4b4843367839cd49"
$ExpectedFlatDerivedManifestSha256 = "a4f318a194af7d5437032c992cfcf2f8f6c6afc4a16e9c5d08087b6306f041db"
if (-not $BaseModel) {
    $BaseModel = Join-Path $ProjectRoot "models\google-gemma-4-12B-bnb-nf4"
}
$BaseModel = [IO.Path]::GetFullPath($BaseModel)
$Manifest = Join-Path $BaseModel "anchor_quantization_manifest.json"
$LegacyBridgeManifest = Join-Path $BaseModel "anchor_vllm_u8_bridge_manifest.json"
$FlatBridgeManifest = Join-Path $BaseModel "anchor_vllm_u8_flat_bridge_manifest.json"
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Frozen NF4 quantization manifest is missing: $Manifest"
}
if (Test-Path -LiteralPath $FlatBridgeManifest -PathType Leaf) {
    $ObservedFlatBridgeManifestSha256 = (
        Get-FileHash -LiteralPath $FlatBridgeManifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($ObservedFlatBridgeManifestSha256 -ne $ExpectedFlatBridgeManifestSha256) {
        throw "Derived vLLM U8 flat-v2 bridge manifest SHA-256 mismatch"
    }
    $ObservedManifestSha256 = (
        Get-FileHash -LiteralPath $Manifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($ObservedManifestSha256 -ne $ExpectedFlatDerivedManifestSha256) {
        throw "Derived vLLM U8 flat-v2 quantization manifest SHA-256 mismatch"
    }
    $BridgeAudit = Get-Content -LiteralPath $FlatBridgeManifest -Raw | ConvertFrom-Json
    $DerivedConfig = Join-Path $BaseModel "config.json"
    $ObservedConfigSha256 = (
        Get-FileHash -LiteralPath $DerivedConfig -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $BridgeAudit.schema_version -ne "anchor.vllm-bnb-nf4-u8-flat-bridge.v2" -or
        $BridgeAudit.derived.quant_storage -ne "uint8" -or
        $BridgeAudit.conversion.shape_transform -ne "BF16.view(U8).reshape(-1,1)" -or
        $BridgeAudit.conversion.converted_tensor_count -ne 331 -or
        $BridgeAudit.verification.expected_converted_tensor_count -ne 331 -or
        -not $BridgeAudit.verification.all_tensor_payload_bytes_equal -or
        -not $BridgeAudit.verification.all_converted_shapes_flat_column -or
        $ObservedConfigSha256 -ne $BridgeAudit.derived.config_sha256
    ) {
        throw "Derived vLLM U8 flat-v2 bridge contract verification failed"
    }
} elseif (Test-Path -LiteralPath $LegacyBridgeManifest -PathType Leaf) {
    throw "Legacy derived-vllm-u8 v1 is serving-incompatible: packed weights are [N,2], not vLLM flat columns; use derived-vllm-u8-flat-v2"
} else {
    $ObservedManifestSha256 = (
        Get-FileHash -LiteralPath $Manifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($ObservedManifestSha256 -ne $ExpectedManifestSha256) {
        throw "Frozen NF4 quantization manifest SHA-256 mismatch"
    }
}

if (-not $WslUser) {
    $DetectedWslUser = & wsl.exe -d $Distro -- id -un
    if ($LASTEXITCODE -ne 0 -or -not $DetectedWslUser) {
        throw "Could not detect the default user for WSL distro $Distro"
    }
    $WslUser = $DetectedWslUser.Trim()
}
if ($WslUser -notmatch '^[A-Za-z_][A-Za-z0-9_-]*$') {
    throw "WslUser contains unsupported characters"
}

if (-not $VllmBinDir) {
    $WslHome = if ($WslUser -eq "root") { "/root" } else { "/home/$WslUser" }
    $VllmBinDir = "$WslHome/.venvs/anchor-vllm/bin"
}
if (-not $VllmBinDir.StartsWith("/") -or $VllmBinDir.Contains(":")) {
    throw "VllmBinDir must be an absolute WSL path without ':'"
}
$VllmExecutable = "$VllmBinDir/vllm"
& wsl.exe -d $Distro -u $WslUser -- test -x $VllmExecutable
if ($LASTEXITCODE -ne 0) {
    throw "vLLM executable is missing or not executable: $VllmExecutable; configure -VllmBinDir if the serving environment is elsewhere"
}

# Keep the serving environment first while retaining WSL's NVIDIA bridge tools.
# This explicit PATH also works for non-login shells launched by wsl.exe.
$WslPath = "$VllmBinDir`:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

function Convert-ToWslPath([string]$Path) {
    $PortablePath = $Path.Replace('\', '/')
    $Converted = & wsl.exe -d $Distro -- wslpath -a $PortablePath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not convert path for WSL: $Path"
    }
    return $Converted.Trim()
}

$BashScript = Convert-ToWslPath (Join-Path $PSScriptRoot "start_formal_af_serial_vllm_wsl.sh")
$BaseModelWsl = Convert-ToWslPath $BaseModel
$Arguments = @(
    $BashScript,
    "--base-model", $BaseModelWsl,
    "--port", $Port,
    "--max-model-len", $MaxModelLength,
    "--gpu-memory-utilization", $GpuMemoryUtilization
)
if ($ApiKey) {
    $Arguments += @("--api-key", $ApiKey)
}
if ($PrintCommand) {
    $Arguments += "--print-command"
    Write-Host "WSL runtime: distro=$Distro user=$WslUser PATH=$WslPath"
}

Write-Host "Delegating one-active-LoRA formal server to WSL2 distro $Distro."
& wsl.exe -d $Distro -u $WslUser -- env "PATH=$WslPath" bash @Arguments
exit $LASTEXITCODE
