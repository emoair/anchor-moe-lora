[CmdletBinding()]
param(
    [switch]$ConfirmDownload,
    [switch]$DirectPhysicalRoute,
    [string]$SourceAddress = ""
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Revision = '7074ef12ea2a6f70a228943c1336553333c22786'
$ExpectedBytes = [int64]106492326
$ExpectedSha256 = '0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69'
$RelativeTarget = "artifacts\swebench\source\SWE-bench__SWE-bench\$Revision\train-00000-of-00001.parquet"
$Target = Join-Path $RepoRoot $RelativeTarget
$Url = "https://huggingface.co/datasets/SWE-bench/SWE-bench/resolve/$Revision/data/train-00000-of-00001.parquet"

$Route = Get-NetRoute -AddressFamily IPv4 -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
    Sort-Object RouteMetric, InterfaceMetric |
    Select-Object -First 1 InterfaceAlias, InterfaceIndex, NextHop, RouteMetric, InterfaceMetric

if ($DirectPhysicalRoute) {
    if ([string]::IsNullOrWhiteSpace($SourceAddress)) {
        throw 'DirectPhysicalRoute requires -SourceAddress set to the physical NIC IPv4 address.'
    }
    $Binding = Get-NetIPAddress -AddressFamily IPv4 -IPAddress $SourceAddress -ErrorAction SilentlyContinue |
        Select-Object -First 1 InterfaceAlias, InterfaceIndex, IPAddress
    if ($null -eq $Binding) {
        throw "SourceAddress $SourceAddress is not assigned to a local IPv4 interface."
    }
    $PhysicalAdapter = Get-NetAdapter -Physical -ErrorAction Stop |
        Where-Object {
            $_.ifIndex -eq $Binding.InterfaceIndex -and $_.Status -eq 'Up'
        } |
        Select-Object -First 1 Name, ifIndex, Status, InterfaceDescription
    if ($null -eq $PhysicalAdapter) {
        throw (
            "SourceAddress $SourceAddress maps to interface index " +
            "$($Binding.InterfaceIndex), which is not an Up physical adapter. " +
            "Virtual, TUN, TAP, VPN, and disconnected adapters are refused."
        )
    }
    Write-Host (
        "Route mode : direct physical binding " +
        "($($PhysicalAdapter.Name), ifIndex $($PhysicalAdapter.ifIndex), $SourceAddress)"
    )
}
else {
    Write-Host "Route mode : system default ($($Route.InterfaceAlias), next hop $($Route.NextHop))"
}
Write-Host "Source     : SWE-bench/SWE-bench train @ $Revision"
Write-Host "Size       : $ExpectedBytes bytes ($([math]::Round($ExpectedBytes / 1MB, 2)) MiB)"
Write-Host "Target     : $Target"
Write-Host "SHA-256    : $ExpectedSha256"
Write-Host 'Scope      : train parquet only; this script never requests dev/test/heldout files.'

if (-not $ConfirmDownload) {
    Write-Host 'Dry run only. Re-run with -ConfirmDownload after checking size, target, and route.'
    exit 0
}

if (Test-Path -LiteralPath $Target) {
    $Existing = Get-Item -LiteralPath $Target
    $ExistingHash = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Existing.Length -eq $ExpectedBytes -and $ExistingHash -eq $ExpectedSha256) {
        Write-Host 'Pinned train parquet already exists and matches size/hash; no download needed.'
        exit 0
    }
    throw 'Target exists but does not match the pinned size/hash; move it aside and retry.'
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
$CurlArgs = @('--location', '--fail', '--retry', '3', '--retry-delay', '2')
if ($DirectPhysicalRoute) {
    $CurlArgs += @('--noproxy', '*', '--interface', $SourceAddress)
}
$CurlArgs += @('--output', $Target, $Url)
& curl.exe @CurlArgs
if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE"
}

$Item = Get-Item -LiteralPath $Target
$Hash = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Item.Length -ne $ExpectedBytes -or $Hash -ne $ExpectedSha256) {
    throw 'Downloaded train parquet failed pinned size/SHA-256 verification.'
}
Write-Host 'Download complete and verified.'
