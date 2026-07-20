[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ProfilePath,
    [string]$ManifestPath,
    [string]$BinaryPath,
    [string]$StateHome,
    [string]$BaseUrl,
    [string]$ModelId,
    [ValidateSet('openai_responses','openai_chat')][string]$Protocol,
    [ValidateSet('none','reasoning.effort','reasoning_effort')][string]$ReasoningField,
    [string]$ReasoningEffort,
    [string]$KeyEnv,
    [ValidateSet('direct','proxy','inherit')][string]$NetworkMode,
    [string]$ProxyUrlEnv,
    [int]$Port = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
function Resolve-RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}
function Set-ProcessEnv([string]$Name, [AllowNull()][string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
}
function Assert-PhysicalProviderRoute([string]$TargetBaseUrl) {
    if (-not (Get-Command Find-NetRoute -ErrorAction SilentlyContinue) -or
        -not (Get-Command Get-NetAdapter -ErrorAction SilentlyContinue)) {
        throw 'Physical-route preflight requires the Windows NetTCPIP and NetAdapter cmdlets'
    }
    $targetUri = [Uri]$TargetBaseUrl
    $addresses = @([System.Net.Dns]::GetHostAddresses($targetUri.DnsSafeHost) |
        Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork })
    if ($addresses.Count -eq 0) { throw 'Physical-route preflight could not resolve an IPv4 provider address' }
    $physicalIndexes = @(Get-NetAdapter -Physical -ErrorAction Stop |
        Where-Object { $_.Status -eq 'Up' } | ForEach-Object { [int]$_.InterfaceIndex })
    if ($physicalIndexes.Count -eq 0) { throw 'Physical-route preflight found no active physical adapter' }
    foreach ($address in $addresses) {
        $route = @(Find-NetRoute -RemoteIPAddress $address.IPAddressToString -ErrorAction Stop |
            Where-Object { $_.DestinationPrefix } | Select-Object -First 1)
        if ($route.Count -eq 0) { throw 'Physical-route preflight could not resolve the provider route' }
        $routeIndex = [int]$route[0].InterfaceIndex
        if ($physicalIndexes -notcontains $routeIndex) {
            $alias = [string]$route[0].InterfaceAlias
            throw "Direct mode is not physically direct: provider traffic currently selects non-physical interface '$alias' (ifIndex $routeIndex). Disable TUN/change routing, then retry. This launcher will not alter system routes."
        }
    }
}

$ProfilePath = Resolve-RepoPath $ProfilePath
if (-not $ManifestPath) { $ManifestPath = Join-Path $repoRoot 'artifacts\tooling\ccswitch-patched\route-manifest.json' }
$ManifestPath = Resolve-RepoPath $ManifestPath
& py (Join-Path $repoRoot 'scripts\tooling\validate_ccswitch_route.py') --manifest $ManifestPath --profile $ProfilePath --require-ready
if ($LASTEXITCODE -ne 0) { throw 'CC Switch route manifest/profile validation failed' }

$profile = Get-Content -LiteralPath $ProfilePath -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $BinaryPath) { $BinaryPath = $manifest.binary.path }
$BinaryPath = Resolve-RepoPath $BinaryPath
if (-not (Test-Path -LiteralPath $BinaryPath)) { throw 'Patched CC Switch binary is missing' }
$actualBinarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $BinaryPath).Hash.ToLowerInvariant()
if ($actualBinarySha -ne $manifest.binary.sha256) { throw 'Patched CC Switch binary hash mismatch' }

if (-not $BaseUrl) { $BaseUrl = [string]$profile.base_url }
if (-not $Protocol) { $Protocol = [string]$profile.protocol }
if (-not $KeyEnv) { $KeyEnv = [string]$profile.key_env }
if (-not $ReasoningField) { $ReasoningField = [string]$profile.reasoning.field }
if (-not $ReasoningEffort) { $ReasoningEffort = [string]$profile.reasoning.effort }
if (-not $NetworkMode) {
    $NetworkMode = if ($profile.network -and $profile.network.mode) { [string]$profile.network.mode } else { 'direct' }
}
if (-not $ProxyUrlEnv) {
    $ProxyUrlEnv = if ($profile.network -and $profile.network.proxy_url_env) { [string]$profile.network.proxy_url_env } else { 'ANCHOR_ROUTE_UPSTREAM_PROXY_URL' }
}
$requirePhysicalRoute = [bool]($profile.network -and $profile.network.require_physical_route)
if ($Port -eq 0) { $Port = [int]$profile.route.port }
if (-not $StateHome) { $StateHome = Join-Path $repoRoot ("runs\ccswitch-anchor-route\" + $profile.profile_id) }
$StateHome = Resolve-RepoPath $StateHome

$credential = [Environment]::GetEnvironmentVariable($KeyEnv, 'Process')
if ([string]::IsNullOrWhiteSpace($credential)) {
    throw "Credential env '$KeyEnv' is not set in this process"
}

$proxyUrl = $null
switch ($NetworkMode) {
    'direct' {
        foreach ($name in @('HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy')) {
            Set-ProcessEnv $name $null
        }
        Set-ProcessEnv 'NO_PROXY' '*'
        Set-ProcessEnv 'no_proxy' '*'
        if ($requirePhysicalRoute) { Assert-PhysicalProviderRoute $BaseUrl }
    }
    'proxy' {
        if ($ProxyUrlEnv -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            throw 'Proxy URL environment variable name is invalid'
        }
        $proxyUrl = [Environment]::GetEnvironmentVariable($ProxyUrlEnv, 'Process')
        if ([string]::IsNullOrWhiteSpace($proxyUrl)) {
            throw "Network mode proxy requires env '$ProxyUrlEnv' in this process"
        }
        $parsedProxy = $null
        if (-not [Uri]::TryCreate($proxyUrl, [UriKind]::Absolute, [ref]$parsedProxy) -or
            $parsedProxy.Scheme -notin @('http','https','socks5','socks5h')) {
            throw 'Proxy URL must be an absolute http, https, socks5, or socks5h URL'
        }
    }
    'inherit' { }
}

$selection = $profile.model_selection
$manualModel = if ($ModelId) { $ModelId } else { [string]$selection.manual_model_id }
$forceManual = [bool]$selection.force_manual_model -or [bool]$ModelId
$mode = [string]$selection.mode
$resolvedModel = $manualModel
if (-not $forceManual -and $mode -ne 'manual') {
    try {
        $discoveryPath = if ($selection.discovery_path) { [string]$selection.discovery_path } else { '/models' }
        $modelsUrl = $BaseUrl.TrimEnd('/') + '/' + $discoveryPath.TrimStart('/')
        $headers = @{ Authorization = 'Bearer ' + $credential }
        $invokeParams = @{ Method = 'Get'; Uri = $modelsUrl; Headers = $headers; TimeoutSec = 60 }
        if ($NetworkMode -eq 'proxy') { $invokeParams.Proxy = $proxyUrl }
        elseif ($NetworkMode -eq 'direct' -and (Get-Command Invoke-RestMethod).Parameters.ContainsKey('NoProxy')) {
            $invokeParams.NoProxy = $true
        }
        $response = Invoke-RestMethod @invokeParams
        $ids = @()
        if ($response.data) { $ids += @($response.data | ForEach-Object { $_.id }) }
        if ($response.models) { $ids += @($response.models | ForEach-Object { if ($_.id) { $_.id } elseif ($_.name) { $_.name } }) }
        $ids = @($ids | Where-Object { $_ } | Select-Object -Unique)
        $pattern = [string]$selection.preferred_model_pattern
        $matches = if ($pattern) { @($ids | Where-Object { $_ -match $pattern }) } else { $ids }
        if ($matches.Count -eq 1) { $resolvedModel = [string]$matches[0] }
        elseif ($ids -contains $manualModel) { $resolvedModel = $manualModel }
        else { throw 'Model discovery was ambiguous or did not contain the manual fallback ID' }
    } catch {
        if ($mode -eq 'discover') { throw }
        Write-Warning 'Model discovery failed; using the explicit manual model ID.'
        $resolvedModel = $manualModel
    }
}

New-Item -ItemType Directory -Force -Path $StateHome | Out-Null
Set-ProcessEnv 'CC_SWITCH_TEST_HOME' $StateHome
Set-ProcessEnv 'ANCHOR_ROUTE_ENABLED' '1'
Set-ProcessEnv 'ANCHOR_ROUTE_BASE_URL' $BaseUrl
Set-ProcessEnv 'ANCHOR_ROUTE_MODEL' $resolvedModel
Set-ProcessEnv 'ANCHOR_ROUTE_API_FORMAT' $Protocol
Set-ProcessEnv 'ANCHOR_ROUTE_API_KEY_ENV' $KeyEnv
Set-ProcessEnv 'ANCHOR_ROUTE_REASONING_FIELD' $ReasoningField
Set-ProcessEnv 'ANCHOR_ROUTE_REASONING_EFFORT' $ReasoningEffort
Set-ProcessEnv 'ANCHOR_ROUTE_LISTEN_ADDRESS' ([string]$profile.route.listen_address)
Set-ProcessEnv 'ANCHOR_ROUTE_PORT' ([string]$Port)
Set-ProcessEnv 'ANCHOR_ROUTE_MAX_RETRIES' ([string]$profile.route.max_retries)
Set-ProcessEnv 'ANCHOR_ROUTE_USER_AGENT' ([string]$profile.route.user_agent)
Set-ProcessEnv 'ANCHOR_ROUTE_NETWORK_MODE' $NetworkMode
Set-ProcessEnv 'ANCHOR_ROUTE_PROXY_URL_ENV' $ProxyUrlEnv
if ($profile.pricing) {
    Set-ProcessEnv 'ANCHOR_ROUTE_PRICE_INPUT_PER_MILLION' ([string]$profile.pricing.input_per_million)
    Set-ProcessEnv 'ANCHOR_ROUTE_PRICE_OUTPUT_PER_MILLION' ([string]$profile.pricing.output_per_million)
    Set-ProcessEnv 'ANCHOR_ROUTE_PRICE_CACHE_READ_PER_MILLION' ([string]$profile.pricing.cache_read_per_million)
    Set-ProcessEnv 'ANCHOR_ROUTE_PRICE_CACHE_CREATION_PER_MILLION' ([string]$profile.pricing.cache_creation_per_million)
}

Write-Host "Starting isolated route: profile=$($profile.profile_id), protocol=$Protocol, model=$resolvedModel, reasoning=$ReasoningField/$ReasoningEffort, network=$NetworkMode, port=$Port"
& $BinaryPath
exit $LASTEXITCODE
