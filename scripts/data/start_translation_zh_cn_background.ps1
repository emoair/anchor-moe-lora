[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_]*$')]
    [string]$ApiKeyEnv,

    [ValidatePattern('^[0-3](?:[,\s]+[0-3])*$')]
    [string]$Parts = '0,1,2,3',

    [ValidateRange(1, 7)]
    [int]$ConcurrencyPerPart = 4,

    [ValidateRange(0, 10)]
    [int]$MaxRetries = 2,

    [ValidateRange(0, 10)]
    [int]$RowMaxRetries = 2,

    [ValidateRange(30, 3600)]
    [int]$TimeoutSeconds = 600,

    [ValidateRange(256, 32768)]
    [int]$MaxTokens = 4096,

    [ValidateRange(1, 1000000)]
    [int]$MaxRequestsPerPart = 5000,

    [ValidateRange(1000, 100000000)]
    [int]$MaxOutputTokensPerPart = 5000000,

    [ValidateRange(1, 1000)]
    [int]$ProgressEvery = 10,

    [string]$Provider = 'custom-openai-responses',
    [ValidateSet('openai_responses')]
    [string]$Protocol = 'openai_responses',
    [string]$BaseUrl = 'https://ark.cn-beijing.volces.com/api/coding/v3',
    [string]$Model = 'glm-5-2-260617',
    [string]$Python = 'C:\Users\Air\AppData\Local\Programs\Python\Python310\python.exe',
    [switch]$PromptForApiKey,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PrepareScript = Join-Path $RepoRoot 'scripts\data\translation_zh_cn.py'
$Runner = Join-Path $RepoRoot 'scripts\data\run_translation_zh_cn.py'
$WorkRoot = Join-Path $RepoRoot 'artifacts\compact_mvp_v2b\translation_zh_cn_v1'
$ShardDir = Join-Path $WorkRoot 'shards'
$JournalDir = Join-Path $WorkRoot 'journals'
$LogDir = Join-Path $WorkRoot 'logs'
$LaunchManifest = Join-Path $WorkRoot 'background_launch.json'
$SelectedParts = @($Parts -split '[,\s]+' | ForEach-Object { [int]$_ } | Sort-Object -Unique)
if ($SelectedParts.Count -eq 0) {
    throw 'At least one translation part must be selected'
}

foreach ($required in @($Python, $PrepareScript, $Runner)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required launcher dependency is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $WorkRoot, $LogDir | Out-Null

# Preparation is deterministic and refuses source-snapshot drift. It never
# discovers inputs outside the allowlisted compact-MVP registry.
$WorkManifest = Join-Path $ShardDir 'work_manifest.json'
if (-not (Test-Path -LiteralPath $WorkManifest -PathType Leaf)) {
    & $Python $PrepareScript prepare | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Translation shard preparation failed with exit code $LASTEXITCODE"
    }
}

# Validate all four source bindings and any resume journals before checking a
# credential or starting network work. Provider values are not needed here.
$preflights = @()
foreach ($part in $SelectedParts) {
    $preflightJson = & $Python $Runner --part $part --preflight
    if ($LASTEXITCODE -ne 0) {
        throw "Translation preflight failed for part $part with exit code $LASTEXITCODE"
    }
    $preflight = ($preflightJson -join "`n") | ConvertFrom-Json
    if ($preflight.status -ne 'preflight_ok') {
        throw "Translation preflight did not pass for part $part"
    }
    $preflights += $preflight
}

if ($PreflightOnly) {
    [pscustomobject]@{
        status = 'preflight_ok'
        total_concurrency = $SelectedParts.Count * $ConcurrencyPerPart
        parts = $preflights
        work_root = $WorkRoot
        heldout_content_read = $false
        benchmark_record_content_read = $false
    } | ConvertTo-Json -Depth 8
    exit 0
}

# Read the credential only from the inherited process environment. Never put
# its value in an argument, file, manifest, log, or PowerShell output.
$promptedCredential = $false
$credential = [Environment]::GetEnvironmentVariable($ApiKeyEnv, 'Process')
if ([string]::IsNullOrWhiteSpace($credential)) {
    if (-not $PromptForApiKey) {
        throw "The process environment variable '$ApiKeyEnv' is not set"
    }
    $secureCredential = Read-Host `
        "Enter the translation API credential for this launch only" `
        -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureCredential)
    try {
        $credential = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        if ([string]::IsNullOrWhiteSpace($credential)) {
            throw 'The prompted translation API credential was empty'
        }
        [Environment]::SetEnvironmentVariable($ApiKeyEnv, $credential, 'Process')
        $promptedCredential = $true
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        $secureCredential.Dispose()
    }
}
$credential = $null

try {
    # Fail closed rather than launching duplicate writers against the same journals.
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(?:\.exe)?$' -and
        $_.CommandLine -match 'run_translation_zh_cn\.py'
    }
    foreach ($process in $active) {
        foreach ($part in $SelectedParts) {
            if ($process.CommandLine -match "(?:^|\s)--part\s+$part(?:\s|$)") {
                throw "A translation runner is already active for part $part"
            }
        }
    }

    $startedAt = (Get-Date).ToUniversalTime().ToString('o')
    # Keep live, unselected runners in the replacement manifest. This permits
    # a failed subset to be resumed without making an already-running part
    # disappear from the control-plane state or launching a duplicate writer.
    $processes = @(
        foreach ($process in $active) {
            if ($process.CommandLine -notmatch '(?:^|\s)--part\s+(\d)(?:\s|$)') {
                continue
            }
            $activePart = [int]$Matches[1]
            if ($SelectedParts -contains $activePart) {
                continue
            }
            [pscustomobject]@{
                part = $activePart
                pid = [int]$process.ProcessId
                status = 'running'
                stdout = Join-Path $LogDir ("part-{0:d3}.stdout.log" -f $activePart)
                stderr = Join-Path $LogDir ("part-{0:d3}.stderr.log" -f $activePart)
                shard = Join-Path $ShardDir ("part-{0:d3}.jsonl" -f $activePart)
                journal = Join-Path $JournalDir ("part-{0:d3}.jsonl" -f $activePart)
            }
        }
    )
    foreach ($part in $SelectedParts) {
        $stdout = Join-Path $LogDir ("part-{0:d3}.stdout.log" -f $part)
        $stderr = Join-Path $LogDir ("part-{0:d3}.stderr.log" -f $part)
        $arguments = @(
            $Runner,
            '--part', [string]$part,
            '--shard-dir', $ShardDir,
            '--journal-dir', $JournalDir,
            '--concurrency', [string]$ConcurrencyPerPart,
            '--progress-every', [string]$ProgressEvery,
            '--row-max-retries', [string]$RowMaxRetries,
            '--provider', $Provider,
            '--protocol', $Protocol,
            '--base-url', $BaseUrl,
            '--model', $Model,
            '--api-key-env', $ApiKeyEnv,
            '--max-requests', [string]$MaxRequestsPerPart,
            '--max-output-tokens-total', [string]$MaxOutputTokensPerPart,
            '--max-tokens', [string]$MaxTokens,
            '--timeout-seconds', [string]$TimeoutSeconds,
            '--max-retries', [string]$MaxRetries
        )
        $process = Start-Process -FilePath $Python `
            -ArgumentList $arguments `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $processes += [pscustomobject]@{
            part = $part
            pid = $process.Id
            status = 'running'
            stdout = $stdout
            stderr = $stderr
            shard = Join-Path $ShardDir ("part-{0:d3}.jsonl" -f $part)
            journal = Join-Path $JournalDir ("part-{0:d3}.jsonl" -f $part)
        }
    }

    $manifestParts = @($processes | ForEach-Object { [int]$_.part } | Sort-Object -Unique)
    $manifest = [ordered]@{
        schema_version = 'anchor.translation-background-launch.v1'
        status = 'running'
        started_at = $startedAt
        provider = $Provider
        protocol = $Protocol
        base_url = $BaseUrl
        model = $Model
        api_key_env = $ApiKeyEnv
        credential_persisted = $false
        parts = $manifestParts
        part_count = $manifestParts.Count
        concurrency_per_part = $ConcurrencyPerPart
        total_concurrency = $manifestParts.Count * $ConcurrencyPerPart
        max_retries = $MaxRetries
        row_max_retries = $RowMaxRetries
        timeout_seconds = $TimeoutSeconds
        max_tokens = $MaxTokens
        max_requests_per_part = $MaxRequestsPerPart
        max_output_tokens_per_part = $MaxOutputTokensPerPart
        work_root = $WorkRoot
        processes = $processes
        heldout_content_read = $false
        benchmark_record_content_read = $false
    }
    $temporaryManifest = "$LaunchManifest.tmp"
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryManifest -Encoding UTF8
    Move-Item -LiteralPath $temporaryManifest -Destination $LaunchManifest -Force
}
finally {
    if ($promptedCredential) {
        [Environment]::SetEnvironmentVariable($ApiKeyEnv, $null, 'Process')
    }
}

$manifest | ConvertTo-Json -Depth 8
