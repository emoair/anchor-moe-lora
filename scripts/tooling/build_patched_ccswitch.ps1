[CmdletBinding()]
param(
    [string]$Worktree,
    [string]$CargoTargetDir,
    [switch]$StaticOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$patchPath = Join-Path $repoRoot 'patches\cc-switch\v3.16.5-anchor-opencode-route.patch'
$manifestPath = Join-Path $repoRoot 'artifacts\tooling\ccswitch-patched\route-manifest.json'
$artifactDir = Split-Path -Parent $manifestPath
$pinnedCommit = '8d1b3306d09a27b9d8fc29694791d8421aba5f93'
$upstreamUrl = 'https://github.com/farion1231/cc-switch.git'
if (-not $Worktree) {
    $Worktree = Join-Path $repoRoot 'runs\cc-switch-build\patched-v3.16.5'
}
$Worktree = [IO.Path]::GetFullPath($Worktree)

New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $Worktree '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Worktree) | Out-Null
    & git clone --filter=blob:none --no-checkout $upstreamUrl $Worktree
    if ($LASTEXITCODE -ne 0) { throw 'Failed to clone pinned CC Switch source' }
    & git -C $Worktree checkout --detach $pinnedCommit
    if ($LASTEXITCODE -ne 0) { throw 'Failed to checkout pinned CC Switch commit' }
}

$head = (& git -C $Worktree rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $pinnedCommit) {
    throw "CC Switch source is not at pinned commit $pinnedCommit"
}

$anchorSource = Join-Path $Worktree 'src-tauri\src\anchor_route.rs'
if (-not (Test-Path -LiteralPath $anchorSource)) {
    & git -C $Worktree apply --check $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Anchor route patch does not apply cleanly' }
    & git -C $Worktree apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Failed to apply Anchor route patch' }
} else {
    & git -C $Worktree apply --reverse --check $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Existing worktree does not exactly contain the pinned patch' }
}

$patchSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $patchPath).Hash.ToLowerInvariant()
if ($StaticOnly) {
    Write-Host 'Pinned patch applies cleanly. Static-only mode leaves route ready=false.'
    exit 0
}

$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo) {
    throw 'Rust/Cargo is not installed. The route remains fail-closed (ready=false).'
}

$cargoManifest = Join-Path $Worktree 'src-tauri\Cargo.toml'
if (-not $CargoTargetDir) {
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        throw 'LOCALAPPDATA is required for the Windows Cargo target directory.'
    }
    $CargoTargetDir = Join-Path $env:LOCALAPPDATA (
        'anchor-moe-lora\ccswitch-build\' + $patchSha.Substring(0, 16)
    )
}
$CargoTargetDir = [IO.Path]::GetFullPath($CargoTargetDir)
New-Item -ItemType Directory -Force -Path $CargoTargetDir | Out-Null
$previousCargoTargetDir = $env:CARGO_TARGET_DIR
$env:CARGO_TARGET_DIR = $CargoTargetDir

function Invoke-CargoChecked {
    param(
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [Parameter(Mandatory=$true)][string]$FailureMessage
    )
    & $cargo.Source @Arguments
    $cargoExitCode = $LASTEXITCODE
    if ($cargoExitCode -ne 0) {
        throw "$FailureMessage (cargo exit=$cargoExitCode)"
    }
}

try {
    Invoke-CargoChecked `
        -Arguments @('test','--manifest-path',$cargoManifest,'--lib','anchor_route') `
        -FailureMessage 'Anchor route Rust tests failed'
    Invoke-CargoChecked `
        -Arguments @('test','--manifest-path',$cargoManifest,'--lib','test_extract_auth_from_named_process_environment') `
        -FailureMessage 'Runtime environment credential test failed'
    Invoke-CargoChecked `
        -Arguments @('build','--release','--manifest-path',$cargoManifest,'--bin','anchor-opencode-route') `
        -FailureMessage 'Anchor route binary build failed'
}
finally {
    $env:CARGO_TARGET_DIR = $previousCargoTargetDir
}

$extension = if ($IsWindows -or $env:OS -eq 'Windows_NT') { '.exe' } else { '' }
$builtBinary = Join-Path $CargoTargetDir "release\anchor-opencode-route$extension"
if (-not (Test-Path -LiteralPath $builtBinary)) { throw 'Built route binary is missing' }
& py (Join-Path $repoRoot 'tests\ccswitch_anchor_route_behavior.py') --binary $builtBinary
if ($LASTEXITCODE -ne 0) { throw 'Anchor route binary behavior smoke failed' }
$artifactBinary = Join-Path $artifactDir "anchor-opencode-route$extension"
Copy-Item -LiteralPath $builtBinary -Destination $artifactBinary -Force
$binarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifactBinary).Hash.ToLowerInvariant()
$relativeBinary = "artifacts/tooling/ccswitch-patched/anchor-opencode-route$extension"

$manifest = [ordered]@{
    schema_version = 'anchor.ccswitch-route-manifest.v1'
    ready = $true
    upstream = [ordered]@{ repository = $upstreamUrl; tag = 'v3.16.5'; commit = $pinnedCommit }
    patch = [ordered]@{ path = 'patches/cc-switch/v3.16.5-anchor-opencode-route.patch'; sha256 = $patchSha }
    binary = [ordered]@{ path = $relativeBinary; sha256 = $binarySha }
    route = [ordered]@{
        app_type = 'anchor-opencode'
        base_url = 'http://127.0.0.1:15731/anchor/v1'
        health_url = 'http://127.0.0.1:15731/anchor/health'
        status_url = 'http://127.0.0.1:15731/anchor/status'
        content_free_health_status = $true
        default_network_mode = 'direct'
        supported_network_modes = @('direct','proxy','inherit')
    }
    secret_persisted = $false
    runtime_config = @('protocol','base_url','model_id','model_discovery','reasoning','pricing','key_env','network_policy','port','retries','user_agent')
    verified_tests = @(
        [ordered]@{ name = 'cargo-test-anchor-route'; status = 'passed' },
        [ordered]@{ name = 'cargo-test-runtime-env-credential'; status = 'passed' },
        [ordered]@{ name = 'cargo-release-build'; status = 'passed' },
        [ordered]@{ name = 'binary-behavior-smoke'; status = 'passed' }
    )
}
$json = $manifest | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($manifestPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Write-Host "Built and verified: $artifactBinary"
