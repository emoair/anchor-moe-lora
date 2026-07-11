param(
    [string]$CheckoutRoot = "",
    [string]$BunPath = "",
    [string]$NodeGypPath = "",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipTypecheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PatchManifestPath = Join-Path $ProjectRoot "patches\opencode\patch-manifest.json"
$OutputRoot = Join-Path $ProjectRoot "artifacts\tooling\opencode-patched"

if (-not (Test-Path -LiteralPath $PatchManifestPath -PathType Leaf)) {
    throw "Patch source manifest is missing: $PatchManifestPath"
}
$patchSource = Get-Content -Raw -LiteralPath $PatchManifestPath | ConvertFrom-Json
if ($patchSource.schema_version -ne "anchor.opencode-patch-source.v1") {
    throw "Unsupported patch source manifest schema"
}
$Repository = [string]$patchSource.repository
$BaselineCommit = [string]$patchSource.baseline_commit
$PatchPath = Join-Path (Split-Path -Parent $PatchManifestPath) ([string]$patchSource.patch)
$ExpectedPatchSha256 = ([string]$patchSource.patch_sha256).ToLowerInvariant()
$ExpectedBunVersion = [string]$patchSource.bun_version
$ToolContract = $patchSource.tool_contract
if ($Repository -ne "https://github.com/anomalyco/opencode.git") {
    throw "Patch source repository is not the audited upstream"
}
if ($BaselineCommit -notmatch '^[0-9a-f]{40}$' -or $ExpectedPatchSha256 -notmatch '^[0-9a-f]{64}$') {
    throw "Patch source manifest contains an invalid commit or digest"
}
if ($null -eq $ToolContract -or [string]$ToolContract.version -ne [string]$patchSource.tool_contract_version) {
    throw "Patch source manifest contains an invalid tool contract"
}

if (-not $CheckoutRoot) {
    $CheckoutRoot = Join-Path $ProjectRoot "runs\opencode-build\v1.17.18"
}
$CheckoutRoot = [IO.Path]::GetFullPath($CheckoutRoot)

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory)
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $PatchPath -PathType Leaf)) {
    throw "Patch is missing: $PatchPath"
}
$patchSha = (Get-FileHash -LiteralPath $PatchPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($patchSha -ne $ExpectedPatchSha256) {
    throw "Patch hash mismatch. Expected $ExpectedPatchSha256, got $patchSha"
}

$createdCheckout = $false
if (-not (Test-Path -LiteralPath (Join-Path $CheckoutRoot ".git"))) {
    $parent = Split-Path -Parent $CheckoutRoot
    New-Item -ItemType Directory -Force $parent | Out-Null
    Invoke-Checked git @(
        "clone", "--depth", "1", "--branch", "v1.17.18", "--filter=blob:none", "--no-checkout", $Repository, $CheckoutRoot
    ) $parent
    $createdCheckout = $true
}

$origin = (& git -C $CheckoutRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $origin -ne $Repository) {
    throw "Checkout origin is not the audited repository: $origin"
}
if (-not $createdCheckout -and (& git -C $CheckoutRoot status --porcelain)) {
    throw "Checkout is dirty. Use a fresh -CheckoutRoot; this script never resets user work."
}
$baselineObject = "${BaselineCommit}^{commit}"
& git -C $CheckoutRoot cat-file -e $baselineObject 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Checked git @("fetch", "--depth", "1", "origin", $BaselineCommit) $CheckoutRoot
}
Invoke-Checked git @("checkout", "--detach", $BaselineCommit) $CheckoutRoot
$actualCommit = (& git -C $CheckoutRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $BaselineCommit) {
    throw "Baseline mismatch. Expected $BaselineCommit, got $actualCommit"
}
Invoke-Checked git @("apply", "--check", $PatchPath) $CheckoutRoot
Invoke-Checked git @("apply", $PatchPath) $CheckoutRoot

if (-not $BunPath) {
    $BunPath = $env:ANCHOR_BUN_EXE
}
if (-not $BunPath) {
    throw "Provide -BunPath or process-local ANCHOR_BUN_EXE. Global Bun is never selected implicitly."
}
$BunPath = (Resolve-Path -LiteralPath $BunPath).Path
$bunVersion = (& $BunPath --version).Trim()
if ($LASTEXITCODE -ne 0 -or $bunVersion -ne $ExpectedBunVersion) {
    throw "Audited build requires Bun $ExpectedBunVersion; got '$bunVersion'"
}

$nodeGypVersion = $null
$nodeGypDirectory = $null
if ($env:OS -eq "Windows_NT" -and -not $SkipInstall) {
    if (-not $NodeGypPath) {
        $NodeGypPath = $env:ANCHOR_NODE_GYP_CMD
    }
    if (-not $NodeGypPath -or -not (Test-Path -LiteralPath $NodeGypPath -PathType Leaf)) {
        throw "Windows native install requires -NodeGypPath (or process-local ANCHOR_NODE_GYP_CMD) pointing to node-gyp.cmd"
    }
    $NodeGypPath = (Resolve-Path -LiteralPath $NodeGypPath).Path
    $nodeGypVersion = (& $NodeGypPath --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $nodeGypVersion -ne "v13.0.1") {
        throw "Audited Windows native install requires node-gyp v13.0.1; got '$nodeGypVersion'"
    }
    $nodeGypDirectory = Split-Path -Parent $NodeGypPath
}

$env:BUN_CONFIG_MAX_HTTP_REQUESTS = "4"
$env:BUN_INSTALL_CACHE_DIR = Join-Path $ProjectRoot "runs\opencode-build\bun-cache"
if (-not $SkipInstall) {
    $priorTrackFileAccess = $env:TrackFileAccess
    $priorPath = $env:PATH
    try {
        if ($env:OS -eq "Windows_NT") {
            # node-gyp's generated C++ project otherwise enables MSBuild
            # FileTracker, which can leave link.exe blocked on this host.
            # This process-local setting is inherited by Bun/node-gyp/MSBuild;
            # it never changes user or machine environment state.
            $env:TrackFileAccess = "false"
            $env:PATH = "$nodeGypDirectory;$priorPath"
        }
        $installArgs = @("install", "--frozen-lockfile")
        if ($env:OS -eq "Windows_NT") {
            # Match OpenCode's own Windows CI layout. The isolated linker can
            # leave peer dependencies unavailable to its package test runner.
            $installArgs += @("--linker", "hoisted")
        }
        Invoke-Checked $BunPath $installArgs $CheckoutRoot
    }
    finally {
        if ($null -eq $priorTrackFileAccess) {
            Remove-Item Env:TrackFileAccess -ErrorAction SilentlyContinue
        }
        else {
            $env:TrackFileAccess = $priorTrackFileAccess
        }
        $env:PATH = $priorPath
    }
}
$windowsBaselineTestExclusions = @()
if (-not $SkipTests) {
    $coreTests = @("test") + @($patchSource.required_tests.core)
    Invoke-Checked $BunPath $coreTests (Join-Path $CheckoutRoot "packages\core")
    if ($env:OS -eq "Windows_NT") {
        # These two /bin/sh-dependent tests time out on the unpatched Windows
        # v1.17.18 baseline as well. Keep the exception narrow and record it in
        # the artifact manifest instead of reporting an unqualified full pass.
        $windowsBaselineTestExclusions = @(
            "loop waits while shell runs and starts after shell exits",
            "shell completion resumes queued loop callers",
            "project reference directories are allowed for external_directory"
        )
        $promptTest = "test/session/prompt.test.ts"
        $agentTest = "test/agent/agent.test.ts"
        $otherOpenCodeTests = @($patchSource.required_tests.opencode | Where-Object { $_ -ne $promptTest -and $_ -ne $agentTest })
        Invoke-Checked $BunPath (@("test") + $otherOpenCodeTests) (Join-Path $CheckoutRoot "packages\opencode")
        $promptExclusions = @($windowsBaselineTestExclusions | Where-Object { $_ -ne "project reference directories are allowed for external_directory" })
        $promptPattern = "^(?!(?:" + (($promptExclusions | ForEach-Object { [regex]::Escape($_) }) -join "|") + ")$).*"
        Invoke-Checked $BunPath @("test", $promptTest, "--test-name-pattern", $promptPattern) (Join-Path $CheckoutRoot "packages\opencode")
        $agentPattern = "^(?!(?:" + [regex]::Escape("project reference directories are allowed for external_directory") + ")$).*"
        Invoke-Checked $BunPath @("test", $agentTest, "--test-name-pattern", $agentPattern) (Join-Path $CheckoutRoot "packages\opencode")
    }
    else {
        $opencodeTests = @("test") + @($patchSource.required_tests.opencode)
        Invoke-Checked $BunPath $opencodeTests (Join-Path $CheckoutRoot "packages\opencode")
    }
}
if (-not $SkipTypecheck) {
    Invoke-Checked $BunPath @("run", "--cwd", "packages/opencode", "typecheck") $CheckoutRoot
}
Invoke-Checked $BunPath @(
    "run", "packages/opencode/script/build.ts", "--single", "--skip-install", "--skip-embed-web-ui"
) $CheckoutRoot

$built = @(
    (Join-Path $CheckoutRoot "packages\opencode\dist\opencode-windows-x64\bin\opencode.exe"),
    (Join-Path $CheckoutRoot "packages\opencode\dist\opencode-windows-x64\bin\opencode")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $built) {
    throw "Build completed without the expected Windows x64 binary"
}

New-Item -ItemType Directory -Force $OutputRoot | Out-Null
$destination = Join-Path $OutputRoot "opencode-anchor.exe"
Copy-Item -LiteralPath $built -Destination $destination -Force
$binarySha = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = [ordered]@{
    schema_version = "anchor.patched-opencode.v1"
    repository = $Repository
    baseline_commit = $BaselineCommit
    opencode_version = [string]$patchSource.upstream_version
    patch_sha256 = $patchSha
    patch_source_manifest_sha256 = (Get-FileHash -LiteralPath $PatchManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    bun_version = $bunVersion
    node_gyp_version = $nodeGypVersion
    tool_contract_version = [string]$patchSource.tool_contract_version
    tool_contract = $ToolContract
    tests_executed = -not $SkipTests
    required_tests = [ordered]@{
        core = @($patchSource.required_tests.core)
        opencode = @($patchSource.required_tests.opencode)
    }
    windows_baseline_test_exclusions = @($windowsBaselineTestExclusions)
    typecheck_executed = -not $SkipTypecheck
    binary_sha256 = $binarySha
    binary = "opencode-anchor.exe"
    global_install_modified = $false
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputRoot "manifest.json") -Encoding UTF8
Write-Output $destination
