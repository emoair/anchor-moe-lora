param(
    [string]$CheckoutRoot = "",
    [string]$BunPath = "",
    [string]$BunSha256 = "",
    [string]$NodeGypPath = "",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipTypecheck
)

<#
Build the Windows x64 member of the pinned OpenCode patch bundle.

This script deliberately creates or consumes one clean, target-specific checkout. It
never calls git reset/clean and refuses a non-clean existing checkout, including a
previously patched checkout. Use a new -CheckoutRoot after an interrupted build.

The WSL/Linux counterpart is scripts/tooling/build_patched_opencode_wsl.sh. Once both
platform manifests exist, create/check the cross-platform bundle with
scripts/tooling/assemble_opencode_bundle.py.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PatchManifestPath = Join-Path $ProjectRoot "patches\opencode\patch-manifest.json"
$OutputRoot = Join-Path $ProjectRoot "artifacts\tooling\opencode-patched"
$Target = "windows-x64"

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

function Invoke-Git {
    param([string[]]$Arguments, [string]$WorkingDirectory = "")
    $gitArguments = @("-c", "core.autocrlf=false")
    if ($WorkingDirectory) {
        $gitArguments += @("-C", $WorkingDirectory)
    }
    $gitArguments += $Arguments
    & git @gitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') exited with code $LASTEXITCODE"
    }
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

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
$RequiredToolContractVersion = "anchor.execution-tool-contract.v3"
if ($Repository -ne "https://github.com/anomalyco/opencode.git") {
    throw "Patch source repository is not the audited upstream"
}
if ($BaselineCommit -notmatch '^[0-9a-f]{40}$' -or $ExpectedPatchSha256 -notmatch '^[0-9a-f]{64}$') {
    throw "Patch source manifest contains an invalid commit or digest"
}
if ($null -eq $ToolContract -or [string]$ToolContract.version -ne [string]$patchSource.tool_contract_version) {
    throw "Patch source manifest contains an invalid tool contract"
}
if ([string]$patchSource.tool_contract_version -ne $RequiredToolContractVersion) {
    throw "Formal OpenCode builds require $RequiredToolContractVersion; v2 artifacts must not be rebuilt or marked ready"
}
if ([string]$ToolContract.model_bash_policy.network -ne "none-with-supervisor-unix-socket-loopback-bridge" -or
    [string]$ToolContract.model_bash_policy.workdir -ne "/testbed" -or
    [string]$ToolContract.hidden_official_eval.network -ne "none") {
    throw "Patch source manifest does not contain the formal v3 isolation contract"
}
if (-not (Test-Path -LiteralPath $PatchPath -PathType Leaf)) {
    throw "Patch is missing: $PatchPath"
}
$patchSha = Get-Sha256 $PatchPath
if ($patchSha -ne $ExpectedPatchSha256) {
    throw "Patch hash mismatch. Expected $ExpectedPatchSha256, got $patchSha"
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Windows x64 build requires a 64-bit host"
}
if (-not $CheckoutRoot) {
    $CheckoutRoot = Join-Path $ProjectRoot "runs\opencode-build\worktrees\$Target"
}
$CheckoutRoot = [IO.Path]::GetFullPath($CheckoutRoot)

# A checkout with the canonical patch applied is intentionally considered dirty. Do
# not reuse it: this keeps the source contract simple and preserves user work.
if (Test-Path -LiteralPath $CheckoutRoot) {
    if (-not (Test-Path -LiteralPath (Join-Path $CheckoutRoot ".git"))) {
        throw "CheckoutRoot exists but is not a Git checkout: $CheckoutRoot"
    }
    $existingOrigin = (& git -c core.autocrlf=false -C $CheckoutRoot remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or $existingOrigin -ne $Repository) {
        throw "Checkout origin is not the audited repository: $existingOrigin"
    }
    $existingStatus = @(& git -c core.autocrlf=false -C $CheckoutRoot status --porcelain=v1)
    if ($LASTEXITCODE -ne 0 -or $existingStatus.Count -ne 0) {
        throw "Checkout is dirty. Use a fresh -CheckoutRoot; this script never resets user work."
    }
}
else {
    $parent = Split-Path -Parent $CheckoutRoot
    New-Item -ItemType Directory -Force $parent | Out-Null
    Invoke-Git @("clone", "--depth", "1", "--branch", "v$($patchSource.upstream_version)", "--filter=blob:none", "--no-checkout", $Repository, $CheckoutRoot) $parent
}

$baselineObject = "${BaselineCommit}^{commit}"
& git -c core.autocrlf=false -C $CheckoutRoot cat-file -e $baselineObject 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Git @("fetch", "--depth", "1", "origin", $BaselineCommit) $CheckoutRoot
}
Invoke-Git @("checkout", "--detach", $BaselineCommit) $CheckoutRoot
$actualCommit = (& git -c core.autocrlf=false -C $CheckoutRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $BaselineCommit) {
    throw "Baseline mismatch. Expected $BaselineCommit, got $actualCommit"
}
Invoke-Git @("apply", "--check", $PatchPath) $CheckoutRoot
Invoke-Git @("apply", $PatchPath) $CheckoutRoot

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
$bunSha = Get-Sha256 $BunPath
if ($BunSha256) {
    $expectedBunSha = $BunSha256.ToLowerInvariant()
    if ($expectedBunSha -notmatch '^[0-9a-f]{64}$' -or $bunSha -ne $expectedBunSha) {
        throw "Bun SHA-256 does not match -BunSha256"
    }
}

$nodeGypVersion = $null
$nodeGypDirectory = $null
if (-not $SkipInstall) {
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
$env:BUN_INSTALL_CACHE_DIR = Join-Path $ProjectRoot "runs\opencode-build\bun-cache\$Target"
if (-not $SkipInstall) {
    $priorTrackFileAccess = $env:TrackFileAccess
    $priorPath = $env:PATH
    try {
        # node-gyp's generated C++ project otherwise enables MSBuild FileTracker,
        # which can leave link.exe blocked on this host. Both changes are process-only.
        $env:TrackFileAccess = "false"
        $env:PATH = "$nodeGypDirectory;$priorPath"
        Invoke-Checked $BunPath @("install", "--frozen-lockfile", "--linker", "hoisted") $CheckoutRoot
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

# Bun workspace packages are junctions. Reusing a dependency tree is permitted
# only when every @opencode-ai workspace link resolves back into this exact
# checkout; otherwise a clean v3 tree can silently execute stale v2 source.
$workspaceScope = Join-Path $CheckoutRoot "node_modules\@opencode-ai"
if (-not (Test-Path -LiteralPath $workspaceScope -PathType Container)) {
    throw "OpenCode workspace dependency scope is missing: $workspaceScope"
}
$workspacePackageRoot = [IO.Path]::GetFullPath((Join-Path $CheckoutRoot "packages")).TrimEnd([char[]]"\/") + [IO.Path]::DirectorySeparatorChar
$workspaceLinks = @(Get-ChildItem -LiteralPath $workspaceScope -Force -Directory)
if ($workspaceLinks.Count -eq 0) {
    throw "OpenCode workspace dependency scope contains no packages"
}
foreach ($workspaceLink in $workspaceLinks) {
    $targets = @($workspaceLink.Target | Where-Object { $_ })
    if ($workspaceLink.LinkType -notin @("Junction", "SymbolicLink") -or $targets.Count -ne 1) {
        throw "Workspace package is not one audited link: $($workspaceLink.FullName)"
    }
    $resolvedTarget = [IO.Path]::GetFullPath([string]$targets[0])
    if (-not $resolvedTarget.StartsWith($workspacePackageRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Workspace package resolves outside the current checkout: $($workspaceLink.Name) -> $resolvedTarget"
    }
}

$windowsBaselineTestExclusions = @()
if (-not $SkipTests) {
    # Windows child-process and Git-heavy tests can exceed Bun's 5 s default
    # under load even when their assertions complete correctly in isolation.
    $coreTests = @("test", "--timeout", "60000") + @($patchSource.required_tests.core)
    Invoke-Checked $BunPath $coreTests (Join-Path $CheckoutRoot "packages\core")
    # These upstream Windows baseline cases are known /bin/sh-dependent timeouts.
    $windowsBaselineTestExclusions = @(
        "loop waits while shell runs and starts after shell exits",
        "shell completion resumes queued loop callers",
        "cancel with queued callers resolves all cleanly",
        "project reference directories are allowed for external_directory"
    )
    $promptTest = "test/session/prompt.test.ts"
    $agentTest = "test/agent/agent.test.ts"
    $otherOpenCodeTests = @($patchSource.required_tests.opencode | Where-Object { $_ -ne $promptTest -and $_ -ne $agentTest })
    Invoke-Checked $BunPath (@("test", "--timeout", "60000") + $otherOpenCodeTests) (Join-Path $CheckoutRoot "packages\opencode")
    $promptExclusions = @($windowsBaselineTestExclusions | Where-Object { $_ -ne "project reference directories are allowed for external_directory" })
    $promptPattern = "^(?!(?:" + (($promptExclusions | ForEach-Object { [regex]::Escape($_) }) -join "|") + ")$).*"
    Invoke-Checked $BunPath @("test", "--timeout", "60000", $promptTest, "--test-name-pattern", $promptPattern) (Join-Path $CheckoutRoot "packages\opencode")
    $agentPattern = "^(?!(?:" + [regex]::Escape("project reference directories are allowed for external_directory") + ")$).*"
    Invoke-Checked $BunPath @("test", "--timeout", "60000", $agentTest, "--test-name-pattern", $agentPattern) (Join-Path $CheckoutRoot "packages\opencode")
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
$platformDirectory = Join-Path $OutputRoot $Target
New-Item -ItemType Directory -Force $platformDirectory | Out-Null
$platformDestination = Join-Path $platformDirectory "opencode-anchor.exe"
Copy-Item -LiteralPath $built -Destination $platformDestination -Force
$binarySha = Get-Sha256 $platformDestination

# The root copy preserves the current Windows-only executor contract. Bundle members
# always use the symmetric <target>/opencode-anchor[.exe] layout below.
$destination = Join-Path $OutputRoot "opencode-anchor.exe"
Copy-Item -LiteralPath $platformDestination -Destination $destination -Force
if ((Get-Sha256 $destination) -ne $binarySha) {
    throw "Legacy Windows compatibility copy does not match the platform artifact"
}
$patchManifestSha = Get-Sha256 $PatchManifestPath
$lockPath = Join-Path $CheckoutRoot "bun.lock"
$lockSha = if (Test-Path -LiteralPath $lockPath -PathType Leaf) { Get-Sha256 $lockPath } else { $null }
$source = [ordered]@{
    repository = $Repository
    baseline_commit = $BaselineCommit
    opencode_version = [string]$patchSource.upstream_version
    patch_sha256 = $patchSha
    patch_source_manifest_sha256 = $patchManifestSha
    bun_version = $bunVersion
    tool_contract_version = [string]$patchSource.tool_contract_version
    tool_contract = $ToolContract
    lockfile_sha256 = $lockSha
}
$platformManifest = [ordered]@{
    schema_version = "anchor.patched-opencode.platform.v1"
    target = $Target
    platform = [ordered]@{ os = "windows"; arch = "x64"; libc = $null }
    source = $source
    bun = [ordered]@{ version = $bunVersion; sha256 = $bunSha }
    node_gyp_version = $nodeGypVersion
    install = [ordered]@{ executed = -not $SkipInstall; linker = "hoisted"; cache_scope = $Target }
    checks = [ordered]@{
        tests_executed = -not $SkipTests
        required_tests = [ordered]@{ core = @($patchSource.required_tests.core); opencode = @($patchSource.required_tests.opencode) }
        test_exclusions = @($windowsBaselineTestExclusions)
        workspace_link_audit = [ordered]@{ executed = $true; count = $workspaceLinks.Count; required_root = "checkout/packages" }
        typecheck_executed = -not $SkipTypecheck
        build_smoke_executed = $true
    }
    binary = [ordered]@{ path = "$Target/opencode-anchor.exe"; sha256 = $binarySha }
    global_install_modified = $false
}
$platformManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputRoot "$Target.manifest.json") -Encoding UTF8

# Preserve the existing single-Windows artifact contract while the bundle manifest is
# assembled separately after the Linux member has been built.
$legacyManifest = [ordered]@{
    schema_version = "anchor.patched-opencode.v1"
    repository = $Repository
    baseline_commit = $BaselineCommit
    opencode_version = [string]$patchSource.upstream_version
    patch_sha256 = $patchSha
    patch_source_manifest_sha256 = $patchManifestSha
    bun_version = $bunVersion
    node_gyp_version = $nodeGypVersion
    tool_contract_version = [string]$patchSource.tool_contract_version
    tool_contract = $ToolContract
    tests_executed = -not $SkipTests
    required_tests = [ordered]@{ core = @($patchSource.required_tests.core); opencode = @($patchSource.required_tests.opencode) }
    windows_baseline_test_exclusions = @($windowsBaselineTestExclusions)
    workspace_link_audit = [ordered]@{ executed = $true; count = $workspaceLinks.Count; required_root = "checkout/packages" }
    typecheck_executed = -not $SkipTypecheck
    binary_sha256 = $binarySha
    binary = "opencode-anchor.exe"
    global_install_modified = $false
}
$legacyManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputRoot "manifest.json") -Encoding UTF8
Write-Output $destination
