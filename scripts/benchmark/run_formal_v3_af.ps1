param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$')]
    [string]$VersionId,
    [switch]$Finalize,
    [switch]$Execute,
    [switch]$Resume,
    [switch]$AuthorizeHeldoutAccess,
    [switch]$NoVram,
    [string]$Control = "configs/benchmark/formal_v3_af_control.json",
    [string]$BaseUrl = "http://127.0.0.1:8000/v1",
    [string]$AdminBaseUrl = "http://127.0.0.1:8000",
    [string]$ServerProjectRoot = "/mnt/d/LLM/anchor-moe-lora",
    [string]$OutputDir = "",
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$LockStream = $null

function Resolve-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

try {
    if ($VersionId -match '(?i)formal[-_]v2') {
        throw "formal-v3 VersionId must not name or reuse formal-v2"
    }
    $ControlPath = Resolve-ProjectPath $Control
    $BundleDir = Resolve-ProjectPath (
        "artifacts/formal_v3/evaluation/registries/$VersionId"
    )
    $BenchmarkPath = Join-Path $BundleDir "benchmark.json"
    $Materializer = Join-Path $ProjectRoot "scripts/benchmark/materialize_formal_v3_af.py"

    if ($Finalize) {
        & $Python $Materializer `
            --control $ControlPath `
            --project-root $ProjectRoot `
            --version-id $VersionId `
            --output-dir $BundleDir `
            --finalize
        if ($LASTEXITCODE -ne 0) {
            throw "formal-v3 registry finalization is BLOCKED (exit $LASTEXITCODE)"
        }
    }

    if (-not (Test-Path -LiteralPath $BenchmarkPath -PathType Leaf)) {
        & $Python $Materializer `
            --control $ControlPath `
            --project-root $ProjectRoot `
            --version-id $VersionId
        if ($LASTEXITCODE -ne 0) {
            throw "formal-v3 A-F evaluation is BLOCKED; complete/finalize training first"
        }
        throw "formal-v3 sources are ready but the immutable bundle is not finalized; use -Finalize"
    }

    & $Python -m anchor_mvp.benchmark.formal_v3_preflight `
        --benchmark $BenchmarkPath `
        --project-root $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "formal-v3 A-F offline preflight is BLOCKED (exit $LASTEXITCODE)"
    }
    if (-not $Execute) {
        Write-Host "formal-v3 A-F offline preflight READY; heldout case content was not read."
        return
    }
    if (-not $AuthorizeHeldoutAccess) {
        throw "live heldout evaluation requires explicit -AuthorizeHeldoutAccess"
    }

    $VersionOutputRoot = Resolve-ProjectPath "runs/formal-v3/evaluation/$VersionId"
    $ResolvedOutput = if ($OutputDir) {
        Resolve-ProjectPath $OutputDir
    } else {
        Join-Path $VersionOutputRoot "run"
    }
    $ExpectedPrefix = $VersionOutputRoot.TrimEnd('\') + '\'
    if (-not $ResolvedOutput.StartsWith(
        $ExpectedPrefix,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "formal-v3 output must remain inside $VersionOutputRoot"
    }
    if ($Resume -and -not (Test-Path -LiteralPath $ResolvedOutput -PathType Container)) {
        throw "-Resume requires the existing same-version output directory"
    }
    if (-not $Resume -and (Test-Path -LiteralPath $ResolvedOutput)) {
        throw "fresh formal-v3 output already exists; choose a new VersionId or use -Resume"
    }

    [IO.Directory]::CreateDirectory($VersionOutputRoot) | Out-Null
    $LockPath = Join-Path $VersionOutputRoot "evaluation.lock"
    try {
        $LockStream = [IO.File]::Open(
            $LockPath,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
    }
    catch {
        throw "another formal-v3 evaluator owns $LockPath"
    }

    $Arguments = @(
        "-m", "anchor_mvp.benchmark", "formal-run",
        "--specs", $BenchmarkPath,
        "--cases", (Join-Path $ProjectRoot "configs/benchmark/heldout_cases_v1.jsonl"),
        "--fixtures-root", (Join-Path $ProjectRoot "examples/benchmark/fixtures"),
        "--manifest", (Join-Path $ProjectRoot "artifacts/benchmark/heldout_v1/manifest.json"),
        "--leak-audit", (Join-Path $ProjectRoot "artifacts/benchmark/heldout_v1/leak_audit.prebulk.json"),
        "--project-root", $ProjectRoot,
        "--base-url", $BaseUrl,
        "--admin-base-url", $AdminBaseUrl,
        "--server-project-root", $ServerProjectRoot,
        "--output-dir", $ResolvedOutput,
        "--serial-runtime-lora",
        "--authorize-heldout-access"
    )
    if ($Resume) { $Arguments += "--resume" }
    if ($NoVram) { $Arguments += "--no-vram" }
    & $Python @Arguments
    exit $LASTEXITCODE
}
finally {
    if ($null -ne $LockStream) {
        $Path = $LockStream.Name
        $LockStream.Dispose()
        Remove-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    }
    if ($null -eq $PreviousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
}
