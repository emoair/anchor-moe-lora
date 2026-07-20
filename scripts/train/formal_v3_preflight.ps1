param(
    [switch]$DeepBaseChecksum,
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Config = Join-Path $ProjectRoot "configs/training/formal_v3_lowmem_base.yaml"
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

try {
    $SnapshotPath = Join-Path $ProjectRoot "artifacts/formal_v3/dataset/manifest.json"
    if (-not (Test-Path -LiteralPath $SnapshotPath -PathType Leaf)) {
        throw "formal-v3 immutable snapshot is missing: $SnapshotPath"
    }
    $Snapshot = Get-Content -LiteralPath $SnapshotPath -Raw | ConvertFrom-Json
    $SnapshotSha = [string]$Snapshot.snapshot_sha256
    if ($SnapshotSha -notmatch '^[0-9a-f]{64}$') {
        throw "formal-v3 snapshot_sha256 is missing or invalid"
    }
    $Configs = [ordered]@{
        B = "configs/training/formal_v3_lowmem_mixed.yaml"
        C = "configs/training/formal_v3_lowmem_common.yaml"
        D = "configs/training/formal_v3_lowmem_budget.yaml"
        E = "configs/training/formal_v3_lowmem_adaptive.yaml"
        F = "configs/training/formal_v3_lowmem_adaptive_budget.yaml"
    }
    foreach ($Entry in $Configs.GetEnumerator()) {
        $Output = Join-Path $ProjectRoot (
            "artifacts/formal_v3/schedules/$SnapshotSha/$($Entry.Key).json"
        )
        & $Python (Join-Path $ProjectRoot "scripts/train/materialize_formal_v3_schedule.py") `
            --config (Join-Path $ProjectRoot $Entry.Value) `
            --arm $Entry.Key `
            --output $Output
        if ($LASTEXITCODE -ne 0) {
            throw "formal-v3 schedule preflight failed for arm $($Entry.Key)"
        }
    }
    $Arguments = @(
        "-m", "anchor_mvp.training", "preflight",
        "--config", $Config,
        "--dry-run"
    )
    if ($DeepBaseChecksum) { $Arguments += "--deep-base-checksum" }
    & $Python @Arguments
    exit $LASTEXITCODE
}
finally {
    if ($null -eq $PreviousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
}
