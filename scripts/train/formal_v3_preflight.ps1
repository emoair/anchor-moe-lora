param(
    [switch]$DeepBaseChecksum,
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Config = Join-Path $ProjectRoot "configs/training/formal_v3_lowmem_common.yaml"
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

try {
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
