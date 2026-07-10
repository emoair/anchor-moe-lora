param(
    [switch]$DeepBaseChecksum,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Config = Join-Path $ProjectRoot "configs/training/gemma4_12b_qlora_smoke.yaml"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if (-not $Python) {
    if ($env:ANCHOR_TRAIN_PYTHON -and (Test-Path $env:ANCHOR_TRAIN_PYTHON)) {
        $Python = $env:ANCHOR_TRAIN_PYTHON
    } else {
        $Python = (& py -3.10 -c "import sys; print(sys.executable)").Trim()
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
