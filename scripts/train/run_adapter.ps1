param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate", "mixed_all")]
    [string]$Adapter,

    [ValidateSet(16, 32, 64)]
    [int]$Rank = 16,

    [switch]$Execute,
    [switch]$AllowModelDownload,
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
    "-m", "anchor_mvp.training",
    "--config", $Config,
    "--adapter", $Adapter,
    "--rank", "$Rank"
)

if ($Execute) {
    $Arguments += "--execute"
    if ($AllowModelDownload) {
        $Arguments += "--allow-model-download"
    }
} else {
    $Arguments += "--dry-run"
}

& $Python @Arguments
exit $LASTEXITCODE
