param(
    [ValidateSet("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate")]
    [string]$Adapter = "frontend_gen",
    [switch]$Execute,
    [switch]$AllowModelDownload,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Config = Join-Path $ProjectRoot "configs/training/gemma4_12b_qlora_one_step.yaml"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$PreviousAllocatorConf = $env:PYTORCH_CUDA_ALLOC_CONF
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"

if (-not $Python) {
    if ($env:ANCHOR_TRAIN_PYTHON -and (Test-Path $env:ANCHOR_TRAIN_PYTHON)) {
        $Python = $env:ANCHOR_TRAIN_PYTHON
    } else {
        $Python = (& py -3.10 -c "import sys; print(sys.executable)").Trim()
    }
}

$Arguments = @(
    "-m", "anchor_mvp.training", "smoke-gate",
    "--config", $Config,
    "--adapter", $Adapter,
    "--rank", "16"
)
if ($Execute) {
    $Arguments += "--execute"
    if ($AllowModelDownload) { $Arguments += "--allow-model-download" }
} else {
    $Arguments += "--dry-run"
}

try {
    & $Python @Arguments
    $Code = $LASTEXITCODE
} finally {
    if ($null -eq $PreviousAllocatorConf) {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    } else {
        $env:PYTORCH_CUDA_ALLOC_CONF = $PreviousAllocatorConf
    }
}
if ($Code -ne 0) {
    Write-Error "smoke-gate child process failed with native exit code $Code"
    exit 1
}
exit 0
