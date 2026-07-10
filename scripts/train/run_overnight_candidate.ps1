param(
    [ValidateSet("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate", "mixed_all")]
    [string]$Adapter = "frontend_gen",
    [ValidateSet(2, 3, 4, 16, 32, 64)]
    [int]$Rank = 16,
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Config = Join-Path $ProjectRoot "configs/training/gemma4_12b_qlora_overnight.yaml"
$LogDir = Join-Path $ProjectRoot "artifacts/logs"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $LogDir "overnight-$Adapter-r$Rank-$Stamp.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$PreviousAllocatorConfig = $env:PYTORCH_CUDA_ALLOC_CONF
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"

try {
    "[$(Get-Date -Format o)] starting $Adapter rank=$Rank config=$Config" | Tee-Object -FilePath $LogPath
    & $Python -m anchor_mvp.training `
        --config $Config `
        --adapter $Adapter `
        --rank $Rank `
        --execute 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "training exited with code $LASTEXITCODE"
    }
    "[$(Get-Date -Format o)] completed $Adapter rank=$Rank" | Tee-Object -FilePath $LogPath -Append
}
finally {
    if ($null -eq $PreviousAllocatorConfig) {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    } else {
        $env:PYTORCH_CUDA_ALLOC_CONF = $PreviousAllocatorConfig
    }
}
