param(
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$PreviousAllocatorConf = $env:PYTORCH_CUDA_ALLOC_CONF
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"
try {
    & $Python (Join-Path $PSScriptRoot "export_bnb_nf4.py")
    exit $LASTEXITCODE
} finally {
    if ($null -eq $PreviousAllocatorConf) {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    } else {
        $env:PYTORCH_CUDA_ALLOC_CONF = $PreviousAllocatorConf
    }
}
