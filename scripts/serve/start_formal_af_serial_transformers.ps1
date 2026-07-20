param(
    [string]$ProjectRoot = "",
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe",
    [string]$FormalConfig = "configs/benchmark/formal_partial_v1_af.json",
    [string]$BaseModel = "models/google-gemma-4-12B-bnb-nf4",
    [string]$ProcessorManifest = "configs/serving/formal_af_windows_processor.json",
    [int]$Port = 8000,
    [ValidateSet(1024, 2048)]
    [int]$MaxModelLength = 2048,
    [switch]$DisableTf32,
    [switch]$DisableUnpaddedDecodeFastPath,
    [switch]$PreflightOnly,
    [switch]$PrintCommand
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
} else {
    $ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable is missing: $Python"
}

$Arguments = @(
    "-m", "anchor_mvp.serving.windows_serial_server",
    "--project-root", $ProjectRoot,
    "--formal-config", $FormalConfig,
    "--base-model", $BaseModel,
    "--processor-manifest", $ProcessorManifest,
    "--host", "127.0.0.1",
    "--port", $Port,
    "--max-model-length", $MaxModelLength,
    "--api-key-env", "ANCHOR_VLLM_API_KEY"
)
if ($DisableTf32) { $Arguments += "--disable-tf32" }
if ($DisableUnpaddedDecodeFastPath) {
    $Arguments += "--disable-unpadded-decode-fast-path"
}
if ($PreflightOnly) { $Arguments += "--preflight-only" }

if ($PrintCommand) {
    Write-Output (($Python, $Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' ')
    exit 0
}

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
try {
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
