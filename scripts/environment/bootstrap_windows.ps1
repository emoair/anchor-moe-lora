param(
    [string]$EnvironmentName = "anchor-mvp",
    [string]$CondaExe = "C:\ProgramData\Anaconda3\Scripts\conda.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Conda executable not found: $CondaExe"
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EnvironmentRoot = Join-Path $env:USERPROFILE ".conda\envs\$EnvironmentName"
$PythonExe = Join-Path $EnvironmentRoot "python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    & $CondaExe create -n $EnvironmentName python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Could not create conda environment" }
}

& $PythonExe -m pip install --disable-pip-version-check `
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA PyTorch" }

& $PythonExe -m pip install --disable-pip-version-check `
    "transformers>=5.10.1,<6" "peft>=0.19" trl datasets accelerate `
    bitsandbytes==0.48.2 sentencepiece safetensors pyyaml httpx `
    "pydantic>=2,<3" protobuf pytest
if ($LASTEXITCODE -ne 0) { throw "Could not install Anchor-MoE-LoRA dependencies" }

& $PythonExe -m pip install --disable-pip-version-check -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Could not install Anchor-MoE-LoRA editable package" }

& $PythonExe -m pytest $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Anchor-MoE-LoRA tests failed" }

Write-Host "Environment ready: $EnvironmentName"
Write-Host "Python: $PythonExe"
