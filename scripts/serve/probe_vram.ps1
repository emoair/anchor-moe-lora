param(
    [ValidateSet("pre_start", "post_start", "post_request", "manual")]
    [string]$Label = "manual",
    [string]$Distro = "Ubuntu-22.04"
)

$ErrorActionPreference = "Stop"
$NativeScript = (Resolve-Path (Join-Path $PSScriptRoot "probe_vram_wsl.sh")).Path.Replace('\', '/')
$WslScript = (& wsl.exe -d $Distro -- wslpath -a $NativeScript).Trim()
if (-not $WslScript) {
    throw "Could not resolve the WSL VRAM probe path."
}
& wsl.exe -d $Distro -- bash $WslScript --label $Label
exit $LASTEXITCODE
