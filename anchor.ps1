[CmdletBinding()]
param(
    [ValidateSet(
        "menu",
        "status",
        "ui",
        "preflight",
        "docs",
        "distill-swebench",
        "distill-synthetic",
        "dashboard",
        "run"
    )]
    [string]$Action = "menu",
    [string]$Config = "configs/data/automation.yaml",
    [string]$SWEConfig = "configs/data/swebench_full_bank.formal.yaml",
    [string]$SWECoordinatorConfig = "configs/data/swebench_five_stage.ccswitch.yaml",
    [switch]$ConfirmLive,
    [switch]$Resume,
    [int]$Concurrency = 0,
    [int]$MaxTasks = 0,
    [switch]$ConfirmLegacySynthetic,
    [switch]$AllowIncomplete,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$automationLauncher = Join-Path $repoRoot "scripts\data\start_automation.ps1"
$openCodeArtifacts = Join-Path $repoRoot "artifacts\tooling\opencode-patched"
$openCodeVerifier = Join-Path $repoRoot "scripts\tooling\assemble_opencode_bundle.py"
$sandboxConfig = Join-Path $repoRoot "configs\tooling\opencode_distillation_ramp.yaml"
$ccSwitchRouteManifest = Join-Path $repoRoot "artifacts\tooling\ccswitch-patched\route-manifest.json"
$ccSwitchRouteVerifier = Join-Path $repoRoot "scripts\tooling\validate_ccswitch_route.py"
$sweCoordinator = Join-Path $repoRoot "scripts\tooling\run_swebench_ccswitch.py"
$fullBankBuilder = Join-Path $repoRoot "scripts\data\build_swebench_full_bank.py"
$fullBankManifest = Join-Path $repoRoot "artifacts\swebench\full-bank-v1\manifest.json"
$dashboardScript = Join-Path $repoRoot "scripts\observability\distillation_dashboard.py"
$formalGateReader = Join-Path $repoRoot "scripts\observability\formal_gate_status.py"
$openCodeDryPreflight = Join-Path $repoRoot "scripts\tooling\run_live.py"

function Write-AnchorTitle {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== Anchor-MoE-LoRA · $Title ===" -ForegroundColor Cyan
}

function Resolve-AnchorPython {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $candidates += [pscustomobject]@{ Value = $PythonExe; Prefix = @() }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_MVP_PYTHON)) {
        $candidates += [pscustomobject]@{ Value = $env:ANCHOR_MVP_PYTHON; Prefix = @() }
    }
    $candidates += [pscustomobject]@{
        Value = (Join-Path $repoRoot ".venv\Scripts\python.exe")
        Prefix = @()
    }
    if (-not [string]::IsNullOrWhiteSpace($HOME)) {
        $candidates += [pscustomobject]@{
            Value = (Join-Path $HOME ".conda\envs\anchor-mvp\python.exe")
            Prefix = @()
        }
    }
    $candidates += [pscustomobject]@{ Value = "py.exe"; Prefix = @("-3.11") }

    foreach ($candidate in $candidates) {
        $application = if (Test-Path -LiteralPath $candidate.Value -PathType Leaf) {
            (Resolve-Path -LiteralPath $candidate.Value).Path
        }
        else {
            $command = Get-Command $candidate.Value -CommandType Application `
                -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -ne $command) { $command.Source } else { $null }
        }
        if ([string]::IsNullOrWhiteSpace($application)) {
            continue
        }
        & $application @($candidate.Prefix) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{
                Executable = $application
                Prefix = @($candidate.Prefix)
            }
        }
    }
    throw "找不到项目 Python 3.10+ / Project Python 3.10+ was not found."
}

function Invoke-AutomationLauncher {
    param([string[]]$Arguments)

    $powershell = (Get-Command powershell.exe -CommandType Application).Source
    $childArguments = @(
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", $automationLauncher,
        "-Config", $Config
    )
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $childArguments += @("-PythonExe", $PythonExe)
    }
    $childArguments += $Arguments
    & $powershell @childArguments | ForEach-Object { Write-Host $_ }
    $childExitCode = $LASTEXITCODE
    return $childExitCode
}

function Invoke-FullBankPreflight {
    param([switch]$RequireLaunchReady)

    $sweConfigPath = if ([IO.Path]::IsPathRooted($SWEConfig)) {
        [IO.Path]::GetFullPath($SWEConfig)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $SWEConfig))
    }
    if (-not (Test-Path -LiteralPath $fullBankBuilder -PathType Leaf) -or
        -not (Test-Path -LiteralPath $sweConfigPath -PathType Leaf)) {
        return [pscustomobject]@{
            ExitCode = 4
            Payload = $null
            ConfigPath = $sweConfigPath
        }
    }
    $python = Resolve-AnchorPython
    $arguments = @($python.Prefix) + @(
        $fullBankBuilder,
        "--config", $sweConfigPath,
        "--preflight"
    )
    if ($RequireLaunchReady) {
        $arguments += "--require-launch-ready"
    }
    $raw = @(& $python.Executable @arguments 2>$null)
    $exitCode = $LASTEXITCODE
    $payload = $null
    if ($raw.Count -gt 0) {
        try {
            $payload = ($raw -join "`n") | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            $payload = $null
            if ($exitCode -eq 0) { $exitCode = 4 }
        }
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Payload = $payload
        ConfigPath = $sweConfigPath
    }
}

function Format-ReadyState {
    param($Value)
    if ($Value -eq $true) { return "READY" }
    if ($Value -eq $false) { return "NOT READY" }
    return "UNAVAILABLE"
}

function New-FailClosedFormalPreflight {
    param([string]$ReasonCode)

    return [pscustomobject]@{
        schema_version = "anchor.swebench-ccswitch-preflight.v1"
        content_free = $true
        offline = $true
        provider_requests = 0
        credentials_read = $false
        sample_bodies_read = $false
        sample_bodies_printed = $false
        heldout_files_read = $false
        component_ready = $false
        bank_ready = $false
        execution_contract_ready = $false
        live_start_allowed = $false
        reason_code = $ReasonCode
        live_started = $false
    }
}

function Test-ContentFreeFormalPreflight {
    param($Payload)

    if ($null -eq $Payload) { return $false }
    $required = @(
        "schema_version", "content_free", "offline", "provider_requests",
        "credentials_read", "sample_bodies_read", "sample_bodies_printed",
        "heldout_files_read", "component_ready", "bank_ready",
        "execution_contract_ready", "live_start_allowed", "reason_code",
        "live_started"
    )
    foreach ($name in $required) {
        if ($null -eq $Payload.PSObject.Properties[$name]) { return $false }
    }
    if ($Payload.schema_version -ne "anchor.swebench-ccswitch-preflight.v1" -or
        $Payload.content_free -ne $true -or $Payload.offline -ne $true -or
        $Payload.provider_requests -ne 0 -or $Payload.credentials_read -ne $false -or
        $Payload.sample_bodies_read -ne $false -or
        $Payload.sample_bodies_printed -ne $false -or
        $Payload.heldout_files_read -ne $false -or
        $Payload.live_started -ne $false) {
        return $false
    }
    foreach ($name in @(
        "component_ready", "bank_ready", "execution_contract_ready", "live_start_allowed"
    )) {
        if ($Payload.PSObject.Properties[$name].Value -isnot [bool]) { return $false }
    }
    if ($Payload.reason_code -isnot [string] -or
        $Payload.reason_code -notmatch '^[a-z0-9_]{1,80}$') {
        return $false
    }
    return $true
}

function Invoke-ContentFreeFormalPreflight {
    if (-not (Test-Path -LiteralPath $formalGateReader -PathType Leaf)) {
        return [pscustomobject]@{
            ExitCode = 4
            Payload = (New-FailClosedFormalPreflight "formal_gate_reader_missing")
        }
    }
    try {
        $python = Resolve-AnchorPython
        $arguments = @($python.Prefix) + @($formalGateReader, "--root", $repoRoot)
        $raw = @(& $python.Executable @arguments 2>$null)
        $exitCode = $LASTEXITCODE
        $payload = $null
        if ($exitCode -eq 0 -and $raw.Count -gt 0) {
            try {
                $payload = ($raw -join "`n") | ConvertFrom-Json -ErrorAction Stop
            }
            catch { $payload = $null }
        }
        if (-not (Test-ContentFreeFormalPreflight $payload)) {
            return [pscustomobject]@{
                ExitCode = 4
                Payload = (New-FailClosedFormalPreflight "formal_gate_payload_invalid")
            }
        }
        return [pscustomobject]@{ ExitCode = 0; Payload = $payload }
    }
    catch {
        return [pscustomobject]@{
            ExitCode = 4
            Payload = (New-FailClosedFormalPreflight "formal_gate_reader_failed")
        }
    }
}

function Show-Status {
    Write-AnchorTitle "状态 / Status"
    Write-Host "紧凑只读状态；不读取样本/heldout/旧 status 正文，不加载密钥，不发 Provider 请求。"
    Write-Host "Compact read-only status; no sample/heldout/legacy status body, credential, or provider request."

    $dashboardListening = Test-DashboardListening
    $openCodePresent = (Test-Path -LiteralPath (Join-Path $openCodeArtifacts "opencode-anchor.exe") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $openCodeArtifacts "linux-x64\opencode-anchor") -PathType Leaf)
    $sandboxPresent = Test-Path -LiteralPath $sandboxConfig -PathType Leaf
    $sweConfigPath = if ([IO.Path]::IsPathRooted($SWEConfig)) { $SWEConfig } else { Join-Path $repoRoot $SWEConfig }
    $sweConfigPresent = Test-Path -LiteralPath $sweConfigPath -PathType Leaf
    $coordinatorPresent = Test-Path -LiteralPath $sweCoordinator -PathType Leaf

    $routeState = "MISSING"
    if (Test-Path -LiteralPath $ccSwitchRouteManifest -PathType Leaf) {
        $python = Resolve-AnchorPython
        & $python.Executable @($python.Prefix) $ccSwitchRouteVerifier `
            --manifest $ccSwitchRouteManifest --require-ready 1>$null 2>$null
        $routeState = if ($LASTEXITCODE -eq 0) { "READY + ATTESTED" } else { "PRESENT, NOT READY" }
    }

    Write-Host ("{0,-30} {1}" -f "Dashboard 127.0.0.1:8765", $(if ($dashboardListening) { "LISTENING" } else { "STOPPED" }))
    Write-Host ("{0,-30} {1}" -f "Patched OpenCode dual target", $(if ($openCodePresent) { "PRESENT" } else { "MISSING" }))
    Write-Host ("{0,-30} {1}" -f "Sandbox config", $(if ($sandboxPresent) { "PRESENT" } else { "MISSING" }))
    Write-Host ("{0,-30} {1}" -f "CC Switch component", $routeState)
    Write-Host ("{0,-30} {1}" -f "WSL/Podman route reachability", "NOT PROBED / E2E UNKNOWN")
    Write-Host ("{0,-30} {1}" -f "SWE five-stage config", $(if ($sweConfigPresent) { "PRESENT" } else { "MISSING" }))
    Write-Host ("{0,-30} {1}" -f "SWE coordinator", $(if ($coordinatorPresent) { "PRESENT" } else { "MISSING" }))

    $formal = Invoke-ContentFreeFormalPreflight
    $gates = $formal.Payload
    Write-Host ("{0,-30} {1}" -f "Formal component gate", (Format-ReadyState $gates.component_ready))
    Write-Host ("{0,-30} {1}" -f "Formal bank gate", (Format-ReadyState $gates.bank_ready))
    Write-Host ("{0,-30} {1}" -f "Execution contract gate", (Format-ReadyState $gates.execution_contract_ready))
    $officialEvaluationReady = $gates.execution_contract.official_evaluation_contract_ready
    Write-Host ("{0,-30} {1} (NON-BLOCKING)" -f "Official heldout eval gate", (Format-ReadyState $officialEvaluationReady))
    $liveState = if ($gates.live_start_allowed -eq $true) { "READY" } else { "BLOCKED" }
    $liveColor = if ($gates.live_start_allowed -eq $true) { "Green" } else { "Red" }
    Write-Host ("{0,-30} {1} ({2})" -f "Formal LIVE start gate", $liveState, $gates.reason_code) -ForegroundColor $liveColor
    if (Test-Path -LiteralPath $fullBankManifest -PathType Leaf) {
        try {
            $bankManifest = Get-Content -LiteralPath $fullBankManifest -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
            $enCount = [int]$bankManifest.bilingual.counts.'en-US'
            $zhCount = [int]$bankManifest.bilingual.counts.'zh-CN'
            $zhManifestPresent = $bankManifest.bilingual.translation_manifest_present -eq $true
            Write-Host ("{0,-30} en-US={1}; zh-CN={2} (routing assignment only)" -f "Language-route assignment", $enCount, $zhCount)
            Write-Host ("{0,-30} {1}" -f "zh-CN localized text", $(if ($zhManifestPresent) { "MANIFEST PRESENT" } else { "MANIFEST MISSING" }))
            if (-not $zhManifestPresent) {
                Write-Host "9504/9504 仅是语言路由分配，不代表 9504 条中文正文已经生成。" -ForegroundColor Yellow
                Write-Host "9504/9504 is locale routing only, not completed Chinese body text." -ForegroundColor Yellow
            }
        }
        catch {
            Write-Host ("{0,-30} {1}" -f "Language-route assignment", "UNAVAILABLE")
        }
    }

    $routes = @(Get-NetRoute -AddressFamily IPv4 -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue)
    $virtual = @($routes | Where-Object {
        $_.InterfaceAlias -match '(?i)tun|tap|clash|flclash|wintun|vpn' -or
        $_.NextHop -match '^198\.(18|19)\.'
    }).Count -gt 0
    $physical = @($routes | Where-Object {
        $_.InterfaceAlias -notmatch '(?i)tun|tap|clash|flclash|wintun|vpn' -and
        $_.NextHop -notmatch '^198\.(18|19)\.' -and $_.NextHop -ne '0.0.0.0'
    }).Count -gt 0
    Write-Host ("{0,-30} virtual={1}; physical={2}; pinned=false" -f "Default route audit", $virtual, $physical)
    if ($virtual) {
        Write-Host "注意：NO_PROXY 只能绕过代理环境变量，不能覆盖 TUN 默认路由。" -ForegroundColor Yellow
        Write-Host "NO_PROXY bypasses proxy variables only; it does not pin the physical NIC." -ForegroundColor Yellow
    }

    if ($dashboardListening) {
        try {
            $snapshot = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/snapshot" -TimeoutSec 2
            $profile = $snapshot.control.provider_profile
            if ($null -ne $profile) {
                Write-Host ("{0,-30} model={1}; reasoning={2}; pricing={3}" -f `
                    "Managed provider profile", $profile.model, `
                    $(if ($profile.reasoning_enabled) { $profile.reasoning_effort } else { "off" }), `
                    $profile.pricing_route)
            }
        }
        catch {
            Write-Host ("{0,-30} {1}" -f "Managed provider profile", "UNAVAILABLE")
        }
    }
}

function Show-Preflight {
    Write-AnchorTitle "预检 / Preflight"
    Write-Host "1/5 Content-free formal gates (no candidate/heldout/credential read)"
    $formal = Invoke-ContentFreeFormalPreflight
    $gates = $formal.Payload
    Write-Host ("  component_ready={0}; bank_ready={1}; execution_contract_ready={2}; live_start_allowed={3}; reason={4}" -f `
        $gates.component_ready, $gates.bank_ready, $gates.execution_contract_ready, `
        $gates.live_start_allowed, $gates.reason_code)

    $python = Resolve-AnchorPython
    Write-Host "2/5 Patched OpenCode bundle hashes"
    & $python.Executable @($python.Prefix) $openCodeVerifier `
        --artifact-root $openCodeArtifacts --check
    if ($LASTEXITCODE -ne 0) {
        throw "OpenCode 产物核验失败 / artifact verification failed."
    }

    Write-Host "3/5 OpenCode + sandbox dry preflight"
    if (-not (Test-Path -LiteralPath $sandboxConfig -PathType Leaf)) {
        throw "缺少沙箱配置 / sandbox config is missing: $sandboxConfig"
    }
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    try {
        & $python.Executable @($python.Prefix) $openCodeDryPreflight --batch-config $sandboxConfig
        if ($LASTEXITCODE -ne 0) {
            throw "OpenCode 沙箱 dry preflight 失败 / sandbox dry preflight failed."
        }
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
    }

    Write-Host "4/5 CC Switch metadata"
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    try {
        & $python.Executable @($python.Prefix) `
            -m anchor_mvp.integrations.ccswitch_metadata `
            check --offline `
            --state-dir (Join-Path $repoRoot "runs\ccswitch-metadata-readonly")
        if ($LASTEXITCODE -ne 0) {
            throw "CC Switch 元数据核验失败 / metadata validation failed."
        }
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
    }
    Write-Host "5/5 CC Switch Anchor route readiness"
    $routeReady = $false
    if (Test-Path -LiteralPath $ccSwitchRouteManifest -PathType Leaf) {
        & $python.Executable @($python.Prefix) $ccSwitchRouteVerifier `
            --manifest $ccSwitchRouteManifest --require-ready
        $routeReady = $LASTEXITCODE -eq 0
    }
    if (-not $routeReady) {
        Write-Host "Anchor CC Switch route patch/binary manifest is missing or NOT READY." -ForegroundColor Yellow
        Write-Host "双魔改实时链尚未端到端核验；distill-swebench 仍可做离线预检，但 LIVE 保持关闭。" -ForegroundColor Yellow
        Write-Host "The dual modified chain is not E2E verified; offline preflight remains available, LIVE remains blocked." -ForegroundColor Yellow
    }
    $gateColor = if ($gates.live_start_allowed -eq $true) { "Green" } else { "Red" }
    Write-Host ("Formal gates: component_ready={0}; bank_ready={1}; execution_contract_ready={2}; live_start_allowed={3}; reason={4}" -f `
        $gates.component_ready, $gates.bank_ready, $gates.execution_contract_ready, `
        $gates.live_start_allowed, $gates.reason_code) -ForegroundColor $gateColor
    Write-Host "本预检没有启动模型、代理、沙箱或网络请求。"
    Write-Host "This preflight started no model, proxy, sandbox, or network request."
    if (($formal.ExitCode -ne 0 -or $gates.live_start_allowed -ne $true -or -not $routeReady) -and
        -not $AllowIncomplete) {
        Write-Host "默认预检失败关闭；仅组件诊断可显式加 -AllowIncomplete。" -ForegroundColor Red
        Write-Host "Default preflight fails closed; add -AllowIncomplete only for component diagnostics." -ForegroundColor Red
        exit 4
    }
}

function Test-DashboardListening {
    $listening = $false
    $netCommand = Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue
    if ($null -ne $netCommand) {
        $listener = Get-NetTCPConnection `
            -LocalAddress "127.0.0.1" `
            -LocalPort 8765 `
            -State Listen `
            -ErrorAction SilentlyContinue
        $listening = $null -ne $listener
    }
    return $listening
}

function Show-UI {
    Write-AnchorTitle "面板 / UI"
    if (Test-DashboardListening) {
        Write-Host "面板已恢复 / UI is ready: http://127.0.0.1:8765/" -ForegroundColor Green
        return
    }
    $python = Resolve-AnchorPython
    $logRoot = Join-Path $repoRoot "runs\dashboard-launch"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $arguments = @($python.Prefix) + @(
        $dashboardScript,
        "--host", "127.0.0.1",
        "--port", "8765"
    )
    $process = Start-Process `
        -FilePath $python.Executable `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "stderr.log") `
        -PassThru
    for ($attempt = 0; $attempt -lt 20; $attempt += 1) {
        Start-Sleep -Milliseconds 250
        if (Test-DashboardListening) {
            Write-Host "面板已启动 / UI started: http://127.0.0.1:8765/ (PID $($process.Id))" -ForegroundColor Green
            Write-Host "浏览器不会自动打开 / Browser was not opened automatically."
            return
        }
        if ($process.HasExited) {
            break
        }
    }
    throw "面板启动失败；查看 runs/dashboard-launch/stderr.log / UI startup failed."
}

function Show-Docs {
    Write-AnchorTitle "文档 / Docs"
    $documents = @(
        "QUICKSTART.zh-CN.md",
        "QUICKSTART.md",
        "docs\data_pipeline.md",
        "docs\tooling_opencode.md",
        "docs\distillation_dashboard.zh-CN.md",
        "docs\teacher_providers.md",
        "docs\teacher_providers.zh-CN.md",
        "docs\PROJECT_STATUS.md"
    )
    foreach ($document in $documents) {
        Write-Host (Join-Path $repoRoot $document)
    }
    Write-Host "仅列出路径；不会修改文件或打开外部程序。"
    Write-Host "Paths only; no file is modified and no external program is opened."
}

function Start-SWEBenchDistillation {
    Write-AnchorTitle "SWE-bench 五段蒸馏 / five-stage distillation"
    $missing = @()
    if (-not (Test-Path -LiteralPath $openCodeArtifacts -PathType Container)) {
        $missing += "patched OpenCode artifacts"
    }
    if (-not (Test-Path -LiteralPath $sandboxConfig -PathType Leaf)) {
        $missing += "OpenCode sandbox config"
    }
    if (-not (Test-Path -LiteralPath $ccSwitchRouteManifest -PathType Leaf)) {
        $missing += "Anchor CC Switch route patch/binary manifest"
    }
    $sweConfigPath = if ([IO.Path]::IsPathRooted($SWEConfig)) {
        $SWEConfig
    }
    else {
        Join-Path $repoRoot $SWEConfig
    }
    if (-not (Test-Path -LiteralPath $sweConfigPath -PathType Leaf)) {
        $missing += "CC Switch-bound SWE-bench config"
    }
    $coordinatorConfigPath = if ([IO.Path]::IsPathRooted($SWECoordinatorConfig)) {
        [IO.Path]::GetFullPath($SWECoordinatorConfig)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $SWECoordinatorConfig))
    }
    if (-not (Test-Path -LiteralPath $coordinatorConfigPath -PathType Leaf)) {
        $missing += "formal SWE-bench coordinator config"
    }
    if (-not (Test-Path -LiteralPath $sweCoordinator -PathType Leaf)) {
        $missing += "SWE-bench OpenCode+CC Switch coordinator"
    }
    if ($missing.Count -gt 0) {
        Write-Host "拒绝启动，缺少 / Launch refused; missing:" -ForegroundColor Red
        foreach ($item in $missing) {
            Write-Host "  - $item" -ForegroundColor Red
        }
        Write-Host "该入口绝不改走 synthetic direct / This entry never falls back to synthetic direct." -ForegroundColor Yellow
        exit 4
    }

    if ($Concurrency -lt 0 -or $MaxTasks -lt 0) {
        Write-Host "Concurrency/MaxTasks 必须为 0（使用配置）或正整数。" -ForegroundColor Red
        Write-Host "Concurrency/MaxTasks must be 0 (use config) or a positive integer." -ForegroundColor Red
        exit 2
    }
    if ($Resume -and -not $ConfirmLive) {
        Write-Host "-Resume 仅可与 -ConfirmLive 一起使用 / -Resume requires -ConfirmLive." -ForegroundColor Red
        exit 2
    }

    Write-Host "1/3 Full-bank offline launch gate"
    $fullBank = Invoke-FullBankPreflight -RequireLaunchReady
    if ($fullBank.ExitCode -ne 0 -or $null -eq $fullBank.Payload -or
        $fullBank.Payload.launch_ready -ne $true) {
        Write-Host "Full-bank launch gate failed; no live process was started." -ForegroundColor Red
        Write-Host "完整题库启动门失败；未启动任何实时进程。" -ForegroundColor Red
        exit 4
    }
    Write-Host ("  launch_ready={0}; publication_ready={1}; training_ready={2}" -f `
        $fullBank.Payload.launch_ready, $fullBank.Payload.publication_ready, $fullBank.Payload.training_ready)
    if ($fullBank.Payload.training_ready -ne $true) {
        Write-Host "  training_ready=false 仅表示运行态 Gold/真实工具结果/中文本地化清单尚未齐全；它不授予 LIVE 权限。" -ForegroundColor Yellow
        Write-Host "  training_ready=false means runtime Gold/tool/localization manifests are pending; it does not grant LIVE permission." -ForegroundColor Yellow
    }

    Write-Host "2/3 CC Switch component hash attestation"
    $python = Resolve-AnchorPython
    & $python.Executable @($python.Prefix) $ccSwitchRouteVerifier `
        --manifest $ccSwitchRouteManifest --require-ready
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CC Switch component attestation failed; no live process was started." -ForegroundColor Red
        exit 4
    }
    Write-Host "  这里只证明组件；WSL/Podman 实时可达性只在显式 LIVE 启动后探测。" -ForegroundColor Yellow
    Write-Host "  This proves the component only; WSL/Podman reachability is probed only after explicit LIVE opt-in." -ForegroundColor Yellow

    Write-Host "3/3 Formal coordinator offline gates"
    $offlineArguments = @($python.Prefix) + @(
        $sweCoordinator,
        "--config", $coordinatorConfigPath
    )
    $offlineRaw = @(& $python.Executable @offlineArguments)
    $offlineExitCode = $LASTEXITCODE
    $offlineText = $offlineRaw -join "`n"
    if (-not [string]::IsNullOrWhiteSpace($offlineText)) {
        Write-Host $offlineText
    }
    if ($offlineExitCode -ne 0) {
        Write-Host "Formal coordinator offline preflight failed; no credential was read and no live process was started." -ForegroundColor Red
        exit $offlineExitCode
    }
    try {
        $offlineReport = $offlineText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        Write-Host "Formal coordinator returned invalid offline gate JSON; LIVE is blocked." -ForegroundColor Red
        exit 4
    }

    if (-not $ConfirmLive) {
        Write-Host "  默认仅离线预检：不读凭据、不启动路由/OpenCode/沙箱、不发 Provider 请求。" -ForegroundColor Green
        Write-Host "  Offline preflight only: no credential read, route/OpenCode/sandbox start, or provider request." -ForegroundColor Green
        if ($offlineReport.live_start_allowed -eq $true) {
            Write-Host "  四闸门已允许尝试 LIVE；真跑请重新执行并显式增加 -ConfirmLive。" -ForegroundColor Yellow
            Write-Host "  All four gates allow a LIVE attempt; re-run with -ConfirmLive." -ForegroundColor Yellow
        }
        else {
            Write-Host ("  LIVE 仍关闭：execution_contract_ready={0}; live_start_allowed={1}; reason={2}" -f `
                $offlineReport.execution_contract_ready, $offlineReport.live_start_allowed, $offlineReport.reason_code) -ForegroundColor Red
            Write-Host "  状态来自正式协调器兼容的 content-free gate payload；请按 reason_code 修复。" -ForegroundColor Red
            Write-Host "  Gates come from the coordinator-compatible content-free payload; fix the reported reason_code." -ForegroundColor Red
        }
        exit 0
    }

    if ($offlineReport.component_ready -ne $true -or
        $offlineReport.bank_ready -ne $true -or
        $offlineReport.execution_contract_ready -ne $true -or
        $offlineReport.live_start_allowed -ne $true) {
        Write-Host ("LIVE blocked before credential/route/API: component={0}; bank={1}; execution_contract={2}; live_start_allowed={3}; reason={4}" -f `
            $offlineReport.component_ready, $offlineReport.bank_ready, `
            $offlineReport.execution_contract_ready, $offlineReport.live_start_allowed, `
            $offlineReport.reason_code) -ForegroundColor Red
        exit 4
    }

    $controlRunId = "anchor-" + ([Guid]::NewGuid().ToString("N"))
    $liveArguments = @($offlineArguments) + @(
        "--confirm-live",
        "--control-run-id", $controlRunId
    )
    if ($Resume) {
        $liveArguments += "--resume"
    }
    if ($Concurrency -gt 0) {
        $liveArguments += @("--concurrency", "$Concurrency")
    }
    if ($MaxTasks -gt 0) {
        $liveArguments += @("--max-tasks", "$MaxTasks")
    }
    Write-Host "LIVE 已显式确认且四闸门通过：协调器将探测实时路由/容器，再读取进程环境凭据并启动或续跑。" -ForegroundColor Cyan
    Write-Host "LIVE confirmed and all gates passed: the coordinator probes route/container reachability, then reads process credentials and starts or resumes." -ForegroundColor Cyan
    & $python.Executable @liveArguments
    $coordinatorExitCode = $LASTEXITCODE
    if ($coordinatorExitCode -ne 0) {
        Write-Host "SWE-bench coordinator exited with code $coordinatorExitCode." -ForegroundColor Red
    }
    exit $coordinatorExitCode
}

function Start-LegacySyntheticDistillation {
    Write-AnchorTitle "Legacy synthetic 蒸馏 / distillation"
    Write-Host "警告：此路径是 CompatibleTeacher synthetic 五专家数据，不含真实 patched OpenCode 工具轨迹。" -ForegroundColor Yellow
    Write-Host "Warning: this is the CompatibleTeacher synthetic path, not patched OpenCode tool trajectories." -ForegroundColor Yellow
    if (-not $ConfirmLegacySynthetic) {
        Write-Host "未启动。显式同意请加 -ConfirmLegacySynthetic；key 必须已在 YAML 指定的进程环境变量中。"
        Write-Host "Not started. Add -ConfirmLegacySynthetic; the key must already be in the YAML-declared process environment."
        exit 5
    }
    $code = Invoke-AutomationLauncher -Arguments @()
    exit $code
}

function Refuse-AmbiguousRun {
    Write-AnchorTitle "运行 / Run"
    Write-Host "run 名称过于含糊，已拒绝 / Ambiguous 'run' was refused." -ForegroundColor Red
    Write-Host "请选择 distill-swebench 或明确标注 legacy 的 distill-synthetic。"
    Write-Host "Choose distill-swebench or explicitly legacy distill-synthetic."
    exit 3
}

if ($Action -eq "menu") {
    Write-AnchorTitle "控制台 / Console"
    Write-Host "1) status              状态（只读）"
    Write-Host "2) ui                  恢复/显示 8765 面板"
    Write-Host "3) preflight           双魔改链预检"
    Write-Host "4) distill-swebench    正式链离线预检（LIVE 需 -ConfirmLive）"
    Write-Host "5) distill-synthetic   明确标注的 legacy synthetic"
    Write-Host "6) docs                文档路径（只读）"
    $selection = Read-Host "选择 / Select [1-6]"
    $Action = switch ($selection) {
        "1" { "status" }
        "2" { "ui" }
        "3" { "preflight" }
        "4" { "distill-swebench" }
        "5" { "distill-synthetic" }
        "6" { "docs" }
        default { throw "无效选择 / Invalid selection." }
    }
}

switch ($Action) {
    "status" { Show-Status }
    "preflight" { Show-Preflight }
    "ui" { Show-UI }
    "dashboard" { Show-UI }
    "docs" { Show-Docs }
    "distill-swebench" { Start-SWEBenchDistillation }
    "distill-synthetic" { Start-LegacySyntheticDistillation }
    "run" { Refuse-AmbiguousRun }
    default { throw "不支持的操作 / Unsupported action: $Action" }
}
