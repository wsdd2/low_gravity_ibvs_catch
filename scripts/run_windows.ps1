param(
    [string]$EnvironmentPath = "E:\isaacsim\venv-6.1",
    [switch]$Headless,
    [double]$Duration = 0,
    [int]$Seed = 7,
    [int]$PhysicsHz = 120,
    [int]$RenderHz = 15,
    [double]$TeleopLinearSpeed = 0.12,
    [double]$TeleopAngularSpeed = 45.0,
    [switch]$DisableTeleop,
    [switch]$AutoGrasp,
    [switch]$Cheats,
    [switch]$OracleBaseline,
    [double]$BallRadius = 0.045,
    [double]$BallSpeed = 0.035,
    [double]$GraspTimeout = 20.0,
    [double]$DespawnMargin = 0.12,
    [double]$RespawnDelay = 1.0,
    [switch]$SinglePass,
    [string]$CaptureDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $EnvironmentPath "Scripts\python.exe"
$Scenario = Join-Path $ProjectRoot "space_grasp_sim.py"

if (-not (Test-Path $Python)) {
    throw "Isaac Sim Python was not found at '$Python'. Run install_isaacsim_windows.ps1 first."
}

& $Python -m pip show isaacsim-core *> $null
if ($LASTEXITCODE -ne 0) {
    throw @"
Only the Isaac Sim compatibility checker is installed. Install the full bundle first:
  .\scripts\install_isaacsim_windows.ps1
"@
}

$Arguments = @(
    $Scenario,
    "--duration", $Duration,
    "--seed", $Seed,
    "--physics-hz", $PhysicsHz,
    "--render-hz", $RenderHz,
    "--teleop-linear-speed", $TeleopLinearSpeed,
    "--teleop-angular-speed", $TeleopAngularSpeed,
    "--ball-radius", $BallRadius,
    "--speed", $BallSpeed,
    "--grasp-timeout", $GraspTimeout,
    "--despawn-margin", $DespawnMargin,
    "--respawn-delay", $RespawnDelay,
    "--save-usd", (Join-Path $ProjectRoot "outputs\space_grasp_scene.usda")
)

if ($Headless) {
    $Arguments += "--headless"
}
if ($DisableTeleop) {
    $Arguments += "--disable-teleop"
}
if ($AutoGrasp) {
    $Arguments += "--auto-grasp"
}
if ($Cheats) {
    $Arguments += "--cheats"
}
if ($OracleBaseline) {
    $Arguments += "--oracle-baseline"
}
if ($SinglePass) {
    $Arguments += "--single-pass"
}
if ($CaptureDir) {
    $Arguments += @("--capture-dir", [System.IO.Path]::GetFullPath($CaptureDir))
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Sim scenario failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
