param(
    [string]$EnvironmentPath = "E:\isaacsim\venv-6.1",
    [switch]$CompatibilityOnly,
    [switch]$AcceptEula
)

$ErrorActionPreference = "Stop"
$IsaacSimVersion = "6.1.0.0"
$EnvironmentPath = [System.IO.Path]::GetFullPath($EnvironmentPath)
$EnvironmentParent = Split-Path -Parent $EnvironmentPath
$PipCache = Join-Path $EnvironmentParent "pip-cache"

function Assert-NativeSuccess {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Isaac Sim environment: $EnvironmentPath"
Write-Host "Pip download cache:     $PipCache"

New-Item -ItemType Directory -Force -Path $EnvironmentParent, $PipCache | Out-Null
$env:PIP_CACHE_DIR = $PipCache

$EnvironmentCreated = $false
if (Get-Command py.exe -ErrorAction SilentlyContinue) {
    & py.exe -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12); print(sys.version)"
    if ($LASTEXITCODE -eq 0) {
        & py.exe -3.12 -m venv $EnvironmentPath
        Assert-NativeSuccess "Virtual environment creation"
        $EnvironmentCreated = $true
    }
}
if (-not $EnvironmentCreated -and (Get-Command python3.12.exe -ErrorAction SilentlyContinue)) {
    & python3.12.exe -m venv $EnvironmentPath
    Assert-NativeSuccess "Virtual environment creation"
    $EnvironmentCreated = $true
}
if (-not $EnvironmentCreated -and (Get-Command python.exe -ErrorAction SilentlyContinue)) {
    & python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Using Python 3.12 from the currently active environment."
        & python.exe -m venv $EnvironmentPath
        Assert-NativeSuccess "Virtual environment creation"
        $EnvironmentCreated = $true
    }
}
if (-not $EnvironmentCreated -and (Get-Command conda.exe -ErrorAction SilentlyContinue)) {
    Write-Host "Python 3.12 was not found globally; creating the environment with Conda."
    & conda.exe create --yes --prefix $EnvironmentPath python=3.12 pip
    Assert-NativeSuccess "Conda environment creation"
    $EnvironmentCreated = $true
}
if (-not $EnvironmentCreated) {
    throw @"
Python 3.12 and Conda were not found. Install Python, open a new PowerShell, then rerun:
  winget install --exact --id Python.Python.3.12
"@
}

$Python = Join-Path $EnvironmentPath "Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
Assert-NativeSuccess "Pip bootstrap"

if ($CompatibilityOnly) {
    & $Python -m pip install "isaacsim[compatibility-check]==$IsaacSimVersion" `
        --extra-index-url https://pypi.nvidia.com
    Assert-NativeSuccess "Isaac Sim compatibility checker installation"
}
else {
    # The host reports a CUDA 13-capable driver. This is the CUDA 13 wheel from
    # the Isaac Sim 6.1 installation guide; it does not install a system CUDA toolkit.
    & $Python -m pip install torch==2.11.0 `
        --index-url https://download.pytorch.org/whl/cu130
    Assert-NativeSuccess "PyTorch installation"
    & $Python -m pip install "isaacsim[all,extscache]==$IsaacSimVersion" `
        --extra-index-url https://pypi.nvidia.com
    Assert-NativeSuccess "Isaac Sim installation"
}

if ($AcceptEula) {
    [Environment]::SetEnvironmentVariable("OMNI_KIT_ACCEPT_EULA", "YES", "User")
    $env:OMNI_KIT_ACCEPT_EULA = "YES"
    Write-Host "NVIDIA Omniverse EULA acceptance was saved for the current user."
}
else {
    Write-Warning @"
The EULA has not been accepted automatically. On first launch, review and answer the
NVIDIA prompt, or rerun with -AcceptEula after reviewing:
https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html
"@
}

Write-Host ""
Write-Host "Installation command finished. Next checks:"
Write-Host "  & '$Python' -m pip show isaacsim"
Write-Host "  & '$(Join-Path $EnvironmentPath "Scripts\isaacsim.exe")' isaacsim.exp.compatibility_check"
