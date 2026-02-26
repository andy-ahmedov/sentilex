param(
    [switch]$OneFile,
    [string]$VenvPath = ".venv-win"
)

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This script must be run on Windows PowerShell."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$venvAbs = Join-Path $repoRoot $VenvPath
if (-not (Test-Path $venvAbs)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $venvAbs
        if ($LASTEXITCODE -ne 0) {
            & py -3.10 -m venv $venvAbs
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.10/3.11 not found via py launcher. Install Python 3.10 or 3.11 and retry."
        }
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $venvAbs
    }
    else {
        throw "Python launcher not found. Install Python 3.10+ and retry."
    }
}

$venvPython = Join-Path $venvAbs "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtualenv python not found at $venvPython"
}

$venvVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([Version]$venvVersion -ge [Version]"3.12") {
    throw "Detected Python $venvVersion in $VenvPath. Use Python 3.10 or 3.11 for Sentilex desktop build (pymorphy2 compatibility)."
}

Get-Process -Name "Sentilex" -ErrorAction SilentlyContinue | Stop-Process -Force
if (Test-Path (Join-Path $repoRoot "build")) {
    Remove-Item (Join-Path $repoRoot "build") -Recurse -Force
}
if (Test-Path (Join-Path $repoRoot "dist")) {
    Remove-Item (Join-Path $repoRoot "dist") -Recurse -Force
}

Write-Host "[build] pip upgrade"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE"
}

Write-Host "[build] pin setuptools<81"
& $venvPython -m pip install "setuptools<81"
if ($LASTEXITCODE -ne 0) {
    throw "setuptools pin failed with exit code $LASTEXITCODE"
}

Write-Host "[build] install runtime deps"
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "runtime dependency install failed with exit code $LASTEXITCODE"
}

Write-Host "[build] install desktop deps"
& $venvPython -m pip install -r desktop/requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "desktop dependency install failed with exit code $LASTEXITCODE"
}

Write-Host "[build] install pyinstaller"
& $venvPython -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "pyinstaller install failed with exit code $LASTEXITCODE"
}

if ($OneFile) {
    $env:SENTILEX_ONEFILE = "1"
}
else {
    $env:SENTILEX_ONEFILE = "0"
}

Write-Host "[build] run pyinstaller"
& $venvPython -m PyInstaller --clean --noconfirm desktop/sentilex.spec
if ($LASTEXITCODE -ne 0) {
    throw "pyinstaller build failed with exit code $LASTEXITCODE"
}

if ($OneFile) {
    Write-Host "Build complete (onefile): dist/Sentilex.exe"
}
else {
    Write-Host "Build complete (onedir): dist/Sentilex/"
}
