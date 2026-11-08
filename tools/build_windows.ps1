param(
    [switch]$OneFile,
    [string]$VenvPath = ".venv-win"
)

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This script must be run on Windows PowerShell."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
Set-Location $repoRoot

if ($repoRoot.StartsWith("\\") -and $VenvPath -eq ".venv-win") {
    # Windows Python cannot reliably discover site-packages on case-sensitive WSL UNC paths.
    $venvAbs = Join-Path $env:LOCALAPPDATA "SentilexBuild\venv"
}
elseif ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $venvAbs = $VenvPath
}
else {
    $venvAbs = Join-Path $repoRoot $VenvPath
}
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
    throw "Detected Python $venvVersion in $VenvPath. Use Python 3.10 or 3.11 for SentiSoft desktop build (pymorphy2 compatibility)."
}

# Keep running user executables and all previous dist artifacts untouched.
$stageRoot = Join-Path $env:LOCALAPPDATA ("SentilexBuild\stage-" + [guid]::NewGuid().ToString("N"))
$stageDist = Join-Path $stageRoot "dist"
$stageWork = Join-Path $stageRoot "build"
New-Item -ItemType Directory -Force -Path $stageDist | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "dist") | Out-Null

Write-Host "[build] ensure pip"
& $venvPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed with exit code $LASTEXITCODE"
    }
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

if (-not (Test-Path (Join-Path $repoRoot "scripts\data\nrc2022\Russian-NRC-EmoLex.txt"))) {
    Write-Host "[build] prepare local NRC 2022 research resource"
    & $venvPython tools/prepare_nrc2022.py
    if ($LASTEXITCODE -ne 0) { throw "NRC 2022 resource preparation failed" }
}

Write-Host "[build] run pyinstaller"
& $venvPython -m PyInstaller --clean --noconfirm desktop/sentilex.spec --distpath $stageDist --workpath $stageWork
if ($LASTEXITCODE -ne 0) {
    throw "pyinstaller build failed with exit code $LASTEXITCODE"
}

if ($OneFile) {
    $target = Join-Path $repoRoot "dist\SentiSoft-NRC2022.exe"
    Copy-Item (Join-Path $stageDist "SentiSoft-NRC2022.exe") $target -Force
    Write-Host "Build complete (onefile): $target"
}
else {
    $target = Join-Path $repoRoot "dist\SentiSoft-NRC2022"
    Copy-Item (Join-Path $stageDist "SentiSoft-NRC2022") (Join-Path $repoRoot "dist") -Recurse -Force
    Write-Host "Build complete (onedir): $target"
}
Write-Host "NTFS staging retained: $stageRoot"
