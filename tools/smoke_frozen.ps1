param(
    [string]$NrcPath = "",
    [string]$ReportName = "frozen-smoke",
    [string]$ExePath = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
$sourceDir = Join-Path $repoRoot "dist"
$exeDir = Join-Path $env:LOCALAPPDATA "SentilexBuild\smoke-$ReportName"
New-Item -ItemType Directory -Force -Path $exeDir | Out-Null
$exe = Join-Path $exeDir "SentiSoft.exe"
if (-not $ExePath) { $ExePath = Join-Path $sourceDir "SentiSoft.exe" }
Copy-Item -LiteralPath $ExePath -Destination $exe -Force
$report = Join-Path $exeDir "$ReportName.json"
if (Test-Path $report) { Remove-Item $report }
$keys = @("SENTILEX_SMOKE_INPUT", "SENTILEX_SMOKE_NRC", "SENTILEX_SMOKE_REPORT")
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }
try {
    $env:SENTILEX_SMOKE_INPUT = Join-Path $repoRoot "texts\chast__7.txt"
    $env:SENTILEX_SMOKE_NRC = $NrcPath
    $env:SENTILEX_SMOKE_REPORT = $report
    $process = Start-Process -FilePath $exe -WorkingDirectory $exeDir -PassThru
    if (-not $process.WaitForExit(180000)) {
        Stop-Process -Id $process.Id -Force
        throw "Frozen smoke timed out"
    }
    if (-not (Test-Path $report)) { throw "Frozen smoke did not produce a report; exit $($process.ExitCode)" }
    $data = Get-Content $report -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($data.error) { throw $data.error }
    $paths = @($data.result_txt, $data.result_png, $data.result_py)
    if ($NrcPath) {
        if (-not $data.emotions) { throw "Missing emotion outputs" }
        $paths += @($data.emotions.png, $data.emotions.txt, $data.emotions.csv, $data.emotions.html)
    } elseif ($data.emotions) { throw "Unexpected emotion outputs without NRC" }
    foreach ($path in $paths) {
        $resolved = Join-Path $exeDir $path
        if (-not (Test-Path $resolved) -or (Get-Item $resolved).Length -eq 0) { throw "Missing or empty output: $resolved" }
    }
    $actual = [IO.File]::ReadAllText((Join-Path $exeDir $data.result_txt)).Replace("`r`n", "`n")
    $expected = [IO.File]::ReadAllText((Join-Path $repoRoot "results\chast_7_result.txt")).Replace("`r`n", "`n")
    if ($actual -cne $expected) { throw "Frozen TXT differs from golden baseline (after CRLF normalization)" }
    Copy-Item $report (Join-Path $sourceDir "$ReportName.json") -Force
    Copy-Item (Join-Path $exeDir "$ReportName.png") (Join-Path $sourceDir "$ReportName.png") -Force
    if ($NrcPath) {
        Copy-Item (Join-Path $exeDir "${ReportName}_markers.png") (Join-Path $sourceDir "${ReportName}_markers.png") -Force
    }
    Write-Host "[smoke-frozen] OK: real exe worker/UI and golden TXT; NRC enabled: $([bool]$NrcPath)"
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], "Process") }
}
