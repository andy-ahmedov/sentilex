param(
    [string]$ReportName = "frozen-smoke",
    [string]$ExePath = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
$sourceDir = Join-Path $repoRoot "dist"
$exeDir = Join-Path $env:LOCALAPPDATA "SentilexBuild\smoke-$ReportName"
New-Item -ItemType Directory -Force -Path $exeDir | Out-Null
$exe = Join-Path $exeDir "SentiSoft-NRC2022.exe"
if (-not $ExePath) { $ExePath = Join-Path $sourceDir "SentiSoft-NRC2022.exe" }
Copy-Item -LiteralPath $ExePath -Destination $exe -Force
$report = Join-Path $exeDir "$ReportName.json"
if (Test-Path $report) { Remove-Item $report }
$keys = @("SENTILEX_SMOKE_INPUT", "SENTILEX_SMOKE_OFFLINE", "SENTILEX_SMOKE_REPORT")
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }
try {
    $env:SENTILEX_SMOKE_INPUT = Join-Path $repoRoot "texts\chast__7.txt"
    $env:SENTILEX_SMOKE_OFFLINE = "1"
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
    if (-not $data.emotions) { throw "Missing built-in NRC2022 outputs" }
    $paths += @($data.emotions.png, $data.emotions.txt, $data.emotions.csv, $data.emotions.html,
                $data.emotions.density_png, $data.emotions.sentences_csv)
    if ($data.dictionary.entries -ne 14154 -or $data.dictionary.categories.Count -ne 8 -or
        -not $data.dictionary.readme_exists -or
        $data.dictionary.translation -ne "Russian; Google Translate; August 2022" -or
        -not $data.dictionary.source_exists -or -not $data.dictionary.checksum_exists -or
        -not $data.dictionary.offline -or
        $data.dictionary.sha256 -ne "96849cb99a395e1e749030fa6c115860e2664f7690a1beac82c47485d42ce75a") {
        throw "Bundled dictionary provenance/offline verification failed"
    }
    foreach ($path in $paths) {
        $resolved = Join-Path $exeDir $path
        if (-not (Test-Path $resolved) -or (Get-Item $resolved).Length -eq 0) { throw "Missing or empty output: $resolved" }
    }
    $actual = [IO.File]::ReadAllText((Join-Path $exeDir $data.result_txt)).Replace("`r`n", "`n")
    $expected = [IO.File]::ReadAllText((Join-Path $repoRoot "results\chast_7_result.txt")).Replace("`r`n", "`n")
    if ($actual -cne $expected) { throw "Frozen TXT differs from golden baseline (after CRLF normalization)" }
    Copy-Item $report (Join-Path $sourceDir "$ReportName.json") -Force
    Copy-Item (Join-Path $exeDir "$ReportName.png") (Join-Path $sourceDir "$ReportName.png") -Force
    Copy-Item (Join-Path $exeDir "${ReportName}_markers.png") (Join-Path $sourceDir "${ReportName}_markers.png") -Force
    Copy-Item (Join-Path $exeDir "${ReportName}_density.png") (Join-Path $sourceDir "${ReportName}_density.png") -Force
    Write-Host "[smoke-frozen] OK: real offline exe worker/UI, bundled NRC2022, six emotion artifacts and golden TXT"
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], "Process") }
}
