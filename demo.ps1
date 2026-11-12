# Demo launcher (ASCII-only on purpose: Windows PowerShell 5.1 reads UTF-8 without BOM as ANSI).
# Usage:  run_demo.bat                 -> default query (see code/demo_search.py default)
#         run_demo.bat "手机壳"        -> custom query
#         run_demo.bat "手机壳" 5 5    -> query, topk, images
param(
    [string]$Query,
    [int]$Topk = 5,
    [int]$Images = 3000
)

$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "code\demo_search.py"

if (-not (Test-Path $py)) {
    Write-Host "[ERROR] python venv not found: $py" -ForegroundColor Red
    Write-Host "Run this first:  uv venv --python 3.12 .venv   then reinstall deps." -ForegroundColor Yellow
    exit 1
}

$argv = @($script, "--images", $Images, "--topk", $Topk)
if ($Query) { $argv += @("--query", $Query) }

$env:PYTHONIOENCODING = "utf-8"
& $py @argv
$code = $LASTEXITCODE

$outDir = Join-Path $root "results\demo"
Write-Host ""
if ($code -eq 0) {
    Write-Host "[OK] result images are in: $outDir" -ForegroundColor Green
    if (Test-Path $outDir) { Start-Process explorer.exe $outDir }
} else {
    Write-Host "[FAIL] exit code $code" -ForegroundColor Red
}
