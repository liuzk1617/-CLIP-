# Products-10K 分批续传：每批并行 4 个，带单传输超时；循环直到 39 个分片字节数全部一致。
$ErrorActionPreference = 'Continue'
$root = "D:\Desktop\课内\人工智能导论\期末大作业"
$dst = "$root\data\raw\products10k_v1\data"
$log = "$root\logs\download_p10k_resume.log"
$api = "https://hf-mirror.com/api/datasets/nyris/products10k-v1/tree/main/data"
$base = "https://hf-mirror.com/datasets/nyris/products10k-v1/resolve/main/"

"=== resume start $(Get-Date -Format s) ===" | Out-File $log -Encoding utf8

for ($round = 1; $round -le 8; $round++) {
    $json = curl.exe -sL --connect-timeout 20 $api
    $files = $json | ConvertFrom-Json | Where-Object { $_.path -like '*.parquet' }
    $bad = @()
    foreach ($f in $files) {
        $name = Split-Path $f.path -Leaf
        $t = Join-Path $dst $name
        $have = if (Test-Path $t) { (Get-Item $t).Length } else { 0 }
        if ($have -ne $f.size) { $bad += [pscustomobject]@{ name = $name; path = $f.path; size = $f.size; have = $have } }
    }
    $missingGB = (($bad | Measure-Object size -Sum).Sum - ($bad | Measure-Object have -Sum).Sum) / 1GB
    "round $round : 不完整 $($bad.Count) 个, 需补 {0:N2} GB" -f $missingGB | Tee-Object -FilePath $log -Append
    if ($bad.Count -eq 0) { break }

    for ($i = 0; $i -lt $bad.Count; $i += 4) {
        $batch = $bad[$i..([Math]::Min($i + 3, $bad.Count - 1))]
        $cargs = @('-sL', '--parallel', '--parallel-max', '4', '--max-time', '420',
                   '--retry', '3', '--retry-delay', '2', '--retry-all-errors',
                   '-C', '-', '--connect-timeout', '20')
        foreach ($b in $batch) {
            $cargs += '-o'; $cargs += (Join-Path $dst $b.name)
            $cargs += ($base + $b.path)
        }
        & curl.exe @cargs 2>&1 | Out-Null
        $sz = ($batch | ForEach-Object { if (Test-Path (Join-Path $dst $_.name)) { (Get-Item (Join-Path $dst $_.name)).Length } else { 0 } } | Measure-Object -Sum).Sum
        "  batch done: {0:N2} GB (batch {1})" -f ($sz / 1GB), ($i / 4 + 1) | Tee-Object -FilePath $log -Append
    }
}

$json = curl.exe -sL --connect-timeout 20 $api
$files = $json | ConvertFrom-Json | Where-Object { $_.path -like '*.parquet' }
$ok = 0
foreach ($f in $files) {
    $t = Join-Path $dst (Split-Path $f.path -Leaf)
    if ((Test-Path $t) -and (Get-Item $t).Length -eq $f.size) { $ok++ }
}
$total = (Get-ChildItem $dst -Filter *.parquet | Measure-Object Length -Sum).Sum
"=== resume done $(Get-Date -Format s) : 完整 $ok/39, 总计 {0:N2} GB ===" -f ($total / 1GB) | Tee-Object -FilePath $log -Append
