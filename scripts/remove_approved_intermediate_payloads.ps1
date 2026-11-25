$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$report = Join-Path $root 'runs/project363/project_cleanup_20261027'
$datasets = @('YOLO363_filtered_720p_20261026', 'YOLO363_filtered_720p_20261026_strict')
$targets = @()
$protected = @()
foreach ($dataset in $datasets) {
    $datasetRoot = (Resolve-Path -LiteralPath (Join-Path $root "data/$dataset")).Path
    if ($datasetRoot -ne (Join-Path $root "data\$dataset")) { throw 'Unexpected dataset path' }
    foreach ($kind in @('vehicle', 'light')) {
        foreach ($category in @('images', 'labels')) {
            $expected = Join-Path $datasetRoot "$kind\$category"
            $resolved = (Resolve-Path -LiteralPath $expected).Path
            if ($resolved -ne $expected -or -not $resolved.StartsWith($datasetRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Deletion target outside approved dataset'
            }
            foreach ($ancestor in @($datasetRoot, (Join-Path $datasetRoot $kind), $resolved)) {
                if (((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw "Linked target or ancestor: $ancestor"
                }
            }
            $entries = @(Get-ChildItem -LiteralPath $resolved -Recurse -Force)
            if (@($entries | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count -gt 0) {
                throw "Linked content: $resolved"
            }
            $files = @($entries | Where-Object { -not $_.PSIsContainer })
            $targets += [PSCustomObject]@{path=$resolved;files=$files.Count;bytes=($files | Measure-Object -Property Length -Sum).Sum;deleted=$false}
        }
    }
    foreach ($file in Get-ChildItem -LiteralPath $datasetRoot -File -Recurse -Force) {
        if (-not @($targets | Where-Object { $file.FullName.StartsWith($_.path + '\', [StringComparison]::OrdinalIgnoreCase) }).Count) {
            $protected += [PSCustomObject]@{path=$file.FullName;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
        }
    }
}
$receipt = [PSCustomObject]@{status='VERIFIED_TARGETS';approved_scope='Only images and labels in two named intermediate datasets';targets=$targets;preserved_files=$protected;started_utc=[DateTime]::UtcNow.ToString('o')}
$receiptPath = Join-Path $report 'intermediate_payload_cleanup.json'
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
foreach ($target in $targets) {
    Write-Output "Deleting verified target: $($target.path) ($($target.files) files)"
    Remove-Item -LiteralPath $target.path -Recurse -Force
    if (Test-Path -LiteralPath $target.path) { throw 'Target still exists after deletion' }
    $target.deleted = $true
    $receipt.status = 'DELETING_APPROVED_PAYLOADS'
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
}
foreach ($file in $protected) {
    if ((Get-FileHash -LiteralPath $file.path -Algorithm SHA256).Hash -ne $file.sha256) {
        throw "Preserved file changed: $($file.path)"
    }
}
$receipt.status = 'COMPLETE_PRESERVED_METADATA_HASHES_VERIFIED'
$receipt | Add-Member -NotePropertyName completed_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o'))
$receipt | Add-Member -NotePropertyName deleted_bytes -NotePropertyValue (($targets | Measure-Object -Property bytes -Sum).Sum)
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Output "Complete. Deleted logical bytes: $($receipt.deleted_bytes)"
