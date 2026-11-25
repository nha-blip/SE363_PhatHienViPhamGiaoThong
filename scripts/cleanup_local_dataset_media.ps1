param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$data = (Resolve-Path -LiteralPath (Join-Path $root 'data')).Path
$reportDirectory = Join-Path $root 'runs/project363/dataset_cleanup_20261013'
$protected = @('YOLO363_training_working', 'Tracking363_working', 'VLM363_working/external_compilation_15') | ForEach-Object {
    (Resolve-Path -LiteralPath (Join-Path $data $_)).Path
}
$compilation = @(Get-ChildItem -LiteralPath $data -File | Where-Object { $_.Name -like 'AQOcupd*.mp4' })
if ($compilation.Count -ne 1) { throw 'Expected exactly one original compilation video' }
$protected += $compilation[0].FullName
$sourceDirectories = @('archive', 'archive (1)', 'dataset_train_rgb', 'MIO-TCD-Localization', 'imptc_samples')
$mediaExtensions = @('.mp4', '.avi', '.mkv', '.mov', '.webm', '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.zip', '.cache')

function Assert-SafeTarget([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($data + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside data: $absolute"
    }
    foreach ($keep in $protected) {
        if ($absolute.Equals($keep, [StringComparison]::OrdinalIgnoreCase) -or
            $absolute.StartsWith($keep + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
            $keep.StartsWith($absolute + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Protected target: $absolute"
        }
    }
    $current = Get-Item -LiteralPath $absolute -Force
    while ($current.FullName -ne $data) {
        if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($current.FullName)" }
        $current = Get-Item -LiteralPath (Split-Path -Parent $current.FullName) -Force
    }
}

function Get-Inventory([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $Path -File -Recurse -Force) } else { @($item) }
    return [pscustomobject]@{path=$Path; files=$files.Count; bytes=[long](($files | Measure-Object Length -Sum).Sum)}
}

$targets = [Collections.Generic.List[object]]::new()
foreach ($name in $sourceDirectories) {
    $path = Join-Path $data $name
    if (Test-Path -LiteralPath $path) {
        Assert-SafeTarget $path
        $inventory = Get-Inventory $path
        $targets.Add([pscustomobject]@{path=$path; recursive=$true; files=$inventory.files; bytes=$inventory.bytes})
    }
}
$remainingRoots = @(Get-ChildItem -LiteralPath $data -Directory | Where-Object {
    $_.Name -notin ($sourceDirectories + @('YOLO363_training_working', 'Tracking363_working'))
})
foreach ($directory in $remainingRoots) {
    foreach ($file in Get-ChildItem -LiteralPath $directory.FullName -File -Recurse -Force) {
        if ($file.Extension.ToLowerInvariant() -notin $mediaExtensions) { continue }
        if ($file.FullName.StartsWith($protected[2] + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { continue }
        Assert-SafeTarget $file.FullName
        $targets.Add([pscustomobject]@{path=$file.FullName; recursive=$false; files=1; bytes=$file.Length})
    }
}
foreach ($file in Get-ChildItem -LiteralPath $data -File) {
    if ($file.Extension.ToLowerInvariant() -eq '.zip') {
        Assert-SafeTarget $file.FullName
        $targets.Add([pscustomobject]@{path=$file.FullName; recursive=$false; files=1; bytes=$file.Length})
    }
}
$snapshots = @($protected | ForEach-Object { Get-Inventory $_ })
$criticalFiles = @($compilation[0].FullName,
    (Join-Path $root 'config/vehicle_dataset.yaml'), (Join-Path $root 'config/traffic_light_dataset.yaml'),
    (Join-Path $data 'Vehicle_Tracking/VNTraffic/VNTraffic_GroundTruth.txt'),
    (Join-Path $data 'Vehicle_Tracking/VNTraffic/VNTraffic_GroundTruth.json'),
    (Join-Path $data 'VLM363_working/external_compilation_15/gt_draft_revision_001/human_event_review.json'))
$criticalFiles += @(Get-ChildItem -LiteralPath (Join-Path $data 'Vehicle_Tracking') -File -Recurse | Where-Object { $_.Extension -in @('.txt', '.json') } | ForEach-Object { $_.FullName })
$hashes = @($criticalFiles | Select-Object -Unique | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 })
$report = [ordered]@{status='PLANNED'; user_scope_approval_date='2026-10-13'; user_reported_drive_backup=$true;
    drive_backup_independently_verified=$false; protected=$snapshots; critical_hashes=$hashes; targets=@($targets.ToArray());
    total_files=($targets | Measure-Object files -Sum).Sum; total_bytes=($targets | Measure-Object bytes -Sum).Sum;
    deleted_targets=0; deleted_logical_bytes=[long]0; error=$null;
    limitation='Restore removed original media from Drive before full-video expansion or historical QA playback'}
$planPath = Join-Path $reportDirectory 'cleanup_execution_plan.json'
if (Test-Path -LiteralPath $planPath) { throw "Execution plan already exists: $planPath" }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding UTF8
Write-Output ('Planned {0} files, {1:N2} GiB' -f $report.total_files, ($report.total_bytes / 1GB))
if (-not $Execute) { return }
try {
    foreach ($target in $targets) {
        Assert-SafeTarget $target.path
        if ($target.recursive) {
            $links = @(Get-ChildItem -LiteralPath $target.path -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
            if ($links.Count) { throw "Nested reparse points: $($target.path)" }
            Write-Output "Removing source directory: $($target.path)"
            Remove-Item -LiteralPath $target.path -Recurse -Force
        } else {
            Remove-Item -LiteralPath $target.path -Force
        }
        $report.deleted_targets++
        $report.deleted_logical_bytes += $target.bytes
    }
    foreach ($before in $snapshots) {
        $after = Get-Inventory $before.path
        if ($after.files -ne $before.files -or $after.bytes -ne $before.bytes) { throw "Protected inventory changed: $($before.path)" }
    }
    foreach ($before in $hashes) {
        if ((Get-FileHash -LiteralPath $before.Path -Algorithm SHA256).Hash -ne $before.Hash) { throw "Critical hash changed: $($before.Path)" }
    }
    if (@($targets | Where-Object { Test-Path -LiteralPath $_.path }).Count) { throw 'Some removal targets remain' }
    $report.status = 'COMPLETE'
} catch {
    $report.status = 'PARTIAL_OR_VERIFICATION_FAILED'
    $report.error = $_.Exception.Message
    throw
} finally {
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reportDirectory 'cleanup_result.json') -Encoding UTF8
}
Write-Output ('Cleanup COMPLETE; protected inventories and critical hashes verified; removed {0:N2} GiB logical data' -f ($report.deleted_logical_bytes / 1GB))
