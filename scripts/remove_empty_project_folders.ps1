$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$report = Join-Path $root 'runs/project363/project_cleanup_20261027'
$excluded = @('.git', '.codex', '.agents', '.aws', '.tools', 'bai_ca_nhan_cnn_spark')
$protected = @($root)
foreach ($name in @('data','weights','src','scripts','tests','config','notebook','runs')) {
    $protected += Join-Path $root $name
}
foreach ($name in @('YOLO363_balanced_720p_20261027','YOLO363_training_working','Tracking363_working','VLM363_working')) {
    $excluded += "data\$name"
}
$stack = [Collections.Generic.Stack[string]]::new()
$stack.Push($root)
$folders = [Collections.Generic.List[string]]::new()
$skipped = [Collections.Generic.List[string]]::new()
while ($stack.Count -gt 0) {
    $parent = $stack.Pop()
    try { $children = @([IO.Directory]::EnumerateDirectories($parent)) }
    catch { $skipped.Add($parent); continue }
    foreach ($directory in $children) {
        $relative = $directory.Substring($root.Length + 1)
        if ($relative -in $excluded) { continue }
        if (([IO.File]::GetAttributes($directory) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            $skipped.Add($directory)
            continue
        }
        $folders.Add($directory)
        $stack.Push($directory)
    }
}
$deleted = [Collections.Generic.List[string]]::new()
foreach ($folder in ($folders | Sort-Object { $_.Length } -Descending)) {
    if ($folder -in $protected) { continue }
    if ($folder -in $skipped) { continue }
    $resolved = (Resolve-Path -LiteralPath $folder).Path
    if (-not $resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Outside workspace' }
    $ancestor = $resolved
    while ($ancestor -ne $root) {
        if (([IO.File]::GetAttributes($ancestor) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Linked path: $ancestor"
        }
        $ancestor = [IO.Directory]::GetParent($ancestor).FullName
    }
    try {
        $entries = [IO.Directory]::EnumerateFileSystemEntries($resolved).GetEnumerator()
        $nonempty = $entries.MoveNext()
    }
    catch { $skipped.Add($resolved); continue }
    if ($entries -is [IDisposable]) { $entries.Dispose() }
    if ($nonempty) { continue }
    [IO.Directory]::Delete($resolved, $false)
    $deleted.Add($resolved)
}
$result = [PSCustomObject]@{status='COMPLETE_EMPTY_DIRECTORIES_ONLY';deleted_count=$deleted.Count;deleted_paths=@($deleted.ToArray());skipped_link_or_inaccessible_paths=@($skipped.ToArray());excluded_subtrees=$excluded;completed_utc=[DateTime]::UtcNow.ToString('o');nonempty_files_deleted=$false}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report 'empty_directory_cleanup.json') -Encoding UTF8
Write-Output "Deleted empty directories: $($deleted.Count)"
$deleted | Select-Object -First 35
