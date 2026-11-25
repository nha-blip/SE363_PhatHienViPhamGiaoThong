param(
    [string]$DatasetRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'data\bosch_yolo_v1')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$classNames = @('red', 'yellow', 'green', 'off')
$classColors = @{
    0 = [System.Drawing.Color]::Red
    1 = [System.Drawing.Color]::Gold
    2 = [System.Drawing.Color]::LimeGreen
    3 = [System.Drawing.Color]::DeepSkyBlue
}
$qaRoot = Join-Path $DatasetRoot 'qa'
$manifest = Get-Content -Raw (Join-Path $qaRoot 'preview_manifest.json') | ConvertFrom-Json
$font = New-Object System.Drawing.Font('Arial', 14, [System.Drawing.FontStyle]::Bold)
$filenameFont = New-Object System.Drawing.Font('Arial', 16, [System.Drawing.FontStyle]::Bold)

try {
    foreach ($groupProperty in $manifest.groups.PSObject.Properties) {
        $group = $groupProperty.Name
        foreach ($item in @($groupProperty.Value)) {
            $outputPath = Join-Path $qaRoot $item.output
            $outputDirectory = Split-Path $outputPath -Parent
            New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

            $sourceImage = [System.Drawing.Image]::FromFile($item.image)
            try {
                $scale = [Math]::Min(1.0, 960.0 / $sourceImage.Width)
                $outputWidth = [Math]::Max(1, [int][Math]::Round($sourceImage.Width * $scale))
                $outputHeight = [Math]::Max(1, [int][Math]::Round($sourceImage.Height * $scale))
                $canvas = New-Object System.Drawing.Bitmap($outputWidth, $outputHeight)
                try {
                    $graphics = [System.Drawing.Graphics]::FromImage($canvas)
                    try {
                        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                        $graphics.DrawImage($sourceImage, 0, 0, $outputWidth, $outputHeight)

                        $labelLines = Get-Content $item.label
                        foreach ($line in $labelLines) {
                            if ([string]::IsNullOrWhiteSpace($line)) { continue }
                            $fields = $line -split '\s+'
                            if ($fields.Count -ne 5) { continue }
                            $classId = [int]$fields[0]
                            $xCenter = [double]$fields[1]
                            $yCenter = [double]$fields[2]
                            $boxWidth = [double]$fields[3]
                            $boxHeight = [double]$fields[4]
                            $left = [int][Math]::Round(($xCenter - $boxWidth / 2) * $outputWidth)
                            $top = [int][Math]::Round(($yCenter - $boxHeight / 2) * $outputHeight)
                            $width = [int][Math]::Round($boxWidth * $outputWidth)
                            $height = [int][Math]::Round($boxHeight * $outputHeight)
                            $color = $classColors[$classId]
                            $pen = New-Object System.Drawing.Pen($color, 2)
                            $brush = New-Object System.Drawing.SolidBrush($color)
                            try {
                                $graphics.DrawRectangle($pen, $left, $top, $width, $height)
                                $text = "$($classNames[$classId]) $width`x$height"
                                $graphics.DrawString($text, $font, $brush, [Math]::Max(0, $left), [Math]::Max(0, $top - 18))
                            } finally {
                                $pen.Dispose()
                                $brush.Dispose()
                            }
                        }

                        $filename = Split-Path $item.image -Leaf
                        $background = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(190, 0, 0, 0))
                        $foreground = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
                        try {
                            $graphics.FillRectangle($background, 0, 0, $outputWidth, 24)
                            $graphics.DrawString($filename, $filenameFont, $foreground, 4, 2)
                        } finally {
                            $background.Dispose()
                            $foreground.Dispose()
                        }
                        $canvas.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
                    } finally {
                        $graphics.Dispose()
                    }
                } finally {
                    $canvas.Dispose()
                }
            } finally {
                $sourceImage.Dispose()
            }
        }
    }
} finally {
    $font.Dispose()
    $filenameFont.Dispose()
}
