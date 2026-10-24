# Silent exact-size photo printing for DZ PhotoID Studio (Windows).
# Called by app.py via: powershell -ExecutionPolicy Bypass -File print_sheet.ps1
#   -Path <jpg> [-Printer <name>] [-PaperW <mm>] [-PaperH <mm>] [-Mode fill|exact]
# fill  = image covers the whole paper (ID sheets whose size == paper size)
# exact = image drawn at its true physical size (from its DPI), centered on paper
param(
  [Parameter(Mandatory = $true)][string]$Path,
  [string]$Printer = "",
  [double]$PaperW = 152.4,
  [double]$PaperH = 101.6,
  [string]$Mode = "fill"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$img = [System.Drawing.Image]::FromFile($Path)
$doc = New-Object System.Drawing.Printing.PrintDocument
# StandardPrintController = no on-screen printing dialog (truly silent)
$doc.PrintController = New-Object System.Drawing.Printing.StandardPrintController
if ($Printer -ne "") { $doc.PrinterSettings.PrinterName = $Printer }

$w100 = [int][Math]::Round($PaperW / 25.4 * 100)
$h100 = [int][Math]::Round($PaperH / 25.4 * 100)
$paper = New-Object System.Drawing.Printing.PaperSize("PhotoID-Custom", $w100, $h100)
$doc.DefaultPageSettings.PaperSize = $paper
$doc.DefaultPageSettings.Margins = New-Object System.Drawing.Printing.Margins(0, 0, 0, 0)
$doc.DefaultPageSettings.Landscape = $img.Width -gt $img.Height

$script:img = $img
$script:mode = $Mode
$handler = {
  param($sender, $e)
  $b = $e.PageBounds
  $im = $script:img
  $g = $e.Graphics
  $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $imgLand = $im.Width -gt $im.Height
  $pageLand = $b.Width -gt $b.Height
  if ($imgLand -ne $pageLand) {
    # Driver reports the paper rotated: rotate the drawing to match the image.
    $g.TranslateTransform([float]($b.Width / 2), [float]($b.Height / 2))
    $g.RotateTransform(90)
    $g.TranslateTransform([float](-$b.Height / 2), [float](-$b.Width / 2))
    $dest = New-Object System.Drawing.Rectangle(0, 0, $b.Height, $b.Width)
  } else {
    $dest = New-Object System.Drawing.Rectangle(0, 0, $b.Width, $b.Height)
  }
  if ($script:mode -eq "exact") {
    $dpi = $im.HorizontalResolution
    if ($dpi -le 0) { $dpi = 300 }
    $nw = [int][Math]::Round($im.Width / $dpi * 100)
    $nh = [int][Math]::Round($im.Height / $dpi * 100)
    $x = [int](($dest.Width - $nw) / 2)
    $y = [int](($dest.Height - $nh) / 2)
    $dest = New-Object System.Drawing.Rectangle($x, $y, $nw, $nh)
  }
  $g.DrawImage($im, $dest)
  $e.HasMorePages = $false
}
$doc.add_PrintPage($handler)

try {
  $doc.Print()
  Write-Output "OK"
} catch {
  Write-Error $_.Exception.Message
  exit 1
} finally {
  $doc.Dispose()
  $img.Dispose()
}
