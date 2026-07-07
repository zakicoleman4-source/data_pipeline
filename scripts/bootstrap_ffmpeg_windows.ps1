<#
.SYNOPSIS
  Download Windows ffmpeg essentials (ffmpeg + ffprobe) into vendor/ffmpeg/bin.
  Called by setup.bat. Requires internet and PowerShell 5+.
#>
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Bin = Join-Path $Root 'vendor\ffmpeg\bin'
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

$Url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
$Zip = Join-Path $env:TEMP ("ffmpeg-essentials-" + [Guid]::NewGuid().ToString() + '.zip')
$Extract = Join-Path $env:TEMP ("ffmpeg-extract-" + [Guid]::NewGuid().ToString())

Write-Host "Downloading: $Url"
Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing

Write-Host "Extracting to: $Extract"
Expand-Archive -Path $Zip -DestinationPath $Extract -Force

$ffmpeg = Get-ChildItem -Path $Extract -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
if (-not $ffmpeg) {
    throw "ffmpeg.exe not found in extracted archive."
}
$ffprobe = Join-Path $ffmpeg.DirectoryName 'ffprobe.exe'
if (-not (Test-Path $ffprobe)) {
    throw "ffprobe.exe not found next to ffmpeg.exe"
}

Copy-Item -Path $ffmpeg.FullName -Destination (Join-Path $Bin 'ffmpeg.exe') -Force
Copy-Item -Path $ffprobe -Destination (Join-Path $Bin 'ffprobe.exe') -Force

Remove-Item $Zip -Force -ErrorAction SilentlyContinue
Remove-Item $Extract -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Done. Binaries:"
Get-Item (Join-Path $Bin 'ffmpeg.exe'), (Join-Path $Bin 'ffprobe.exe') | Format-Table FullName, Length

& (Join-Path $Bin 'ffmpeg.exe') -version | Select-Object -First 1
