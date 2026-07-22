# Downloads pip wheels for GeoTIFF basemap support into vendor/wheels.
# Run ONCE on an online PC (same Python MAJOR.MINOR as offline), then copy vendor\wheels USB → offline.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Req = Join-Path $Root "requirements-basemap.txt"
$Dest = Join-Path $Root "vendor\wheels"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Push-Location $Root
try {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m pip download --upgrade --dest $Dest -r $Req
    }
    else {
        python -m pip download --upgrade --dest $Dest -r $Req
    }
    Write-Host ""
    Write-Host "Done. Wheels are in: $Dest"
    Write-Host "Copy 'vendor\wheels' together with the repo to your offline PC, then run setup_offline_windows.bat"
} finally {
    Pop-Location
}
