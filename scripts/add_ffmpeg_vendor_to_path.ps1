# Idempotent: append repo vendor\ffmpeg\bin to the *user* PATH (not system).
# Run after bootstrap_ffmpeg_windows.ps1, or if those binaries were copied in manually.
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $RepoRoot 'vendor\ffmpeg\bin'
$ffmpeg = Join-Path $Bin 'ffmpeg.exe'
if (-not (Test-Path $ffmpeg)) {
    Write-Error "Missing $ffmpeg — run setup_ffmpeg_windows.bat or scripts\bootstrap_ffmpeg_windows.ps1 first."
    exit 1
}
$BinFull = (Resolve-Path $Bin).Path.TrimEnd('\')
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $userPath) { $userPath = '' }
$segments = @(
    $userPath -split ';' |
    ForEach-Object { $_.Trim().TrimEnd('\') } |
    Where-Object { $_ -ne '' }
)
if ($segments -contains $BinFull) {
    Write-Host "Already on user PATH: $BinFull"
    exit 0
}
$newPath = if ($userPath.Trim()) { "$userPath;$BinFull" } else { $BinFull }
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
Write-Host "Added to user PATH: $BinFull"
Write-Host "Open a NEW terminal (or sign out) so PATH updates are picked up."
exit 0
