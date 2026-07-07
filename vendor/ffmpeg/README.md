# Portable media tools

The Python pipeline invokes **the external converter** and **the probe tool**. Either:

1. Run **`setup.bat`** from the repo root (recommended — downloads automatically), or
2. Install the converter system-wide so it is on `PATH`.

## Layout (after setup)

Binaries are placed here by the setup script:

| File | Purpose |
|------|---------|
| `bin/ffmpeg.exe` | Windows |
| `bin/ffprobe.exe` | Windows |

The app resolves `vendor/ffmpeg/bin/ffmpeg` before looking at `PATH` (see `data_pipeline/ffmpeg_paths.py`).

Override paths if needed:

```text
set PHONE_PIPELINE_FFMPEG=C:\path\to\ffmpeg.exe
set PHONE_PIPELINE_FFPROBE=C:\path\to\ffprobe.exe
```

## Manual download

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_ffmpeg_windows.ps1
```

## Legal

The external converter is licensed as LGPL/GPL depending on build configuration. Use official builds from reputable sources (e.g. gyan.dev Windows builds).
