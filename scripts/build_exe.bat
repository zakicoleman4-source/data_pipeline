@echo off
:: One-shot exe build from source. Run from the repo root.
:: Requires Python 3.10+ on PATH and internet for pip.
::
:: Steps:
::   1. Install runtime deps (numpy, scipy, opencv-python, tkinterdnd2, pillow)
::   2. Install PyInstaller
::   3. Bootstrap portable ffmpeg under vendor/ffmpeg/ (skipped if already present)
::   4. Build dist/client_pipeline/client_pipeline.exe
::
:: Output: dist\client_pipeline\client_pipeline.exe (+ _internal\ alongside)
:: Hand the whole dist\client_pipeline\ folder to the operator.

setlocal EnableExtensions
cd /d "%~dp0\.."

echo.
echo ====================================
echo   client_pipeline - build exe
echo ====================================
echo.
echo Working directory: %CD%
echo.

:: --- 1. Locate Python ---
where py >nul 2>&1
if errorlevel 1 (
  where python >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Neither 'py' nor 'python' found on PATH.
    echo        Install Python 3.10+ from https://python.org/downloads
    echo        and check "Add to PATH" during install.
    pause
    exit /b 1
  )
  set "PY=python"
) else (
  set "PY=py -3"
)
echo Using: %PY%
%PY% --version
echo.

:: --- 2. pip install runtime + build deps ---
echo Installing runtime dependencies...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: pip install -r requirements.txt failed.
  pause
  exit /b 1
)
%PY% -m pip install pyinstaller
if errorlevel 1 (
  echo ERROR: pip install pyinstaller failed.
  pause
  exit /b 1
)

:: --- 3. Bootstrap vendor\ffmpeg if missing ---
if not exist "vendor\ffmpeg\bin\ffmpeg.exe" (
  echo.
  echo Downloading portable FFmpeg...
  powershell -ExecutionPolicy Bypass -File "scripts\bootstrap_ffmpeg_windows.ps1"
  if errorlevel 1 (
    echo ERROR: ffmpeg bootstrap failed. Check internet connection.
    pause
    exit /b 1
  )
) else (
  echo FFmpeg already present at vendor\ffmpeg\bin\ffmpeg.exe
)

:: --- 4. Build exe ---
echo.
echo Building exe (5-10 min)...
%PY% -m PyInstaller data_to_frames.spec --noconfirm --clean
if errorlevel 1 (
  echo ERROR: pyinstaller build failed.
  pause
  exit /b 1
)

:: --- 5. Report ---
if exist "dist\client_pipeline\client_pipeline.exe" (
  echo.
  echo ====================================
  echo   Build succeeded
  echo ====================================
  for %%S in ("dist\client_pipeline\client_pipeline.exe") do echo   exe: %%~zS bytes
  echo   path: %CD%\dist\client_pipeline\client_pipeline.exe
  echo.
  echo Hand the whole dist\client_pipeline\ folder to the operator.
  echo The exe will NOT work on its own - _internal\ must stay next to it.
) else (
  echo ERROR: exe not found at dist\client_pipeline\client_pipeline.exe
  pause
  exit /b 1
)
echo.
pause
exit /b 0
