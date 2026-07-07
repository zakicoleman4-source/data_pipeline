@echo off
REM data_to_frames Windows installer wrapper.
REM   - Resolves the Python on PATH
REM   - Calls install.py which handles pip + gtsam-via-conda fallback
REM   - Pauses at the end so users see the report before the window closes
REM
REM Usage:
REM   setup.bat             :: required deps + try gtsam
REM   setup.bat --dev       :: + pytest + pyinstaller
REM   setup.bat --no-gtsam  :: skip gtsam entirely
setlocal ENABLEDELAYEDEXPANSION

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python.exe not on PATH.
    echo Install Python 3.10+ from https://python.org/downloads/ ^(check "Add to PATH"^).
    pause
    exit /b 1
)

echo === data_to_frames installer ===
python "%~dp0install.py" %*
set INSTALL_RC=%ERRORLEVEL%

echo.
if %INSTALL_RC% NEQ 0 (
    echo Installer exited with code %INSTALL_RC%. See messages above.
) else (
    echo Installation OK. Launch GUI with:  python -m data_pipeline
)
pause
exit /b %INSTALL_RC%
