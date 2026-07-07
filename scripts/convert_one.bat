@echo off
:: Drag-and-drop convenience wrapper.
:: Drop a measurements_*.txt onto this .bat file and it will produce
:: a sibling .obs using the new "relaxed" filter (device-agnostic).
::
:: Usage:
::   convert_one.bat "C:\path\to\measurements_xxx.txt"
::   or drag-drop the .txt onto convert_one.bat in Explorer.
::
:: Output:
::   <same folder>\<same name>.obs  (next to the input)

setlocal
if "%~1"=="" (
  echo Usage: convert_one.bat ^<measurements_*.txt^>
  echo   or drag-drop a .txt file onto this .bat.
  pause
  exit /b 1
)

set "INPUT=%~f1"
set "OUTPUT=%~dpn1.obs"
set "SCRIPT_DIR=%~dp0..\vendor\android_rinex\src"

echo Converting:
echo   input : %INPUT%
echo   output: %OUTPUT%
echo.

pushd "%SCRIPT_DIR%"
py -3 gnsslogger_to_rnx.py "%INPUT%" -o "%OUTPUT%" --keep-level relaxed
set "RC=%ERRORLEVEL%"
popd

echo.
if "%RC%"=="0" (
  echo Done. RINEX OBS written to:
  echo   %OUTPUT%
) else (
  echo Conversion failed (exit code %RC%^). See above for the error.
)
pause
exit /b %RC%
