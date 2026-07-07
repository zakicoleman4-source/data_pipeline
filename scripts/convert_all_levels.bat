@echo off
:: Run gnsslogger_to_rnx three times on the same input, once per strictness
:: preset, so the operator can compare side-by-side which preset their device
:: needs.
::
:: Usage:
::   convert_all_levels.bat "C:\path\to\measurements_xxx.txt"
::   or drag-drop the .txt file onto this .bat.
::
:: Output (next to the input):
::   <name>_strict.obs       Google decimeter defaults (CNo>=20, SVT-unc<=500ns)
::   <name>_relaxed.obs      CNo>=15, SVT-unc<=1us, ignore multipath flag  (default)
::   <name>_permissive.obs   CNo>=10, SVT-unc<=2us, also keep cycle-slip rows
::
:: Prints epoch + size of each so the operator can pick.

setlocal EnableDelayedExpansion
if "%~1"=="" (
  echo Usage: convert_all_levels.bat ^<measurements_*.txt^>
  echo   or drag-drop a .txt file onto this .bat.
  pause
  exit /b 1
)

set "INPUT=%~f1"
set "STEM=%~dpn1"
set "SCRIPT_DIR=%~dp0..\vendor\android_rinex\src"

echo Input:  !INPUT!
echo Script: !SCRIPT_DIR!
echo.

pushd "%SCRIPT_DIR%"
for %%L in (strict relaxed permissive) do (
  set "OUT=%STEM%_%%L.obs"
  echo --- %%L ---
  py -3 gnsslogger_to_rnx.py "!INPUT!" -o "!OUT!" --keep-level %%L
  if exist "!OUT!" (
    for %%S in ("!OUT!") do echo   %%~zS bytes  ^=^>  !OUT!
  ) else (
    echo   FAILED  ^=^>  !OUT!
  )
  echo.
)
popd

echo Done. Three .obs files written next to the input.
pause
exit /b 0
