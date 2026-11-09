@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"

cd /d "%~dp0"
if errorlevel 1 goto :path_failed

if exist ".venv\Scripts\python.exe" goto :run_venv

set "PY="
py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py"
if defined PY goto :run_system
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if not defined PY goto :python_missing

:run_system
"%PY%" diagnostics.py
if errorlevel 1 goto :diagnostics_failed
goto :success

:run_venv
".venv\Scripts\python.exe" diagnostics.py
if errorlevel 1 goto :diagnostics_failed

:success
echo.
echo Diagnostic report created in logs\diagnostics.txt
pause
exit /b 0

:path_failed
echo Could not open the project folder.
goto :failed

:python_missing
echo Python was not found. Run run.bat first, or install Python 3.10 or newer.
goto :failed

:diagnostics_failed
echo Diagnostics did not complete. Review the error above and logs\studio.log.

:failed
pause
exit /b 1
