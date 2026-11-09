@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Keep this launcher ASCII-only. cmd.exe is unreliable with UTF-8 batch
rem source on some Windows installations, especially inside parenthesized blocks.
chcp 65001 >nul
set "PYTHONUTF8=1"
title Algeria Photo ID Studio

cd /d "%~dp0"
if errorlevel 1 goto :path_failed

set "CHECK_ONLY="
if /I "%~1"=="--check" set "CHECK_ONLY=1"

rem Locate a usable Python 3.10+ installation. Do not use the Windows Store alias.
set "PY="
py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py"
if defined PY goto :python_found

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :python_found

goto :python_missing

:python_found
echo Using Python:
"%PY%" --version

if exist ".venv\Scripts\python.exe" goto :venv_ready

:create_venv
echo Creating the project virtual environment...
"%PY%" -m venv ".venv"
if errorlevel 1 goto :venv_failed

:venv_ready
set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "%VPY%" goto :venv_failed
if not exist "requirements.txt" goto :requirements_missing

rem Skip pip only when both requirements and real runtime imports are healthy.
rem This keeps normal launches fast while repairing a missing cv2 or stale venv.
"%VPY%" -c "import hashlib,pathlib,sys; r=pathlib.Path('requirements.txt'); h=pathlib.Path('.venv/.reqhash'); sys.exit(0 if h.is_file() and h.read_text().strip()==hashlib.sha256(r.read_bytes()).hexdigest() else 1)" >nul 2>&1
if errorlevel 1 goto :install_packages
"%VPY%" -c "import cv2,numpy,onnxruntime,gradio,PIL,requests,tqdm,mtcnnruntime" >nul 2>&1
if errorlevel 1 goto :install_packages
goto :packages_ready

:install_packages
echo Installing or repairing required packages...
"%VPY%" -m pip install --disable-pip-version-check --prefer-binary -r "requirements.txt"
if errorlevel 1 goto :package_failed
"%VPY%" -c "import hashlib,pathlib; r=pathlib.Path('requirements.txt'); pathlib.Path('.venv/.reqhash').write_text(hashlib.sha256(r.read_bytes()).hexdigest())"
if errorlevel 1 goto :package_failed

:packages_ready

echo Verifying runtime imports...
"%VPY%" -c "import cv2, numpy, onnxruntime, gradio, PIL, requests, tqdm, mtcnnruntime; print('Runtime dependencies are ready.')"
if errorlevel 1 goto :import_failed

if defined CHECK_ONLY goto :success

echo Checking the background-removal model...
"%VPY%" "scripts\download_weights.py"
if errorlevel 1 goto :weights_failed

echo Starting the application. It will open in your browser.
"%VPY%" "app.py"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" goto :app_failed
goto :success

:success
if defined CHECK_ONLY echo Runtime check completed successfully.
if not defined CHECK_ONLY echo Application stopped.
if defined CHECK_ONLY exit /b 0
pause
exit /b 0

:path_failed
echo Could not open the project folder.
goto :failed

:python_missing
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ and run this file again.
goto :failed

:venv_failed
echo The virtual environment could not be created or opened.
goto :failed

:requirements_missing
echo requirements.txt is missing from the project folder.
goto :failed

:package_failed
if defined REPAIR_TRIED goto :package_failed_final
set "REPAIR_TRIED=1"
echo Package installation failed. Rebuilding the virtual environment once and retrying...
rmdir /s /q ".venv" >nul 2>&1
goto :create_venv

:package_failed_final
echo Package installation failed again after an automatic environment rebuild.
echo Check your internet connection and run this file again.
goto :failed

:import_failed
echo A required package is still unavailable after installation.
echo Run diagnose.bat and send logs\diagnostics.txt if this persists.
goto :failed

:weights_failed
echo The background-removal model could not be prepared.
echo Check your internet connection and run this file again.
goto :failed

:app_failed
echo The application exited with code %APP_EXIT%.
goto :failed

:failed
pause
exit /b 1
