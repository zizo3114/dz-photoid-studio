@echo off
title DZ PhotoID Studio - Allow phone access (one time only)
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Asking for administrator permission...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
netsh advfirewall firewall delete rule name="DZ PhotoID Studio" >nul 2>&1
netsh advfirewall firewall add rule name="DZ PhotoID Studio" dir=in action=allow protocol=TCP localport=7860-7880 profile=any
echo.
echo DONE - your phone (same Wi-Fi) can now open the app, e.g.:
echo http://192.168.1.7:7860
echo.
pause
