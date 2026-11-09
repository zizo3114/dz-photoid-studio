@echo off
title DZ PhotoID Studio - Allow phone access (one time only)
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Asking for administrator permission...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
netsh advfirewall firewall delete rule name="DZ PhotoID Studio" >nul 2>&1
netsh advfirewall firewall add rule name="DZ PhotoID Studio" dir=in action=allow protocol=TCP localport=7860-7880 profile=private
if errorlevel 1 goto :rule_failed
echo.
echo DONE - allowed only on Private network profile.
echo Keep Windows network profile set to Private for your trusted Wi-Fi.
echo Your phone (same Wi-Fi) can now open the app, e.g.:
echo http://192.168.1.7:7860
echo.
pause
exit /b 0

:rule_failed
echo.
echo FAILED - the firewall rule was NOT created (see the error above).
echo Run this file again and accept the administrator prompt.
echo.
pause
