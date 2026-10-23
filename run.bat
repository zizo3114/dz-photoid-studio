@echo off
chcp 65001 >nul
title DZ PhotoID Studio
cd /d "%~dp0"

echo ============================================================
echo   DZ PhotoID Studio - صانع صور الهوية - الجزائر
echo   صور 3.5 x 4.5 سم - قص ذكي - ازالة خلفية - طباعة 300 DPI
echo ============================================================
echo.

rem ---- 1) ابحث عن بايثون يعمل فعلا (الفحص يستبعد اختصار متجر مايكروسوفت الوهمي) ----
set PY=
where py >nul 2>nul
if errorlevel 1 goto :try_python
py -3 -c "import sys" >nul 2>nul
if errorlevel 1 goto :try_python
set PY=py -3
goto :have_py

:try_python
where python >nul 2>nul
if errorlevel 1 goto :no_py
python -c "import sys" >nul 2>nul
if errorlevel 1 goto :no_py
set PY=python
goto :have_py

:no_py
where winget >nul 2>nul
if errorlevel 1 goto :no_winget
echo [تنبيه] بايثون غير مثبت على هذا الجهاز.
echo سيثبت الآن تلقائيا عبر winget - انتظر حتى ينتهي...
echo.
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
echo.
echo [تم] أعد تشغيل هذا الملف run.bat مرة أخرى بعد انتهاء التثبيت.
pause
exit /b 0

:no_winget
echo [تنبيه] بايثون غير مثبت ولا توجد أداة winget.
echo سيفتح البرنامج مباشرة كملف - أغلب المزايا تعمل،
echo لكن يفضل تثبيت بايثون من python.org لأفضل نتيجة.
echo.
start "" "index.html"
pause
exit /b 0

:have_py
echo [تشغيل] الخادم المحلي يعمل الآن... سيفتح المتصفح تلقائيا.
echo ملاحظة: أول مرة فقط، يحتاج البرنامج إنترنت لتحميل نموذج
echo إزالة الخلفية وكشف الوجه. بعدها يعمل من ذاكرة المتصفح،
echo وكل صورك تبقى على جهازك فقط.
echo.
%PY% serve.py
pause
