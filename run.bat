@echo off
chcp 65001 >nul
title DZ PhotoID Studio - صانع صور الهوية
cd /d "%~dp0"
set PORT=8765

echo ============================================================
echo   DZ PhotoID Studio - صانع صور الهوية - الجزائر
echo   صور 3.5 x 4.5 سم - قص ذكي - ازالة خلفية - طباعة 300 DPI
echo ============================================================
echo.

rem ---- 1) ابحث عن بايثون ----
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY ( where python >nul 2>nul && set PY=python )

rem ---- 2) اذا لم يوجد بايثون، حاول تثبيته بـ winget ----
if not defined PY (
  where winget >nul 2>nul
  if not errorlevel 1 (
    echo [تنبيه] بايثون غير مثبت على هذا الجهاز.
    echo سيثبت الآن تلقائيا عبر winget - انتظر حتى ينتهي...
    echo.
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.
    echo [تم] اعد تشغيل هذا الملف run.bat مرة اخرى بعد انتهاء التثبيت.
    pause
    exit /b 0
  )
  echo [تنبيه] بايثون غير مثبت ولا توجد اداة winget.
  echo سيفتح البرنامج مباشرة كملف - اغلب المزايا تعمل،
  echo لكن يفضل تثبيت بايثون من python.org لافضل نتيجة.
  echo.
  start "" "index.html"
  pause
  exit /b 0
)

rem ---- 3) شغل خادم محلي وافتح المتصفح ----
echo [تشغيل] الخادم المحلي يعمل الآن...
echo [تشغيل] افتح المتصفح على: http://localhost:%PORT%
echo.
echo ملاحظة: اول مرة فقط، يحتاج البرنامج انترنت لتحميل نماذج
echo الذكاء الاصطناعي (ازالة الخلفية وكشف الوجه). بعدها يعمل
echo من ذاكرة المتصفح، وكل صورك تبقى على جهازك فقط.
echo.
echo لايقاف البرنامج: اغلق هذه النافذة او اضغط Ctrl+C
echo ============================================================
start "" /min cmd /c "timeout /t 2 >nul && start http://localhost:%PORT%"
%PY% -m http.server %PORT% --bind 127.0.0.1
pause
