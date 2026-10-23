@echo off
chcp 65001 >nul
title صانع صور الهوية — الجزائر
cd /d %~dp0

REM ---------- البحث عن بايثون حقيقي يعمل (تخطّي نسخة متجر مايكروسوفت الوهمية) ----------
set PY=
for %%P in (py python python3) do (
  if not defined PY (
    %%P -c "import sys; sys.exit(0)" >nul 2>&1 && set PY=%%P
  )
)
if not defined PY (
  echo لم يتم العثور على بايثون. جارٍ محاولة تثبيته تلقائيا...
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  echo أعد تشغيل هذا الملف بعد انتهاء التثبيت.
  pause
  exit /b 1
)
for /f "tokens=*" %%V in ('%PY% --version 2^>^&1') do set PYVER=%%V
echo تم العثور على بايثون: %PY% - %PYVER%

REM ---------- إنشاء البيئة الافتراضية أول مرة ----------
if not exist .venv\Scripts\python.exe (
  echo إنشاء البيئة الافتراضية...
  %PY% -m venv .venv
)
set VPY=.venv\Scripts\python.exe

REM ---------- تثبيت المكتبات أول مرة ----------
if not exist .venv\.installed (
  echo تثبيت المكتبات - قد يأخذ بضع دقائق أول مرة...
  %VPY% -m pip install --upgrade pip
  %VPY% -m pip install --prefer-binary -r requirements.txt
  if errorlevel 1 (
    echo.
    echo فشل تثبيت المكتبات. أغلق هذا النافذة وشغّل الملف مرة أخرى -
    echo فكثير من الأعطال المؤقتة تُحل في المحاولة الثانية.
    echo وإن تكرر نفس الخطأ، صوّر الشاشة كاملة وأرسلها.
    pause
    exit /b 1
  )
  echo ok> .venv\.installed
)

REM ---------- التأكد من وجود نموذج إزالة الخلفية ----------
%VPY% scripts\download_weights.py

REM ---------- تشغيل البرنامج ----------
echo جارٍ تشغيل البرنامج... سيفتح في متصفحك تلقائيا.
%VPY% app.py
pause
