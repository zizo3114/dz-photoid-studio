#!/usr/bin/env bash
# صانع صور الهوية — الجزائر (Linux/macOS)
cd "$(dirname "$0")" || exit 1
PY=python3
command -v $PY >/dev/null 2>&1 || { echo "Python 3 غير موجود — ثبّته أولاً."; exit 1; }
[ -d .venv ] || $PY -m venv .venv
VPY=.venv/bin/python
if [ ! -f .venv/.installed ]; then
  $VPY -m pip install --upgrade pip
  $VPY -m pip install -r requirements.txt || { echo "فشل التثبيت."; exit 1; }
  touch .venv/.installed
fi
$VPY scripts/download_weights.py
$VPY app.py
