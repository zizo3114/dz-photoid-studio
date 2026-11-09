#!/usr/bin/env bash
# صانع صور الهوية — الجزائر
set -Eeuo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3 غير موجود — ثبّته أولاً."; exit 1; }
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' || {
  echo "يتطلب البرنامج Python 3.10 أو أحدث."; exit 1;
}

[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY=.venv/bin/python
HASH_FILE=.venv/.reqhash
[ -f requirements.txt ] || { echo "ملف requirements.txt مفقود من مجلد المشروع."; exit 1; }
REQ_HASH=$("$VPY" -c 'import hashlib; print(hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest())')
OLD_HASH=$(cat "$HASH_FILE" 2>/dev/null || true)
if [ "$REQ_HASH" != "$OLD_HASH" ] || ! "$VPY" -c 'import cv2, numpy, onnxruntime, gradio, PIL, requests, tqdm, mtcnnruntime' >/dev/null 2>&1; then
  echo "تثبيت/إصلاح المكتبات حسب requirements.txt..."
  "$VPY" -m pip install --disable-pip-version-check --prefer-binary -r requirements.txt
  printf '%s' "$REQ_HASH" > "$HASH_FILE"
fi
"$VPY" -c 'import cv2, numpy, onnxruntime, gradio, PIL, requests, tqdm, mtcnnruntime; print("المكتبات جاهزة.")'

if [ "${1:-}" = "--check" ]; then
  echo "اكتمل فحص بيئة التشغيل بنجاح."
  exit 0
fi

"$VPY" scripts/download_weights.py || { echo "تعذّر تجهيز نموذج إزالة الخلفية — تحقق من الاتصال وحاول مجدداً."; exit 1; }
exec "$VPY" app.py
