# -*- coding: utf-8 -*-
"""Local diagnostics for DZ PhotoID Studio. Does not upload or modify user photos."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


def _configure_console_utf8() -> None:
    """Prevent Arabic diagnostics from failing in legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            pass


_configure_console_utf8()

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "outputs"
MODEL = ROOT / "hivision" / "creator" / "weights" / "hivision_modnet.onnx"
DB = ROOT / "studio.db"
SETTINGS = ROOT / "settings.json"

checks: list[tuple[str, str, str]] = []
def add(name: str, status: str, detail: str = "") -> None:
    checks.append((name, status, detail))

IN_PROJECT_VENV = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
add("Python", "OK" if sys.version_info >= (3, 10) else "FAIL",
    f"{platform.python_version()} ({platform.architecture()[0]})")
add("بيئة التشغيل", "OK" if IN_PROJECT_VENV else "WARN",
    "بيئة المشروع .venv مفعّلة" if IN_PROJECT_VENV else
    "هذا الفحص يعمل ببايثون النظام لا ببيئة المشروع .venv؛ شغّل run.bat مرة أولاً أو أعد تشغيل التشخيص بعد إنشاء البيئة.")

modules = {
    "gradio": "gradio", "numpy": "numpy", "OpenCV": "cv2", "Pillow": "PIL",
    "ONNX Runtime": "onnxruntime", "MTCNN Runtime": "mtcnnruntime", "requests": "requests",
}
for label, module in modules.items():
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "loaded")
        if module == "gradio":
            status = "OK" if version == "6.31.0" else "WARN"
            detail = f"{version} — النسخة المثبتة الموصى بها 6.31.0"
        else:
            status, detail = "OK", str(version)
        add(label, status, detail)
    except Exception as exc:
        # Before the project virtual environment exists, missing packages in
        # the machine-wide Python are not evidence that the project install is broken.
        add(label, "FAIL" if IN_PROJECT_VENV else "WARN",
            f"{type(exc).__name__}: {exc}" if IN_PROJECT_VENV else
            f"غير موجود في بايثون النظام؛ سيُثبّت داخل .venv عند تشغيل run.bat ({type(exc).__name__})")

if not MODEL.is_file():
    add("نموذج إزالة الخلفية", "FAIL", "الملف غير موجود — شغّل run.bat لتنزيله")
elif MODEL.stat().st_size < 20_000_000:
    add("نموذج إزالة الخلفية", "FAIL",
        f"الملف ناقص ({MODEL.stat().st_size / (1024*1024):.1f} MB من 24.7) — احذفه وشغّل run.bat")
else:
    try:
        import onnxruntime as _ort
        _ort.InferenceSession(str(MODEL), providers=[])
        add("نموذج إزالة الخلفية", "OK", f"{MODEL.stat().st_size / (1024*1024):.1f} MB — يُفتح سليماً")
    except ImportError:
        add("نموذج إزالة الخلفية", "OK", f"{MODEL.stat().st_size / (1024*1024):.1f} MB (فحص الحجم فقط)")
    except Exception as exc:
        add("نموذج إزالة الخلفية", "FAIL",
            f"الملف تالف ولا يفتحه ONNX ({type(exc).__name__}) — احذفه وشغّل run.bat لإعادة تنزيله")

try:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="diag_", dir=OUT_DIR, delete=True) as f:
        f.write(b"write-test")
        f.flush()
    add("صلاحية مجلد outputs", "OK", "يمكن إنشاء ملف مؤقت وحذفه")
except Exception as exc:
    add("صلاحية مجلد outputs", "FAIL", f"{type(exc).__name__}: {exc}")

if SETTINGS.exists():
    try:
        value = json.loads(SETTINGS.read_text(encoding="utf-8"))
        add("settings.json", "OK" if isinstance(value, dict) else "WARN",
            "JSON سليم" if isinstance(value, dict) else "المحتوى ليس كائناً JSON")
    except Exception as exc:
        add("settings.json", "FAIL", f"{type(exc).__name__}: {exc}")
else:
    add("settings.json", "OK", "سيُنشأ تلقائياً عند أول حفظ")

if DB.exists():
    try:
        con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=5)
        result = con.execute("PRAGMA quick_check").fetchone()[0]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        add("studio.db", "OK" if result == "ok" else "FAIL",
            f"quick_check={result}; tables={', '.join(sorted(tables)) or 'empty'}")
    except Exception as exc:
        add("studio.db", "FAIL", f"{type(exc).__name__}: {exc}")
else:
    add("studio.db", "OK", "لا يوجد سجل سابق بعد؛ سيُنشأ عند التشغيل")

if IN_PROJECT_VENV:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT,
                           capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
        detail = (r.stdout or r.stderr or "No broken requirements found.").strip()
        add("توافق المكتبات (pip check)", "OK" if r.returncode == 0 else "FAIL", detail[:1000])
    except Exception as exc:
        add("توافق المكتبات (pip check)", "WARN", f"تعذّر الفحص: {type(exc).__name__}: {exc}")
else:
    add("توافق المكتبات (pip check)", "WARN",
        "تم تخطيه لأن الفحص يعمل خارج .venv؛ أعد تشغيل diagnose.bat بعد تشغيل run.bat لأول مرة.")

lines = ["DZ PHOTOID STUDIO — LOCAL DIAGNOSTIC REPORT", f"Date: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}", f"Root: {ROOT}", ""]
for name, status, detail in checks:
    lines.append(f"[{status}] {name}: {detail}")
report = "\n".join(lines) + "\n"
print(report)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / "diagnostics.txt").write_text(report, encoding="utf-8")
failed = [x for x in checks if x[1] == "FAIL"]
print(f"التقرير محفوظ في: {LOG_DIR / 'diagnostics.txt'}")
print(f"عدد الفحوص: {len(checks)} | إخفاقات: {len(failed)}")
sys.exit(1 if failed else 0)
