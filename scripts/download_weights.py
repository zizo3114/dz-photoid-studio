# -*- coding: utf-8 -*-
"""
تنزيل نموذج إزالة الخلفية (hivision_modnet.onnx) عند الحاجة.
النموذج يأتي مع نسخة ZIP جاهزاً؛ هذه السكربت شبكة أمان لمن حمّل المشروع من GitHub
(الملفات الكبيرة مستثناة من المستودع).
"""
import os
import sys
import urllib.request


def _configure_console_utf8():
    """Prevent Arabic status messages from failing in legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            pass


_configure_console_utf8()

URL = ("https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/"
       "download/pretrained-model/hivision_modnet.onnx")
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "hivision", "creator", "weights", "hivision_modnet.onnx")
MIN_SIZE = 20_000_000  # النموذج الحقيقي ≈ 24.7MB


def _model_ok(path):
    """النموذج صالح فقط إذا كان حجمه كافياً ويُفتح فعلياً بمحرك ONNX.

    فحص الحجم وحده كان يقبل ملفاً مقطوعاً عند 20MB من أصل 24.7MB فيبقى
    التثبيت مسموماً للأبد؛ التحميل الحقيقي هنا يكشف أي قطع أو تلف.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) < MIN_SIZE:
            return False
    except OSError:
        return False
    try:
        import onnxruntime as ort
        ort.InferenceSession(path, providers=[])
        return True
    except ImportError:
        return True  # المحرك غير متاح في هذا المفسر: فحص الحجم هو المتاح
    except Exception:
        return False


def main():
    if _model_ok(DEST):
        print("النموذج موجود — لا حاجة للتنزيل.")
        return 0
    if os.path.exists(DEST):
        print("النموذج الموجود ناقص أو تالف — إعادة تنزيله.")
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("جارٍ تنزيل نموذج إزالة الخلفية (24.7MB)…")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    tmp = DEST + ".part"
    try:
        with urllib.request.urlopen(req, timeout=90) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done * 100 // total}%", end="", flush=True)
        print()
        if total and done != total:
            raise RuntimeError(
                f"انقطع التنزيل ({done} من {total} بايت) — تحقق من الاتصال وحاول مجدداً.")
        if os.path.getsize(tmp) < MIN_SIZE:
            raise RuntimeError("الملف النازل أصغر من المتوقع — تحقق من الاتصال وحاول مجدداً.")
        if not _model_ok(tmp):
            raise RuntimeError("الملف النازل لا يفتحه محرك ONNX (مقطوع أو تالف) — حاول مجدداً.")
        os.replace(tmp, DEST)
        print("تم تنزيل النموذج بنجاح.")
        return 0
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"فشل تنزيل النموذج: {e}")
        print("يمكنك تنزيله يدوياً من:")
        print(URL)
        print(f"وضعه في: {DEST}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
