# -*- coding: utf-8 -*-
"""
تنزيل النماذج عند الحاجة:
- نموذج إزالة الخلفية (hivision_modnet.onnx) — إلزامي لعمل البرنامج.
- نموذج تجميل الوجوه بالذكاء الاصطناعي (gfpgan_1.4.onnx) — اختياري:
  إن تعذّر تنزيله يعمل البرنامج عادياً ويرجع التجميل لمحركه الكلاسيكي.
النماذج تأتي مع نسخة ZIP جاهزة ما أمكن؛ هذه السكربت شبكة أمان لمن حمّل
المشروع من GitHub (الملفات الكبيرة مستثناة من المستودع).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "hivision", "creator", "weights")

MODELS = [
    {
        "label": "نموذج إزالة الخلفية",
        "size_label": "24.7MB",
        "url": ("https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/"
                "download/pretrained-model/hivision_modnet.onnx"),
        "dest": os.path.join(WEIGHTS, "hivision_modnet.onnx"),
        "min_size": 20_000_000,  # النموذج الحقيقي ≈ 24.7MB
        "required": True,
    },
    {
        "label": "نموذج تجميل الوجوه بالذكاء الاصطناعي (GFPGAN)",
        "size_label": "340MB",
        "url": ("https://github.com/facefusion/facefusion-assets/releases/"
                "download/models-3.0.0/gfpgan_1.4.onnx"),
        "dest": os.path.join(WEIGHTS, "gfpgan_1.4.onnx"),
        "min_size": 330_000_000,  # النموذج الحقيقي ≈ 340MB
        "required": False,
        "fallback_note": "تنبيه: بدون هذا النموذج يعمل التجميل بالمحرك الكلاسيكي فقط — "
                         "أعد تشغيل البرنامج وهو متصل بالإنترنت لإعادة المحاولة.",
    },
    {
        "label": "نموذج قصّ الخلفية الاحترافي (ISNet)",
        "size_label": "179MB",
        "url": ("https://github.com/danielgatis/rembg/releases/"
                "download/v0.0.0/isnet-general-use.onnx"),
        "dest": os.path.join(WEIGHTS, "isnet-general-use.onnx"),
        "min_size": 170_000_000,  # النموذج الحقيقي ≈ 179MB
        "required": False,
        "fallback_note": "تنبيه: بدون هذا النموذج يعمل قصّ الخلفية بالمحرك الأساسي — "
                         "أعد تشغيل البرنامج وهو متصل بالإنترنت لإعادة المحاولة.",
    },
]


def _model_ok(path, min_size):
    """النموذج صالح فقط إذا كان حجمه كافياً ويُفتح فعلياً بمحرك ONNX.

    فحص الحجم وحده كان يقبل ملفاً مقطوعاً فيبقى التثبيت مسموماً للأبد؛
    التحميل الحقيقي هنا يكشف أي قطع أو تلف.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) < min_size:
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


def _download(model):
    """تنزيل نموذج واحد مع التحقق الكامل. يعيد True عند النجاح."""
    dest = model["dest"]
    if _model_ok(dest, model["min_size"]):
        print(f"{model['label']} موجود — لا حاجة للتنزيل.")
        return True
    if os.path.exists(dest):
        print(f"{model['label']} الموجود ناقص أو تالف — إعادة تنزيله.")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"جارٍ تنزيل {model['label']} ({model['size_label']})…")
    req = urllib.request.Request(model["url"], headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
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
        if os.path.getsize(tmp) < model["min_size"]:
            raise RuntimeError("الملف النازل أصغر من المتوقع — تحقق من الاتصال وحاول مجدداً.")
        if not _model_ok(tmp, model["min_size"]):
            raise RuntimeError("الملف النازل لا يفتحه محرك ONNX (مقطوع أو تالف) — حاول مجدداً.")
        os.replace(tmp, dest)
        print(f"تم تنزيل {model['label']} بنجاح.")
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"فشل تنزيل {model['label']}: {e}")
        print("يمكنك تنزيله يدوياً من:")
        print(model["url"])
        print(f"وضعه في: {dest}")
        return False


def main():
    ok = True
    for model in MODELS:
        if _download(model):
            continue
        if model["required"]:
            ok = False
        elif model.get("fallback_note"):
            print(model["fallback_note"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
