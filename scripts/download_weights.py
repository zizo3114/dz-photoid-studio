# -*- coding: utf-8 -*-
"""
تنزيل نموذج إزالة الخلفية (hivision_modnet.onnx) عند الحاجة.
النموذج يأتي مع نسخة ZIP جاهزاً؛ هذه السكربت شبكة أمان لمن حمّل المشروع من GitHub
(الملفات الكبيرة مستثناة من المستودع).
"""
import os
import sys
import urllib.request

URL = ("https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/"
       "download/pretrained-model/hivision_modnet.onnx")
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "hivision", "creator", "weights", "hivision_modnet.onnx")
MIN_SIZE = 20_000_000  # النموذج الحقيقي ≈ 24.7MB


def main():
    if os.path.exists(DEST) and os.path.getsize(DEST) >= MIN_SIZE:
        print("النموذج موجود — لا حاجة للتنزيل.")
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("جارٍ تنزيل نموذج إزالة الخلفية (24.7MB)…")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    tmp = DEST + ".part"
    try:
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
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
        if os.path.getsize(tmp) < MIN_SIZE:
            raise RuntimeError("الملف النازل أصغر من المتوقع — تحقق من الاتصال وحاول مجدداً.")
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
