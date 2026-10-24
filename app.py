# -*- coding: utf-8 -*-
"""
صانع صور الهوية — الجزائر (نسخة سطح المكتب)

واجهة عربية مبنية على محرك مشروع HivisionIDPhotos مفتوح المصدر
(https://github.com/Zeyi-Lin/HivisionIDPhotos — رخصة Apache-2.0).
المحرك: كشف وجه MTCNN بعتبات صارمة + قاعدة «وجه واحد فقط»،
وقصّ ذكي بنسب رأس قياسية، وإزالة خلفية بنموذج MODNet المهيّأ لصور الهوية.

مسار جودة عالٍ (HQ): المحرك يعمل داخلياً على نسخة مصغّرة (2000px)؛
عند توفر صورة أصلية أكبر نعيد بناء القص النهائي من بكسلات الأصلية كاملة
الدقة بنفس هندسة المحرك، مع تحسين حدة خفيف قياسي — فتبقى التفاصيل.

وضع يدوي كامل: تدوير بالدرجة، تقريب وإزاحة، وأدلة قص احترافية على
المعاينة فقط (بيضوي رأس مزدوج بقياسات المليمتر، شريط منطقة العينين،
خطوط عرض الرأس وخط المنتصف) — لا تظهر الأدلة في أي ملف نهائي.
كل المعالجة على جهاز المستخدم، والإعدادات محفوظة في settings.json.
"""
import os
import shutil

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import gradio as gr

from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from hivision.creator.context import Context, Params
from hivision.creator.photo_adjuster import adjust_photo, IDphotos_cut, move
from hivision.creator.utils import get_box, detect_distance
from hivision.error import FaceError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
DPI = 300
HD_FACTOR = 1.4526      # نسبة النسخة عالية الدقة إلى القياسية
HQ_MIN_EDGE = 2100      # فوق هذا الحد نفعّل مسار الجودة العالية
HQ_MAX_EDGE = 6000      # سقف أبعاد المسار عالي الجودة (توازن جودة/ذاكرة)

# ----------------------------- المقاسات والألوان والأوراق -----------------------------

SIZES = {
    "جزائرية رسمية 3.5 × 4.5 سم": (531, 413),   # (الارتفاع، العرض) بالبكسل @300dpi
    "صغيرة 2.5 × 3.5 سم": (413, 295),
    "كبيرة 3.5 × 4.9 سم": (579, 413),
    "جواز سفر 5 × 5 سم": (591, 591),
    "مقاس مخصص…": None,
}

COLORS = {
    "أبيض": (255, 255, 255),
    "أزرق": (67, 142, 219),
    "أزرق فاتح": (168, 197, 230),
    "رمادي فاتح": (240, 240, 240),
    "أحمر": (215, 30, 25),
    "لون مخصص…": None,
}

SHEETS = {
    "بدون ورقة طباعة": None,
    "8 صور — ورقة 10 × 15 سم": ((1800, 1200), 4, 2),   # ورق المختبر 6×4 بوصة بالضبط عند 300 DPI (152.4×101.6مم)
    "12 صورة — ورقة 15 × 20 سم": ((2400, 1800), 4, 3),  # ورق المختبر الحقيقي 8×6 بوصة (203×152مم) عند 300 DPI
}

NOTIF_SLOTS = 3   # إشعارات ظاهرة في نفس الوقت كحد أقصى
LIST_ROWS = 6     # صور ظاهرة في قائمة الصور كحد أقصى

MODES = ["تلقائي — كشف الوجه والقص الذكي", "يدوي — أنا أضبط القص بنفسي"]

# معاملات القص القياسية (قيم المحرك المُختبَرة على 8 بورتريهات: ذقن ≈ 71.6%،
# عينان ≈ 44.6%، قمة ≈ 10.9% — مطابقة لقوالب صور الهوية المعتمدة)
CROP_PARAMS = dict(head_measure_ratio=0.20, head_height_ratio=0.45,
                   head_top_range=(0.12, 0.10))
# نسب الأدلة القياسية من ارتفاع الصورة (توافق القالب المرجعي 35×45مم):
# الذقن عند 70% من الأعلى (31.5مم)، العينان عند 44%، القمة المستهدفة 8%
GUIDE_CROWN, GUIDE_CHIN, GUIDE_EYE = 0.08, 0.70, 0.44

DEFAULT_SETTINGS = {
    "mode": MODES[0],
    "size": "جزائرية رسمية 3.5 × 4.5 سم",
    "custom_w": 35, "custom_h": 45,
    "color": "أبيض", "custom_color": "#ffffff",
    "sheet": "8 صور — ورقة 10 × 15 سم",
    "align": True, "kb": 0,
    "angle": 0.0, "zoom": 1.0, "pan_x": 0, "pan_y": 0, "guides": True,
    "brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "sharpness": 1.0,
    "shadows": 0, "highlights": 0, "temperature": 0,
    "inbox": "", "watch": False, "automake": False, "printer": "",
    "ui_cut": True, "ui_color": False, "ui_paper": False,
    "ui_studio": False, "ui_list": True,
}


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        merged = dict(DEFAULT_SETTINGS)
        for k, v in saved.items():
            if k in merged:
                merged[k] = v
        return merged
    except Exception:
        return dict(DEFAULT_SETTINGS)


_last_saved_json = None


def save_settings_dict(values):
    """حفظ الإعدادات — مع تخطّي الكتابة إن لم يتغيّر شيء (منعاً لأي تكرار زائد)."""
    global _last_saved_json
    try:
        payload = json.dumps(values, ensure_ascii=False, sort_keys=True)
        if payload == _last_saved_json:
            return
        _last_saved_json = payload
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


SETTINGS = load_settings()
_creator = None


def get_creator() -> IDCreator:
    """محرك واحد يُعاد استعماله (تحميل النماذج مرة واحدة فقط)."""
    global _creator
    if _creator is None:
        _creator = IDCreator()
        choose_handler(_creator, "hivision_modnet", "mtcnn")
        # تعطيل إضافة التجميل: لا حاجة لها في صور الهوية، وتحتاج ملفاً ثنائياً
        # (جدول ألوان LUT) — تجاوزها يجعل البرنامج يعمل بنفس الشكل في كل النسخ.
        _creator.beauty_handler = lambda ctx: None
    return _creator


def mm_to_px(mm: float) -> int:
    return int(round(float(mm) / 25.4 * DPI))


def hex_to_rgb(value: str):
    value = (value or "#ffffff").lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def resolve_size(size_label, custom_w_mm, custom_h_mm):
    size = SIZES.get(size_label)
    if size is not None:
        return size
    try:
        w_px, h_px = mm_to_px(custom_w_mm), mm_to_px(custom_h_mm)
    except Exception:
        raise gr.Error("المقاس المخصص غير صحيح — أدخل العرض والارتفاع بالمليمتر.")
    if not (100 <= w_px <= 3000 and 100 <= h_px <= 3000):
        raise gr.Error("المقاس المخصص خارج الحدود المعقولة (بين 8 و250 مم تقريباً).")
    return (h_px, w_px)


def resolve_color(color_label, custom_color):
    rgb = COLORS.get(color_label)
    if rgb is None:
        rgb = hex_to_rgb(custom_color)
    return rgb


# ----------------------------- معالجة الصور -----------------------------

def normalize_input(img) -> np.ndarray:
    """
    تحويل الصورة المرفوعة إلى BGR بثلاث قنوات — وهو ما يتوقعه المحرك.
    عند استلام مسار ملف (وهو الحال من الواجهة): فك ترميز يحفظ قيم البكسل
    حرفياً كما تفعل برامج صور الهوية الاحترافية (لا تحويلات فضاء ألوان —
    الألوان تبقى كما في الملف تماماً)، مع تطبيق اتجاه EXIF. الصور الشفافة
    تُدمج على أبيض أولاً.
    """
    if img is None:
        return None
    if isinstance(img, (str, os.PathLike)):
        pil = Image.open(img)
        pil = ImageOps.exif_transpose(pil)
        if pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info):
            rgba = pil.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            pil = pil.convert("RGB")
        img = np.array(pil)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        rgba = Image.fromarray(img).convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = np.array(Image.alpha_composite(white, rgba).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def composite_bg(bgra: np.ndarray, rgb) -> Image.Image:
    """تركيب الصورة المقصوصة (BGRA) على لون خلفية موحّد."""
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    fg = Image.fromarray(rgba, mode="RGBA")
    canvas = Image.new("RGBA", fg.size, tuple(rgb) + (255,))
    return Image.alpha_composite(canvas, fg).convert("RGB")


def finalize_from_crop(crop_bgra: np.ndarray, target_w: int, target_h: int, rgb) -> Image.Image:
    """تصغير قصّ BGRA عالي الدقة إلى المقاس الهدف بـ LANCZOS وتركيبه على الخلفية."""
    rgba = cv2.cvtColor(crop_bgra, cv2.COLOR_BGRA2RGBA)
    fg = Image.fromarray(rgba, mode="RGBA")
    if fg.size != (target_w, target_h):
        fg = fg.resize((target_w, target_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), tuple(rgb) + (255,))
    return Image.alpha_composite(canvas, fg).convert("RGB")


def mild_sharpen(img: Image.Image) -> Image.Image:
    """تحسين حدة خفيف قياسي (قناع إزالة ضباب) — يعيد للتفاصيل حضورها بعد التصغير."""
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=40, threshold=3))


def apply_color_adjustments(img: Image.Image, brightness=1.0, contrast=1.0,
                            saturation=1.0, sharpness=1.0, shadows=0,
                            highlights=0, temperature=0) -> Image.Image:
    """ضبط الألوان: حرارة اللون والظلال والإبرازات (numpy) ثم السطوع والتباين والتشبع والحدة (PIL)."""
    if (brightness == 1.0 and contrast == 1.0 and saturation == 1.0
            and sharpness == 1.0 and not shadows and not highlights and not temperature):
        return img
    arr = np.asarray(img).astype(np.float32) / 255.0
    if temperature:
        f = float(temperature) / 100.0 * 0.25
        arr[..., 0] = np.clip(arr[..., 0] * (1 + f), 0, 1)   # الأحمر — دفء
        arr[..., 2] = np.clip(arr[..., 2] * (1 - f), 0, 1)   # الأزرق — برودة
    if shadows or highlights:
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]
        if shadows:
            arr = arr + (float(shadows) / 100.0) * 0.30 * (1.0 - lum) ** 2
        if highlights:
            arr = arr + (float(highlights) / 100.0) * 0.30 * lum ** 2
        arr = np.clip(arr, 0, 1)
    out = Image.fromarray((arr * 255).astype(np.uint8))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(brightness))
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(float(contrast))
    if saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(float(saturation))
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(float(sharpness))
    return out


def build_virtual_bgra(orig_bgr: np.ndarray, capped_matting: np.ndarray):
    """
    بناء نسخة افتراضية كاملة الدقة من صورة الإزالة: بكسلات الأصلية الحقيقية
    + قناة الشفافية من المحرك مكبّرة. تُستخدم في مسار الجودة العالية.
    تعيد None إن كانت الأصلية صغيرة أصلاً (لا فائدة منها).
    """
    oh, ow = orig_bgr.shape[:2]
    if max(ow, oh) <= HQ_MIN_EDGE:
        return None
    f = min(1.0, HQ_MAX_EDGE / max(ow, oh))
    vw, vh = int(ow * f), int(oh * f)
    orig_v = (cv2.resize(orig_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
              if f < 1.0 else orig_bgr)
    alpha_v = cv2.resize(capped_matting[:, :, 3], (vw, vh),
                         interpolation=cv2.INTER_LINEAR)
    return cv2.merge((orig_v[:, :, 0], orig_v[:, :, 1], orig_v[:, :, 2], alpha_v))


def hq_rerender(orig_bgr: np.ndarray, creator: IDCreator, size_hw, face_alignment: bool):
    """
    إعادة بناء قصّ المحرك من الأصلية كاملة الدقة:
    نأخذ مستطيل الوجه الذي كشفه المحرك على النسخة المصغّرة، نطابقه على
    النسخة الافتراضية كاملة الدقة، ونعيد تشغيل حسابات القص نفسها.
    تعيد مصفوفة BGRA للقص عالي الدقة أو None عند عدم الإمكانية.
    """
    try:
        ctx = creator.ctx
        if ctx is None or ctx.face is None or ctx.face.get("rectangle") is None:
            return None
        roll = ctx.face.get("roll_angle") or 0
        base_bgr = orig_bgr
        if face_alignment and abs(roll) > 2:
            # المحرك دوّر صورته داخلياً لتصحيح الميلان: ندوّر الأصلية كاملة الدقة
            # بنفس الزاوية والأداة حتى تبقى الهندسة متطابقة ويعمل مسار HQ
            # (سابقاً كانت الصور المائلة الرأس تسقط إلى مسار مصغّر ضبابي)
            from hivision.creator.rotation_adjust import rotate_bound_4channels
            _alpha = np.full(orig_bgr.shape[:2], 255, dtype=np.uint8)
            base_bgr = rotate_bound_4channels(orig_bgr, _alpha, -1 * roll)[0]
        virt = build_virtual_bgra(base_bgr, ctx.matting_image)
        if virt is None:
            return None
        ch, cw = ctx.origin_image.shape[:2]
        k = virt.shape[1] / cw
        rect = ctx.face["rectangle"]
        ctx2 = Context(Params(size=size_hw))
        ctx2.matting_image = virt
        ctx2.face = {"rectangle": [rect[0] * k, rect[1] * k, rect[2] * k, rect[3] * k],
                     "roll_angle": roll}
        hd_a, _std_a, _clothing, _typo = adjust_photo(ctx2)
        return hd_a
    except Exception:
        return None


def render_manual(matting_bgra: np.ndarray, angle, zoom, pan_x, pan_y,
                  target_w: int, target_h: int, rgb) -> Image.Image:
    """
    القص اليدوي: تدوير بالدرجة حول المركز، ثم تقريب (تغطية للإطار)،
    ثم إزاحة، ثم قص نافذة الإطار وتركيبها على لون الخلفية.
    """
    rgba = cv2.cvtColor(matting_bgra, cv2.COLOR_BGRA2RGBA)
    img = Image.fromarray(rgba, mode="RGBA")
    if angle:
        img = img.rotate(float(angle), resample=Image.BICUBIC,
                         expand=False, fillcolor=(0, 0, 0, 0))
    base = max(target_w / img.width, target_h / img.height)
    scale = base * max(0.05, float(zoom))
    new_w, new_h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    excess_x = max(0, new_w - target_w)
    excess_y = max(0, new_h - target_h)
    fx = min(1.0, max(0.0, 0.5 + float(pan_x) / 200.0))
    fy = min(1.0, max(0.0, 0.5 + float(pan_y) / 200.0))
    left, top = int(excess_x * fx), int(excess_y * fy)
    crop = img.crop((left, top, left + target_w, top + target_h))
    if crop.size != (target_w, target_h):  # حماية عند الحواف بعد التدوير
        padded = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        padded.paste(crop, (0, 0))
        crop = padded
    canvas = Image.new("RGBA", (target_w, target_h), tuple(rgb) + (255,))
    return Image.alpha_composite(canvas, crop).convert("RGB")


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _vlabel(base: Image.Image, text: str, xy, fill, size: int = 13):
    """رسم نص عمودي (مدوّر 90°) على الصورة — بعرض يُقاس حسب النص."""
    f = _font(size)
    try:
        tw = int(f.getlength(text)) + 8
    except Exception:
        tw = 8 + len(text) * (size // 2 + 2)
    tmp = Image.new("RGBA", (tw, size + 10), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.text((3, 3), text, font=f, fill=fill)
    tmp = tmp.rotate(90, expand=True)
    base.paste(tmp, xy, tmp)


def draw_guides(img: Image.Image) -> Image.Image:
    """
    أدلة قص احترافية على المعاينة فقط — بهوامش قياس خارجية كما في قوالب
    برامج صور الهوية المعتمدة: أسهم أبعاد خضراء حول الصورة (الارتفاع الكلي،
    العرض، موضع الذقن من الأعلى مع مجال التسامح، وبعد العينين عن الأسفل)،
    وعلى الصورة نفسها: منطقة قمة الرأس، بيضوي رأس خارجي/داخلي، شريط العينين،
    خط المنتصف. النسب القياسية: قمة 8%، ذقن 70%، عينان 44% من الارتفاع —
    وهي ما ينتجه القص التلقائي فعلياً (قياسات مُتحققة على بورتريهات حقيقية).
    """
    photo = img.copy().convert("RGB")
    w, h = photo.size
    total_h_mm = h / DPI * 25.4
    total_w_mm = w / DPI * 25.4
    f_big = max(12, int(w * 0.030))
    f_small = max(10, int(w * 0.024))

    mx_l = int(w * 0.16)          # هامش يسار (سهم الارتفاع)
    mx_r = int(w * 0.46)          # هامش يمين (سهما الذقن والعينين + التسميات)
    mx_t = int(h * 0.04)
    mx_b = int(h * 0.12)          # هامش أسفل (سهم العرض)
    W2, H2 = w + mx_l + mx_r, h + mx_t + mx_b
    canvas = Image.new("RGB", (W2, H2), (246, 246, 246))
    canvas.paste(photo, (mx_l, mx_t))
    d = ImageDraw.Draw(canvas, "RGBA")
    green = (26, 158, 75)
    red = (225, 38, 38)
    blue = (52, 120, 220)
    yellow = (245, 197, 0)

    crown_y = mx_t + h * GUIDE_CROWN
    chin_y = mx_t + h * GUIDE_CHIN
    eye_y = mx_t + h * GUIDE_EYE
    ox0, ox1 = mx_l + w * 0.23, mx_l + w * 0.77     # البيضوي الخارجي
    ix0, ix1 = mx_l + w * 0.27, mx_l + w * 0.73     # البيضوي الداخلي
    crown_in = mx_t + h * 0.13

    # --- على الصورة: المناطق والبيضويات والخطوط ---
    d.rectangle([mx_l, mx_t, mx_l + w, crown_y], fill=(225, 38, 38, 42))
    for hx in range(mx_l - int(h * 0.08), mx_l + w, 13):
        d.line([(hx, crown_y), (hx + (crown_y - mx_t), mx_t)], fill=(225, 38, 38, 66), width=1)
    d.rectangle([mx_l, mx_t + h * (GUIDE_EYE - 0.04), mx_l + w, mx_t + h * (GUIDE_EYE + 0.04)],
                fill=(90, 160, 235, 40))
    for x in (ox0, ox1):
        d.line([(x, mx_t), (x, mx_t + h)], fill=blue + (170,), width=1)
    d.ellipse([ox0, crown_y, ox1, chin_y], outline=red + (255,), width=3)
    steps = 90
    for i in range(steps):
        if i % 2 == 0:
            d.arc([ix0, crown_in, ix1, chin_y], start=i / steps * 360,
                  end=(i + 1) / steps * 360, fill=red + (225,), width=2)
    d.line([(mx_l + w / 2, mx_t), (mx_l + w / 2, mx_t + h)], fill=yellow + (255,), width=2)
    d.line([(mx_l, eye_y), (mx_l + w, eye_y)], fill=yellow + (170,), width=1)
    d.line([(mx_l, chin_y), (mx_l + w, chin_y)], fill=green + (200,), width=1)
    cx = mx_l + w / 2
    d.polygon([(cx - 7, crown_y - 9), (cx + 7, crown_y - 9), (cx, crown_y)], fill=yellow + (255,))
    d.rectangle([mx_l, mx_t, mx_l + w, mx_t + h], outline=(150, 150, 150), width=1)

    # --- أسهم الأبعاد في الهوامش (خضراء، كالقوالب المعتمدة) ---
    def varrow(x, y0, y1):
        d.line([(x, y0), (x, y1)], fill=green, width=2)
        d.polygon([(x - 5, y0 + 8), (x + 5, y0 + 8), (x, y0)], fill=green)
        d.polygon([(x - 5, y1 - 8), (x + 5, y1 - 8), (x, y1)], fill=green)

    def harrow(y, x0, x1):
        d.line([(x0, y), (x1, y)], fill=green, width=2)
        d.polygon([(x0 + 8, y - 5), (x0 + 8, y + 5), (x0, y)], fill=green)
        d.polygon([(x1 - 8, y - 5), (x1 - 8, y + 5), (x1, y)], fill=green)

    # الارتفاع الكلي — يسار
    varrow(mx_l - 12, mx_t, mx_t + h)
    _vlabel(canvas, f"{total_h_mm:.1f} mm", (mx_l - 34, int(mx_t + h * 0.40)), green + (255,), f_big)
    # العرض الكلي — أسفل
    harrow(mx_t + h + 14, mx_l, mx_l + w)
    d.text((mx_l + w / 2 - 26, mx_t + h + 22), f"{total_w_mm:.1f} mm",
           font=_font(f_big), fill=green)
    # موضع الذقن من الأعلى — يمين قريب
    chin_mm = total_h_mm * GUIDE_CHIN
    varrow(mx_l + w + 12, mx_t, chin_y)
    _vlabel(canvas, f"{chin_mm:.1f} mm ({chin_mm * 0.914:.1f}-{chin_mm * 1.086:.1f})",
            (mx_l + w + 16, int(mx_t + h * 0.26)), green + (255,), f_small)
    # بعد العينين عن الأسفل — يمين بعيد
    eye_mm = total_h_mm * (1 - GUIDE_EYE)
    varrow(mx_l + w + int(mx_r * 0.52), eye_y, mx_t + h)
    _vlabel(canvas, f"{eye_mm:.1f} mm ({eye_mm * 0.90:.1f}-{eye_mm * 1.10:.1f})",
            (mx_l + w + int(mx_r * 0.52) + 5, int(mx_t + h * 0.62)), green + (255,), f_small)
    return canvas


def save_jpg(img: Image.Image, path: str, target_kb: float = 0) -> str:
    """
    حفظ JPG بأعلى جودة (100 وبدون تخفيض ألوان 4:4:4) مع ختم 300 DPI.
    عند طلب حجم مستهدف: تخفيض تدريجي للجودة حتى الوصول إليه.
    """
    if target_kb and target_kb > 0:
        limit = float(target_kb) * 1024
        quality = 100
        while True:
            img.save(path, "JPEG", quality=quality, subsampling=0, dpi=(DPI, DPI))
            if os.path.getsize(path) <= limit or quality <= 35:
                break
            quality -= 6
    else:
        img.save(path, "JPEG", quality=100, subsampling=0, dpi=(DPI, DPI))
    return path


def build_sheet(tile: Image.Image, sheet_cfg) -> Image.Image:
    """
    تجميع النسخ على ورقة طباعة بقياس حقيقي (300 DPI):
    توزيع متوازن — فراغات متساوية بين الصور وعند الحواف —
    وخطوط قص رمادية تمتد من طرف الورقة إلى طرفها لتسهيل القص الدقيق.
    """
    (sheet_w, sheet_h), cols, rows = sheet_cfg
    tw, th = tile.size
    gap_x = (sheet_w - cols * tw) / (cols + 1)
    gap_y = (sheet_h - rows * th) / (rows + 1)
    xs = [round((i + 1) * gap_x + i * tw) for i in range(cols)]
    ys = [round((j + 1) * gap_y + j * th) for j in range(rows)]
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    for y in ys:
        for x in xs:
            sheet.paste(tile, (x, y))
    draw = ImageDraw.Draw(sheet)
    line = (176, 176, 176)
    for x in xs:  # خطوط عمودية بطول الورقة كاملاً عند حدَّي كل صورة
        draw.line([(x, 0), (x, sheet_h)], fill=line, width=1)
        draw.line([(x + tw - 1, 0), (x + tw - 1, sheet_h)], fill=line, width=1)
    for y in ys:  # خطوط أفقية بعرض الورقة كاملاً
        draw.line([(0, y), (sheet_w, y)], fill=line, width=1)
        draw.line([(0, y + th - 1), (sheet_w, y + th - 1)], fill=line, width=1)
    return sheet


# ----------------------------- أحداث الواجهة -----------------------------

PREVIEW_ARGS = ("angle", "zoom", "pan_x", "pan_y", "size_label", "custom_w",
                "custom_h", "color_label", "custom_color", "guides", "brightness",
                "contrast", "saturation", "sharpness", "shadows", "highlights",
                "temperature")


_mtcnn_detector = None


def _get_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        from mtcnnruntime import MTCNN
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector


def compute_auto_init(matting_bgra, width, height):
    """
    استخراج ضبط ابتدائي للوضع اليدوي يطابق تأطير القص التلقائي:
    يُحاكي نافذة adjust_photo (نفس الحسابات والخطوة الثانية وخطوة الإزاحة)
    على نسخة إزالة الخلفية مع مستطيل الوجه من MTCNN، ثم يحوّل النافذة إلى
    (تكبير، إزاحة أفقية، إزاحة عمودية) لمنزلقات الوضع اليدوي.
    يعيد None عند عدم كشف وجه واحد واضح.
    """
    import math
    try:
        rgb = np.array(composite_bg(matting_bgra, (255, 255, 255)))
        faces, _lm = _get_mtcnn().detect(rgb)
        if faces is None or len(faces) == 0:
            return None
        # عند تعدد الوجوه (صورة جماعية) نوسّط على الأكبر — نقطة بداية يحرّكها المستخدم
        faces = sorted(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
        n_faces = len(faces)
        fx1, fy1, fx2, fy2 = [float(v) for v in faces[0][:4]]
        x, y, w, h = fx1, fy1, fx2 - fx1, fy2 - fy1
        mh, mw = matting_bgra.shape[:2]
        face_center = (x + w / 2, y + h / 2)
        crop_measure = (w * h) / CROP_PARAMS["head_measure_ratio"]
        rrs = math.sqrt(crop_measure / (height * width))
        crop_size = (int(height * rrs), int(width * rrs))
        x1 = int(face_center[0] - crop_size[1] / 2)
        y1 = int(face_center[1] - crop_size[0] * CROP_PARAMS["head_height_ratio"])
        x2, y2 = x1 + crop_size[1], y1 + crop_size[0]
        cut = IDphotos_cut(x1, y1, x2, y2, matting_bgra)
        cut = cv2.resize(cut, (crop_size[1], crop_size[0]))
        y_top, _yb, x_left, x_right = get_box(cut.astype(np.uint8), model=2,
                                              correction_factor=0)
        if x_left > 0 or x_right > 0:
            cut_value_top = int(((x_left + x_right) * (height / width)) / 2)
        else:
            cut_value_top = 0
        status_top, move_value = detect_distance(
            y_top - cut_value_top, crop_size[0],
            max=CROP_PARAMS["head_top_range"][0], min=CROP_PARAMS["head_top_range"][1])
        X1 = x1 + x_left
        Y1 = y1 + cut_value_top + status_top * move_value
        X2 = x2 - x_right
        Y2 = y2 - cut_value_top + status_top * move_value
        win_img = IDphotos_cut(X1, Y1, X2, Y2, matting_bgra)
        _moved, y_high = move(win_img.astype(np.uint8))
        Y1 -= y_high
        Y2 -= y_high
        win_w, win_h = X2 - X1, Y2 - Y1
        if win_w <= 10 or win_h <= 10:
            return None
        s_total = width / win_w
        base = max(width / mw, height / mh)
        zoom = max(0.2, min(5.0, s_total / base))
        ex = mw * s_total - width
        ey = mh * s_total - height
        fx = (X1 * s_total) / ex if ex > 1 else 0.5
        fy = (Y1 * s_total) / ey if ey > 1 else 0.5
        pan_x = max(-100.0, min(100.0, (fx - 0.5) * 200))
        pan_y = max(-100.0, min(100.0, (fy - 0.5) * 200))
        return float(zoom), float(pan_x), float(pan_y), n_faces
    except Exception:
        return None


def on_upload(img, mode, *preview_vals):
    """عند رفع صورة: إزالة الخلفية مرة واحدة + بناء نسخة HQ + توسيط الوجه تلقائياً للوضع اليدوي."""
    skip = gr.skip()
    if img is None:
        return None, None, "", skip, skip, skip
    try:
        bgr = normalize_input(img)
        creator = get_creator()
        result = creator(bgr, change_bg_only=True)
        state = {"matting": result.standard,
                 "matting_hq": build_virtual_bgra(bgr, result.standard),
                 "init": None}
        height, width = resolve_size(preview_vals[4], preview_vals[5], preview_vals[6])
        _init = compute_auto_init(result.standard, width, height)
        state["init"] = _init[:3] if _init else None
        state["init_faces"] = _init[3] if _init else 0
    except Exception as e:
        return None, None, f"تعذّر تجهيز الصورة: {e}", skip, skip, skip
    init = state["init"]
    vals = list(preview_vals)
    if init is not None:
        vals[1], vals[2], vals[3] = init
    preview = manual_preview(state, *vals) if mode == MODES[1] else None
    hq_note = " بدقة عالية كاملة" if state["matting_hq"] is not None else ""
    if init is not None:
        if state.get("init_faces", 1) > 1:
            status = (f"تم تجهيز الصورة{hq_note} ✓ ووُسّطت على أكبر وجه ظاهر — "
                      "حرّك الإطار بالمنزلقات نحو الشخص المقصود إن كان غيره، أو اصنع مباشرة.")
        else:
            status = (f"تم تجهيز الصورة{hq_note} ✓ ووُسّط الوجه تلقائياً كما في القص الذكي — "
                      "في الوضع اليدوي حرّك قليلاً يميناً أو يساراً أو ارفع وأنزل إن أردت، أو اصنع مباشرة.")
        return state, preview, status, init[0], init[1], init[2]
    status = (f"تم تجهيز الصورة{hq_note} ✓ — لم يُكتشف وجه واحد واضح للتوسيط التلقائي "
              "(صورة جماعية أو وجه غير واضح)، اضبط القص يدوياً بالمنزلقات.")
    return state, preview, status, skip, skip, skip


def manual_preview(state, angle, zoom, pan_x, pan_y, size_label, custom_w, custom_h,
                   color_label, custom_color, guides, brightness, contrast,
                   saturation, sharpness, shadows, highlights, temperature):
    """المعاينة تعمل على النسخة المصغّرة (سريعة) — النتيجة النهائية تُبنى بدقة كاملة."""
    if not state or "matting" not in state:
        return None
    try:
        height, width = resolve_size(size_label, custom_w, custom_h)
        rgb = resolve_color(color_label, custom_color)
        img = render_manual(state["matting"], angle, zoom, pan_x, pan_y, width, height, rgb)
        img = apply_color_adjustments(img, brightness, contrast, saturation,
                                      sharpness, shadows, highlights, temperature)
        if guides:
            img = draw_guides(img)
        return np.array(img)
    except Exception:
        return None


def make_id_photo(img, state, mode, size_label, custom_w_mm, custom_h_mm,
                  color_label, custom_color, sheet_label, align_face, target_kb,
                  angle, zoom, pan_x, pan_y,
                  brightness, contrast, saturation, sharpness,
                  shadows, highlights, temperature):
    height, width = resolve_size(size_label, custom_w_mm, custom_h_mm)
    rgb = resolve_color(color_label, custom_color)
    adj = dict(brightness=brightness, contrast=contrast, saturation=saturation,
               sharpness=sharpness, shadows=shadows, highlights=highlights,
               temperature=temperature)
    hq_used = False
    small_src = False

    if mode == MODES[1]:
        # ---------- الوضع اليدوي ----------
        if not state or "matting" not in state:
            raise gr.Error("ارفع الصورة أولاً وانتظر رسالة «تم تجهيز الصورة» — الضبط اليدوي يعمل على الصورة المجهّزة.")
        _hq = state.get("matting_hq")
        source = _hq if _hq is not None else state["matting"]
        hq_used = _hq is not None
        std_img = render_manual(source, angle, zoom, pan_x, pan_y, width, height, rgb)
        hd_w, hd_h = round(width * HD_FACTOR), round(height * HD_FACTOR)
        hd_img = render_manual(source, angle, zoom, pan_x, pan_y, hd_w, hd_h, rgb)
    else:
        # ---------- الوضع التلقائي ----------
        if img is None:
            raise gr.Error("ارفع صورة أولاً — اضغط على مربع الصورة واختر صورة من جهازك.")
        bgr = normalize_input(img)
        creator = get_creator()
        try:
            result = creator(bgr, size=(height, width), face_alignment=bool(align_face), **CROP_PARAMS)
        except FaceError as e:
            n = getattr(e, "face_num", 0)
            if n == 0:
                raise gr.Error("لم يظهر أي وجه في الصورة. جرّب صورة أوضح، يكون فيها الوجه من الأمام ومضاءً جيداً — أو بدّل إلى الوضع اليدوي واضبط القص بنفسك.")
            raise gr.Error(f"الصورة تحتوي {n} وجوه — صورة الهوية يجب أن تحتوي وجهاً واحداً فقط. اختر صورة فيها شخص واحد.")
        except Exception as e:
            raise gr.Error(f"حدث خطأ أثناء المعالجة: {e}")
        try:
            import math as _math
            _rect = creator.ctx.face["rectangle"]
            _rrs = _math.sqrt((_rect[2] * _rect[3] / CROP_PARAMS["head_measure_ratio"]) / (height * width))
            small_src = int(width * _rrs) < width
        except Exception:
            small_src = False
        crop_hq = hq_rerender(bgr, creator, (height, width), bool(align_face))
        if crop_hq is not None:
            hq_used = True
            std_img = finalize_from_crop(crop_hq, width, height, rgb)
            hd_img = finalize_from_crop(crop_hq, round(width * HD_FACTOR),
                                        round(height * HD_FACTOR), rgb)
        else:
            hd_img = composite_bg(result.hd, rgb)
            std_img = (hd_img.resize((width, height), Image.LANCZOS)
                       if hd_img.size != (width, height)
                       else composite_bg(result.standard, rgb))

    std_img = mild_sharpen(apply_color_adjustments(std_img, **adj))
    hd_img = mild_sharpen(apply_color_adjustments(hd_img, **adj))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    files = []
    files.append(save_jpg(std_img, os.path.join(OUT_DIR, f"photo_{stamp}.jpg"), target_kb or 0))
    files.append(save_jpg(hd_img, os.path.join(OUT_DIR, f"photo_{stamp}_hd.jpg"), 0))

    sheet_np = None
    sheet_cfg = SHEETS.get(sheet_label)
    if sheet_cfg is not None:
        sheet = build_sheet(std_img, sheet_cfg)
        files.append(save_jpg(sheet, os.path.join(OUT_DIR, f"sheet_{stamp}.jpg"), 0))
        sheet_np = np.array(sheet)

    if mode == MODES[1] and state and "matting" in state:
        _src = state.get("matting_hq")
        _src = _src if _src is not None else state["matting"]
        _sh, _sw = _src.shape[:2]
        _base = max(width / _sw, height / _sh)
        small_src = (width / (_base * max(float(zoom), 0.01))) < width
    _notes = []
    if target_kb and target_kb > 0:
        _notes.append(f"طُبّق ضغط الحجم إلى {int(target_kb)} KB بطلبك — الضغط يخفّض الجودة قليلاً بطبيعته")
    _active = [nm for nm, off in (("السطوع", adj["brightness"] != 1.0), ("التباين", adj["contrast"] != 1.0),
                                  ("التشبع", adj["saturation"] != 1.0), ("الحدة", adj["sharpness"] != 1.0),
                                  ("الظلال", bool(adj["shadows"])), ("الإبرازات", bool(adj["highlights"])),
                                  ("حرارة اللون", bool(adj["temperature"]))) if off]
    if _active:
        _notes.append("تعديلات ألوان مفعّلة (" + "، ".join(_active) + ") — هي وحدها ما يغيّر الألوان عن الأصل؛ للإلغاء اضغط «إعادة ضبط الألوان»")
    if small_src:
        _notes.append("الصورة الأصلية صغيرة أو الوجه بعيد فيها — النتيجة مكبّرة رقمياً وقد تبدو أقل حدة عند التكبير الشديد؛ صورة أقرب وأكبر تعطي نتيجة أفضل")
    mode_name = "يدوي" if mode == MODES[1] else "تلقائي"
    quality_note = " بمسار الجودة العالية من الأصلية كاملة الدقة" if hq_used else ""
    status = (
        f"تم بنجاح ✓ ({mode_name}{quality_note}) — صورة {width}×{height} بكسل (300 DPI) بخلفية "
        f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X} وبصيغة JPG عالية الجودة. "
        f"حُفظت الملفات في مجلد outputs ويمكنك تنزيلها أو طباعتها من الأسفل."
    )
    if _notes:
        status += " | تنبيه: " + "؛ ".join(_notes) + "."
    return np.array(std_img), np.array(hd_img), sheet_np, files, status


# ----------------------------- الطباعة -----------------------------
# دالتان منفصلتان (واحدة للصورة وواحدة للورقة المقسمة) — بلا أي مكوّنات حالة
# وسيطة، حتى تصل الوسائط مرتبة تماماً كما هي معرّفة.

_PRINT_BODY = """
  if (!files || !files.length) { alert('اصنع الصورة أولاً، ثم اطبعها'); return; }
  const nm = f => ((f && (f.name || f.path || f.url)) || '');
  let pick = files.filter(FILTER_EXPR).pop();
  const f = pick || files[files.length - 1];
  const url = (f && f.url) || (f && f.path ? '/file=' + f.path : null);
  if (!url) { alert('تعذّر العثور على الملف للطباعة'); return; }
  let css = '';
  CSS_EXPR
  const w = window.open('', '_blank', 'width=900,height=700');
  if (!w) { window.open(url, '_blank'); return; }
  w.document.write('<html><head><meta charset="utf-8"><title>طباعة</title><style>' + css + '</style></head><body><img src="' + url + '" onload="setTimeout(function(){ window.print(); }, 250);"></body></html>');
  w.document.close();
"""

_SHEET_CSS = """
  const size = sheetLabel.includes('15 × 20') ? '20.32cm 15.24cm' : '15.24cm 10.16cm';
  css = '@page { size: ' + size + '; margin: 0; } html, body { margin: 0; padding: 0; } img { width: 100%; height: 100%; display: block; }';
"""

_PHOTO_CSS = """
  const presets = {'3.5 × 4.5': 35, '2.5 × 3.5': 25, '3.5 × 4.9': 35, '5 × 5': 50};
  let wmm = 35;
  for (const k in presets) { if (sizeLabel.includes(k)) { wmm = presets[k]; break; } }
  if (sizeLabel.includes('مخصص')) wmm = parseFloat(cwMm) || 35;
  css = '@page { margin: 5mm; } body { margin: 0; } img { width: ' + wmm + 'mm; display: block; }';
"""


# ===================== الاستوديو: مراقبة المجلد + طباعة مباشرة =====================


def _sec_label(title, opened):
    return ("▼ " if opened else "◀ ") + title


def clean_folder(p):
    """تنظيف مسار مجلد منسوخ من ويندوز (نسخ كمسار يضيف علامتي تنصيص)."""
    if not p:
        return ""
    return str(p).strip().strip('"').strip().strip("'")


# ===================== قاعدة بيانات الاستوديو (دائمة) =====================
# سجلّ SQLite داخل مجلد البرنامج: كل صورة واصلة تُعرَّف ببصمة محتواها (SHA-256)
# لا باسمها — فالصورة نفسها تُعرَف ولو تغيّر اسمها أو مجلدها، وحالتها
# (وصلت/صُنعت/طُبعت وعدد مرات الطباعة) تعيش عبر الجلسات وإعادات التشغيل.
import hashlib
import sqlite3

DB_FILE = os.path.join(BASE_DIR, "studio.db")


def _fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _db_connect():
    con = sqlite3.connect(DB_FILE, timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS photos(
        fp TEXT PRIMARY KEY,
        name TEXT, src_path TEXT, local_path TEXT,
        arrived_at TEXT, made INTEGER DEFAULT 0,
        printed INTEGER DEFAULT 0, print_count INTEGER DEFAULT 0,
        last_printed_at TEXT)""")
    return con


def _row_to_dict(r):
    return {"fp": r[0], "name": r[1], "src": r[2], "path": r[3],
            "time": (r[4] or "")[11:16], "made": bool(r[5]),
            "printed": bool(r[6]), "print_count": r[7],
            "last_printed": (r[8] or "")[11:16]}


def db_list(limit=60):
    """أحدث الصور المسجلة (الأحدث أولاً) كقواميس جاهزة للعرض."""
    try:
        con = _db_connect()
        rows = con.execute(
            "SELECT fp, name, src_path, local_path, arrived_at, made, "
            "printed, print_count, last_printed_at FROM photos "
            "ORDER BY arrived_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        con.close()
        return [_row_to_dict(r) for r in rows]
    except Exception:
        return []


def db_register(fp, name, src_path, local_path):
    """
    تسجيل وصول صورة: إن كانت بصمتها جديدة أُنشئ سجلّها (created=True)،
    وإن كانت معروفة حُدّث وقت وصولها ومسارها فقط — بلا أي تكرار.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    con = _db_connect()
    cur = con.execute("SELECT fp FROM photos WHERE fp = ?", (fp,))
    created = cur.fetchone() is None
    if created:
        con.execute("INSERT INTO photos(fp, name, src_path, local_path, "
                    "arrived_at) VALUES(?,?,?,?,?)",
                    (fp, name, src_path, local_path, now))
    else:
        con.execute("UPDATE photos SET src_path = ?, local_path = ?, "
                    "arrived_at = ? WHERE fp = ?",
                    (src_path, local_path, now, fp))
    con.commit()
    r = con.execute("SELECT fp, name, src_path, local_path, arrived_at, "
                    "made, printed, print_count, last_printed_at "
                    "FROM photos WHERE fp = ?", (fp,)).fetchone()
    con.close()
    return _row_to_dict(r), created


def db_mark(fp, made=None, printed=False):
    """تحديث حالة صورة: صُنعت و/أو طُبعت (الطبع يزيد العدّاد ووقت آخر طباعة)."""
    con = _db_connect()
    if made is not None:
        con.execute("UPDATE photos SET made = ? WHERE fp = ?",
                    (1 if made else 0, fp))
    if printed:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        con.execute("UPDATE photos SET printed = 1, "
                    "print_count = print_count + 1, last_printed_at = ? "
                    "WHERE fp = ?", (now, fp))
    con.commit()
    con.close()


def db_mark_by_local(local_path):
    """ربط طباعة مباشرة بصورتها في السجل عبر مسار النسخة المحلية."""
    try:
        con = _db_connect()
        r = con.execute("SELECT fp FROM photos WHERE local_path = ?",
                        (local_path,)).fetchone()
        con.close()
        if r:
            db_mark(r[0], printed=True)
            return True
    except Exception:
        pass
    return False


def render_dynamic(pd):
    """بناء تحديثات الإشعارات وقائمة الصور من الحالة (قيم فقط — بلا مكوّنات)."""
    pd = pd or {"photos": [], "notifs": []}
    photos, notifs = pd.get("photos", []), pd.get("notifs", [])
    by_fp = {p["fp"]: p for p in photos}
    out = [pd]
    for i in range(NOTIF_SLOTS):
        if i < len(notifs):
            n = notifs[i]
            rec = by_fp.get(n["pid"])
            nm = rec["name"] if rec else "صورة"
            tm = rec["time"] if rec else ""
            kind = n.get("kind", "new")
            if kind == "again_printed" and rec:
                txt = (f"⚠ **تنبيه: صورة مطبوعة سابقاً وصلت مجدداً** — {nm}  \n"
                       f"🕒 وصلت {tm} · طُبعت {rec['print_count']} مرة "
                       f"(آخرها {rec['last_printed']})")
            elif kind == "again_unprinted":
                txt = (f"📸 **وصلت مجدداً** — {nm}  \n"
                       f"🕒 {tm} · موجودة في قائمتك ولم تُطبع بعد")
            else:
                txt = f"📸 **صورة جديدة وصلت** — {nm}  \n🕒 {tm}"
            out += [gr.update(visible=True), gr.update(value=txt)]
        else:
            out += [gr.update(visible=False), gr.update(value="")]
    for j in range(LIST_ROWS):
        if j < len(photos):
            p = photos[j]
            if p.get("printed"):
                chip = (f"🟢 **تمت طباعتها** (×{p['print_count']} — "
                        f"آخرها {p['last_printed']})")
            else:
                chip = "🔴 **لم تُطبع بعد**"
            out += [gr.update(visible=True),
                    gr.update(value=f"**{p['name']}** — 🕒 {p['time']} — {chip}")]
        else:
            out += [gr.update(visible=False), gr.update(value="")]
    return tuple(out)


def _file_complete(path):
    """
    اكتمال ملف صورة بلا فكّ كامل: JPEG الكامل ينتهي بعلامة FFD9 وPNG
    بمقطع IEND — ملف ما زال يُنسخ لا يحمل خاتمته بعد.
    """
    try:
        low = path.lower()
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            n = f.tell()
            if low.endswith((".jpg", ".jpeg")):
                if n < 4:
                    return False
                f.seek(n - 2)
                return f.read(2) == b"\xff\xd9"
            if low.endswith(".png"):
                if n < 12:
                    return False
                f.seek(n - 12)
                return f.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
            return True
    except OSError:
        return False


def _inbox_copy(path):
    """
    نسخ صورة الاستقبال إلى outputs/inbox داخل مجلد البرنامج.
    ضروري: غراديو يرفض (InvalidPathError) عرض أي ملف تُرجعه الدوال إن لم يكن
    داخل مجلد العمل أو المؤقت — ومجلد الكاميرا خارجي بطبيعته. النسخة مطابقة
    بايتاً للأصل، والأصل لا يُلمس.
    """
    try:
        dst_dir = os.path.join(BASE_DIR, "outputs", "inbox")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(path))
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(path):
            return dst
        shutil.copy2(path, dst)
        return dst
    except Exception:
        return path


def newest_inbox_image(folder):
    """أحدث صورة (JPG/PNG) في مجلد الاستقبال، أو None."""
    try:
        best, best_t = None, -1.0
        for name in os.listdir(folder):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                p = os.path.join(folder, name)
                try:
                    t = os.path.getmtime(p)
                except OSError:
                    continue
                if t > best_t:
                    best, best_t = p, t
        return best
    except OSError:
        return None


def _lan_ip():
    try:
        import socket
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        ip = _s.getsockname()[0]
        _s.close()
        return ip
    except Exception:
        return None


def watch_tick(wstate, photos, inbox_dir, watch_on, auto_make, cur_img, mode,
               sheet_label, align_face, kb,
               angle, zoom, pan_x, pan_y, size_label, cw, ch,
               color_label, custom_color, guides,
               bright, contrast, satur, sharp, shadows, highlights, temp):
    """
    نبضة المؤقّت: مسح كامل لمجلد الاستقبال في كل نبضة —
    • كل صورة جديدة (حتى لو وصلت دفعة كاملة معاً) تُسجَّل في قاعدة البيانات
      بعد استقرار حجمها، لا أحدث صورة فقط.
    • أول نبضة في الجلسة «بذرة» صامتة: تسجيل بلا إشعارات وبلا لمس للوحة
      (تحديث الصفحة لا يزيف وصولاً جديداً ولا يمسح عملك).
    • صورة معروفة البصمة تعود → إشعار (تحذير إن طُبعت سابقاً) بلا تكرار سجل.
    • القائمة تُقرأ دائماً من قاعدة البيانات: الحقيقة واحدة لكل الجلسات.
    """
    _s = gr.skip()
    SKIP12 = (_s,) * 12
    SKIPDYN = (_s,) * (1 + NOTIF_SLOTS * 2 + LIST_ROWS * 2)
    pd = {"photos": [], "notifs": list((photos or {}).get("notifs", []))}
    now_ts = datetime.datetime.now().timestamp()
    kept = [n for n in pd["notifs"] if n.get("expires", 0) > now_ts]
    dyn_changed = len(kept) != len(pd["notifs"])
    pd["notifs"] = kept
    folder = clean_folder(inbox_dir)
    if not watch_on or not folder or not os.path.isdir(folder):
        if dyn_changed:
            pd["photos"] = db_list()
            return SKIP12 + render_dynamic(pd)
        return SKIP12 + SKIPDYN

    ws = dict(wstate or {})
    present = dict(ws.get("present") or {})
    pending = dict(ws.get("pending") or {})
    first = not ws.get("seeded")

    scan = {}
    try:
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fp_path = os.path.join(folder, fname)
                try:
                    scan[fp_path] = os.path.getsize(fp_path)
                except OSError:
                    pass
    except OSError:
        return SKIP12 + (render_dynamic({**pd, "photos": db_list()})
                         if dyn_changed else SKIPDYN)

    # الاستقرار أولاً: لا تُعالج إلا صورة كانت معلّقة من نبضة سابقة وثبت حجمها
    ready = []
    for p, info in list(pending.items()):
        if p not in scan:
            pending.pop(p)
        elif scan[p] == info["size"]:
            ready.append((p, bool(info.get("silent"))))
            pending.pop(p)
        else:
            info["size"] = scan[p]
    # ثم تسجيل الظهورات الجديدة كمعلّقة (تُعالج في النبضة التالية إن ثبتت)
    for p in scan:
        if p not in present and p not in pending:
            pending[p] = {"size": scan[p], "silent": bool(first)}
    ws["present"] = scan
    ws["pending"] = pending
    ws["seeded"] = True

    if not ready:
        if dyn_changed:
            pd["photos"] = db_list()
            return (ws,) + (_s,) * 11 + render_dynamic(pd)
        return (ws,) + (_s,) * 11 + SKIPDYN

    ready.sort(key=lambda t: os.path.getmtime(t[0]))
    known_fps = {r["fp"] for r in db_list(500)}
    preview_vals = [angle, zoom, pan_x, pan_y, size_label, cw, ch,
                    color_label, custom_color, guides,
                    bright, contrast, satur, sharp, shadows, highlights, temp]
    base = None
    lines = []
    loud = 0
    for path, silent in ready:
        local = _inbox_copy(path)
        if not _file_complete(local):
            # ملف لم يكتمل نزوله بعد: يبقى معلّقاً حتى تظهر خاتمته
            pending[path] = {"size": scan.get(path, -1), "silent": silent}
            continue
        try:
            fp = _fingerprint(local)
        except Exception:
            continue
        if silent and fp in known_fps:
            continue  # معروفة من قبل: لا نلمس سجلّها (وقتها وحالتها محفوظان)
        row, created = db_register(fp, os.path.basename(path), path, local)
        if silent:
            continue
        loud += 1
        kind = ("new" if created else
                "again_printed" if row["printed"] else "again_unprinted")
        pd["notifs"] = [n for n in pd["notifs"] if n["pid"] != fp]
        pd["notifs"].append({"pid": fp, "kind": kind,
                             "expires": now_ts + 30})
        st, preview, ostatus, nzoom, npanx, npany = on_upload(
            local, mode, *preview_vals)
        name = os.path.basename(path)
        if st is None:
            lines.append(f"📥 وصلت ({name}) لكن تعذّر تجهيزها: {ostatus}")
            base = (ws, local, None, None, _s, _s, _s, _s,
                    "\n".join(lines), _s, _s, _s)
            continue
        if not auto_make:
            lines.append(f"📥 وصلت صورة من الكاميرا: {name} — {ostatus}")
            base = (ws, local, st, preview, _s, _s, _s, _s,
                    "\n".join(lines), nzoom, npanx, npany)
            continue
        ez, epx, epy = zoom, pan_x, pan_y
        if st.get("init"):
            ez, epx, epy = st["init"]
        try:
            m = make_id_photo(local, st, mode, size_label, cw, ch,
                              color_label, custom_color, sheet_label,
                              align_face, kb, angle, ez, epx, epy,
                              bright, contrast, satur, sharp,
                              shadows, highlights, temp)
            db_mark(fp, made=True)
            lines.append(f"📥 ({name}) — صُنعت تلقائياً ✓\n{m[4]}")
            base = (ws, local, st, preview, m[0], m[1], m[2], m[3],
                    "\n".join(lines), nzoom, npanx, npany)
        except gr.Error as e:
            msg = getattr(e, "message", None) or str(e)
            lines.append(f"📥 وصلت ({name}) لكن تعذّر صنعها: {msg}")
            base = (ws, local, st, preview, _s, _s, _s, _s,
                    "\n".join(lines), nzoom, npanx, npany)
        except Exception as e:
            lines.append(f"📥 وصلت ({name}) وخطأ أثناء الصنع: {e}")
            base = (ws, local, st, preview, _s, _s, _s, _s,
                    "\n".join(lines), nzoom, npanx, npany)
    if loud > 1:
        lines.insert(0, f"📥 وصلت {loud} صور دفعة واحدة — سُجّلت وصُنعت كلها:")
    pd["photos"] = db_list()
    dyn = render_dynamic(pd)
    if base is None:
        return (ws,) + (_s,) * 11 + dyn
    if loud > 1:
        base = base[:8] + ("\n".join(lines),) + base[9:]
    return base + dyn


def quick_make_print(photos, which, idx, kind, mode, printer, align_face, kb,
                     *preview_vals):
    """
    زر «تجهيز مباشر + طبع»: يجهّز صورة من القائمة/الإشعار بورقة 8 أو 12
    ويطبعها مباشرة على الطابعة المختارة (تنفيذ على الخادم — يعمل من الهاتف).
    """
    _s = gr.skip()
    pd = {"photos": db_list(),
          "notifs": list((photos or {}).get("notifs", []))}
    rec = None
    if which == "slot":
        if 0 <= idx < len(pd["notifs"]):
            pid = pd["notifs"][idx]["pid"]
            rec = next((x for x in pd["photos"] if x["fp"] == pid), None)
    elif 0 <= idx < len(pd["photos"]):
        rec = pd["photos"][idx]
    if rec is None:
        return (pd, _s, _s, _s, _s, _s) + render_dynamic({**pd, 'photos': db_list()})[1:]
    sheet_label = ("8 صور — ورقة 10 × 15 سم" if kind == "8"
                   else "12 صورة — ورقة 15 × 20 سم")
    try:
        st, _pv, _ost, _nz, _nx, _ny = on_upload(rec["path"], mode, *preview_vals)
        if st is None:
            return (pd, _s, _s, _s, _s,
                    f"تعذّر تجهيز الصورة {rec['name']} — جرّب صنعها من اللوحة يدوياً.") \
                + render_dynamic({**pd, 'photos': db_list()})[1:]
        ez, epx, epy = preview_vals[1], preview_vals[2], preview_vals[3]
        if st.get("init"):
            ez, epx, epy = st["init"]
        m = make_id_photo(rec["path"], st, mode, preview_vals[4], preview_vals[5],
                          preview_vals[6], preview_vals[7], preview_vals[8],
                          sheet_label, align_face, kb, preview_vals[0],
                          ez, epx, epy, *preview_vals[10:17])
        status = f"⚡ تجهيز مباشر ({kind} صور) — {rec['name']} ✓\n{m[4]}"
        db_mark(rec["fp"], made=True)
        if printer:
            pr = direct_print(m[3], printer, True, sheet_label)
            status += "\n" + pr
            if pr.startswith("🖨"):
                db_mark(rec["fp"], printed=True)
                if rec.get("printed"):
                    status += "\n(إعادة طباعة — هذه الصورة كانت مطبوعة سابقاً.)"
        else:
            status += "\n⚠ صُنعت الورقة لكن لم تُطبع: اختر الطابعة أولاً من قسم الاستوديو."
        if which == "slot":
            pd["notifs"] = [n for n in pd["notifs"] if n["pid"] != rec["fp"]]
        return (pd, m[0], m[1], m[2], m[3], status) + render_dynamic({**pd, 'photos': db_list()})[1:]
    except gr.Error as e:
        msg = getattr(e, "message", None) or str(e)
        return (pd, _s, _s, _s, _s,
                f"⚡ تعذّر التجهيز المباشر لـ{rec['name']}: {msg}") + render_dynamic({**pd, 'photos': db_list()})[1:]
    except Exception as e:
        return (pd, _s, _s, _s, _s,
                f"⚡ خطأ أثناء التجهيز المباشر لـ{rec['name']}: {e}") + render_dynamic({**pd, 'photos': db_list()})[1:]


def list_printers():
    """أسماء الطابعات المثبتة (ويندوز فقط — عبر PowerShell)."""
    if os.name != "nt":
        return []
    try:
        import subprocess
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-Printer | Select-Object -ExpandProperty Name"],
                           capture_output=True, text=True, timeout=25,
                           encoding="utf-8", errors="replace")
        return [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    except Exception:
        return []


def _pick_result(files, want_sheet):
    paths = []
    for f in (files or []):
        p = f.get("path") if isinstance(f, dict) else str(f)
        if p:
            paths.append(p)
    if not paths:
        return None
    for p in paths:
        b = os.path.basename(p).lower()
        if want_sheet and b.startswith("sheet"):
            return p
        if not want_sheet and b.startswith("photo") and not b.endswith("_hd.jpg"):
            return p
    return paths[-1] if want_sheet else paths[0]


def direct_print(files, printer, want_sheet, sheet_label):
    """
    طباعة مباشرة صامتة على طابعة الحاسوب (ويندوز): تُنفَّذ على الخادم نفسه،
    فتعمل حتى لو ضغط الزر من متصفح الهاتف — بالمقاس الحقيقي عبر print_sheet.ps1.
    """
    if os.name != "nt":
        return "الطباعة المباشرة الصامتة تعمل على ويندوز فقط — شغّل البرنامج على حاسوب الويندوز."
    target = _pick_result(files, want_sheet)
    if not target or not os.path.isfile(target):
        return "اصنع صورة الهوية أولاً — لا يوجد ملف نتيجة جاهز للطباعة بعد."
    ps1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print_sheet.ps1")
    if not os.path.isfile(ps1):
        return "ملف الطباعة print_sheet.ps1 غير موجود — ضعه بجانب app.py في نفس المجلد."
    if want_sheet and sheet_label and "15 × 20" in sheet_label:
        pw, ph, mode = 203.2, 152.4, "fill"
    elif want_sheet:
        pw, ph, mode = 152.4, 101.6, "fill"
    else:
        pw, ph, mode = 152.4, 101.6, "exact"
    try:
        import subprocess
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", ps1, "-Path", target,
                            "-Printer", printer or "",
                            "-PaperW", str(pw), "-PaperH", str(ph), "-Mode", mode],
                           capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace")
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if r.returncode == 0 and "OK" in out:
            kind = "الورقة المقسمة" if want_sheet else "الصورة"
            return (f"🖨 أُرسلت {kind} إلى الطابعة ({printer or 'الافتراضية'}) "
                    f"— تُطبع الآن بالمقاس الحقيقي بلا أي نافذة.")
        return f"تعذّرت الطباعة المباشرة: {err or out or 'خطأ غير معروف من الطابعة'}"
    except Exception as e:
        return f"تعذّرت الطباعة المباشرة: {e}"


PRINT_SHEET_JS = ("(files, sheetLabel, sizeLabel, cwMm, chMm) => {" +
                  _PRINT_BODY.replace("FILTER_EXPR", "f => nm(f).includes('sheet_')")
                             .replace("CSS_EXPR", _SHEET_CSS) + "}")
PRINT_PHOTO_JS = ("(files, sheetLabel, sizeLabel, cwMm, chMm) => {" +
                  _PRINT_BODY.replace("FILTER_EXPR", "f => nm(f).includes('photo_') && !nm(f).includes('_hd')")
                             .replace("CSS_EXPR", _PHOTO_CSS) + "}")

# ----------------------------- الواجهة -----------------------------

CUSTOM_CSS = """
.gradio-container { direction: rtl; font-family: Tahoma, 'Segoe UI', sans-serif; }
#app-title { text-align: center; padding: 18px 10px 4px; }
#app-title h1 { font-size: 30px; margin: 0; color: #0b6b3a; }
#app-title p { color: #555; margin: 6px 0 0; }
.section-head { margin-top: 8px; }
.section-head button { justify-content: flex-start; font-weight: 700; font-size: 15px; }
#make-btn { font-size: 19px !important; padding: 14px !important; }
footer { display: none !important; }
"""

S = SETTINGS

with gr.Blocks(title="صانع صور الهوية — الجزائر") as demo:
    gr.HTML("""
    <div id="app-title">
      <h1>🇩🇿 صانع صور الهوية — الجزائر</h1>
      <p>قصّ ذكي أو ضبط يدوي احترافي بأدلة قياس، جودة كاملة من صورتك الأصلية، وطباعة مباشرة — كل شيء على جهازك</p>
    </div>
    """)

    state = gr.State(None)
    watch_state = gr.State({"present": {}, "pending": {}, "seeded": False})
    photos_state = gr.State({"photos": db_list(), "notifs": []})

    # ======== إشعارات الصور الجديدة (تعيش 30 ثانية) ========
    notif_slots = []
    for _i in range(NOTIF_SLOTS):
        with gr.Group(visible=False) as _ng:
            _nmd = gr.Markdown("")
            with gr.Row():
                _nb8 = gr.Button("⚡ تجهيز مباشر 8 صور + طبع", size="sm", variant="primary")
                _nb12 = gr.Button("⚡ تجهيز مباشر 12 صورة + طبع", size="sm")
                _nbh = gr.Button("إخفاء", size="sm")
        notif_slots.append((_ng, _nmd, _nb8, _nb12, _nbh))


    with gr.Row():
        # ======== لوحة التحكم (يمين الشاشة) ========
        with gr.Column(scale=6):
            in_image = gr.Image(label="١ — ارفع صورتك", type="filepath",
                                sources=["upload", "webcam"], height=280)
            mode_radio = gr.Radio(MODES, value=S["mode"], label="٢ — وضع العمل")
            out_preview = gr.Image(label="معاينة العمل (الأدلة هنا فقط — لا تظهر في النتائج أبداً)",
                                   type="numpy", interactive=False, height=330,
                                   visible=(S["mode"] == MODES[1]))
            guide_md = gr.Markdown(
                "دليل الأدلة (كما في القوالب المعتمدة): الأسهم الخضراء حول الصورة = الارتفاع والعرض الكليان، "
                "وموضع الذقن من الأعلى، وبعد العينين عن الأسفل — بالمليمتر مع مجال التسامح. "
                "وعلى الصورة: ضع قمة الرأس عند المنطقة الحمراء، والذقن عند الخط الأخضر، والعينين داخل الشريط الأزرق.",
                visible=(S["mode"] == MODES[1]))

            cut_head = gr.Button(_sec_label("✂️ القص والمقاس", S["ui_cut"]), elem_classes=["section-head"])
            with gr.Group(visible=S["ui_cut"]) as cut_group:
                size_radio = gr.Radio(list(SIZES.keys()), value=S["size"], label="مقاس الصورة")
                with gr.Row(visible=(S["size"] == "مقاس مخصص…")) as custom_size_row:
                    custom_w = gr.Number(label="العرض (مم)", value=S["custom_w"], minimum=8, maximum=250)
                    custom_h = gr.Number(label="الارتفاع (مم)", value=S["custom_h"], minimum=8, maximum=250)
                color_radio = gr.Radio(list(COLORS.keys()), value=S["color"], label="لون الخلفية")
                custom_color = gr.ColorPicker(label="اختر لونك", value=S["custom_color"],
                                              visible=(S["color"] == "لون مخصص…"))
                with gr.Group(visible=(S["mode"] == MODES[0])) as auto_group:
                    align_chk = gr.Checkbox(label="تصحيح ميلان الوجه تلقائياً (حسب العينين)", value=S["align"])
                    kb_num = gr.Number(label="حجم ملف مستهدف (KB) — 0 يعني أعلى جودة (يُضغط لهذه المرة فقط ولا يُحفظ)", value=0, minimum=0)
                with gr.Group(visible=(S["mode"] == MODES[1])) as manual_group:
                    angle_slider = gr.Slider(-15, 15, value=S["angle"], step=0.1,
                                             label="تدوير الصورة (درجة) — لاستقامة الكتفين والرأس")
                    angle_num = gr.Number(label="الزاوية بدقة (اكتب الرقم ثم Enter)", value=S["angle"],
                                          minimum=-15, maximum=15, step=0.1)
                    zoom_slider = gr.Slider(0.2, 5.0, value=S["zoom"], step=0.01,
                                            label="التقريب والتبعيد (1 = ملء الإطار)")
                    pan_x = gr.Slider(-100, 100, value=S["pan_x"], step=1, label="إزاحة أفقية")
                    pan_y = gr.Slider(-100, 100, value=S["pan_y"], step=1, label="إزاحة عمودية")
                    guides_chk = gr.Checkbox(label="إظهار أدلة القص في المعاينة", value=S["guides"])
                    center_btn = gr.Button("🎯 توسيط الوجه تلقائياً", size="sm")
                    reset_manual_btn = gr.Button("إعادة ضبط اليدوي", size="sm")

            color_head = gr.Button(_sec_label("🎨 الألوان والإضاءة", S["ui_color"]), elem_classes=["section-head"])
            with gr.Group(visible=S["ui_color"]) as color_group:
                bright_s = gr.Slider(0.5, 1.5, value=S["brightness"], step=0.01, label="السطوع")
                contrast_s = gr.Slider(0.5, 1.5, value=S["contrast"], step=0.01, label="التباين")
                satur_s = gr.Slider(0, 2.0, value=S["saturation"], step=0.01, label="تشبع الألوان")
                sharp_s = gr.Slider(0, 3.0, value=S["sharpness"], step=0.05, label="الحدة")
                shadows_s = gr.Slider(-100, 100, value=S["shadows"], step=1, label="الظلال (تفتيح المناطق الداكنة)")
                highlights_s = gr.Slider(-100, 100, value=S["highlights"], step=1, label="الإبرازات (المناطق الفاتحة)")
                temp_s = gr.Slider(-50, 50, value=S["temperature"], step=1, label="حرارة اللون (بارد ← دافئ)")
                reset_color_btn = gr.Button("إعادة ضبط الألوان", size="sm")

            paper_head = gr.Button(_sec_label("🖨 الورقة والطباعة", S["ui_paper"]), elem_classes=["section-head"])
            with gr.Group(visible=S["ui_paper"]) as paper_group:
                sheet_radio = gr.Radio(list(SHEETS.keys()), value=S["sheet"], label="ورقة الطباعة")
                with gr.Row():
                    print_photo_btn = gr.Button("🖨 طباعة الصورة", size="sm")
                    print_sheet_btn = gr.Button("🖨 طباعة الورقة المقسمة", size="sm")
                gr.Markdown("الطباعة بحجم حقيقي (100%) — عند الطباعة اختر «الحجم الفعلي» ولا تختر «ملء الصفحة».")

            studio_head = gr.Button(_sec_label("📥 الاستوديو — الكاميرا والهاتف والطباعة المباشرة", S["ui_studio"]), elem_classes=["section-head"])
            with gr.Group(visible=S["ui_studio"]) as studio_group:
                inbox_txt = gr.Textbox(label="مجلد استقبال صور الكاميرا (نفس مجلد وجهة EOS Utility)",
                                       value=S["inbox"], placeholder="مثال: D:\\Studio\\Incoming")
                watch_chk = gr.Checkbox(label="مراقبة المجلد — تحميل أي صورة جديدة تلقائياً", value=S["watch"])
                automake_chk = gr.Checkbox(label="صنع صورة الهوية تلقائياً فور وصول صورة جديدة", value=S["automake"])
                printer_dd = gr.Dropdown(choices=list_printers(), value=(S["printer"] or None),
                                         label="طابعة الطباعة المباشرة", allow_custom_value=True)
                with gr.Row():
                    refresh_printers_btn = gr.Button("🔄 تحديث الطابعات", size="sm")
                    direct_sheet_btn = gr.Button("🖨 طباعة الورقة مباشرة", size="sm")
                    direct_photo_btn = gr.Button("🖨 طباعة الصورة مباشرة", size="sm")

            make_btn = gr.Button("اصنع صورة الهوية", variant="primary", elem_id="make-btn")
            with gr.Row():
                save_settings_btn = gr.Button("💾 احفظ إعداداتي", size="sm")
                reset_settings_btn = gr.Button("↩️ إعادة ضبط الإعدادات", size="sm")

        # ======== النتائج: ثلاث صور فقط ========
        with gr.Column(scale=4):
            list_head = gr.Button(_sec_label("🗂 قائمة الصور الواصلة", S["ui_list"]), elem_classes=["section-head"])
            with gr.Group(visible=S["ui_list"]) as list_group:
                photo_rows = []
                for _j in range(LIST_ROWS):
                    with gr.Group(visible=False) as _rg:
                        _rmd = gr.Markdown("")
                        with gr.Row():
                            _rb8 = gr.Button("⚡ تجهيز 8 صور + طبع", size="sm", variant="primary")
                            _rb12 = gr.Button("⚡ تجهيز 12 صورة + طبع", size="sm")
                    photo_rows.append((_rg, _rmd, _rb8, _rb12))
                gr.Markdown("السجل محفوظ دائماً في قاعدة بيانات البرنامج — يبقى بعد الإغلاق وتحديث الصفحة.")

            out_std = gr.Image(label="الصورة القياسية", type="numpy", interactive=False, height=330)
            out_hd = gr.Image(label="نسخة عالية الدقة", type="numpy", interactive=False, height=250)
            out_sheet = gr.Image(label="ورقة الطباعة", type="numpy", interactive=False, height=250)
            out_status = gr.Textbox(label="الحالة", interactive=False)
            out_files = gr.File(label="تنزيل الملفات (JPG بدقة 300 DPI)", file_count="multiple")

    # ---------- حفظ الإعدادات (بمنع التكرار داخل save_settings_dict) ----------
    SETTINGS_KEYS = ["mode", "size", "custom_w", "custom_h", "color", "custom_color",
                     "sheet", "align", "angle", "zoom", "pan_x", "pan_y", "guides",
                     "brightness", "contrast", "saturation", "sharpness",
                     "shadows", "highlights", "temperature",
                     "inbox", "watch", "automake", "printer"]
    SETTINGS_INPUTS = [mode_radio, size_radio, custom_w, custom_h, color_radio,
                       custom_color, sheet_radio, align_chk, angle_slider,
                       zoom_slider, pan_x, pan_y, guides_chk, bright_s, contrast_s,
                       satur_s, sharp_s, shadows_s, highlights_s, temp_s,
                       inbox_txt, watch_chk, automake_chk, printer_dd]

    def _persist(*vals):
        save_settings_dict(dict(zip(SETTINGS_KEYS, vals)))

    for _c in SETTINGS_INPUTS:
        _c.change(_persist, inputs=SETTINGS_INPUTS, outputs=None)

    # ---------- أقسام قابلة للطي بحالة محفوظة ----------
    SECTION_DEFS = [("ui_cut", cut_head, cut_group, "✂️ القص والمقاس"),
                    ("ui_color", color_head, color_group, "🎨 الألوان والإضاءة"),
                    ("ui_paper", paper_head, paper_group, "🖨 الورقة والطباعة"),
                    ("ui_studio", studio_head, studio_group,
                     "📥 الاستوديو — الكاميرا والهاتف والطباعة المباشرة"),
                    ("ui_list", list_head, list_group, "🗂 قائمة الصور الواصلة")]
    SECTION_OUTS = []
    for _k, _h, _g, _t in SECTION_DEFS:
        SECTION_OUTS += [_g, _h]

    def _section_updates(s):
        out = []
        for key, _h, _g, title in SECTION_DEFS:
            o = bool(s.get(key))
            out += [gr.update(visible=o), gr.update(value=_sec_label(title, o))]
        return out

    def _mk_toggle(key, title):
        def _toggle():
            d = load_settings()
            newv = not d.get(key, False)
            d[key] = newv
            save_settings_dict(d)
            return gr.update(visible=newv), gr.update(value=_sec_label(title, newv))
        return _toggle

    for _k, _h, _g, _t in SECTION_DEFS:
        _h.click(_mk_toggle(_k, _t), inputs=None, outputs=[_g, _h])

    # ---------- مخرجات الإشعارات وقائمة الصور ----------
    DYN_OUTPUTS = [photos_state]
    for _slot in notif_slots:
        DYN_OUTPUTS += [_slot[0], _slot[1]]
    for _row in photo_rows:
        DYN_OUTPUTS += [_row[0], _row[1]]

    # ---------- إظهار/إخفاء ----------
    size_radio.change(lambda v: gr.update(visible=(v == "مقاس مخصص…")),
                      inputs=size_radio, outputs=custom_size_row)
    color_radio.change(lambda v: gr.update(visible=(v == "لون مخصص…")),
                       inputs=color_radio, outputs=custom_color)

    def _mode_switch(m):
        return (gr.update(visible=(m == MODES[0])),
                gr.update(visible=(m == MODES[1])),
                gr.update(visible=(m == MODES[1])))

    mode_radio.change(_mode_switch, inputs=mode_radio,
                      outputs=[auto_group, manual_group, out_preview])

    # ---------- المعاينة: أحداث محسوبة بعناية ----------
    # المنزلقات تُطلق المعاينة عند الإفلات فقط (release) لا أثناء السحب،
    # وخانة الزاوية عند الإدخال (submit) — لا حلقات إعادة بناء أبداً.
    PREVIEW_INPUTS = [angle_slider, zoom_slider, pan_x, pan_y, size_radio, custom_w,
                      custom_h, color_radio, custom_color, guides_chk, bright_s,
                      contrast_s, satur_s, sharp_s, shadows_s, highlights_s, temp_s]

    def _preview(state_v, *vals):
        return manual_preview(state_v, *vals)

    in_image.change(on_upload, inputs=[in_image, mode_radio] + PREVIEW_INPUTS,
                    outputs=[state, out_preview, out_status, zoom_slider, pan_x, pan_y])

    for _c in (angle_slider, zoom_slider, pan_x, pan_y, bright_s, contrast_s,
               satur_s, sharp_s, shadows_s, highlights_s, temp_s):
        _c.release(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)
    for _c in (size_radio, color_radio, custom_color, guides_chk):
        _c.change(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)
    custom_w.change(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)
    custom_h.change(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)
    mode_radio.change(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)

    angle_slider.release(lambda v: v, inputs=angle_slider, outputs=angle_num)
    angle_num.submit(lambda v: v, inputs=angle_num, outputs=angle_slider)
    angle_num.submit(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)

    def _reset_manual(*vals):
        d = dict(zip(SETTINGS_KEYS, vals))
        d.update({"angle": 0.0, "zoom": 1.0, "pan_x": 0, "pan_y": 0})
        save_settings_dict(d)
        return 0.0, 0.0, 1.0, 0, 0

    def _apply_init(state_v, *vals):
        init = (state_v or {}).get("init")
        if not init:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        v = list(vals)
        v[1], v[2], v[3] = init
        return init[0], init[1], init[2], manual_preview(state_v, *v)

    center_btn.click(_apply_init, inputs=[state] + PREVIEW_INPUTS,
                     outputs=[zoom_slider, pan_x, pan_y, out_preview])

    reset_manual_btn.click(_reset_manual, inputs=SETTINGS_INPUTS,
                           outputs=[angle_slider, angle_num, zoom_slider, pan_x, pan_y])

    def _reset_colors(*vals):
        d = dict(zip(SETTINGS_KEYS, vals))
        d.update({"brightness": 1.0, "contrast": 1.0, "saturation": 1.0,
                  "sharpness": 1.0, "shadows": 0, "highlights": 0, "temperature": 0})
        save_settings_dict(d)
        return 1.0, 1.0, 1.0, 1.0, 0, 0, 0

    reset_color_btn.click(_reset_colors, inputs=SETTINGS_INPUTS,
                          outputs=[bright_s, contrast_s, satur_s, sharp_s,
                                   shadows_s, highlights_s, temp_s])

    # ---------- الصنع ----------
    make_btn.click(
        make_id_photo,
        inputs=[in_image, state, mode_radio, size_radio, custom_w, custom_h,
                color_radio, custom_color, sheet_radio, align_chk, kb_num,
                angle_slider, zoom_slider, pan_x, pan_y,
                bright_s, contrast_s, satur_s, sharp_s,
                shadows_s, highlights_s, temp_s],
        outputs=[out_std, out_hd, out_sheet, out_files, out_status],
    )

    # ---------- الطباعة ----------
    print_photo_btn.click(None, inputs=[out_files, sheet_radio,
                                        size_radio, custom_w, custom_h], js=PRINT_PHOTO_JS)
    print_sheet_btn.click(None, inputs=[out_files, sheet_radio,
                                        size_radio, custom_w, custom_h], js=PRINT_SHEET_JS)

    # ---------- الاستوديو: طباعة مباشرة + مراقبة المجلد ----------
    refresh_printers_btn.click(lambda: gr.update(choices=list_printers()),
                               inputs=None, outputs=printer_dd)

    def _after_direct(status, photos, img):
        if status.startswith("🖨") and isinstance(img, str) and img:
            if db_mark_by_local(img):
                status += "\n✓ سُجّلت هذه الطباعة في قائمة الصور."
        pd = {"photos": db_list(),
              "notifs": list((photos or {}).get("notifs", []))}
        return (status,) + render_dynamic(pd)

    def _dprint_sheet(files, printer, sheet_label, photos, img):
        return _after_direct(direct_print(files, printer, True, sheet_label),
                             photos, img)

    def _dprint_photo(files, printer, photos, img):
        return _after_direct(direct_print(files, printer, False, None),
                             photos, img)

    direct_sheet_btn.click(_dprint_sheet,
                           inputs=[out_files, printer_dd, sheet_radio,
                                   photos_state, in_image],
                           outputs=[out_status] + DYN_OUTPUTS)
    direct_photo_btn.click(_dprint_photo,
                           inputs=[out_files, printer_dd, photos_state,
                                   in_image],
                           outputs=[out_status] + DYN_OUTPUTS)

    watch_timer = gr.Timer(1.5)
    watch_timer.tick(
        watch_tick,
        inputs=[watch_state, photos_state, inbox_txt, watch_chk, automake_chk,
                in_image, mode_radio, sheet_radio, align_chk, kb_num] + PREVIEW_INPUTS,
        outputs=[watch_state, in_image, state, out_preview, out_std, out_hd,
                 out_sheet, out_files, out_status, zoom_slider, pan_x,
                 pan_y] + DYN_OUTPUTS)

    # ---------- أزرار التجهيز المباشر + إخفاء الإشعارات + مسح القائمة ----------
    QUICK_INPUTS = [photos_state, mode_radio, printer_dd, align_chk,
                    kb_num] + PREVIEW_INPUTS
    QUICK_OUTPUTS = [photos_state, out_std, out_hd, out_sheet, out_files,
                     out_status] + DYN_OUTPUTS[1:]

    def _mk_quick(which, idx, kind):
        def _h(photos, mode, printer, align, kb, *pv):
            return quick_make_print(photos, which, idx, kind, mode, printer,
                                    align, kb, *pv)
        return _h

    def _mk_dismiss(idx):
        def _d(photos):
            pd = {"photos": list((photos or {}).get("photos", [])),
                  "notifs": list((photos or {}).get("notifs", []))}
            if 0 <= idx < len(pd["notifs"]):
                pd["notifs"].pop(idx)
            return render_dynamic(pd)
        return _d

    for _i, _slot in enumerate(notif_slots):
        _slot[2].click(_mk_quick("slot", _i, "8"), inputs=QUICK_INPUTS,
                       outputs=QUICK_OUTPUTS)
        _slot[3].click(_mk_quick("slot", _i, "12"), inputs=QUICK_INPUTS,
                       outputs=QUICK_OUTPUTS)
        _slot[4].click(_mk_dismiss(_i), inputs=[photos_state],
                       outputs=DYN_OUTPUTS)
    for _j, _row in enumerate(photo_rows):
        _row[2].click(_mk_quick("row", _j, "8"), inputs=QUICK_INPUTS,
                      outputs=QUICK_OUTPUTS)
        _row[3].click(_mk_quick("row", _j, "12"), inputs=QUICK_INPUTS,
                      outputs=QUICK_OUTPUTS)


    # ---------- حفظ/استرجاع الإعدادات ----------
    # المشكلة التي أزعجت المستخدم: قيم البداية تتجمّد لحظة إقلاع الخادم، فكل
    # تحديث للصفحة كان يعرض قيم الإقلاع لا آخر ما حُفظ. الحل: عند كل تحميل
    # للصفحة (demo.load) نعيد قراءة settings.json ونملأ المكوّنات من جديد،
    # + زر حفظ صريح يلتقط حتى الحقول النصية التي لم يُضغط خارجها بعد.
    def _settings_values(s):
        vals = [s[k] for k in SETTINGS_KEYS]
        vals[SETTINGS_KEYS.index("printer")] = \
            vals[SETTINGS_KEYS.index("printer")] or None
        return vals

    def _save_now(*vals):
        save_settings_dict(dict(zip(SETTINGS_KEYS, vals)))
        return ("💾 تم حفظ إعداداتك ✓ — ستبقى كما هي بعد تحديث الصفحة "
                "أو إعادة فتح البرنامج أو من هاتفك.")

    def _reset_now():
        save_settings_dict(dict(DEFAULT_SETTINGS))
        vis = DEFAULT_SETTINGS["mode"] == MODES[1]
        return (*_settings_values(dict(DEFAULT_SETTINGS)),
                gr.update(visible=vis), gr.update(visible=vis),
                *_section_updates(dict(DEFAULT_SETTINGS)),
                "↩️ تمت إعادة ضبط كل الإعدادات إلى قيمها الأصلية.")

    def _load_saved():
        s = load_settings()
        vis = s["mode"] == MODES[1]
        return (*_settings_values(s),
                gr.update(visible=vis), gr.update(visible=vis),
                *_section_updates(s),
                *render_dynamic({"photos": db_list(), "notifs": []}))

    save_settings_btn.click(_save_now, inputs=SETTINGS_INPUTS, outputs=out_status)
    reset_settings_btn.click(_reset_now, inputs=None,
                             outputs=SETTINGS_INPUTS + [out_preview, guide_md]
                                     + SECTION_OUTS + [out_status])
    demo.load(_load_saved, inputs=None,
              outputs=SETTINGS_INPUTS + [out_preview, guide_md] + SECTION_OUTS
                      + DYN_OUTPUTS)

    gr.HTML("""
    <p style="text-align:center;color:#888;font-size:12px">
    <p style="text-align:center;color:#888;font-size:13px">صُنع بواسطة <b>Zizo3114</b></p>
    """)

if __name__ == "__main__":
    # اختيار أول منفذ فارغ تلقائياً: إن بقيت نسخة قديمة من البرنامج تعمل في
    # الخلفية (وهي ما تسبب خطأ «المنفذ مشغول») تُفتح الجديدة على المنفذ التالي.
    import socket

    def _free_port(start=7860):
        for _p in range(start, start + 20):
            with socket.socket() as _s:
                try:
                    _s.bind(("127.0.0.1", _p))
                    return _p
                except OSError:
                    continue
        return start

    # الشبكة المحلية مفتوحة (0.0.0.0) ليصل الهاتف على نفس الوايفاي إلى اللوحة.
    # كلمة سر اختيارية: عيّن متغير البيئة PHOTOID_AUTH على صورة user:pass قبل التشغيل.
    _auth = None
    _ea = os.environ.get("PHOTOID_AUTH", "")
    if ":" in _ea:
        _u, _pw = _ea.split(":", 1)
        _auth = (_u, _pw)
    print("الهاتف (نفس الوايفاي): http://" + str(_lan_ip()) + ":<المنفذ الظاهر أسفله>")
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=_free_port(),
                auth=_auth,
                theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
                css=CUSTOM_CSS)
