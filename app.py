# -*- coding: utf-8 -*-
"""
صانع صور الهوية — الجزائر (نسخة سطح المكتب)

واجهة عربية مبنية على محرك مشروع HivisionIDPhotos مفتوح المصدر
(https://github.com/Zeyi-Lin/HivisionIDPhotos — رخصة Apache-2.0).
المحرك: كشف وجه MTCNN بعتبات صارمة + قاعدة «وجه واحد فقط»،
وقصّ ذكي بنسب رأس قياسية، وإزالة خلفية بنموذج MODNet المهيّأ لصور الهوية.
يوجد أيضاً وضع يدوي كامل: تدوير بالدرجة، تقريب وإزاحة، خطوط إرشاد
(خط المنتصف، خط العينين ودائرتاهما، قمة الرأس، الذقن)، وضبط ألوان.
كل المعالجة تتم على جهاز المستخدم — بدون إنترنت بعد التثبيت الأول.
وتُحفظ كل الإعدادات في settings.json لتعود كما تركتها في كل تشغيل.
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance
import gradio as gr

from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from hivision.error import FaceError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
DPI = 300
HD_FACTOR = 1.4526  # نسبة النسخة عالية الدقة إلى القياسية (كما في المحرك)

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
    "8 صور — ورقة 10 × 15 سم": ((1772, 1181), 4, 2),   # (عرض، ارتفاع) الورقة، أعمدة، صفوف
    "12 صورة — ورقة 15 × 20 سم": ((2362, 1772), 4, 3),
}

MODES = ["تلقائي — كشف الوجه والقص الذكي", "يدوي — أنا أضبط القص بنفسي"]

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


def save_settings_dict(values):
    try:
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

def normalize_input(img: np.ndarray) -> np.ndarray:
    """
    تحويل أي صورة مرفوعة إلى BGR بثلاث قنوات — وهو ما يتوقعه المحرك.
    الصور الشفافة (PNG بقناة ألفا) تُدمج أولاً على خلفية بيضاء؛
    بدورها الأصلي كانت هذه الصور تعطّل نموذج إزالة الخلفية.
    """
    if img is None:
        return None
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


def draw_guides(img: Image.Image) -> Image.Image:
    """خطوط إرشاد احترافية تُرسم على المعاينة فقط — لا تظهر في الصورة النهائية."""
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    cx = w / 2
    red = (238, 52, 52, 230)
    green = (28, 178, 76, 230)
    amber = (245, 166, 0, 235)
    # خط المنتصف العمودي
    d.line([(cx, 0), (cx, h)], fill=red, width=2)
    # قمة الرأس والذقن
    top_y, chin_y = h * 0.08, h * 0.80
    d.line([(0, top_y), (w, top_y)], fill=amber, width=2)
    d.line([(0, chin_y), (w, chin_y)], fill=amber, width=2)
    # إطار الرأس البيضوي بين القمة والذقن
    d.ellipse([w * 0.235, top_y, w * 0.765, chin_y], outline=amber, width=2)
    # خط العينين ودائرتا العينين
    eye_y = h * 0.55
    d.line([(0, eye_y), (w, eye_y)], fill=green, width=2)
    r = w * 0.048
    for ex in (cx - w * 0.135, cx + w * 0.135):
        d.ellipse([ex - r, eye_y - r, ex + r, eye_y + r], outline=green, width=2)
    return out


def save_jpg(img: Image.Image, path: str, target_kb: float = 0) -> str:
    """
    حفظ JPG بأعلى جودة (97 وبدون تخفيض ألوان 4:4:4) مع ختم 300 DPI.
    عند طلب حجم مستهدف: تخفيض تدريجي للجودة حتى الوصول إليه.
    """
    if target_kb and target_kb > 0:
        limit = float(target_kb) * 1024
        quality = 97
        while True:
            img.save(path, "JPEG", quality=quality, subsampling=0, dpi=(DPI, DPI))
            if os.path.getsize(path) <= limit or quality <= 35:
                break
            quality -= 6
    else:
        img.save(path, "JPEG", quality=97, subsampling=0, dpi=(DPI, DPI))
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

def on_upload(img, mode, angle, zoom, pan_x, pan_y, size_label, custom_w, custom_h,
              color_label, custom_color, guides, brightness, contrast, saturation,
              sharpness, shadows, highlights, temperature):
    """عند رفع صورة: إزالة الخلفية مرة واحدة وتخزينها — الضبط اليدوي بعدها فوري."""
    if img is None:
        return None, None, ""
    try:
        bgr = normalize_input(img)
        result = get_creator()(bgr, change_bg_only=True)
        state = {"matting": result.standard}
    except Exception as e:
        return None, None, f"تعذّر تجهيز الصورة: {e}"
    preview = None
    if mode == MODES[1]:
        preview = manual_preview(state, angle, zoom, pan_x, pan_y, size_label,
                                 custom_w, custom_h, color_label, custom_color,
                                 guides, brightness, contrast, saturation,
                                 sharpness, shadows, highlights, temperature)
    return state, preview, "تم تجهيز الصورة وإزالة خلفيتها ✓ — جاهزة للصنع أو للضبط اليدوي."


def manual_preview(state, angle, zoom, pan_x, pan_y, size_label, custom_w, custom_h,
                   color_label, custom_color, guides, brightness, contrast,
                   saturation, sharpness, shadows, highlights, temperature):
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

    if mode == MODES[1]:
        # ---------- الوضع اليدوي ----------
        if not state or "matting" not in state:
            raise gr.Error("ارفع الصورة أولاً وانتظر رسالة «تم تجهيز الصورة» — الضبط اليدوي يعمل على الصورة المجهّزة.")
        matting = state["matting"]
        std_img = render_manual(matting, angle, zoom, pan_x, pan_y, width, height, rgb)
        hd_w, hd_h = round(width * HD_FACTOR), round(height * HD_FACTOR)
        hd_img = render_manual(matting, angle, zoom, pan_x, pan_y, hd_w, hd_h, rgb)
    else:
        # ---------- الوضع التلقائي ----------
        if img is None:
            raise gr.Error("ارفع صورة أولاً — اضغط على مربع الصورة واختر صورة من جهازك.")
        bgr = normalize_input(img)
        creator = get_creator()
        try:
            result = creator(bgr, size=(height, width), face_alignment=bool(align_face))
        except FaceError as e:
            n = getattr(e, "face_num", 0)
            if n == 0:
                raise gr.Error("لم يظهر أي وجه في الصورة. جرّب صورة أوضح، يكون فيها الوجه من الأمام ومضاءً جيداً — أو بدّل إلى الوضع اليدوي واضبط القص بنفسك.")
            raise gr.Error(f"الصورة تحتوي {n} وجوه — صورة الهوية يجب أن تحتوي وجهاً واحداً فقط. اختر صورة فيها شخص واحد.")
        except Exception as e:
            raise gr.Error(f"حدث خطأ أثناء المعالجة: {e}")
        std_img = composite_bg(result.standard, rgb)
        hd_img = composite_bg(result.hd, rgb)

    std_img = apply_color_adjustments(std_img, **adj)
    hd_img = apply_color_adjustments(hd_img, **adj)

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

    mode_name = "يدوي" if mode == MODES[1] else "تلقائي"
    status = (
        f"تم بنجاح ✓ ({mode_name}) — صورة {width}×{height} بكسل (300 DPI) بخلفية "
        f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X} وبصيغة JPG عالية الجودة. "
        f"حُفظت الملفات في مجلد outputs ويمكنك تنزيلها من الأسفل."
    )
    return np.array(std_img), np.array(hd_img), sheet_np, files, status


# ----------------------------- الواجهة -----------------------------

CUSTOM_CSS = """
.gradio-container { direction: rtl; font-family: Tahoma, 'Segoe UI', sans-serif; }
#app-title { text-align: center; padding: 18px 10px 4px; }
#app-title h1 { font-size: 30px; margin: 0; color: #0b6b3a; }
#app-title p { color: #555; margin: 6px 0 0; }
#make-btn { font-size: 19px !important; padding: 14px !important; }
footer { display: none !important; }
"""

S = SETTINGS

with gr.Blocks(title="صانع صور الهوية — الجزائر") as demo:
    gr.HTML("""
    <div id="app-title">
      <h1>🇩🇿 صانع صور الهوية — الجزائر</h1>
      <p>قصّ ذكي أو ضبط يدوي كامل، إزالة خلفية احترافية، وضبط ألوان — كل شيء على جهازك وبدون إنترنت، وإعداداتك محفوظة</p>
    </div>
    """)

    state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=5):
            in_image = gr.Image(label="١ — ارفع صورتك", type="numpy",
                                sources=["upload", "webcam"], height=300)
            mode_radio = gr.Radio(MODES, value=S["mode"], label="٢ — وضع العمل")
            size_radio = gr.Radio(list(SIZES.keys()), value=S["size"], label="٣ — مقاس الصورة")
            with gr.Row(visible=(S["size"] == "مقاس مخصص…")) as custom_size_row:
                custom_w = gr.Number(label="العرض (مم)", value=S["custom_w"], minimum=8, maximum=250)
                custom_h = gr.Number(label="الارتفاع (مم)", value=S["custom_h"], minimum=8, maximum=250)
            color_radio = gr.Radio(list(COLORS.keys()), value=S["color"], label="٤ — لون الخلفية")
            custom_color = gr.ColorPicker(label="اختر لونك", value=S["custom_color"],
                                          visible=(S["color"] == "لون مخصص…"))
            sheet_radio = gr.Radio(list(SHEETS.keys()), value=S["sheet"], label="٥ — ورقة الطباعة")

            with gr.Group(visible=(S["mode"] == MODES[0])) as auto_group:
                align_chk = gr.Checkbox(label="تصحيح ميلان الوجه تلقائياً (حسب العينين)", value=S["align"])
                kb_num = gr.Number(label="حجم ملف مستهدف (KB) — 0 يعني أعلى جودة", value=S["kb"], minimum=0)

            with gr.Group(visible=(S["mode"] == MODES[1])) as manual_group:
                gr.Markdown("**الضبط اليدوي** — حرّك المنزلقات وشاهد المعاينة مباشرة. الخطوط الملوّنة إرشادية فقط ولا تظهر في النتيجة.")
                angle_slider = gr.Slider(-15, 15, value=S["angle"], step=0.1,
                                         label="تدوير الصورة (درجة) — لاستقامة الكتفين والرأس")
                angle_num = gr.Number(label="الزاوية بدقة", value=S["angle"], minimum=-15, maximum=15, step=0.1)
                zoom_slider = gr.Slider(0.3, 4.0, value=S["zoom"], step=0.01,
                                        label="التقريب والتبعيد (1 = ملء الإطار)")
                pan_x = gr.Slider(-100, 100, value=S["pan_x"], step=1, label="إزاحة أفقية")
                pan_y = gr.Slider(-100, 100, value=S["pan_y"], step=1, label="إزاحة عمودية")
                guides_chk = gr.Checkbox(label="إظهار خطوط الإرشاد في المعاينة", value=S["guides"])
                reset_manual_btn = gr.Button("إعادة ضبط اليدوي (صفّارة الكل)", size="sm")

            with gr.Accordion("ضبط الألوان والإضاءة", open=False):
                bright_s = gr.Slider(0.5, 1.5, value=S["brightness"], step=0.01, label="السطوع")
                contrast_s = gr.Slider(0.5, 1.5, value=S["contrast"], step=0.01, label="التباين")
                satur_s = gr.Slider(0, 2.0, value=S["saturation"], step=0.01, label="تشبع الألوان")
                sharp_s = gr.Slider(0, 3.0, value=S["sharpness"], step=0.05, label="الحدة")
                shadows_s = gr.Slider(-100, 100, value=S["shadows"], step=1, label="الظلال (تفتيح المناطق الداكنة)")
                highlights_s = gr.Slider(-100, 100, value=S["highlights"], step=1, label="الإبرازات (المناطق الفاتحة)")
                temp_s = gr.Slider(-50, 50, value=S["temperature"], step=1, label="حرارة اللون (بارد ← دافئ)")
                reset_color_btn = gr.Button("إعادة ضبط الألوان", size="sm")

            make_btn = gr.Button("اصنع صورة الهوية", variant="primary", elem_id="make-btn")

        with gr.Column(scale=5):
            out_preview = gr.Image(label="معاينة الضبط اليدوي (بخطوط الإرشاد)", type="numpy",
                                   interactive=False, height=300,
                                   visible=(S["mode"] == MODES[1]))
            gr.Markdown(
                "دليل الخطوط: 🔴 **خط المنتصف** — مرّر منه الأنف &nbsp;·&nbsp; "
                "🟢 **خط العينين ودائرتاهما** — ضع العينين داخلهما &nbsp;·&nbsp; "
                "🟠 **البيضوي** — الرأس داخله، قمّته عند الخط العلوي وذقنه عند السفلي",
                visible=(S["mode"] == MODES[1]))
            out_std = gr.Image(label="الصورة القياسية", type="numpy", interactive=False, height=300)
            out_hd = gr.Image(label="نسخة عالية الدقة", type="numpy", interactive=False, height=220)
            out_sheet = gr.Image(label="ورقة الطباعة", type="numpy", interactive=False, height=220)
            out_status = gr.Textbox(label="الحالة", interactive=False)
            out_files = gr.File(label="تنزيل الملفات (JPG بدقة 300 DPI)", file_count="multiple")

    # ---------- حفظ الإعدادات تلقائياً عند أي تغيير ----------
    SETTINGS_KEYS = ["mode", "size", "custom_w", "custom_h", "color", "custom_color",
                     "sheet", "align", "kb", "angle", "zoom", "pan_x", "pan_y", "guides",
                     "brightness", "contrast", "saturation", "sharpness",
                     "shadows", "highlights", "temperature"]
    SETTINGS_INPUTS = [mode_radio, size_radio, custom_w, custom_h, color_radio,
                       custom_color, sheet_radio, align_chk, kb_num, angle_slider,
                       zoom_slider, pan_x, pan_y, guides_chk, bright_s, contrast_s,
                       satur_s, sharp_s, shadows_s, highlights_s, temp_s]

    def _persist(*vals):
        save_settings_dict(dict(zip(SETTINGS_KEYS, vals)))

    for _c in SETTINGS_INPUTS:
        _c.change(_persist, inputs=SETTINGS_INPUTS, outputs=None)

    # ---------- إظهار/إخفاء حسب الاختيارات ----------
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

    # ---------- الرفع والمعاينة اليدوية ----------
    PREVIEW_INPUTS = [angle_slider, zoom_slider, pan_x, pan_y, size_radio, custom_w,
                      custom_h, color_radio, custom_color, guides_chk, bright_s,
                      contrast_s, satur_s, sharp_s, shadows_s, highlights_s, temp_s]

    in_image.change(on_upload,
                    inputs=[in_image, mode_radio] + PREVIEW_INPUTS,
                    outputs=[state, out_preview, out_status])

    def _preview(state_v, *vals):
        return manual_preview(state_v, *vals)

    for _c in PREVIEW_INPUTS:
        _c.change(_preview, inputs=[state] + PREVIEW_INPUTS, outputs=out_preview)

    angle_slider.change(lambda v: v, inputs=angle_slider, outputs=angle_num)
    angle_num.change(lambda v: v, inputs=angle_num, outputs=angle_slider)

    def _reset_manual(*vals):
        d = dict(zip(SETTINGS_KEYS, vals))
        d.update({"angle": 0.0, "zoom": 1.0, "pan_x": 0, "pan_y": 0})
        save_settings_dict(d)
        return 0.0, 0.0, 1.0, 0, 0

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

    gr.HTML("""
    <p style="text-align:center;color:#888;font-size:12px">
      مبني على محرك <a href="https://github.com/Zeyi-Lin/HivisionIDPhotos" target="_blank">HivisionIDPhotos</a>
      مفتوح المصدر (رخصة Apache-2.0) — كشف الوجه MTCNN وإزالة الخلفية MODNet.
      الوضع التلقائي يطلب وجهاً واحداً واضحاً ويرفض ما عداه؛ والوضع اليدوي يترك القص بين يديك بالكامل.
    </p>
    """)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860,
                theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
                css=CUSTOM_CSS)
