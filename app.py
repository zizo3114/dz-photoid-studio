# -*- coding: utf-8 -*-
"""
صانع صور الهوية — الجزائر (نسخة سطح المكتب)

واجهة عربية مبنية على محرك مشروع HivisionIDPhotos مفتوح المصدر
(https://github.com/Zeyi-Lin/HivisionIDPhotos — رخصة Apache-2.0).
المحرك: كشف وجه MTCNN بعتبات صارمة + قاعدة «وجه واحد فقط»،
وقصّ ذكي بنسب رأس قياسية، وإزالة خلفية بنموذج MODNet المهيّأ لصور الهوية.
كل المعالجة تتم على جهاز المستخدم — بدون إنترنت بعد التثبيت الأول.
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw
import gradio as gr

from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from hivision.error import FaceError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 300

# ----------------------------- المقاسات والألوان -----------------------------

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


def save_jpg(img: Image.Image, path: str, target_kb: float = 0) -> str:
    """حفظ JPG بختم 300 DPI — مع ضغط تدريجي إذا طُلب حجم ملف مستهدف."""
    if target_kb and target_kb > 0:
        limit = float(target_kb) * 1024
        quality = 95
        while quality >= 30:
            img.save(path, "JPEG", quality=quality, dpi=(DPI, DPI))
            if os.path.getsize(path) <= limit or quality == 30:
                break
            quality -= 7
    else:
        img.save(path, "JPEG", quality=95, dpi=(DPI, DPI))
    return path


def build_sheet(tile: Image.Image, sheet_cfg) -> Image.Image:
    """تجميع نسخ الصورة على ورقة طباعة بقياس حقيقي (300 DPI) مع خطوط قص خفيفة."""
    (sheet_w, sheet_h), cols, rows = sheet_cfg
    tw, th = tile.size
    gap = 10
    block_w = cols * tw + (cols - 1) * gap
    block_h = rows * th + (rows - 1) * gap
    x0 = (sheet_w - block_w) // 2
    y0 = (sheet_h - block_h) // 2
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (tw + gap)
            y = y0 + r * (th + gap)
            sheet.paste(tile, (x, y))
            draw.rectangle([x, y, x + tw - 1, y + th - 1], outline=(200, 200, 200), width=1)
    return sheet


# ----------------------------- الدالة الرئيسية -----------------------------

def make_id_photo(img, size_label, custom_w_mm, custom_h_mm,
                  color_label, custom_color, sheet_label,
                  align_face, target_kb):
    if img is None:
        raise gr.Error("ارفع صورة أولاً — اضغط على مربع الصورة واختر صورة من جهازك.")

    # المقاس
    size = SIZES.get(size_label)
    if size is None:
        try:
            w_px, h_px = mm_to_px(custom_w_mm), mm_to_px(custom_h_mm)
        except Exception:
            raise gr.Error("المقاس المخصص غير صحيح — أدخل العرض والارتفاع بالمليمتر.")
        if not (100 <= w_px <= 3000 and 100 <= h_px <= 3000):
            raise gr.Error("المقاس المخصص خارج الحدود المعقولة (بين 8 و250 مم تقريباً).")
        size = (h_px, w_px)
    height, width = size

    # اللون
    rgb = COLORS.get(color_label)
    if rgb is None:
        rgb = hex_to_rgb(custom_color)

    bgr = normalize_input(img)
    creator = get_creator()
    try:
        result = creator(
            bgr,
            size=(height, width),
            face_alignment=bool(align_face),
        )
    except FaceError as e:
        n = getattr(e, "face_num", 0)
        if n == 0:
            raise gr.Error("لم يظهر أي وجه في الصورة. جرّب صورة أوضح، يكون فيها الوجه من الأمام ومضاءً جيداً.")
        raise gr.Error(f"الصورة تحتوي {n} وجوه — صورة الهوية يجب أن تحتوي وجهاً واحداً فقط. اختر صورة فيها شخص واحد.")
    except Exception as e:
        raise gr.Error(f"حدث خطأ أثناء المعالجة: {e}")

    std_img = composite_bg(result.standard, rgb)
    hd_img = composite_bg(result.hd, rgb)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    files = []
    std_path = save_jpg(std_img, os.path.join(OUT_DIR, f"photo_{stamp}.jpg"), target_kb or 0)
    files.append(std_path)
    hd_path = save_jpg(hd_img, os.path.join(OUT_DIR, f"photo_{stamp}_hd.jpg"), 0)
    files.append(hd_path)

    sheet_np = None
    sheet_cfg = SHEETS.get(sheet_label)
    if sheet_cfg is not None:
        sheet = build_sheet(std_img, sheet_cfg)
        sheet_path = save_jpg(sheet, os.path.join(OUT_DIR, f"sheet_{stamp}.jpg"), 0)
        files.append(sheet_path)
        sheet_np = np.array(sheet)

    status = (
        f"تم بنجاح ✓ — صورة {width}×{height} بكسل (300 DPI) بخلفية "
        f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}. "
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

with gr.Blocks(title="صانع صور الهوية — الجزائر") as demo:
    gr.HTML("""
    <div id="app-title">
      <h1>🇩🇿 صانع صور الهوية — الجزائر</h1>
      <p>قصّ ذكي بمقاسات رسمية وإزالة خلفية احترافية — كل شيء يعمل على جهازك بدون إنترنت</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            in_image = gr.Image(label="١ — ارفع صورتك", type="numpy",
                                sources=["upload", "webcam"], height=340)
            size_radio = gr.Radio(list(SIZES.keys()), value="جزائرية رسمية 3.5 × 4.5 سم",
                                  label="٢ — مقاس الصورة")
            with gr.Row(visible=False) as custom_size_row:
                custom_w = gr.Number(label="العرض (مم)", value=35, minimum=8, maximum=250)
                custom_h = gr.Number(label="الارتفاع (مم)", value=45, minimum=8, maximum=250)
            color_radio = gr.Radio(list(COLORS.keys()), value="أبيض", label="٣ — لون الخلفية")
            custom_color = gr.ColorPicker(label="اختر لونك", value="#ffffff", visible=False)
            sheet_radio = gr.Radio(list(SHEETS.keys()), value="8 صور — ورقة 10 × 15 سم",
                                   label="٤ — ورقة الطباعة")
            with gr.Accordion("خيارات متقدمة", open=False):
                align_chk = gr.Checkbox(label="تصحيح ميلان الوجه تلقائياً", value=True)
                kb_num = gr.Number(label="حجم ملف مستهدف (KB) — 0 يعني أفضل جودة", value=0, minimum=0)
            make_btn = gr.Button("اصنع صورة الهوية", variant="primary", elem_id="make-btn")

        with gr.Column(scale=5):
            out_std = gr.Image(label="الصورة القياسية", type="numpy", interactive=False, height=340)
            out_hd = gr.Image(label="نسخة عالية الدقة", type="numpy", interactive=False, height=240)
            out_sheet = gr.Image(label="ورقة الطباعة", type="numpy", interactive=False, height=240)
            out_status = gr.Textbox(label="الحالة", interactive=False)
            out_files = gr.File(label="تنزيل الملفات (JPG بدقة 300 DPI)", file_count="multiple")

    size_radio.change(lambda v: gr.update(visible=(v == "مقاس مخصص…")),
                      inputs=size_radio, outputs=custom_size_row)
    color_radio.change(lambda v: gr.update(visible=(v == "لون مخصص…")),
                       inputs=color_radio, outputs=custom_color)

    make_btn.click(
        make_id_photo,
        inputs=[in_image, size_radio, custom_w, custom_h,
                color_radio, custom_color, sheet_radio,
                align_chk, kb_num],
        outputs=[out_std, out_hd, out_sheet, out_files, out_status],
    )

    gr.HTML("""
    <p style="text-align:center;color:#888;font-size:12px">
      مبني على محرك <a href="https://github.com/Zeyi-Lin/HivisionIDPhotos" target="_blank">HivisionIDPhotos</a>
      مفتوح المصدر (رخصة Apache-2.0) — كشف الوجه MTCNN وإزالة الخلفية MODNet.
      البرنامج يطلب وجهاً واحداً واضحاً في الصورة، ويرفض الصور بدون وجه بدل قصّها خطأً.
    </p>
    """)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860,
                theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
                css=CUSTOM_CSS)
