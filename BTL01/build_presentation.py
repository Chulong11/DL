from pathlib import Path
import csv

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "baocao.pptx"

NAVY = RGBColor(15, 30, 55)
BLUE = RGBColor(27, 105, 174)
CYAN = RGBColor(37, 184, 213)
TEAL = RGBColor(19, 148, 133)
ORANGE = RGBColor(242, 153, 74)
WHITE = RGBColor(255, 255, 255)
LIGHT = RGBColor(243, 247, 251)
MID = RGBColor(102, 116, 135)
DARK = RGBColor(28, 39, 55)
GREEN = RGBColor(31, 157, 109)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def rect(slide, x, y, w, h, color, radius=False, line=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.color.rgb = line or color
    return shape


def textbox(slide, text, x, y, w, h, size=18, color=DARK, bold=False,
            font="Aptos", align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
            margin=0.06):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return box


def bullets(slide, items, x, y, w, h, size=19, color=DARK, spacing=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(spacing)
        p.line_spacing = 1.05
        p.text = "•  " + p.text
    return box


def base_slide(title, section=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    rect(slide, 0, 0, 13.333, 0.12, CYAN)
    if section:
        textbox(slide, section.upper(), 0.65, 0.32, 3.2, 0.3, 10, BLUE, True)
    textbox(slide, title, 0.65, 0.65 if section else 0.45, 12.0, 0.65, 28, NAVY, True)
    rect(slide, 0.65, 1.32, 0.72, 0.05, CYAN)
    return slide


def footer(slide, n):
    textbox(slide, "FUNCTION APPROXIMATION MLP", 0.65, 7.12, 4.0, 0.22, 9, MID, True)
    textbox(slide, f"{n:02d}", 12.2, 7.08, 0.48, 0.25, 10, MID, True, align=PP_ALIGN.RIGHT)


def add_picture_fit(slide, path, x, y, w, h):
    from PIL import Image
    path = Path(path)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w-pw)/2), Inches(y + (h-ph)/2),
                                    width=Inches(pw), height=Inches(ph))


def metric_card(slide, x, y, w, label, value, accent=BLUE, note=None):
    rect(slide, x, y, w, 1.18, LIGHT, True, LIGHT)
    rect(slide, x, y, 0.08, 1.18, accent)
    textbox(slide, label.upper(), x+0.25, y+0.16, w-0.4, 0.25, 10, MID, True)
    textbox(slide, value, x+0.25, y+0.43, w-0.4, 0.45, 23, NAVY, True)
    if note:
        textbox(slide, note, x+0.25, y+0.88, w-0.4, 0.2, 9, MID)


def add_table(slide, headers, rows, x, y, w, h, widths=None, font_size=12):
    table = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for i, frac in enumerate(widths):
            table.columns[i].width = Inches(w * frac)
    for c, header in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = header
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font_size); p.font.bold = True; p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            cell.margin_left = cell.margin_right = Inches(0.05)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size); p.font.color.rgb = DARK
                p.alignment = PP_ALIGN.CENTER
    return table


# 1 — Cover
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
rect(slide, 0, 0, 0.18, 7.5, CYAN)
rect(slide, 9.7, 0, 3.633, 7.5, RGBColor(20, 50, 83))
for x, y, s, c in [(10.4, 0.8, 1.25, CYAN), (11.5, 2.0, .65, BLUE), (10.25, 3.0, 1.75, TEAL), (11.7, 5.0, .9, ORANGE)]:
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(s), Inches(s))
    shp.fill.solid(); shp.fill.fore_color.rgb = c; shp.fill.transparency = 18; shp.line.fill.background()
textbox(slide, "DEEP LEARNING", 0.85, 0.75, 5.5, 0.35, 13, CYAN, True)
textbox(slide, "FUNCTION\nAPPROXIMATION\nWITH MLP", 0.85, 1.35, 8.5, 2.6, 34, WHITE, True)
textbox(slide, "Xấp xỉ hàm Gaussian hai chiều bằng mạng perceptron đa tầng", 0.9, 4.25, 7.8, 0.7, 19, RGBColor(205, 220, 236))
textbox(slide, "NHÓM THỰC HIỆN  •  2026", 0.9, 6.55, 5.0, 0.35, 12, WHITE, True)

# 2 — Team and agenda
slide = base_slide("Nhóm thực hiện & nội dung", "Tổng quan")
names = ["Nhữ Dương Luân — 3123411184", "Trần Lê Tuấn Kiệt — 3123411172",
         "Chu Thế Long — 3123411178", "Nguyễn Mai Hiển — 3123411093"]
rect(slide, .65, 1.7, 5.65, 4.85, LIGHT, True, LIGHT)
textbox(slide, "THÀNH VIÊN", .95, 2.0, 3.0, .3, 12, BLUE, True)
for i, name in enumerate(names):
    textbox(slide, f"{i+1:02d}", .95, 2.55+i*.78, .45, .35, 13, CYAN, True)
    textbox(slide, name, 1.55, 2.52+i*.78, 4.25, .42, 16, NAVY, i == 0)
textbox(slide, "NỘI DUNG", 7.05, 1.95, 3.0, .3, 12, BLUE, True)
agenda = ["Bài toán & dữ liệu", "Kiến trúc và quy trình huấn luyện", "Ba thí nghiệm A / B / C", "Kết luận & hướng phát triển"]
for i, item in enumerate(agenda):
    rect(slide, 7.05, 2.48+i*.83, 5.2, .62, WHITE, True, RGBColor(218, 227, 238))
    textbox(slide, str(i+1), 7.25, 2.62+i*.83, .3, .3, 14, BLUE, True, align=PP_ALIGN.CENTER)
    textbox(slide, item, 7.75, 2.59+i*.83, 4.15, .34, 16, DARK)

# 3 — Problem
slide = base_slide("Bài toán xấp xỉ hàm", "01 · Bài toán")
textbox(slide, "MỤC TIÊU", .75, 1.72, 2.0, .3, 12, BLUE, True)
textbox(slide, "Xây dựng MLP bằng PyTorch để học ánh xạ", .75, 2.08, 5.4, .55, 21, NAVY, True)
rect(slide, .75, 2.85, 5.2, 1.05, NAVY, True, NAVY)
textbox(slide, "f(x₁, x₂)  →  ŷ", 1.1, 3.05, 4.5, .55, 29, WHITE, True, align=PP_ALIGN.CENTER)
bullets(slide, ["Đầu vào: hai biến liên tục x₁, x₂", "Đầu ra: một giá trị Gaussian thực", "Loss: Mean Squared Error (MSE)", "Test chỉ dùng cho đánh giá cuối"], .75, 4.25, 5.6, 2.1, 17)
add_picture_fit(slide, ROOT/"plots"/"gaussian_surface.png", 6.6, 1.6, 6.1, 4.9)
textbox(slide, "Bề mặt hàm Gaussian mục tiêu", 8.25, 6.35, 3.4, .25, 11, MID, False, align=PP_ALIGN.CENTER)

# 4 — Data
slide = base_slide("Chuẩn bị dữ liệu", "01 · Bài toán")
add_picture_fit(slide, ROOT/"plots"/"dataset_scatter.png", .65, 1.55, 6.1, 4.95)
textbox(slide, "Gaussian 2D", 7.15, 1.62, 2.2, .35, 13, BLUE, True)
textbox(slide, "f(x) = 1/(2π√det(C)) · exp[−½(x−μ)ᵀC⁻¹(x−μ)]", 7.15, 2.05, 5.45, .85, 17, NAVY, True)
bullets(slide, ["μ = [0, 0]ᵀ", "C = [[0.3, −0.5], [−0.5, 2]]", "x₁ ∈ [−2, 2]  |  x₂ ∈ [−4, 4]", "Dữ liệu được xáo trộn với seed = 42"], 7.15, 3.1, 5.4, 1.95, 16)
metric_card(slide, 7.15, 5.35, 1.65, "Train", "2.000", TEAL)
metric_card(slide, 8.98, 5.35, 1.65, "Validation", "400", BLUE)
metric_card(slide, 10.81, 5.35, 1.65, "Test", "100", ORANGE)

# 5 — Architecture
slide = base_slide("Kiến trúc MLP", "02 · Phương pháp")
textbox(slide, "Kiến trúc tổng quát", .75, 1.65, 3.0, .35, 13, BLUE, True)
layers = [(1.0, "INPUT", "2", BLUE), (4.0, "HIDDEN", "H", TEAL), (7.0, "HIDDEN", "…", TEAL), (10.0, "OUTPUT", "1", ORANGE)]
for i, (x, label, count, color) in enumerate(layers):
    if i < len(layers)-1:
        textbox(slide, "→", x+2.05, 3.0, .65, .5, 30, MID, True, align=PP_ALIGN.CENTER)
    rect(slide, x, 2.35, 1.65, 2.0, color, True, color)
    textbox(slide, label, x+0.12, 2.62, 1.4, .3, 11, WHITE, True, align=PP_ALIGN.CENTER)
    textbox(slide, count, x+0.15, 3.02, 1.35, .55, 29, WHITE, True, align=PP_ALIGN.CENTER)
    textbox(slide, "neurons" if label != "OUTPUT" else "value", x+0.15, 3.7, 1.35, .25, 10, WHITE, False, align=PP_ALIGN.CENTER)
rect(slide, .9, 5.05, 11.55, 1.15, LIGHT, True, LIGHT)
facts = [("Activation", "Tanh"), ("Optimizer", "Adam"), ("Loss", "MSE"), ("Learning rate", "0.001")]
for i, (k, v) in enumerate(facts):
    xx = 1.15 + i*2.85
    textbox(slide, k.upper(), xx, 5.28, 2.3, .23, 10, MID, True, align=PP_ALIGN.CENTER)
    textbox(slide, v, xx, 5.58, 2.3, .32, 18, NAVY, True, align=PP_ALIGN.CENTER)

# 6 — Training process
slide = base_slide("Quy trình huấn luyện & đánh giá", "02 · Phương pháp")
steps = [("01", "Khởi tạo", "nn.Module + DataLoader"), ("02", "Huấn luyện", "zero_grad → backward → step"),
         ("03", "Đánh giá", "eval() + no_grad()"), ("04", "Chọn model", "Min Validation MSE"),
         ("05", "Lưu kết quả", "Checkpoint + history")]
for i, (num, title, desc) in enumerate(steps):
    x = .65 + i*2.48
    rect(slide, x, 2.05, 2.08, 2.25, LIGHT, True, RGBColor(220, 229, 239))
    textbox(slide, num, x+.18, 2.27, .45, .32, 12, CYAN, True)
    textbox(slide, title, x+.18, 2.75, 1.72, .4, 17, NAVY, True)
    textbox(slide, desc, x+.18, 3.28, 1.72, .65, 13, MID)
    if i < 4:
        textbox(slide, "→", x+2.06, 2.88, .42, .45, 22, BLUE, True, align=PP_ALIGN.CENTER)
metric_card(slide, 1.1, 5.05, 3.25, "Epoch", "100", BLUE, "mỗi lượt chạy")
metric_card(slide, 5.05, 5.05, 3.25, "Batch size", "32", TEAL, "mini-batch")
metric_card(slide, 9.0, 5.05, 3.25, "Seeds", "11 · 22 · 33", ORANGE, "đánh giá độ ổn định")

# 7 — Experimental design
slide = base_slide("Thiết kế ba thí nghiệm", "03 · Thực nghiệm")
cards = [
    ("A", "SỐ MẪU", "N = 200 → 2.000", "30 lượt chạy", BLUE),
    ("B", "SỐ NEURON", "H = 1 → 10", "30 lượt chạy", TEAL),
    ("C", "SỐ TẦNG ẨN", "L = 1 → 4", "12 lượt chạy", ORANGE),
]
for i, (letter, title, variable, runs, color) in enumerate(cards):
    x = .75 + i*4.16
    rect(slide, x, 1.85, 3.7, 4.25, LIGHT, True, LIGHT)
    rect(slide, x, 1.85, 3.7, .18, color)
    textbox(slide, letter, x+.28, 2.25, .72, .7, 32, color, True)
    textbox(slide, title, x+.28, 3.08, 3.0, .35, 13, MID, True)
    textbox(slide, variable, x+.28, 3.62, 3.0, .55, 21, NAVY, True)
    textbox(slide, runs, x+.28, 4.43, 3.0, .35, 15, color, True)
    textbox(slide, "Các yếu tố còn lại được giữ cố định", x+.28, 5.12, 2.95, .6, 12, MID)

# 8 — Experiment A
slide = base_slide("Thí nghiệm A · Ảnh hưởng số mẫu", "03 · Kết quả")
add_picture_fit(slide, ROOT/"results"/"plots"/"A"/"experiment_A_performance.png", .55, 1.55, 7.3, 4.9)
metric_card(slide, 8.15, 1.7, 4.45, "Validation MSE", "0,001482 → 0,000393", BLUE, "200 mẫu → 2.000 mẫu")
metric_card(slide, 8.15, 3.08, 4.45, "Mức giảm", "73,5%", GREEN, "sai số validation trung bình")
bullets(slide, ["Cả ba seed cùng xu hướng giảm MSE", "Khoảng cách train–validation nhỏ", "Best epoch chủ yếu 97–100", "Chưa thấy overfitting rõ ràng"], 8.15, 4.55, 4.45, 1.75, 14, DARK, 4)

# 9 — Experiment B
slide = base_slide("Thí nghiệm B · Ảnh hưởng số neuron", "03 · Kết quả")
add_picture_fit(slide, ROOT/"results"/"plots"/"B"/"experiment_B_performance.png", .55, 1.5, 7.4, 4.95)
metric_card(slide, 8.12, 1.68, 4.5, "Cấu hình tốt nhất", "9 neuron", TEAL, "Mean Val MSE = 0,000374")
metric_card(slide, 8.12, 3.06, 4.5, "Cải thiện so với 1 neuron", "86,2%", GREEN, "0,002710 → 0,000374")
bullets(slide, ["1–2 neuron: dấu hiệu underfitting", "Xu hướng giảm không hoàn toàn đơn điệu", "5 neuron có độ phân tán cao", "9 và 10 neuron chưa cho thấy overfitting"], 8.12, 4.53, 4.5, 1.8, 14, DARK, 4)

# 10 — Experiment C performance
slide = base_slide("Thí nghiệm C · Ảnh hưởng độ sâu", "03 · Kết quả")
add_picture_fit(slide, ROOT/"results"/"plots"/"C"/"experiment_C_performance.png", .55, 1.5, 7.4, 4.95)
metric_card(slide, 8.12, 1.62, 4.5, "Validation MSE", "3,93×10⁻⁴ → 5,03×10⁻⁶", ORANGE, "1 tầng → 4 tầng")
metric_card(slide, 8.12, 3.0, 4.5, "Bước cải thiện lớn nhất", "1 → 2 tầng", GREEN, "giảm khoảng 95,8%")
metric_card(slide, 8.12, 4.38, 4.5, "Thời gian trung bình", "8,92 s → 13,22 s", BLUE, "1 tầng → 4 tầng / một lượt")

# 11 — Experiment C convergence
slide = base_slide("Thí nghiệm C · Hội tụ và độ ổn định", "03 · Kết quả")
add_picture_fit(slide, ROOT/"results"/"plots"/"C"/"experiment_C_learning_curves.png", .55, 1.5, 8.05, 5.15)
rect(slide, 8.95, 1.7, 3.7, 4.7, LIGHT, True, LIGHT)
textbox(slide, "NHẬN XÉT", 9.28, 2.02, 2.0, .3, 12, BLUE, True)
bullets(slide, ["2–4 tầng hội tụ nhanh và đạt loss thấp", "Train/validation bám sát nhau", "Best epoch: 93–100", "Chưa thấy phân kỳ hoặc overfitting rõ", "Độ sâu tăng làm thời gian train tăng"], 9.22, 2.55, 3.05, 3.1, 14, DARK, 7)

# 12 — Independent test evaluation
slide = base_slide("Đánh giá cuối trên tập test độc lập", "03 · Kết quả")
add_picture_fit(slide, ROOT/"results"/"plots"/"test"/"test_true_vs_pred.png", .5, 1.48, 6.15, 5.15)
add_picture_fit(slide, ROOT/"results"/"plots"/"test"/"true_vs_approximated_surface.png", 6.8, 1.5, 5.9, 2.55)
metric_card(slide, 6.95, 4.25, 1.72, "Test MSE", "3,26×10⁻⁶", BLUE)
metric_card(slide, 8.83, 4.25, 1.72, "Test MAE", "0,001424", TEAL)
metric_card(slide, 10.71, 4.25, 1.72, "Test RMSE", "0,001806", ORANGE)
textbox(slide, "R² = 0,999079  •  100 mẫu test  •  chỉ đánh giá một lần sau khi chọn model bằng validation",
        7.02, 5.72, 5.3, .62, 13, NAVY, True, align=PP_ALIGN.CENTER)

# 13 — C stability + summary table
slide = base_slide("Tổng hợp kết quả", "04 · Kết luận")
headers = ["Thí nghiệm", "Biến khảo sát", "Cấu hình nổi bật", "Mean Val MSE", "Kết luận"]
rows = [
    ["A", "Số mẫu", "N = 2.000", "0,000393", "Nhiều dữ liệu → tốt hơn"],
    ["B", "Số neuron", "H = 9", "0,000374", "Cân bằng tốt nhất"],
    ["C", "Số tầng", "L = 4", "0,000005", "Sai số thấp nhất"],
]
add_table(slide, headers, rows, .65, 1.65, 12.05, 2.1, [.12,.18,.19,.17,.34], 12)
textbox(slide, "KẾT LUẬN CHÍNH", .75, 4.18, 2.7, .3, 12, BLUE, True)
bullets(slide, ["Quy mô dữ liệu, chiều rộng và độ sâu đều ảnh hưởng rõ đến khả năng xấp xỉ.",
                "Trong phạm vi khảo sát, tăng độ sâu tạo mức cải thiện mạnh nhất.",
                "Model chọn bằng validation đạt Test MSE 3,26×10⁻⁶ và R² 0,999079.",
                "Không có bằng chứng overfitting rõ ràng; best epoch thường gần cuối lịch huấn luyện."],
        .75, 4.62, 11.8, 1.65, 17, DARK, 8)

# 14 — Recommendations
slide = base_slide("Hạn chế & hướng phát triển", "04 · Kết luận")
left = ["Lịch huấn luyện cố định 100 epoch", "Số mẫu tăng làm tăng số bước cập nhật", "Chưa báo cáo MSE trên test", "Mỗi cấu hình mới chỉ có 3 seed"]
right = ["Dùng early stopping và learning-rate scheduler", "Chuẩn hóa số optimizer steps khi so sánh", "Mở rộng seed và kiểm định thống kê", "Demo: generate_data.py → experiments.py → evaluate_best_model.py"]
for x, title, items, color in [(.75, "HẠN CHẾ", left, ORANGE), (6.85, "ĐỀ XUẤT", right, TEAL)]:
    rect(slide, x, 1.75, 5.75, 4.75, LIGHT, True, LIGHT)
    rect(slide, x, 1.75, 5.75, .16, color)
    textbox(slide, title, x+.35, 2.18, 2.5, .35, 13, color, True)
    for i, item in enumerate(items):
        textbox(slide, f"{i+1:02d}", x+.35, 2.92+i*.78, .42, .3, 12, color, True)
        textbox(slide, item, x+1.0, 2.86+i*.78, 4.15, .55, 15, DARK)

# 15 — Closing
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
rect(slide, 0, 0, 13.333, .14, CYAN)
textbox(slide, "KẾT LUẬN", .85, .8, 2.2, .35, 12, CYAN, True)
textbox(slide, "MLP xấp xỉ hiệu quả\nhàm Gaussian hai chiều", .85, 1.55, 8.5, 1.45, 31, WHITE, True)
textbox(slide, "Kiến trúc 4 tầng ẩn × 10 neuron đạt Test MSE ≈ 3,26 × 10⁻⁶\nvà R² = 0,999079 trên 100 mẫu test độc lập.", .9, 3.35, 8.8, 1.05, 19, RGBColor(207, 221, 236))
textbox(slide, "CẢM ƠN!", .9, 5.9, 3.4, .6, 27, CYAN, True)
textbox(slide, "Q & A", 10.2, 2.45, 2.2, 1.0, 39, WHITE, True, align=PP_ALIGN.CENTER)
rect(slide, 10.1, 3.58, 2.4, .07, CYAN)

for i, slide in enumerate(prs.slides, start=1):
    if i not in (1, 15):
        footer(slide, i)

prs.core_properties.title = "Function Approximation with MLP"
prs.core_properties.subject = "Xấp xỉ hàm Gaussian 2D bằng PyTorch MLP"
prs.core_properties.author = "Nhóm Deep Learning"
prs.core_properties.keywords = "Deep Learning, MLP, Gaussian, PyTorch"
prs.save(OUT)
print(f"Created {OUT} with {len(prs.slides)} slides")
