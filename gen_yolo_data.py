#!/usr/bin/env python3
"""Synthetic splash-ad dataset for training a YOLO skip-button detector.

Intentionally broader and differently-styled than testset/ (the VLM benchmark
eval set) so evaluating on it isn't pure memorization: random palettes, layout
composition, fonts, button styles/positions, hard negatives, and ~10% images
with no skip button at all.
"""
import json, math, os, random, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_data")
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
N_VAL = int(sys.argv[2]) if len(sys.argv) > 2 else 240

CJK_FONTS = ["/System/Library/Fonts/STHeiti Medium.ttc",
             "/System/Library/Fonts/STHeiti Light.ttc",
             "/System/Library/Fonts/Hiragino Sans GB.ttc"]
LAT_FONTS = ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]

def usable(paths):
    ok = []
    for p in paths:
        try:
            ImageFont.truetype(p, 20)
            ok.append(p)
        except Exception:
            pass
    return ok

CJK_FONTS, LAT_FONTS = usable(CJK_FONTS), usable(LAT_FONTS)

SKIP_TEXTS_CJK = ["跳过", "跳过广告", "跳过 {n}", "{n} 跳过", "{n}s 跳过", "{n} | 跳过",
                  "跳过广告 {n}", "跳过 >", "跳过广告 >", "关闭", "关闭广告"]
SKIP_TEXTS_LAT = ["Skip", "Skip Ad", "SKIP", "Skip {n}s", "{n}s Skip", "Skip >"]
CTA_TEXTS = ["立即下载", "立即购买", "点击了解更多", "查看详情", "立即领取", "去看看",
             "打开应用", "免费试用"]
HARD_NEG = ["{n}秒后自动进入", "广告 {n}s", "第 {n} 期", "倒计时 {n}", "WiFi 已预载",
            "摇一摇 跳转详情页", "向上滑动 了解更多", "跳转到第三方页面"]
SLOGANS = ["年中大促", "新品上市", "限时秒杀", "超值优惠", "品牌焕新", "开学季特惠",
           "会员日", "狂欢购物节", "精选好物", "焕然一新", "冬日暖心", "春夏新风尚"]

def rand_color(lo=0, hi=255):
    return tuple(random.randint(lo, hi) for _ in range(3))

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def contrast(bg):
    return (250, 250, 250) if sum(bg) < 380 else (25, 25, 30)

def draw_bg(d, W, H):
    style = random.random()
    c1, c2 = rand_color(), rand_color()
    horiz = random.random() < 0.3
    for i in range(H if not horiz else W):
        t = i / (H if not horiz else W)
        col = lerp(c1, c2, t)
        if horiz:
            d.line([(i, 0), (i, H)], fill=col)
        else:
            d.line([(0, i), (W, i)], fill=col)
    # random decorative shapes
    for _ in range(random.randint(3, 12)):
        x, y = random.randint(-100, W), random.randint(-100, H)
        r = random.randint(20, 300)
        col = rand_color()
        kind = random.random()
        if kind < 0.4:
            d.ellipse([x, y, x + r, y + r],
                      outline=col if random.random() < 0.5 else None,
                      fill=col if random.random() < 0.4 else None, width=random.randint(2, 10))
        elif kind < 0.8:
            d.rounded_rectangle([x, y, x + r, y + int(r * random.uniform(0.4, 1.4))],
                                random.randint(0, 40),
                                fill=col if random.random() < 0.6 else None,
                                outline=col, width=random.randint(1, 6))
        else:
            d.polygon([(x, y), (x + r, y + random.randint(-80, 80)),
                       (x + random.randint(0, r), y + r)], fill=col)
    return c1

def draw_content(d, W, H):
    """Random slogans / product cards / text blocks."""
    fg = rand_color()
    for _ in range(random.randint(1, 3)):
        f = ImageFont.truetype(random.choice(CJK_FONTS), random.randint(48, 140))
        t = random.choice(SLOGANS)
        x = random.randint(0, max(1, W - 500))
        y = random.randint(int(H * 0.08), int(H * 0.75))
        d.text((x, y), t, font=f, fill=fg)
    if random.random() < 0.6:  # product card
        cw = random.randint(300, W - 200)
        ch = random.randint(300, 900)
        cx = random.randint(0, W - cw)
        cy = random.randint(int(H * 0.15), max(int(H * 0.15) + 1, H - ch - 300))
        d.rounded_rectangle([cx, cy, cx + cw, cy + ch], random.randint(10, 50),
                            fill=rand_color(), outline=rand_color(), width=3)
    if random.random() < 0.7:  # bottom brand strip
        sh = random.randint(140, 260)
        d.rectangle([0, H - sh, W, H], fill=rand_color())
        f = ImageFont.truetype(random.choice(CJK_FONTS), random.randint(40, 70))
        d.text((random.randint(60, W // 2), H - sh + sh // 3),
               random.choice(SLOGANS), font=f, fill=rand_color())

def fmt(t):
    return t.replace("{n}", str(random.randint(1, 9)))

def draw_skip(od, W, H):
    """Draw one skip button; returns (x0,y0,x1,y1)."""
    kind = random.random()
    corner = random.choices(["tr", "tl", "br", "bl", "bc"], [0.45, 0.15, 0.2, 0.05, 0.15])[0]
    mx = random.randint(30, 130)
    my = random.randint(60, 260) if corner in ("tr", "tl") else random.randint(240, 420)
    if kind < 0.25:  # countdown ring
        r = random.randint(45, 110)
        cx = r + mx if corner in ("tl", "bl") else W - r - mx
        if corner == "bc":
            cx = W // 2 + random.randint(-100, 100)
        cy = r + my if corner in ("tr", "tl") else H - r - my
        alpha = random.randint(90, 200)
        od.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, alpha))
        od.arc([cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5], start=-90,
               end=-90 + 360 * random.uniform(0.15, 0.95),
               fill=random.choice([(255, 200, 40), (255, 255, 255), (80, 200, 255)]),
               width=random.randint(4, 10))
        label = random.choice(["跳过", str(random.randint(1, 9)), "跳过", "Skip"])
        latin = label.isascii()
        f = ImageFont.truetype(random.choice(LAT_FONTS if latin else CJK_FONTS), int(r * random.uniform(0.45, 0.7)))
        tw = od.textlength(label, font=f)
        od.text((cx - tw / 2, cy - r * 0.35), label, font=f, fill=(255, 255, 255))
        return (cx - r, cy - r, cx + r, cy + r)
    # text pill / bare text
    latin = random.random() < 0.25
    label = fmt(random.choice(SKIP_TEXTS_LAT if latin else SKIP_TEXTS_CJK))
    fsz = random.randint(26, 60)
    f = ImageFont.truetype(random.choice(LAT_FONTS if latin else CJK_FONTS), fsz)
    tw = od.textlength(label, font=f)
    pad_x = random.randint(12, 44)
    pad_y = random.randint(8, 28)
    bw, bh = tw + 2 * pad_x, fsz + 2 * pad_y
    if corner == "bc":
        x0 = W / 2 - bw / 2 + random.randint(-80, 80)
    elif corner in ("tl", "bl"):
        x0 = mx
    else:
        x0 = W - bw - mx
    y0 = my if corner in ("tr", "tl") else H - bh - my
    x1, y1 = x0 + bw, y0 + bh
    style = random.random()
    rad = (y1 - y0) / 2 if random.random() < 0.7 else random.randint(4, 18)
    if style < 0.45:
        od.rounded_rectangle([x0, y0, x1, y1], rad, fill=(0, 0, 0, random.randint(80, 190)))
        col = (255, 255, 255)
    elif style < 0.6:
        od.rounded_rectangle([x0, y0, x1, y1], rad, fill=(255, 255, 255, random.randint(150, 230)))
        col = (30, 30, 35)
    elif style < 0.75:
        od.rounded_rectangle([x0, y0, x1, y1], rad, outline=(255, 255, 255, 230),
                             width=random.randint(2, 5))
        col = (255, 255, 255)
    else:
        col = random.choice([(255, 255, 255), (240, 240, 240), (20, 20, 25)])
    od.text((x0 + pad_x, y0 + pad_y - fsz * 0.1), label, font=f, fill=col)
    return (x0, y0, x1, y1)

def draw_distractors(od, W, H):
    if random.random() < 0.55:  # big CTA
        bw = random.randint(400, 700)
        x0 = W / 2 - bw / 2 + random.randint(-60, 60)
        y0 = random.randint(int(H * 0.6), int(H * 0.8))
        od.rounded_rectangle([x0, y0, x0 + bw, y0 + random.randint(100, 160)],
                             random.randint(20, 80), fill=(random.randint(180, 255), random.randint(30, 90), random.randint(30, 90)))
        f = ImageFont.truetype(random.choice(CJK_FONTS), random.randint(52, 76))
        t = random.choice(CTA_TEXTS)
        tw = od.textlength(t, font=f)
        od.text((x0 + bw / 2 - tw / 2, y0 + 25), t, font=f, fill=(255, 255, 255))
    for _ in range(random.randint(0, 2)):  # hard negatives
        t = fmt(random.choice(HARD_NEG))
        f = ImageFont.truetype(random.choice(CJK_FONTS), random.randint(30, 54))
        x = random.randint(30, max(31, int(W - od.textlength(t, font=f) - 30)))
        y = random.randint(int(H * 0.05), int(H * 0.9))
        if random.random() < 0.4:
            tw = od.textlength(t, font=f)
            od.rounded_rectangle([x - 20, y - 12, x + tw + 20, y + f.size + 12],
                                 20, fill=(0, 0, 0, random.randint(60, 150)))
        od.text((x, y), t, font=f, fill=(255, 255, 255, random.randint(180, 255)))

def gen_one(path_img, path_lbl):
    W = random.choice([720, 1080, 1080, 1080, 1440])
    H = int(W * random.uniform(2.0, 2.35))
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    draw_bg(d, W, H)
    draw_content(d, W, H)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    draw_distractors(od, W, H)
    boxes = []
    if random.random() > 0.10:  # 10% negatives: no skip button
        boxes.append(draw_skip(od, W, H))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.2)))
    img.save(path_img, quality=90)
    with open(path_lbl, "w") as f:
        for (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

random.seed(7)
for split, n in [("train", N_TRAIN), ("val", N_VAL)]:
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    for i in range(n):
        gen_one(f"{OUT}/images/{split}/{split}_{i:05d}.jpg",
                f"{OUT}/labels/{split}/{split}_{i:05d}.txt")
    print(split, n, "done", flush=True)

with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: skip\n")
print("dataset ready:", OUT)
