#!/usr/bin/env python3
"""Generate synthetic splash-ad screenshots with ground-truth skip-button rects.

Mimics common CN app splash ads: full-bleed creative, brand strip at bottom,
skip button in a corner (pill / countdown ring / bare text), plus distractor
CTAs (立即下载 / 摇一摇 / swipe-up) that must NOT be clicked.
"""
import json, math, os, random
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 2400
OUT = os.path.join(os.path.dirname(__file__), "testset")
CJK = "/System/Library/Fonts/STHeiti Medium.ttc"
LAT = "/System/Library/Fonts/Helvetica.ttc"

random.seed(42)

def font(sz, latin=False):
    return ImageFont.truetype(LAT if latin else CJK, sz)

PALETTES = [
    ((244, 240, 235), (40, 40, 45)),    # light cream bg
    ((24, 28, 44), (240, 240, 245)),    # dark navy bg
    ((250, 235, 238), (60, 30, 40)),    # pink light
    ((16, 60, 42), (235, 245, 240)),    # deep green
    ((248, 248, 250), (30, 30, 30)),    # near white
    ((45, 20, 65), (240, 235, 250)),    # purple dark
]

SLOGANS = [
    ("焕新一夏", "全场低至 5 折起"),
    ("新品首发", "限时抢购中"),
    ("品质生活", "从这里开始"),
    ("超级品牌日", "领券立减 100"),
    ("周年庆典", "会员专享福利"),
    ("春日上新", "满 299 减 60"),
]

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def draw_creative(d, img, bg, fg, idx):
    """Full-bleed ad creative: gradient, shapes, product card, big slogan."""
    top = bg
    bot = lerp(bg, (255, 255, 255) if sum(bg) > 380 else (0, 0, 0), 0.25)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=lerp(top, bot, y / H))
    # decorative circles
    for _ in range(5):
        r = random.randint(60, 220)
        x, y = random.randint(0, W), random.randint(200, H - 400)
        c = lerp(bg, fg, 0.12)
        d.ellipse([x - r, y - r, x + r, y + r], outline=c, width=6)
    # product card in middle
    cw, ch = 640, 760
    cx, cy = (W - cw) // 2, 700
    card_bg = lerp(bg, (255, 255, 255) if sum(bg) < 380 else (0, 0, 0), 0.10)
    d.rounded_rectangle([cx, cy, cx + cw, cy + ch], 40, fill=card_bg,
                        outline=lerp(bg, fg, 0.3), width=3)
    # fake product: box + swatches
    d.rounded_rectangle([cx + 120, cy + 90, cx + cw - 120, cy + 420], 24,
                        fill=lerp(bg, fg, 0.35))
    for i in range(3):
        d.ellipse([cx + 140 + i * 130, cy + 480, cx + 240 + i * 130, cy + 580],
                  fill=lerp(bg, fg, 0.2 + 0.15 * i))
    slogan, sub = SLOGANS[idx % len(SLOGANS)]
    f1, f2 = font(110), font(52)
    w1 = d.textlength(slogan, font=f1)
    d.text(((W - w1) / 2, 320), slogan, font=f1, fill=fg)
    w2 = d.textlength(sub, font=f2)
    d.text(((W - w2) / 2, 480), sub, font=f2, fill=lerp(bg, fg, 0.75))
    # brand strip at bottom (the "app logo" area of splash ads)
    d.rectangle([0, H - 220, W, H], fill=lerp(bg, (255, 255, 255) if sum(bg) > 380 else (10, 10, 14), 0.5))
    bf = font(64)
    brand = "极速视频"
    bw = d.textlength(brand, font=bf)
    d.ellipse([W / 2 - bw / 2 - 110, H - 175, W / 2 - bw / 2 - 20, H - 85], fill=(230, 80, 60))
    d.text((W / 2 - bw / 2 + 10, H - 165), brand, font=bf, fill=fg if sum(bg) < 380 else (40, 40, 45))
    return slogan


def add_distractor(d, kind, fg, bg):
    rects = []
    if kind == "cta":
        # big rounded CTA button lower third
        x0, y0, x1, y1 = W // 2 - 330, 1750, W // 2 + 330, 1880
        d.rounded_rectangle([x0, y0, x1, y1], 65, fill=(235, 60, 60))
        t = "立即下载"
        f = font(72)
        tw = d.textlength(t, font=f)
        d.text(((W - tw) / 2, y0 + 22), t, font=f, fill=(255, 255, 255))
        rects.append([x0, y0, x1, y1])
    elif kind == "shake":
        x0, y0, x1, y1 = W // 2 - 280, 1700, W // 2 + 280, 1900
        d.rounded_rectangle([x0, y0, x1, y1], 40, outline=(255, 255, 255), width=5)
        f = font(58)
        t = "摇一摇 跳转详情页"
        tw = d.textlength(t, font=f)
        d.text(((W - tw) / 2, y0 + 105), t, font=f, fill=(255, 255, 255))
        d.ellipse([W / 2 - 45, y0 + 20, W / 2 + 45, y0 + 95], outline=(255, 255, 255), width=5)
        rects.append([x0, y0, x1, y1])
    elif kind == "swipe":
        f = font(56)
        t = "向上滑动 了解更多"
        tw = d.textlength(t, font=f)
        y = 1980
        d.text(((W - tw) / 2, y), t, font=f, fill=(255, 255, 255))
        d.polygon([(W / 2 - 40, y - 40), (W / 2 + 40, y - 40), (W / 2, y - 100)],
                  fill=(255, 255, 255))
        rects.append([W / 2 - tw / 2, y - 100, W / 2 + tw / 2, y + 70])
    return rects


def skip_pill(d, label, cx, cy, pad_x, pad_y, fsz, style, latin=False):
    """Draw pill-style skip button centered at (cx, cy); returns rect."""
    f = font(fsz, latin)
    tw = d.textlength(label, font=f)
    x0, y0 = cx - tw / 2 - pad_x, cy - fsz / 2 - pad_y
    x1, y1 = cx + tw / 2 + pad_x, cy + fsz / 2 + pad_y
    r = (y1 - y0) / 2
    if style == "dark":
        d.rounded_rectangle([x0, y0, x1, y1], r, fill=(0, 0, 0, 140))
        col = (255, 255, 255)
    elif style == "light":
        d.rounded_rectangle([x0, y0, x1, y1], r, fill=(255, 255, 255, 210))
        col = (40, 40, 45)
    elif style == "outline":
        d.rounded_rectangle([x0, y0, x1, y1], r, outline=(255, 255, 255), width=3)
        col = (255, 255, 255)
    else:  # bare text
        col = (255, 255, 255)
    d.text((cx - tw / 2, cy - fsz / 2 - fsz * 0.12), label, font=f, fill=col)
    return [int(x0), int(y0), int(x1), int(y1)]


def skip_ring(d, cx, cy, radius, label, frac):
    """Countdown ring with text inside; returns rect."""
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0, 150))
    d.arc([cx - radius + 6, cy - radius + 6, cx + radius - 6, cy + radius - 6],
          start=-90, end=-90 + 360 * frac, fill=(255, 200, 40), width=8)
    f = font(int(radius * 0.6))
    tw = d.textlength(label, font=f)
    d.text((cx - tw / 2, cy - radius * 0.38), label, font=f, fill=(255, 255, 255))
    return [cx - radius, cy - radius, cx + radius, cy + radius]


# (name, corner, maker) — corner: tr/tl/br/bc
CASES = []

def case(name, desc, fn):
    CASES.append((name, desc, fn))

M = 48  # corner margin

case("tr_pill_5", "top-right dark pill 跳过 5", lambda d: skip_pill(d, "跳过 5", W - 170, 140, 34, 22, 44, "dark"))
case("tr_pill_ad", "top-right dark pill 跳过广告", lambda d: skip_pill(d, "跳过广告", W - 190, 150, 30, 20, 42, "dark"))
case("tl_pill_3", "top-left dark pill 跳过 3", lambda d: skip_pill(d, "跳过 3", 170, 140, 34, 22, 44, "dark"))
case("br_pill", "bottom-right pill 跳过 above brand strip", lambda d: skip_pill(d, "跳过", W - 150, H - 320, 36, 24, 46, "dark"))
case("tr_ring", "top-right countdown ring 跳过", lambda d: skip_ring(d, W - 150, 160, 95, "跳过", 0.6))
case("tr_ring_num", "top-right countdown ring with 3s", lambda d: skip_ring(d, W - 150, 160, 90, "3", 0.4))
case("tr_bare_skip", "top-right bare text Skip", lambda d: skip_pill(d, "Skip", W - 130, 130, 18, 14, 46, "bare", latin=True))
case("tr_seg", "top-right segmented 5s|跳过", lambda d: skip_pill(d, "5s | 跳过", W - 190, 140, 30, 20, 42, "dark"))
case("tr_tiny", "top-right tiny pill 跳过", lambda d: skip_pill(d, "跳过", W - 110, 110, 20, 14, 30, "dark"))
case("tr_light", "top-right light pill 跳过 2", lambda d: skip_pill(d, "跳过 2", W - 170, 140, 32, 20, 42, "light"))
case("tr_outline", "top-right outline pill Skip Ad", lambda d: skip_pill(d, "Skip Ad", W - 175, 140, 28, 18, 40, "outline", latin=True))
case("bc_pill", "bottom-center small 跳过广告 >", lambda d: skip_pill(d, "跳过广告 >", W // 2, H - 300, 26, 16, 36, "dark"))
case("tl_ring", "top-left countdown ring 跳过", lambda d: skip_ring(d, 150, 160, 92, "跳过", 0.75))
case("br_bare", "bottom-right bare 跳过 3s", lambda d: skip_pill(d, "跳过 3s", W - 160, H - 300, 22, 14, 38, "bare"))
case("tr_wifi", "top-right pill with wifi预载 label nearby", lambda d: skip_pill(d, "跳过", W - 140, 150, 34, 22, 44, "dark"))

# distractor pairings: every case gets creative; some add cta/shake/swipe
DISTRACT = {
    "tr_pill_5": "cta", "tr_pill_ad": "shake", "tl_pill_3": "swipe",
    "br_pill": "cta", "tr_ring": "shake", "tr_ring_num": "cta",
    "tr_bare_skip": "swipe", "tr_seg": None, "tr_tiny": "cta",
    "tr_light": None, "tr_outline": "shake", "bc_pill": "cta",
    "tl_ring": None, "br_bare": "swipe", "tr_wifi": "cta",
}

# 15 base cases x varied palettes; plus 5 repeats with different palettes = 20
RUNS = [(n, dsc, fn, i) for i, (n, dsc, fn) in enumerate(CASES)]
extra = [0, 4, 8, 2, 5]  # rerun some tricky cases on different palettes
for j, k in enumerate(extra):
    n, dsc, fn = CASES[k]
    RUNS.append((n + "_alt", dsc + " (alt palette)", fn, len(CASES) + j))

os.makedirs(OUT, exist_ok=True)
meta = []
for name, desc, fn, idx in RUNS:
    base = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(base)
    bg, fg = PALETTES[idx % len(PALETTES)]
    draw_creative(d, base, bg, fg, idx)
    dk = DISTRACT.get(name.replace("_alt", ""))
    distract_rects = []
    # overlay layer with alpha for translucent pills
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    if dk:
        distract_rects = add_distractor(od, dk, fg, bg)
    gt = fn(od)
    img = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    fname = f"{name}.png"
    img.save(os.path.join(OUT, fname))
    if "wifi" in name:  # small caption under the wifi case's pill
        pass
    meta.append({"file": fname, "desc": desc, "w": W, "h": H,
                 "gt": gt, "distractors": distract_rects})

with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"generated {len(meta)} images in {OUT}")
