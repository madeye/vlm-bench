#!/usr/bin/env python3
"""Import real phone screenshots into yolo_data as train samples.

- Dedupes by content hash
- Drops near-blank loading screens (mean luminance > 250)
- Positives from real_shots/meta.json get YOLO labels; everything else
  is an empty-label hard negative
- Writes N augmented copies per unique image (color jitter / slight crop /
  horizontal flip for negatives only — flip would break asymmetric skip
  positions on positives unless box is mirrored)

Usage:
  python import_real_shots.py              # default: 6 augs/image → train
  python import_real_shots.py --augs 10
  python import_real_shots.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil

from PIL import Image, ImageEnhance, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "real_shots", "raw")
META = os.path.join(ROOT, "real_shots", "meta.json")
NEG = os.path.join(ROOT, "neg_shots")
OUT_IMG = os.path.join(ROOT, "yolo_data", "images", "train")
OUT_LBL = os.path.join(ROOT, "yolo_data", "labels", "train")
PREFIX = "real_"


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_luma(im: Image.Image) -> float:
    g = im.convert("L").resize((32, 32))
    hist = g.histogram()
    total = sum(i * c for i, c in enumerate(hist))
    return total / (32 * 32)


def yolo_line(box, w, h) -> str:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2 / w
    cy = (y0 + y1) / 2 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def augment(im: Image.Image, box, rng: random.Random, allow_flip: bool):
    """Return (image, box_or_None). box is xyxy in pixel coords or None for neg."""
    w, h = im.size
    out = im.copy()
    # mild brightness/contrast/color
    if rng.random() < 0.8:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.8:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        out = ImageEnhance.Color(out).enhance(rng.uniform(0.8, 1.2))
    # slight random crop (keep ≥92% of each side)
    if rng.random() < 0.6:
        scale = rng.uniform(0.92, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        x0 = rng.randint(0, w - nw)
        y0 = rng.randint(0, h - nh)
        out = out.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.BILINEAR)
        if box is not None:
            # map box through crop+resize
            sx, sy = w / nw, h / nh
            bx0 = (box[0] - x0) * sx
            by0 = (box[1] - y0) * sy
            bx1 = (box[2] - x0) * sx
            by1 = (box[3] - y0) * sy
            # drop if crop removed most of the box
            if bx1 <= 0 or by1 <= 0 or bx0 >= w or by0 >= h:
                box = None  # became a negative accidentally — skip this aug
            else:
                box = (
                    max(0, bx0),
                    max(0, by0),
                    min(w, bx1),
                    min(h, by1),
                )
                if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    box = None
    # horizontal flip: only for pure negatives (skip labels are asymmetric)
    if allow_flip and box is None and rng.random() < 0.5:
        out = ImageOps.mirror(out)
    return out, box


def collect_sources():
    paths = []
    for d in (RAW, NEG):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                paths.append(os.path.join(d, name))
    return paths


def load_positives():
    pos = {}
    if os.path.isfile(META):
        for row in json.load(open(META)):
            pos[row["file"]] = row["gt"]
    return pos


def next_index() -> int:
    n = 0
    if os.path.isdir(OUT_IMG):
        for f in os.listdir(OUT_IMG):
            if f.startswith(PREFIX) and f.endswith(".jpg"):
                try:
                    n = max(n, int(f[len(PREFIX) : f.rfind(".")].split("_")[0]) + 1)
                except ValueError:
                    pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augs", type=int, default=6, help="augmented copies per unique image")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--replace",
        action="store_true",
        help="delete previous real_* train samples first",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    positives = load_positives()
    sources = collect_sources()
    print(f"sources: {len(sources)} files, positives labeled: {len(positives)}")

    # dedupe + filter blanks
    seen = {}
    kept = []
    blank = 0
    for p in sources:
        md = file_md5(p)
        if md in seen:
            continue
        im = Image.open(p).convert("RGB")
        if mean_luma(im) > 250:
            blank += 1
            continue
        seen[md] = p
        kept.append((p, im))
    print(f"unique non-blank: {len(kept)}  (dropped blank={blank}, dups={len(sources)-len(kept)-blank})")

    if args.dry_run:
        for p, _ in kept:
            kind = "POS" if os.path.basename(p) in positives else "neg"
            print(f"  [{kind}] {os.path.relpath(p, ROOT)}")
        print(f"would write ~{len(kept) * args.augs} train samples with prefix {PREFIX}")
        return

    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LBL, exist_ok=True)

    if args.replace:
        for d in (OUT_IMG, OUT_LBL):
            for f in os.listdir(d):
                if f.startswith(PREFIX):
                    os.remove(os.path.join(d, f))

    idx = next_index()
    n_pos = n_neg = 0
    for src, im0 in kept:
        base = os.path.basename(src)
        box0 = positives.get(base)
        # also match without path if meta uses just filename
        w, h = im0.size
        for a in range(args.augs):
            if a == 0:
                im, box = im0, box0
            else:
                im, box = augment(im0, box0, rng, allow_flip=(box0 is None))
                if box0 is not None and box is None:
                    # crop killed the positive; retry without crop-ish by using original
                    im, box = im0.copy(), box0
                    if rng.random() < 0.8:
                        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.9, 1.1))
            name = f"{PREFIX}{idx:05d}"
            img_path = os.path.join(OUT_IMG, f"{name}.jpg")
            lbl_path = os.path.join(OUT_LBL, f"{name}.txt")
            im.save(img_path, quality=90)
            with open(lbl_path, "w") as f:
                if box is not None:
                    f.write(yolo_line(box, w, h))
                    n_pos += 1
                else:
                    n_neg += 1
            idx += 1

    print(f"wrote pos_instances={n_pos} neg_instances={n_neg} under {OUT_IMG}")
    print(f"total train images now: {len([f for f in os.listdir(OUT_IMG) if f.endswith(('.jpg','.png'))])}")


if __name__ == "__main__":
    main()
