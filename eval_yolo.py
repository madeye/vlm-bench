#!/usr/bin/env python3
"""Evaluate the trained YOLO skip detector on the VLM benchmark test set.

Same metric as the VLM benchmark: top-confidence detection's center inside
the GT button rect = hit. Also reports distractor (CTA) misclicks and
latency. Usage: eval_yolo.py [weights] [conf]
"""
import json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/detect/yolo_runs/skip_v1/weights/best.pt"
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

from ultralytics import YOLO

model = YOLO(WEIGHTS)
meta = json.load(open(f"{ROOT}/testset/meta.json"))

hits = misses = none = bad = 0
lat = []
rows = []
for m in meta:
    img = f"{ROOT}/testset/{m['file']}"
    t0 = time.time()
    r = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0]
    lat.append(time.time() - t0)
    gt = m["gt"]
    if len(r.boxes) == 0:
        none += 1
        rows.append((m["file"], "no_det", None))
        continue
    b = max(r.boxes, key=lambda b: float(b.conf))
    x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hit = gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]
    isbad = any(d[0] <= cx <= d[2] and d[1] <= cy <= d[3] for d in m["distractors"])
    hits += hit
    misses += (not hit)
    bad += isbad
    rows.append((m["file"], "HIT" if hit else "miss", round(float(b.conf), 2)))

for f, s, c in rows:
    print(f"{f:<22} {s:<7} conf={c}")
print(f"\nhits={hits}/{len(meta)}  no_detection={none}  cta_misclicks={bad}")
print(f"median latency (host CPU/MPS, incl. pre/post): {sorted(lat)[len(lat)//2]*1000:.0f} ms")
