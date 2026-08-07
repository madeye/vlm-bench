#!/usr/bin/env python3
"""Pass 4: native grounding prompt at 672 max-dim (latency/accuracy midpoint)."""
import json, os, subprocess, sys, time
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run_bench as rb

PROMPT = ("This is a screenshot of a mobile app splash advertisement. "
          "Locate the button used to skip or close the ad (usually labeled "
          "'跳过', 'Skip', '关闭' or shown as a countdown circle). "
          'Output only its bounding box in JSON: {"bbox_2d": [x1, y1, x2, y2]}')

IMGDIR = os.path.join(ROOT, "testset672")
RESULTS = os.path.join(ROOT, "results4.jsonl")
SUBSET = {"internvl3-1b", "internvl3-2b", "qwen25-vl-3b-q8mm", "qwen2-vl-2b"}

os.makedirs(IMGDIR, exist_ok=True)
meta = json.load(open(os.path.join(ROOT, "testset", "meta.json")))
for m in meta:
    dst = os.path.join(IMGDIR, m["file"])
    if not os.path.exists(dst):
        im = Image.open(os.path.join(ROOT, "testset", m["file"]))
        s = 672 / max(im.size)
        im.resize((int(im.width * s), int(im.height * s))).save(dst)

done = set()
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["model"], r["image"]))
            except Exception:
                pass

images = [m["file"] for m in meta]
out = open(RESULTS, "a")
for mid, mpath, mmpath in rb.MODELS:
    if mid not in SUBSET or not (rb.ready(mpath) and rb.ready(mmpath)):
        continue
    for img in images:
        if (mid, img) in done:
            continue
        t0 = time.time()
        try:
            p = subprocess.run(
                [rb.CLI, "-m", mpath, "--mmproj", mmpath,
                 "--image", os.path.join(IMGDIR, img),
                 "--temp", "0", "-n", "64", "-p", PROMPT],
                capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace")
            raw, rc = p.stdout.strip(), p.returncode
        except subprocess.TimeoutExpired:
            raw, rc = "", -9
        rec = {"model": mid, "image": img, "rc": rc,
               "wall_s": round(time.time() - t0, 2), "raw": raw, "timing": ""}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{mid}] {img} rc={rc} {rec['wall_s']}s :: {raw[:70]!r}", flush=True)
print("BENCH4 PASS COMPLETE")
