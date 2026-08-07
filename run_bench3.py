#!/usr/bin/env python3
"""Pass 3: native grounding prompt (JSON bbox_2d) at 896 max-dim input."""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run_bench as rb

PROMPT = ("This is a screenshot of a mobile app splash advertisement. "
          "Locate the button used to skip or close the ad (usually labeled "
          "'跳过', 'Skip', '关闭' or shown as a countdown circle). "
          'Output only its bounding box in JSON: {"bbox_2d": [x1, y1, x2, y2]}')

IMGDIR = os.path.join(ROOT, "testset896")
RESULTS = os.path.join(ROOT, "results3.jsonl")

done = set()
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["model"], r["image"]))
            except Exception:
                pass

meta = json.load(open(os.path.join(ROOT, "testset", "meta.json")))
images = [m["file"] for m in meta]
out = open(RESULTS, "a")
for mid, mpath, mmpath in rb.MODELS:
    if not (rb.ready(mpath) and rb.ready(mmpath)):
        print(f"[skip] {mid}", flush=True)
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
print("BENCH3 PASS COMPLETE")
