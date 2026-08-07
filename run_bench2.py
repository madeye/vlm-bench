#!/usr/bin/env python3
"""Prompt-variant pass: Qwen-native grounding prompt (JSON bbox_2d, xyxy px).

Only for models with real grounding ability; results go to results2.jsonl.
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run_bench as rb

PROMPT2 = ("This is a screenshot of a mobile app splash advertisement. "
           "Locate the button used to skip or close the ad (usually labeled "
           "'跳过', 'Skip', '关闭' or shown as a countdown circle). "
           'Output only its bounding box in JSON: {"bbox_2d": [x1, y1, x2, y2]}')

SUBSET = {"internvl3-2b", "qwen2-vl-2b", "qwen25-vl-3b-q8mm", "minicpm-v26",
          "smolvlm2-500m"}

RESULTS2 = os.path.join(ROOT, "results2.jsonl")
done = set()
if os.path.exists(RESULTS2):
    with open(RESULTS2) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["model"], r["image"]))
            except Exception:
                pass

meta = json.load(open(os.path.join(ROOT, "testset", "meta.json")))
images = [m["file"] for m in meta]
out = open(RESULTS2, "a")
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
                 "--image", os.path.join(rb.IMGDIR, img),
                 "--temp", "0", "-n", "64", "-p", PROMPT2],
                capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace")
            raw, rc = p.stdout.strip(), p.returncode
        except subprocess.TimeoutExpired:
            raw, rc = "", -9
        rec = {"model": mid, "image": img, "rc": rc,
               "wall_s": round(time.time() - t0, 2), "raw": raw, "timing": ""}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{mid}] {img} rc={rc} :: {raw[:80]!r}", flush=True)
print("BENCH2 PASS COMPLETE")
