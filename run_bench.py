#!/usr/bin/env python3
"""Run llama-mtmd-cli grounding benchmark for ad-skipper candidate models.

Mirrors the app's L3 path: 448 max-dim image, same prompt, temp 0, n=64.
Resumable: skips (model, image) pairs already in results.jsonl.
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
CLI = "/Volumes/DATA/workspace/llama.cpp/build/bin/llama-mtmd-cli"
MDIR = os.path.join(ROOT, "models")
BUNDLED = "/Volumes/DATA/workspace/ad-skipper/core/build/generated/assets/downloadBundledModel/bundled_model"
IMGDIR = os.path.join(ROOT, "testset448")
RESULTS = os.path.join(ROOT, "results.jsonl")

PROMPT = ("This is a screenshot of a mobile app showing a splash advertisement. "
          "Locate the button used to skip or close the ad (usually labeled "
          "'跳过', 'Skip', '关闭' or shown as a countdown circle). "
          "Reply with only the bounding box as [ymin, xmin, ymax, xmax] "
          "normalized to 0-1000.")

MODELS = [
    ("smolvlm2-256m", f"{BUNDLED}/SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
     f"{BUNDLED}/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"),
    ("smolvlm2-500m", f"{MDIR}/SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
     f"{MDIR}/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"),
    ("lfm2-vl-450m", f"{MDIR}/LFM2-VL-450M-Q8_0.gguf",
     f"{MDIR}/mmproj-LFM2-VL-450M-Q8_0.gguf"),
    ("internvl3-2b", f"{MDIR}/InternVL3-2B-Instruct-Q4_K_M.gguf",
     f"{MDIR}/mmproj-InternVL3-2B-Instruct-Q8_0.gguf"),
    ("internvl3-1b", f"{MDIR}/InternVL3-1B-Instruct-Q8_0.gguf",
     f"{MDIR}/mmproj-InternVL3-1B-Instruct-Q8_0.gguf"),
    ("qwen2-vl-2b", f"{MDIR}/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
     f"{MDIR}/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"),
    ("qwen25-vl-3b-q8mm", f"{MDIR}/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
     f"{MDIR}/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"),
    ("qwen25-vl-3b-f16mm", f"{MDIR}/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
     f"{MDIR}/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"),
    ("minicpm-v26", f"{MDIR}/minicpm-v26-ggml-model-Q4_K_M.gguf",
     f"{MDIR}/minicpm-v26-mmproj-model-f16.gguf"),
]

def ready(path):
    # bundled files have no .done marker
    return os.path.exists(path) and (path.startswith(BUNDLED) or os.path.exists(path + ".done"))

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
for mid, mpath, mmpath in MODELS:
    if not (ready(mpath) and ready(mmpath)):
        print(f"[skip] {mid}: files not ready", flush=True)
        continue
    for img in images:
        if (mid, img) in done:
            continue
        t0 = time.time()
        try:
            p = subprocess.run(
                [CLI, "-m", mpath, "--mmproj", mmpath,
                 "--image", os.path.join(IMGDIR, img),
                 "--temp", "0", "-n", "64", "-p", PROMPT],
                capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace")
            raw = p.stdout.strip()
            rc = p.returncode
            timing = "\n".join(l for l in p.stderr.splitlines()
                               if "eval time" in l or "total time" in l)
        except subprocess.TimeoutExpired:
            raw, rc, timing = "", -9, "TIMEOUT"
        wall = time.time() - t0
        rec = {"model": mid, "image": img, "rc": rc, "wall_s": round(wall, 2),
               "raw": raw, "timing": timing}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{mid}] {img} rc={rc} {wall:.1f}s :: {raw[:80]!r}", flush=True)
print("BENCH PASS COMPLETE")
