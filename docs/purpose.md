# What this repository is for

## One-sentence purpose

**vlm-bench** is a controlled offline suite for comparing vision models on **mobile splash-ad skip-button grounding**, so the ad-skipper app can pick a size/accuracy trade-off for its image-layer detector (L3).

## The product problem

The ad-skipper app helps users dismiss interstitial / splash ads. Part of that pipeline is visual:

1. Take a screenshot of the foreground app.
2. Decide whether a **skip / close / countdown** control is present.
3. Return a bounding box and **tap its center**.

That is harder than it sounds:

- Skip controls are often **tiny** (small pills, rings, bare “跳过”).
- Ads include **distractor CTAs** (立即下载, 摇一摇, swipe-up) that must not be tapped.
- The model must run **on a phone**: model download size, RAM, and latency all matter.
- VLM outputs use **different coordinate conventions** (xyxy vs yxyx, 0–1000 norm vs absolute pixels).

Shipping the wrong model or parser looks like “AI doesn’t work” even when the model is capable under the right settings.

## What we use this repo for

| Use | How |
|-----|-----|
| Compare GGUF VLMs | `download.sh` + `run_bench*.py` + `score_all.py` |
| Diagnose app bugs | Reproduce app prompt / 448 px / yxyx parse and show 0/20 |
| Pick resolution & format | Same model at 448 / 672 / 896 / 1120 with native JSON bbox prompts |
| Train a tiny specialist | Synthetic YOLO data + real hard negatives → YOLO11n |
| Validate real-device UI | adb captures from Douban / Hupu; measure FP and real splash hits |
| Record decisions | `REPORT.md` tables and recommendations for the app team |

## What this repo is **not**

- **Not** the production ad-skipper Android/iOS app (that lives in a separate repo).
- **Not** a general VQA or OCR benchmark.
- **Not** a claim that synthetic ads equal the full real-world distribution — real splash recall still needs more field screenshots.
- **Not** a hosted leaderboard; everything is local scripts + gitignored artifacts.

## How results feed the app

Conclusions from this bench drive the L3 design:

```
L1  Accessibility / node match (fast, free)
L2  OCR text heuristics
L3a Bundled specialized detector (YOLO — small, default)
L3b Downloaded VLM (e.g. InternVL3-2B) as fallback
```

with image-layer work often gated to early session windows so cost stays bounded.

See **[REPORT.md](../REPORT.md)** for the evidence behind those choices (and for known parser/resolution bugs that made every VLM score 0 under the original app config).

## Success criteria for experiments here

An experiment is useful if it answers at least one of:

1. Does this model **hit** skip buttons under the **tap-center** metric?
2. Does it **avoid distractor CTAs**?
3. What **size / latency / resolution** does that require?
4. On **real UI**, does it false-positive when there is no splash ad?

If a change does not move those numbers (or explain them), it does not belong as a “result” in this repo.
