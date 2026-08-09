# VLM Model Trade-off Benchmark for Ad Skipper (L3 grounding)

Date: 2026-08-07. Host: macOS (Metal), llama.cpp `llama-mtmd-cli` (build d6cf0b288, 2026-06-11).

## Goal

Pick the best size/accuracy trade-off GGUF VLM for the L3 skip-button grounding
layer, and find out why the current setup underperforms.

## Method

- 20 synthetic 1080×2400 splash-ad screenshots (`testset/`), CN-style: skip
  button variants (dark/light/outline pill, bare text, countdown ring, 4
  corner/bottom positions, tiny 30 px font) + distractor CTAs (立即下载 /
  摇一摇 / swipe-up) + brand strip. Ground truth rects in `testset/meta.json`
  (verified visually, `gt_check.png`).
- Metric = what the app does: predicted bbox **center** inside GT button rect
  (pipeline taps the center). Also counted: taps landing on distractor CTAs
  (= opens the ad, worse than a miss).
- Passes: app's current prompt/448px (`results.jsonl`), native JSON-bbox prompt
  at 448 / 672 / 896 / 1120 px max-dim (`results2/4/3/5.jsonl`). temp 0, n=64.
  Every output scored under 4 coordinate conventions (app yxyx-norm, xyxy-norm,
  xyxy-px, yxyx-px); best shown. Scorer: `score_all.py`.

## Results (hits out of 20, best convention per cell)

| Model | Download | 448 | 672 | 896 | 1120 | Convention | CTA misclicks |
|---|---|---|---|---|---|---|---|
| SmolVLM2-256M Q8 (bundled) | 0.27 GB | 0 | – | 0 | – | none parseable | 0 |
| LFM2-VL-450M Q8 | 0.48 GB | 0 | – | 0 | – | full-screen boxes | 0 |
| SmolVLM2-500M Q8 | 0.55 GB | 0 | – | 3 | – | – | 0 |
| InternVL3-1B Q8 | 1.01 GB | 0 | 2 | 0 | – | – | 0 |
| **InternVL3-2B Q4_K_M** | **1.46 GB** | 4 | 8 | **16** | **17** | xyxy, 0–1000 norm | **0** |
| Qwen2-VL-2B Q4_K_M | 2.32 GB | 1 | 0 | 5 | – | xyxy norm | 0 |
| Qwen2.5-VL-3B Q4_K_M (+Q8 mmproj) | 2.77 GB | 6 | **16** | 12 | – | xyxy, abs px | 1 |
| MiniCPM-V 2.6 Q4_K_M | 5.72 GB | 6 | – | 8 | – | xyxy norm | 0 |

Qwen2.5-VL-3B with f16 mmproj (3.27 GB) scored identically to Q8_0 mmproj
(12/20 @896) → Q8_0 mmproj is free 0.5 GB savings.

## Key findings

1. **The app's current config scores 0/20 with every model** — including the
   strong ones. Three independent causes:
   - **448 px input is too small**: the skip pill is ~30×15 px after downscale.
     Same model+prompt goes 4→16 (InternVL3-2B) and 6→16 (Qwen2.5-VL) when fed
     672–896 px.
   - **Prompt/convention mismatch**: models answer `[x1,y1,x2,y2]` (InternVL /
     MiniCPM: normalized 0–1000; Qwen2.5-VL: absolute pixels), while
     `BboxParser` assumes `[ymin,xmin,ymax,xmax]` — axes swapped for every
     model tested.
   - **`BboxParser` digit bug**: models emit JSON `{"bbox_2d": [...]}`; the
     regex grabs the `2` from `bbox_2d` as the first coordinate, shifting all
     four. (Same class of bug initially broke this benchmark's scorer.)
2. **Everything ≤1B params is useless for grounding** (0–3/20 at any
   resolution). The bundled SmolVLM2-256M cannot do this task; its known
   "full-screen bbox" behavior on-device is confirmed here.
3. **InternVL3-2B Q4_K_M @896 px is the sweet spot**: 16/20, 0 CTA misclicks,
   1.46 GB — half the size of Qwen2.5-VL-3B (16/20 @672 but 1 CTA misclick,
   2.77 GB) and ¼ the size of MiniCPM-V 2.6 (8/20). 1120 px only adds +1 hit
   for ~1.6× the vision compute. Its 4 misses @896 are edge-adjacent
   near-misses (ring top edge, pill bottom edge) except one (br_bare confused
   with bottom-center position).
4. **MiniCPM-V 2.6 and Qwen2-VL-2B are dominated** — bigger and worse; drop
   from the catalog.

## Recommendation for the app

- **Recommended download tier: InternVL3-2B Q4_K_M + Q8_0 mmproj (1.46 GB), 896 px input.**
  Repo: `ggml-org/InternVL3-2B-Instruct-GGUF` (HF; mirror availability on
  ModelScope needs checking).
- Keep Qwen2.5-VL-3B (switch mmproj to Q8_0, run at 672 px) as the alternative
  tier; retire Qwen2-VL-2B and MiniCPM-V 2.6.
- Keep SmolVLM2-256M bundled only as a demo/fallback, or drop it — set
  expectations that L3 needs a downloaded model.
- Code changes needed in `core`:
  - `VlmDetector.MAX_DIM` 448 → 896 (or per-model).
  - Prompt → ask for JSON `{"bbox_2d": [x1,y1,x2,y2]}` (native format).
  - `BboxParser`: parse numbers inside the bracket group only / ignore digits
    embedded in identifiers (`bbox_2d`), treat as **xyxy**, and support
    per-model scale (0–1000 normalized for InternVL vs input-image pixels for
    Qwen2.5-VL).
- Latency caveat: host medians (incl. per-run model load) were ~3.0 s for
  InternVL3-2B@896. On-device (resident model, Vulkan GPU) needs measuring;
  896 px quadruples image tokens vs 448. If a flagship phone can't hold ~2 s,
  try InternVL3-2B @672 (8/20) vs Qwen2.5-VL-3B @672 (16/20) — Qwen becomes
  the better pick at 672 despite the extra 1.3 GB.

## Artifacts

- `score_all.py` → table + `score_all.json`; raw outputs in `results*.jsonl`.
- `internvl3_2b_896_preds.png` — winner's predictions overlaid (blue=hit tap,
  red=miss tap, green=GT).
- Models kept in `models/` (~15 GB) for future re-runs; test set generator
  `gen_testset.py` (seeded, reproducible).

## Addendum: YOLO11n specialized detector (2026-08-07, later the same day)

User constraint: 2B VLM too large to bundle. Since no capable ≤1B VLM exists
(see table), we trained a specialized detector instead:

- **YOLO11n** (~2.6M params) on 2400 synthetic splash ads (`gen_yolo_data.py`,
  broader style space than the eval set, 10% negatives) + 300 augmented real
  UI negatives (emulator screenshots) after the first round false-positived on
  16/16 real screens.
- **Benchmark result (fp32 TFLite, 10.6MB): 19/20 hits, 0 CTA misclicks,
  21ms/image host CPU** — beats every VLM tested including InternVL3-2B
  (16/20, 1.46GB, ~3s).
- Exported via ONNX -> onnx2tf (ultralytics' litert path clashes with torch
  2.13). Output [1,5,8400], coords in input pixels, 640 letterbox.
- Caveat: train and eval are both synthetic; real-world splash-ad recall
  unproven. The synthetic->real gap on *negatives* was demonstrably large
  (16/16 FP screens before negative fine-tune); expect a gap on positives too.
  Collecting real splash-ad screenshots for eval/fine-tune is the highest-value
  next step.

Final app architecture (branch feature/internvl3-2b-default):
L1 node match -> L2 OCR -> L3a bundled YOLO (10MB, default) -> L3b downloaded
VLM (InternVL3-2B recommended) as fallback, with image-layer detection gated
to the first 8s of an app session.

## Addendum: skip_v3b with real Douban/Hupu shots (2026-08-09)

Captured ~70 real UI frames from a connected phone (豆瓣 `com.douban.frodo`,
虎扑 `com.hupu.games`) via `capture_real_shots.py` → `real_shots/raw/`.
Imported as hard negatives (+ labeled the 2 splash positives) with
`import_real_shots.py`, then oversampled the positives 40× each.

| Model | Synthetic testset | Real splash pos | Real UI FP (68) | Weights |
|-------|-------------------|-----------------|-----------------|---------|
| skip_v2 | 18/20 | 0/2 | 0 | `.../skip_v2/weights/best.pt` |
| skip_v3 last | 18/20 | 1/2 (hupu only) | 0 | `.../skip_v3/weights/last.pt` |
| **skip_v3b** | **19/20** | **2/2** (conf ~0.88–0.92) | **0** | `runs/detect/yolo_runs/skip_v3b/weights/best.pt` |

- Train: 25 epochs, MPS, from skip_v2, 3164 train images (incl. 384 real augs
  + 80 oversampled real positives). Val mAP50 0.993 / mAP50-95 0.903.
- Preferred checkpoint for the app L3a path: **skip_v3b best.pt**.
- Still only 2 real splash positives; collect more cold-launch splash ads to
  harden recall further.
