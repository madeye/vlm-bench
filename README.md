# vlm-bench

**What is this?** An offline research / engineering benchmark for **finding and tapping “skip ad” buttons** on mobile splash screens.

It exists to answer one practical question for the **ad-skipper** app:

> Which on-device vision model (or specialized detector) should the app use for L3 “look at the screenshot and ground the skip button” — and at what size / resolution / cost?

This repo is **not** the production app. It is the harness used to generate data, run candidates, score them with the same metric the app uses (tap the box center), and record conclusions.

---

## Why it exists

Chinese-style splash ads often show a small **跳过 / Skip / countdown ring** control plus loud **distractor CTAs** (立即下载, 摇一摇, swipe-up). The app’s image-layer path must:

1. **Find** the skip control (bounding box)
2. **Tap its center** (not the download button)
3. Fit **on a phone** (size + latency matter)

Early VLMs and parser settings scored **0/20** under the app’s original config. This suite isolates whether the failure is model size, input resolution, prompt/coord convention, or something else — and later validates a tiny **YOLO11n** specialist as a bundleable L3a detector.

Full measured results and recommendations: **[REPORT.md](REPORT.md)**.  
Agent-oriented conventions: **[AGENTS.md](AGENTS.md)**.  
Longer narrative docs: **[docs/](docs/)**.

---

## What’s in the box

| Area | Role |
|------|------|
| **VLM benchmark** | Download GGUF VLMs, run `llama-mtmd-cli`, score bbox outputs |
| **Synthetic testset** | 20 seeded 1080×2400 splash scenes with GT skip rects + distractors |
| **YOLO track** | Train a specialized skip detector; eval with the same hit metric |
| **Real phone captures** | Hard negatives (and a few real splash positives) from Douban / Hupu |
| **ggml-yolo** | Experimental ONNX → ggml host validation (not required for main bench) |

### Headline findings (snapshot)

| Approach | Footprint | Hits / 20 | Role |
|----------|-----------|-----------|------|
| App’s original VLM config (@448, yxyx parse) | varies | **0/20** | Broken for this task |
| InternVL3-2B Q4_K_M @896 | ~1.46 GB | **16–17/20** | Best **downloadable** VLM tier |
| YOLO11n specialized (**skip_v3b**) | ~5–11 MB | **19/20** synthetic; real splash **2/2** on labeled set | Best **bundled** L3a path |

Suggested product stack (from the report):  
**L1 node match → L2 OCR → L3a bundled YOLO → L3b downloaded VLM fallback.**

---

## Metric (shared everywhere)

- Original image space: **1080×2400**
- **Hit** = predicted box **center** lands inside the GT skip rect  
  (matches “tap center of predicted bbox”)
- **CTA misclick** = center lands on a distractor (worse than a miss)

GT lives in `testset/meta.json`. Visual check: `gt_check.png`.

---

## Quick start

```bash
# Python ≥3.11
uv sync                    # or: python -m venv .venv && pip install -e .

# Optional: GGUF models + yolo11n base weights
./download.sh              # uses local https_proxy as written in the script

# Rebuild synthetic eval set (seeded, reproducible)
python gen_testset.py

# VLM pass (needs llama-mtmd-cli; paths at top of run_bench.py)
python run_bench.py
python score_all.py

# YOLO: data → train elsewhere / existing weights → eval
python gen_yolo_data.py
python eval_yolo.py runs/detect/yolo_runs/skip_v3b/weights/best.pt 0.25
```

VLM CLI paths in `run_bench.py` currently assume sibling checkouts on this machine (`/Volumes/DATA/workspace/llama.cpp`, ad-skipper bundled model). Edit those constants if your layout differs.

---

## Repo layout

```
vlm-bench/
├── README.md              # you are here
├── REPORT.md              # full experiment write-up + tables
├── AGENTS.md              # instructions for coding agents
├── docs/                  # purpose, workflow, data notes
│
├── gen_testset.py         # synthetic eval images + meta.json
├── run_bench*.py          # VLM inference passes → results*.jsonl
├── score.py / score_all.py
│
├── gen_yolo_data.py       # synthetic YOLO train/val
├── eval_yolo.py           # YOLO vs testset (hit metric)
├── capture_real_shots.py  # adb: Douban/Hupu screenshots
├── import_real_shots.py   # real frames → yolo_data train
│
├── testset/               # full-res eval set + meta.json
├── testset{448,672,...}/  # scaled inputs for VLM passes
├── real_shots/            # phone captures (raw gitignored)
├── neg_shots/             # older real UI hard negatives
├── models/                # GGUFs (gitignored)
├── yolo_data/             # YOLO dataset (images gitignored)
├── runs/                  # Ultralytics train artifacts (gitignored)
└── ggml-yolo/             # experimental ggml export/run
```

Large local artifacts are gitignored (`models/`, `yolo_data/images/`, `runs/`, `real_shots/raw/`, venvs).

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/purpose.md](docs/purpose.md) | Problem, non-goals, how this feeds the app |
| [docs/workflow.md](docs/workflow.md) | End-to-end VLM + YOLO workflows |
| [docs/data.md](docs/data.md) | Datasets, labels, real captures |
| [REPORT.md](REPORT.md) | Measured results and product recommendations |

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Max Lv.
