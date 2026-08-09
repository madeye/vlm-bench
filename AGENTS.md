# vlm-bench — agent instructions

Benchmark suite for **ad-skip button grounding** on mobile splash ads. Compares GGUF VLMs (via llama.cpp) and a specialized YOLO11n detector for the ad-skipper app L3 path.

Canonical findings and recommendations: [`REPORT.md`](REPORT.md).

## What this repo is (and is not)

- **Is**: offline eval harness + synthetic data generators + scoring + YOLO train/export experiments + ggml YOLO host validation.
- **Is not**: the production ad-skipper app. App code lives elsewhere (`/Volumes/DATA/workspace/ad-skipper`). Do not invent app APIs here; mirror what the benches already encode (prompt, `BboxParser` conventions, tap-center metric).

## Layout

| Path | Role |
|------|------|
| `gen_testset.py` | Seeded synthetic eval set → `testset/` + `meta.json` |
| `run_bench.py` … `run_bench4.py` | VLM inference passes (`llama-mtmd-cli`); append `results*.jsonl` |
| `score.py` / `score_all.py` | Score JSONL vs GT; multi-convention hit rate |
| `gen_yolo_data.py` | Broader synthetic train/val for YOLO → `yolo_data/` |
| `eval_yolo.py` | YOLO eval on VLM testset (same hit metric) |
| `ggml-yolo/` | ONNX→gguf/prog conversion + host ggml executor + ORT compare |
| `models/` | Local GGUFs (gitignored; ~15 GB). Marker files: `*.done` |
| `testset/` | Full-res 1080×2400 eval images + `meta.json` |
| `testset{448,672,896,1120}/` | Max-dim scaled copies for each pass |
| `neg_shots/` | Older real UI screenshots (hard negatives) |
| `real_shots/` | Phone captures (`raw/`, labeled positives in `meta.json`) |
| `capture_real_shots.py` | adb walk of Douban/Hupu (etc.) → `real_shots/raw/` |
| `import_real_shots.py` | Dedupe/filter/augment real shots → `yolo_data` train |
| `runs/` | Ultralytics train/export artifacts (gitignored) |

## Environment

- Python **≥3.11**. Prefer project env:
  - `uv sync` (see `pyproject.toml` / `uv.lock`), or existing `.venv` / `yolo-venv` for heavy YOLO export stacks.
- System deps: **Pillow** needs macOS CJK fonts used in generators (`STHeiti`, etc.).
- VLM runtime: `llama-mtmd-cli` (paths hard-coded in `run_bench.py`: local llama.cpp build and optional ad-skipper bundled model dir). Prefer updating those constants over assuming `PATH`.
- Downloads: `./download.sh` (uses `https_proxy=http://127.0.0.1:7890`). Idempotent via `models/*.done`.

## Metric (do not change without updating all scorers)

- Image space: **1080×2400** original.
- **Hit** = center of predicted bbox falls inside GT skip-button rect (`meta.json` `gt` as `[x0,y0,x1,y1]`).
- **CTA misclick** = center falls inside any `distractors` rect (worse than a miss).
- App-style parse historically assumed **yxyx normalized 0–1000**; many models emit **xyxy** (0–1000 or absolute px). `score_all.py` scores multiple conventions; prefer its bracket-aware number extract (avoids the `2` in `bbox_2d`).

## Common workflows

```bash
# Deps
uv sync

# Models + yolo11n.pt
./download.sh

# Rebuild eval set (seeded) and scaled variants as needed
python gen_testset.py

# VLM passes (resumable; appends JSONL)
python run_bench.py    # app prompt @448 → results.jsonl
python run_bench2.py   # native JSON prompt @448 → results2.jsonl
python run_bench4.py   # @672 → results4.jsonl
python run_bench3.py   # @896 → results3.jsonl

# Aggregate scores
python score_all.py    # → score_all.json + table

# YOLO data / train / eval (ultralytics)
python gen_yolo_data.py [N_train] [N_val]
# Real phone UI (hard negatives + any labeled splash positives)
python capture_real_shots.py both 20   # needs adb device; Douban + Hupu
python import_real_shots.py --augs 6   # → yolo_data/images|labels/train/real_*
# train via ultralytics CLI; preferred weights: runs/detect/yolo_runs/skip_v3b/weights/best.pt
# (see REPORT.md skip_v3b addendum: 19/20 synthetic, 2/2 real splash, 0 FP)
python eval_yolo.py runs/detect/yolo_runs/skip_v3b/weights/best.pt 0.25

# ggml YOLO convert + host validate
python ggml-yolo/convert.py [path/to/model.onnx]
# build/run ggml-yolo/run_host against out/; python ggml-yolo/compare.py
```

## Conventions for code changes

- Keep scripts **standalone and readable** (few deps). Match existing style: stdlib + PIL/numpy/ultralytics/onnx as already used; no new frameworks without a clear need.
- **Resumable JSONL**: never truncate `results*.jsonl` unless the user asks; skip keys already present.
- Prefer **relative paths from `ROOT`**; if you must hard-code absolute host paths (CLI binary, sibling repos), keep them in one place at the top of the script.
- Generators: keep **reproducible seeds** unless explicitly diversifying train data; do not silently change the eval set semantics (`gen_testset.py` seed 42, 20 fixed scenarios).
- Large artifacts stay **gitignored**: `models/`, `yolo-venv/`, `.venv/`, `runs/`, `yolo_data/images/`, `ggml-yolo/out/`. Do not commit GGUFs, train images, or venvs.
- Do not rewrite `REPORT.md` conclusions unless new measured results justify it; append dated notes if needed.
- Git: work on a feature branch; do not commit to `main` directly. Do not commit unless asked.

## Sibling / external paths (this machine)

| Path | Use |
|------|-----|
| `/Volumes/DATA/workspace/llama.cpp/build/bin/llama-mtmd-cli` | Default VLM CLI in benches |
| `/Volumes/DATA/workspace/ad-skipper/.../bundled_model` | SmolVLM2-256M bundled baseline |
| HF repos listed in `download.sh` | GGUF + mmproj sources |

## Known pitfalls

1. **448 px input** is too small for skip pills after downscale; strong VLMs need ~672–896 max-dim.
2. **Parser digit bug**: greedily scanning all numbers grabs `2` from `bbox_2d` — use bracket-first extract (`score_all.first4`).
3. **Convention mismatch**: InternVL often 0–1000 xyxy; Qwen2.5-VL often absolute pixels; app assumed yxyx.
4. **≤1B VLMs** fail grounding here; YOLO11n specialized detector is the small-footprint winner (see REPORT addendum).
5. YOLO export: ultralytics LiteRT path can clash with torch; existing path was ONNX → onnx2tf. NCNN/ggml work is experimental under `ggml-yolo/`.
6. Train/eval are largely **synthetic**; real splash-ad recall is unproven — real screenshots are high value.

## When unsure

Prefer reading `REPORT.md`, then the relevant `run_bench*.py` / `score_all.py` / `eval_yolo.py`, over inventing new metrics or prompt formats. Match the app’s tap-center definition unless the task is explicitly to change it.
