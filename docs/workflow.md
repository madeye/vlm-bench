# Workflows

Prerequisites: Python ≥3.11, `uv sync` (or an equivalent venv with `pyproject.toml` deps).  
VLM runs also need a built **llama.cpp** `llama-mtmd-cli`.

## A. VLM candidate bake-off

Goal: rank GGUF VLMs for skip-button grounding.

```bash
./download.sh                 # models/ + yolo11n.pt
python gen_testset.py         # testset/ + meta.json (seeded)
# optional: pre-scaled dirs testset448/672/896/1120 already used by passes

python run_bench.py           # app-style prompt @448 → results.jsonl
python run_bench2.py          # native JSON bbox @448 → results2.jsonl
python run_bench4.py          # @672 → results4.jsonl
python run_bench3.py          # @896 → results3.jsonl
# (1120 pass was results5.jsonl in earlier work)

python score_all.py           # multi-convention table → score_all.json
```

Notes:

- JSONL runs are **resumable** (skip completed model×image pairs).
- Scoring tries several coordinate conventions; the app historically assumed **yxyx normalized 0–1000**, while strong models often emit **xyxy**.
- Prefer bracket-aware number parsing (see `score_all.py`) so `bbox_2d` does not inject a spurious `2`.

Edit `CLI`, `BUNDLED`, and `MODELS` paths at the top of `run_bench.py` for your machine.

## B. YOLO specialist train / eval

Goal: a small detector that can ship bundled (L3a).

```bash
python gen_yolo_data.py [N_train] [N_val]   # default 2400 / 240
# train with ultralytics, e.g.:
#   yolo detect train model=yolo11n.pt data=yolo_data/data.yaml ...

python eval_yolo.py path/to/weights.pt 0.25
```

Preferred local checkpoint after real-data fine-tune:

```text
runs/detect/yolo_runs/skip_v3b/weights/best.pt
```

Eval uses the **same** hit metric as the VLM bench (center of top-confidence box inside GT).

## C. Real phone hard negatives / splash positives

Goal: reduce false positives on normal app UI and measure real splash recall.

```bash
# Phone connected via adb (Douban + Hupu installed)
python capture_real_shots.py both 20    # → real_shots/raw/

# Label any true splash frames in real_shots/meta.json (xyxy pixels),
# then import into YOLO train as empty-label negatives + labeled positives:
python import_real_shots.py --augs 6

# Fine-tune from previous best (example):
yolo detect train \
  model=runs/detect/yolo_runs/skip_v2/weights/best.pt \
  data=yolo_data/data.yaml epochs=25 device=mps \
  project=yolo_runs name=skip_v3b
```

`real_shots/raw/` is gitignored (large). Keep `real_shots/meta.json` for positive labels.

## D. Scoring philosophy (do not reinvent casually)

1. Always report **hits / N** under tap-center.
2. Always count **CTA misclicks** separately.
3. When comparing VLMs, report the **best coordinate convention** *and* the app convention, so parser bugs are visible.
4. Prefer fixed seeds for eval generators; diversify only train generators.

## Related artifacts

| Artifact | Meaning |
|----------|---------|
| `results*.jsonl` | Raw VLM text + wall time per image |
| `score_all.json` | Aggregated multi-pass scores |
| `gt_check.png` | GT boxes overlaid for visual audit |
| `internvl3_2b_896_preds.png` | Example VLM prediction overlay |
| `yolo_train*.log` | Ultralytics train logs |
| `REPORT.md` | Human-readable conclusions |
