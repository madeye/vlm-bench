# Data

## Evaluation set (`testset/`)

- **Purpose:** Fixed benchmark for VLM and YOLO hit-rate.
- **Generator:** `gen_testset.py` (seeded, reproducible).
- **Resolution:** 1080×2400 full-bleed synthetic splash ads (CN-style).
- **Contents:** ~20 images covering:
  - Skip styles: dark/light/outline pill, bare text, countdown ring, tiny font
  - Positions: corners / bottom
  - Distractors: 立即下载 / 摇一摇 / swipe-up style CTAs + brand strip
- **Labels:** `testset/meta.json`

```json
{
  "file": "tr_pill_5.png",
  "desc": "top-right dark pill 跳过 5",
  "w": 1080,
  "h": 2400,
  "gt": [x0, y0, x1, y1],
  "distractors": [[x0, y0, x1, y1], ...]
}
```

Scaled copies used as VLM inputs: `testset448/`, `testset672/`, `testset896/`, `testset1120/` (max-dimension resize).

## YOLO training set (`yolo_data/`)

- **Purpose:** Train a single-class detector (`skip`).
- **Generator:** `gen_yolo_data.py` — intentionally **broader** style space than the eval set (so val is not pure memorization).
- **Layout:** Ultralytics YOLO format

```text
yolo_data/
  data.yaml
  images/train|val/*.jpg
  labels/train|val/*.txt   # class cx cy w h (normalized); empty = negative
```

- ~10% of synthetic images have **no** skip button (background negatives).
- Train images are gitignored; regenerate with `gen_yolo_data.py` when needed.

## Real UI captures

### `neg_shots/`

Older real device / emulator screenshots (settings, chrome, home, …) used as **hard negatives** after early YOLO models false-positived on every real screen.

### `real_shots/`

Phone captures (primarily 豆瓣 / 虎扑) via `capture_real_shots.py`:

| Path | Role |
|------|------|
| `real_shots/raw/` | Full screenshots (gitignored) |
| `real_shots/meta.json` | Labeled **positive** splash frames with `gt` boxes |
| `real_shots/*_gt.png` | Optional visual QA overlays |

Import into YOLO train:

```bash
python import_real_shots.py --augs 6
```

- Dedupes by content hash
- Drops near-blank loaders
- Unlabeled frames → empty labels (negatives)
- Frames in `meta.json` → YOLO `skip` boxes (+ color/crop augs)

## Metric alignment

Whether synthetic or real:

- **Positive:** center of predicted box inside `gt`
- **Negative:** any detection at operating conf is a **false positive**
- Distractors on eval ads are tracked as **CTA misclicks**

Do not mix “IoU-only” ranking with product decisions unless you also report the tap-center metric the app uses.
