#!/usr/bin/env python3
"""Unified scoring across all benchmark passes and coordinate conventions.

For each (pass, model): hit rate under the app's current parser, and under
each alternate convention — so we can pick best model AND the right
prompt/parser for it. Hit = predicted bbox center inside GT button rect.
"""
import json, os, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 2400
NUM = re.compile(r"-?\d+(?:\.\d+)?")

PASSES = [
    ("p1_app448", "results.jsonl", 448 / 2400),
    ("p2_native448", "results2.jsonl", 448 / 2400),
    ("p3_native896", "results3.jsonl", 896 / 2400),
    ("p4_native672", "results4.jsonl", 672 / 2400),
]

meta = {m["file"]: m for m in json.load(open(os.path.join(ROOT, "testset", "meta.json")))}

BRACKET = re.compile(r"\[([^\[\]]*)\]")
CLEAN_NUM = re.compile(r"(?<![A-Za-z_0-9])-?\d+(?:\.\d+)?")

def first4(text):
    # Prefer numbers inside a bracket group (avoids the '2' in 'bbox_2d');
    # fall back to identifier-filtered numbers anywhere.
    for m in BRACKET.finditer(text):
        nums = [float(x) for x in CLEAN_NUM.findall(m.group(1))]
        if len(nums) >= 4:
            return nums[:4]
    nums = [float(x) for x in CLEAN_NUM.findall(text)]
    return nums[:4] if len(nums) >= 4 else None

def adaptive_norm(v):
    if all(0 <= x <= 1 for x in v):
        s = 1.0
    elif all(0 <= x <= 100 for x in v):
        s = 100.0
    else:
        s = 1000.0
    return [min(max(x / s, 0.0), 1.0) for x in v]

def centers(text, S):
    """All candidate interpretations -> center in original px."""
    v = first4(text)
    if v is None:
        return None
    sw, sh = W * S, H * S
    out = {}
    n = adaptive_norm(v)
    if n[0] < n[2] and n[1] < n[3]:
        # app convention: yxyx normalized (exact BboxParser port)
        out["app_yxyx_norm"] = ((n[1] + n[3]) / 2 * W, (n[0] + n[2]) / 2 * H)
        # same numbers read as xyxy normalized
        out["xyxy_norm"] = ((n[0] + n[2]) / 2 * W, (n[1] + n[3]) / 2 * H)
    # absolute pixels wrt the scaled input image
    if v[0] < v[2] and v[1] < v[3]:
        if v[2] <= sw * 1.15 and v[3] <= sh * 1.15:
            out["xyxy_px"] = ((v[0] + v[2]) / 2 / S, (v[1] + v[3]) / 2 / S)
        if v[2] <= sh * 1.15 and v[3] <= sw * 1.15:
            out["yxyx_px"] = ((v[1] + v[3]) / 2 / S, (v[0] + v[2]) / 2 / S)
    return out

def inside(pt, rect, pad=0):
    x, y = pt
    return rect[0] - pad <= x <= rect[2] + pad and rect[1] - pad <= y <= rect[3] + pad

summary = {}
for pname, fname, S in PASSES:
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        continue
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["model"]].append(r)
    for model, recs in rows.items():
        conv_hits = defaultdict(int)
        conv_bad = defaultdict(int)
        no_nums = 0
        walls = []
        for r in recs:
            gt = meta[r["image"]]["gt"]
            dist = meta[r["image"]]["distractors"]
            walls.append(r["wall_s"])
            cs = centers(r["raw"], S)
            if cs is None:
                no_nums += 1
                continue
            for name, c in cs.items():
                if inside(c, gt):
                    conv_hits[name] += 1
                if any(inside(c, d) for d in dist):
                    conv_bad[name] += 1
        n = len(recs)
        best = max(conv_hits.items(), key=lambda kv: kv[1]) if conv_hits else ("-", 0)
        summary[(pname, model)] = {
            "n": n, "no_nums": no_nums,
            "app_hits": conv_hits.get("app_yxyx_norm", 0),
            "conv_hits": dict(conv_hits), "conv_bad": dict(conv_bad),
            "best_conv": best[0], "best_hits": best[1],
            "median_wall_s": sorted(walls)[len(walls) // 2],
        }

json.dump({f"{k[0]}|{k[1]}": v for k, v in summary.items()},
          open(os.path.join(ROOT, "score_all.json"), "w"), indent=1)

print(f"{'pass':<14}{'model':<20}{'app':>7}{'best conv':>16}{'best':>7}{'no#':>5}{'med_s':>7}")
for (pname, model), s in sorted(summary.items()):
    print(f"{pname:<14}{model:<20}{s['app_hits']:>4}/{s['n']:<3}"
          f"{s['best_conv']:>16}{s['best_hits']:>4}/{s['n']:<3}{s['no_nums']:>4}{s['median_wall_s']:>7.1f}")
