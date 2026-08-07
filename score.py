#!/usr/bin/env python3
"""Score benchmark results against ground truth.

Primary metric = what the app does: BboxParser (first 4 numbers, scale
heuristic, [ymin,xmin,ymax,xmax]) -> rect center -> tap. Hit = tap lands
inside GT button rect. Also scores alternate coordinate conventions to
detect model/parser mismatches, and harmful taps on distractor CTAs.
"""
import json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 2400
S = 448 / 2400  # testset448 scale
SW, SH = int(W * S), int(H * S)  # 201x448-ish

NUM = re.compile(r"-?\d+(?:\.\d+)?")

def parse_nums(text):
    nums = [float(m) for m in NUM.findall(text)]
    return nums[:4] if len(nums) >= 4 else None

def app_parse(text):
    """Exact port of BboxParser.parse: returns (ymin,xmin,ymax,xmax) in 0..1."""
    v = parse_nums(text)
    if not v:
        return None
    if all(0 <= x <= 1 for x in v):
        scale = 1.0
    elif all(0 <= x <= 100 for x in v):
        scale = 100.0
    else:
        scale = 1000.0
    n = [min(max(x / scale, 0.0), 1.0) for x in v]
    if n[0] >= n[2] or n[1] >= n[3]:
        return None
    return n

def center_from_yxyx(n):
    return ((n[1] + n[3]) / 2 * W, (n[0] + n[2]) / 2 * H)

def alt_centers(text):
    """Alternate interpretations -> dict name->(cx,cy) in original px."""
    v = parse_nums(text)
    if not v:
        return {}
    out = {}
    # xyxy with the app's same adaptive scale heuristic (1 / 100 / 1000)
    if all(0 <= x <= 1 for x in v):
        scale = 1.0
    elif all(0 <= x <= 100 for x in v):
        scale = 100.0
    else:
        scale = 1000.0
    n = [min(max(x / scale, 0.0), 1.0) for x in v]
    if n[0] < n[2] and n[1] < n[3]:
        out["xyxy_norm"] = ((n[0] + n[2]) / 2 * W, (n[1] + n[3]) / 2 * H)
    # xyxy absolute pixels in 448-scaled image
    if v[0] < v[2] and v[1] < v[3] and v[2] <= SW * 1.2 and v[3] <= SH * 1.2:
        out["xyxy_px448"] = ((v[0] + v[2]) / 2 / S, (v[1] + v[3]) / 2 / S)
    # yxyx absolute pixels in 448-scaled image
    if v[0] < v[2] and v[1] < v[3] and v[2] <= SH * 1.2 and v[3] <= SW * 1.2:
        out["yxyx_px448"] = ((v[1] + v[3]) / 2 / S, (v[0] + v[2]) / 2 / S)
    return out

def inside(pt, rect, pad=0):
    x, y = pt
    return rect[0] - pad <= x <= rect[2] + pad and rect[1] - pad <= y <= rect[3] + pad

def eval_time_ms(timing):
    # "llama_perf_context_print: eval time = 123.45 ms / 64 runs ..."
    tot = None
    for line in timing.splitlines():
        m = re.search(r"total time\s*=\s*([\d.]+)\s*ms", line)
        if m:
            tot = float(m.group(1))
    return tot

meta = {m["file"]: m for m in json.load(open(os.path.join(ROOT, "testset", "meta.json")))}
rows = defaultdict(list)
with open(os.path.join(ROOT, "results.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        rows[r["model"]].append(r)

report = {}
for model, recs in sorted(rows.items()):
    n = len(recs)
    app_hit = app_near = parse_fail = bad = 0
    best_hit = defaultdict(int)  # interpretation -> hits (incl app)
    lat = []
    per_img = []
    for r in recs:
        gt = meta[r["image"]]["gt"]
        dist = meta[r["image"]]["distractors"]
        t = eval_time_ms(r["timing"])
        if t:
            lat.append(t)
        parsed = app_parse(r["raw"])
        entry = {"img": r["image"], "raw": r["raw"][:120]}
        if parsed is None:
            parse_fail += 1
            entry["app"] = "parse_fail"
        else:
            c = center_from_yxyx(parsed)
            hit = inside(c, gt)
            near = inside(c, gt, pad=40)
            if hit:
                app_hit += 1
                best_hit["app_yxyx"] += 1
            if near:
                app_near += 1
            if any(inside(c, dr) for dr in dist):
                bad += 1
            entry["app"] = "HIT" if hit else ("near" if near else "miss")
            entry["click"] = [round(c[0]), round(c[1])]
        for name, c in alt_centers(r["raw"]).items():
            if inside(c, gt):
                best_hit[name] += 1
        per_img.append(entry)
    report[model] = {
        "n": n, "app_hit": app_hit, "app_near": app_near,
        "parse_fail": parse_fail, "bad_click": bad,
        "alt_hits": dict(best_hit),
        "median_total_ms": sorted(lat)[len(lat) // 2] if lat else None,
        "per_img": per_img,
    }

json.dump(report, open(os.path.join(ROOT, "score.json"), "w"), ensure_ascii=False, indent=1)

print(f"{'model':<22}{'hit':>5}{'near':>6}{'pfail':>7}{'bad':>5}{'best-alt':>22}{'med_ms':>9}")
for m, s in report.items():
    alts = s["alt_hits"]
    best = max(alts.items(), key=lambda kv: kv[1]) if alts else ("-", 0)
    print(f"{m:<22}{s['app_hit']:>3}/{s['n']:<3}{s['app_near']:>4}{s['parse_fail']:>7}"
          f"{s['bad_click']:>5}{best[0]:>16}={best[1]:<4}"
          f"{s['median_total_ms'] or 0:>9.0f}")
