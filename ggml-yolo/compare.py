#!/usr/bin/env python3
"""Compare host-ggml dumps against the ORT per-node reference.

Reports the FIRST node (in program order) whose output diverges, so layout
bugs can be localized. Handles the ggml<->ONNX dim reversal by comparing
flattened arrays (both are row-major/C-order in their own convention, and
ggml ne = reversed ONNX dims, so raw element order matches).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ref = dict(np.load(os.path.join(OUT, "ref.npz")))
ref = {k.replace("__", "/"): v for k, v in ref.items()}

# program order
order = []
for line in open(os.path.join(OUT, "yolo.prog")):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    outs = line.split("<-")[0].split(None, 1)[1].strip()
    for o in outs.split(","):
        order.append(o.strip())

man = {}
for line in open(os.path.join(OUT, "dump", "manifest.txt")):
    name, shape = line.rsplit(" ", 1)
    man[name] = [int(x) for x in shape.strip().split(",")]

def load_dump(name):
    safe = name.replace("/", "_")
    p = os.path.join(OUT, "dump", safe + ".bin")
    return np.fromfile(p, dtype=np.float32)

worst = []
first_bad = None
for name in order:
    if name not in man or name not in ref:
        continue
    got = load_dump(name)
    exp = ref[name].astype(np.float32).reshape(-1)
    if got.size != exp.size:
        print(f"[SIZE] {name}: ggml {got.size} vs ort {exp.size} (ne={man[name]}, ort={ref[name].shape})")
        if first_bad is None:
            first_bad = name
        continue
    denom = np.abs(exp).max() + 1e-6
    rel = np.abs(got - exp).max() / denom
    cos = float(np.dot(got, exp) / (np.linalg.norm(got) * np.linalg.norm(exp) + 1e-9))
    worst.append((rel, cos, name))
    if rel > 1e-2 and first_bad is None:
        first_bad = name
        print(f"[FIRST DIVERGENCE] {name}")
        print(f"  ne={man[name]} ort={ref[name].shape} relmax={rel:.4g} cos={cos:.5f}")
        print(f"  ggml[:6]={got[:6]}")
        print(f"  ort [:6]={exp[:6]}")

worst.sort(reverse=True)
print("\n=== worst 8 nodes by rel error ===")
for rel, cos, name in worst[:8]:
    print(f"  rel={rel:.4g} cos={cos:.5f}  {name}")

if "output0" in man and "output0" in ref:
    got = load_dump("output0"); exp = ref["output0"].astype(np.float32).reshape(-1)
    if got.size == exp.size:
        rel = np.abs(got - exp).max() / (np.abs(exp).max() + 1e-6)
        print(f"\noutput0 relmax={rel:.4g}  {'PASS' if rel < 1e-2 else 'FAIL'}")
