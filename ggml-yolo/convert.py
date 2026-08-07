#!/usr/bin/env python3
"""Convert a YOLO11n ONNX export into a ggml-runnable form.

Emits, into ggml-yolo/out/:
  - yolo.gguf   : all weight/constant tensors, stored in ggml layout
                  (ggml ne = reversed ONNX dims), f32
  - yolo.prog   : flat op-program, one node per line, concrete attributes
  - ref.npz     : ONNX Runtime output for EVERY node (per-node validation)

Dimension convention everywhere: a tensor with ONNX logical shape
[d0,d1,...,dk] (row-major, last fastest) becomes ggml ne = [dk,...,d1,d0]
(ne[0] fastest). ONNX axis a on an r-dim tensor -> ggml dim (r-1-a).
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import onnxruntime as ort
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/Volumes/DATA/workspace/vlm-bench/runs/detect/yolo_runs/skip_v2/weights/last.onnx"
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

model = onnx.load(SRC)
model = shape_inference.infer_shapes(model)
g = model.graph

# ---- gather initializers and Constant-node outputs as numpy ----
tensors = {}  # name -> np.ndarray (ONNX logical layout)
for init in g.initializer:
    tensors[init.name] = numpy_helper.to_array(init)

for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                tensors[n.output[0]] = numpy_helper.to_array(a.t)

# ONNX logical shapes for every value (for reshape/-1 resolution & validation)
shapes = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    dims = [d.dim_value if (d.dim_value > 0) else 1 for d in vi.type.tensor_type.shape.dim]
    shapes[vi.name] = dims
for name, arr in tensors.items():
    shapes[name] = list(arr.shape)
shapes["images"] = [1, 3, 640, 640]

def onnx_to_ne(dims):
    """Reverse ONNX dims -> ggml ne, padded to 4 with 1s."""
    ne = list(reversed(dims))
    while len(ne) < 4:
        ne.append(1)
    return ne[:4]

# ---- run ORT with every node output exposed, for per-node reference ----
all_outs = []
for n in g.node:
    all_outs.extend(o for o in n.output if o)
extra = [o for o in all_outs if o not in {x.name for x in g.output}]
vinfos = [onnx.helper.make_empty_tensor_value_info(o) for o in extra]
model.graph.output.extend(vinfos)
onnx.save(model, os.path.join(OUT, "_allout.onnx"))

rng = np.random.default_rng(0)
img = rng.random((1, 3, 640, 640), dtype=np.float32)
sess = ort.InferenceSession(os.path.join(OUT, "_allout.onnx"),
                            providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
vals = sess.run(out_names, {"images": img})
ref = {k: v for k, v in zip(out_names, vals)}
ref["images"] = img
np.savez(os.path.join(OUT, "ref.npz"), **{k.replace("/", "__"): v for k, v in ref.items()})
# raw input for the C++ executor (same bytes ORT saw): ggml layout == C-order here
img.astype(np.float32).tofile(os.path.join(OUT, "input.bin"))
print(f"ORT reference: {len(ref)} tensors, output0 shape {ref['output0'].shape}")

# ---- decide which tensors are runtime graph tensors (go into gguf) ----
# A `tensors` entry is a runtime input to some op if it's consumed as a data
# operand (not as a shape/param arg). Shape/param args are resolved to attrs.
PARAM_INPUTS = {  # op -> set of input indices that are shape/param args (skip)
    "Reshape": {1}, "Resize": {1, 2, 3}, "Slice": {1, 2, 3, 4},
    "Split": {1}, "Unsqueeze": {1}, "Squeeze": {1},
}
runtime_const = set()
for n in g.node:
    for i, inp in enumerate(n.input):
        if not inp or inp not in tensors:
            continue
        if i in PARAM_INPUTS.get(n.op_type, set()):
            continue
        runtime_const.add(inp)

# ---- write gguf ----
writer = gguf.GGUFWriter(os.path.join(OUT, "yolo.gguf"), "yolo11n")
for name in sorted(runtime_const):
    arr = tensors[name].astype(np.float32)
    ne = onnx_to_ne(list(arr.shape))
    # ggml stores ne[0] fastest; ONNX row-major already has last-dim fastest,
    # and ne is reversed dims, so the raw C-order bytes match ggml layout.
    flat = np.ascontiguousarray(arr).reshape(-1)
    writer.add_tensor(name, flat, raw_shape=list(reversed(ne)))  # gguf wants ONNX-order dims
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"gguf: {len(runtime_const)} runtime tensors")

# ---- helpers to resolve param args ----
def const(name):
    return tensors[name]

def axis_to_dim(axis, rank):
    if axis < 0:
        axis += rank
    return rank - 1 - axis

# ---- emit program ----
lines = []
def emit(op, outs, ins, attrs=""):
    lines.append(f"{op} {','.join(outs)} <- {','.join(ins) if ins else '-'} | {attrs}")

for n in g.node:
    op = n.op_type
    if op == "Constant":
        continue
    ins = list(n.input)
    outs = list(n.output)
    if op == "Conv":
        a = {x.name: x for x in n.attribute}
        s = list(a["strides"].ints) if "strides" in a else [1, 1]
        p = list(a["pads"].ints) if "pads" in a else [0, 0, 0, 0]
        grp = a["group"].i if "group" in a else 1
        real_ins = [ins[0], ins[1]] + ([ins[2]] if len(ins) > 2 else [])
        oc = shapes[ins[1]][0]
        dw = 1 if grp == oc and grp != 1 else 0
        emit("Conv", outs, real_ins,
             f"s={s[0]},{s[1]};p={p[0]},{p[1]};dw={dw}")
    elif op in ("Sigmoid",):
        emit("Sigmoid", outs, ins)
    elif op in ("Mul", "Add", "Sub", "Div", "MatMul"):
        emit(op, outs, ins)
    elif op == "Concat":
        ax = [x for x in n.attribute if x.name == "axis"][0].i
        rank = len(shapes.get(outs[0], shapes.get(ins[0], [0, 0, 0, 0])))
        emit("Concat", outs, ins, f"dim={axis_to_dim(ax, rank)}")
    elif op == "MaxPool":
        a = {x.name: x for x in n.attribute}
        k = list(a["kernel_shape"].ints)
        s = list(a["strides"].ints) if "strides" in a else [1, 1]
        p = list(a["pads"].ints) if "pads" in a else [0, 0, 0, 0]
        emit("MaxPool", outs, [ins[0]], f"k={k[0]},{k[1]};s={s[0]},{s[1]};p={p[0]},{p[1]}")
    elif op == "Reshape":
        target = list(const(ins[1]).astype(np.int64))
        # resolve 0 (=copy) and -1 (=infer) against input logical shape
        insh = shapes[ins[0]]
        tot = int(np.prod(insh))
        res = []
        for i, d in enumerate(target):
            res.append(int(insh[i]) if d == 0 else int(d))
        if -1 in res:
            known = int(np.prod([d for d in res if d != -1]))
            res[res.index(-1)] = tot // known
        ne = onnx_to_ne(res)
        emit("Reshape", outs, [ins[0]], f"ne={','.join(map(str, ne))}")
    elif op == "Transpose":
        perm = list([x for x in n.attribute if x.name == "perm"][0].ints)
        r = len(perm)
        # ggml_permute(a, axis0..3): source ggml dim i lands at position axis_i.
        # ONNX perm p: output onnx dim j = input onnx dim p[j]. With ggml dim
        # g = r-1-onnx_dim, source ggml dim i maps to dst ggml dim r-1-j where
        # p[j] == r-1-i.
        gperm = [r - 1 - perm.index(r - 1 - i) for i in range(r)]
        while len(gperm) < 4:
            gperm.append(len(gperm))
        emit("Transpose", outs, [ins[0]], f"perm={','.join(map(str, gperm))}")
    elif op == "Softmax":
        ax = [x for x in n.attribute if x.name == "axis"][0].i
        rank = len(shapes.get(ins[0], [0, 0, 0, 0]))
        emit("Softmax", outs, [ins[0]], f"dim={axis_to_dim(ax, rank)}")
    elif op == "Resize":
        # nearest 2x upsample (scales const = [1,1,2,2])
        emit("Resize", outs, [ins[0]], "scale=2")
    elif op == "Split":
        a = {x.name: x for x in n.attribute}
        ax = a["axis"].i if "axis" in a else 0
        if "split" in a:
            sizes = list(a["split"].ints)
        elif len(ins) > 1 and ins[1] in tensors:
            sizes = list(const(ins[1]).astype(np.int64))
        else:
            rank = len(shapes[ins[0]])
            total = shapes[ins[0]][ax]
            sizes = [total // len(outs)] * len(outs)
        rank = len(shapes[ins[0]])
        emit("Split", outs, [ins[0]],
             f"dim={axis_to_dim(ax, rank)};sizes={','.join(map(str, sizes))}")
    elif op == "Slice":
        starts = const(ins[1]).astype(np.int64)
        ends = const(ins[2]).astype(np.int64)
        axes = const(ins[3]).astype(np.int64) if len(ins) > 3 else np.arange(len(starts))
        rank = len(shapes[ins[0]])
        # single-axis slices only (true for this graph)
        assert len(axes) == 1, f"multi-axis slice unsupported: {n.name}"
        ax = int(axes[0]); st = int(starts[0]); en = int(ends[0])
        dimlen = shapes[ins[0]][ax]
        en = min(en, dimlen)
        emit("Slice", outs, [ins[0]], f"dim={axis_to_dim(ax, rank)};start={st};end={en}")
    else:
        raise SystemExit(f"unhandled op {op} ({n.name})")

with open(os.path.join(OUT, "yolo.prog"), "w") as f:
    f.write(f"# input images ne=640,640,3,1 ; output output0\n")
    f.write("\n".join(lines) + "\n")
print(f"program: {len(lines)} ops -> out/yolo.prog")
