// Host ggml executor for the YOLO11n op-program (validation build, CPU backend).
// Loads out/yolo.gguf + out/yolo.prog, runs the graph, dumps every node's
// output to out/dump/ for per-node comparison against the ORT reference.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

static std::string DIR;

struct Op {
    std::string type;
    std::vector<std::string> outs, ins;
    std::map<std::string, std::string> attr;
};

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> v; std::string t; std::stringstream ss(s);
    while (std::getline(ss, t, d)) if (!t.empty()) v.push_back(t);
    return v;
}
static std::vector<int> ints(const std::string& s) {
    std::vector<int> v; for (auto& x : split(s, ',')) v.push_back(std::atoi(x.c_str()));
    return v;
}

// parse "a=1,2;b=3" -> map
static std::map<std::string,std::string> parse_attr(const std::string& s) {
    std::map<std::string,std::string> m;
    for (auto& kv : split(s, ';')) {
        auto p = kv.find('=');
        if (p != std::string::npos) m[kv.substr(0,p)] = kv.substr(p+1);
    }
    return m;
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "out";

    // ---- load weights from gguf ----
    struct ggml_context* wctx = nullptr;
    struct gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &wctx };
    std::string gguf_path = DIR + "/yolo.gguf";
    struct gguf_context* gctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gctx) { fprintf(stderr, "failed to load %s\n", gguf_path.c_str()); return 1; }
    std::map<std::string, ggml_tensor*> T;
    for (ggml_tensor* t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
        T[t->name] = t;
    fprintf(stderr, "loaded %zu weight tensors\n", T.size());

    // ---- compute context ----
    size_t mem = (size_t)3 * 1024 * 1024 * 1024;
    struct ggml_init_params cp = { mem, nullptr, /*no_alloc*/ false };
    struct ggml_context* ctx = ggml_init(cp);

    // ---- input ----
    ggml_tensor* img = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 640, 640, 3, 1);
    ggml_set_name(img, "images");
    {
        std::ifstream f(DIR + "/input.bin", std::ios::binary);
        f.read((char*)img->data, ggml_nbytes(img));
    }
    T["images"] = img;

    auto need = [&](const std::string& n) -> ggml_tensor* {
        auto it = T.find(n);
        if (it == T.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(2); }
        return it->second;
    };

    // ---- parse program ----
    std::vector<Op> ops;
    std::ifstream pf(DIR + "/yolo.prog");
    std::string line;
    while (std::getline(pf, line)) {
        if (line.empty() || line[0] == '#') continue;
        // "TYPE outs <- ins | attrs"
        std::stringstream ss(line);
        Op op;
        ss >> op.type;
        std::string rest; std::getline(ss, rest);
        auto arrow = rest.find("<-");
        auto bar = rest.find('|');
        std::string outs = rest.substr(0, arrow);
        std::string inss = rest.substr(arrow + 2, bar - (arrow + 2));
        std::string attrs = rest.substr(bar + 1);
        // trim + split by comma
        auto clean = [](std::string s){ std::string o; for(char c:s) if(c!=' ') o+=c; return o; };
        op.outs = split(clean(outs), ',');
        std::string ci = clean(inss);
        if (ci != "-") op.ins = split(ci, ',');
        op.attr = parse_attr(clean(attrs));
        ops.push_back(op);
    }
    fprintf(stderr, "parsed %zu ops\n", ops.size());

    auto CONT = [&](ggml_tensor* t){ return ggml_cont(ctx, t); };

    std::vector<std::pair<std::string, ggml_tensor*>> dumps;

    for (auto& op : ops) {
        ggml_tensor* y = nullptr;
        auto& A = op.attr;
        if (op.type == "Conv") {
            ggml_tensor* x = need(op.ins[0]);
            ggml_tensor* w = need(op.ins[1]);
            auto s = ints(A["s"]); auto p = ints(A["p"]);
            int dw = A.count("dw") ? std::atoi(A["dw"].c_str()) : 0;
            if (dw) {
                // Direct depthwise (GGML_OP_CONV_2D_DW): F32 kernel, dedicated
                // shader. Avoids ggml_conv_2d_dw's F16 im2col + mul_mat_vec_f16
                // path, whose shader crashes the Adreno pipeline compiler.
                y = ggml_conv_2d_dw_direct(ctx, w, x, s[0], s[1], p[0], p[1], 1, 1);
            } else {
                y = ggml_conv_2d(ctx, w, x, s[0], s[1], p[0], p[1], 1, 1);
            }
            if (op.ins.size() > 2) {
                ggml_tensor* b = need(op.ins[2]);
                ggml_tensor* br = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
                y = ggml_add(ctx, y, br);
            }
        } else if (op.type == "Sigmoid") {
            y = ggml_sigmoid(ctx, need(op.ins[0]));
        } else if (op.type == "Mul") {
            y = ggml_mul(ctx, need(op.ins[0]), need(op.ins[1]));
        } else if (op.type == "Add") {
            y = ggml_add(ctx, need(op.ins[0]), need(op.ins[1]));
        } else if (op.type == "Sub") {
            y = ggml_sub(ctx, need(op.ins[0]), need(op.ins[1]));
        } else if (op.type == "Div") {
            y = ggml_div(ctx, need(op.ins[0]), need(op.ins[1]));
        } else if (op.type == "Concat") {
            int dim = std::atoi(A["dim"].c_str());
            y = need(op.ins[0]);
            for (size_t k = 1; k < op.ins.size(); k++)
                y = ggml_concat(ctx, y, need(op.ins[k]), dim);
        } else if (op.type == "MaxPool") {
            auto k = ints(A["k"]); auto s = ints(A["s"]); auto p = ints(A["p"]);
            y = ggml_pool_2d(ctx, need(op.ins[0]), GGML_OP_POOL_MAX,
                             k[0], k[1], s[0], s[1], (float)p[0], (float)p[1]);
        } else if (op.type == "Reshape") {
            auto ne = ints(A["ne"]);
            y = ggml_reshape_4d(ctx, CONT(need(op.ins[0])), ne[0], ne[1], ne[2], ne[3]);
        } else if (op.type == "Transpose") {
            auto pm = ints(A["perm"]);
            y = CONT(ggml_permute(ctx, need(op.ins[0]), pm[0], pm[1], pm[2], pm[3]));
        } else if (op.type == "Softmax") {
            int dim = std::atoi(A["dim"].c_str());
            ggml_tensor* x = need(op.ins[0]);
            if (dim == 0) y = ggml_soft_max(ctx, x);
            else {
                int ax[4] = {0,1,2,3}; ax[0] = dim; ax[dim] = 0;
                ggml_tensor* xp = CONT(ggml_permute(ctx, x, ax[0], ax[1], ax[2], ax[3]));
                ggml_tensor* sm = ggml_soft_max(ctx, xp);
                y = CONT(ggml_permute(ctx, sm, ax[0], ax[1], ax[2], ax[3]));
            }
        } else if (op.type == "Resize") {
            y = ggml_upscale(ctx, need(op.ins[0]), 2, GGML_SCALE_MODE_NEAREST);
        } else if (op.type == "Split") {
            int dim = std::atoi(A["dim"].c_str());
            auto sizes = ints(A["sizes"]);
            ggml_tensor* x = need(op.ins[0]);
            size_t off = 0;
            for (size_t k = 0; k < op.outs.size(); k++) {
                int64_t ne[4] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
                ne[dim] = sizes[k];
                ggml_tensor* v = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3],
                                              x->nb[1], x->nb[2], x->nb[3],
                                              off * x->nb[dim]);
                off += sizes[k];
                ggml_tensor* vc = CONT(v);
                T[op.outs[k]] = vc;
                dumps.push_back({op.outs[k], vc});
            }
            continue;
        } else if (op.type == "Slice") {
            int dim = std::atoi(A["dim"].c_str());
            int st = std::atoi(A["start"].c_str());
            int en = std::atoi(A["end"].c_str());
            ggml_tensor* x = need(op.ins[0]);
            int64_t ne[4] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
            ne[dim] = en - st;
            ggml_tensor* v = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], (size_t)st * x->nb[dim]);
            y = CONT(v);
        } else if (op.type == "MatMul") {
            // ONNX C = A @ B  ->  ggml_mul_mat(cont(transpose(Bg)), Ag)
            ggml_tensor* Ag = need(op.ins[0]);
            ggml_tensor* Bg = need(op.ins[1]);
            ggml_tensor* Bt = CONT(ggml_transpose(ctx, Bg));
            y = ggml_mul_mat(ctx, Bt, Ag);
        } else {
            fprintf(stderr, "unhandled op %s\n", op.type.c_str());
            return 3;
        }
        ggml_set_name(y, op.outs[0].c_str());
        T[op.outs[0]] = y;
        dumps.push_back({op.outs[0], y});
    }

    ggml_tensor* out = need("output0");
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, out);
    // ensure all dumped nodes are computed
    for (auto& d : dumps) ggml_build_forward_expand(graph, d.second);

    int nth = 4;
    ggml_graph_compute_with_ctx(ctx, graph, nth);
    fprintf(stderr, "computed. output0 ne = %lld,%lld,%lld,%lld\n",
            (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3]);

    // ---- dump ----
    std::string dd = DIR + "/dump";
    mkdir(dd.c_str(), 0755);
    std::ofstream man(dd + "/manifest.txt");
    for (auto& d : dumps) {
        ggml_tensor* t = d.second;
        std::string safe = d.first;
        for (char& c : safe) if (c == '/') c = '_';
        std::ofstream f(dd + "/" + safe + ".bin", std::ios::binary);
        f.write((char*)t->data, ggml_nbytes(t));
        man << d.first << " " << t->ne[0] << "," << t->ne[1] << ","
            << t->ne[2] << "," << t->ne[3] << "\n";
    }
    fprintf(stderr, "dumped %zu tensors to %s\n", dumps.size(), dd.c_str());
    return 0;
}
