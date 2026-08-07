#!/bin/zsh
# Download candidate VLM GGUFs for ad-skipper benchmark (via local proxy).
set -u
export https_proxy=http://127.0.0.1:7890
DIR=/Volumes/DATA/workspace/vlm-bench/models
mkdir -p "$DIR"

hf() { echo "https://huggingface.co/$1/resolve/main/$2"; }

# repo|file pairs
FILES=(
  "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF|SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
  "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF|mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
  "ggml-org/LFM2-VL-450M-GGUF|LFM2-VL-450M-Q8_0.gguf"
  "ggml-org/LFM2-VL-450M-GGUF|mmproj-LFM2-VL-450M-Q8_0.gguf"
  "ggml-org/InternVL3-2B-Instruct-GGUF|InternVL3-2B-Instruct-Q4_K_M.gguf"
  "ggml-org/InternVL3-2B-Instruct-GGUF|mmproj-InternVL3-2B-Instruct-Q8_0.gguf"
  "bartowski/Qwen2-VL-2B-Instruct-GGUF|Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
  "bartowski/Qwen2-VL-2B-Instruct-GGUF|mmproj-Qwen2-VL-2B-Instruct-f16.gguf"
  "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF|Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
  "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF|mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
  "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF|mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
  "openbmb/MiniCPM-V-2_6-gguf|ggml-model-Q4_K_M.gguf"
  "openbmb/MiniCPM-V-2_6-gguf|mmproj-model-f16.gguf"
)

dl() {
  local repo=${1%%|*} file=${1##*|}
  local out="$DIR/$file"
  # MiniCPM files have generic names; prefix them
  case $repo in openbmb/*) out="$DIR/minicpm-v26-$file";; esac
  [[ -f "$out.done" ]] && { echo "SKIP $file"; return 0; }
  for i in 1 2 3 4 5; do
    curl -sSL -C - --retry 3 -o "$out" "$(hf "$repo" "$file")" && { touch "$out.done"; echo "OK $file"; return 0; }
    echo "retry $i $file"; sleep 5
  done
  echo "FAIL $file"; return 1
}

# 3 concurrent download slots
i=0
for f in "${FILES[@]}"; do
  dl "$f" &
  (( ++i % 3 == 0 )) && wait
done
wait
echo "ALL DONE"
ls -lh "$DIR"
