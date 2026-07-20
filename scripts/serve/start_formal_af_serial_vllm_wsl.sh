#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
base_model="$project_root/models/google-gemma-4-12B-bnb-nf4"
port="8000"
max_model_len="2048"
gpu_memory_utilization="0.82"
api_key=""
print_command="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) base_model="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --print-command) print_command="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$max_model_len" != "1024" && "$max_model_len" != "2048" ]]; then
  echo "RTX 3080 Ti formal profiles allow max-model-len 1024 or 2048 only" >&2
  exit 2
fi
if [[ ! -f "$base_model/anchor_quantization_manifest.json" ]]; then
  echo "Frozen NF4 quantization manifest is missing: $base_model" >&2
  exit 2
fi
if [[ "$print_command" == "true" && -n "$api_key" ]]; then
  echo "--print-command refuses --api-key to prevent secret disclosure" >&2
  exit 2
fi

vllm_args=(
  serve "$base_model"
  --host 127.0.0.1
  --port "$port"
  --served-model-name gemma4-12b-base-q4
  --enable-lora
  --max-loras 1
  --max-cpu-loras 1
  --max-lora-rank 16
  --max-model-len "$max_model_len"
  --max-num-seqs 1
  --gpu-memory-utilization "$gpu_memory_utilization"
  --kv-cache-dtype auto
  --language-model-only
  --enforce-eager
  --no-enable-prefix-caching
  --no-enable-chunked-prefill
  --quantization bitsandbytes
  --load-format bitsandbytes
)
if [[ -n "$api_key" ]]; then
  vllm_args+=(--api-key "$api_key")
fi

if [[ "$print_command" == "true" ]]; then
  printf 'VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_V2_MODEL_RUNNER=0 vllm '
  printf '%q ' "${vllm_args[@]}"
  printf '\n'
  exit 0
fi

echo "Starting local-only formal A-F vLLM with one-active-LoRA runtime updates."
echo "Base=$base_model max_loras=1 max_cpu_loras=1 max_model_len=$max_model_len"
bash "$script_dir/probe_vram_wsl.sh" --label formal_af_pre_start
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# vLLM 0.25.1 selects its V2 GPU model runner for this dense architecture.
# That runner requires UVA, while vLLM deliberately disables pinned memory by
# default under WSL2. The formal compatibility path uses the supported V1
# runner override; this avoids UVA without mutating the installed vLLM package.
export VLLM_USE_V2_MODEL_RUNNER=0
exec vllm "${vllm_args[@]}"
