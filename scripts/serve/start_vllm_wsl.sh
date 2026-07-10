#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
base_model="$project_root/models/google-gemma-4-12B-base"
frontend_adapter=""
review_adapter=""
security_adapter=""
mixed_adapter=""
quantization="bitsandbytes"
load_format=""
api_key=""
port="8000"
profile="3080ti-safe"
max_model_len=""
gpu_memory_utilization="0.88"
print_command="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) base_model="$2"; shift 2 ;;
    --frontend-adapter) frontend_adapter="$2"; shift 2 ;;
    --review-adapter) review_adapter="$2"; shift 2 ;;
    --security-adapter) security_adapter="$2"; shift 2 ;;
    --mixed-adapter) mixed_adapter="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --quantization) quantization="$2"; shift 2 ;;
    --load-format) load_format="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --print-command) print_command="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$quantization" in
  bitsandbytes)
    if [[ -z "$load_format" ]]; then load_format="bitsandbytes"; fi
    if [[ "$load_format" != "bitsandbytes" ]]; then
      echo "bitsandbytes requires --load-format bitsandbytes" >&2
      exit 2
    fi
    ;;
  compressed-tensors)
    if [[ -z "$load_format" ]]; then load_format="auto"; fi
    if [[ "$load_format" != "auto" ]]; then
      echo "compressed-tensors requires --load-format auto" >&2
      exit 2
    fi
    ;;
  *) echo "Unsupported quantization: $quantization" >&2; exit 2 ;;
esac

case "$profile" in
  3080ti-safe)
    profile_default_model_len="1024"
    profile_args=(
      --enforce-eager
      --no-enable-prefix-caching
      --no-enable-chunked-prefill
    )
    ;;
  throughput)
    profile_default_model_len="2048"
    profile_args=(
      --enable-prefix-caching
      --enable-chunked-prefill
    )
    ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

if [[ -z "$max_model_len" ]]; then
  max_model_len="$profile_default_model_len"
fi
if [[ "$max_model_len" != "1024" && "$max_model_len" != "2048" ]]; then
  echo "RTX 3080 Ti profiles allow max-model-len 1024 or 2048 only" >&2
  exit 2
fi

for name in base_model frontend_adapter review_adapter security_adapter mixed_adapter; do
  if [[ -z "${!name}" ]]; then
    echo "Missing required argument: ${name}" >&2
    exit 2
  fi
done

vllm_args=(
  serve "$base_model"
  --host 127.0.0.1
  --port "$port"
  --served-model-name base-model
  --enable-lora
  --max-loras 1
  --max-cpu-loras 4
  --max-lora-rank 16
  --max-model-len "$max_model_len"
  --max-num-seqs 1
  --gpu-memory-utilization "$gpu_memory_utilization"
  --kv-cache-dtype auto
  --language-model-only
  --lora-modules
  "lora-frontend-gen=$frontend_adapter"
  "lora-code-review=$review_adapter"
  "lora-security-audit=$security_adapter"
  "lora-mixed-all=$mixed_adapter"
)
vllm_args+=("${profile_args[@]}")

# No speculative-config/draft model is accepted by this launcher. Both profiles
# are intentionally text-only and leave KV cache dtype at officially safe auto.

if [[ -n "$quantization" ]]; then
  vllm_args+=(--quantization "$quantization")
fi
vllm_args+=(--load-format "$load_format")
if [[ -n "$api_key" ]]; then
  vllm_args+=(--api-key "$api_key")
fi

if [[ "$print_command" == "true" ]]; then
  printf 'vllm '
  printf '%q ' "${vllm_args[@]}"
  printf '\n'
  exit 0
fi

echo "Starting local-only vLLM inside WSL2."
echo "Profile=$profile max_model_len=$max_model_len max_num_seqs=1 max_lora_rank=16 kv_cache_dtype=auto"
echo "A Hub model id may cause vLLM to download weights when this command is invoked."
bash "$(dirname "$0")/probe_vram_wsl.sh" --label pre_start
exec vllm "${vllm_args[@]}"
