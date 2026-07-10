#!/usr/bin/env bash
set -euo pipefail

label="manual"
if [[ "${1:-}" == "--label" ]]; then
  label="${2:-manual}"
fi

echo "anchor_vram_probe label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader || true
