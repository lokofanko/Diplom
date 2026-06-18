#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
mkdir -p logs data/generated experiments/VAE_MBL_Neural_nZBG_BRSA

for n in 1 2; do
  echo "=== $(date) | start BR-SAScore target_n_zbg=${n} ==="
  "${PYTHON_BIN}" scripts/vae_finetune_mbl_neural_nzbg_brsascore.py \
    --config "configs/finetune_vae_mbl_neural_nzbg_brsa_${n}.yaml" \
    2>&1 | tee "logs/vae_mbl_neural_nzbg_brsa_${n}.log"
  echo "=== $(date) | done BR-SAScore target_n_zbg=${n} ==="
done
