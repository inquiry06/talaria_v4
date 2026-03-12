#!/usr/bin/env bash
# =============================================================================
# TALARIA Phase 1: Self-Supervised Pre-training
# =============================================================================
set -euo pipefail

# --- Environment ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# --- Configuration ---
CONFIG="${1:-configs/pretrain.yaml}"

# --- GPU Setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[TALARIA] Phase 1: Pre-training"
echo "[TALARIA] Config:  ${CONFIG}"
echo "[TALARIA] GPU:     ${CUDA_VISIBLE_DEVICES}"
echo "[TALARIA] Started: $(date)"

# --- Run ---
python -m src.training.pretrain \
    --config "${CONFIG}"

echo "[TALARIA] Phase 1 complete: $(date)"
