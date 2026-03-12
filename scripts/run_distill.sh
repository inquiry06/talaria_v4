#!/usr/bin/env bash
# =============================================================================
# TALARIA Phase 2: Knowledge Distillation (Teacher -> Student)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# --- Arguments ---
CONFIG="${1:-configs/distill.yaml}"
TEACHER_CKPT="${2:-}"

if [ -z "${TEACHER_CKPT}" ]; then
    # Auto-detect latest pretrain checkpoint
    TEACHER_CKPT=$(ls -t experiments/pretrain_*/checkpoints/best.ckpt 2>/dev/null | head -1)
    if [ -z "${TEACHER_CKPT}" ]; then
        echo "[ERROR] No teacher checkpoint found. Run Phase 1 first, or pass checkpoint as second argument."
        echo "Usage: bash scripts/run_distill.sh [config] [teacher_ckpt]"
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[TALARIA] Phase 2: Knowledge Distillation"
echo "[TALARIA] Config:          ${CONFIG}"
echo "[TALARIA] Teacher ckpt:    ${TEACHER_CKPT}"
echo "[TALARIA] GPU:             ${CUDA_VISIBLE_DEVICES}"
echo "[TALARIA] Started:         $(date)"

python -m src.training.distill \
    --config "${CONFIG}" \
    --teacher_ckpt "${TEACHER_CKPT}"

echo "[TALARIA] Phase 2 complete: $(date)"
