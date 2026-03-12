#!/usr/bin/env bash
# =============================================================================
# TALARIA Phase 3: Task-Specific Fine-tuning (Frozen Encoder)
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
CONFIG="${1:-configs/finetune.yaml}"
STUDENT_CKPT="${2:-}"

if [ -z "${STUDENT_CKPT}" ]; then
    # Auto-detect latest distillation checkpoint
    STUDENT_CKPT=$(ls -t experiments/distill_*/checkpoints/best.ckpt 2>/dev/null | head -1)
    if [ -z "${STUDENT_CKPT}" ]; then
        echo "[WARNING] No student checkpoint found. Fine-tuning from scratch."
        STUDENT_CKPT=""
    fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[TALARIA] Phase 3: Fine-tuning"
echo "[TALARIA] Config:         ${CONFIG}"
echo "[TALARIA] Student ckpt:   ${STUDENT_CKPT:-'(none - training from scratch)'}"
echo "[TALARIA] GPU:            ${CUDA_VISIBLE_DEVICES}"
echo "[TALARIA] Started:        $(date)"

CKPT_ARG=""
if [ -n "${STUDENT_CKPT}" ]; then
    CKPT_ARG="--student_ckpt ${STUDENT_CKPT}"
fi

python -m src.training.finetune \
    --config "${CONFIG}" \
    ${CKPT_ARG}

echo "[TALARIA] Phase 3 complete: $(date)"
