#!/usr/bin/env bash
# Serve a Perceval PRM (a VLM fine-tuned to localize perceptual errors in a
# response) over an OpenAI-compatible endpoint. The training script reads
# PRM_BASE (space-separated list of base URLs) and load-balances across
# replicas, so launch one of these per replica.
#
# Required env:
#   PERCEVAL_PRM_MODEL_PATH    -- PRM checkpoint to serve
#   CUDA_VISIBLE_DEVICES       -- GPU(s) to bind
#   PORT                       -- HTTP port (defaults to 12298)

set -euo pipefail

: "${PERCEVAL_PRM_MODEL_PATH:?Set PERCEVAL_PRM_MODEL_PATH (path to a PRM checkpoint)}"
PORT="${PORT:-12298}"
GPU_MEM_UTIL="${PERCEVAL_PRM_GPU_MEM:-0.8}"
MAX_MODEL_LEN="${PERCEVAL_PRM_MAX_LEN:-32768}"
TP="${PERCEVAL_PRM_TP:-1}"

LOG_DIR="${PERCEVAL_LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

exec vllm serve "${PERCEVAL_PRM_MODEL_PATH}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP}" \
    --served-model-name prm \
    --trust-remote-code \
    --uvicorn-log-level info \
    --host 0.0.0.0 \
    2>&1 | tee "${LOG_DIR}/prm_server_${PORT}.log"
