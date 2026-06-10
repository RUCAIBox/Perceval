#!/usr/bin/env bash
# Serve a Qwen2.5-VL model as the LLM-as-judge backing the outcome reward.
# The training script reads LLM_AS_A_JUDGE_BASE (a space-separated list of
# OpenAI-compatible base URLs) and load-balances across replicas, so launch
# one of these per replica with a different PORT / CUDA_VISIBLE_DEVICES.
#
# Required env:
#   PERCEVAL_JUDGE_MODEL_PATH  -- VLM checkpoint to serve as the judge
#   CUDA_VISIBLE_DEVICES       -- GPU(s) to bind (single-GPU by default)
#   PORT                       -- HTTP port (defaults to 9999)

set -euo pipefail

: "${PERCEVAL_JUDGE_MODEL_PATH:?Set PERCEVAL_JUDGE_MODEL_PATH (e.g. Qwen/Qwen2.5-VL-7B-Instruct)}"
PORT="${PORT:-9999}"
GPU_MEM_UTIL="${PERCEVAL_JUDGE_GPU_MEM:-0.8}"
MAX_MODEL_LEN="${PERCEVAL_JUDGE_MAX_LEN:-32768}"
TP="${PERCEVAL_JUDGE_TP:-1}"

LOG_DIR="${PERCEVAL_LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

exec vllm serve "${PERCEVAL_JUDGE_MODEL_PATH}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP}" \
    --served-model-name judge \
    --trust-remote-code \
    --disable-log-requests \
    --host 0.0.0.0 \
    2>&1 | tee "${LOG_DIR}/judge_server_${PORT}.log"
