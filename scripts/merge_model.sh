#!/usr/bin/env bash
# Convert a single FSDP checkpoint to HuggingFace format.
# Usage: BASE_DIR=/path/to/run STEP=60 bash merge_model.sh
set -euo pipefail

: "${BASE_DIR:?Set BASE_DIR (e.g. \$PERCEVAL_RESULTS_DIR/perceval-3b-vstar-trm)}"
: "${STEP:?Set STEP (global step number to merge)}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOCAL_DIR="${BASE_DIR}/global_step_${STEP}/actor"
TARGET_DIR="${LOCAL_DIR}/huggingface"

mkdir -p "${TARGET_DIR}"

python "${SCRIPT_DIR}/legacy_model_merger.py" merge \
    --backend fsdp \
    --tie-word-embedding \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"
