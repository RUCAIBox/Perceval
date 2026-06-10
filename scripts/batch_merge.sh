#!/usr/bin/env bash
# Merge several FSDP checkpoints in parallel.
# Usage: BASE_DIR=/path/to/run STEPS="60 120 180" bash batch_merge.sh
set -euo pipefail

: "${BASE_DIR:?Set BASE_DIR (e.g. \$PERCEVAL_RESULTS_DIR/perceval-3b-vstar-trm)}"
STEPS="${STEPS:-}"
if [[ -z "${STEPS}" ]]; then
    echo "Set STEPS to a space-separated list of step numbers, e.g. STEPS='60 120 180'." >&2
    exit 2
fi
MAX_WORKERS="${MAX_WORKERS:-6}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "${SCRIPT_DIR}/batch_merge.py" \
    --base_dir "${BASE_DIR}" \
    --steps ${STEPS} \
    --max_workers "${MAX_WORKERS}" \
    --script_path "${SCRIPT_DIR}/legacy_model_merger.py"
