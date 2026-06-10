# Shared bootstrap used by every Perceval training script. Sourced, not run.
# Resolves the repo root, loads configs/perceval.env if it exists, and exports
# defaults so downstream Hydra overrides can interpolate them.

set -euo pipefail

# Repo root = three levels up from this file (examples/perceval/train/_common.sh).
PERCEVAL_REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
export PERCEVAL_REPO_ROOT

# Load user config if present. Any variable already set in the environment wins
# (env > config file > script default), matching how 12-factor apps behave.
_PERCEVAL_ENV_FILE="${PERCEVAL_ENV_FILE:-${PERCEVAL_REPO_ROOT}/configs/perceval.env}"
if [[ -f "${_PERCEVAL_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "${_PERCEVAL_ENV_FILE}"
    set +a
fi

# Mandatory variables -- fail fast with a useful message instead of letting
# Hydra emit a confusing error 30 seconds in.
require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[perceval] ERROR: required env var '${name}' is not set." >&2
        echo "[perceval] Copy configs/perceval.env.example to configs/perceval.env and fill it in," >&2
        echo "[perceval]   or export ${name} in your shell before launching." >&2
        exit 2
    fi
}

require_var PERCEVAL_MODEL_PATH
require_var PERCEVAL_TRAIN_DATA
require_var PERCEVAL_VAL_DATA
require_var PERCEVAL_RESULTS_DIR
require_var PERCEVAL_LOG_DIR
require_var LLM_AS_A_JUDGE_BASE
require_var PRM_BASE

mkdir -p "${PERCEVAL_RESULTS_DIR}" "${PERCEVAL_LOG_DIR}"

# Reward function lives in the in-tree verl package; we just need its absolute
# path because verl's Hydra config takes a path string.
REWARD_FUNCTION_PATH="${PERCEVAL_REPO_ROOT}/verl/utils/reward_score/hallu_token_reward_vstar.py"
REWARD_FUNCTION_NAME=adaptive_comput_score
export REWARD_FUNCTION_PATH REWARD_FUNCTION_NAME
