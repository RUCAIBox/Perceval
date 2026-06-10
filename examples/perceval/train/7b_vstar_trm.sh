#!/usr/bin/env bash
# 7B variant: Qwen2.5-VL-7B-Instruct, 8 GPUs, val_before_train disabled.
# Equivalent to the original 7b_train/qwen7b_base-vstar-trm-7b.sh.
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset no_proxy || true

TBS=128
UBS=32
N=8
TEMP=1.0
N_GPU="${PERCEVAL_N_GPU:-8}"
DYNAMIC_BSZ=False
ADV_ESTIMATOR=trm
PENALTY_PERCENTAGE=0.1

VAL_TEMP=0.0
VAL_N=1
VAL_DO_SAMPLE=False
VAL_BEFORE_TRAIN=False  # 7B run skips val_before_train to save warmup time

REWARD_MANAGER=trm
PROCESS_REWARD=True
MAX_WORKERS="${PERCEVAL_REWARD_WORKERS:-768}"

RUN_NAME="${RUN_NAME:-perceval-7b-vstar-trm}"

python3 -u -m verl.trainer.main_ppo \
    reward_model.reward_manager=${REWARD_MANAGER} \
    +reward_model.reward_kwargs.verify_process=${PROCESS_REWARD} \
    +reward_model.reward_kwargs.record_path=${PERCEVAL_LOG_DIR}/${RUN_NAME} \
    +reward_model.reward_kwargs.max_workers=${MAX_WORKERS} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    +algorithm.penalty_percentage=${PENALTY_PERCENTAGE} \
    data.train_files="${PERCEVAL_TRAIN_DATA}" \
    data.val_files="${PERCEVAL_VAL_DATA}" \
    data.train_batch_size=${TBS} \
    data.max_prompt_length=20000 \
    data.max_response_length=4000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path="${PERCEVAL_MODEL_PATH}" \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${UBS} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.actor.use_dynamic_bsz=${DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=${TEMP} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT:-perceval}" \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir=${PERCEVAL_RESULTS_DIR}/${RUN_NAME} \
    custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    custom_reward_function.name="${REWARD_FUNCTION_NAME}" "$@" \
    2>&1 | tee "${PERCEVAL_LOG_DIR}/${RUN_NAME}.log"
