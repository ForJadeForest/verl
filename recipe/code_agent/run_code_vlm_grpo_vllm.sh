#!/bin/bash

set -x

# export LLM_AS_A_JUDGE_BASE="http://28.12.131.135:8000/v1"
# export WANDB_API_KEY=070da1234f26b6663537b7c68d1084b0eaecd342
# export WANDB_PROJECT="CodeAgent"
# export NCCL_IB_GID_INDEX="3"
# export NCCL_IB_SL="3"
# export NCCL_CHECKS_DISABLE="1"
# export NCCL_P2P_DISABLE="0"
# export NCCL_IB_DISABLE="0"
# export NCCL_LL_THRESHOLD="16384"
# export NCCL_IB_CUDA_SUPPORT="1"
# export NCCL_SOCKET_IFNAME="bond1"
# export UCX_NET_DEVICES="bond1"
# export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
# export NCCL_COLLNET_ENABLE="0"
# export SHARP_COLL_ENABLE_SAT="0"
# export NCCL_NET_GDR_LEVEL="2"
# export NCCL_IB_QPS_PER_CONNECTION="4"
# export NCCL_IB_TC="160"
# export NCCL_PXN_DISABLE="0"
# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"
# wandb login



PROJECT_NAME="CodeAgent"
EXPERIMENT_NAME="Qwen2.5-VL-7B-CropMath-Thyme-NoCode-Maze-ThymeCode-AgentLoop"

BASEDIR=/apdcephfs_gy5/share_303588738/yingzhepeng/results/verl_checkpoints
SAVE_CHECKPOINT_DIR=${BASEDIR}
SAVE_DIR="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

mkdir -p ${SAVE_DIR}
mkdir -p ${SAVE_DIR}/logs
mkdir -p ${SAVE_DIR}/rollout_data

TRAIN_DATASET_DIR="/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train"
VAL_DATASET_DIR="/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val"

DEEPEYES_TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR}/deepeyes_train_47k.parquet"
GROUND_R1_TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR}/ground_r1_32k.parquet"



DATASET_TRAIN=${DEEPEYES_TRAIN_DATASET_DIR}
DATASET_VAL=${VAL_DATASET_DIR}/lmm_eval_lite_v2.parquet
   
REF_MODEL_PATH=/root/models/Qwen2.5-VL-7B-CropMath-Thyme-NoCode-Maze-ThymeCode

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=/mnt/private_yingzhepeng/code/rl/verl/recipe/code_agent/configs \
    --config-name='code_vlm_multiturn_grpo' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=128 \
    data.max_prompt_length=9000 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=5000 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/mnt/private_yingzhepeng/code/rl/verl/recipe/code_agent/configs/jupyter_run_config.yaml \
    +reward_model.log_reward_keys=['acc_reward','format_reward','acc','code_error_reward'] \
    +reward_model.reward_mask=-100 \
    +trainer.wandb_proxy=http://star-proxy.oa.com:3128 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.rollout_data_dir=${SAVE_DIR}/rollout_data \
    trainer.log_val_generations=50 \
    trainer.nnodes=4 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_DIR} \
    +trainer.tensorboard_dir=${SAVE_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_DIR}/logs/rl_logging_board \
    trainer.total_epochs=10 2>&1 | tee ${SAVE_DIR}/logs/${EXPERIMENT_NAME}.log



cp ${SAVE_DIR}/logs/${EXPERIMENT_NAME}.log .logs/${EXPERIMENT_NAME}/train.log

echo "Please see the train.log for more details: ${SAVE_DIR}/logs/${EXPERIMENT_NAME}.log or ./logs/${EXPERIMENT_NAME}/train.log"

cp -r ${SAVE_DIR}/rollout_data ./logs/${EXPERIMENT_NAME}/rollout_data

echo "Please see the rollout_data for more details: ${SAVE_DIR}/rollout_data or ./logs/${EXPERIMENT_NAME}/rollout_data"
