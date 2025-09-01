#!/bin/bash
set -x

MODEL_PATHS=(
    "/root/models/modelA"
    "/root/models/modelB"
    "/root/models/modelC"
)

SANDBOX_URL="http://29.157.70.224:8080"
JUDGE_MODEL_API="http://28.12.131.135:8000/v1"
EVAL_API_URL="http://${LOCAL_IP}:8000/v1"
NUM_WORKERS=12
HRBENCH_BENCH_PATH="./data/hrbench"
VSTAR_BENCH_PATH="./data/vstar_bench"

echo "EVAL_API_URL: ${EVAL_API_URL}"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "========== Running eval for ${MODEL_NAME} =========="

    # 启动 vllm serve
    vllm serve "${MODEL_PATH}" \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --served-model-name "${MODEL_NAME}" &

    # 等待服务启动（可按需要调大）
    sleep 20

    # vstar 基准测试
    python -m evaluate.vstar.vstar \
        --eval_api_url "${EVAL_API_URL}" \
        --judge_api_url "${JUDGE_MODEL_API}" \
        --vstar_bench_path "${VSTAR_BENCH_PATH}" \
        --use_code_tool \
        --sandbox_url "${SANDBOX_URL}" \
        --num_workers "${NUM_WORKERS}" &

    # hrbench 基准测试
    python -m evaluate.hrbench.hrbench \
        --eval_api_url "${EVAL_API_URL}" \
        --judge_api_url "${JUDGE_MODEL_API}" \
        --hrbench_path "${HRBENCH_BENCH_PATH}" \
        --use_code_tool \
        --sandbox_url "${SANDBOX_URL}" \
        --num_workers "${NUM_WORKERS}" &

    wait
    # 停掉服务，等待释放
    pkill -f "vllm serve"
    sleep 30
done