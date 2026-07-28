#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd)"

num_gpus="${NUM_GPUS:-8}"
expert_parallel_size="${EXPERT_PARALLEL_SIZE:-8}"
num_experts="${NUM_EXPERTS:-8}"
run_unit_tests="${RUN_UNIT_TESTS:-1}"
dispatcher_backend="${DISPATCHER_BACKEND:-hybridep}"
log_dir="${KIMI_K3_LOG_DIR:-${repo_root}/logs/moe_perf}"
log_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${KIMI_K3_LOG_FILE:-${log_dir}/kimi_k3_${dispatcher_backend}_ep8_${log_timestamp}.log}"

mkdir -p "${log_dir}"
exec > >(tee -a "${log_file}") 2>&1
echo "[kimi_k3] log_file=${log_file}"

if [[ "${num_gpus}" -ne 8 || "${expert_parallel_size}" -ne 8 ]]; then
    echo "Kimi K3 distributed validation requires NUM_GPUS=8 and EXPERT_PARALLEL_SIZE=8." >&2
    exit 2
fi
if (( num_experts % expert_parallel_size != 0 )); then
    echo "NUM_EXPERTS must be divisible by EXPERT_PARALLEL_SIZE." >&2
    exit 2
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export MCORE_DEBUG_DENSE_ROUTING="${MCORE_DEBUG_DENSE_ROUTING:-1}"

case "${dispatcher_backend}" in
    alltoall)
        dispatcher_args=(--moe-token-dispatcher-type alltoall)
        ;;
    hybridep)
        dispatcher_args=(
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend hybridep
            --moe-router-fusion
        )
        ;;
    *)
        echo "DISPATCHER_BACKEND must be 'alltoall' or 'hybridep'." >&2
        exit 2
        ;;
esac

if [[ "${run_unit_tests}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 pytest -q -s \
        tests/unit_tests/fusions/test_cutedsl_situ_glu.py \
        tests/unit_tests/transformer/moe/test_grouped_mlp.py::test_situ_glu_activation_flag_aliases \
        tests/unit_tests/transformer/moe/test_latent_moe_layer.py::TestLatentMoELayer::test_latent_up_projection_rmsnorm
fi

torchrun --standalone --nproc-per-node="${num_gpus}" \
    tests/functional_tests/test_cases/common/moe_perf/recipe_frontend.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --expert-model-parallel-size "${expert_parallel_size}" \
    --num-layers 1 \
    --hidden-size 256 \
    --num-attention-heads 4 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --micro-batch-size 1 \
    --global-batch-size "${num_gpus}" \
    --num-experts "${num_experts}" \
    --moe-router-topk 1 \
    --moe-router-pre-softmax \
    --moe-router-score-function sigmoid \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type none \
    --moe-router-enable-expert-bias \
    "${dispatcher_args[@]}" \
    --moe-grouped-gemm \
    --moe-mlp-glu-interleave-size 32 \
    --moe-ffn-hidden-size 256 \
    --moe-shared-expert-intermediate-size 256 \
    --moe-latent-size 128 \
    --moe-latent-up-projection-rmsnorm \
    --normalization RMSNorm \
    --situ-glu \
    --fp8-format hybrid \
    --fp8-recipe mxfp8 \
    --bf16 \
    --disable-bias-linear \
    --use-transformer-engine-op-fuser \
    --moe-perf-warmup-iters "${MOE_PERF_WARMUP_ITERS:-1}" \
    --moe-perf-iters "${MOE_PERF_ITERS:-1}"
