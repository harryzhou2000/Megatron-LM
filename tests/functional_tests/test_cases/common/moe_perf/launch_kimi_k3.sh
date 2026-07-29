#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd)"

num_gpus="${NUM_GPUS:-8}"
expert_parallel_size="${EXPERT_PARALLEL_SIZE:-8}"
num_experts="${NUM_EXPERTS:-8}"
router_topk="${ROUTER_TOPK:-1}"
run_unit_tests="${RUN_UNIT_TESTS:-1}"
dispatcher_backend="${DISPATCHER_BACKEND:-hybridep}"
bias_update_method="${BIAS_UPDATE_METHOD:-quantile}"
qb_num_bins="${QB_NUM_BINS:-1000}"
full_iter_cuda_graph="${FULL_ITER_CUDA_GRAPH:-0}"
paged_stash="${PAGED_STASH:-0}"
expert_rank_capacity_factor="${EXPERT_RANK_CAPACITY_FACTOR:-}"
log_dir="${KIMI_K3_LOG_DIR:-${repo_root}/logs/moe_perf}"
log_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${KIMI_K3_LOG_FILE:-${log_dir}/kimi_k3_${dispatcher_backend}_ep${expert_parallel_size}_${bias_update_method}_${log_timestamp}.log}"

mkdir -p "${log_dir}"
exec > >(tee -a "${log_file}") 2>&1
echo "[kimi_k3] log_file=${log_file}"

if [[ "${num_gpus}" -ne "${expert_parallel_size}" ]]; then
    echo "This no-TP launch requires NUM_GPUS to equal EXPERT_PARALLEL_SIZE." >&2
    exit 2
fi
if (( num_experts % expert_parallel_size != 0 )); then
    echo "NUM_EXPERTS must be divisible by EXPERT_PARALLEL_SIZE." >&2
    exit 2
fi
if (( router_topk <= 0 || router_topk >= num_experts )); then
    echo "ROUTER_TOPK must be greater than zero and less than NUM_EXPERTS." >&2
    exit 2
fi
if [[ "${bias_update_method}" != "quantile" && "${bias_update_method}" != "sign" ]]; then
    echo "BIAS_UPDATE_METHOD must be 'quantile' or 'sign'." >&2
    exit 2
fi
if (( qb_num_bins <= 0 )); then
    echo "QB_NUM_BINS must be greater than zero." >&2
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
        )
        ;;
    *)
        echo "DISPATCHER_BACKEND must be 'alltoall' or 'hybridep'." >&2
        exit 2
        ;;
esac

graph_args=()
if [[ "${full_iter_cuda_graph}" == "1" ]]; then
    graph_args+=(--moe-perf-full-iter-cuda-graph)
fi
if [[ "${paged_stash}" == "1" ]]; then
    graph_args+=(--moe-perf-paged-stash)
fi
if [[ -n "${expert_rank_capacity_factor}" ]]; then
    graph_args+=(
        --moe-perf-expert-rank-capacity-factor
        "${expert_rank_capacity_factor}"
    )
fi

if [[ "${run_unit_tests}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 pytest -q -s \
        tests/unit_tests/fusions/test_cutedsl_situ_glu.py \
        tests/unit_tests/transformer/moe/test_grouped_mlp.py::test_situ_glu_activation_flag_aliases \
        tests/unit_tests/transformer/moe/test_latent_moe_layer.py::TestLatentMoELayer::test_latent_up_projection_rmsnorm
    CUDA_VISIBLE_DEVICES=0 pytest -q -s \
        tests/unit_tests/transformer/moe/test_routers.py -k quantile
    CUDA_VISIBLE_DEVICES=0 pytest -q -s \
        tests/unit_tests/distributed/test_finalize_model_grads.py::TestFinalizeModelGradsMoEExpertBias::test_finalize_model_grads_updates_quantile_bias_bounds_and_resets_histogram
fi

echo "[kimi_k3] num_gpus=${num_gpus} ep=${expert_parallel_size} experts=${num_experts} topk=${router_topk} bias_update_method=${bias_update_method} qb_num_bins=${qb_num_bins} full_iter_cuda_graph=${full_iter_cuda_graph} paged_stash=${paged_stash} expert_rank_capacity_factor=${expert_rank_capacity_factor:-none}"

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
    --moe-router-topk "${router_topk}" \
    --moe-router-pre-softmax \
    --moe-router-score-function sigmoid \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type none \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-method "${bias_update_method}" \
    --moe-router-qb-num-bins "${qb_num_bins}" \
    --moe-router-fusion \
    "${dispatcher_args[@]}" \
    "${graph_args[@]}" \
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
