# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone MoE-layer performance frontend.

This entrypoint consumes the normal Megatron full-model command-line arguments,
derives the real GPT or Hybrid MoE layer specs, and runs only those MoE MLP
paths on synthetic hidden states. It intentionally does not build tokenizers,
datasets, embeddings, attention layers, output projections, optimizers, FSDP
wrappers, or pipeline schedules.
"""

from __future__ import annotations

import functools
import os
import statistics
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import fused_topk_with_score_function_supports_qb
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, get_shared_capture_stream
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import RandomSTE
from megatron.core.transformer.moe.paged_stash import (
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module, import_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import configure_nvtx_profiling
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import initialize_megatron
from megatron.training.training import _moe_layer_flops

try:
    from megatron.post_training.arguments import add_modelopt_args

    _HAS_MODELOPT = True
except ImportError:
    add_modelopt_args = None
    _HAS_MODELOPT = False


@dataclass(frozen=True)
class MoELayerSpec:
    """MoE layer spec with its original full-model layer number."""

    layer_number: int
    spec: ModuleSpec
    source: str


def _add_moe_perf_args(parser):
    if _HAS_MODELOPT and add_modelopt_args is not None:
        parser = add_modelopt_args(parser)
    group = parser.add_argument_group(title="MoE layer perf")
    group.add_argument(
        "--moe-perf-warmup-iters",
        type=int,
        default=5,
        help="Warmup iterations for the standalone MoE layer perf frontend.",
    )
    group.add_argument(
        "--moe-perf-iters",
        type=int,
        default=20,
        help="Measured iterations for the standalone MoE layer perf frontend.",
    )
    group.add_argument(
        "--moe-perf-num-layers",
        type=int,
        default=None,
        help=(
            "Override the number of original MoE layers to instantiate. By default all "
            "MoE layers found in the GPT or Hybrid main decoder pattern are instantiated."
        ),
    )
    group.add_argument(
        "--moe-perf-seed",
        type=int,
        default=1234,
        help="Random seed for synthetic hidden states and backward gradients.",
    )
    group.add_argument(
        "--moe-perf-full-iter-cuda-graph",
        action="store_true",
        help="Capture and replay the whole standalone MoE forward+backward iteration.",
    )
    group.add_argument(
        "--moe-perf-cuda-graph-warmup-iters",
        type=int,
        default=3,
        help="Frontend-only warmup iterations before full-iteration CUDA graph capture.",
    )
    group.add_argument(
        "--moe-perf-paged-stash",
        action="store_true",
        help="Enable the existing MoE paged-stash manager around standalone MoE iterations.",
    )
    group.add_argument(
        "--moe-perf-expert-rank-capacity-factor",
        type=float,
        default=None,
        help=(
            "Override --moe-expert-rank-capacity-factor for the frontend. Setting this gives "
            "HybridEP a static per-rank token budget and avoids dynamic num_permuted_tokens sizing."
        ),
    )
    return parser


def _warn_ignored_args(args) -> None:
    ignored = []
    checks = {
        "use_megatron_fsdp": "FSDP wrappers are not constructed",
        "use_distributed_optimizer": "optimizer and distributed optimizer are not constructed",
        "overlap_grad_reduce": "gradient-reduce overlap is a full-training feature",
        "overlap_param_gather": "parameter-gather overlap is a full-training feature",
        "fine_grained_activation_offloading": "activation offload is not modeled in this frontend",
        "sequence_packing_scheduler": "THD/packed-sequence data path is not constructed",
        "use_varlen_dataset": "varlen dataset/THD data path is not constructed",
        "sft": "SFT/THD data path is not constructed",
        "cuda_graph_scope": "full-model CUDA graph capture is not constructed",
        "cuda_graph_modules": "MoE partial CUDA graph capture is not constructed",
        "use_cpu_initialization": (
            "GPU initialization is used to avoid CPU RNG mutation inside GPU RNG contexts"
        ),
    }
    for attr, reason in checks.items():
        value = getattr(args, attr, None)
        if value not in (None, False, [], "", "none"):
            ignored.append(f"--{attr.replace('_', '-')} ({reason})")

    if getattr(args, "recompute_granularity", None) not in (None, "none"):
        ignored.append("--recompute-granularity/--recompute-modules (recompute is disabled)")
    if getattr(args, "offload_modules", None):
        ignored.append("--offload-modules (activation offload is disabled)")
    if getattr(args, "qkv_format", None) == "thd":
        ignored.append("--qkv-format thd (attention/data path is skipped)")

    for item in ignored:
        warnings.warn(f"MoE perf frontend ignoring {item}", stacklevel=2)


def _disable_unsupported_full_model_features(args) -> None:
    """Disable features that need the full training stack around the MoE layer."""

    args.recompute_granularity = None
    args.recompute_modules = None
    args.fine_grained_activation_offloading = False
    args.offload_modules = None
    args.cuda_graph_scope = None
    args.cuda_graph_modules = []
    args.cuda_graph_impl = "full_iteration" if args.moe_perf_full_iter_cuda_graph else "none"
    if args.moe_perf_full_iter_cuda_graph:
        args.check_for_nan_in_loss_and_grad = False
    args.use_megatron_fsdp = False
    args.use_distributed_optimizer = False
    args.overlap_grad_reduce = False
    args.overlap_param_gather = False
    args.use_cpu_initialization = False


def _force_balanced_routing(args) -> None:
    args.moe_router_force_load_balancing = True


def _is_moe_layer_spec(spec: ModuleSpec) -> bool:
    if not isinstance(spec, ModuleSpec):
        return False
    submodules = getattr(spec, "submodules", None)
    mlp = getattr(submodules, "mlp", None)
    if isinstance(mlp, functools.partial):
        return mlp.func is MoELayer
    if isinstance(mlp, ModuleSpec):
        return mlp.module is MoELayer
    return mlp is MoELayer


def _resolve_hybrid_stack_spec(args, config):
    if args.spec is None:
        raise ValueError("Hybrid MoE perf requires --spec, matching the HybridModel pretrain path.")
    stack_spec = import_module(args.spec)
    if not isinstance(stack_spec, ModuleSpec) and callable(stack_spec):
        stack_spec = stack_spec(config)
    if not isinstance(stack_spec, ModuleSpec):
        raise TypeError(
            f"Hybrid --spec must resolve to ModuleSpec, got {type(stack_spec).__name__}"
        )
    return stack_spec


def _hybrid_moe_layer_specs(args, config) -> list[MoELayerSpec]:
    stack_spec = _resolve_hybrid_stack_spec(args, config)
    moe_spec = stack_spec.submodules.moe_layer
    if not isinstance(moe_spec, ModuleSpec):
        raise TypeError(f"Hybrid moe_layer spec must be ModuleSpec, got {type(moe_spec).__name__}")

    pattern = args.hybrid_layer_pattern or args.hybrid_override_pattern
    parsed = parse_hybrid_pattern(pattern)
    if parsed.mtp_pattern:
        warnings.warn(
            "MoE perf frontend only instantiates main-decoder MoE layers; MTP MoE layers "
            "are not included yet.",
            stacklevel=2,
        )
    if parsed.main_pattern is None:
        raise ValueError("Hybrid MoE perf could not determine the main hybrid layer pattern.")

    layer_specs = []
    layer_number = 0
    for symbol in parsed.main_pattern:
        if symbol == Symbols.PIPE:
            continue
        layer_number += 1
        if symbol == Symbols.MOE:
            layer_specs.append(MoELayerSpec(layer_number, moe_spec, "hybrid"))
    return layer_specs


def _gpt_moe_layer_specs(args, config) -> list[MoELayerSpec]:
    if args.spec is not None:
        spec = import_module(args.spec)
        if callable(spec) and not isinstance(spec, ModuleSpec):
            spec = spec(config)
        if isinstance(spec, ModuleSpec) and _is_moe_layer_spec(spec):
            return [
                MoELayerSpec(layer_number, spec, "gpt")
                for layer_number in range(1, config.num_layers + 1)
            ]
        raise ValueError(
            "GPT --spec did not resolve to a single MoE TransformerLayer spec. "
            "Use the standard GPT MoE spec path or provide a MoE layer ModuleSpec."
        )

    if config.experimental_attention_variant is not None:
        specs = get_transformer_block_with_experimental_attention_variant_spec(config).layer_specs
    else:
        use_te = args.transformer_impl == "transformer_engine"
        specs = get_gpt_decoder_layer_specs(
            config,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
        )
    if hasattr(specs, "layer_specs"):
        specs = specs.layer_specs
    return [
        MoELayerSpec(layer_number, spec, "gpt")
        for layer_number, spec in enumerate(specs, start=1)
        if _is_moe_layer_spec(spec)
    ]


def _select_moe_layer_specs(args, config) -> list[MoELayerSpec]:
    is_hybrid = bool(
        getattr(args, "is_hybrid_model", False)
        or getattr(args, "hybrid_layer_pattern", None)
        or getattr(args, "hybrid_override_pattern", None)
    )
    layer_specs = (
        _hybrid_moe_layer_specs(args, config) if is_hybrid else _gpt_moe_layer_specs(args, config)
    )
    if not layer_specs:
        raise ValueError("No MoE layers were found in the resolved full-model configuration.")

    if args.moe_perf_num_layers is not None:
        if args.moe_perf_num_layers <= 0:
            raise ValueError("--moe-perf-num-layers must be positive.")
        if args.moe_perf_num_layers > len(layer_specs):
            raise ValueError(
                f"--moe-perf-num-layers={args.moe_perf_num_layers} exceeds the "
                f"{len(layer_specs)} MoE layers found in the full model."
            )
        layer_specs = layer_specs[: args.moe_perf_num_layers]
    return layer_specs


def _build_moe_layers(layer_specs: Iterable[MoELayerSpec], config) -> torch.nn.ModuleList:
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    layers = []
    for local_index, layer_spec in enumerate(layer_specs):
        layer = build_module(
            layer_spec.spec,
            config=config,
            layer_number=layer_spec.layer_number,
            pg_collection=pg_collection,
            add_layer_offset=False,
            name=f"moe_perf.layers.{local_index}",
        )
        if not isinstance(layer, TransformerLayer) or not getattr(layer, "is_moe_layer", False):
            raise TypeError(
                f"Resolved layer {layer_spec.layer_number} did not instantiate a TransformerLayer "
                "with a MoELayer MLP."
            )
        layers.append(layer)
    return torch.nn.ModuleList(layers).cuda().train()


def _ensure_main_grad_buffers(module: torch.nn.Module) -> None:
    """Allocate minimal main_grad buffers normally provided by DDP/FSDP."""

    for param in module.parameters():
        if not param.requires_grad:
            continue
        if not hasattr(param, "main_grad"):
            param.main_grad = torch.zeros_like(param.data)
        if not hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = False


def _zero_layer_grads(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.grad = None
        if hasattr(param, "main_grad"):
            param.main_grad.zero_()
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = False


def _router_bias_modules(layers: torch.nn.ModuleList) -> list[torch.nn.Module]:
    return [
        module
        for module in layers.modules()
        if getattr(module, "expert_bias", None) is not None and module.training
    ]


def _verify_router_bias_metadata(config, layers: torch.nn.ModuleList) -> dict[int, tuple[int, ...]]:
    routers = _router_bias_modules(layers)
    if not config.moe_router_enable_expert_bias:
        if routers:
            raise RuntimeError("Expert-bias router buffers exist while the feature is disabled.")
        return {}
    if not routers:
        raise RuntimeError("No expert-bias router buffers were found in the MoE perf layers.")

    pointers = {}
    for router in routers:
        pointers[id(router)] = (
            router.expert_bias.data_ptr(),
            *(
                (router.qb_histogram.data_ptr(), router.qb_bin_bounds.data_ptr())
                if config.moe_router_bias_update_method == "quantile"
                else ()
            ),
        )
        if config.moe_router_bias_update_method == "quantile":
            if not fused_topk_with_score_function_supports_qb:
                raise RuntimeError(
                    "Quantile Balancing requested, but the installed Transformer Engine "
                    "router does not expose the QB API."
                )
            if router.qb_histogram_mode != "fused_atomic":
                raise RuntimeError(
                    f"Expected fused_atomic QB histogram mode, got {router.qb_histogram_mode!r}."
                )
            if "qb_bin_bounds" not in router.state_dict():
                raise RuntimeError("qb_bin_bounds must be persistent router state.")
            if "qb_histogram" in router.state_dict():
                raise RuntimeError("qb_histogram must remain nonpersistent step-local state.")

    print_rank_0(
        "[moe_perf] router_bias_metadata "
        f"method={config.moe_router_bias_update_method} "
        f"router_count={len(routers)} "
        f"te_qb_api={fused_topk_with_score_function_supports_qb} "
        f"qb_histogram_mode={'fused_atomic' if config.moe_router_bias_update_method == 'quantile' else 'none'} "
        f"qb_num_bins={config.moe_router_qb_num_bins}"
    )
    return pointers


def _verify_router_bias_pointers(
    layers: torch.nn.ModuleList, expected_pointers: dict[int, tuple[int, ...]]
) -> None:
    for router in _router_bias_modules(layers):
        actual = (
            router.expert_bias.data_ptr(),
            *(
                (router.qb_histogram.data_ptr(), router.qb_bin_bounds.data_ptr())
                if router.qb_histogram is not None
                else ()
            ),
        )
        if actual != expected_pointers[id(router)]:
            raise RuntimeError(
                f"Router bias/QB CUDA storage changed: expected={expected_pointers[id(router)]}, "
                f"actual={actual}."
            )


def _verify_qb_histograms(
    config, layers: torch.nn.ModuleList, expected_tokens_per_router: int
) -> None:
    if config.moe_router_bias_update_method != "quantile":
        return
    expected_count = expected_tokens_per_router * config.num_moe_experts
    for router in _router_bias_modules(layers):
        actual_count = int(router.qb_histogram.sum().item())
        if actual_count != expected_count:
            raise RuntimeError(
                f"QB histogram count mismatch: expected={expected_count}, actual={actual_count}."
            )


def _time_router_bias_finalization(config, layers: torch.nn.ModuleList) -> float:
    if not config.moe_router_enable_expert_bias:
        return 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _update_router_expert_bias(
        [layers],
        config,
        tp_dp_cp_group=parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        ),
    )
    reset_model_temporary_tensors(config, [layers])
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _verify_situ_glu_fused_backend(layers: torch.nn.ModuleList) -> None:
    """Verify TE selected its joint CuTe grouped-MLP op rather than basic-op fallback."""
    selected = []
    for layer in layers:
        experts = layer.mlp.experts
        fused_sequential = experts._fused_ops[0]
        for module_group in fused_sequential._module_groups:
            forward_ops = getattr(module_group, "_forward_ops", ())
            selected.extend(type(op).__name__ for op, _ in forward_ops)
    expected = "GroupedMLP_CuTeGEMMGLU"
    if expected not in selected:
        raise RuntimeError(
            "SiTU-GLU requested, but TE did not select the fused MXFP8 "
            f"FC1 -> SiTU-GLU -> FC2 CuTe DSL op; selected={selected}."
        )
    print_rank_0(f"[moe_perf] verified_situ_glu_backend={expected}")


def _zero_static_input_grad(hidden_states: torch.Tensor) -> None:
    if hidden_states.grad is not None:
        hidden_states.grad.zero_()


def _reset_paged_stash_for_iteration(config, enabled: bool) -> None:
    if not enabled:
        return
    paged_stash_reset(enabled=True, config=config)
    paged_stash_init_chunk_handler(vp_size=None, vp_stage=None)


def _check_paged_stash_status(enabled: bool) -> tuple[bool, bool]:
    if not enabled:
        return False, False
    overflow = bool(check_paged_stash_overflow().view(-1)[0].item())
    host_spill = bool(check_paged_stash_host_spill().view(-1)[0].item())
    return overflow, host_spill


def _quantization_context(config, layer_number: int):
    if config.fp8:
        if config.fp8_recipe == Fp8Recipe.delayed:
            return nullcontext()
        return get_fp8_context(config, layer_number - 1)
    if config.fp4:
        return get_fp4_context(config, layer_number - 1)
    return nullcontext()


def _moe_only_flops_per_iteration(args, layer_specs: list[MoELayerSpec]) -> float:
    """Compute MoE-only GEMM FLOPs for one frontend iteration."""

    # The frontend runs exactly one synthetic microbatch per data-parallel rank
    # per measured iteration. Router, token permutation, aux loss, elementwise
    # activations, normalization, and communication are intentionally excluded.
    total_tokens = args.micro_batch_size * args.data_parallel_size * args.seq_length
    moe_ffn_hidden_size = (
        args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    )
    shared_expert_size = args.moe_shared_expert_intermediate_size or 0
    forward_backward_expansion_factor = 3
    return (
        forward_backward_expansion_factor
        * len(layer_specs)
        * _moe_layer_flops(
            total_tokens,
            args.hidden_size,
            moe_ffn_hidden_size,
            shared_expert_size,
            args.moe_router_topk,
            args.moe_latent_size,
            gated_linear_unit=args.swiglu or args.use_situ_glu,
        )
    )


def _forward_moe_layers(hidden_states, config, layers, layer_specs):
    for layer, layer_spec in zip(layers, layer_specs):
        with _quantization_context(config, layer_spec.layer_number):
            hidden_states = layer._forward_mlp(hidden_states)
    return hidden_states


def _run_eager_iteration(
    *, args, config, layers, layer_specs, hidden_template, backward_grad, paged_stash_enabled
):
    _reset_paged_stash_for_iteration(config, paged_stash_enabled)
    _zero_layer_grads(layers)
    hidden_states = hidden_template.detach().clone().requires_grad_(True)

    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    fwd_start.record()
    hidden_states = _forward_moe_layers(hidden_states, config, layers, layer_specs)
    fwd_end.record()

    bwd_start.record()
    hidden_states.backward(backward_grad)
    bwd_end.record()
    torch.cuda.synchronize()
    return fwd_start.elapsed_time(fwd_end), bwd_start.elapsed_time(bwd_end)


def _build_full_iteration_wrapper(
    *, args, config, layers, layer_specs, hidden_template, static_backward_grad, paged_stash_enabled
):
    static_hidden = hidden_template.detach().clone().requires_grad_(True)
    static_hidden.grad = torch.zeros_like(static_hidden)

    def forward_backward_func(**kwargs):
        del kwargs
        reset_model_temporary_tensors(config, [layers])
        _zero_layer_grads(layers)
        _zero_static_input_grad(static_hidden)
        hidden_states = _forward_moe_layers(static_hidden, config, layers, layer_specs)
        hidden_states.backward(static_backward_grad)

    wrapper = FullCudaGraphWrapper(
        forward_backward_func,
        # Run wrapper warmups before capture, but _call_full_iteration_wrapper()
        # executes both warmup and capture on FullCudaGraphWrapper's side stream.
        # This mirrors full training: lazy compiler/autograd setup finishes
        # before capture, while AccumulateGrad nodes are created on the same
        # stream later used for capture.
        cuda_graph_warmup_steps=max(0, args.moe_perf_cuda_graph_warmup_iters),
        use_single_mempool=True,
    )
    wrapper.reset_cuda_graph(stage='training')
    return wrapper, static_hidden


def _call_full_iteration_wrapper(*, args, config, wrapper, layers, paged_stash_enabled):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    capture_stream = get_shared_capture_stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        _reset_paged_stash_for_iteration(config, paged_stash_enabled)
        start.record()
        wrapper(
            model=[layers],
            data_iterator=None,
            num_microbatches=1,
            seq_length=args.seq_length,
            forward_only=False,
        )
        end.record()
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    return start.elapsed_time(end), 0.0


def _uses_nsys_profile(args) -> bool:
    """Return whether standard Megatron arguments request an NSys capture."""
    return bool(args.profile and not args.use_pytorch_profiler)


def _moe_perf_iteration_plan(args) -> tuple[int, int]:
    """Return total iterations and the first iteration included in frontend metrics."""
    if not _uses_nsys_profile(args):
        return args.moe_perf_warmup_iters + args.moe_perf_iters, args.moe_perf_warmup_iters

    profile_start = args.profile_step_start
    profile_end = args.profile_step_end
    if profile_start < 0 or profile_end <= profile_start:
        raise ValueError(
            "NSys profiling requires 0 <= --profile-step-start < --profile-step-end, "
            f"got {profile_start} and {profile_end}."
        )

    # Match the normal training loop: start before iteration profile_start, stop
    # after completed iteration profile_end, and schedule no later distributed work.
    return profile_end, profile_start


class _NSysProfiler:
    """Mirror Megatron's CUDA-profiler API window for the standalone frontend."""

    def __init__(self, args):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else args.rank
        self.enabled = _uses_nsys_profile(args) and (
            len(args.profile_ranks) == 0 or rank in args.profile_ranks
        )
        self.profile_start = args.profile_step_start
        self.profile_end = args.profile_step_end
        self.nvtx_ranges = args.nvtx_ranges
        self.record_shapes = args.record_shapes
        self._active = False
        self._nvtx_context = None

    def start_if_needed(self, iteration: int) -> None:
        """Start NSys immediately before the configured frontend iteration."""
        if not self.enabled or self._active or iteration != self.profile_start:
            return
        if self.nvtx_ranges:
            configure_nvtx_profiling(True)
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        self._active = True
        try:
            self._nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=self.record_shapes)
            self._nvtx_context.__enter__()
        except Exception:
            self.close()
            raise

    def stop_if_needed(self, completed_iteration: int) -> None:
        """Stop NSys after the configured number of iterations has completed."""
        if self._active and completed_iteration == self.profile_end:
            self.close()

    def close(self) -> None:
        """Close an active profile window, including on an exceptional exit."""
        if not self._active:
            return
        if self.nvtx_ranges:
            configure_nvtx_profiling(False)
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
        if self._nvtx_context is not None:
            self._nvtx_context.__exit__(None, None, None)
        self._nvtx_context = None
        self._active = False


def _run_moe_perf(
    args, config, layers: torch.nn.ModuleList, layer_specs: list[MoELayerSpec]
) -> None:
    if args.moe_perf_warmup_iters < 0 or args.moe_perf_iters <= 0:
        raise ValueError("--moe-perf-warmup-iters must be >= 0 and --moe-perf-iters must be > 0.")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dtype = config.params_dtype or torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(args.moe_perf_seed)
    hidden_template = torch.randn(
        args.seq_length,
        args.micro_batch_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    backward_grad_template = torch.randn(
        args.seq_length,
        args.micro_batch_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )

    forward_timings = []
    backward_timings = []
    router_bias_finalize_timings = []
    max_allocated = []
    flops_per_iteration = _moe_only_flops_per_iteration(args, layer_specs)
    total_iters, measurement_start = _moe_perf_iteration_plan(args)
    paged_stash_enabled = args.moe_perf_paged_stash
    router_bias_pointers = _verify_router_bias_metadata(config, layers)
    nsys_profiler = _NSysProfiler(args)
    graph_wrapper = None
    if args.moe_perf_full_iter_cuda_graph:
        graph_wrapper, _ = _build_full_iteration_wrapper(
            args=args,
            config=config,
            layers=layers,
            layer_specs=layer_specs,
            hidden_template=hidden_template,
            static_backward_grad=backward_grad_template,
            paged_stash_enabled=paged_stash_enabled,
        )

    try:
        for iteration in range(total_iters):
            nsys_profiler.start_if_needed(iteration)
            random_ste_generator = getattr(RandomSTE, "generator", None)
            if random_ste_generator is not None:
                random_ste_generator.manual_seed(random_ste_generator.initial_seed())
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            _zero_layer_grads(layers)
            reset_model_temporary_tensors(config, [layers])
            torch.cuda.reset_peak_memory_stats()

            if graph_wrapper is not None:
                forward_ms, backward_ms = _call_full_iteration_wrapper(
                    args=args,
                    config=config,
                    wrapper=graph_wrapper,
                    layers=layers,
                    paged_stash_enabled=paged_stash_enabled,
                )
            else:
                forward_ms, backward_ms = _run_eager_iteration(
                    args=args,
                    config=config,
                    layers=layers,
                    layer_specs=layer_specs,
                    hidden_template=hidden_template,
                    backward_grad=backward_grad_template,
                    paged_stash_enabled=paged_stash_enabled,
                )
            if iteration == 0:
                _verify_qb_histograms(
                    config,
                    layers,
                    expected_tokens_per_router=args.seq_length * args.micro_batch_size,
                )
            router_bias_finalize_ms = _time_router_bias_finalization(config, layers)
            _verify_router_bias_pointers(layers, router_bias_pointers)
            if iteration == 0 and config.use_situ_glu:
                _verify_situ_glu_fused_backend(layers)
            paged_stash_overflow, paged_stash_host_spill = _check_paged_stash_status(
                paged_stash_enabled
            )
            iteration_ms = forward_ms + backward_ms
            throughput = flops_per_iteration / ((iteration_ms / 1000.0) * 10**12 * args.world_size)
            nsys_profiler.stop_if_needed(iteration + 1)

            if iteration >= measurement_start:
                forward_timings.append(forward_ms)
                backward_timings.append(backward_ms)
                router_bias_finalize_timings.append(router_bias_finalize_ms)
                max_allocated.append(torch.cuda.max_memory_allocated())

            if args.rank == 0:
                current_iter = iteration + 1
                consumed_samples = current_iter * args.micro_batch_size * args.data_parallel_size
                print(
                    f" iteration {current_iter:8d}/{total_iters:8d} | "
                    f"consumed samples: {consumed_samples:12d} | "
                    f"elapsed time per iteration (ms): {iteration_ms:.1f} | "
                    f"throughput per GPU (TFLOP/s/GPU): {throughput:.1f} | "
                    f"global batch size: {args.micro_batch_size * args.data_parallel_size:5d} |",
                    flush=True,
                )
                print(
                    f"[moe_perf] iter={iteration} measured={iteration >= measurement_start} "
                    f"forward_ms={forward_ms:.3f} backward_ms={backward_ms:.3f} "
                    f"router_bias_finalize_ms={router_bias_finalize_ms:.3f} "
                    f"router_bias_update_method={config.moe_router_bias_update_method} "
                    f"throughput_tflops_per_gpu={throughput:.1f} "
                    f"full_iter_cuda_graph={graph_wrapper is not None} "
                    f"paged_stash={paged_stash_enabled} "
                    f"paged_stash_overflow={paged_stash_overflow} "
                    f"paged_stash_host_spill={paged_stash_host_spill} "
                    f"max_allocated={torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GiB",
                    flush=True,
                )
    finally:
        nsys_profiler.close()

    if args.rank == 0:
        forward_ms = statistics.mean(forward_timings)
        backward_ms = statistics.mean(backward_timings)
        router_bias_finalize_ms = statistics.mean(router_bias_finalize_timings)
        iteration_ms = forward_ms + backward_ms
        end_to_end_ms = iteration_ms + router_bias_finalize_ms
        throughput = flops_per_iteration / ((iteration_ms / 1000.0) * 10**12 * args.world_size)
        print(
            "[moe_perf] summary "
            f"layers={len(layers)} "
            f"layer_numbers={[spec.layer_number for spec in layer_specs]} "
            f"forward_ms_mean={forward_ms:.3f} "
            f"forward_ms_stdev={statistics.pstdev(forward_timings) if len(forward_timings) > 1 else 0.0:.3f} "
            f"backward_ms_mean={backward_ms:.3f} "
            f"backward_ms_stdev={statistics.pstdev(backward_timings) if len(backward_timings) > 1 else 0.0:.3f} "
            f"router_bias_finalize_ms_mean={router_bias_finalize_ms:.3f} "
            f"router_bias_finalize_ms_stdev={statistics.pstdev(router_bias_finalize_timings) if len(router_bias_finalize_timings) > 1 else 0.0:.3f} "
            f"router_bias_update_method={config.moe_router_bias_update_method} "
            f"end_to_end_ms_mean={end_to_end_ms:.3f} "
            f"throughput_tflops_per_gpu={throughput:.1f} "
            f"flops_per_iteration={flops_per_iteration:.6e} "
            f"max_allocated_gib_mean={statistics.mean(max_allocated) / (1024 ** 3):.3f}",
            flush=True,
        )


def main() -> None:
    program_start = time.time()
    args = parse_args(extra_args_provider=_add_moe_perf_args)
    validate_args(args, defaults={"tokenizer_type": "NullTokenizer"})
    _warn_ignored_args(args)
    _disable_unsupported_full_model_features(args)
    _force_balanced_routing(args)
    if args.moe_perf_expert_rank_capacity_factor is not None:
        # Keep the frontend-only static HybridEP buffer budget out of ordinary
        # TransformerConfig validation. The perf harness applies it to the completed
        # config below and reports any actual overflow; this allows QB to test a
        # graph-safe no-overflow path without enabling production token dropping.
        args.use_transformer_engine_op_fuser = True
    set_global_variables(args, build_tokenizer=False)
    initialize_megatron()

    args = get_args()
    config = core_transformer_config_from_args(args)
    config.moe_router_force_load_balancing = True
    config.recompute_granularity = None
    config.recompute_modules = None
    config.fine_grained_activation_offloading = False
    config.offload_modules = None
    config.cuda_graph_scope = None
    config.cuda_graph_modules = []
    config.cuda_graph_impl = "none"
    config.moe_paged_stash = args.moe_perf_paged_stash
    if args.moe_perf_expert_rank_capacity_factor is not None:
        config.moe_expert_rank_capacity_factor = args.moe_perf_expert_rank_capacity_factor
        config.use_transformer_engine_op_fuser = True
    layer_specs = _select_moe_layer_specs(args, config)
    if args.rank == 0:
        print_rank_0(
            "[moe_perf] constructing MoE layers "
            f"source={layer_specs[0].source} "
            f"count={len(layer_specs)} "
            f"layer_numbers={[spec.layer_number for spec in layer_specs]} "
            f"force_load_balancing={config.moe_router_force_load_balancing} "
            f"router_bias_update_method={config.moe_router_bias_update_method} "
            f"qb_num_bins={config.moe_router_qb_num_bins} "
            f"latent_up_rmsnorm={config.moe_latent_up_projection_rmsnorm} "
            f"situ_glu={config.use_situ_glu} "
            f"situ_glu_impl={config.situ_glu_impl}"
        )
    layers = _build_moe_layers(layer_specs, config)
    _ensure_main_grad_buffers(layers)
    _run_moe_perf(args, config, layers, layer_specs)

    if args.moe_perf_full_iter_cuda_graph:
        # Full-iteration CUDA graphs may capture collectives. Normal process
        # group teardown can then hang in torchrun/NCCL atexit cleanup, the same
        # issue handled by gpt_static_inference.py. This frontend has already
        # printed the summary by this point, so exit directly for graph runs.
        if args.rank == 0:
            print(f"[moe_perf] done elapsed_sec={time.time() - program_start:.3f}", flush=True)
        os._exit(0)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if args.rank == 0:
        print(f"[moe_perf] done elapsed_sec={time.time() - program_start:.3f}", flush=True)


if __name__ == "__main__":
    main()
