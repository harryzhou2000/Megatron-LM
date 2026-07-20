# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental Qwen3.5-VL MIMO entry point.

This entry point mirrors ``pretrain_multimodal.py`` but builds a colocated
``MimoModel`` instead of the standalone ``Qwen35VLModel``.  The first target is
baseline-equivalent colocated training with the normal Megatron wrapper handling
DDP/FSDP for the whole returned model chunk.

Current scope:
- Qwen3.5-VL only.
- Colocated modules only.
- PP=1 and CP=1.
- Whole-model DDP/FSDP wrapping through the standard Megatron training path.

Independent per-module FSDP and bridged/non-colocated MIMO are intentionally not
enabled here yet.
"""

import importlib
import os
import sys
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from examples.mimo.training.topology import ModuleGridSpec, create_topology
from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.forward_step import get_batch, loss_func
from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VISION_KWARGS,
    get_qwen35_vl_vision_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    post_language_config,
    set_vision_flops_metadata,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from examples.multimodal_dev.models.qwen35_vl.specs import (
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_spec,
)
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import unwrap_model
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args


@dataclass(frozen=True)
class _VisualDescriptor:
    desc_id: int
    src_rank: int
    sample_idx: int
    row_idx: int
    patch_start: int
    patch_end: int
    token_count: int
    cost: int


class _VariableAllGather(torch.autograd.Function):
    """Autograd-preserving all-gather for tensors with variable dim 0."""

    @staticmethod
    def forward(ctx, tensor, group, sizes):
        local_size = tensor.shape[0]
        max_size = max(max(sizes), 1)
        padded = torch.zeros((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        if local_size > 0:
            padded[:local_size] = tensor.contiguous()
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=group)
        ctx.group = group
        ctx.local_size = local_size
        ctx.local_rank = dist.get_rank(group)
        return torch.stack(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Each rank's input appears at output[rank] on every peer. Sum the
        # complete output-gradient stack across peers, then select this rank's
        # slice as the gradient for the local input.
        grad_stack = grad_output.contiguous()
        dist.all_reduce(grad_stack, group=ctx.group)
        return grad_stack[ctx.local_rank, : ctx.local_size].contiguous(), None, None


def add_multimodal_mimo_args(parser):
    """Add Qwen3.5-VL MIMO-specific arguments."""
    parser = add_multimodal_args(parser)
    group = parser.add_argument_group("Qwen3.5-VL MIMO", "MIMO colocated parallelism arguments")
    group.add_argument(
        "--mimo-vision-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Vision encoder TP size for colocated MIMO. Defaults to global TP size.",
    )
    group.add_argument(
        "--mimo-disable-module-grid-map",
        action="store_true",
        default=False,
        help="Disable module_to_grid_map and use legacy all-modules-colocated MIMO behavior.",
    )
    group.add_argument(
        "--mimo-vision-mdp",
        action="store_true",
        default=False,
        help=(
            "Enable experimental MDP-style dynamic image scheduling for Qwen MIMO. "
            "Currently supports colocated vision_tp == language_tp, PP=CP=1, padded batches."
        ),
    )
    group.add_argument(
        "--mimo-vision-mdp-scheduler",
        type=str,
        default="lpt",
        choices=["static", "lpt"],
        help="Image scheduling policy for --mimo-vision-mdp.",
    )
    group.add_argument(
        "--mimo-vision-mdp-cost-proxy",
        type=str,
        default="p2",
        choices=["p2", "raw_patch"],
        help="Image cost proxy for LPT scheduling.",
    )
    return parser


def _build_module_parallel_context(args):
    """Build colocated MIMO grids and process-group collections."""
    if getattr(args, "mimo_disable_module_grid_map", False):
        return None, None, None

    world_size = dist.get_world_size()
    vision_tp = getattr(args, "mimo_vision_tensor_model_parallel_size", None)
    if vision_tp is None:
        vision_tp = args.tensor_model_parallel_size
    if world_size % vision_tp != 0:
        raise ValueError(f"world_size={world_size} must be divisible by vision TP={vision_tp}")
    vision_dp = world_size // vision_tp

    language_tp = args.tensor_model_parallel_size
    # The MIMO grid describes activation/batch layout for the colocated bridge,
    # not MoE expert ownership. Keep EP out of this grid so the language DP
    # dimension matches the dense/activation batch replicas.
    language_ep = 1
    denom = language_tp
    if world_size % denom != 0:
        raise ValueError(f"world_size={world_size} must be divisible by language TP={language_tp}")
    language_dp = world_size // denom
    if vision_tp > language_tp and language_tp != 1:
        raise ValueError(
            "pretrain_multimodal_mimo supports vision TP > language TP only when "
            f"language TP is 1; got vision TP={vision_tp}, language TP={language_tp}"
        )
    if language_tp > vision_tp:
        if language_tp % vision_tp != 0:
            raise ValueError(
                "language TP must be divisible by vision TP for MIMO fan-in; "
                f"got language TP={language_tp}, vision TP={vision_tp}"
            )
        fan_in_scale = language_tp // vision_tp
        if args.micro_batch_size % fan_in_scale != 0:
            raise ValueError(
                "micro_batch_size must be divisible by MIMO fan-in scale when "
                "language TP > vision TP; "
                f"micro_batch_size={args.micro_batch_size}, fan_in_scale={fan_in_scale}"
            )

    vision_size = vision_tp * vision_dp
    language_size = language_tp * language_dp * language_ep
    if vision_size != world_size or language_size != world_size:
        raise ValueError(
            "Colocated MIMO grids must both span the full world. "
            f"Got vision TP*DP={vision_tp}*{vision_dp}={vision_size}, "
            f"language TP*DP*EP={language_tp}*{language_dp}*{language_ep}={language_size}, "
            f"world_size={world_size}."
        )

    topology = create_topology(
        [
            ModuleGridSpec(name="images", num_ranks=world_size, tp=vision_tp),
            ModuleGridSpec(
                name=MIMO_LANGUAGE_MODULE_KEY,
                num_ranks=world_size,
                tp=language_tp,
                ep=language_ep,
            ),
        ]
    )
    module_to_grid_map = {
        "images": topology.grids["images"],
        MIMO_LANGUAGE_MODULE_KEY: topology.grids[MIMO_LANGUAGE_MODULE_KEY],
    }
    return (
        module_to_grid_map,
        topology.module_pgs["images"],
        topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
    )


class Qwen35VLMimoModel(MimoModel):
    """Colocated MIMO wrapper for Qwen3.5-VL.

    ``MimoModel`` already owns the language/module plumbing, but Qwen3.5-VL
    needs an adapter for the existing multimodal_dev batch schema and MRoPE.
    """

    def __init__(
        self,
        mimo_config: MimoModelConfig,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        spatial_merge_size: int = 2,
        vision_tp_group=None,
        cp_group=None,
        tp_group=None,
        vision_dp_group=None,
        language_dp_group=None,
        vision_mdp_enabled: bool = False,
        vision_mdp_scheduler: str = "lpt",
        vision_mdp_cost_proxy: str = "p2",
        vision_grid=None,
        language_grid=None,
    ) -> None:
        super().__init__(mimo_config, cp_group=cp_group, tp_group=tp_group)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size
        self.vision_tp_group = vision_tp_group
        self.language_tp_group = tp_group
        self.vision_dp_group = vision_dp_group
        self.language_dp_group = language_dp_group
        self.vision_mdp_enabled = vision_mdp_enabled
        self.vision_mdp_scheduler = vision_mdp_scheduler
        self.vision_mdp_cost_proxy = vision_mdp_cost_proxy
        self.vision_grid = vision_grid
        self.language_grid = language_grid

    def compute_position_ids(self, input_ids, image_grid_thw=None, packed_seq_params=None):
        """Compute Qwen3.5-VL 3D MRoPE position IDs."""
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            packed_seq_params=packed_seq_params,
        )
        return position_ids

    def get_text_embeddings(self, input_ids, position_ids, special_token_ids):
        """Get full text embeddings before MIMO performs multimodal alignment.

        GPT embeddings should normally scatter under SP so MTP receives SP-local
        next-token embeddings. MIMO alignment needs the opposite: a full flat
        text-token list. Temporarily disable only this lookup's scatter and
        restore the embedding module state before the language model/MTP forward.
        """
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for special_token_id in special_token_ids.values():
            text_mask &= input_ids != special_token_id

        batch_idx, seq_idx = text_mask.nonzero(as_tuple=True)
        input_ids_text = input_ids[batch_idx, seq_idx].unsqueeze(0)

        if position_ids is None:
            position_ids_text = None
        elif position_ids.dim() == 3:
            position_ids_text = position_ids[0, batch_idx, seq_idx].unsqueeze(0)
        else:
            position_ids_text = position_ids[batch_idx, seq_idx].unsqueeze(0)

        embedding_layer = unwrap_model(self.language_model).embedding
        scatter = getattr(embedding_layer, "scatter_to_sequence_parallel", None)
        reduce_scatter = getattr(embedding_layer, "reduce_scatter_embeddings", None)
        word_reduce_scatter = getattr(
            embedding_layer.word_embeddings, "reduce_scatter_embeddings", None
        )
        # MIMO aligns text and vision using the full local token list. Under SP,
        # RoPE/no-position embeddings may reduce-scatter inside VocabParallelEmbedding
        # before LanguageModelEmbedding sees the tensor, so disable both wrapper
        # and child flags only for this lookup. Restore them before GPT/MTP runs.
        if scatter is not None:
            embedding_layer.scatter_to_sequence_parallel = False
        if reduce_scatter is not None:
            embedding_layer.reduce_scatter_embeddings = False
        if word_reduce_scatter is not None:
            embedding_layer.word_embeddings.reduce_scatter_embeddings = False
        try:
            return embedding_layer(input_ids=input_ids_text, position_ids=position_ids_text).squeeze(1)
        finally:
            if scatter is not None:
                embedding_layer.scatter_to_sequence_parallel = scatter
            if reduce_scatter is not None:
                embedding_layer.reduce_scatter_embeddings = reduce_scatter
            if word_reduce_scatter is not None:
                embedding_layer.word_embeddings.reduce_scatter_embeddings = word_reduce_scatter

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_mask=None,
        padding_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        decoder_input=None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Adapt multimodal_dev batches to MIMO's ``modality_inputs`` format."""
        self._debug_mdp_stage(
            "forward_start",
            pixel_is_none=pixel_values is None,
            pixel_shape=None if pixel_values is None else tuple(pixel_values.shape),
            grid_is_none=image_grid_thw is None,
            grid_shape=None if image_grid_thw is None else tuple(image_grid_thw.shape),
            mdp=self.vision_mdp_enabled,
        )
        if decoder_input is not None:
            raise NotImplementedError(
                "decoder_input is not supported by pretrain_multimodal_mimo yet"
            )
        if (input_ids == self.video_token_id).any().item():
            raise ValueError("pretrain_multimodal_mimo currently supports image tokens only")
        if self.vision_mdp_enabled:
            self._debug_mdp_stage("before_normalize_visual_inputs")
            pixel_values, image_grid_thw = self._normalize_mdp_visual_inputs(
                pixel_values, image_grid_thw, device=input_ids.device
            )
            self._debug_mdp_stage(
                "after_normalize_visual_inputs",
                pixel_shape=tuple(pixel_values.shape),
                grid_shape=tuple(image_grid_thw.shape),
            )
        vision_stats = None
        if pixel_values is not None:
            self._debug_mdp_stage("before_validate_vision_inputs")
            vision_stats = self._validate_vision_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )
            self._debug_mdp_stage(
                "after_validate_vision_inputs",
                stats={k: v for k, v in vision_stats.items() if not torch.is_tensor(v)},
            )
        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )

        modality_inputs = None
        image_bridge_split_sizes = None
        if pixel_values is not None:
            if (
                vision_stats is not None
                and vision_stats["num_image_tokens"] == 0
                and not self._uses_colocated_bridge_for_visuals()
                and not (self.vision_mdp_enabled and self._can_use_mdp_scheduler())
            ):
                image_embeddings = torch.empty(
                    (0, self.config.hidden_size), dtype=pixel_values.dtype, device=pixel_values.device
                )
                modality_inputs = {"images": {"precomputed_embeddings": image_embeddings}}
            elif self.vision_mdp_enabled and self._can_use_mdp_scheduler():
                self._debug_mdp_stage(
                    "before_encode_images_with_mdp",
                    schedule_group_size=dist.get_world_size(self._mdp_schedule_group()),
                    schedule_group_rank=dist.get_rank(self._mdp_schedule_group()),
                    num_workers=self._mdp_num_workers(),
                    worker_rank=self._mdp_worker_rank(),
                )
                image_embeddings = self._encode_images_with_mdp(
                    pixel_values,
                    image_grid_thw,
                    vision_stats,
                    packed=packed_seq_params is not None,
                )
                modality_inputs = {"images": {"precomputed_embeddings": image_embeddings}}
            else:
                pixel_values, image_grid_thw, image_bridge_split_sizes = (
                    self._prepare_vision_inputs_for_bridge(
                        pixel_values, image_grid_thw, vision_stats
                    )
                )
                if image_bridge_split_sizes is not None and sum(image_bridge_split_sizes) == 0:
                    image_embeddings = torch.empty(
                        (0, self.config.hidden_size),
                        dtype=pixel_values.dtype,
                        device=pixel_values.device,
                    )
                    modality_inputs = {"images": {"precomputed_embeddings": image_embeddings}}
                else:
                    modality_inputs = {
                        "images": {
                            "qwen35_vision_encoder": {
                                "pixel_values": pixel_values,
                                "grid_thw": image_grid_thw,
                            }
                        }
                    }

        output, _ = self._forward_all_modules(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            labels=labels,
            modality_inputs=modality_inputs,
            image_bridge_split_sizes=image_bridge_split_sizes,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        return output

    def _debug_mdp_stage(self, stage, **kwargs):
        if os.getenv("MIMO_MDP_DEBUG", "0") != "1":
            return
        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[MIMO MDP][rank {rank}] {stage} {kwargs}", flush=True)

    def _validate_vision_inputs(
        self, input_ids, pixel_values, image_grid_thw, packed_seq_params=None
    ):
        """Validate Qwen vision tensors before MIMO bridge fan-out."""
        if image_grid_thw is None:
            raise ValueError("image_grid_thw must be provided when pixel_values is provided")
        if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError(
                "image_grid_thw must have shape [num_images_or_videos, 3], "
                f"got {tuple(image_grid_thw.shape)}"
            )
        if pixel_values.dim() == 0:
            raise ValueError("pixel_values must have at least one dimension")
        token_counts = self._image_token_counts_per_sample(input_ids, packed_seq_params)
        if image_grid_thw.shape[0] == 0:
            if pixel_values.shape[0] != 0 or int(token_counts.sum().item()) != 0:
                raise ValueError(
                    "Empty image_grid_thw requires empty pixel_values and no image tokens"
                )
            empty_counts = image_grid_thw.new_zeros(0, dtype=torch.long)
            return {
                "num_patches": 0,
                "num_grid_rows": 0,
                "num_image_tokens": 0,
                "patch_counts": empty_counts,
                "merged_patch_counts": empty_counts,
                "sample_grid_ranges": self._sample_grid_ranges(
                    token_counts=token_counts,
                    merged_patch_counts=empty_counts,
                ),
                "num_samples": int(token_counts.numel()),
            }
        if (image_grid_thw <= 0).any().item():
            raise ValueError("image_grid_thw entries must be positive")

        patch_counts = image_grid_thw.to(torch.long).prod(dim=1)
        num_patches = int(patch_counts.sum().item())
        if pixel_values.shape[0] != num_patches:
            raise ValueError(
                "pixel_values first dimension must match sum(T*H*W) from image_grid_thw: "
                f"pixel_values.shape[0]={pixel_values.shape[0]}, sum_grid={num_patches}"
            )

        merge_area = self.spatial_merge_size * self.spatial_merge_size
        if (patch_counts % merge_area != 0).any().item():
            raise ValueError(
                "Each visual grid T*H*W must be divisible by "
                f"spatial_merge_size^2={merge_area}; got {patch_counts.tolist()}"
            )

        num_visual_tokens = int((patch_counts // merge_area).sum().item())
        num_image_tokens = int(token_counts.sum().item())
        if num_image_tokens != num_visual_tokens:
            raise ValueError(
                "Number of image tokens must match post-merge visual tokens: "
                f"image_tokens={num_image_tokens}, visual_tokens={num_visual_tokens}"
            )
        if (
            not self.vision_mdp_enabled
            and self._needs_uniform_visual_token_counts_for_bridge()
            and token_counts.numel() > 1
            and token_counts.unique().numel() != 1
        ):
            raise ValueError(
                "pretrain_multimodal_mimo currently requires a uniform number of image "
                "tokens per sample because the colocated bridge equal-slices flattened "
                f"visual embeddings; got per-sample counts={token_counts.tolist()}"
            )

        return {
            "num_patches": num_patches,
            "num_grid_rows": int(image_grid_thw.shape[0]),
            "num_image_tokens": num_image_tokens,
            "patch_counts": patch_counts,
            "merged_patch_counts": patch_counts // merge_area,
            "sample_grid_ranges": self._sample_grid_ranges(
                token_counts=token_counts,
                merged_patch_counts=patch_counts // merge_area,
            ),
            "num_samples": int(token_counts.numel()),
        }

    @staticmethod
    def _sample_grid_ranges(token_counts, merged_patch_counts):
        """Map each language sample to its contiguous image_grid_thw row range."""
        # This relies on the dataset collating image_grid_thw in sample-major order.
        # It matches aggregate visual-token counts, not explicit image occurrence
        # boundaries in the prompt; malformed/reordered metadata with identical
        # totals can still map to the wrong sample before later validation fails.
        ranges = []
        row = 0
        for count in token_counts.tolist():
            start = row
            remaining = int(count)
            while remaining > 0 and row < merged_patch_counts.numel():
                remaining -= int(merged_patch_counts[row].item())
                row += 1
            if remaining != 0:
                raise ValueError(
                    "Could not map image tokens to contiguous image_grid_thw rows; "
                    f"remaining tokens for sample={remaining}"
                )
            ranges.append((start, row))
        if row != merged_patch_counts.numel():
            raise ValueError(
                "image_grid_thw contains rows that were not consumed by input_ids image tokens: "
                f"consumed_rows={row}, total_rows={merged_patch_counts.numel()}"
            )
        return ranges

    def _image_token_counts_per_sample(self, input_ids, packed_seq_params=None):
        """Return image-token counts per local sample for BSHD or packed THD input."""
        if packed_seq_params is not None:
            flat = input_ids.reshape(-1)
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded.to(device=flat.device)
            counts = [
                (
                    flat[int(cu_seqlens[i].item()) : int(cu_seqlens[i + 1].item())]
                    == self.image_token_id
                ).sum()
                for i in range(cu_seqlens.numel() - 1)
            ]
            return torch.stack(counts).to(torch.long) if counts else flat.new_zeros(0).long()
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [B, S], got {tuple(input_ids.shape)}")
        return (input_ids == self.image_token_id).sum(dim=1).to(torch.long)

    def _uses_colocated_bridge_for_visuals(self):
        if self.vision_grid is None or self.language_grid is None:
            return False
        if self.vision_mdp_enabled and self._can_use_mdp_scheduler():
            return False
        return self._grid_dim_size(self.vision_grid, "dp") != self._grid_dim_size(
            self.language_grid, "dp"
        )

    def _needs_uniform_visual_token_counts_for_bridge(self):
        if self.vision_grid is None or self.language_grid is None:
            return True
        vision_dp_size = self._grid_dim_size(self.vision_grid, "dp")
        language_dp_size = self._grid_dim_size(self.language_grid, "dp")
        # Equal-DP needs no slicing and fan-out has variable split sizes. Fan-in
        # still uses equal all-gather, so keep the conservative uniform check.
        return vision_dp_size > language_dp_size

    def _prepare_vision_inputs_for_bridge(self, pixel_values, image_grid_thw, vision_stats):
        """Redistribute local visual tensors for the colocated bridge.

        The full language MBS stays local. For vision-DP fan-in
        (language_tp > vision_tp), each vision DP slot encodes one MBS shard.
        For fan-out (vision_tp > language_tp), vision TP peers encode a grouped
        batch and the bridge narrows it back to language DP ranks.
        """
        if self.vision_grid is None or self.language_grid is None:
            return pixel_values, image_grid_thw, None

        vision_dp_size = self._grid_dim_size(self.vision_grid, "dp")
        language_dp_size = self._grid_dim_size(self.language_grid, "dp")
        if vision_dp_size == language_dp_size:
            return pixel_values, image_grid_thw, None
        if vision_dp_size > language_dp_size:
            pixel_values, image_grid_thw = self._slice_vision_inputs_for_fan_in(
                pixel_values, image_grid_thw, vision_stats
            )
            return pixel_values, image_grid_thw, None
        return self._gather_vision_inputs_for_fan_out(pixel_values, image_grid_thw, vision_stats)

    def _encode_images_with_mdp(self, pixel_values, image_grid_thw, vision_stats, packed=False):
        """Schedule variable-size images across the vision DP group and restore local order."""
        self._debug_mdp_stage("mdp_encode_start")
        self._validate_mdp_layout(packed=packed)
        schedule_group = self._mdp_schedule_group()
        local_descs = self._local_visual_descriptors(vision_stats)
        self._debug_mdp_stage(
            "before_gather_pixels",
            schedule_group_size=dist.get_world_size(schedule_group),
            schedule_group_rank=dist.get_rank(schedule_group),
            pixel_shape=tuple(pixel_values.shape),
            grid_shape=tuple(image_grid_thw.shape),
            local_descs=len(local_descs),
        )
        gathered_pixels = self._all_gather_variable_first_dim(pixel_values, schedule_group)
        self._debug_mdp_stage("after_gather_pixels")
        gathered_grids = self._all_gather_variable_first_dim(image_grid_thw, schedule_group)
        self._debug_mdp_stage("after_gather_grids")

        all_descs = []
        desc_id = 0
        for src_rank, grid in enumerate(gathered_grids):
            patch_counts = grid.to(torch.long).prod(dim=1).tolist()
            # Rebuild row-local descriptors for every source rank. Sample ownership is
            # only needed for this rank's final selection, so non-local rows use sample -1.
            patch_start = 0
            for row_idx, patch_count in enumerate(patch_counts):
                token_count = patch_count // (self.spatial_merge_size * self.spatial_merge_size)
                sample_idx = -1
                if src_rank == dist.get_rank(schedule_group):
                    for desc in local_descs:
                        if desc.row_idx == row_idx:
                            sample_idx = desc.sample_idx
                            break
                all_descs.append(
                    _VisualDescriptor(
                        desc_id=desc_id,
                        src_rank=src_rank,
                        sample_idx=sample_idx,
                        row_idx=row_idx,
                        patch_start=patch_start,
                        patch_end=patch_start + patch_count,
                        token_count=token_count,
                        cost=self._visual_cost(patch_count),
                    )
                )
                patch_start += patch_count
                desc_id += 1

        assignments = self._assign_visual_work(all_descs, self._mdp_num_workers())
        my_worker = self._mdp_worker_rank()
        my_descs = [desc for desc in all_descs if assignments[desc.desc_id] == my_worker]
        self._debug_mdp_schedule(local_descs, all_descs, assignments, my_descs)

        local_pixels, local_grid = self._materialize_assigned_visual_inputs(
            my_descs, gathered_pixels, gathered_grids, pixel_values
        )
        vision_submodule = unwrap_model(self.modality_submodules["images"])
        if local_grid.numel() == 0:
            if all_descs:
                dummy_pixels, dummy_grid = self._dummy_vision_inputs(pixel_values)
                dummy_embeddings = vision_submodule.encoders["qwen35_vision_encoder"](
                    pixel_values=dummy_pixels, grid_thw=dummy_grid
                )
                local_embeddings = dummy_embeddings[:0] + dummy_embeddings.sum() * 0.0
            else:
                local_embeddings = torch.empty(
                    (0, self.config.hidden_size), dtype=pixel_values.dtype, device=pixel_values.device
                )
        else:
            local_embeddings = vision_submodule.encoders["qwen35_vision_encoder"](
                pixel_values=local_pixels, grid_thw=local_grid
            )
        self._debug_mdp_backward(local_embeddings, my_descs)
        local_meta = torch.tensor(
            [[desc.desc_id, desc.token_count] for desc in my_descs],
            dtype=torch.long,
            device=pixel_values.device,
        )
        if local_meta.numel() == 0:
            local_meta = torch.empty((0, 2), dtype=torch.long, device=pixel_values.device)
        gathered_embeddings = self._all_gather_variable_first_dim(
            local_embeddings, schedule_group, differentiable=True
        )
        gathered_meta = self._all_gather_variable_first_dim(local_meta, schedule_group)
        embedding_by_desc = {}
        for emb, meta in zip(gathered_embeddings, gathered_meta):
            offset = 0
            for desc_id, token_count in meta.tolist():
                embedding_by_desc.setdefault(int(desc_id), []).append(emb[offset : offset + int(token_count)])
                offset += int(token_count)
        embedding_by_desc = {
            desc_id: torch.stack(chunks, dim=0).mean(dim=0)
            for desc_id, chunks in embedding_by_desc.items()
        }

        output_chunks = []
        local_rank = dist.get_rank(schedule_group)
        local_desc_ids = [desc.desc_id for desc in all_descs if desc.src_rank == local_rank]
        for desc_id in local_desc_ids:
            output_chunks.append(embedding_by_desc[desc_id])
        if output_chunks:
            return torch.cat(output_chunks, dim=0)
        empty_output = torch.empty(
            (0, self.config.hidden_size), dtype=local_embeddings.dtype, device=local_embeddings.device
        )
        if all_descs:
            # Even ranks with no local source images must participate in the
            # differentiable all-gather backward; otherwise peers hang in the
            # collective. Keep a zero-valued dependency on the gathered outputs.
            zero_dep = sum(emb.sum() for emb in gathered_embeddings) * 0.0
            empty_output = empty_output + zero_dep
        return empty_output

    def _debug_mdp_schedule(self, local_descs, all_descs, assignments, my_descs):
        if os.getenv("MIMO_MDP_DEBUG", "0") != "1":
            return
        if dist.get_rank(self._mdp_schedule_group()) != 0:
            return
        rank = dist.get_rank()
        global_rows = [
            {
                "desc": desc.desc_id,
                "src": desc.src_rank,
                "sample": desc.sample_idx,
                "row": desc.row_idx,
                "patches": desc.patch_end - desc.patch_start,
                "tokens": desc.token_count,
                "cost": desc.cost,
            }
            for desc in all_descs
        ]
        assigned = [
            {
                "desc": desc.desc_id,
                "src": desc.src_rank,
                "worker": assignments[desc.desc_id],
                "row": desc.row_idx,
                "patches": desc.patch_end - desc.patch_start,
                "tokens": desc.token_count,
                "cost": desc.cost,
            }
            for desc in all_descs
        ]
        loads = [0 for _ in range(self._mdp_num_workers())]
        counts = [0 for _ in loads]
        source_loads = [0 for _ in range(dist.get_world_size(self._mdp_schedule_group()))]
        source_counts = [0 for _ in source_loads]
        for desc in all_descs:
            worker = assignments[desc.desc_id]
            loads[worker] += desc.cost
            counts[worker] += 1
            source_loads[desc.src_rank] += desc.cost
            source_counts[desc.src_rank] += 1
        print(
            f"[MIMO MDP][rank {rank}] global_images={global_rows} "
            f"assignments={assigned} source_counts={source_counts} source_costs={source_loads} "
            f"worker_counts={counts} worker_costs={loads}",
            flush=True,
        )

    def _debug_mdp_backward(self, local_embeddings, my_descs):
        if os.getenv("MIMO_MDP_DEBUG", "0") != "1" or not local_embeddings.requires_grad:
            return

        rank = dist.get_rank()
        assigned = [(desc.desc_id, desc.token_count) for desc in my_descs]

        def _print_grad(grad):
            grad_f = grad.float()
            norm = grad_f.norm().item() if grad.numel() else 0.0
            abs_max = grad_f.abs().max().item() if grad.numel() else 0.0
            print(
                f"[MIMO MDP][rank {rank}][backward] local_embedding_shape="
                f"{tuple(local_embeddings.shape)} grad_norm={norm:.6e} "
                f"grad_abs_max={abs_max:.6e} assigned={assigned}",
                flush=True,
            )

        local_embeddings.register_hook(_print_grad)

    def _normalize_mdp_visual_inputs(self, pixel_values, image_grid_thw, device):
        if self._mdp_schedule_group() is None:
            raise ValueError("MIMO vision MDP requires an MDP scheduling process group")
        if image_grid_thw is None:
            image_grid_thw = torch.empty((0, 3), dtype=torch.long, device=device)
        if pixel_values is None:
            dtype = (
                torch.bfloat16
                if self.config.bf16
                else torch.float16
                if self.config.fp16
                else torch.float32
            )
            pixel_values = torch.empty((0, self._vision_patch_dim()), dtype=dtype, device=device)
        return pixel_values, image_grid_thw

    @staticmethod
    def _vision_patch_dim():
        return (
            VISION_KWARGS["in_channels"]
            * VISION_KWARGS["temporal_patch_size"]
            * VISION_KWARGS["patch_size"]
            * VISION_KWARGS["patch_size"]
        )

    def _dummy_vision_inputs(self, like_tensor):
        merge = self.spatial_merge_size
        grid = torch.tensor([[1, merge, merge]], dtype=torch.long, device=like_tensor.device)
        pixels = torch.zeros(
            (merge * merge, like_tensor.shape[1]), dtype=like_tensor.dtype, device=like_tensor.device
        )
        return pixels, grid

    def _validate_mdp_layout(self, packed):
        if self.vision_grid is None or self.language_grid is None or self._mdp_schedule_group() is None:
            raise ValueError("MIMO vision MDP requires module grids and an MDP scheduling group")
        if not self._can_use_mdp_scheduler():
            raise ValueError(
                "MIMO vision MDP scheduler currently supports equal TP/DP or vision fan-out only"
            )

    def _can_use_mdp_scheduler(self):
        if self.vision_grid is None or self.language_grid is None:
            return False
        vision_tp = self._grid_dim_size(self.vision_grid, "tp")
        language_tp = self._grid_dim_size(self.language_grid, "tp")
        vision_dp = self._grid_dim_size(self.vision_grid, "dp")
        language_dp = self._grid_dim_size(self.language_grid, "dp")
        return (vision_tp == language_tp and vision_dp == language_dp) or (
            vision_tp > language_tp and language_tp == 1 and vision_dp < language_dp
        )

    def _mdp_schedule_group(self):
        if self.vision_grid is None or self.language_grid is None:
            return None
        if self._grid_dim_size(self.vision_grid, "tp") == self._grid_dim_size(
            self.language_grid, "tp"
        ):
            return self.vision_dp_group
        return self.language_dp_group

    def _mdp_num_workers(self):
        if self._grid_dim_size(self.vision_grid, "tp") == self._grid_dim_size(
            self.language_grid, "tp"
        ):
            return dist.get_world_size(self.vision_dp_group)
        return self._grid_dim_size(self.vision_grid, "dp")

    def _mdp_worker_rank(self):
        if self._grid_dim_size(self.vision_grid, "tp") == self._grid_dim_size(
            self.language_grid, "tp"
        ):
            return dist.get_rank(self.vision_dp_group)
        return self._grid_dp_index(self.vision_grid)

    def _local_visual_descriptors(self, vision_stats):
        descs = []
        for sample_idx, (row_start, row_end) in enumerate(vision_stats["sample_grid_ranges"]):
            for row_idx in range(row_start, row_end):
                patch_start = int(vision_stats["patch_counts"][:row_idx].sum().item())
                patch_count = int(vision_stats["patch_counts"][row_idx].item())
                descs.append(
                    _VisualDescriptor(
                        desc_id=-1,
                        src_rank=dist.get_rank(self._mdp_schedule_group()),
                        sample_idx=sample_idx,
                        row_idx=row_idx,
                        patch_start=patch_start,
                        patch_end=patch_start + patch_count,
                        token_count=int(vision_stats["merged_patch_counts"][row_idx].item()),
                        cost=self._visual_cost(patch_count),
                    )
                )
        return descs

    def _visual_cost(self, patch_count):
        if self.vision_mdp_cost_proxy == "raw_patch":
            hidden = self.config.hidden_size
            return patch_count * patch_count * hidden + patch_count * hidden * hidden
        return patch_count * patch_count

    def _assign_visual_work(self, descs, num_workers):
        assignments = {}
        if self.vision_mdp_scheduler == "static":
            for idx, desc in enumerate(descs):
                assignments[desc.desc_id] = idx % num_workers
            return assignments
        loads = [0 for _ in range(num_workers)]
        for desc in sorted(descs, key=lambda d: (-d.cost, d.desc_id)):
            worker = min(range(num_workers), key=lambda idx: (loads[idx], idx))
            assignments[desc.desc_id] = worker
            loads[worker] += desc.cost
        return assignments

    @staticmethod
    def _materialize_assigned_visual_inputs(descs, gathered_pixels, gathered_grids, like_tensor):
        pixel_chunks = [gathered_pixels[d.src_rank][d.patch_start : d.patch_end] for d in descs]
        grid_chunks = [gathered_grids[d.src_rank][d.row_idx : d.row_idx + 1] for d in descs]
        if not pixel_chunks:
            return like_tensor[:0], torch.empty((0, 3), dtype=torch.long, device=like_tensor.device)
        return torch.cat(pixel_chunks, dim=0).contiguous(), torch.cat(grid_chunks, dim=0).contiguous()

    @staticmethod
    def _all_gather_variable_first_dim(tensor, group, differentiable=False):
        size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
        sizes = [torch.empty_like(size) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, size, group=group)
        sizes = [int(s.item()) for s in sizes]
        trailing_shape = torch.tensor(list(tensor.shape[1:]), dtype=torch.long, device=tensor.device)
        gathered_trailing = [torch.empty_like(trailing_shape) for _ in sizes]
        dist.all_gather(gathered_trailing, trailing_shape, group=group)
        if any(not torch.equal(trailing_shape, other) for other in gathered_trailing):
            raise ValueError(
                "_all_gather_variable_first_dim requires identical non-leading dimensions"
            )
        if differentiable:
            gathered = _VariableAllGather.apply(tensor, group, sizes)
            return [gathered[idx, :size].contiguous() for idx, size in enumerate(sizes)]
        max_size = max(max(sizes), 1)
        padded_shape = (max_size, *tensor.shape[1:])
        padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
        if tensor.shape[0] > 0:
            padded[: tensor.shape[0]] = tensor.contiguous()
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=group)
        return [item[:size].contiguous() for item, size in zip(gathered, sizes)]

    def _slice_vision_inputs_for_fan_in(self, pixel_values, image_grid_thw, vision_stats):
        """Give each vision DP slot its shard of the local language MBS."""
        vision_dp_size = self._grid_dim_size(self.vision_grid, "dp")
        language_dp_size = self._grid_dim_size(self.language_grid, "dp")
        if vision_dp_size % language_dp_size != 0:
            raise ValueError(
                "vision DP must be divisible by language DP for MIMO fan-in; "
                f"got vision DP={vision_dp_size}, language DP={language_dp_size}"
            )
        scale = vision_dp_size // language_dp_size
        num_samples = vision_stats["num_samples"]
        if num_samples % scale != 0:
            raise ValueError(
                "micro batch size must be divisible by MIMO fan-in scale; "
                f"num_samples={num_samples}, fan_in_scale={scale}"
            )

        vision_dp_idx = self._grid_dp_index(self.vision_grid)
        slot = vision_dp_idx % scale
        samples_per_slot = num_samples // scale
        sample_start = slot * samples_per_slot
        sample_end = sample_start + samples_per_slot

        sample_ranges = vision_stats["sample_grid_ranges"]
        row_start = sample_ranges[sample_start][0]
        row_end = sample_ranges[sample_end - 1][1]
        patch_counts = vision_stats["patch_counts"]
        patch_start = int(patch_counts[:row_start].sum().item())
        patch_end = int(patch_counts[:row_end].sum().item())
        return pixel_values[patch_start:patch_end], image_grid_thw[row_start:row_end]

    def _gather_vision_inputs_for_fan_out(self, pixel_values, image_grid_thw, vision_stats):
        """Gather local language-DP samples into the colocated vision TP group."""
        vision_tp_size = (
            dist.get_world_size(self.vision_tp_group) if self.vision_tp_group is not None else 1
        )
        language_tp_size = (
            dist.get_world_size(self.language_tp_group) if self.language_tp_group is not None else 1
        )
        if vision_tp_size == 1 or vision_tp_size == language_tp_size:
            return pixel_values, image_grid_thw
        if language_tp_size != 1:
            raise ValueError(
                "pretrain_multimodal_mimo only supports heterogeneous vision TP when "
                f"language TP is 1; got vision TP={vision_tp_size}, language TP={language_tp_size}"
            )

        self._debug_fanout(
            "before_input_gather",
            pixel_shape=tuple(pixel_values.shape),
            grid_shape=tuple(image_grid_thw.shape),
            stats={k: v for k, v in vision_stats.items() if not torch.is_tensor(v)},
        )
        local_stats = torch.tensor(
            [
                vision_stats["num_patches"],
                vision_stats["num_grid_rows"],
                vision_stats["num_image_tokens"],
            ],
            dtype=torch.long,
            device=pixel_values.device,
        )
        gathered_stats = torch.empty(
            (vision_tp_size, local_stats.numel()), dtype=torch.long, device=pixel_values.device
        )
        dist.all_gather_into_tensor(gathered_stats, local_stats, group=self.vision_tp_group)
        split_sizes = [int(size) for size in gathered_stats[:, 2].tolist()]
        self._debug_fanout(
            "after_stats_gather",
            gathered_stats=gathered_stats.cpu().tolist(),
            split_sizes=split_sizes,
        )

        gathered_pixels = self._all_gather_variable_first_dim(pixel_values, self.vision_tp_group)
        gathered_grids = self._all_gather_variable_first_dim(image_grid_thw, self.vision_tp_group)
        pixels = torch.cat(gathered_pixels, dim=0)
        grids = torch.cat(gathered_grids, dim=0)
        self._debug_fanout(
            "after_input_gather",
            pixel_shape=tuple(pixels.shape),
            grid_shape=tuple(grids.shape),
        )
        return pixels, grids, split_sizes

    @staticmethod
    def _grid_dim_size(grid, dim_name):
        return grid.shape[grid.dim_names.index(dim_name)]

    @staticmethod
    def _grid_dp_index(grid):
        rank = dist.get_rank()
        for dp_idx, tp_group in enumerate(grid.get_rank_enum(["tp"])):
            if rank in tp_group:
                return dp_idx
        raise RuntimeError(f"rank {rank} is not in grid rank span")

    def _forward_all_modules(
        self,
        input_ids,
        position_ids,
        attention_mask,
        loss_mask,
        labels,
        modality_inputs,
        image_bridge_split_sizes=None,
        packing_kwargs=None,
        packed_seq_params=None,
        padding_mask=None,
    ):
        """Forward colocated Qwen3.5-VL MIMO while preserving ``input_ids`` for MTP."""
        if packing_kwargs is not None:
            raise NotImplementedError(
                "packing_kwargs is not supported by pretrain_multimodal_mimo yet"
            )

        modality_embeddings = {}
        for modality_name, submodule in self.modality_submodules.items():
            if (
                modality_inputs
                and modality_name in modality_inputs
                and "precomputed_embeddings" in modality_inputs[modality_name]
            ):
                modality_embeddings[modality_name] = modality_inputs[modality_name][
                    "precomputed_embeddings"
                ]
                continue
            if (
                modality_inputs
                and modality_name in modality_inputs
                and modality_inputs[modality_name] is not None
            ):
                embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                if embeddings is not None:
                    if modality_name == "images" and image_bridge_split_sizes is not None:
                        embeddings._mimo_bridge_split_sizes = image_bridge_split_sizes
                    self._debug_fanout(
                        "vision_encoder_output",
                        embeddings_shape=tuple(embeddings.shape),
                        split_sizes=image_bridge_split_sizes,
                    )
                    modality_embeddings[modality_name] = embeddings

        if self.colocated_comms and not (
            modality_inputs
            and "images" in modality_inputs
            and "precomputed_embeddings" in modality_inputs["images"]
        ):
            self._debug_fanout(
                "before_colocated_bridge",
                images_shape=(
                    tuple(modality_embeddings["images"].shape)
                    if "images" in modality_embeddings
                    else None
                ),
            )
            modality_embeddings = self._apply_colocated_comms(modality_embeddings)
            self._debug_fanout(
                "after_colocated_bridge",
                images_shape=(
                    tuple(modality_embeddings["images"].shape)
                    if "images" in modality_embeddings
                    else None
                ),
            )

        self._validate_image_embedding_count(input_ids, modality_embeddings)

        text_embeddings = self.get_text_embeddings(input_ids, position_ids, self.special_token_ids)
        if "images" in modality_embeddings and modality_embeddings["images"].numel() == 0:
            # Empty image slices still need a graph edge so variable fan-out/MDP
            # collectives run their backward on every rank in the group.
            text_embeddings = text_embeddings + modality_embeddings["images"].sum() * 0.0
        modality_embeddings["text"] = text_embeddings
        combined_embeddings = self.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,
            input_ids=input_ids,
            special_token_ids=self.special_token_ids,
        )

        combined_embeddings, labels, loss_mask, packed_seq_params = self._shard_language_inputs(
            embeddings=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

        lm_input_ids = input_ids
        lm_position_ids = position_ids
        if self.config.sequence_parallel:
            # THD PackedSeqParams intentionally remains global/duplicated across
            # TP ranks. Match MCore GPT's THD+SP contract by sharding sequence
            # tensors but not the packed metadata.
            lm_input_ids = self._sp_shard_batch_sequence(input_ids)
            lm_position_ids = self._sp_shard_position_ids(position_ids)
        if padding_mask is not None and self.config.sequence_parallel:
            padding_mask = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    padding_mask.transpose(0, 1).contiguous(), group=self.language_model.tp_group
                )
                .transpose(0, 1)
                .contiguous()
            )

        lm_output = self.language_model(
            input_ids=lm_input_ids,
            position_ids=lm_position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )
        return lm_output, loss_mask

    def _sp_shard_batch_sequence(self, tensor):
        """Shard a [B, S] tensor along S using the language TP/SP group."""
        return (
            tensor_parallel.scatter_to_sequence_parallel_region(
                tensor.transpose(0, 1).contiguous(), group=self.language_model.tp_group
            )
            .transpose(0, 1)
            .contiguous()
        )

    def _sp_shard_position_ids(self, position_ids):
        """Shard Qwen MRoPE position IDs to match SP-local decoder input."""
        if position_ids is None:
            return None
        if position_ids.dim() == 3:
            return (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    position_ids.permute(2, 0, 1).contiguous(), group=self.language_model.tp_group
                )
                .permute(1, 2, 0)
                .contiguous()
            )
        return self._sp_shard_batch_sequence(position_ids)

    def _validate_image_embedding_count(self, input_ids, modality_embeddings):
        """Fail early when visual embeddings cannot align to local image tokens."""
        num_image_tokens = int((input_ids == self.image_token_id).sum().item())
        image_embeddings = modality_embeddings.get("images")
        if num_image_tokens and image_embeddings is None:
            raise ValueError(
                f"input_ids contain {num_image_tokens} image tokens but no image embeddings were produced"
            )
        if image_embeddings is not None and image_embeddings.size(0) != num_image_tokens:
            raise ValueError(
                "Number of image tokens does not match bridged image embeddings: "
                f"image_tokens={num_image_tokens}, image_embeddings={image_embeddings.size(0)}"
            )

    def _debug_fanout(self, stage, **kwargs):
        if os.getenv("MIMO_MDP_DEBUG", "0") != "1":
            return
        rank = dist.get_rank()
        vision_tp_rank = (
            dist.get_rank(self.vision_tp_group) if self.vision_tp_group is not None else 0
        )
        language_tp_rank = (
            dist.get_rank(self.language_tp_group) if self.language_tp_group is not None else 0
        )
        print(
            f"[MIMO fanout][rank {rank}][vtp {vision_tp_rank}][ltp {language_tp_rank}] "
            f"{stage} {kwargs}",
            flush=True,
        )


def _build_qwen35_vl_mimo_model(
    args, language_config, vision_config, pre_process=True, post_process=True
):
    """Build a colocated Qwen3.5-VL MIMO model."""
    if getattr(args, "use_megatron_fsdp", False):
        raise ValueError(
            "pretrain_multimodal_mimo does not support --use-megatron-fsdp yet. "
            "Whole-model FSDP uses one process-group layout, which is unsafe for "
            "heterogeneous MIMO vision/language TP-DP layouts."
        )
    if getattr(args, "use_torch_fsdp2", False):
        raise ValueError("pretrain_multimodal_mimo does not support --use-torch-fsdp2 yet")

    module_to_grid_map, vision_pg_collection, language_pg_collection = (
        _build_module_parallel_context(args)
    )
    if vision_pg_collection is not None:
        vision_config.tensor_model_parallel_size = vision_pg_collection.tp.size()
        vision_config.context_parallel_size = 1
        vision_config.pipeline_model_parallel_size = 1
        vision_config.expert_model_parallel_size = 1

    language_spec = get_qwen35_vl_language_spec(config=language_config, vp_stage=None, pp_rank=None)

    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=language_config,
            spec=language_spec,
            use_transformer_engine=(args.transformer_impl == "transformer_engine"),
            vp_stage=None,
            pp_rank=None,
        )

    spatial_merge_size = VISION_KWARGS["spatial_merge_size"]
    vkw = dict(VISION_KWARGS)
    vkw["out_hidden_size"] = language_config.hidden_size

    vision_encoder_spec = ModuleSpec(
        module=Qwen35VLVisionEncoder,
        params={
            "config": vision_config,
            "transformer_layer_spec": get_qwen35_vl_vision_spec(),
            "in_channels": vkw["in_channels"],
            "patch_size": vkw["patch_size"],
            "temporal_patch_size": vkw["temporal_patch_size"],
            "spatial_merge_size": vkw["spatial_merge_size"],
            "out_hidden_size": vkw["out_hidden_size"],
            "max_num_positions": vkw["max_num_positions"],
            "pg_collection": vision_pg_collection,
        },
    )
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": vision_pg_collection},
        submodules={"encoders": {"qwen35_vision_encoder": vision_encoder_spec}},
    )

    share_embeddings = not getattr(args, "untie_embeddings_and_output_weights", False)
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": language_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "parallel_output": True,
            "share_embeddings_and_output_weights": share_embeddings,
            "position_embedding_type": "mrope",
            "rotary_percent": ROTARY_PERCENT,
            "rotary_base": ROTARY_BASE,
            "mtp_block_spec": mtp_block_spec,
        },
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)},
        module_to_grid_map=module_to_grid_map,
    )
    return Qwen35VLMimoModel(
        mimo_config,
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        spatial_merge_size=spatial_merge_size,
        vision_tp_group=vision_pg_collection.tp if vision_pg_collection is not None else None,
        cp_group=language_pg_collection.cp if language_pg_collection is not None else None,
        tp_group=language_pg_collection.tp if language_pg_collection is not None else None,
        vision_dp_group=vision_pg_collection.dp if vision_pg_collection is not None else None,
        language_dp_group=language_pg_collection.dp if language_pg_collection is not None else None,
        vision_mdp_enabled=getattr(args, "mimo_vision_mdp", False),
        vision_mdp_scheduler=getattr(args, "mimo_vision_mdp_scheduler", "lpt"),
        vision_mdp_cost_proxy=getattr(args, "mimo_vision_mdp_cost_proxy", "p2"),
        vision_grid=module_to_grid_map["images"] if module_to_grid_map is not None else None,
        language_grid=(
            module_to_grid_map[MIMO_LANGUAGE_MODULE_KEY] if module_to_grid_map is not None else None
        ),
    )


def _prepare_vision_cuda_graph_args(args):
    """Enable graph-safe RNG when only the vision tower uses local CUDA graphs."""
    if not getattr(args, "vision_layer_cuda_graph", False):
        return
    args.te_rng_tracker = True
    if getattr(args, "cuda_graph_impl", "none") == "none":
        args._multimodal_language_cuda_graph_impl = "none"
        args.cuda_graph_impl = "local"
        from megatron.core.transformer.enums import InferenceCudaGraphScope

        args.inference_cuda_graph_scope = InferenceCudaGraphScope.layer


def model_provider(pre_process: bool = True, post_process: bool = True, **kwargs):
    """Build a Qwen3.5-VL MIMO model from normal multimodal_dev args."""
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")
    if model_arch != "qwen35_vl":
        raise ValueError("pretrain_multimodal_mimo currently supports only --model-arch qwen35_vl")

    language_config = core_transformer_config_from_args(args)
    if getattr(args, "_multimodal_language_cuda_graph_impl", None) is not None:
        language_config.cuda_graph_impl = args._multimodal_language_cuda_graph_impl
        language_config.cuda_graph_modules = []
    post_language_config(language_config, args)

    vision_config = get_qwen35_vl_vision_config(
        num_layers_override=getattr(args, "vision_num_layers", None),
        variant=getattr(args, "model_variant", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16
    vision_config.qwen_vision_max_packed_tokens = getattr(
        args, "vision_max_packed_tokens", None
    )
    vision_config.qwen_vision_max_packed_sequences = getattr(
        args, "vision_max_packed_sequences", None
    )
    vision_config.qwen_vision_max_grid_size = getattr(args, "vision_max_grid_size", None)
    if getattr(args, "vision_layer_cuda_graph", False):
        vision_config.cuda_graph_impl = "local"
        vision_config.cuda_graph_modules = []
    if getattr(args, "recompute_vision", False):
        vision_config.recompute_granularity = "full"
        vision_config.recompute_method = "uniform"
        vision_config.recompute_num_layers = 1

    set_vision_flops_metadata(args, language_config, vision_config)
    return _build_qwen35_vl_mimo_model(
        args, language_config, vision_config, pre_process=pre_process, post_process=post_process
    )


def _resolve_provider_fn(provider_fn):
    """Resolve a provider that may be a dotted import path string."""
    if isinstance(provider_fn, str):
        module_path, func_name = provider_fn.rsplit(".", 1)
        provider_fn = getattr(importlib.import_module(module_path), func_name)
    return provider_fn


def datasets_provider(train_val_test_num_samples):
    """Reuse qwen35_vl multimodal_dev dataset providers."""
    args = get_args()
    provider = getattr(args, "dataset_provider", "mock")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    available = MODEL_REGISTRY["qwen35_vl"].get("dataset_providers", {})
    if provider not in available:
        raise ValueError(
            f"Unknown dataset provider '{provider}' for qwen35_vl. "
            f"Available: {list(available.keys())}"
        )
    return _resolve_provider_fn(available[provider])(train_val_test_num_samples)


def forward_step(data_iterator, model):
    """Forward step for Qwen3.5-VL MIMO training."""
    batch = get_batch(data_iterator)
    if batch is None:
        return None, None

    pixel_values = batch.get("pixel_values", None)
    if (
        pixel_values is not None
        and pixel_values.is_floating_point()
        and pixel_values.dtype == torch.float32
    ):
        pixel_values = pixel_values.bfloat16()

    output_tensor = model(
        input_ids=batch["input_ids"],
        position_ids=batch.get("position_ids"),
        attention_mask=batch.get("attention_mask", None),
        labels=batch.get("labels", None),
        loss_mask=batch.get("loss_mask", None),
        padding_mask=batch.get("padding_mask", None),
        pixel_values=pixel_values,
        image_grid_thw=batch.get("image_grid_thw", None),
        packed_seq_params=batch.get("packed_seq_params", None),
    )

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(batch["input_ids"], dtype=torch.float)
    loss_mask = MultimodalModel.cp_split_loss_mask(loss_mask, batch.get("packed_seq_params", None))

    return output_tensor, partial(loss_func, loss_mask)


if __name__ == "__main__":
    datasets_provider.is_distributed = True

    args = parse_and_validate_args(extra_args_provider=add_multimodal_mimo_args, args_defaults={})
    if args.pipeline_model_parallel_size > 1:
        raise ValueError("pretrain_multimodal_mimo currently supports only PP=1")
    if args.context_parallel_size > 1:
        raise ValueError("pretrain_multimodal_mimo currently supports only CP=1")

    _prepare_vision_cuda_graph_args(args)
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config, datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step
    )
