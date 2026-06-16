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
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
from megatron.core.enums import ModelType
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args


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
        "--mimo-vision-data-parallel-size",
        type=int,
        default=None,
        help="Vision encoder DP size for colocated MIMO. Defaults to world_size / vision TP.",
    )
    group.add_argument(
        "--mimo-disable-module-grid-map",
        action="store_true",
        default=False,
        help="Disable module_to_grid_map and use legacy all-modules-colocated MIMO behavior.",
    )
    return parser


def _create_mimo_grid(tp: int, dp: int, ep: int = 1) -> HyperCommGrid:
    """Create a colocated MIMO HyperCommGrid and required process groups."""
    grid = HyperCommGrid(
        shape=[tp, 1, 1, dp, ep, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=0,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["tp", "cp"])
    grid.create_pg(["ep"])
    grid.create_pg(["expt_dp"])
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "ep", "pp"])
    grid.create_pg(["dp", "ep"])
    grid.create_pg(["tp", "cp", "ep", "pp", "dp"])
    return grid


def _pg_collection_from_grid(grid: HyperCommGrid) -> ProcessGroupCollection:
    """Build a ProcessGroupCollection from a MIMO HyperCommGrid."""
    pg = ProcessGroupCollection()
    pg.tp = grid.get_pg("tp")
    pg.cp = grid.get_pg("cp")
    pg.pp = grid.get_pg("pp")
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.ep = grid.get_pg("ep")
    pg.tp_cp = grid.get_pg(["tp", "cp"])
    pg.expt_dp = grid.get_pg("expt_dp")
    pg.mp = grid.get_pg(["tp", "pp"])
    pg.tp_ep_pp = grid.get_pg(["tp", "ep", "pp"])
    pg.intra_dist_opt = grid.get_pg(["tp", "cp", "ep", "pp", "dp"])
    return pg


def _build_module_parallel_context(args):
    """Build colocated MIMO grids and process-group collections."""
    if getattr(args, "mimo_disable_module_grid_map", False):
        return None, None, None

    world_size = dist.get_world_size()
    vision_tp = getattr(args, "mimo_vision_tensor_model_parallel_size", None)
    if vision_tp is None:
        vision_tp = args.tensor_model_parallel_size
    vision_dp = getattr(args, "mimo_vision_data_parallel_size", None)
    if vision_dp is None:
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
        raise ValueError(
            f"world_size={world_size} must be divisible by language TP={language_tp}"
        )
    language_dp = world_size // denom

    vision_size = vision_tp * vision_dp
    language_size = language_tp * language_dp * language_ep
    if vision_size != world_size or language_size != world_size:
        raise ValueError(
            "Colocated MIMO grids must both span the full world. "
            f"Got vision TP*DP={vision_tp}*{vision_dp}={vision_size}, "
            f"language TP*DP*EP={language_tp}*{language_dp}*{language_ep}={language_size}, "
            f"world_size={world_size}."
        )

    vision_grid = _create_mimo_grid(tp=vision_tp, dp=vision_dp, ep=1)
    language_grid = _create_mimo_grid(tp=language_tp, dp=language_dp, ep=language_ep)
    module_to_grid_map = {"images": vision_grid, MIMO_LANGUAGE_MODULE_KEY: language_grid}
    return (
        module_to_grid_map,
        _pg_collection_from_grid(vision_grid),
        _pg_collection_from_grid(language_grid),
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
        cp_group=None,
        tp_group=None,
    ) -> None:
        super().__init__(mimo_config, cp_group=cp_group, tp_group=tp_group)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size

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
        if decoder_input is not None:
            raise NotImplementedError(
                "decoder_input is not supported by pretrain_multimodal_mimo yet"
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "packed_seq_params is not supported by pretrain_multimodal_mimo yet"
            )
        if padding_mask is not None and padding_mask.any():
            raise NotImplementedError(
                "padding_mask forwarding is not supported by pretrain_multimodal_mimo yet"
            )
        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )

        modality_inputs = None
        if pixel_values is not None:
            modality_inputs = {
                "images": {
                    "qwen35_vision_encoder": {
                        "pixel_values": pixel_values,
                        "grid_thw": image_grid_thw,
                    }
                }
            }

        output, _ = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            labels=labels,
            modality_inputs=modality_inputs,
        )
        return output


def _build_qwen35_vl_mimo_model(
    args, language_config, vision_config, pre_process=True, post_process=True
):
    """Build a colocated Qwen3.5-VL MIMO model."""
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
        cp_group=language_pg_collection.cp if language_pg_collection is not None else None,
        tp_group=language_pg_collection.tp if language_pg_collection is not None else None,
    )


def model_provider(pre_process: bool = True, post_process: bool = True, **kwargs):
    """Build a Qwen3.5-VL MIMO model from normal multimodal_dev args."""
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")
    if model_arch != "qwen35_vl":
        raise ValueError("pretrain_multimodal_mimo currently supports only --model-arch qwen35_vl")

    language_config = core_transformer_config_from_args(args)
    post_language_config(language_config, args)

    vision_config = get_qwen35_vl_vision_config(
        num_layers_override=getattr(args, "vision_num_layers", None),
        variant=getattr(args, "model_variant", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16
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

    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config, datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step
    )
