# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-native Qwen3.5-VL vision encoder.

Architecture (matches HF ``Qwen3VLVisionModel`` exactly):

  PatchEmbed (Conv3d)
    → learned position embedding (bilinear interpolation)
    → 2D Vision RoPE
    → TransformerBlock × N  (with PackedSeqParams / THD attention)
    → PatchMerger  (per-token LN → spatial merge → MLP)

Key design choices:
  * ``Conv3d`` patch embedding is replicated across TP ranks (no MCore
    equivalent for 3D convolutions).
  * ``PatchMerger`` MLP uses ``ColumnParallelLinear`` / ``RowParallelLinear``
    for TP sharding.
  * Inherits from ``VisionModule``.
  * Expects pixel values in block-merge order (as produced by the HF
    processor) so the merger's simple reshape is correct.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

# -------------------------------------------------------------------
# PatchEmbed — Conv3d (replicated, no TP sharding)
# -------------------------------------------------------------------

class Qwen35VLPatchEmbed(MegatronModule):
    """3D convolution patch embedding matching HF ``Qwen3VLVisionPatchEmbed``.

    Uses ``nn.Conv3d`` with kernel = stride = ``[temporal_patch_size,
    patch_size, patch_size]`` and ``bias=True``.  The module is replicated
    across TP ranks (no MCore equivalent for 3D conv).

    Args:
        config: TransformerConfig (used by MegatronModule base).
        in_channels: Number of input channels (3 for RGB).
        hidden_size: Output embedding dimension.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
    ):
        super().__init__(config=config)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel = [temporal_patch_size, patch_size, patch_size]
        self.proj = torch.nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pixel_values: ``[total_patches, C * T * pH * pW]``
                pre-extracted flat patches.

        Returns:
            Patch embeddings ``[total_patches, hidden_size]``.
        """
        target_dtype = self.proj.weight.dtype
        pixel_values = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values.to(dtype=target_dtype)).view(
            -1, self.hidden_size
        )


# -------------------------------------------------------------------
# VisionRotaryEmbedding — 1D frequency table
# -------------------------------------------------------------------

class Qwen35VLVisionRotaryEmbedding(MegatronModule):
    """1D rotary position frequency table for the vision transformer.

    Generates RoPE frequencies for integer positions ``0 .. seqlen-1``.
    The encoder maps 2D (row, col) positions to embeddings via table
    lookup.  Matches HF ``Qwen3VLVisionRotaryEmbedding``.

    Args:
        dim: Frequency dimension (``head_dim // 2``).
        theta: RoPE base frequency.
        config: Optional TransformerConfig for MegatronModule base.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        config: Optional[TransformerConfig] = None,
    ):
        super().__init__(config=config)
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_inv_freq(self, device: torch.device) -> Tensor:
        """Return ``inv_freq`` in float32 on *device*.

        Always recomputes in float32 regardless of the buffer's stored dtype.
        This matches Bridge's lazy-init behaviour where ``inv_freq`` is
        constructed fresh (in float32) on the first forward call, after any
        ``model.bfloat16()`` cast has already occurred.
        """
        return 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0, self.dim, 2,
                    dtype=torch.float32, device=device,
                )
                / self.dim
            )
        )

    def forward(
        self,
        seqlen: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Frequency lookup table for positions ``0 .. seqlen-1``.

        Args:
            seqlen: Number of positions.
            device: Runtime device (required for meta-init safety).

        Returns:
            ``[seqlen, dim // 2]`` frequencies.
        """
        if device is None:
            if self.inv_freq.device.type != "meta":
                device = self.inv_freq.device
            else:
                device = torch.device(
                    "cuda", torch.cuda.current_device()
                )
        inv_freq = self._get_inv_freq(device)
        seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        return torch.outer(seq, inv_freq)


# -------------------------------------------------------------------
# PatchMerger — per-token LN, spatial merge, TP-sharded MLP
# -------------------------------------------------------------------

class Qwen35VLPatchMerger(MegatronModule):
    """Spatial patch merger matching HF ``Qwen3VLVisionPatchMerger``.

    Per-token ``LayerNorm`` on ``hidden_size`` → reshape to merge
    ``spatial_merge_size ** 2`` adjacent patches → two-layer MLP
    (``ColumnParallelLinear`` → GELU → ``RowParallelLinear``).

    MLP dimensions: ``merge_dim → merge_dim → out_hidden_size``
    where ``merge_dim = hidden_size * spatial_merge_size ** 2``.

    Args:
        config: TransformerConfig (provides TP settings, init_method).
        hidden_size: Per-token hidden size from the ViT.
        out_hidden_size: Output dimension (language model hidden_size).
        spatial_merge_size: Merge factor per spatial dimension.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int = 1152,
        out_hidden_size: int = 3584,
        spatial_merge_size: int = 2,
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.spatial_merge_size = spatial_merge_size
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        merge_dim = self.merge_dim
        tp_group = pg_collection.tp if pg_collection is not None else None

        self.patch_norm = TENorm(config=config, hidden_size=hidden_size, eps=1e-6)
        self.linear_fc1 = build_module(
            ColumnParallelLinear,
            merge_dim,
            merge_dim,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            tp_group=tp_group,
        )
        self.linear_fc2 = build_module(
            RowParallelLinear,
            merge_dim,
            out_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Merge patches spatially.

        Args:
            hidden_states: ``[total_patches, hidden_size]`` in block-merge
                order from the ViT transformer blocks.

        Returns:
            ``[total_merged_patches, out_hidden_size]``.
        """
        hidden_states = self.patch_norm(hidden_states)
        merged = hidden_states.view(-1, self.merge_dim)
        merged, _ = self.linear_fc1(merged)
        # Match official HuggingFace Qwen3VLVisionPatchMerger (default approximate='none').
        merged = torch.nn.functional.gelu(merged, approximate="none")
        merged, _ = self.linear_fc2(merged)
        return merged


# -------------------------------------------------------------------
# Qwen35VLVisionEncoder — top-level encoder module
# -------------------------------------------------------------------

class Qwen35VLVisionEncoder(VisionModule):
    """Megatron-native Qwen3.5-VL vision encoder.

    Processes image / video inputs through:

    1. ``Qwen35VLPatchEmbed``  (Conv3d)
    2. Learned ``nn.Embedding`` position table with bilinear interpolation
    3. 2D Vision RoPE from ``(row, col)`` patch positions
    4. ``TransformerBlock`` × N  with ``PackedSeqParams`` (THD attention)
    5. ``Qwen35VLPatchMerger``

    Output dimension matches the language model ``hidden_size``.

    Args:
        config: Vision ``TransformerConfig``.
        transformer_layer_spec: ``ModuleSpec`` for ViT layers.
        in_channels: Image channels (3 for RGB).
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor.
        out_hidden_size: Output dim (language decoder hidden_size).
        max_num_positions: Size of the learned position table.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec = None,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 3584,
        max_num_positions: int = 2304,
        pg_collection=None,
    ):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.spatial_merge_size = spatial_merge_size

        # --- Patch embedding (Conv3d) ---
        self.patch_embed = Qwen35VLPatchEmbed(
            config=config,
            in_channels=in_channels,
            hidden_size=config.hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )

        # --- Learned position embedding with bilinear interpolation ---
        self.pos_embed = torch.nn.Embedding(
            max_num_positions, config.hidden_size,
        )
        self.num_grid_per_side = int(max_num_positions ** 0.5)

        # --- Vision rotary embeddings ---
        head_dim = config.hidden_size // config.num_attention_heads
        self.rot_pos_emb = Qwen35VLVisionRotaryEmbedding(
            head_dim // 2, config=config,
        )

        # --- Transformer blocks ---
        if transformer_layer_spec is None:
            from examples.multimodal_dev.models.qwen35_vl.specs import get_qwen35_vl_vision_spec
            transformer_layer_spec = get_qwen35_vl_vision_spec()

        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=False,
            pg_collection=pg_collection,
        )

        # --- Patch merger ---
        self.merger = Qwen35VLPatchMerger(
            config=config,
            hidden_size=config.hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            pg_collection=pg_collection,
        )

    # ---------------------------------------------------------------
    # Learned position embedding with bilinear interpolation
    # ---------------------------------------------------------------

    def _fast_pos_embed_interpolate(
        self, grid_thw: Tensor,
    ) -> Tensor:
        """Bilinear interpolation of the learned 2D position table.

        Matches HF ``Qwen3VLVisionModel.fast_pos_embed_interpolate``.

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) in patch-grid units.

        Returns:
            ``[total_patches, hidden_size]`` position embeddings in
            block-merge order.
        """
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device
        n = self.num_grid_per_side

        idx_list: List[List[int]] = [[] for _ in range(4)]
        weight_list: List[List[float]] = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)
            h_idxs = torch.linspace(0, n - 1, h)
            w_idxs = torch.linspace(0, n - 1, w)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=n - 1)
            w_ceil = (w_floor + 1).clip(max=n - 1)

            dh = h_idxs - h_floor.float()
            dw = w_idxs - w_floor.float()

            base_h = h_floor * n
            base_h_ceil = h_ceil * n

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(
            idx_list, dtype=torch.long, device=device,
        )
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=device,
        )
        pos_embeds = (
            self.pos_embed(idx_tensor).to(device)
            * weight_tensor[:, :, None]
        )
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1]
            + pos_embeds[2] + pos_embeds[3]
        )

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        merge = self.spatial_merge_size
        result = []
        for pe, t, h, w in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws,
        ):
            pe = pe.repeat(t, 1)
            pe = (
                pe.view(
                    t, h // merge, merge, w // merge, merge, -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pe)

        return torch.cat(result)

    def _packed_token_metadata(
        self,
        grid_thw: Tensor,
        total_tokens: int,
        real_total_tokens: int,
    ):
        """Return per-token image-local row/col coordinates for packed vision tokens."""
        device = grid_thw.device
        image_lens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        image_cu = F.pad(image_lens.cumsum(dim=0, dtype=torch.long), (1, 0), value=0)
        token_pos = torch.arange(total_tokens, device=device, dtype=torch.long)
        valid = token_pos < int(real_total_tokens)
        safe_pos = torch.minimum(
            token_pos,
            token_pos.new_full((), max(int(real_total_tokens) - 1, 0)),
        )
        image_idx = torch.searchsorted(image_cu, safe_pos, right=True) - 1
        image_idx = image_idx.clamp(min=0, max=grid_thw.shape[0] - 1)

        height = grid_thw[image_idx, 1].long()
        width = grid_thw[image_idx, 2].long()
        spatial_tokens = height * width
        local_pos = token_pos - image_cu[image_idx]
        spatial_pos = torch.remainder(local_pos, spatial_tokens.clamp(min=1))

        merge = int(self.spatial_merge_size)
        merge_area = merge * merge
        merged_w = (width // merge).clamp(min=1)
        block_idx = spatial_pos // merge_area
        intra_idx = spatial_pos % merge_area
        row = (block_idx // merged_w) * merge + intra_idx // merge
        col = (block_idx % merged_w) * merge + intra_idx % merge
        row = row.masked_fill(~valid, 0)
        col = col.masked_fill(~valid, 0)
        return row, col, height, width, valid

    @staticmethod
    def _pad_grid_thw(grid_thw: Tensor, max_num_rows: Optional[int]) -> Tensor:
        """Pad image grid metadata rows with zero-sized dummy images."""
        if max_num_rows is None:
            return grid_thw
        if grid_thw.shape[0] > max_num_rows:
            raise ValueError(
                f"Qwen vision image metadata rows ({grid_thw.shape[0]}) exceed configured "
                f"static target ({max_num_rows}). Increase --vision-max-packed-sequences "
                "or use a larger bucket."
            )
        if grid_thw.shape[0] == max_num_rows:
            return grid_thw
        pad_shape = (int(max_num_rows) - grid_thw.shape[0], grid_thw.shape[1])
        return torch.cat((grid_thw, grid_thw.new_zeros(pad_shape)), dim=0)

    def _fast_pos_embed_interpolate_static(
        self,
        grid_thw: Tensor,
        total_tokens: int,
        real_total_tokens: int,
    ) -> Tensor:
        """Vectorized learned position interpolation for static packed vision buckets."""
        row, col, height, width, valid = self._packed_token_metadata(
            grid_thw, total_tokens, real_total_tokens
        )
        n = self.num_grid_per_side
        dtype = self.pos_embed.weight.dtype

        h_den = (height - 1).clamp(min=1).to(torch.float32)
        w_den = (width - 1).clamp(min=1).to(torch.float32)
        h_pos = row.to(torch.float32) * float(n - 1) / h_den
        w_pos = col.to(torch.float32) * float(n - 1) / w_den

        h_floor = h_pos.floor().long()
        w_floor = w_pos.floor().long()
        h_ceil = (h_floor + 1).clamp(max=n - 1)
        w_ceil = (w_floor + 1).clamp(max=n - 1)
        dh = (h_pos - h_floor.to(torch.float32)).to(dtype)
        dw = (w_pos - w_floor.to(torch.float32)).to(dtype)

        idx00 = h_floor * n + w_floor
        idx01 = h_floor * n + w_ceil
        idx10 = h_ceil * n + w_floor
        idx11 = h_ceil * n + w_ceil
        out = (
            self.pos_embed(idx00) * ((1 - dh) * (1 - dw)).unsqueeze(-1)
            + self.pos_embed(idx01) * ((1 - dh) * dw).unsqueeze(-1)
            + self.pos_embed(idx10) * (dh * (1 - dw)).unsqueeze(-1)
            + self.pos_embed(idx11) * (dh * dw).unsqueeze(-1)
        )
        return out * valid.to(dtype).unsqueeze(-1)

    # ---------------------------------------------------------------
    # 2D Vision RoPE
    # ---------------------------------------------------------------

    def _compute_rotary_pos_emb(self, grid_thw: Tensor) -> Tensor:
        """Compute 2D Vision RoPE for all patches in block-merge order.

        Matches HF ``Qwen3VLVisionModel.rot_pos_emb``.

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) per image.

        Returns:
            ``[total_patches, head_dim // 2]`` raw RoPE frequencies.
        """
        merge = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(int(h), int(w)) for _, h, w in grid_thw_list)
        freq_table = self.rot_pos_emb(
            max_hw, device=grid_thw.device,
        )
        device = freq_table.device

        total_tokens = sum(
            int(t) * int(h) * int(w) for t, h, w in grid_thw_list
        )
        pos_ids = torch.empty(
            (total_tokens, 2), dtype=torch.long, device=device,
        )

        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames = int(num_frames)
            height = int(height)
            width = int(width)
            merged_h = height // merge
            merged_w = width // merge

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge, device=device)
            intra_col = torch.arange(merge, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(
                merged_h, merged_w, merge, merge,
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge, merge,
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            n_tokens = coords.shape[0]
            pos_ids[offset: offset + n_tokens] = coords
            offset += n_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def _compute_rotary_pos_emb_static(
        self,
        grid_thw: Tensor,
        total_tokens: int,
        real_total_tokens: int,
        max_grid_size: int,
    ) -> Tensor:
        """Vectorized 2D Vision RoPE for static packed vision buckets."""
        row, col, _, _, valid = self._packed_token_metadata(
            grid_thw, total_tokens, real_total_tokens
        )
        freq_table = self.rot_pos_emb(max_grid_size, device=grid_thw.device)
        pos_ids = torch.stack((row, col), dim=-1).clamp(max=max_grid_size - 1)
        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings * valid.to(embeddings.dtype).unsqueeze(-1)

    # ---------------------------------------------------------------
    # PackedSeqParams for variable-length attention
    # ---------------------------------------------------------------

    @staticmethod
    def _pad_first_dim(tensor: Tensor, target_len: Optional[int]) -> Tensor:
        """Pad a tensor's leading dimension with zeros to ``target_len``."""
        if target_len is None:
            return tensor
        actual_len = tensor.shape[0]
        if actual_len > target_len:
            raise ValueError(
                f"Qwen vision packed tokens ({actual_len}) exceed configured static target "
                f"({target_len}). Increase --vision-max-packed-tokens or use a larger bucket."
            )
        if actual_len == target_len:
            return tensor
        pad_shape = (target_len - actual_len, *tensor.shape[1:])
        return torch.cat((tensor, tensor.new_zeros(pad_shape)), dim=0)

    @staticmethod
    def _pad_cu_seqlens(cu_seqlens: Tensor, max_num_sequences: Optional[int]) -> Tensor:
        """Pad THD cumulative sequence metadata to a fixed number of entries."""
        if max_num_sequences is None:
            return cu_seqlens
        target_entries = int(max_num_sequences) + 1
        if cu_seqlens.numel() > target_entries:
            raise ValueError(
                f"Qwen vision packed sequences ({cu_seqlens.numel() - 1}) exceed configured "
                f"static target ({max_num_sequences}). Increase --vision-max-packed-sequences "
                "or use a larger bucket."
            )
        if cu_seqlens.numel() == target_entries:
            return cu_seqlens
        pad = cu_seqlens[-1:].expand(target_entries - cu_seqlens.numel())
        return torch.cat((cu_seqlens, pad), dim=0)

    @staticmethod
    def _build_packed_seq_params(
        grid_thw: Tensor,
        max_total_tokens: Optional[int] = None,
        max_num_sequences: Optional[int] = None,
        real_total_tokens: Optional[int] = None,
    ) -> PackedSeqParams:
        """Build ``PackedSeqParams`` from grid dimensions.

        Each temporal frame of each image forms a separate sub-sequence
        in the packed THD layout, matching HF's ``cu_seqlens`` computation.

        Args:
            grid_thw: ``[num_images, 3]``.

        Returns:
            ``PackedSeqParams`` for ``TransformerBlock``.
        """
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        if max_total_tokens is not None:
            if real_total_tokens is None:
                real_total_tokens = int(cu_seqlens[-1].item())
            if real_total_tokens > max_total_tokens:
                raise ValueError(
                    f"Qwen vision packed tokens ({real_total_tokens}) exceed configured static "
                    f"target ({max_total_tokens}). Increase --vision-max-packed-tokens or use a "
                    "larger bucket."
                )
            if real_total_tokens < max_total_tokens:
                cu_seqlens = torch.cat(
                    (
                        cu_seqlens,
                        cu_seqlens.new_tensor([int(max_total_tokens)]),
                    ),
                    dim=0,
                )
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(seq_lens.max().item()) if seq_lens.numel() else 0
            cu_seqlens = Qwen35VLVisionEncoder._pad_cu_seqlens(cu_seqlens, max_num_sequences)
            pad_between_seqs = False
        else:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(seq_lens.max().item()) if seq_lens.numel() else 0
            cu_seqlens = Qwen35VLVisionEncoder._pad_cu_seqlens(cu_seqlens, max_num_sequences)
            pad_between_seqs = None

        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens if max_total_tokens is not None else None,
            cu_seqlens_kv_padded=cu_seqlens if max_total_tokens is not None else None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            pad_between_seqs=pad_between_seqs,
        )

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def forward(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
    ) -> Tensor:
        """Encode images / video frames.

        Args:
            pixel_values: ``[total_patches, C * T * pH * pW]``
                pre-extracted flat patches in block-merge order.
            grid_thw: ``[num_images, 3]`` (T, H, W) in patch-grid units.

        Returns:
            ``[total_merged_patches, out_hidden_size]`` visual embeddings.
        """
        real_patch_tokens = pixel_values.shape[0]
        max_packed_tokens = getattr(self.config, "qwen_vision_max_packed_tokens", None)
        max_packed_sequences = getattr(self.config, "qwen_vision_max_packed_sequences", None)
        max_grid_size = getattr(self.config, "qwen_vision_max_grid_size", None)
        static_bucket_flags = {
            "--vision-max-packed-tokens": max_packed_tokens,
            "--vision-max-packed-sequences": max_packed_sequences,
            "--vision-max-grid-size": max_grid_size,
        }
        static_bucket_enabled = any(value is not None for value in static_bucket_flags.values())
        if static_bucket_enabled and not all(value is not None for value in static_bucket_flags.values()):
            missing_flags = ", ".join(
                name for name, value in static_bucket_flags.items() if value is None
            )
            raise ValueError(
                "Qwen vision static token padding requires all static bucket flags: "
                f"{missing_flags}."
            )
        static_metadata = static_bucket_enabled
        if static_metadata:
            expected_patch_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
            if real_patch_tokens != expected_patch_tokens:
                raise ValueError(
                    f"Qwen vision pixel_values tokens ({real_patch_tokens}) do not match "
                    f"image_grid_thw token count ({expected_patch_tokens})."
                )
            actual_grid_size = int(grid_thw[:, 1:].max().item())
            if actual_grid_size > int(max_grid_size):
                raise ValueError(
                    f"Qwen vision grid size ({actual_grid_size}) exceeds configured static "
                    f"target ({max_grid_size}). Increase --vision-max-grid-size or use a "
                    "larger bucket."
                )
        static_grid_thw = (
            self._pad_grid_thw(grid_thw, int(max_packed_sequences))
            if static_metadata and max_packed_sequences is not None
            else grid_thw
        )
        if max_packed_tokens is not None:
            merge_area = self.spatial_merge_size ** 2
            if max_packed_tokens % merge_area != 0:
                raise ValueError(
                    "--vision-max-packed-tokens must be divisible by "
                    f"spatial_merge_size**2 ({merge_area}) so the patch merger remains static."
                )
            pixel_values = self._pad_first_dim(pixel_values, int(max_packed_tokens))

        # 1. Patch embedding (Conv3d)
        hidden_states = self.patch_embed(pixel_values)

        # 2. Learned position embedding (bilinear interpolation)
        if static_metadata:
            pos_embeds = self._fast_pos_embed_interpolate_static(
                static_grid_thw, int(max_packed_tokens), real_patch_tokens
            )
        else:
            pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 3. 2D Vision RoPE
        if static_metadata:
            rot_freqs = self._compute_rotary_pos_emb_static(
                static_grid_thw, int(max_packed_tokens), real_patch_tokens, int(max_grid_size)
            )
        else:
            rot_freqs = self._compute_rotary_pos_emb(grid_thw)
            if max_packed_tokens is not None:
                hidden_states = self._pad_first_dim(hidden_states, int(max_packed_tokens))
                rot_freqs = self._pad_first_dim(rot_freqs, int(max_packed_tokens))
        emb = torch.cat((rot_freqs, rot_freqs), dim=-1)
        rot_freqs_expanded = emb.unsqueeze(1).unsqueeze(1)

        # 4. Transformer blocks with PackedSeqParams
        packed_seq_params = self._build_packed_seq_params(
            static_grid_thw,
            max_total_tokens=max_packed_tokens,
            max_num_sequences=max_packed_sequences,
            real_total_tokens=real_patch_tokens,
        )
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=rot_freqs_expanded,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = hidden_states.squeeze(1)

        # 5. Patch merger
        merged = self.merger(hidden_states)
        if max_packed_tokens is not None:
            merged = merged[: real_patch_tokens // (self.spatial_merge_size ** 2)]
        return merged
