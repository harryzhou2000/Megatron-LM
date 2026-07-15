# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for Qwen3.5-VL vision RoPE fusion."""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.qwen35_vl.specs import _apply_rope_fp32
from megatron.core.models.common.embeddings import rope_utils as rope_utils_module
from megatron.core.transformer.transformer_config import TransformerConfig


class FakeCPGroup:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    rope_utils_module.fused_apply_rotary_pos_emb_thd is None, reason="Fused THD RoPE not available."
)
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_qwen_vision_fused_thd_rope_matches_unfused_exact_freqs(cp_size):
    """Qwen vision uses exact per-token 2D frequencies in packed THD order."""
    torch.manual_seed(456)
    seq_lens = torch.tensor([16, 32, 48], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seq_lens.cumsum(dim=0), (1, 0), value=0)
    max_seqlen = int(seq_lens.max().item())
    total_tokens = int(cu_seqlens[-1].item()) // cp_size
    num_heads = 4
    head_dim = 32
    freqs = torch.randn(
        int(cu_seqlens[-1].item()), 1, 1, head_dim, device="cuda", dtype=torch.float32
    )
    config_unfused = TransformerConfig(num_layers=1, num_attention_heads=num_heads)
    config_fused = TransformerConfig(
        num_layers=1, num_attention_heads=num_heads, apply_rope_fusion=True
    )

    for cp_rank in range(cp_size):
        cp_group = FakeCPGroup(size=cp_size, rank=cp_rank)
        unfused_input = torch.randn(
            total_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        fused_input = unfused_input.detach().clone().requires_grad_()

        unfused_output = _apply_rope_fp32(
            unfused_input,
            freqs,
            config_unfused,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            max_seqlen=max_seqlen,
        )
        fused_output = _apply_rope_fp32(
            fused_input,
            freqs,
            config_fused,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            max_seqlen=max_seqlen,
        )

        torch.testing.assert_close(fused_output, unfused_output, rtol=5e-3, atol=5e-3)
