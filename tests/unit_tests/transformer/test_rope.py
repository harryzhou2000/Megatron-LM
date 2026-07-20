# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings import rope_utils as rope_utils_module
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False

from tests.unit_tests.test_utilities import Utils


class FakeCPGroup:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class TestMultimodalRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = MultimodalRotaryEmbedding(self.kv_channels, self.rotary_percent)

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, MultimodalRotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(torch.Tensor(3, 1, 64), mrope_section=[16, 24, 24])
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope_cpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=True
        )
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        del self.rope_cpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_cpu_init, RotaryEmbedding)
        assert self.rope_cpu_init.inv_freq.device.type == 'cpu'
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_forward(self):
        output = self.rope_cpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestQKVRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.seq_len = 64
        self.num_heads = 1
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )
        self.transformer_config = TransformerConfig(
            num_attention_heads=self.num_heads, num_layers=1, apply_rope_fusion=True
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAVE_FUSED_QKV_ROPE, reason="Fused QKV RoPE not available.")
    def test_gpu_forward(self):
        pos_embed = self.rope_gpu_init(self.seq_len)
        assert pos_embed.shape[0] == self.seq_len
        assert pos_embed.shape[1] == 1
        assert pos_embed.shape[2] == 1
        assert pos_embed.shape[3] == self.kv_channels
        assert pos_embed.dtype == torch.float32
        assert pos_embed.device.type == 'cuda'

        qkv_split_arg_list = [self.kv_channels * 4, self.kv_channels, self.kv_channels]
        # Create input tensors
        qkv = torch.randn(self.seq_len, 1, self.num_heads, self.kv_channels * 6, device="cuda")
        (query_in, key_in, value_in) = torch.split(qkv, qkv_split_arg_list, dim=3)

        query_in = query_in.reshape(query_in.shape[0], query_in.shape[1], -1, self.kv_channels)
        q_out_ref = apply_rotary_pos_emb(query_in, pos_embed, self.transformer_config)
        k_out_ref = apply_rotary_pos_emb(key_in, pos_embed, self.transformer_config)
        q_out, k_out, _ = apply_fused_qkv_rotary_pos_emb(
            qkv, pos_embed, pos_embed, qkv_split_arg_list
        )

        assert (
            q_out_ref.numel() == q_out.numel()
        ), f"Output sizes do not match for Q: {q_out.shape} != {q_out_ref.shape}"
        assert (
            k_out_ref.numel() == k_out.numel()
        ), f"Output sizes do not match for K: {k_out.shape} != {k_out_ref.shape}"
        assert torch.allclose(q_out_ref, q_out), f"Outputs do not match for Q"
        assert torch.allclose(k_out_ref, k_out), f"Outputs do not match for K"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    rope_utils_module.fused_apply_rotary_pos_emb_thd is None, reason="Fused THD RoPE not available."
)
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_fused_thd_rope_with_start_positions_matches_unfused(cp_size):
    """Exact packed frequencies require per-sequence offsets in fused THD RoPE."""
    torch.manual_seed(123)
    seq_lens = torch.tensor([16, 32, 48], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seq_lens.cumsum(dim=0), (1, 0), value=0)
    max_seqlen = int(seq_lens.max().item())
    total_tokens = int(cu_seqlens[-1].item()) // cp_size
    num_heads = 4
    head_dim = 32
    freqs = torch.randn(
        int(cu_seqlens[-1].item()), 1, 1, head_dim, device="cuda", dtype=torch.float32
    )
    grad = torch.randn(total_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)

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

        unfused_output = apply_rotary_pos_emb(
            unfused_input.float(),
            freqs,
            config_unfused,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            max_seqlen=max_seqlen,
        ).to(torch.bfloat16)
        fused_output = apply_rotary_pos_emb(
            fused_input.float(),
            freqs,
            config_fused,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            max_seqlen=max_seqlen,
            start_positions=cu_seqlens[:-1],
        ).to(torch.bfloat16)

        unfused_grad = torch.autograd.grad(unfused_output, unfused_input, grad, retain_graph=True)[
            0
        ]
        fused_grad = torch.autograd.grad(fused_output, fused_input, grad)[0]

        torch.testing.assert_close(fused_output, unfused_output, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(fused_grad, unfused_grad, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    rope_utils_module.fused_apply_rotary_pos_emb_thd is None, reason="Fused THD RoPE not available."
)
@pytest.mark.parametrize(
    ("seq_lens", "match"),
    [
        ([15], "divisible by cp_size"),
        ([6], "CP-local sequence length"),
    ],
)
def test_fused_thd_rope_with_start_positions_rejects_unsupported_cp_layout(seq_lens, match):
    """The CP fallback assumes divisible sequence lengths and even local zigzag chunks."""
    cu_seqlens = torch.nn.functional.pad(
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda").cumsum(dim=0), (1, 0), value=0
    )
    num_heads = 2
    head_dim = 16
    cp_group = FakeCPGroup(size=2, rank=0)
    total_tokens = int(cu_seqlens[-1].item()) // cp_group.size()
    t = torch.randn(total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float32)
    freqs = torch.randn(int(cu_seqlens[-1].item()), 1, 1, head_dim, device="cuda")
    config = TransformerConfig(num_layers=1, num_attention_heads=num_heads, apply_rope_fusion=True)

    with pytest.raises((AssertionError, RuntimeError), match=match):
        apply_rotary_pos_emb(
            t,
            freqs,
            config,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            max_seqlen=max(seq_lens),
            start_positions=cu_seqlens[:-1],
        )
