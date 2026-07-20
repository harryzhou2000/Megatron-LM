# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for Qwen3.5-VL vision static THD staging helpers."""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.qwen35_vl.vision_encoder import (
    Qwen35VLVisionEncoder,
    Qwen35VLVisionRotaryEmbedding,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _make_tiny_vision_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=16,
        ffn_hidden_size=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-6,
        normalization="LayerNorm",
        gated_linear_unit=False,
        activation_func=lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        bias_activation_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_rope_fusion=True,
        bf16=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )


def _make_metadata_only_encoder(hidden_size=8, max_num_positions=64):
    encoder = object.__new__(Qwen35VLVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.spatial_merge_size = 2
    encoder.num_grid_per_side = int(max_num_positions**0.5)
    encoder.pos_embed = torch.nn.Embedding(max_num_positions, hidden_size)
    with torch.no_grad():
        values = torch.arange(max_num_positions * hidden_size, dtype=torch.float32)
        encoder.pos_embed.weight.copy_(values.view(max_num_positions, hidden_size) / 1000.0)
    encoder.rot_pos_emb = Qwen35VLVisionRotaryEmbedding(dim=4)
    return encoder


def test_static_vision_thd_pads_tokens_and_cu_seqlens():
    grid_thw = torch.tensor([[1, 14, 14], [1, 28, 28]], dtype=torch.long)
    real_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
    target_tokens = 1024

    packed_seq_params = Qwen35VLVisionEncoder._build_packed_seq_params(
        grid_thw,
        max_total_tokens=target_tokens,
        max_num_sequences=3,
        real_total_tokens=real_tokens,
    )

    assert packed_seq_params.qkv_format == "thd"
    assert packed_seq_params.max_seqlen_q == target_tokens
    assert packed_seq_params.max_seqlen_kv == target_tokens
    assert packed_seq_params.pad_between_seqs is False
    assert torch.equal(
        packed_seq_params.cu_seqlens_q,
        torch.tensor([0, 196, 980, 1024], dtype=torch.int32),
    )
    assert torch.equal(packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_q_padded)
    assert torch.equal(packed_seq_params.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv_padded)

    x = torch.arange(real_tokens * 2, dtype=torch.float32).view(real_tokens, 2)
    padded = Qwen35VLVisionEncoder._pad_first_dim(x, target_tokens)
    assert padded.shape == (target_tokens, 2)
    assert torch.equal(padded[:real_tokens], x)
    assert torch.count_nonzero(padded[real_tokens:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_static_vision_thd_metadata_stays_on_gpu_with_fixed_shapes():
    grid_thw = torch.tensor([[1, 14, 14], [1, 28, 28]], dtype=torch.long, device="cuda")
    real_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
    target_tokens = 1024
    max_num_sequences = 3

    packed_seq_params = Qwen35VLVisionEncoder._build_packed_seq_params(
        grid_thw,
        max_total_tokens=target_tokens,
        max_num_sequences=max_num_sequences,
        real_total_tokens=real_tokens,
    )

    assert packed_seq_params.cu_seqlens_q.device.type == "cuda"
    assert packed_seq_params.cu_seqlens_kv.device.type == "cuda"
    assert packed_seq_params.cu_seqlens_q_padded.device.type == "cuda"
    assert packed_seq_params.cu_seqlens_kv_padded.device.type == "cuda"
    assert packed_seq_params.cu_seqlens_q.shape == (max_num_sequences + 1,)
    assert packed_seq_params.cu_seqlens_kv.shape == (max_num_sequences + 1,)
    assert packed_seq_params.max_seqlen_q == target_tokens
    assert packed_seq_params.max_seqlen_kv == target_tokens
    assert packed_seq_params.pad_between_seqs is False

    x = torch.ones(real_tokens, 8, device="cuda", dtype=torch.bfloat16)
    padded = Qwen35VLVisionEncoder._pad_first_dim(x, target_tokens)
    assert padded.device.type == "cuda"
    assert padded.shape == (target_tokens, 8)
    assert torch.count_nonzero(padded[real_tokens:]) == 0


def test_static_vision_thd_rejects_small_buckets():
    grid_thw = torch.tensor([[1, 14, 14], [1, 28, 28]], dtype=torch.long)

    with pytest.raises(ValueError, match="packed tokens"):
        Qwen35VLVisionEncoder._build_packed_seq_params(
            grid_thw,
            max_total_tokens=512,
            max_num_sequences=3,
            real_total_tokens=980,
        )

    with pytest.raises(ValueError, match="packed sequences"):
        Qwen35VLVisionEncoder._build_packed_seq_params(
            grid_thw,
            max_total_tokens=1024,
            max_num_sequences=1,
            real_total_tokens=980,
        )


def test_static_vision_metadata_matches_dynamic_prefix():
    encoder = _make_metadata_only_encoder()
    grid_thw = torch.tensor([[1, 4, 4], [1, 6, 6]], dtype=torch.long)
    real_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
    target_tokens = 64
    static_grid_thw = encoder._pad_grid_thw(grid_thw, max_num_rows=3)
    assert static_grid_thw.shape == (3, 3)
    assert torch.equal(static_grid_thw[:2], grid_thw)
    assert torch.count_nonzero(static_grid_thw[2:]) == 0

    dynamic_pos = encoder._fast_pos_embed_interpolate(grid_thw)
    static_pos = encoder._fast_pos_embed_interpolate_static(
        static_grid_thw, target_tokens, real_tokens
    )
    torch.testing.assert_close(static_pos[:real_tokens], dynamic_pos)
    assert torch.count_nonzero(static_pos[real_tokens:]) == 0

    dynamic_rope = encoder._compute_rotary_pos_emb(grid_thw)
    static_rope = encoder._compute_rotary_pos_emb_static(
        static_grid_thw, target_tokens, real_tokens, max_grid_size=6
    )
    torch.testing.assert_close(static_rope[:real_tokens], dynamic_rope)
    assert torch.count_nonzero(static_rope[real_tokens:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_static_vision_encoder_matches_dynamic_forward_backward():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(2026)
    try:
        torch.manual_seed(2026)
        device = torch.device("cuda", torch.cuda.current_device())
        grid_thw = torch.tensor([[1, 4, 4], [1, 6, 6]], dtype=torch.long, device=device)
        real_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        pixel_dim = 3 * 1 * 2 * 2
        target_tokens = 64

        dynamic_config = _make_tiny_vision_config()
        static_config = _make_tiny_vision_config()
        static_config.qwen_vision_max_packed_tokens = target_tokens
        static_config.qwen_vision_max_packed_sequences = 3
        static_config.qwen_vision_max_grid_size = 6

        dynamic_model = Qwen35VLVisionEncoder(
            config=dynamic_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).to(device).eval()
        static_model = Qwen35VLVisionEncoder(
            config=static_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).to(device).eval()
        static_model.load_state_dict(dynamic_model.state_dict())

        dynamic_pixels = torch.randn(real_tokens, pixel_dim, device=device, requires_grad=True)
        static_pixels = dynamic_pixels.detach().clone().requires_grad_()

        dynamic_out = dynamic_model(dynamic_pixels, grid_thw)
        static_out = static_model(static_pixels, grid_thw)
        assert dynamic_out.shape == static_out.shape
        output_diff = (static_out - dynamic_out).abs()
        print(
            "static_vs_dynamic_output "
            f"max={output_diff.max().item():.6e} mean={output_diff.mean().item():.6e} "
            f"bitwise={torch.equal(static_out, dynamic_out)}",
            flush=True,
        )
        torch.testing.assert_close(static_out, dynamic_out, rtol=2e-4, atol=2e-4)

        grad_out = torch.randn_like(dynamic_out)
        dynamic_out.backward(grad_out)
        static_out.backward(grad_out)

        pixel_grad_diff = (static_pixels.grad - dynamic_pixels.grad).abs()
        print(
            "static_vs_dynamic_pixel_grad "
            f"max={pixel_grad_diff.max().item():.6e} mean={pixel_grad_diff.mean().item():.6e} "
            f"bitwise={torch.equal(static_pixels.grad, dynamic_pixels.grad)}",
            flush=True,
        )
        torch.testing.assert_close(static_pixels.grad, dynamic_pixels.grad, rtol=2e-4, atol=2e-4)

        dynamic_grads = dict(dynamic_model.named_parameters())
        worst_param = ("", 0.0, 0.0, True)
        for name, static_param in static_model.named_parameters():
            dynamic_grad = dynamic_grads[name].grad
            static_grad = static_param.grad
            if dynamic_grad is None and static_grad is None:
                continue
            assert dynamic_grad is not None, name
            assert static_grad is not None, name
            grad_diff = (static_grad - dynamic_grad).abs()
            max_diff = grad_diff.max().item() if grad_diff.numel() else 0.0
            mean_diff = grad_diff.mean().item() if grad_diff.numel() else 0.0
            bitwise = torch.equal(static_grad, dynamic_grad)
            if max_diff > worst_param[1]:
                worst_param = (name, max_diff, mean_diff, bitwise)
            torch.testing.assert_close(static_grad, dynamic_grad, rtol=2e-4, atol=2e-4, msg=name)
        print(
            "static_vs_dynamic_worst_param_grad "
            f"name={worst_param[0]} max={worst_param[1]:.6e} "
            f"mean={worst_param[2]:.6e} bitwise={worst_param[3]}",
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(
    reason=(
        "Full encoder capture still builds PackedSeqParams inside forward; "
        "only the pre-staged transformer layer is graph-captured today."
    ),
    strict=True,
)
def test_static_vision_encoder_cuda_graph_forward_backward():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(2027)
    try:
        torch.manual_seed(2027)
        device = torch.device("cuda", torch.cuda.current_device())
        grid_thw = torch.tensor([[1, 4, 4], [1, 6, 6]], dtype=torch.long, device=device)
        real_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        pixel_dim = 3 * 1 * 2 * 2
        target_tokens = 64

        eager_config = _make_tiny_vision_config()
        graph_config = _make_tiny_vision_config()
        for config in (eager_config, graph_config):
            config.qwen_vision_max_packed_tokens = target_tokens
            config.qwen_vision_max_packed_sequences = 3
            config.qwen_vision_max_grid_size = 6

        eager_model = Qwen35VLVisionEncoder(
            config=eager_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).to(device).eval()
        graph_model = Qwen35VLVisionEncoder(
            config=graph_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).to(device).eval()
        graph_model.load_state_dict(eager_model.state_dict())

        eager_pixels = torch.randn(real_tokens, pixel_dim, device=device, requires_grad=True)
        graph_pixels = eager_pixels.detach().clone().requires_grad_()

        eager_out = eager_model(eager_pixels, grid_thw)
        torch.cuda.synchronize()
        graphed_model = torch.cuda.make_graphed_callables(
            graph_model, (graph_pixels, grid_thw), num_warmup_iters=3
        )
        graph_out = graphed_model(graph_pixels, grid_thw)

        output_diff = (graph_out - eager_out).abs()
        print(
            "cuda_graph_static_output "
            f"max={output_diff.max().item():.6e} mean={output_diff.mean().item():.6e} "
            f"bitwise={torch.equal(graph_out, eager_out)}",
            flush=True,
        )
        torch.testing.assert_close(graph_out, eager_out, rtol=2e-4, atol=2e-4)

        grad_out = torch.randn_like(eager_out)
        eager_out.backward(grad_out)
        graph_out.backward(grad_out)

        pixel_grad_diff = (graph_pixels.grad - eager_pixels.grad).abs()
        print(
            "cuda_graph_static_pixel_grad "
            f"max={pixel_grad_diff.max().item():.6e} mean={pixel_grad_diff.mean().item():.6e} "
            f"bitwise={torch.equal(graph_pixels.grad, eager_pixels.grad)}",
            flush=True,
        )
        torch.testing.assert_close(graph_pixels.grad, eager_pixels.grad, rtol=2e-4, atol=2e-4)

        eager_params = dict(eager_model.named_parameters())
        worst_param = ("", 0.0, 0.0, True)
        for name, graph_param in graph_model.named_parameters():
            eager_grad = eager_params[name].grad
            graph_grad = graph_param.grad
            if eager_grad is None and graph_grad is None:
                continue
            assert eager_grad is not None, name
            assert graph_grad is not None, name
            grad_diff = (graph_grad - eager_grad).abs()
            max_diff = grad_diff.max().item() if grad_diff.numel() else 0.0
            mean_diff = grad_diff.mean().item() if grad_diff.numel() else 0.0
            bitwise = torch.equal(graph_grad, eager_grad)
            if max_diff > worst_param[1]:
                worst_param = (name, max_diff, mean_diff, bitwise)
            torch.testing.assert_close(graph_grad, eager_grad, rtol=2e-4, atol=2e-4, msg=name)
        print(
            "cuda_graph_static_worst_param_grad "
            f"name={worst_param[0]} max={worst_param[1]:.6e} "
            f"mean={worst_param[2]:.6e} bitwise={worst_param[3]}",
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()


class _VisionDecoderGraphWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, max_seqlen: int):
        super().__init__()
        self.decoder = decoder
        self.max_seqlen = max_seqlen

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=self.max_seqlen,
            max_seqlen_kv=self.max_seqlen,
            pad_between_seqs=False,
        )
        return self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_static_vision_transformer_layer_cuda_graph_forward_backward():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(
        2028, te_rng_tracker=True, use_cudagraphable_rng=True, force_reset_rng=True
    )
    try:
        torch.manual_seed(2028)
        device = torch.device("cuda", torch.cuda.current_device())
        target_tokens = 256
        hidden_size = 256
        head_dim = 64
        cu_seqlens = torch.tensor([0, 64, 192, 256], dtype=torch.int32, device=device)

        eager_config = _make_tiny_vision_config()
        graph_config = _make_tiny_vision_config()
        for config in (eager_config, graph_config):
            config.hidden_size = hidden_size
            config.num_attention_heads = 4
            config.kv_channels = head_dim
            config.ffn_hidden_size = 512
            config.bf16 = True
            config.params_dtype = torch.bfloat16
            config.pipeline_dtype = torch.bfloat16
        eager_encoder = Qwen35VLVisionEncoder(
            config=eager_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=64,
            max_num_positions=64,
        ).to(device).bfloat16().train()
        graph_encoder = Qwen35VLVisionEncoder(
            config=graph_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=64,
            max_num_positions=64,
        ).to(device).bfloat16().train()
        graph_encoder.load_state_dict(eager_encoder.state_dict())

        eager_layer = _VisionDecoderGraphWrapper(eager_encoder.decoder, max_seqlen=target_tokens)
        graph_layer = _VisionDecoderGraphWrapper(graph_encoder.decoder, max_seqlen=target_tokens)

        eager_hidden = torch.randn(
            target_tokens, 1, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        graph_hidden = eager_hidden.detach().clone().requires_grad_()
        rotary_pos_emb = torch.randn(target_tokens, 1, 1, head_dim, device=device)

        eager_out = eager_layer(eager_hidden, rotary_pos_emb, cu_seqlens)
        torch.cuda.synchronize()
        from transformer_engine.pytorch.graph import make_graphed_callables

        graphed_layer = make_graphed_callables(
            graph_layer, (graph_hidden, rotary_pos_emb, cu_seqlens), num_warmup_iters=3
        )
        graph_out = graphed_layer(graph_hidden, rotary_pos_emb, cu_seqlens)

        output_diff = (graph_out - eager_out).abs()
        print(
            "cuda_graph_layer_output "
            f"max={output_diff.max().item():.6e} mean={output_diff.mean().item():.6e} "
            f"bitwise={torch.equal(graph_out, eager_out)}",
            flush=True,
        )
        torch.testing.assert_close(graph_out, eager_out, rtol=5e-2, atol=5e-2)

        grad_out = torch.randn_like(eager_out)
        eager_out.backward(grad_out)
        graph_out.backward(grad_out)

        hidden_grad_diff = (graph_hidden.grad - eager_hidden.grad).abs()
        print(
            "cuda_graph_layer_hidden_grad "
            f"max={hidden_grad_diff.max().item():.6e} mean={hidden_grad_diff.mean().item():.6e} "
            f"bitwise={torch.equal(graph_hidden.grad, eager_hidden.grad)}",
            flush=True,
        )
        torch.testing.assert_close(graph_hidden.grad, eager_hidden.grad, rtol=5e-2, atol=5e-2)

        eager_params = dict(eager_layer.named_parameters())
        worst_param = ("", 0.0, 0.0, True)
        for name, graph_param in graph_layer.named_parameters():
            eager_grad = eager_params[name].grad
            graph_grad = graph_param.grad
            if eager_grad is None and graph_grad is None:
                continue
            assert eager_grad is not None, name
            assert graph_grad is not None, name
            grad_diff = (graph_grad - eager_grad).abs()
            max_diff = grad_diff.max().item() if grad_diff.numel() else 0.0
            mean_diff = grad_diff.mean().item() if grad_diff.numel() else 0.0
            bitwise = torch.equal(graph_grad, eager_grad)
            if max_diff > worst_param[1]:
                worst_param = (name, max_diff, mean_diff, bitwise)
            torch.testing.assert_close(graph_grad, eager_grad, rtol=5e-2, atol=5e-2, msg=name)
        print(
            "cuda_graph_layer_worst_param_grad "
            f"name={worst_param[0]} max={worst_param[1]:.6e} "
            f"mean={worst_param[2]:.6e} bitwise={worst_param[3]}",
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()
