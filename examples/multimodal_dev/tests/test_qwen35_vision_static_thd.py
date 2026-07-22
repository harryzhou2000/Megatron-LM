# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for Qwen3.5-VL vision static THD staging helpers."""

import os
import sys
from types import SimpleNamespace

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
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    HAVE_TE_GRAPHS,
    CudaGraphManager,
    VisionTECudaGraphHelper,
    _CudagraphGlobalRecord,
    create_cudagraphs,
)
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
    encoder.config = SimpleNamespace(mrope_section=None)
    encoder.spatial_merge_size = 2
    encoder.num_grid_per_side = int(max_num_positions**0.5)
    encoder.pos_embed = torch.nn.Embedding(max_num_positions, hidden_size)
    with torch.no_grad():
        values = torch.arange(max_num_positions * hidden_size, dtype=torch.float32)
        encoder.pos_embed.weight.copy_(values.view(max_num_positions, hidden_size) / 1000.0)
    encoder.rot_pos_emb = Qwen35VLVisionRotaryEmbedding(dim=4)
    return encoder


def _reset_mcore_cuda_graph_state():
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    _CudagraphGlobalRecord.cudagraph_inference_record = []
    CudaGraphManager.global_mempool = None


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
    assert packed_seq_params.max_seqlen_q == 784
    assert packed_seq_params.max_seqlen_kv == 784
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
    assert packed_seq_params.max_seqlen_q == 784
    assert packed_seq_params.max_seqlen_kv == 784
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


def test_static_vision_encoder_rejects_partial_static_bucket_config():
    encoder = _make_metadata_only_encoder()
    pixel_values = torch.zeros(16, 12)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    partial_configs = [
        dict(
            qwen_vision_max_packed_tokens=64,
            qwen_vision_max_packed_sequences=None,
            qwen_vision_max_grid_size=4,
        ),
        dict(
            qwen_vision_max_packed_tokens=64,
            qwen_vision_max_packed_sequences=1,
            qwen_vision_max_grid_size=None,
        ),
        dict(
            qwen_vision_max_packed_tokens=None,
            qwen_vision_max_packed_sequences=1,
            qwen_vision_max_grid_size=4,
        ),
    ]

    for config_kwargs in partial_configs:
        encoder.config = SimpleNamespace(**config_kwargs)
        with pytest.raises(ValueError, match="static bucket flags"):
            encoder(pixel_values, grid_thw)


def test_static_vision_encoder_rejects_grid_token_mismatch():
    encoder = _make_metadata_only_encoder()
    encoder.config = SimpleNamespace(
        qwen_vision_max_packed_tokens=16,
        qwen_vision_max_packed_sequences=1,
        qwen_vision_max_grid_size=4,
    )
    pixel_values = torch.zeros(15, 12)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="image_grid_thw token count"):
        encoder(pixel_values, grid_thw)


def test_static_vision_encoder_rejects_small_grid_bucket():
    encoder = _make_metadata_only_encoder()
    encoder.config = SimpleNamespace(
        qwen_vision_max_packed_tokens=16,
        qwen_vision_max_packed_sequences=1,
        qwen_vision_max_grid_size=3,
    )
    pixel_values = torch.zeros(16, 12)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="vision-max-grid-size"):
        encoder(pixel_values, grid_thw)


def test_static_vision_thd_uses_true_max_segment_length():
    grid_thw = torch.tensor([[2, 14, 14]], dtype=torch.long)

    packed_seq_params = Qwen35VLVisionEncoder._build_packed_seq_params(
        grid_thw,
        max_total_tokens=392,
        max_num_sequences=3,
        real_total_tokens=392,
    )

    assert torch.equal(
        packed_seq_params.cu_seqlens_q, torch.tensor([0, 196, 392, 392], dtype=torch.int32)
    )
    assert packed_seq_params.max_seqlen_q == 196
    assert packed_seq_params.max_seqlen_kv == 196


def test_static_vision_thd_dummy_tail_bounds_max_seqlen():
    grid_thw = torch.tensor([[1, 14, 14]], dtype=torch.long)

    packed_seq_params = Qwen35VLVisionEncoder._build_packed_seq_params(
        grid_thw,
        max_total_tokens=392,
        max_num_sequences=2,
        real_total_tokens=196,
    )

    assert torch.equal(
        packed_seq_params.cu_seqlens_q, torch.tensor([0, 196, 392], dtype=torch.int32)
    )
    assert packed_seq_params.max_seqlen_q == 196
    assert packed_seq_params.max_seqlen_kv == 196


def test_vision_layer_cuda_graph_rejects_unsupported_args(monkeypatch):
    from examples.multimodal_dev.pretrain_multimodal import _prepare_vision_cuda_graph_args

    base_kwargs = dict(
        vision_layer_cuda_graph=True,
        vision_te_cuda_graph=False,
        vision_max_packed_tokens=392,
        vision_max_packed_sequences=3,
        vision_max_grid_size=14,
        cuda_graph_impl="none",
        te_rng_tracker=False,
        use_megatron_fsdp=False,
        recompute_vision=False,
    )

    with pytest.raises(ValueError, match="Megatron-FSDP"):
        _prepare_vision_cuda_graph_args(SimpleNamespace(**{**base_kwargs, "use_megatron_fsdp": True}))

    with pytest.raises(ValueError, match="recompute-vision"):
        _prepare_vision_cuda_graph_args(SimpleNamespace(**{**base_kwargs, "recompute_vision": True}))

    with pytest.raises(ValueError, match="vision-max-grid-size"):
        _prepare_vision_cuda_graph_args(
            SimpleNamespace(**{**base_kwargs, "vision_max_grid_size": None})
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _prepare_vision_cuda_graph_args(SimpleNamespace(**{**base_kwargs, "vision_te_cuda_graph": True}))

    args = SimpleNamespace(**base_kwargs)
    _prepare_vision_cuda_graph_args(args)
    assert args.te_rng_tracker is True
    assert args._multimodal_language_cuda_graph_impl == "none"

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)
    te_args = SimpleNamespace(
        **{**base_kwargs, "vision_layer_cuda_graph": False, "vision_te_cuda_graph": True}
    )
    _prepare_vision_cuda_graph_args(te_args)
    assert te_args.te_rng_tracker is True
    assert not hasattr(te_args, "_multimodal_language_cuda_graph_impl")
    assert os.environ["NCCL_GRAPH_REGISTER"] == "0"


def test_mimo_rejects_vision_te_cuda_graph():
    from examples.multimodal_dev.pretrain_multimodal_mimo import _prepare_vision_cuda_graph_args

    args = SimpleNamespace(
        vision_layer_cuda_graph=False,
        vision_te_cuda_graph=True,
        vision_max_packed_tokens=392,
        vision_max_packed_sequences=3,
        vision_max_grid_size=14,
        use_megatron_fsdp=False,
        recompute_vision=False,
    )

    with pytest.raises(ValueError, match="pretrain_multimodal_mimo"):
        _prepare_vision_cuda_graph_args(args)


def test_unfused_qwen_vision_thd_rope_uses_start_positions():
    from examples.multimodal_dev.models.base import _NO_CP_GROUP
    from examples.multimodal_dev.models.qwen35_vl.specs import _apply_rope_fp32
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

    config = _make_tiny_vision_config()
    config.apply_rope_fusion = False
    t = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    freqs = torch.randn(5, 1, 1, 8, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    out = _apply_rope_fp32(
        t,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=_NO_CP_GROUP,
        max_seqlen=5,
    )
    expected = _apply_rotary_pos_emb_bshd(t.float().unsqueeze(1), freqs).squeeze(1).to(t.dtype)

    torch.testing.assert_close(out, expected)


class _VisionOnlyWrapper(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def zero_grad_buffer(self):
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE_GRAPHS, reason="TE make_graphed_callables not available")
def test_qwen_vision_te_cuda_graph_static_inputs_include_thd_rope():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(2029)
    try:
        target_tokens = 64
        max_grid_size = 6
        max_sequences = 3
        config = _make_tiny_vision_config()
        config.bf16 = True
        config.params_dtype = torch.bfloat16
        config.pipeline_dtype = torch.bfloat16
        config.cuda_graph_impl = "transformer_engine"
        config.cuda_graph_modules = []
        config.sequence_packing_scheduler = "vision_static"
        config.qwen_vision_max_packed_tokens = target_tokens
        config.qwen_vision_max_packed_sequences = max_sequences
        config.qwen_vision_max_grid_size = max_grid_size
        config.max_vision_cuda_graph_seq_length = target_tokens
        config.max_seqlen_per_dp_cp_rank = max_grid_size**2
        config.thd_max_packed_sequences = max_sequences
        config.cuda_graph_static_total_tokens = target_tokens
        config.cuda_graph_static_max_seqlen = max_grid_size**2
        encoder = Qwen35VLVisionEncoder(
            config=config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).cuda().bfloat16().train()
        helper = VisionTECudaGraphHelper(
            model=[_VisionOnlyWrapper(encoder)],
            vision_config=config,
            vision_seq_length=target_tokens,
            micro_batch_size=2,
            num_microbatches=2,
        )

        sample_args, sample_kwargs = helper._get_sample_arguments(order=[1, -1])

        assert len(sample_args) == config.num_layers * 2
        assert len(sample_kwargs) == config.num_layers * 2
        hidden_states = sample_args[0][0]
        assert hidden_states.shape == (target_tokens, 1, config.hidden_size)
        assert hidden_states.dtype == torch.bfloat16
        kwargs = sample_kwargs[0]
        assert kwargs["rotary_pos_emb"].shape == (target_tokens, 1, 1, config.kv_channels)
        assert kwargs["rotary_pos_emb"].dtype == torch.float32
        assert kwargs["cu_seqlens_q"].shape == (max_sequences + 1,)
        assert torch.equal(
            kwargs["cu_seqlens_q"],
            torch.tensor([0, 36, target_tokens, target_tokens], dtype=torch.int32, device="cuda"),
        )
        assert kwargs["cu_seqlens_q_padded"].shape == (max_sequences + 1,)
        assert "padding_mask" not in kwargs
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE_GRAPHS, reason="TE make_graphed_callables not available")
def test_qwen_vision_te_cuda_graph_create_delete():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(
        2030, te_rng_tracker=True, use_cudagraphable_rng=True, force_reset_rng=True
    )
    try:
        target_tokens = 64
        max_grid_size = 6
        max_sequences = 3
        config = _make_tiny_vision_config()
        config.bf16 = True
        config.params_dtype = torch.bfloat16
        config.pipeline_dtype = torch.bfloat16
        config.cuda_graph_impl = "transformer_engine"
        config.cuda_graph_modules = []
        config.sequence_packing_scheduler = "vision_static"
        config.qwen_vision_max_packed_tokens = target_tokens
        config.qwen_vision_max_packed_sequences = max_sequences
        config.qwen_vision_max_grid_size = max_grid_size
        config.max_vision_cuda_graph_seq_length = target_tokens
        config.max_seqlen_per_dp_cp_rank = max_grid_size**2
        config.thd_max_packed_sequences = max_sequences
        config.cuda_graph_static_total_tokens = target_tokens
        config.cuda_graph_static_max_seqlen = max_grid_size**2
        encoder = Qwen35VLVisionEncoder(
            config=config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).cuda().bfloat16().train()
        helper = VisionTECudaGraphHelper(
            model=[_VisionOnlyWrapper(encoder)],
            vision_config=config,
            vision_seq_length=target_tokens,
            micro_batch_size=2,
            num_microbatches=1,
        )
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=2,
            micro_batch_size=2,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        helper.create_cudagraphs()
        assert helper.graphs_created()
        for layer in helper.callables:
            assert len(layer.cuda_graphs) == 1
        helper.delete_cuda_graphs()
        assert not helper.graphs_created()
    finally:
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE_GRAPHS, reason="TE make_graphed_callables not available")
def test_qwen_vision_te_cuda_graph_replay_matches_eager():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(
        2031, te_rng_tracker=True, use_cudagraphable_rng=True, force_reset_rng=True
    )
    try:
        torch.manual_seed(2031)
        device = torch.device("cuda", torch.cuda.current_device())
        target_tokens = 64
        max_grid_size = 6
        max_sequences = 3

        eager_config = _make_tiny_vision_config()
        graph_config = _make_tiny_vision_config()
        for config in (eager_config, graph_config):
            config.bf16 = True
            config.params_dtype = torch.bfloat16
            config.pipeline_dtype = torch.bfloat16
            config.sequence_packing_scheduler = "vision_static"
            config.qwen_vision_max_packed_tokens = target_tokens
            config.qwen_vision_max_packed_sequences = max_sequences
            config.qwen_vision_max_grid_size = max_grid_size
            config.max_vision_cuda_graph_seq_length = target_tokens
            config.max_seqlen_per_dp_cp_rank = max_grid_size**2
            config.thd_max_packed_sequences = max_sequences
            config.cuda_graph_static_total_tokens = target_tokens
            config.cuda_graph_static_max_seqlen = max_grid_size**2
        graph_config.cuda_graph_impl = "transformer_engine"
        graph_config.cuda_graph_modules = []

        eager_encoder = Qwen35VLVisionEncoder(
            config=eager_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).to(device).bfloat16().train()
        graph_encoder = Qwen35VLVisionEncoder(
            config=graph_config,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            max_num_positions=64,
        ).to(device).bfloat16().train()
        graph_encoder.load_state_dict(eager_encoder.state_dict())

        helper = VisionTECudaGraphHelper(
            model=[_VisionOnlyWrapper(graph_encoder)],
            vision_config=graph_config,
            vision_seq_length=target_tokens,
            micro_batch_size=2,
            num_microbatches=1,
        )
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=2,
            micro_batch_size=2,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )
        helper.create_cudagraphs()
        try:
            cu_seqlens = torch.tensor([0, 16, 52, target_tokens], dtype=torch.int32, device=device)
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens,
                max_seqlen_q=max_grid_size**2,
                max_seqlen_kv=max_grid_size**2,
                pad_between_seqs=False,
            )
            rotary_pos_emb = torch.randn(target_tokens, 1, 1, graph_config.kv_channels, device=device)
            eager_hidden = torch.randn(
                target_tokens,
                1,
                graph_config.hidden_size,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            graph_hidden = eager_hidden.detach().clone().requires_grad_()
            for layer in graph_encoder.decoder.layers:
                layer.current_microbatch = 0

            eager_out = eager_encoder.decoder(
                hidden_states=eager_hidden,
                attention_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
            graph_out = graph_encoder.decoder(
                hidden_states=graph_hidden,
                attention_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
            torch.testing.assert_close(graph_out, eager_out, rtol=5e-3, atol=5e-3)

            grad_out = torch.randn_like(eager_out)
            eager_out.backward(grad_out)
            graph_out.backward(grad_out)
            torch.testing.assert_close(graph_hidden.grad, eager_hidden.grad, rtol=5e-3, atol=5e-3)

            eager_params = dict(eager_encoder.decoder.named_parameters())
            for name, graph_param in graph_encoder.decoder.named_parameters():
                eager_grad = eager_params[name].grad
                graph_grad = graph_param.grad
                if eager_grad is None and graph_grad is None:
                    continue
                assert eager_grad is not None, name
                assert graph_grad is not None, name
                torch.testing.assert_close(graph_grad, eager_grad, rtol=5e-3, atol=5e-3, msg=name)
        finally:
            if helper.graphs_created():
                helper.delete_cuda_graphs()
    finally:
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()


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
    run=False,
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
    _reset_mcore_cuda_graph_state()
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
        record_cu_seqlens = torch.tensor([0, 128, 256, 256], dtype=torch.int32, device=device)
        replay_cu_seqlens = torch.tensor([0, 64, 192, 256], dtype=torch.int32, device=device)

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
        graph_config.cuda_graph_impl = "local"
        graph_config.cuda_graph_modules = []
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

        record_hidden = torch.randn(
            target_tokens, 1, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        rotary_pos_emb = torch.randn(target_tokens, 1, 1, head_dim, device=device)

        # Match --vision-layer-cuda-graph: first pass records MCore local graph metadata,
        # then create_cudagraphs() captures, and subsequent passes replay those graphs.
        record_out = graph_layer(record_hidden, rotary_pos_emb, record_cu_seqlens)
        record_out.backward(torch.randn_like(record_out))
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created
        for layer in graph_encoder.decoder.layers:
            assert hasattr(layer, "cudagraph_manager")
            assert len(layer.cudagraph_manager.cudagraph_runners) == 1
            runner = layer.cudagraph_manager.cudagraph_runners[0]
            assert runner.cudagraph_created
            assert runner.fwd_graph is not None
            assert runner.bwd_graph is not None

        graph_layer.zero_grad(set_to_none=True)

        eager_hidden = torch.randn(
            target_tokens, 1, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        graph_hidden = eager_hidden.detach().clone().requires_grad_()
        eager_out = eager_layer(eager_hidden, rotary_pos_emb, replay_cu_seqlens)
        torch.cuda.synchronize()
        graph_out = graph_layer(graph_hidden, rotary_pos_emb, replay_cu_seqlens)

        output_diff = (graph_out - eager_out).abs()
        print(
            "mcore_local_cuda_graph_layer_output "
            f"max={output_diff.max().item():.6e} mean={output_diff.mean().item():.6e} "
            f"bitwise={torch.equal(graph_out, eager_out)}",
            flush=True,
        )
        torch.testing.assert_close(graph_out, eager_out, rtol=5e-3, atol=5e-3)

        grad_out = torch.randn_like(eager_out)
        eager_out.backward(grad_out)
        graph_out.backward(grad_out)

        hidden_grad_diff = (graph_hidden.grad - eager_hidden.grad).abs()
        print(
            "mcore_local_cuda_graph_layer_hidden_grad "
            f"max={hidden_grad_diff.max().item():.6e} mean={hidden_grad_diff.mean().item():.6e} "
            f"bitwise={torch.equal(graph_hidden.grad, eager_hidden.grad)}",
            flush=True,
        )
        torch.testing.assert_close(graph_hidden.grad, eager_hidden.grad, rtol=5e-3, atol=5e-3)

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
            torch.testing.assert_close(graph_grad, eager_grad, rtol=5e-3, atol=5e-3, msg=name)
        print(
            "mcore_local_cuda_graph_layer_worst_param_grad "
            f"name={worst_param[0]} max={worst_param[1]:.6e} "
            f"mean={worst_param[2]:.6e} bitwise={worst_param[3]}",
            flush=True,
        )
    finally:
        _reset_mcore_cuda_graph_state()
        Utils.destroy_model_parallel()
