# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the standalone MoE performance frontend."""

from types import SimpleNamespace

import pytest

from tests.functional_tests.test_cases.common.moe_perf import recipe_frontend


def _make_args(**overrides):
    values = {
        "profile": False,
        "use_pytorch_profiler": False,
        "profile_step_start": 7,
        "profile_step_end": 8,
        "profile_ranks": [0],
        "nvtx_ranges": True,
        "record_shapes": False,
        "rank": 0,
        "moe_perf_warmup_iters": 10,
        "moe_perf_iters": 150,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_iteration_plan_without_nsys_uses_frontend_counts():
    args = _make_args()

    assert recipe_frontend._moe_perf_iteration_plan(args) == (160, 10)


@pytest.mark.parametrize(
    ("load_balancing_type", "enable_expert_bias", "expected"),
    [("quantile_balancing", False, "quantile"), ("none", True, "sign"), ("none", False, "none")],
)
def test_router_bias_update_method_matches_current_mcore_modes(
    load_balancing_type, enable_expert_bias, expected
):
    config = SimpleNamespace(
        moe_router_load_balancing_type=load_balancing_type,
        moe_router_enable_expert_bias=enable_expert_bias,
    )

    assert recipe_frontend._router_bias_update_method(config) == expected


def test_kimi_moe_flops_count_situ_glu_and_latent_projections():
    args = SimpleNamespace(
        micro_batch_size=1,
        data_parallel_size=2,
        seq_length=3,
        hidden_size=8,
        ffn_hidden_size=32,
        moe_ffn_hidden_size=16,
        moe_shared_expert_intermediate_size=12,
        moe_router_topk=2,
        moe_latent_size=4,
        swiglu=False,
        situ_glu=True,
    )
    total_tokens = 6
    scale_factor = 3.0 / 2.0
    routed_expert_flops = 4 * total_tokens * 4 * 16 * 2 * scale_factor
    latent_projection_flops = 4 * total_tokens * 8 * 4
    shared_expert_flops = 4 * total_tokens * 8 * 12 * scale_factor

    assert recipe_frontend._moe_only_flops_per_iteration(args, [object()]) == 3 * (
        routed_expert_flops + latent_projection_flops + shared_expert_flops
    )


def test_iteration_plan_with_nsys_matches_standard_profile_window():
    args = _make_args(profile=True)

    assert recipe_frontend._moe_perf_iteration_plan(args) == (8, 7)


@pytest.mark.parametrize(("profile_start", "profile_end"), [(-1, 8), (8, 8), (9, 8)])
def test_iteration_plan_rejects_invalid_nsys_window(profile_start, profile_end):
    args = _make_args(profile=True, profile_step_start=profile_start, profile_step_end=profile_end)

    with pytest.raises(ValueError, match="profile-step-start"):
        recipe_frontend._moe_perf_iteration_plan(args)


def test_nsys_profiler_matches_training_start_stop_order(monkeypatch):
    calls = []

    class FakeCudart:
        @staticmethod
        def cudaProfilerStart():
            calls.append("start")
            return 0

        @staticmethod
        def cudaProfilerStop():
            calls.append("stop")
            return 0

    class FakeNVTXContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            calls.append("exit")

    monkeypatch.setattr(recipe_frontend.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        recipe_frontend,
        "configure_nvtx_profiling",
        lambda enabled: calls.append(("configure_nvtx", enabled)),
    )
    monkeypatch.setattr(recipe_frontend.torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(recipe_frontend.torch.cuda, "check_error", lambda error: error)
    monkeypatch.setattr(
        recipe_frontend.torch.autograd.profiler, "emit_nvtx", lambda **kwargs: FakeNVTXContext()
    )

    profiler = recipe_frontend._NSysProfiler(_make_args(profile=True))
    profiler.start_if_needed(6)
    profiler.start_if_needed(7)
    profiler.stop_if_needed(7)
    profiler.stop_if_needed(8)
    profiler.close()

    assert calls == [
        ("configure_nvtx", True),
        "start",
        "enter",
        ("configure_nvtx", False),
        "stop",
        "exit",
    ]


def test_nsys_profiler_ignores_unselected_rank(monkeypatch):
    monkeypatch.setattr(recipe_frontend.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        recipe_frontend.torch.cuda,
        "cudart",
        lambda: pytest.fail("unselected rank must not call CUDA profiler APIs"),
    )

    profiler = recipe_frontend._NSysProfiler(_make_args(profile=True, rank=1, profile_ranks=[0]))
    profiler.start_if_needed(7)
    profiler.stop_if_needed(8)

    assert not profiler.enabled
