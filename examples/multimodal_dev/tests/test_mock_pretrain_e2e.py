# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in eight-GPU Qwen MoE/GDN end-to-end test for empirical mock records.

Run this file with a single pytest controller process. The test itself launches
the eight distributed training workers::

    RUN_MULTIMODAL_PRETRAIN_E2E=1 python -m pytest -xvs \
        examples/multimodal_dev/tests/test_mock_pretrain_e2e.py
"""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_RUN_E2E = os.getenv("RUN_MULTIMODAL_PRETRAIN_E2E") == "1"


@pytest.mark.skipif(not _RUN_E2E, reason="set RUN_MULTIMODAL_PRETRAIN_E2E=1")
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight visible GPUs")
def test_empirical_records_complete_packed_pretrain(tmp_path):
    """Exercise packed records with full-size Qwen MoE/GDN layers and EP8."""

    repo_root = Path(__file__).resolve().parents[3]
    records_path = tmp_path / "records.jsonl"
    records = [
        {"format_version": 1, "llm_sequence_length": 96, "vision_tokens_per_image": [4]},
        {"format_version": 1, "llm_sequence_length": 112, "image_sizes": [[32, 64]]},
        {"format_version": 1, "llm_sequence_length": 80, "image_sizes": []},
    ]
    records_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo_root / "examples/multimodal_dev/pretrain_multimodal.py"),
        "--model-arch",
        "qwen35_vl",
        "--model-variant",
        "35b_a3b",
        "--dataset-provider",
        "mock",
        "--mock-vision-distribution",
        "empirical_record",
        "--mock-vision-records-path",
        str(records_path),
        "--mock-vision-record-sampling",
        "cycle",
        "--mock-random-seed",
        "1234",
        "--use-vanilla-collate-fn",
        "--use-packed-sequence",
        "--image-seq-length",
        "inf",
        "--image-token-id",
        "248056",
        "--image-size",
        "224",
        "--vision-num-layers",
        "1",
        "--total-seq-length",
        "128",
        "--seq-length",
        "128",
        "--max-position-embeddings",
        "256",
        "--num-layers",
        "4",
        "--hidden-size",
        "2048",
        "--ffn-hidden-size",
        "4096",
        "--num-attention-heads",
        "16",
        "--group-query-attention",
        "--num-query-groups",
        "2",
        # Keep the Qwen35 layer shape except for SDPA head width: 256-wide
        # heads have no CI-portable THD backend when sequences are inter-padded.
        "--kv-channels",
        "128",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--swiglu",
        "--disable-bias-linear",
        "--position-embedding-type",
        "rope",
        "--no-rope-fusion",
        "--rotary-percent",
        "0.25",
        "--rotary-base",
        "10000000",
        "--rotary-seq-len-interpolation-factor",
        "1",
        "--qk-layernorm",
        "--attention-output-gate",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--apply-layernorm-1p",
        "--untie-embeddings-and-output-weights",
        "--experimental-attention-variant",
        "gated_delta_net",
        "--linear-attention-freq",
        "4",
        "--linear-conv-kernel-dim",
        "4",
        "--linear-key-head-dim",
        "128",
        "--linear-value-head-dim",
        "128",
        "--linear-num-key-heads",
        "16",
        "--linear-num-value-heads",
        "32",
        "--num-experts",
        "256",
        "--moe-ffn-hidden-size",
        "512",
        "--moe-shared-expert-intermediate-size",
        "512",
        "--moe-shared-expert-gate",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-router-topk",
        "8",
        "--moe-grouped-gemm",
        "--moe-aux-loss-coeff",
        "1e-3",
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-router-dtype",
        "fp32",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "8",
        "--micro-batch-size",
        "4",
        "--global-batch-size",
        "32",
        "--train-iters",
        "3",
        "--lr",
        "1e-4",
        "--min-lr",
        "1e-5",
        "--lr-decay-style",
        "constant",
        "--lr-warmup-iters",
        "0",
        "--weight-decay",
        "0.0",
        "--clip-grad",
        "1.0",
        "--bf16",
        "--use-mcore-models",
        "--transformer-impl",
        "transformer_engine",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "248320",
        "--make-vocab-size-divisible-by",
        "485",
        "--distributed-backend",
        "nccl",
        "--enable-experimental",
        "--log-interval",
        "1",
        "--eval-iters",
        "1",
        "--eval-interval",
        "1000",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{repo_root}:{environment.get('PYTHONPATH', '')}"
    environment["TRITON_CACHE_DIR"] = str(tmp_path / "triton-cache")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "torchinductor-cache")
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    print(result.stdout, end="")

    assert result.returncode == 0
    losses = [float(value) for value in re.findall(r"lm loss:\s*([+\-0-9.eE]+)", result.stdout)]
    assert len(losses) >= 3
    assert all(math.isfinite(loss) for loss in losses)
