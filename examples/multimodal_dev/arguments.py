# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""

import argparse
import math


def parse_image_seq_length(value: str) -> int | float:
    """Parse a non-negative image-token cap or positive infinity."""

    if value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"image sequence length must be a non-negative integer or 'inf', got {value!r}"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"image sequence length must be non-negative, got {parsed}"
        )
    return parsed


def add_multimodal_args(parser):
    """Add multimodal-specific arguments to the Megatron argument parser."""
    group = parser.add_argument_group("Multimodal", "Multimodal model arguments")

    group.add_argument(
        "--model-arch",
        type=str,
        default="qwen35_vl",
        help="Model architecture. Available: qwen35_vl",
    )
    group.add_argument(
        "--model-variant",
        type=str,
        default="proxy",
        help="Model variant (size). E.g. proxy, 9b, 397b_a17b",
    )
    group.add_argument(
        "--dataset-provider", type=str, default="mock", help="Dataset provider: mock"
    )
    group.add_argument(
        "--image-token-id", type=int, default=248056, help="Token ID for image placeholder tokens"
    )
    group.add_argument(
        "--image-size", type=int, default=224, help="Image size (height and width) for mock data"
    )
    group.add_argument(
        "--total-seq-length", type=int, default=1024, help="Total sequence length for mock data"
    )
    group.add_argument(
        "--mock-vision-distribution",
        type=str,
        choices=("random", "empirical_record"),
        default="random",
        help=(
            "Visual-layout source for mock data. 'empirical_record' replays public v1 "
            "JSONL records from --mock-vision-records-path."
        ),
    )
    group.add_argument(
        "--mock-vision-records-path",
        type=str,
        default=None,
        help="Public v1 JSONL file or directory used by empirical mock vision records.",
    )
    group.add_argument(
        "--mock-vision-record-sampling",
        type=str,
        choices=("cycle", "with_replacement"),
        default="cycle",
        help="Select empirical records cyclically or by deterministic replacement sampling.",
    )
    group.add_argument(
        "--mock-random-seed",
        type=int,
        default=1234,
        help="Base seed for deterministic empirical mock sample generation.",
    )
    group.add_argument(
        "--image-seq-length",
        type=parse_image_seq_length,
        default=256,
        help="Maximum image tokens in mock data; use 'inf' to disable this cap.",
    )
    group.add_argument(
        "--vision-num-layers",
        type=int,
        default=None,
        help=("Override for vision backbone depth. " "Useful for proxy perf runs."),
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help=(
            "HuggingFace processor path for real VLM datasets " "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
        ),
    )
    group.add_argument(
        "--recompute-vision",
        action="store_true",
        default=False,
        help=(
            "Enable full activation recomputation for vision encoder layers. "
            "Uses uniform method and recomputes every layer. "
            "Independent of the decoder --recompute-* flags."
        ),
    )
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help=("Pack variable-length sequences into THD format to eliminate " "padding waste."),
    )
    group.add_argument(
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=("Use vanilla collate function to collate the data."),
    )

    return parser
