# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""


def add_multimodal_args(parser):
    """Add multimodal-specific arguments to the Megatron argument parser."""
    group = parser.add_argument_group(
        "Multimodal", "Multimodal model arguments",
    )

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
        "--dataset-provider",
        type=str,
        default="mock",
        help="Dataset provider: mock",
    )
    group.add_argument(
        "--image-token-id",
        type=int,
        default=248056,
        help="Token ID for image placeholder tokens",
    )
    group.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (height and width) for mock data",
    )
    group.add_argument(
        "--mock-variable-images",
        action="store_true",
        default=False,
        help="Generate variable image counts and image sizes in mock Qwen-VL data.",
    )
    group.add_argument(
        "--mock-min-images-per-sample",
        type=int,
        default=1,
        help="Minimum images per sample when --mock-variable-images is enabled.",
    )
    group.add_argument(
        "--mock-max-images-per-sample",
        type=int,
        default=3,
        help="Maximum images per sample when --mock-variable-images is enabled.",
    )
    group.add_argument(
        "--mock-image-size-choices",
        type=str,
        default="224,448",
        help="Comma-separated square image sizes for variable mock images.",
    )
    group.add_argument(
        "--mock-image-size-weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated weights for mock_image_size_choices. "
            "For 224/448/896 with small images more common, use e.g. '8,2,1'."
        ),
    )
    group.add_argument(
        "--mock-image-count-weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated weights for image counts in "
            "[mock_min_images_per_sample, mock_max_images_per_sample]. "
            "For 0-1 with zeros more common, use e.g. '4,1'."
        ),
    )
    group.add_argument(
        "--mock-random-seed",
        type=int,
        default=1234,
        help="Base seed for deterministic per-sample mock data generation.",
    )
    group.add_argument(
        "--mock-variable-seq-length",
        action="store_true",
        default=False,
        help="Generate variable total sequence lengths for mock multimodal samples.",
    )
    group.add_argument(
        "--mock-min-seq-length",
        type=int,
        default=None,
        help="Minimum total sequence length when --mock-variable-seq-length is enabled.",
    )
    group.add_argument(
        "--mock-max-seq-length",
        type=int,
        default=None,
        help="Maximum total sequence length when --mock-variable-seq-length is enabled.",
    )
    group.add_argument(
        "--total-seq-length",
        type=int,
        default=1024,
        help="Total sequence length for mock data",
    )
    group.add_argument(
        "--image-seq-length",
        type=int,
        default=256,
        help="Number of image tokens in mock data",
    )
    group.add_argument(
        "--vision-num-layers",
        type=int,
        default=None,
        help=(
            "Override for vision backbone depth. "
            "Useful for proxy perf runs."
        ),
    )
    group.add_argument(
        "--vision-max-packed-tokens",
        type=int,
        default=None,
        help=(
            "Pad Qwen vision transformer inputs to this fixed total patch-token count. "
            "This is an opt-in static THD staging aid for CUDA graph experiments."
        ),
    )
    group.add_argument(
        "--vision-max-packed-sequences",
        type=int,
        default=None,
        help=(
            "Pad Qwen vision THD cu_seqlens to this fixed number of frame sequences. "
            "Use with --vision-max-packed-tokens for static graph input metadata; include one "
            "extra sequence for the dummy padding tail when token padding may be needed."
        ),
    )
    group.add_argument(
        "--vision-max-grid-size",
        type=int,
        default=None,
        help=(
            "Fixed maximum Qwen vision patch-grid side used to build graph-friendly position "
            "and RoPE metadata without per-microbatch CPU shape queries. Set this to the "
            "largest H/W patch-grid size in the bucket."
        ),
    )
    group.add_argument(
        "--vision-layer-cuda-graph",
        action="store_true",
        default=False,
        help=(
            "Enable local layer-wise CUDA graphs for Qwen vision transformer layers only. "
            "Use with the static vision THD bucket flags."
        ),
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help=(
            "HuggingFace processor path for real VLM datasets "
            "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
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
        help=(
            "Pack variable-length sequences into THD format to eliminate "
            "padding waste."
        ),
    )
    group.add_argument(
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=(
            "Use vanilla collate function to collate the data."
        ),
    )

    return parser
