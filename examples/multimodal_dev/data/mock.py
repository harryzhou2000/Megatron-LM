# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mock dataset for multimodal_dev end-to-end testing.

Generates synthetic image + text data.  Each sample has random text
tokens with image-token placeholders, random pixel values sized for the
vision encoder, 3D MRoPE position IDs, and shifted labels.
"""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.data.mock_records import (
    VisionRecord,
    cap_visual_token_counts,
    closest_square_grid,
    load_vision_records,
    select_vision_record,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index


class MockQwen35VLDataset(Dataset):
    """Synthetic Qwen3.5-VL training samples.

    Args:
        num_samples: Number of samples.
        seq_length: Total sequence length (text + image tokens).
        image_seq_length: Number of image tokens per sample.
        vocab_size: Vocabulary size for random text tokens.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking start of a vision region.
        image_size: Image height and width in pixels.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor.
        vision_distribution: ``"random"`` or ``"empirical_record"`` visual-layout source.
        vision_records_path: JSON/JSONL statistics path required for empirical records.
        vision_record_sampling: ``"cycle"`` or ``"with_replacement"`` record selection.
        vision_records: Optional preloaded empirical records shared by datasets.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 1024,
        image_seq_length: int = 256,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        variable_images: bool = False,
        min_images_per_sample: int = 1,
        max_images_per_sample: int = 3,
        image_size_choices: str = "224,448",
        image_size_weights: str = None,
        image_count_weights: str = None,
        random_seed: int = 1234,
        variable_seq_length: bool = False,
        min_seq_length: int = None,
        max_seq_length: int = None,
        vision_distribution: str = "random",
        vision_records_path: str = None,
        vision_record_sampling: str = "cycle",
        vision_records: Sequence[VisionRecord] | None = None,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.variable_images = variable_images
        self.min_images_per_sample = min_images_per_sample
        self.max_images_per_sample = max_images_per_sample
        self.variable_seq_length = variable_seq_length
        self.vision_distribution = vision_distribution
        self.vision_record_sampling = vision_record_sampling
        self.min_seq_length = (
            min_seq_length if min_seq_length is not None else max(1, seq_length // 2)
        )
        self.max_seq_length = max_seq_length if max_seq_length is not None else seq_length
        if self.variable_seq_length:
            if self.min_seq_length <= 0 or self.max_seq_length < self.min_seq_length:
                raise ValueError(
                    "mock variable sequence length range must satisfy "
                    "0 < min_seq_length <= max_seq_length"
                )
            if self.max_seq_length > self.seq_length:
                raise ValueError(
                    "mock_max_seq_length cannot exceed total_seq_length; "
                    f"got max_seq_length={self.max_seq_length}, total_seq_length={self.seq_length}"
                )
        if self.vision_distribution not in {"random", "empirical_record"}:
            raise ValueError(
                "mock_vision_distribution must be 'random' or 'empirical_record', "
                f"got {self.vision_distribution!r}"
            )
        if self.vision_distribution == "empirical_record":
            if vision_records is not None:
                self.vision_records = tuple(vision_records)
            elif vision_records_path:
                self.vision_records = tuple(load_vision_records(vision_records_path))
            else:
                raise ValueError(
                    "mock_vision_records_path is required when "
                    "mock_vision_distribution=empirical_record"
                )
            if not self.vision_records:
                raise ValueError("Empirical mock vision records must not be empty")
            if self.vision_record_sampling not in {"cycle", "with_replacement"}:
                raise ValueError(
                    "mock_vision_record_sampling must be 'cycle' or "
                    f"'with_replacement', got {self.vision_record_sampling!r}"
                )
        else:
            self.vision_records = None
        self.image_size_choices = [
            int(size.strip()) for size in image_size_choices.split(",") if size.strip()
        ]
        if not self.image_size_choices:
            self.image_size_choices = [image_size]
        self.image_size_weights = None
        if image_size_weights is not None:
            weights = [
                float(weight.strip()) for weight in image_size_weights.split(",") if weight.strip()
            ]
            if len(weights) != len(self.image_size_choices):
                raise ValueError(
                    "mock_image_size_weights must have one weight per mock_image_size_choices; "
                    f"expected={len(self.image_size_choices)}, got={len(weights)}"
                )
            if any(weight < 0 for weight in weights) or sum(weights) <= 0:
                raise ValueError("mock_image_size_weights must be non-negative and nonzero")
            self.image_size_weights = torch.tensor(weights, dtype=torch.float)
        if self.min_images_per_sample < 0 or self.max_images_per_sample < self.min_images_per_sample:
            raise ValueError(
                "mock image count range must satisfy 0 <= min_images <= max_images"
            )
        self.random_seed = random_seed
        self.image_count_weights = None
        if image_count_weights is not None:
            weights = [
                float(weight.strip()) for weight in image_count_weights.split(",") if weight.strip()
            ]
            expected = self.max_images_per_sample - self.min_images_per_sample + 1
            if len(weights) != expected:
                raise ValueError(
                    "mock_image_count_weights must have one weight per possible image count; "
                    f"expected={expected}, got={len(weights)}"
                )
            if any(weight < 0 for weight in weights) or sum(weights) <= 0:
                raise ValueError("mock_image_count_weights must be non-negative and nonzero")
            self.image_count_weights = torch.tensor(weights, dtype=torch.float)

        h_patches = image_size // patch_size
        w_patches = image_size // patch_size
        t_patches = temporal_patch_size
        self.grid_thw = torch.tensor([[t_patches, h_patches, w_patches]])

        self.num_merged_tokens = (
            t_patches
            * (h_patches // spatial_merge_size)
            * (w_patches // spatial_merge_size)
        )
        # Fixed-image mode keeps the historical single-image cap. Variable-image
        # and empirical-record modes use this as the per-sample cap for total
        # post-merge image tokens.
        self.image_seq_length = (
            image_seq_length
            if variable_images or vision_distribution == "empirical_record"
            else min(image_seq_length, self.num_merged_tokens)
        )
        self.total_patches = t_patches * h_patches * w_patches

    def _generator_for_idx(self, idx):
        generator = torch.Generator()
        generator.manual_seed(self.random_seed + idx)
        return generator

    def _sample_grids(self, idx, generator):
        if self.vision_distribution == "empirical_record":
            record = select_vision_record(
                self.vision_records,
                sample_index=idx,
                sampling=self.vision_record_sampling,
                random_seed=self.random_seed,
            )
            rows = [
                closest_square_grid(token_count, self.spatial_merge_size)
                for token_count in cap_visual_token_counts(
                    record.visual_token_counts, self.image_seq_length
                )
            ]
            if not rows:
                return torch.empty((0, 3), dtype=torch.long)
            return torch.tensor(rows, dtype=torch.long)
        if not self.variable_images:
            return self.grid_thw.clone()
        span = self.max_images_per_sample - self.min_images_per_sample + 1
        if self.image_count_weights is None:
            num_images = self.min_images_per_sample + int(
                torch.randint(span, (1,), generator=generator).item()
            )
        else:
            sampled = torch.multinomial(self.image_count_weights, 1, generator=generator).item()
            num_images = self.min_images_per_sample + int(sampled)
        rows = []
        for image_idx in range(num_images):
            if self.image_size_weights is None:
                size_idx = int(
                    torch.randint(len(self.image_size_choices), (1,), generator=generator).item()
                )
            else:
                size_idx = int(
                    torch.multinomial(self.image_size_weights, 1, generator=generator).item()
                )
            image_size = self.image_size_choices[size_idx]
            if image_size % (self.patch_size * self.spatial_merge_size) != 0:
                raise ValueError(
                    "mock image sizes must be divisible by patch_size * spatial_merge_size; "
                    f"got image_size={image_size}, patch_size={self.patch_size}, "
                    f"spatial_merge_size={self.spatial_merge_size}"
                )
            rows.append(
                [
                    self.temporal_patch_size,
                    image_size // self.patch_size,
                    image_size // self.patch_size,
                ]
            )
        return self._cap_grids_to_image_seq_length(rows)

    def _cap_grids_to_image_seq_length(self, rows):
        """Keep image grids within the configured post-merge image-token cap."""
        if self.image_seq_length < 0 or self.image_seq_length == float("inf"):
            if not rows:
                return torch.empty((0, 3), dtype=torch.long)
            return torch.tensor(rows, dtype=torch.long)

        capped_rows = []
        total_merged_tokens = 0
        merge_area = self.spatial_merge_size * self.spatial_merge_size
        for row in rows:
            merged_tokens = row[0] * row[1] * row[2] // merge_area
            if total_merged_tokens + merged_tokens > self.image_seq_length:
                continue
            capped_rows.append(row)
            total_merged_tokens += merged_tokens
        if not capped_rows:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.tensor(capped_rows, dtype=torch.long)

    def _sample_seq_length(self, generator):
        if not self.variable_seq_length:
            return self.seq_length
        span = self.max_seq_length - self.min_seq_length + 1
        return self.min_seq_length + int(torch.randint(span, (1,), generator=generator).item())

    def _cap_grids_to_seq_length(self, image_grid_thw, seq_length):
        if image_grid_thw.numel() == 0:
            return image_grid_thw
        rows = []
        total_image_tokens = 0
        merge_area = self.spatial_merge_size * self.spatial_merge_size
        for row in image_grid_thw.tolist():
            image_tokens = row[0] * row[1] * row[2] // merge_area
            # Keep at least one text token. Each image also needs a vision_start token.
            if total_image_tokens + image_tokens + len(rows) + 1 >= seq_length:
                continue
            rows.append(row)
            total_image_tokens += image_tokens
        if not rows:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def collate_fn(self, batch):
        """Collate mock samples while preserving full-iteration CUDA graph input shape rules."""
        try:
            from megatron.training import get_args

            args = get_args()
            use_vanilla_collate = bool(getattr(args, "use_vanilla_collate_fn", False))
            full_iteration_graph = getattr(args, "cuda_graph_impl", "none") == "full_iteration"
        except AssertionError:
            use_vanilla_collate = False
            full_iteration_graph = False

        if full_iteration_graph:
            if (
                self.variable_images
                or self.variable_seq_length
                or self.vision_distribution == "empirical_record"
            ):
                raise ValueError(
                    "Full-iteration CUDA graph requires a top-level dict with static tensor "
                    "shapes. Variable-image, variable-sequence-length, and empirical-record "
                    "mocks emit variable pixel/grid/token shapes and need an explicit padding "
                    "or bucketing contract before they can be captured."
                )
            return mock_collate_fn(batch)

        if use_vanilla_collate:
            return batch
        return mock_collate_fn(batch)

    def __getitem__(self, idx):
        generator = self._generator_for_idx(idx)
        image_grid_thw = self._sample_grids(idx, generator)
        seq_length = self._sample_seq_length(generator)
        image_grid_thw = self._cap_grids_to_seq_length(image_grid_thw, seq_length)
        patch_counts = image_grid_thw.prod(dim=1) if image_grid_thw.numel() else torch.zeros(0)
        merged_counts = patch_counts // (self.spatial_merge_size * self.spatial_merge_size)
        image_seq_length = int(merged_counts.sum().item())

        # Reserve 1 slot for the vision_start sentinel before image tokens.
        text_length = seq_length - image_seq_length - image_grid_thw.shape[0]
        if text_length <= 0:
            raise ValueError(
                "Mock visual tokens exceed total sequence length: "
                f"seq_length={seq_length}, image_tokens={image_seq_length}, "
                f"num_images={image_grid_thw.shape[0]}"
            )
        text_tokens = torch.randint(
            1, self.vocab_size, (text_length,), dtype=torch.long, generator=generator,
        )
        special_ids = {
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
        }
        for sid in special_ids:
            text_tokens[text_tokens == sid] = 1

        prefix_len = text_length // 2
        suffix_len = text_length - prefix_len
        vision_segments = []
        for token_count in merged_counts.tolist():
            vision_segments.extend(
                [
                    torch.tensor([self.vision_start_token_id], dtype=torch.long),
                    torch.full((int(token_count),), self.image_token_id, dtype=torch.long),
                ]
            )
        input_ids = torch.cat([
            text_tokens[:prefix_len],
            *vision_segments,
            text_tokens[prefix_len: prefix_len + suffix_len],
        ])

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0

        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[-1] = 0

        pixel_dim = (
            3
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        total_patches = int(patch_counts.sum().item())
        pixel_values = torch.randn(total_patches, pixel_dim, generator=generator)

        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        position_ids = position_ids.squeeze(1)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "cu_seqlens": torch.tensor([0, seq_length], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor(
                [0, seq_length], dtype=torch.int32,
            ),
            "max_seqlen": torch.tensor(seq_length, dtype=torch.int32),
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def mock_collate_fn(batch):
    """Collate: handles position_ids ``[3, S]`` stacking."""
    result = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [sample[key] for sample in batch]
        if key == "position_ids":
            result[key] = torch.stack(tensors, dim=1)
        elif key == "image_grid_thw":
            result[key] = torch.cat(tensors, dim=0)
        elif key == "pixel_values":
            result[key] = torch.cat(tensors, dim=0)
        else:
            result[key] = torch.stack(tensors, dim=0)
    return result


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide mock train / val / test datasets."""
    from megatron.training import get_args

    args = get_args()
    vision_distribution = getattr(args, "mock_vision_distribution", "random")
    vision_records_path = getattr(args, "mock_vision_records_path", None)
    vision_records = (
        tuple(load_vision_records(vision_records_path))
        if vision_distribution == "empirical_record" and vision_records_path
        else None
    )
    kwargs = dict(
        seq_length=getattr(args, "total_seq_length", 1024),
        image_seq_length=getattr(args, "image_seq_length", 256),
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", 248056),
        image_size=getattr(args, "image_size", 224),
        variable_images=getattr(args, "mock_variable_images", False),
        min_images_per_sample=getattr(args, "mock_min_images_per_sample", 1),
        max_images_per_sample=getattr(args, "mock_max_images_per_sample", 3),
        image_size_choices=getattr(args, "mock_image_size_choices", "224,448"),
        image_size_weights=getattr(args, "mock_image_size_weights", None),
        image_count_weights=getattr(args, "mock_image_count_weights", None),
        random_seed=getattr(args, "mock_random_seed", 1234),
        variable_seq_length=getattr(args, "mock_variable_seq_length", False),
        min_seq_length=getattr(args, "mock_min_seq_length", None),
        max_seq_length=getattr(args, "mock_max_seq_length", None),
        vision_distribution=vision_distribution,
        vision_records_path=vision_records_path,
        vision_record_sampling=getattr(args, "mock_vision_record_sampling", "cycle"),
        vision_records=vision_records,
    )

    train_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[0], **kwargs,
    )
    val_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[1], **kwargs,
    )
    test_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[2], **kwargs,
    )

    return train_ds, val_ds, test_ds
