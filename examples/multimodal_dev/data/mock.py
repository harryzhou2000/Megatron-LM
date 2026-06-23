# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mock dataset for multimodal_dev end-to-end testing.

Generates synthetic image + text data.  Each sample has random text
tokens with image-token placeholders, random pixel values sized for the
vision encoder, 3D MRoPE position IDs, and shifted labels.
"""

import torch
from torch.utils.data import Dataset

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
        image_count_weights: str = None,
        random_seed: int = 1234,
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
        self.image_size_choices = [
            int(size.strip()) for size in image_size_choices.split(",") if size.strip()
        ]
        if not self.image_size_choices:
            self.image_size_choices = [image_size]
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
        self.image_seq_length = min(
            image_seq_length, self.num_merged_tokens,
        )
        self.total_patches = t_patches * h_patches * w_patches

    def _generator_for_idx(self, idx):
        generator = torch.Generator()
        generator.manual_seed(self.random_seed + idx)
        return generator

    def _sample_grids(self, idx, generator):
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
            image_size = self.image_size_choices[
                int(torch.randint(len(self.image_size_choices), (1,), generator=generator).item())
            ]
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
        if not rows:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        generator = self._generator_for_idx(idx)
        image_grid_thw = self._sample_grids(idx, generator)
        patch_counts = image_grid_thw.prod(dim=1) if image_grid_thw.numel() else torch.zeros(0)
        merged_counts = patch_counts // (self.spatial_merge_size * self.spatial_merge_size)
        image_seq_length = int(merged_counts.sum().item())

        # Reserve 1 slot for the vision_start sentinel before image tokens.
        text_length = self.seq_length - image_seq_length - image_grid_thw.shape[0]
        if text_length <= 0:
            raise ValueError(
                "Mock visual tokens exceed total sequence length: "
                f"seq_length={self.seq_length}, image_tokens={image_seq_length}, "
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
            "cu_seqlens": torch.tensor([0, self.seq_length], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor(
                [0, self.seq_length], dtype=torch.int32,
            ),
            "max_seqlen": torch.tensor(self.seq_length, dtype=torch.int32),
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
        image_count_weights=getattr(args, "mock_image_count_weights", None),
        random_seed=getattr(args, "mock_random_seed", 1234),
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
