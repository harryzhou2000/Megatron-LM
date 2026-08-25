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
    cap_vision_layouts,
    load_vision_records,
    record_to_grids,
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
        vision_distribution: str = "random",
        vision_records_path: str = None,
        vision_record_sampling: str = "cycle",
        vision_records: Sequence[VisionRecord] | None = None,
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
        self.vision_distribution = vision_distribution
        self.vision_record_sampling = vision_record_sampling
        self.random_seed = random_seed

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
                    "mock_vision_record_sampling must be 'cycle' or 'with_replacement', "
                    f"got {self.vision_record_sampling!r}"
                )
            for record in self.vision_records:
                if record.llm_sequence_length > self.seq_length:
                    raise ValueError(
                        f"{record.source} llm_sequence_length={record.llm_sequence_length} "
                        f"exceeds configured maximum {self.seq_length}"
                    )
                record_to_grids(
                    record,
                    patch_size=self.patch_size,
                    temporal_patch_size=self.temporal_patch_size,
                    spatial_merge_size=self.spatial_merge_size,
                )
        else:
            self.vision_records = None

        h_patches = image_size // patch_size
        w_patches = image_size // patch_size
        t_patches = temporal_patch_size
        self.grid_thw = torch.tensor([[t_patches, h_patches, w_patches]])

        self.num_merged_tokens = (
            t_patches * (h_patches // spatial_merge_size) * (w_patches // spatial_merge_size)
        )
        self.image_seq_length = (
            image_seq_length
            if self.vision_distribution == "empirical_record"
            else min(image_seq_length, self.num_merged_tokens)
        )
        self.total_patches = t_patches * h_patches * w_patches

    def __len__(self):
        return self.num_samples

    def _fixed_sample(self):
        """Construct the historical fixed-image sample without changing its RNG behavior."""

        # Reserve 1 slot for the vision_start sentinel before image tokens.
        text_length = self.seq_length - self.image_seq_length - 1
        text_tokens = torch.randint(1, self.vocab_size, (text_length,), dtype=torch.long)
        special_ids = {self.image_token_id, self.video_token_id, self.vision_start_token_id}
        for sid in special_ids:
            text_tokens[text_tokens == sid] = 1

        prefix_len = text_length // 2
        suffix_len = text_length - prefix_len
        input_ids = torch.cat(
            [
                text_tokens[:prefix_len],
                torch.tensor([self.vision_start_token_id], dtype=torch.long),
                torch.full((self.image_seq_length,), self.image_token_id, dtype=torch.long),
                text_tokens[prefix_len : prefix_len + suffix_len],
            ]
        )

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0

        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[-1] = 0

        pixel_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size
        pixel_values = torch.randn(self.total_patches, pixel_dim)

        image_grid_thw = self.grid_thw.clone()

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
            "cu_seqlens_padded": torch.tensor([0, self.seq_length], dtype=torch.int32),
            "max_seqlen": torch.tensor(self.seq_length, dtype=torch.int32),
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _record_sample(self, idx):
        record = select_vision_record(
            self.vision_records,
            sample_index=idx,
            sampling=self.vision_record_sampling,
            random_seed=self.random_seed,
        )
        grids, token_counts = record_to_grids(
            record,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
        )
        grids, token_counts = cap_vision_layouts(grids, token_counts, self.image_seq_length)
        image_grid_thw = (
            torch.tensor(grids, dtype=torch.long)
            if grids
            else torch.empty((0, 3), dtype=torch.long)
        )
        sequence_length = record.llm_sequence_length
        text_length = sequence_length - sum(token_counts) - len(token_counts)

        generator = torch.Generator()
        generator.manual_seed(self.random_seed + idx)
        text_tokens = torch.randint(
            1, self.vocab_size, (text_length,), dtype=torch.long, generator=generator
        )
        special_ids = {self.image_token_id, self.video_token_id, self.vision_start_token_id}
        for special_id in special_ids:
            text_tokens[text_tokens == special_id] = 1

        vision_segments = []
        for token_count in token_counts:
            vision_segments.extend(
                [
                    torch.tensor([self.vision_start_token_id], dtype=torch.long),
                    torch.full((token_count,), self.image_token_id, dtype=torch.long),
                ]
            )
        prefix_length = text_length // 2
        input_ids = torch.cat(
            [text_tokens[:prefix_length], *vision_segments, text_tokens[prefix_length:]]
        )

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[-1] = 0

        real_image_grid_thw = image_grid_thw
        vision_image_grid_thw = (
            image_grid_thw
            if image_grid_thw.numel()
            else torch.tensor(
                [[1, self.spatial_merge_size, self.spatial_merge_size]], dtype=torch.long
            )
        )
        pixel_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size
        total_patches = int(vision_image_grid_thw.prod(dim=1).sum().item())
        pixel_values = torch.randn(total_patches, pixel_dim, generator=generator)

        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=real_image_grid_thw,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "cu_seqlens": torch.tensor([0, sequence_length], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0, sequence_length], dtype=torch.int32),
            "max_seqlen": torch.tensor(sequence_length, dtype=torch.int32),
            "position_ids": position_ids.squeeze(1),
            "pixel_values": pixel_values,
            "image_grid_thw": vision_image_grid_thw,
            "has_real_vision": torch.tensor(bool(grids), dtype=torch.bool),
        }

    def __getitem__(self, idx):
        if self.vision_distribution == "empirical_record":
            return self._record_sample(idx)
        return self._fixed_sample()


def mock_collate_fn(batch):
    """Collate: handles position_ids ``[3, S]`` stacking."""
    result = {}
    keys = batch[0].keys()
    real_vision_indices = [
        index for index, sample in enumerate(batch) if bool(sample.get("has_real_vision", True))
    ]
    vision_indices = real_vision_indices or [0]
    for key in keys:
        if key == "has_real_vision":
            continue
        tensors = [sample[key] for sample in batch]
        if key == "position_ids":
            result[key] = torch.stack(tensors, dim=1)
        elif key == "image_grid_thw":
            result[key] = torch.cat([batch[index][key] for index in vision_indices], dim=0)
        elif key == "pixel_values":
            result[key] = torch.cat([batch[index][key] for index in vision_indices], dim=0)
        else:
            result[key] = torch.stack(tensors, dim=0)
    return result


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide mock train / val / test datasets."""
    from megatron.training import get_args

    args = get_args()
    vision_distribution = getattr(args, "mock_vision_distribution", "random")
    vision_records_path = getattr(args, "mock_vision_records_path", None)
    if vision_distribution == "empirical_record":
        if not vision_records_path:
            raise ValueError(
                "--mock-vision-records-path is required with "
                "--mock-vision-distribution=empirical_record"
            )
        if not getattr(args, "use_vanilla_collate_fn", False):
            raise ValueError(
                "Empirical mock vision records require --use-vanilla-collate-fn because "
                "sample token, grid, and pixel shapes may vary"
            )
        if getattr(args, "cuda_graph_impl", "none") == "full_iteration":
            raise ValueError(
                "Empirical mock vision records do not support full-iteration CUDA graphs; "
                "record-dependent token, grid, and pixel shapes require bucketing"
            )
    vision_records = (
        tuple(load_vision_records(vision_records_path))
        if vision_distribution == "empirical_record"
        else None
    )
    maximum_sequence_length = min(
        getattr(args, "total_seq_length", 1024), getattr(args, "seq_length", 1024)
    )
    kwargs = dict(
        seq_length=maximum_sequence_length,
        image_seq_length=getattr(args, "image_seq_length", 256),
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", 248056),
        image_size=getattr(args, "image_size", 224),
        vision_distribution=vision_distribution,
        vision_records_path=vision_records_path,
        vision_record_sampling=getattr(args, "mock_vision_record_sampling", "cycle"),
        vision_records=vision_records,
        random_seed=getattr(args, "mock_random_seed", 1234),
    )

    train_ds = MockQwen35VLDataset(num_samples=train_val_test_num_samples[0], **kwargs)
    val_ds = MockQwen35VLDataset(num_samples=train_val_test_num_samples[1], **kwargs)
    test_ds = MockQwen35VLDataset(num_samples=train_val_test_num_samples[2], **kwargs)

    return train_ds, val_ds, test_ds
