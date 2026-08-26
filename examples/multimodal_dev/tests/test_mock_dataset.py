# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for empirical Qwen3.5-VL mock sample construction."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch

from examples.multimodal_dev.data.mock import MockQwen35VLDataset, mock_collate_fn


class TestMockDataset(unittest.TestCase):
    """Validate both record modes and text-only batch handling."""

    def _dataset(self, records, **kwargs):
        tempdir = tempfile.TemporaryDirectory()
        path = Path(tempdir.name) / "records.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        dataset = MockQwen35VLDataset(
            num_samples=len(records),
            seq_length=max(record["llm_sequence_length"] for record in records),
            image_seq_length=float("inf"),
            vision_distribution="empirical_record",
            vision_records_path=str(path),
            **kwargs,
        )
        self.addCleanup(tempdir.cleanup)
        return dataset

    def test_variable_lengths_and_both_vision_modes(self):
        dataset = self._dataset(
            [
                {"format_version": 1, "llm_sequence_length": 32, "vision_tokens_per_image": [4]},
                {"format_version": 1, "llm_sequence_length": 48, "image_sizes": [[32, 64]]},
            ]
        )

        token_sample = dataset[0]
        size_sample = dataset[1]

        self.assertEqual(token_sample["input_ids"].numel(), 32)
        self.assertEqual(size_sample["input_ids"].numel(), 48)
        self.assertEqual(int((token_sample["input_ids"] == dataset.image_token_id).sum().item()), 4)
        self.assertEqual(int((size_sample["input_ids"] == dataset.image_token_id).sum().item()), 2)
        self.assertEqual(token_sample["image_grid_thw"].tolist(), [[1, 4, 4]])
        self.assertEqual(size_sample["image_grid_thw"].tolist(), [[1, 2, 4]])
        self.assertEqual(tuple(token_sample["pixel_values"].shape), (16, 1536))
        self.assertEqual(tuple(size_sample["pixel_values"].shape), (8, 1536))
        self.assertEqual(token_sample["cu_seqlens"].tolist(), [0, 32])
        self.assertEqual(size_sample["cu_seqlens"].tolist(), [0, 48])

    def test_finite_cap_replaces_dropped_vision_with_text(self):
        dataset = self._dataset(
            [
                {
                    "format_version": 1,
                    "llm_sequence_length": 64,
                    "vision_tokens_per_image": [10, 8, 4],
                }
            ]
        )
        dataset.image_seq_length = 12

        sample = dataset[0]

        self.assertEqual(sample["input_ids"].numel(), 64)
        self.assertEqual(int((sample["input_ids"] == dataset.image_token_id).sum().item()), 10)
        self.assertEqual(sample["image_grid_thw"].shape[0], 1)

    def test_text_only_record_uses_dummy_vision_inputs(self):
        dataset = self._dataset(
            [{"format_version": 1, "llm_sequence_length": 24, "image_sizes": []}]
        )

        sample = dataset[0]

        self.assertFalse(torch.any(sample["input_ids"] == dataset.image_token_id))
        self.assertEqual(tuple(sample["pixel_values"].shape), (4, 1536))
        self.assertEqual(sample["image_grid_thw"].tolist(), [[1, 2, 2]])
        self.assertFalse(bool(sample["has_real_vision"]))
        self.assertEqual(tuple(sample["position_ids"].shape), (3, 24))
        torch.testing.assert_close(sample["position_ids"], torch.arange(24).expand(3, -1))

    def test_mixed_collate_drops_dummy_rows(self):
        dataset = self._dataset(
            [
                {"format_version": 1, "llm_sequence_length": 32, "image_sizes": []},
                {"format_version": 1, "llm_sequence_length": 32, "vision_tokens_per_image": [4]},
            ]
        )
        batch = [dataset[0], dataset[1]]

        collated = mock_collate_fn(batch)

        self.assertEqual(collated["image_grid_thw"].tolist(), [[1, 4, 4]])
        self.assertEqual(tuple(collated["pixel_values"].shape), (16, 1536))

    def test_default_fixed_mock_shape_is_unchanged(self):
        dataset = MockQwen35VLDataset(num_samples=1, seq_length=128, image_seq_length=256)
        sample = dataset[0]

        self.assertEqual(sample["input_ids"].numel(), 128)
        self.assertEqual(int((sample["input_ids"] == dataset.image_token_id).sum().item()), 98)
        self.assertEqual(sample["image_grid_thw"].tolist(), [[2, 14, 14]])
        self.assertNotIn("has_real_vision", sample)


if __name__ == "__main__":
    unittest.main()
