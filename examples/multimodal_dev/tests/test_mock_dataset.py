# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for Qwen3.5-VL mock dataset sample construction."""

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

from examples.multimodal_dev.data.mock import MockQwen35VLDataset


class TestMockDataset(unittest.TestCase):
    """Validate mock dataset edge cases."""

    def test_empirical_text_only_record_uses_dummy_vision_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            records_path.write_text(json.dumps({"num_images": 0, "tokens_per_image": []}) + "\n")
            dataset = MockQwen35VLDataset(
                num_samples=1,
                seq_length=32,
                image_seq_length=float("inf"),
                vision_distribution="empirical_record",
                vision_records_path=str(records_path),
            )

            sample = dataset[0]

        self.assertFalse(torch.any(sample["input_ids"] == dataset.image_token_id))
        self.assertEqual(tuple(sample["pixel_values"].shape), (4, 1536))
        self.assertEqual(sample["image_grid_thw"].tolist(), [[1, 2, 2]])
        self.assertEqual(tuple(sample["position_ids"].shape), (3, 32))
        torch.testing.assert_close(sample["position_ids"], torch.arange(32).expand(3, -1))


if __name__ == "__main__":
    unittest.main()
