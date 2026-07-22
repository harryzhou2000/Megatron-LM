# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for dependency-free empirical mock-record helpers."""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.arguments import parse_image_seq_length
from examples.multimodal_dev.data.mock_records import (
    cap_visual_token_counts,
    closest_square_grid,
    load_vision_records,
    select_vision_record,
)


class TestMockRecords(unittest.TestCase):
    """Validate record parsing and square-grid synthesis without MCore imports."""

    def test_loads_canonical_and_alternate_record_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            records_path.write_text(
                "\n".join(
                    [
                        json.dumps({"num_imgs": 2, "tok_per_image_arr": [1, 638]}),
                        json.dumps({"num_images": 1, "vision_tokens_per_image": [600]}),
                        json.dumps({"image_count": 2, "images": [{"tokens": 4}, {"tokens": 9}]}),
                        json.dumps({"n_images": 0, "vision_token_counts": []}),
                    ]
                )
            )

            records = load_vision_records(records_path)

        self.assertEqual(
            [record.visual_token_counts for record in records],
            [(1, 638), (600,), (4, 9), ()],
        )

    def test_loads_json_container_and_ignores_directory_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "summary.json").write_text(json.dumps({"samples_with_images": 2}))
            (root / "records.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {"num_images": 1, "tokens_per_image": [48]},
                            {"num_images": 0, "tokens_per_image": []},
                        ]
                    }
                )
            )

            records = load_vision_records(root)

        self.assertEqual([record.visual_token_counts for record in records], [(48,), ()])

    def test_rejects_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            records_path.write_text(json.dumps({"num_images": 2, "tokens_per_image": [64]}) + "\n")

            with self.assertRaisesRegex(ValueError, "image count is 2"):
                load_vision_records(records_path)

    def test_selects_exact_cycle_or_deterministic_replacement_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            records_path.write_text(
                "\n".join(
                    json.dumps({"num_imgs": 1, "tok_per_image_arr": [count]})
                    for count in (4, 8, 12)
                )
            )
            records = load_vision_records(records_path)

        self.assertEqual(
            [select_vision_record(records, index).visual_token_counts for index in range(5)],
            [(4,), (8,), (12,), (4,), (8,)],
        )
        self.assertEqual(
            select_vision_record(records, 17, "with_replacement", 9),
            select_vision_record(records, 17, "with_replacement", 9),
        )

    def test_closest_square_grid_preserves_visual_token_count(self):
        for token_count, expected_grid in ((1, (1, 2, 2)), (600, (1, 48, 50)), (638, (1, 44, 58))):
            grid = closest_square_grid(token_count)
            self.assertEqual(grid, expected_grid)
            self.assertEqual(grid[0] * grid[1] * grid[2] // 4, token_count)

    def test_parse_image_seq_length_accepts_infinity(self):
        self.assertTrue(math.isinf(parse_image_seq_length("inf")))
        self.assertEqual(parse_image_seq_length("2048"), 2048)

    def test_image_seq_length_cap_remains_effective_for_record_layouts(self):
        token_counts = (600, 638, 540)
        self.assertEqual(cap_visual_token_counts(token_counts, 1000), (600,))
        self.assertEqual(cap_visual_token_counts(token_counts, math.inf), token_counts)


if __name__ == "__main__":
    unittest.main()
