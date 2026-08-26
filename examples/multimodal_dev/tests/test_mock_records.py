# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the public mock-vision JSONL v1 contract."""

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
    cap_vision_layouts,
    closest_square_grid,
    load_vision_records,
    record_to_grids,
    select_vision_record,
)


class TestMockRecords(unittest.TestCase):
    """Validate parsing, grid conversion, and deterministic selection."""

    def _load(self, records):
        tempdir = tempfile.TemporaryDirectory()
        path = Path(tempdir.name) / "records.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return tempdir, load_vision_records(path)

    def test_loads_mutually_exclusive_record_modes(self):
        tempdir, records = self._load(
            [
                {"format_version": 1, "llm_sequence_length": 32, "vision_tokens_per_image": [4, 6]},
                {
                    "format_version": 1,
                    "llm_sequence_length": 48,
                    "image_sizes": [[32, 64], [64, 32]],
                },
                {"format_version": 1, "llm_sequence_length": 16, "image_sizes": []},
            ]
        )
        self.addCleanup(tempdir.cleanup)

        self.assertEqual(records[0].vision_tokens_per_image, (4, 6))
        self.assertIsNone(records[0].image_sizes)
        self.assertEqual(records[1].image_sizes, ((32, 64), (64, 32)))
        self.assertIsNone(records[1].vision_tokens_per_image)
        self.assertEqual(records[2].image_sizes, ())

    def test_rejects_missing_or_mixed_payloads(self):
        cases = [
            {"format_version": 1, "llm_sequence_length": 32},
            {
                "format_version": 1,
                "llm_sequence_length": 32,
                "vision_tokens_per_image": [],
                "image_sizes": [],
            },
        ]
        for record in cases:
            with self.subTest(record=record):
                tempdir = tempfile.TemporaryDirectory()
                self.addCleanup(tempdir.cleanup)
                path = Path(tempdir.name) / "records.jsonl"
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    load_vision_records(path)

    def test_rejects_invalid_version_types_and_capacity(self):
        cases = [
            ({"format_version": 2, "llm_sequence_length": 32, "image_sizes": []}, "must be 1"),
            (
                {"format_version": 1, "llm_sequence_length": 8, "vision_tokens_per_image": [7]},
                "at least one text token",
            ),
            (
                {"format_version": 1, "llm_sequence_length": 32, "vision_tokens_per_image": [True]},
                "positive integer",
            ),
        ]
        for record, message in cases:
            with self.subTest(record=record):
                tempdir = tempfile.TemporaryDirectory()
                self.addCleanup(tempdir.cleanup)
                path = Path(tempdir.name) / "records.jsonl"
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_vision_records(path)

    def test_directory_order_and_selection_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, length in (("b.jsonl", 24), ("a.jsonl", 16)):
                (root / name).write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "llm_sequence_length": length,
                            "vision_tokens_per_image": [],
                        }
                    ),
                    encoding="utf-8",
                )
            records = load_vision_records(root)

        self.assertEqual([record.llm_sequence_length for record in records], [16, 24])
        self.assertEqual(select_vision_record(records, 2).llm_sequence_length, 16)
        self.assertEqual(
            select_vision_record(records, 17, "with_replacement", 9),
            select_vision_record(records, 17, "with_replacement", 9),
        )

    def test_token_counts_preserve_arbitrary_layout_sizes(self):
        tempdir, records = self._load(
            [
                {
                    "format_version": 1,
                    "llm_sequence_length": 2048,
                    "vision_tokens_per_image": [1, 600, 638],
                }
            ]
        )
        self.addCleanup(tempdir.cleanup)

        grids, token_counts = record_to_grids(records[0])
        self.assertEqual(grids, ((1, 2, 2), (1, 48, 50), (1, 44, 58)))
        self.assertEqual(token_counts, (1, 600, 638))
        for grid, token_count in zip(grids, token_counts):
            self.assertEqual(math.prod(grid) // 4, token_count)

    def test_image_sizes_preserve_rectangular_geometry(self):
        tempdir, records = self._load(
            [{"format_version": 1, "llm_sequence_length": 64, "image_sizes": [[32, 64], [64, 32]]}]
        )
        self.addCleanup(tempdir.cleanup)

        grids, token_counts = record_to_grids(records[0])
        self.assertEqual(grids, ((1, 2, 4), (1, 4, 2)))
        self.assertEqual(token_counts, (2, 2))

    def test_image_sizes_require_patch_merge_alignment(self):
        tempdir, records = self._load(
            [{"format_version": 1, "llm_sequence_length": 64, "image_sizes": [[48, 64]]}]
        )
        self.addCleanup(tempdir.cleanup)

        with self.assertRaisesRegex(ValueError, "divisible"):
            record_to_grids(records[0])

    def test_cap_drops_whole_images_without_changing_order(self):
        grids = ((1, 20, 20), (1, 24, 24), (1, 8, 8))
        capped_grids, capped_counts = cap_vision_layouts(grids, (100, 144, 16), 120)
        self.assertEqual(capped_grids, (grids[0], grids[2]))
        self.assertEqual(capped_counts, (100, 16))

    def test_parse_image_seq_length(self):
        self.assertTrue(math.isinf(parse_image_seq_length("inf")))
        self.assertEqual(parse_image_seq_length("2048"), 2048)
        with self.assertRaisesRegex(Exception, "non-negative"):
            parse_image_seq_length("-1")

    def test_closest_square_grid_rejects_boolean(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            closest_square_grid(True)


if __name__ == "__main__":
    unittest.main()
