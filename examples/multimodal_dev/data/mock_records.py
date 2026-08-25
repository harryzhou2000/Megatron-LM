# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public record helpers for empirical multimodal mock data.

The JSONL v1 contract stores an LLM-side sequence length and exactly one
vision representation: post-merge token counts or source image dimensions.
This module intentionally has no Torch or Megatron dependencies.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VisionRecord:
    """One public v1 mock-vision record."""

    llm_sequence_length: int
    vision_tokens_per_image: tuple[int, ...] | None
    image_sizes: tuple[tuple[int, int], ...] | None
    source: str


def load_vision_records(records_path: str | Path) -> list[VisionRecord]:
    """Load v1 records from a JSONL file or a flat directory of shards."""

    path = Path(records_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Mock vision records path does not exist: {path}")

    if path.is_file():
        record_files = [path]
    else:
        record_files = sorted(path.glob("*.jsonl"))
    if not record_files:
        raise ValueError(f"No JSONL record files found at: {path}")

    records: list[VisionRecord] = []
    for record_file in record_files:
        if record_file.suffix != ".jsonl":
            raise ValueError(f"Mock vision records must be JSONL, got: {record_file}")
        for source, raw_record in _read_jsonl(record_file):
            records.append(_parse_record(raw_record, source))

    if not records:
        raise ValueError(f"No mock vision records found at: {path}")
    return records


def select_vision_record(
    records: Sequence[VisionRecord],
    sample_index: int,
    sampling: str = "cycle",
    random_seed: int = 1234,
) -> VisionRecord:
    """Select a deterministic record for a virtual dataset index."""

    if not records:
        raise ValueError("Cannot select a mock vision record from an empty record set")
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")
    if sampling == "cycle":
        return records[sample_index % len(records)]
    if sampling == "with_replacement":
        return records[random.Random(random_seed + sample_index).randrange(len(records))]
    raise ValueError(
        "mock vision record sampling must be 'cycle' or 'with_replacement', " f"got {sampling!r}"
    )


def record_to_grids(
    record: VisionRecord,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """Resolve either public record representation into Qwen image grids."""

    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
    if not isinstance(temporal_patch_size, int) or temporal_patch_size <= 0:
        raise ValueError(
            "temporal_patch_size must be a positive integer, " f"got {temporal_patch_size!r}"
        )
    if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be a positive integer, got {spatial_merge_size!r}"
        )

    if record.vision_tokens_per_image is not None:
        token_counts = record.vision_tokens_per_image
        grids = tuple(
            closest_square_grid(token_count, spatial_merge_size) for token_count in token_counts
        )
    else:
        assert record.image_sizes is not None
        alignment = patch_size * spatial_merge_size
        grids_list: list[tuple[int, int, int]] = []
        token_counts_list: list[int] = []
        for image_index, (height, width) in enumerate(record.image_sizes):
            if height % alignment != 0 or width % alignment != 0:
                raise ValueError(
                    f"{record.source} image_sizes[{image_index}] dimensions must be divisible "
                    f"by patch_size * spatial_merge_size ({alignment}), got [{height}, {width}]"
                )
            # A still image has one temporal grid cell. Each flattened patch
            # independently contains ``temporal_patch_size`` duplicated frames.
            grid = (1, height // patch_size, width // patch_size)
            grids_list.append(grid)
            token_counts_list.append(math.prod(grid) // (spatial_merge_size**2))
        grids = tuple(grids_list)
        token_counts = tuple(token_counts_list)

    vision_footprint = sum(token_counts) + len(token_counts)
    if vision_footprint >= record.llm_sequence_length:
        raise ValueError(
            f"{record.source} vision payload needs {vision_footprint} LLM positions, but "
            f"llm_sequence_length is {record.llm_sequence_length}; at least one text token "
            "is required"
        )
    return grids, token_counts


def cap_vision_layouts(
    grids: Sequence[tuple[int, int, int]],
    token_counts: Sequence[int],
    image_seq_length: int | float,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """Drop whole images in record order when an explicit token cap is set."""

    if len(grids) != len(token_counts):
        raise ValueError(
            f"grids and token_counts must have equal lengths, got {len(grids)} and "
            f"{len(token_counts)}"
        )
    if math.isinf(image_seq_length):
        return tuple(grids), tuple(token_counts)
    if image_seq_length < 0:
        raise ValueError(f"image_seq_length must be non-negative, got {image_seq_length}")

    kept_grids: list[tuple[int, int, int]] = []
    kept_counts: list[int] = []
    total = 0
    for grid, token_count in zip(grids, token_counts):
        if total + token_count > image_seq_length:
            continue
        kept_grids.append(grid)
        kept_counts.append(token_count)
        total += token_count
    return tuple(kept_grids), tuple(kept_counts)


@lru_cache(maxsize=None)
def closest_square_grid(
    vision_token_count: int, spatial_merge_size: int = 2
) -> tuple[int, int, int]:
    """Construct a closest-to-square grid preserving a post-merge token count."""

    if not isinstance(vision_token_count, int) or isinstance(vision_token_count, bool):
        raise ValueError(
            f"vision_token_count must be a positive integer, got {vision_token_count!r}"
        )
    if vision_token_count <= 0:
        raise ValueError(f"vision_token_count must be positive, got {vision_token_count}")
    if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be a positive integer, got {spatial_merge_size!r}"
        )

    height_factor = 1
    for candidate in range(1, math.isqrt(vision_token_count) + 1):
        if vision_token_count % candidate == 0:
            height_factor = candidate
    width_factor = vision_token_count // height_factor
    return (1, spatial_merge_size * height_factor, spatial_merge_size * width_factor)


def _read_jsonl(path: Path) -> Iterable[tuple[str, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            source = f"{path}:{line_number}"
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{source} must be a JSON object")
            yield source, value


def _parse_record(record: Mapping[str, Any], source: str) -> VisionRecord:
    version = _positive_int(record.get("format_version"), f"{source} format_version")
    if version != 1:
        raise ValueError(f"{source} format_version must be 1, got {version}")
    sequence_length = _positive_int(
        record.get("llm_sequence_length"), f"{source} llm_sequence_length"
    )

    has_tokens = "vision_tokens_per_image" in record
    has_sizes = "image_sizes" in record
    if has_tokens == has_sizes:
        raise ValueError(
            f"{source} must contain exactly one of vision_tokens_per_image or image_sizes"
        )

    token_counts: tuple[int, ...] | None = None
    image_sizes: tuple[tuple[int, int], ...] | None = None
    if has_tokens:
        values = _sequence(record["vision_tokens_per_image"], f"{source} vision_tokens_per_image")
        token_counts = tuple(
            _positive_int(value, f"{source} vision_tokens_per_image[{index}]")
            for index, value in enumerate(values)
        )
        if sum(token_counts) + len(token_counts) >= sequence_length:
            raise ValueError(
                f"{source} vision payload does not leave at least one text token within "
                f"llm_sequence_length={sequence_length}"
            )
    else:
        values = _sequence(record["image_sizes"], f"{source} image_sizes")
        parsed_sizes: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            pair = _sequence(value, f"{source} image_sizes[{index}]")
            if len(pair) != 2:
                raise ValueError(f"{source} image_sizes[{index}] must be [height, width]")
            parsed_sizes.append(
                (
                    _positive_int(pair[0], f"{source} image_sizes[{index}][0]"),
                    _positive_int(pair[1], f"{source} image_sizes[{index}][1]"),
                )
            )
        image_sizes = tuple(parsed_sizes)

    return VisionRecord(sequence_length, token_counts, image_sizes, source)


def _sequence(value: Any, description: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{description} must be a JSON array")
    return value


def _positive_int(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer, got {value!r}")
    return value
