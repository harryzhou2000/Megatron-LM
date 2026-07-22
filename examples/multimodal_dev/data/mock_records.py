# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dependency-free helpers for empirical multimodal mock records.

The production mock dataset uses these helpers to replay observed visual-token
layouts while generating synthetic pixels.  Keeping parsing and grid synthesis
free of Torch and Megatron imports makes the input contract independently
testable on a CPU-only Python installation.
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


_VISION_TOKEN_KEYS = (
    "tok_per_image_arr",
    "vision_tokens_per_image",
    "image_token_counts",
    "image_tokens",
    "vision_token_counts",
    "tokens_per_image",
)
_IMAGE_COUNT_KEYS = ("num_imgs", "num_images", "image_count", "n_images")
_TOTAL_VISION_TOKEN_KEYS = (
    "total_vision_tokens",
    "num_vision_tokens",
    "vision_token_count",
)
_RECORD_CONTAINER_KEYS = ("records", "samples", "data", "items")
_SUMMARY_FILENAMES = {"summary.json", "stats.json", "metadata.json"}
_IMAGE_ITEM_TOKEN_KEYS = (
    "token_count",
    "vision_tokens",
    "image_tokens",
    "tokens",
)


@dataclass(frozen=True)
class VisionRecord:
    """One empirical visual-token layout.

    Attributes:
        visual_token_counts: Post-merge visual-token count for each image.
        source: Human-readable source location used in validation errors.
    """

    visual_token_counts: tuple[int, ...]
    source: str


def load_vision_records(records_path: str | Path) -> list[VisionRecord]:
    """Load visual-token records from JSON/JSONL files or a directory.

    The canonical format is one JSON object per JSONL line with
    ``tok_per_image_arr``.  Equivalent alternate names are accepted so other
    statistics exports can be replayed without a format-specific loader.

    Args:
        records_path: A JSON/JSONL file or directory containing record files.

    Returns:
        Validated visual-token records in deterministic file/line order.

    Raises:
        FileNotFoundError: If ``records_path`` does not exist.
        ValueError: If no records are found or an input record is invalid.
    """

    path = Path(records_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Mock vision records path does not exist: {path}")

    record_files = _find_record_files(path)
    if not record_files:
        raise ValueError(f"No JSON or JSONL record files found at: {path}")

    records: list[VisionRecord] = []
    for record_file in record_files:
        for source, raw_record in _read_record_file(record_file):
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
    """Select a deterministic empirical record for a virtual sample index.

    ``cycle`` gives exact record frequencies per complete cycle; the Megatron
    data sampler can then shuffle sample indices.  ``with_replacement`` gives
    independent deterministic draws for callers that intentionally want a
    replacement stream.

    Args:
        records: Non-empty record sequence returned by :func:`load_vision_records`.
        sample_index: Virtual dataset index.
        sampling: ``"cycle"`` or ``"with_replacement"``.
        random_seed: Base seed for replacement sampling.

    Returns:
        The selected record.

    Raises:
        ValueError: If inputs are invalid or the sampling mode is unknown.
    """

    if not records:
        raise ValueError("Cannot select a mock vision record from an empty record set")
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")
    if sampling == "cycle":
        return records[sample_index % len(records)]
    if sampling == "with_replacement":
        return records[random.Random(random_seed + sample_index).randrange(len(records))]
    raise ValueError(
        "mock vision record sampling must be 'cycle' or 'with_replacement', "
        f"got {sampling!r}"
    )


def cap_visual_token_counts(
    visual_token_counts: Sequence[int],
    image_seq_length: int | float,
) -> tuple[int, ...]:
    """Apply the mock's per-sample visual-token cap to image token counts.

    Images that would exceed the cap are skipped, matching the legacy
    variable-image mock behavior.  ``inf`` (or a negative cap) retains every
    image.  The language-model total sequence cap is enforced separately by
    the dataset because it also accounts for vision-start tokens.

    Args:
        visual_token_counts: Post-merge visual-token count for each image.
        image_seq_length: Maximum total visual tokens, or ``inf``.

    Returns:
        Kept visual-token counts in their original order.
    """

    if image_seq_length < 0 or math.isinf(image_seq_length):
        return tuple(visual_token_counts)

    kept: list[int] = []
    total = 0
    for token_count in visual_token_counts:
        if total + token_count > image_seq_length:
            continue
        kept.append(token_count)
        total += token_count
    return tuple(kept)


@lru_cache(maxsize=None)
def closest_square_grid(
    visual_token_count: int,
    spatial_merge_size: int = 2,
) -> tuple[int, int, int]:
    """Build a valid, closest-to-square ``(T, H, W)`` patch grid.

    The Qwen vision merger emits ``T * H * W / merge_size**2`` LLM visual
    tokens.  Statistics exports generally retain only that scalar, not source
    geometry.  This deterministic reconstruction uses ``T=1`` and factors the
    token count into the closest possible spatial rectangle.  ``H`` and ``W``
    are both multiples of the spatial merge size, which keeps mRoPE and the
    patch merger consistent.

    Args:
        visual_token_count: Positive post-merge LLM visual-token count.
        spatial_merge_size: Spatial patch merge factor.

    Returns:
        ``(T, H, W)`` in pre-merge patch-grid units.

    Raises:
        ValueError: If inputs are not positive integers.
    """

    if not isinstance(visual_token_count, int) or visual_token_count <= 0:
        raise ValueError(
            "visual_token_count must be a positive integer, "
            f"got {visual_token_count!r}"
        )
    if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
        raise ValueError(
            "spatial_merge_size must be a positive integer, "
            f"got {spatial_merge_size!r}"
        )

    height_factor = 1
    for candidate in range(1, math.isqrt(visual_token_count) + 1):
        if visual_token_count % candidate == 0:
            height_factor = candidate
    width_factor = visual_token_count // height_factor
    return (1, spatial_merge_size * height_factor, spatial_merge_size * width_factor)


def _find_record_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    jsonl_files = sorted(path.glob("*.jsonl"))
    if jsonl_files:
        return jsonl_files
    return sorted(
        record_file
        for record_file in path.glob("*.json")
        if record_file.name not in _SUMMARY_FILENAMES
    )


def _read_record_file(path: Path) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                yield f"{path}:{line_number}", _require_mapping(record, f"{path}:{line_number}")
        return

    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        for item_number, record in enumerate(_records_from_json_payload(payload, str(path)), start=1):
            yield f"{path}:{item_number}", record
        return

    raise ValueError(f"Mock vision records must be JSON or JSONL, got: {path}")


def _records_from_json_payload(payload: Any, source: str) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [_require_mapping(record, f"{source}[{index}]") for index, record in enumerate(payload)]
    mapping = _require_mapping(payload, source)
    if _first_present(mapping, _VISION_TOKEN_KEYS + _IMAGE_COUNT_KEYS) is not None:
        return [mapping]
    for key in _RECORD_CONTAINER_KEYS:
        if key in mapping:
            container = mapping[key]
            if not isinstance(container, list):
                raise ValueError(f"{source}.{key} must be a list of records")
            return [
                _require_mapping(record, f"{source}.{key}[{index}]")
                for index, record in enumerate(container)
            ]
    raise ValueError(
        f"{source} does not contain a record or one of {_RECORD_CONTAINER_KEYS}"
    )


def _parse_record(record: Mapping[str, Any], source: str) -> VisionRecord:
    token_value = _first_present(record, _VISION_TOKEN_KEYS)
    if token_value is None and "images" in record:
        token_value = _tokens_from_images(record["images"], source)

    image_count_value = _first_present(record, _IMAGE_COUNT_KEYS)
    expected_image_count = (
        _as_nonnegative_int(image_count_value, f"{source} image count")
        if image_count_value is not None
        else None
    )

    if token_value is None:
        total_token_value = _first_present(record, _TOTAL_VISION_TOKEN_KEYS)
        if expected_image_count == 0 and total_token_value is None:
            token_counts: tuple[int, ...] = ()
        elif expected_image_count == 1 and total_token_value is not None:
            token_counts = (_as_positive_int(total_token_value, f"{source} total vision tokens"),)
        else:
            raise ValueError(
                f"{source} is missing per-image vision-token counts; expected one of "
                f"{_VISION_TOKEN_KEYS}"
            )
    else:
        token_counts = tuple(
            _as_positive_int(value, f"{source} visual token count")
            for value in _as_sequence(token_value, f"{source} vision-token counts")
        )

    if expected_image_count is not None and expected_image_count != len(token_counts):
        raise ValueError(
            f"{source} image count is {expected_image_count}, but found "
            f"{len(token_counts)} per-image vision-token counts"
        )
    return VisionRecord(visual_token_counts=token_counts, source=source)


def _tokens_from_images(images: Any, source: str) -> list[Any]:
    image_items = _as_sequence(images, f"{source}.images")
    values: list[Any] = []
    for image_index, image in enumerate(image_items):
        if isinstance(image, Mapping):
            value = _first_present(image, _IMAGE_ITEM_TOKEN_KEYS)
            if value is None:
                raise ValueError(
                    f"{source}.images[{image_index}] is missing a vision-token count"
                )
            values.append(value)
        else:
            values.append(image)
    return values


def _as_sequence(value: Any, description: str) -> Sequence[Any]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return value
    raise ValueError(f"{description} must be a sequence or comma-separated string")


def _as_positive_int(value: Any, description: str) -> int:
    parsed = _as_int(value, description)
    if parsed <= 0:
        raise ValueError(f"{description} must be positive, got {parsed}")
    return parsed


def _as_nonnegative_int(value: Any, description: str) -> int:
    parsed = _as_int(value, description)
    if parsed < 0:
        raise ValueError(f"{description} must be non-negative, got {parsed}")
    return parsed


def _as_int(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be an integer, got {value!r}") from exc
    if parsed != value and not (isinstance(value, str) and str(parsed) == value.strip()):
        raise ValueError(f"{description} must be an integer, got {value!r}")
    return parsed


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _require_mapping(value: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be a JSON object")
    return value
