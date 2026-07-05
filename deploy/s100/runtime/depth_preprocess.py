"""Depth preprocessing equivalent to the Isaac Gym training pipeline."""

from typing import Tuple

import numpy as np


def preprocess_depth_meters(
    depth_meters: np.ndarray,
    near_clip: float = 0.0,
    far_clip: float = 2.0,
    output_size: Tuple[int, int] = (58, 87),
) -> np.ndarray:
    """Crop, bicubic-resize and normalize a positive depth map in metres.

    The checkpoint used a 60x106 camera image, removed the bottom two rows and
    four columns at each side, then resized to 58x87. Hardware camera
    registration must produce the same geometry before calling this function.
    """

    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.shape != (60, 106):
        raise ValueError(f"raw depth must have shape (60, 106), got {depth.shape}")
    if not np.all(np.isfinite(depth)):
        raise ValueError("raw depth contains NaN or Inf")
    if not far_clip > near_clip:
        raise ValueError("far_clip must be greater than near_clip")

    cropped = np.clip(depth[:-2, 4:-4], near_clip, far_clip)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for depth resizing") from exc
    height, width = output_size
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = np.asarray(
        Image.fromarray(cropped, mode="F").resize(
            (width, height),
            resample=resampling,
        ),
        dtype=np.float32,
    )
    normalized = (resized - near_clip) / (far_clip - near_clip) - 0.5
    return np.ascontiguousarray(normalized, dtype=np.float32)
