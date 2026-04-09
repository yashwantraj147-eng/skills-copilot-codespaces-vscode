"""Image pre-processing utilities for medicine packaging images."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np


# Target spatial resolution fed into the CNN.
DEFAULT_TARGET_SIZE: Tuple[int, int] = (64, 64)


def resize_bilinear(
    image: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Resize an HxWxC image to (target_h, target_w, C) using bilinear interpolation.

    This pure-NumPy implementation avoids a hard dependency on Pillow/OpenCV
    while producing results suitable for model inference.

    Args:
        image: uint8 or float array of shape (H, W, C).
        target_h: Desired output height in pixels.
        target_w: Desired output width in pixels.

    Returns:
        Resized image with the same dtype and channel count as the input.
    """
    src_h, src_w = image.shape[:2]
    row_idx = np.linspace(0, src_h - 1, target_h)
    col_idx = np.linspace(0, src_w - 1, target_w)

    r0 = np.floor(row_idx).astype(int).clip(0, src_h - 2)
    r1 = r0 + 1
    c0 = np.floor(col_idx).astype(int).clip(0, src_w - 2)
    c1 = c0 + 1

    dr = (row_idx - r0)[:, np.newaxis, np.newaxis]
    dc = (col_idx - c0)[np.newaxis, :, np.newaxis]

    top = image[r0][:, c0] * (1 - dc) + image[r0][:, c1] * dc
    bot = image[r1][:, c0] * (1 - dc) + image[r1][:, c1] * dc
    resized = top * (1 - dr) + bot * dr

    return resized.astype(image.dtype)


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0.0, 1.0].

    Args:
        image: uint8 array of any shape.

    Returns:
        float32 array with the same shape, values in [0, 1].
    """
    return image.astype(np.float32) / 255.0


def to_chw(image: np.ndarray) -> np.ndarray:
    """Convert an (H, W, C) array to (C, H, W) for CNN inference.

    Args:
        image: Array of shape (H, W, C).

    Returns:
        Array of shape (C, H, W).
    """
    return np.transpose(image, (2, 0, 1))


def preprocess_image(
    raw: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Full preprocessing pipeline for a raw medicine packaging image.

    Steps:
        1. Ensure the image is RGB (3 channels).
        2. Resize to *target_size* using bilinear interpolation.
        3. Normalise pixel values to [0, 1].
        4. Reorder axes to (C, H, W) for CNN input.

    Args:
        raw: uint8 NumPy array of shape (H, W) or (H, W, C).
        target_size: ``(height, width)`` tuple for the output spatial dimensions.

    Returns:
        float32 array of shape ``(3, target_size[0], target_size[1])``.

    Raises:
        ValueError: If the image has an unexpected number of channels.
    """
    # Ensure 3-D HxWxC
    if raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 1:
        raw = np.concatenate([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]  # drop alpha
    elif raw.ndim == 3 and raw.shape[2] == 3:
        pass  # already RGB
    else:
        raise ValueError(
            f"Unsupported image shape {raw.shape}. Expected (H, W), "
            "(H, W, 1), (H, W, 3), or (H, W, 4)."
        )

    resized = resize_bilinear(raw, target_size[0], target_size[1])
    normed = normalize(resized)
    chw = to_chw(normed)
    return chw


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into a NumPy uint8 RGB array.

    Tries to use Pillow when available; falls back to a minimal PNG/JPEG
    decoder stub that raises a clear error if the library is absent.

    Args:
        data: Raw image bytes (e.g. from an HTTP multipart upload).

    Returns:
        uint8 array of shape (H, W, 3).

    Raises:
        ImportError: If neither Pillow nor a compatible decoder is available.
        ValueError: If the bytes cannot be decoded as a supported image format.
    """
    try:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to decode image bytes. "
            "Install it with: pip install Pillow"
        ) from exc
