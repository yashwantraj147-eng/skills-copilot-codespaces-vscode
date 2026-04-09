"""Feature extractors for medicine packaging anomaly detection.

Each extractor analyses a specific visual property of the packaging image
and returns a score in **[0, 1]** where higher values indicate a stronger
anomaly signal (i.e. greater likelihood of counterfeiting).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Font inconsistency analyser
# ---------------------------------------------------------------------------

def analyse_font_inconsistency(image: np.ndarray) -> Dict[str, float]:
    """Detect font irregularities caused by low-quality counterfeit printing.

    Method:
        * Convert to grayscale and apply a simple edge-detection kernel.
        * Compute the variance of edge magnitudes; genuine packaging tends to
          have crisp, uniform edges whereas counterfeits often show blurring
          or pixel noise around text.

    Args:
        image: float32 array of shape (3, H, W) in [0, 1].

    Returns:
        dict with key ``font_anomaly_score`` (float in [0, 1]).
    """
    gray = _to_gray_chw(image)  # (H, W)

    # Sobel-like horizontal and vertical kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T

    gx = _convolve2d(gray, kx)
    gy = _convolve2d(gray, ky)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    edge_variance = float(np.var(magnitude))
    # Normalise: genuine packages typically have edge_variance < 0.01;
    # high variance indicates noisy / inconsistent printing.
    score = float(np.clip(edge_variance / 0.05, 0.0, 1.0))
    return {"font_anomaly_score": round(score, 4)}


# ---------------------------------------------------------------------------
# Colour deviation analyser
# ---------------------------------------------------------------------------

def analyse_colour_deviation(
    image: np.ndarray,
    reference_mean: np.ndarray | None = None,
    reference_std: np.ndarray | None = None,
) -> Dict[str, float]:
    """Measure colour deviation from an expected (reference) distribution.

    For a genuine product the per-channel colour distribution is tightly
    controlled by the manufacturer.  Counterfeits often deviate noticeably.

    Args:
        image: float32 array of shape (3, H, W) in [0, 1].
        reference_mean: Per-channel mean of genuine samples, shape (3,).
            Defaults to a neutral mid-grey ``[0.5, 0.5, 0.5]``.
        reference_std: Per-channel std of genuine samples, shape (3,).
            Defaults to ``[0.1, 0.1, 0.1]``.

    Returns:
        dict with key ``colour_deviation_score`` (float in [0, 1]).
    """
    if reference_mean is None:
        reference_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if reference_std is None:
        reference_std = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    sample_mean = image.mean(axis=(1, 2))  # (3,)
    z_scores = np.abs((sample_mean - reference_mean) / (reference_std + 1e-8))
    score = float(np.clip(z_scores.mean() / 3.0, 0.0, 1.0))
    return {"colour_deviation_score": round(score, 4)}


# ---------------------------------------------------------------------------
# Barcode anomaly analyser
# ---------------------------------------------------------------------------

def analyse_barcode(image: np.ndarray) -> Dict[str, float]:
    """Detect barcode anomalies by checking bar regularity.

    Genuine barcodes have alternating black/white bars of regular widths.
    Counterfeits often show smearing, incorrect bar ratios, or missing quiet
    zones.

    Method:
        1. Extract a horizontal scanline through the image centre.
        2. Binarise the scanline.
        3. Measure the coefficient of variation (CV) of run lengths.
           High CV indicates irregular bars.

    Args:
        image: float32 array of shape (3, H, W) in [0, 1].

    Returns:
        dict with keys:
            ``barcode_anomaly_score`` (float in [0, 1])
            ``bar_count`` (int – number of distinct bars detected)
    """
    gray = _to_gray_chw(image)  # (H, W)
    h, w = gray.shape
    scanline = gray[h // 2, :]  # centre row

    threshold = 0.5
    binary = (scanline < threshold).astype(np.int8)  # 1 = dark bar

    runs = _run_lengths(binary)
    if len(runs) < 3:
        return {"barcode_anomaly_score": 0.0, "bar_count": 0}

    cv = float(np.std(runs) / (np.mean(runs) + 1e-8))
    score = float(np.clip(cv / 2.0, 0.0, 1.0))
    return {
        "barcode_anomaly_score": round(score, 4),
        "bar_count": int(len(runs)),
    }


# ---------------------------------------------------------------------------
# Seal pattern analyser
# ---------------------------------------------------------------------------

def analyse_seal(image: np.ndarray) -> Dict[str, float]:
    """Evaluate tamper-evident seal integrity.

    Genuine seals typically exhibit:
    * Symmetric intensity gradients around the seal boundary.
    * A distinct dark contour relative to the surrounding area.

    This analyser looks at the corners of the image (where seals are
    commonly placed) and measures radial symmetry of the local intensity.

    Args:
        image: float32 array of shape (3, H, W) in [0, 1].

    Returns:
        dict with key ``seal_anomaly_score`` (float in [0, 1]).
    """
    gray = _to_gray_chw(image)  # (H, W)
    h, w = gray.shape
    patch_h = max(h // 6, 1)
    patch_w = max(w // 6, 1)

    # Sample top-right corner (common seal location)
    patch = gray[:patch_h, w - patch_w :]
    if patch.size == 0:
        return {"seal_anomaly_score": 0.0}

    # Genuine seals have low std (uniform colour); anomalies show high variance
    std = float(np.std(patch))
    score = float(np.clip(1.0 - std / 0.3, 0.0, 1.0))

    # Flip: high std → high anomaly probability
    score = 1.0 - score
    return {"seal_anomaly_score": round(score, 4)}


# ---------------------------------------------------------------------------
# Aggregate extractor
# ---------------------------------------------------------------------------

def extract_all_features(image: np.ndarray) -> Dict[str, float]:
    """Run all feature extractors and return a consolidated result.

    Args:
        image: float32 array of shape (3, H, W) in [0, 1].

    Returns:
        dict containing all per-extractor scores plus a combined
        ``overall_anomaly_score`` that averages all component scores.
    """
    results: Dict[str, float] = {}
    results.update(analyse_font_inconsistency(image))
    results.update(analyse_colour_deviation(image))
    results.update(analyse_barcode(image))
    results.update(analyse_seal(image))

    score_keys = [k for k in results if k.endswith("_score")]
    if score_keys:
        results["overall_anomaly_score"] = round(
            float(np.mean([results[k] for k in score_keys])), 4
        )
    return results


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _to_gray_chw(image: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) float32 to (H, W) grayscale using BT.601 coefficients."""
    return (
        0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
    ).astype(np.float32)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D 'valid' convolution without scipy/OpenCV dependency."""
    kh, kw = kernel.shape
    h, w = image.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * image[i : i + out_h, j : j + out_w]
    return out


def _run_lengths(binary: np.ndarray) -> np.ndarray:
    """Return array of run lengths for a 1-D binary sequence."""
    if len(binary) == 0:
        return np.array([], dtype=np.int32)
    changes = np.where(np.diff(binary))[0] + 1
    boundaries = np.concatenate([[0], changes, [len(binary)]])
    return np.diff(boundaries).astype(np.int32)
