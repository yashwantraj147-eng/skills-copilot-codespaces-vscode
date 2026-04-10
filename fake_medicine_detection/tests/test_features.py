"""Unit tests for the feature extraction module."""

import numpy as np
import pytest

from fake_medicine_detection.features import (
    _run_lengths,
    analyse_barcode,
    analyse_colour_deviation,
    analyse_font_inconsistency,
    analyse_seal,
    extract_all_features,
)


def _make_image(h: int = 64, w: int = 64, fill: float = 0.5) -> np.ndarray:
    """Return a uniform float32 CHW image."""
    return np.full((3, h, w), fill, dtype=np.float32)


def _make_barcode_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return an image with alternating vertical bars (simulates a barcode)."""
    img = np.zeros((3, h, w), dtype=np.float32)
    for c in range(3):
        for j in range(w):
            img[c, :, j] = 0.0 if j % 4 < 2 else 1.0
    return img


class TestFontInconsistency:
    def test_returns_expected_key(self):
        img = _make_image()
        result = analyse_font_inconsistency(img)
        assert "font_anomaly_score" in result

    def test_uniform_image_low_score(self):
        img = _make_image()
        result = analyse_font_inconsistency(img)
        assert result["font_anomaly_score"] == pytest.approx(0.0, abs=1e-4)

    def test_noisy_image_higher_score(self):
        rng = np.random.default_rng(1)
        img = rng.random((3, 64, 64)).astype(np.float32)
        result = analyse_font_inconsistency(img)
        uniform = analyse_font_inconsistency(_make_image())
        assert result["font_anomaly_score"] > uniform["font_anomaly_score"]

    def test_score_in_range(self):
        rng = np.random.default_rng(2)
        img = rng.random((3, 32, 32)).astype(np.float32)
        score = analyse_font_inconsistency(img)["font_anomaly_score"]
        assert 0.0 <= score <= 1.0


class TestColourDeviation:
    def test_returns_expected_key(self):
        img = _make_image()
        result = analyse_colour_deviation(img)
        assert "colour_deviation_score" in result

    def test_mid_grey_image_zero_deviation(self):
        img = _make_image(fill=0.5)
        result = analyse_colour_deviation(img)
        assert result["colour_deviation_score"] == pytest.approx(0.0, abs=1e-4)

    def test_custom_reference_increases_deviation(self):
        img = _make_image(fill=0.9)
        ref_mean = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        result = analyse_colour_deviation(img, reference_mean=ref_mean)
        assert result["colour_deviation_score"] > 0.0

    def test_score_in_range(self):
        img = _make_image(fill=0.9)
        score = analyse_colour_deviation(img)["colour_deviation_score"]
        assert 0.0 <= score <= 1.0


class TestBarcodeAnalyser:
    def test_returns_expected_keys(self):
        img = _make_barcode_image()
        result = analyse_barcode(img)
        assert "barcode_anomaly_score" in result
        assert "bar_count" in result

    def test_uniform_image_zero_bars(self):
        img = _make_image()
        result = analyse_barcode(img)
        assert result["bar_count"] == 0

    def test_barcode_image_detects_bars(self):
        img = _make_barcode_image()
        result = analyse_barcode(img)
        assert result["bar_count"] > 0

    def test_score_in_range(self):
        img = _make_barcode_image()
        score = analyse_barcode(img)["barcode_anomaly_score"]
        assert 0.0 <= score <= 1.0


class TestSealAnalyser:
    def test_returns_expected_key(self):
        img = _make_image()
        result = analyse_seal(img)
        assert "seal_anomaly_score" in result

    def test_score_in_range(self):
        rng = np.random.default_rng(3)
        img = rng.random((3, 64, 64)).astype(np.float32)
        score = analyse_seal(img)["seal_anomaly_score"]
        assert 0.0 <= score <= 1.0

    def test_uniform_patch_low_anomaly(self):
        img = _make_image(fill=0.5)
        score = analyse_seal(img)["seal_anomaly_score"]
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_empty_image_returns_zero(self):
        img = np.zeros((3, 0, 0), dtype=np.float32)
        score = analyse_seal(img)["seal_anomaly_score"]
        assert score == 0.0


class TestExtractAllFeatures:
    def test_all_keys_present(self):
        img = _make_image()
        result = extract_all_features(img)
        expected = {
            "font_anomaly_score",
            "colour_deviation_score",
            "barcode_anomaly_score",
            "bar_count",
            "seal_anomaly_score",
            "overall_anomaly_score",
        }
        assert expected.issubset(result.keys())

    def test_overall_is_mean_of_component_scores(self):
        img = _make_image()
        result = extract_all_features(img)
        score_keys = [k for k in result if k.endswith("_score") and k != "overall_anomaly_score"]
        expected_mean = np.mean([result[k] for k in score_keys])
        assert result["overall_anomaly_score"] == pytest.approx(expected_mean, abs=1e-4)


class TestInternalUtilities:
    def test_run_lengths_empty_input(self):
        out = _run_lengths(np.array([], dtype=np.int8))
        assert out.size == 0
