"""Unit tests for the preprocessing module."""

import numpy as np
import pytest

from fake_medicine_detection.preprocessing import (
    normalize,
    preprocess_image,
    resize_bilinear,
    to_chw,
)


class TestNormalize:
    def test_uint8_range_maps_to_float(self):
        img = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize(img)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0, abs=1e-6)

    def test_output_dtype_is_float32(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        assert normalize(img).dtype == np.float32


class TestResizeBilinear:
    def test_upscale_shape(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = resize_bilinear(img, 20, 20)
        assert out.shape == (20, 20, 3)

    def test_downscale_shape(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = resize_bilinear(img, 32, 32)
        assert out.shape == (32, 32, 3)

    def test_uniform_image_preserves_value(self):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        out = resize_bilinear(img, 25, 25)
        np.testing.assert_allclose(out, 200, atol=1)


class TestToChw:
    def test_axes_permuted(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        out = to_chw(img)
        assert out.shape == (3, 32, 32)


class TestPreprocessImage:
    def test_output_shape(self):
        raw = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        out = preprocess_image(raw, target_size=(64, 64))
        assert out.shape == (3, 64, 64)

    def test_output_range(self):
        raw = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        out = preprocess_image(raw)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_grayscale_input(self):
        raw = np.full((40, 40), 128, dtype=np.uint8)
        out = preprocess_image(raw, target_size=(32, 32))
        assert out.shape == (3, 32, 32)

    def test_rgba_input_drops_alpha(self):
        raw = np.random.randint(0, 255, (40, 40, 4), dtype=np.uint8)
        out = preprocess_image(raw, target_size=(32, 32))
        assert out.shape == (3, 32, 32)

    def test_single_channel_input(self):
        raw = np.random.randint(0, 255, (40, 40, 1), dtype=np.uint8)
        out = preprocess_image(raw, target_size=(32, 32))
        assert out.shape == (3, 32, 32)

    def test_invalid_channels_raises(self):
        raw = np.zeros((40, 40, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            preprocess_image(raw)
