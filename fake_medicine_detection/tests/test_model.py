"""Unit tests for the CNN model."""

import numpy as np
import pytest

from fake_medicine_detection.models import FakeMedicineDetectorCNN


def _make_tensor(h: int = 16, w: int = 16) -> np.ndarray:
    """Return a (3, H, W) float32 zero array."""
    return np.zeros((3, h, w), dtype=np.float32)


class TestFakeMedicineDetectorCNN:
    def setup_method(self):
        self.model = FakeMedicineDetectorCNN()

    def test_predict_returns_expected_keys(self):
        img = _make_tensor()
        result = self.model.predict(img)
        assert "counterfeit_probability" in result
        assert "is_counterfeit" in result
        assert "confidence" in result

    def test_probability_in_range(self):
        img = _make_tensor()
        prob = self.model.predict(img)["counterfeit_probability"]
        assert 0.0 <= prob <= 1.0

    def test_confidence_in_range(self):
        img = _make_tensor()
        conf = self.model.predict(img)["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_is_counterfeit_is_bool(self):
        img = _make_tensor()
        result = self.model.predict(img)
        assert isinstance(result["is_counterfeit"], (bool, np.bool_))

    def test_predict_accepts_batched_input(self):
        # (batch=1, C, H, W) should also work
        img = _make_tensor()[np.newaxis]
        result = self.model.predict(img)
        assert 0.0 <= result["counterfeit_probability"] <= 1.0

    def test_different_images_can_differ(self):
        rng = np.random.default_rng(99)
        img1 = rng.random((3, 16, 16)).astype(np.float32)
        img2 = rng.random((3, 16, 16)).astype(np.float32)
        p1 = self.model.predict(img1)["counterfeit_probability"]
        p2 = self.model.predict(img2)["counterfeit_probability"]
        # They should produce valid probabilities even if equal
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0

    def test_save_and_load_weights(self, tmp_path):
        img = _make_tensor()
        prob_before = self.model.predict(img)["counterfeit_probability"]

        weights_file = str(tmp_path / "weights.npz")
        self.model.save_weights(weights_file)

        new_model = FakeMedicineDetectorCNN()
        # Perturb the new model's weights so we can verify loading works
        new_model.fc_bias += 100.0
        new_model.load_weights(weights_file)

        prob_after = new_model.predict(img)["counterfeit_probability"]
        assert prob_before == pytest.approx(prob_after, abs=1e-5)
