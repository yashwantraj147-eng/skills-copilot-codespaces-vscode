"""Unit tests for the API module."""

import base64
import io
import json

import numpy as np
import pytest

from fake_medicine_detection.api import _verdict, create_app


def _dummy_image_b64(h: int = 16, w: int = 16) -> str:
    """Create a base64-encoded PNG-like byte payload.

    We use a raw bytes approach so the test has no Pillow dependency.
    The API falls back to a placeholder image when Pillow is absent.
    """
    data = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8).tobytes()
    return base64.b64encode(data).decode()


def _wsgi_request(app, method: str, path: str, body: bytes = b""):
    """Minimal WSGI test client helper."""
    responses = []

    def start_response(status, headers):
        responses.append({"status": status, "headers": headers})

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    result = app(environ, start_response)
    response_body = b"".join(result)
    return responses[0]["status"], json.loads(response_body)


class TestHealthEndpoint:
    def setup_method(self):
        self.app = create_app()

    def test_health_returns_ok(self):
        status, data = _wsgi_request(self.app, "GET", "/health")
        assert status.startswith("200")
        assert data["status"] == "ok"

    def test_unknown_path_returns_404(self):
        status, data = _wsgi_request(self.app, "GET", "/nonexistent")
        assert status.startswith("404")


class TestPredictEndpoint:
    def setup_method(self):
        self.app = create_app()

    def test_predict_with_valid_payload(self):
        payload = json.dumps({"image": _dummy_image_b64()}).encode()
        status, data = _wsgi_request(self.app, "POST", "/predict", payload)
        assert status.startswith("200")
        assert "cnn_detection" in data
        assert "feature_analysis" in data
        assert "verdict" in data

    def test_predict_missing_image_returns_400(self):
        payload = json.dumps({}).encode()
        status, data = _wsgi_request(self.app, "POST", "/predict", payload)
        assert status.startswith("400")
        assert "error" in data

    def test_predict_invalid_json_returns_400(self):
        status, data = _wsgi_request(self.app, "POST", "/predict", b"not json")
        assert status.startswith("400")

    def test_verdict_in_expected_values(self):
        payload = json.dumps({"image": _dummy_image_b64()}).encode()
        _, data = _wsgi_request(self.app, "POST", "/predict", payload)
        assert data["verdict"] in {"GENUINE", "SUSPICIOUS", "COUNTERFEIT"}


class TestVerdict:
    def test_low_scores_genuine(self):
        cnn = {"counterfeit_probability": 0.1}
        features = {"overall_anomaly_score": 0.1}
        assert _verdict(cnn, features) == "GENUINE"

    def test_high_scores_counterfeit(self):
        cnn = {"counterfeit_probability": 0.9}
        features = {"overall_anomaly_score": 0.9}
        assert _verdict(cnn, features) == "COUNTERFEIT"

    def test_mid_scores_suspicious(self):
        cnn = {"counterfeit_probability": 0.5}
        features = {"overall_anomaly_score": 0.5}
        assert _verdict(cnn, features) == "SUSPICIOUS"
