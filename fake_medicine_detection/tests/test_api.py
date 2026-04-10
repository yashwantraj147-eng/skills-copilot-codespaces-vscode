"""Unit tests for the API module."""

import base64
import http.client
import io
import json
import threading
from contextlib import contextmanager
from http.server import HTTPServer

import numpy as np
import pytest

import fake_medicine_detection.api as api_module
from fake_medicine_detection.api import _Handler, _verdict, create_app


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


@contextmanager
def _running_http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _http_request(port: int, method: str, path: str, payload: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = b""
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = json.loads(response.read().decode())
    conn.close()
    return response.status, data


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


class TestHttpHandler:
    def test_get_health_returns_ok(self):
        with _running_http_server() as port:
            status, data = _http_request(port, "GET", "/health")
            assert status == 200
            assert data["status"] == "ok"

    def test_get_unknown_path_returns_404(self):
        with _running_http_server() as port:
            status, data = _http_request(port, "GET", "/missing")
            assert status == 404
            assert data["error"] == "Not found"

    def test_post_unknown_path_returns_404(self):
        with _running_http_server() as port:
            status, data = _http_request(port, "POST", "/missing", payload={})
            assert status == 404
            assert data["error"] == "Not found"

    def test_post_invalid_json_returns_400(self):
        with _running_http_server() as port:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/predict", body=b"{", headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            data = json.loads(response.read().decode())
            conn.close()
            assert response.status == 400
            assert "Invalid JSON" in data["error"]

    def test_post_missing_image_returns_400(self):
        with _running_http_server() as port:
            status, data = _http_request(port, "POST", "/predict", payload={})
            assert status == 400
            assert "Missing 'image' field" in data["error"]

    def test_post_invalid_base64_returns_400(self):
        with _running_http_server() as port:
            status, data = _http_request(port, "POST", "/predict", payload={"image": {"bad": "type"}})
            assert status == 400
            assert "Failed to decode base64 image" in data["error"]

    def test_post_image_decode_error_returns_400(self, monkeypatch):
        def _raise_decode_error(_):
            raise ValueError("decode fail")

        monkeypatch.setattr(api_module, "_decode_image_bytes", _raise_decode_error)
        with _running_http_server() as port:
            status, data = _http_request(port, "POST", "/predict", payload={"image": _dummy_image_b64()})
            assert status == 400
            assert "Failed to decode image" in data["error"]

    def test_post_inference_error_returns_500(self, monkeypatch):
        def _raise_inference_error(_):
            raise RuntimeError("boom")

        monkeypatch.setattr(api_module, "_decode_image_bytes", lambda _: np.zeros((8, 8, 3), dtype=np.uint8))
        monkeypatch.setattr(api_module, "preprocess_image", _raise_inference_error)
        with _running_http_server() as port:
            status, data = _http_request(port, "POST", "/predict", payload={"image": _dummy_image_b64()})
            assert status == 500
            assert data["error"] == "Internal error during inference"

    def test_post_valid_payload_returns_200(self, monkeypatch):
        monkeypatch.setattr(api_module, "_decode_image_bytes", lambda _: np.zeros((8, 8, 3), dtype=np.uint8))
        monkeypatch.setattr(api_module, "preprocess_image", lambda _: np.zeros((3, 8, 8), dtype=np.float32))
        monkeypatch.setattr(
            api_module,
            "extract_all_features",
            lambda _: {
                "font_anomaly_score": 0.1,
                "colour_deviation_score": 0.2,
                "barcode_anomaly_score": 0.3,
                "bar_count": 2,
                "seal_anomaly_score": 0.4,
                "overall_anomaly_score": 0.25,
            },
        )
        monkeypatch.setattr(
            api_module._model,
            "predict",
            lambda _: {
                "counterfeit_probability": 0.3,
                "is_counterfeit": False,
                "confidence": 0.4,
            },
        )

        with _running_http_server() as port:
            status, data = _http_request(port, "POST", "/predict", payload={"image": _dummy_image_b64()})
            assert status == 200
            assert "cnn_detection" in data
            assert "feature_analysis" in data
            assert "verdict" in data


class TestCreateAppExtraBranches:
    def test_predict_invalid_content_length_defaults_to_zero(self):
        app = create_app()
        responses = []

        def start_response(status, headers):
            responses.append({"status": status, "headers": headers})

        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/predict",
            "CONTENT_LENGTH": "invalid",
            "wsgi.input": io.BytesIO(b""),
        }
        result = app(environ, start_response)
        data = json.loads(b"".join(result).decode())
        assert responses[0]["status"].startswith("400")
        assert "error" in data
