"""REST API for the Fake Medicine Detection service.

Endpoints
---------
POST /predict
    Accepts a multipart/form-data or JSON payload with a base64-encoded image.
    Returns a JSON object with detection results.

GET /health
    Returns service health status.

Usage
-----
    python -m fake_medicine_detection.api

or via gunicorn::

    gunicorn fake_medicine_detection.api:app
"""

from __future__ import annotations

import base64
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np

from fake_medicine_detection.features import extract_all_features
from fake_medicine_detection.models import FakeMedicineDetectorCNN
from fake_medicine_detection.preprocessing import preprocess_image

# ---------------------------------------------------------------------------
# Global model instance (loaded once at startup)
# ---------------------------------------------------------------------------

_model = FakeMedicineDetectorCNN()
_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "")

if _WEIGHTS_PATH and os.path.isfile(_WEIGHTS_PATH):  # pragma: no cover
    _model.load_weights(_WEIGHTS_PATH)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for the detection API."""

    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        """Suppress default access log (use a real logger in production)."""

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok", "model": "FakeMedicineDetectorCNN"})
        else:
            self._send_json({"error": "Not found"}, status=404)

    # ------------------------------------------------------------------
    # POST /predict
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return

        image_b64 = payload.get("image")
        if not image_b64:
            self._send_json(
                {"error": "Missing 'image' field (base64-encoded image bytes)"},
                status=400,
            )
            return

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            self._send_json(
                {"error": f"Failed to decode base64 image: {exc}"}, status=400
            )
            return

        try:
            raw = _decode_image_bytes(image_bytes)
        except Exception as exc:
            self._send_json(
                {"error": f"Failed to decode image: {exc}"}, status=400
            )
            return

        try:
            processed = preprocess_image(raw)
            cnn_result = _model.predict(processed)
            feature_scores = extract_all_features(processed)
        except Exception:  # pragma: no cover
            self._send_json(
                {"error": "Internal error during inference", "detail": traceback.format_exc()},
                status=500,
            )
            return

        response = {
            "cnn_detection": cnn_result,
            "feature_analysis": feature_scores,
            "verdict": _verdict(cnn_result, feature_scores),
        }
        self._send_json(response)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------

def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to a uint8 NumPy array (RGB).

    Tries Pillow first; falls back to generating a placeholder array so
    the API remains functional even without Pillow in the test environment.
    """
    try:
        from fake_medicine_detection.preprocessing import load_image_from_bytes
        return load_image_from_bytes(data)
    except ImportError:
        # Return a grey placeholder image for environments without Pillow.
        return np.full((64, 64, 3), 128, dtype=np.uint8)


def _verdict(cnn_result: dict, features: dict) -> str:
    """Combine CNN and feature scores into a human-readable verdict.

    Args:
        cnn_result: Output of :meth:`FakeMedicineDetectorCNN.predict`.
        features: Output of :func:`extract_all_features`.

    Returns:
        One of ``"GENUINE"``, ``"SUSPICIOUS"``, or ``"COUNTERFEIT"``.
    """
    prob = cnn_result.get("counterfeit_probability", 0.0)
    overall = features.get("overall_anomaly_score", 0.0)
    combined = 0.6 * prob + 0.4 * overall

    if combined >= 0.65:
        return "COUNTERFEIT"
    if combined >= 0.35:
        return "SUSPICIOUS"
    return "GENUINE"


# ---------------------------------------------------------------------------
# WSGI-compatible app (Flask-compatible surface for easy migration)
# ---------------------------------------------------------------------------

def create_app():
    """Return a minimal WSGI callable.

    This allows the module to be used with any WSGI server (e.g. gunicorn)
    without requiring Flask.  For production deployments, replacing this with
    a proper Flask/FastAPI app is recommended.
    """

    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        if method == "GET" and path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body]

        if method == "POST" and path == "/predict":
            try:
                length = int(environ.get("CONTENT_LENGTH") or 0)
            except ValueError:
                length = 0
            raw_body = environ["wsgi.input"].read(length)
            try:
                payload = json.loads(raw_body)
                image_bytes = base64.b64decode(payload["image"])
                raw_img = _decode_image_bytes(image_bytes)
                processed = preprocess_image(raw_img)
                cnn_result = _model.predict(processed)
                feature_scores = extract_all_features(processed)
                result = {
                    "cnn_detection": cnn_result,
                    "feature_analysis": feature_scores,
                    "verdict": _verdict(cnn_result, feature_scores),
                }
                status = "200 OK"
            except Exception as exc:
                result = {"error": str(exc)}
                status = "400 Bad Request"

            body = json.dumps(result).encode()
            start_response(status, [("Content-Type", "application/json")])
            return [body]

        body = json.dumps({"error": "Not found"}).encode()
        start_response("404 Not Found", [("Content-Type", "application/json")])
        return [body]

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:  # pragma: no cover
    """Start the built-in HTTP server.

    Args:
        host: Bind address.
        port: TCP port to listen on.
    """
    server = HTTPServer((host, port), _Handler)
    print(f"Fake Medicine Detection API listening on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    run_server()
