# Fake Medicine Detection using Packaging Image Analysis

A CNN-based system that analyses medicine packaging photographs to detect
counterfeit drugs — **no lab required**.

## Overview

A user photographs a medicine packet with their mobile device. The image is
sent to this service, which runs:

| Analysis | What it detects |
|---|---|
| **Font inconsistency** | Blurring, noise, and irregular edges around printed text |
| **Colour deviation** | Per-channel colour drift from the genuine product reference |
| **Barcode anomaly** | Irregular bar widths, smearing, or missing quiet zones |
| **Seal pattern** | Variance anomalies in tamper-evident seal regions |
| **CNN classifier** | End-to-end deep feature extraction and binary classification |

All scores are combined into a final verdict: **GENUINE**, **SUSPICIOUS**, or **COUNTERFEIT**.

---

## Project Structure

```
fake_medicine_detection/
├── __init__.py              # Package metadata
├── models/
│   └── __init__.py          # FakeMedicineDetectorCNN (vectorised NumPy CNN)
├── preprocessing/
│   └── __init__.py          # Image resize, normalise, channel reorder
├── features/
│   └── __init__.py          # Font, colour, barcode, seal analysers
├── api/
│   └── __init__.py          # WSGI-compatible REST API
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_model.py
    └── test_api.py
setup.py
requirements.txt
pytest.ini
```

---

## Quick Start

### Install

```bash
pip install -e ".[image,dev]"
```

Or install just the runtime dependencies:

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
python -m fake_medicine_detection.api
# Listening on 0.0.0.0:8080
```

### Analyse a medicine packet image

```bash
# Encode image to base64 and POST to /predict
IMAGE_B64=$(base64 -w0 /path/to/packet.jpg)
curl -s -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"$IMAGE_B64\"}" | python -m json.tool
```

Example response:

```json
{
  "cnn_detection": {
    "counterfeit_probability": 0.7123,
    "is_counterfeit": true,
    "confidence": 0.4246
  },
  "feature_analysis": {
    "font_anomaly_score": 0.82,
    "colour_deviation_score": 0.54,
    "barcode_anomaly_score": 0.61,
    "bar_count": 38,
    "seal_anomaly_score": 0.43,
    "overall_anomaly_score": 0.60
  },
  "verdict": "COUNTERFEIT"
}
```

### Health check

```bash
curl http://localhost:8080/health
# {"status": "ok", "model": "FakeMedicineDetectorCNN"}
```

---

## Running Tests

```bash
python -m pytest fake_medicine_detection/tests/ -v
```

Expected output: **45 tests pass**.

---

## Architecture

```
Mobile App
    |
    | (base64 image bytes via HTTPS)
    v
REST API  (/predict)
    |
    +-- Preprocessing ────────────────────> CNN Model ──> counterfeit_probability
    |   (resize, normalise, CHW)
    |
    +-- Feature Extractors
    |   ├── Font inconsistency   (Sobel edge variance)
    |   ├── Colour deviation     (Z-score vs. reference distribution)
    |   ├── Barcode anomaly      (run-length coefficient of variation)
    |   └── Seal pattern         (corner patch intensity variance)
    |
    +-- Verdict engine (weighted combination)
            GENUINE / SUSPICIOUS / COUNTERFEIT
```

---

## Using Pre-trained Weights

Save weights after training:

```python
from fake_medicine_detection.models import FakeMedicineDetectorCNN
model = FakeMedicineDetectorCNN()
# ... train model ...
model.save_weights("weights.npz")
```

Load at startup via environment variable:

```bash
MODEL_WEIGHTS_PATH=weights.npz python -m fake_medicine_detection.api
```

---

## Production Notes

- Replace the built-in `http.server` with **gunicorn** or **uvicorn** for production workloads.
- The CNN architecture is intentionally small and dependency-free (pure NumPy). For higher accuracy, swap `FakeMedicineDetectorCNN` with a TensorFlow/PyTorch model that exposes the same `predict()` interface.
- Add HTTPS termination (nginx, AWS ALB, etc.) before exposing to mobile clients.
- Consider rate-limiting and authentication middleware for the `/predict` endpoint.
