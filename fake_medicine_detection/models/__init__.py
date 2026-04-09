"""CNN model for fake medicine packaging detection."""

import numpy as np


# ---------------------------------------------------------------------------
# Model architecture (pure-NumPy reference implementation)
# This keeps the package dependency-free for environments without TensorFlow/
# PyTorch while still being directly replaceable with a real framework model.
# ---------------------------------------------------------------------------

class ConvLayer:
    """Single 2-D convolutional layer (no-framework, for illustration)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        rng = np.random.default_rng(seed=42)
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = rng.standard_normal(
            (out_channels, in_channels, kernel_size, kernel_size)
        ) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply convolution followed by ReLU activation (vectorised im2col).

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Feature map tensor of shape (batch, out_channels, h_out, w_out).
        """
        batch, _, h, w = x.shape
        k = self.kernel_size
        h_out = h - k + 1
        w_out = w - k + 1

        # im2col: reshape input patches into columns
        # Shape: (batch, in_channels*k*k, h_out*w_out)
        cols = np.lib.stride_tricks.as_strided(
            x,
            shape=(batch, self.in_channels, k, k, h_out, w_out),
            strides=(
                x.strides[0],
                x.strides[1],
                x.strides[2],
                x.strides[3],
                x.strides[2],
                x.strides[3],
            ),
        ).reshape(batch, self.in_channels * k * k, h_out * w_out)

        # Reshape weights to (out_channels, in_channels*k*k)
        w_flat = self.weights.reshape(self.out_channels, -1)

        # Correct einsum: (out, in*k*k) x (batch, in*k*k, h*w) -> (batch, out, h*w)
        out = (w_flat @ cols).reshape(batch, self.out_channels, h_out, w_out)
        out += self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        return np.maximum(0, out)  # ReLU


def global_average_pool(x: np.ndarray) -> np.ndarray:
    """Global average pooling: reduce spatial dims to a single value per channel.

    Args:
        x: Tensor of shape (batch, channels, height, width).

    Returns:
        Tensor of shape (batch, channels).
    """
    return x.mean(axis=(2, 3))


class FakeMedicineDetectorCNN:
    """CNN model that detects counterfeit medicine packaging.

    Architecture overview
    ---------------------
    Input  : (B, 3, H, W) RGB image normalised to [0, 1]
    Conv1  : 16 filters, 3×3, ReLU
    Conv2  : 32 filters, 3×3, ReLU
    GAP    : global average pooling → (B, 32)
    FC     : linear classifier → (B, 1) sigmoid score

    The model outputs a **counterfeit probability** in [0, 1] where values
    closer to 1 indicate higher likelihood of being fake.
    """

    # Threshold above which a packet is classified as counterfeit.
    DETECTION_THRESHOLD: float = 0.5

    def __init__(self):
        self.conv1 = ConvLayer(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = ConvLayer(in_channels=16, out_channels=32, kernel_size=3)
        rng = np.random.default_rng(seed=0)
        self.fc_weights = rng.standard_normal((32, 1)) * 0.01
        self.fc_bias = np.zeros(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> dict:
        """Run inference on a single pre-processed image.

        Args:
            image: Float32 array of shape (3, H, W) with values in [0, 1].

        Returns:
            dict with keys:
                ``counterfeit_probability`` (float in [0, 1])
                ``is_counterfeit`` (bool)
                ``confidence`` (float – distance from 0.5 mapped to [0, 1])
        """
        if image.ndim == 3:
            image = image[np.newaxis]  # add batch dim → (1, 3, H, W)

        features = self._extract_features(image)
        prob = float(self._sigmoid(features @ self.fc_weights + self.fc_bias)[0, 0])
        return {
            "counterfeit_probability": round(prob, 4),
            "is_counterfeit": prob >= self.DETECTION_THRESHOLD,
            "confidence": round(abs(prob - 0.5) * 2, 4),
        }

    def load_weights(self, path: str) -> None:
        """Load pre-trained weights from a NumPy .npz archive.

        Args:
            path: File path to a ``.npz`` file produced by :meth:`save_weights`.
        """
        data = np.load(path)
        self.conv1.weights = data["conv1_w"]
        self.conv1.bias = data["conv1_b"]
        self.conv2.weights = data["conv2_w"]
        self.conv2.bias = data["conv2_b"]
        self.fc_weights = data["fc_w"]
        self.fc_bias = data["fc_b"]

    def save_weights(self, path: str) -> None:
        """Save current weights to a NumPy .npz archive.

        Args:
            path: Destination file path (extension ``.npz`` recommended).
        """
        np.savez(
            path,
            conv1_w=self.conv1.weights,
            conv1_b=self.conv1.bias,
            conv2_w=self.conv2.weights,
            conv2_b=self.conv2.bias,
            fc_w=self.fc_weights,
            fc_b=self.fc_bias,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        return global_average_pool(x)  # (batch, 32)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
