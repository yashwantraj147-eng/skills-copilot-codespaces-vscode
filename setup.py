"""Package setup for Fake Medicine Detection."""

from setuptools import find_packages, setup

setup(
    name="fake-medicine-detection",
    version="1.0.0",
    description=(
        "CNN-based fake medicine detection via packaging image analysis "
        "(font inconsistencies, colour deviations, barcode anomalies, seal patterns)"
    ),
    packages=find_packages(exclude=["*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "image": ["Pillow>=9.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fmd-serve=fake_medicine_detection.api:run_server",
        ]
    },
)
