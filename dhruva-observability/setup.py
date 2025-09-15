"""
Dhruva Observability Plugin
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dhruva-observability",
    version="1.0.0",
    author="AI4X Team",
    author_email="team@ai4x.com",
    description="Enterprise observability plugin for Dhruva Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4x/dhruva-observability",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0",
        "prometheus-client>=0.12.0",
        "psutil>=5.8.0",
        "uvicorn>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "gpu": [
            "GPUtil>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dhruva-observability=dhruva_observability.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dhruva_observability": [
            "dashboards/*.json",
            "config/*.yaml",
        ],
    },
)
