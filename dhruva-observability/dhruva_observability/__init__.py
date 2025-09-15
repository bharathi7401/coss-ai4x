"""
Dhruva Observability Plugin

This package provides enterprise-grade observability features for the Dhruva Platform,
including comprehensive metrics, monitoring, and business analytics.
"""

__version__ = "1.0.0"
__author__ = "AI4X Team"
__email__ = "team@ai4x.com"

from .plugin import ObservabilityPlugin
from .metrics import MetricsCollector
from .config import PluginConfig
from .middleware import ObservabilityMiddleware

__all__ = [
    "ObservabilityPlugin",
    "MetricsCollector", 
    "PluginConfig",
    "ObservabilityMiddleware",
]
