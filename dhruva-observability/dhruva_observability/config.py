"""
Configuration system for Dhruva Observability Plugin

Handles environment variables, defaults, and plugin configuration.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PluginConfig:
    """Configuration for Dhruva Observability Plugin."""
    
    # Core settings
    enabled: bool = False
    debug: bool = False
    
    # Customer and app settings
    customers: List[str] = None
    apps: List[str] = None
    default_customer: str = "default"
    default_app: str = "default"
    
    # Endpoint settings
    metrics_path: str = "/enterprise/metrics"
    health_path: str = "/enterprise/health"
    
    # Monitoring settings
    collect_system_metrics: bool = True
    collect_gpu_metrics: bool = True
    collect_db_metrics: bool = True
    
    # SLA settings
    availability_target: float = 100.0
    response_time_target: float = 1.0
    throughput_target: float = 20.0
    
    # Advanced settings
    max_completed_requests: int = 1000
    metrics_update_interval: int = 10
    system_metrics_interval: int = 5
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        if self.customers is None:
            self.customers = self._parse_list(os.getenv("DHRUVA_OBSERVABILITY_CUSTOMERS", "default"))
        if self.apps is None:
            self.apps = self._parse_list(os.getenv("DHRUVA_OBSERVABILITY_APPS", "default"))
            
        # Override with environment variables if set
        self.enabled = os.getenv("DHRUVA_OBSERVABILITY_ENABLED", "false").lower() == "true"
        self.debug = os.getenv("DHRUVA_OBSERVABILITY_DEBUG", "false").lower() == "true"
        
        self.default_customer = os.getenv("DHRUVA_OBSERVABILITY_DEFAULT_CUSTOMER", self.default_customer)
        self.default_app = os.getenv("DHRUVA_OBSERVABILITY_DEFAULT_APP", self.default_app)
        
        self.metrics_path = os.getenv("DHRUVA_OBSERVABILITY_METRICS_PATH", self.metrics_path)
        self.health_path = os.getenv("DHRUVA_OBSERVABILITY_HEALTH_PATH", self.health_path)
        
        self.collect_system_metrics = os.getenv("DHRUVA_OBSERVABILITY_COLLECT_SYSTEM_METRICS", "true").lower() == "true"
        self.collect_gpu_metrics = os.getenv("DHRUVA_OBSERVABILITY_COLLECT_GPU_METRICS", "true").lower() == "true"
        self.collect_db_metrics = os.getenv("DHRUVA_OBSERVABILITY_COLLECT_DB_METRICS", "true").lower() == "true"
        
        # SLA targets
        self.availability_target = float(os.getenv("DHRUVA_OBSERVABILITY_AVAILABILITY_TARGET", self.availability_target))
        self.response_time_target = float(os.getenv("DHRUVA_OBSERVABILITY_RESPONSE_TIME_TARGET", self.response_time_target))
        self.throughput_target = float(os.getenv("DHRUVA_OBSERVABILITY_THROUGHPUT_TARGET", self.throughput_target))
        
        # Advanced settings
        self.max_completed_requests = int(os.getenv("DHRUVA_OBSERVABILITY_MAX_COMPLETED_REQUESTS", self.max_completed_requests))
        self.metrics_update_interval = int(os.getenv("DHRUVA_OBSERVABILITY_METRICS_UPDATE_INTERVAL", self.metrics_update_interval))
        self.system_metrics_interval = int(os.getenv("DHRUVA_OBSERVABILITY_SYSTEM_METRICS_INTERVAL", self.system_metrics_interval))
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated string into list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    
    def is_customer_allowed(self, customer: str) -> bool:
        """Check if customer is in allowed list."""
        return customer in self.customers or "default" in self.customers
    
    def is_app_allowed(self, app: str) -> bool:
        """Check if app is in allowed list."""
        return app in self.apps or "default" in self.apps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "debug": self.debug,
            "customers": self.customers,
            "apps": self.apps,
            "default_customer": self.default_customer,
            "default_app": self.default_app,
            "metrics_path": self.metrics_path,
            "health_path": self.health_path,
            "collect_system_metrics": self.collect_system_metrics,
            "collect_gpu_metrics": self.collect_gpu_metrics,
            "collect_db_metrics": self.collect_db_metrics,
            "availability_target": self.availability_target,
            "response_time_target": self.response_time_target,
            "throughput_target": self.throughput_target,
            "max_completed_requests": self.max_completed_requests,
            "metrics_update_interval": self.metrics_update_interval,
            "system_metrics_interval": self.system_metrics_interval,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PluginConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "PluginConfig":
        """Create configuration from environment variables."""
        return cls()


# Global configuration instance
config = PluginConfig.from_env()
