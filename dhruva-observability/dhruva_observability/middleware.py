"""
FastAPI Middleware for Dhruva Observability

Provides automatic request tracking and metrics collection for FastAPI applications.
"""
import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .metrics import MetricsCollector
from .config import PluginConfig


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic request tracking and metrics collection."""
    
    def __init__(
        self,
        app,
        metrics_collector: Optional[MetricsCollector] = None,
        config: Optional[PluginConfig] = None
    ):
        super().__init__(app)
        self.metrics = metrics_collector or MetricsCollector()
        self.config = config or PluginConfig()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        
        # Skip metrics collection for certain paths
        if self._should_skip_metrics(request):
            return await call_next(request)
        
        # Extract customer and app from headers or default
        customer = self._extract_customer(request)
        app = self._extract_app(request)
        endpoint = request.url.path
        service = self._extract_service(request)
        
        # Start request tracking
        request_id = self.metrics.start_request(customer, app, endpoint, service)
        
        # Add request ID to request state for use in endpoints
        request.state.request_id = request_id
        request.state.customer = customer
        request.state.app = app
        request.state.service = service
        
        start_time = time.time()
        status_code = 200
        
        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code
            return response
            
        except Exception as e:
            # Handle exceptions
            status_code = 500
            if self.config.debug:
                print(f"Request error: {e}")
            raise
            
        finally:
            # End request tracking
            self.metrics.end_request(request_id, status_code)
            
            # Track service request
            self.metrics.service_request(service, customer, app)
            
            # Track resource usage
            self.metrics.track_service_resource_usage(service, customer, app, endpoint)
    
    def _should_skip_metrics(self, request: Request) -> bool:
        """Check if metrics should be skipped for this request."""
        skip_paths = [
            "/metrics",
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
        ]
        
        # Skip if path is in skip list
        if request.url.path in skip_paths:
            return True
            
        # Skip if enterprise is disabled
        if not self.config.enabled:
            return True
            
        return False
    
    def _extract_customer(self, request: Request) -> str:
        """Extract customer identifier from request."""
        # Try to get from header first
        customer = request.headers.get("X-Customer-ID")
        if customer:
            return customer
            
        # Try to get from query params
        customer = request.query_params.get("customer")
        if customer:
            return customer
            
        # Use default customer
        return self.config.default_customer
    
    def _extract_app(self, request: Request) -> str:
        """Extract app identifier from request."""
        # Try to get from header first
        app = request.headers.get("X-App-ID")
        if app:
            return app
            
        # Try to get from query params
        app = request.query_params.get("app")
        if app:
            return app
            
        # Use default app
        return self.config.default_app
    
    def _extract_service(self, request: Request) -> str:
        """Extract service type from request path."""
        path = request.url.path.lower()
        
        if "/nmt" in path or "/translate" in path:
            return "nmt"
        elif "/tts" in path or "/synthesize" in path:
            return "tts"
        elif "/asr" in path or "/transcribe" in path:
            return "asr"
        elif "/llm" in path or "/chat" in path:
            return "llm"
        else:
            return "pipeline"


def create_middleware(
    metrics_collector: Optional[MetricsCollector] = None,
    config: Optional[PluginConfig] = None
) -> Callable:
    """Factory function to create middleware with configuration."""
    
    def middleware_factory(app):
        return ObservabilityMiddleware(app, metrics_collector, config)
    
    return middleware_factory
