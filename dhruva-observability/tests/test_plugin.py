"""
Tests for Dhruva Observability Plugin

Comprehensive test suite for the observability plugin.
"""
import pytest
import time
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dhruva_observability import ObservabilityPlugin, PluginConfig, MetricsCollector


class TestPluginConfig:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PluginConfig()
        assert config.enabled == False
        assert config.debug == False
        assert config.default_customer == "default"
        assert config.default_app == "default"
    
    def test_env_config(self):
        """Test configuration from environment variables."""
        with patch.dict('os.environ', {
            'DHRUVA_OBSERVABILITY_ENABLED': 'true',
            'DHRUVA_OBSERVABILITY_CUSTOMERS': 'cust1,cust2',
            'DHRUVA_OBSERVABILITY_APPS': 'app1,app2',
            'DHRUVA_OBSERVABILITY_DEBUG': 'true'
        }):
            config = PluginConfig()
            assert config.enabled == True
            assert config.debug == True
            assert config.customers == ['cust1', 'cust2']
            assert config.apps == ['app1', 'app2']
    
    def test_customer_allowed(self):
        """Test customer validation."""
        config = PluginConfig()
        config.customers = ['cust1', 'cust2']
        assert config.is_customer_allowed('cust1') == True
        assert config.is_customer_allowed('cust3') == False
    
    def test_app_allowed(self):
        """Test app validation."""
        config = PluginConfig()
        config.apps = ['app1', 'app2']
        assert config.is_app_allowed('app1') == True
        assert config.is_app_allowed('app3') == False


class TestMetricsCollector:
    """Test metrics collection."""
    
    def test_metrics_collector_init(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        assert collector is not None
    
    def test_request_tracking(self):
        """Test request tracking."""
        collector = MetricsCollector()
        
        # Start request
        request_id = collector.start_request("customer1", "app1", "/test", "service1")
        assert request_id is not None
        assert request_id in collector._req
        
        # End request
        collector.end_request(request_id, 200)
        assert request_id not in collector._req
        assert request_id in collector._completed_requests
    
    def test_component_tracking(self):
        """Test component tracking."""
        collector = MetricsCollector()
        
        request_id = collector.start_request("customer1", "app1", "/test", "service1")
        
        # Start component
        collector.start_component(request_id, "nmt")
        assert "nmt" in collector._req[request_id]["components"]
        
        # End component
        collector.end_component(request_id, "nmt", success=True)
        assert "nmt" not in collector._req[request_id]["components"]
        
        collector.end_request(request_id, 200)
    
    def test_service_metrics(self):
        """Test service-specific metrics."""
        collector = MetricsCollector()
        
        # Test NMT metrics
        collector.nmt_chars("customer1", "app1", "en", "hi", 100)
        
        # Test TTS metrics
        collector.tts_chars("customer1", "app1", "en", 50)
        
        # Test ASR metrics
        collector.asr_minutes("customer1", "app1", "en", 0.5)
        
        # Test LLM metrics
        collector.llm_tokens("customer1", "app1", "gpt-3.5-turbo", 150)
    
    def test_context_managers(self):
        """Test context managers."""
        collector = MetricsCollector()
        
        # Test request timer
        with collector.request_timer("customer1", "app1", "/test") as request_id:
            assert request_id is not None
        
        # Test component timer
        request_id = collector.start_request("customer1", "app1", "/test", "service1")
        with collector.component_timer(request_id, "nmt"):
            time.sleep(0.01)
        collector.end_request(request_id, 200)


class TestObservabilityPlugin:
    """Test main plugin functionality."""
    
    def test_plugin_init(self):
        """Test plugin initialization."""
        plugin = ObservabilityPlugin()
        assert plugin is not None
        assert not plugin.is_initialized()
    
    def test_plugin_registration(self):
        """Test plugin registration with FastAPI."""
        app = FastAPI()
        plugin = ObservabilityPlugin()
        
        # Mock config to enable plugin
        plugin.config.enabled = True
        
        plugin.register_plugin(app)
        assert plugin.is_initialized()
    
    def test_plugin_endpoints(self):
        """Test plugin endpoints."""
        app = FastAPI()
        plugin = ObservabilityPlugin()
        plugin.config.enabled = True
        
        plugin.register_plugin(app)
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/enterprise/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["plugin"] == "dhruva-enterprise"
        
        # Test metrics endpoint
        response = client.get("/enterprise/metrics")
        assert response.status_code == 200
        assert "dhruva_observability" in response.text
        
        # Test config endpoint
        response = client.get("/enterprise/config")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data


class TestIntegration:
    """Test integration scenarios."""
    
    def test_full_integration(self):
        """Test full integration with FastAPI app."""
        app = FastAPI()
        
        # Configure plugin
        config = PluginConfig()
        config.enabled = True
        config.debug = True
        
        plugin = ObservabilityPlugin(config)
        plugin.register_plugin(app)
        
        # Add sample endpoint
        @app.post("/test")
        async def test_endpoint(request):
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Test endpoint with headers
        response = client.post(
            "/test",
            headers={"X-Customer-ID": "customer1", "X-App-ID": "app1"}
        )
        assert response.status_code == 200
        
        # Check metrics
        response = client.get("/enterprise/metrics")
        assert response.status_code == 200
        metrics_text = response.text
        
        # Verify metrics are being collected
        assert "dhruva_observability_requests_total" in metrics_text
        assert "customer1" in metrics_text
        assert "app1" in metrics_text


if __name__ == "__main__":
    pytest.main([__file__])
