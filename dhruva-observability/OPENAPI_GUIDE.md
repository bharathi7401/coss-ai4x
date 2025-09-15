# Dhruva Observability Plugin: Python Integration Guide

## Overview

The Dhruva Observability Plugin provides a comprehensive OpenAPI 3.0 specification that documents all available endpoints, request/response schemas, and integration details. This guide focuses on Python integration and usage.

## Available Formats

### 1. YAML Format
- **File**: `openapi.yaml`
- **Use Case**: Human-readable, easy to edit
- **Tools**: Swagger Editor, OpenAPI Generator

## API Endpoints

### Core Endpoints

#### `/enterprise/metrics` (GET)
- **Purpose**: Prometheus-compatible metrics endpoint
- **Response**: Text/plain format with Prometheus metrics
- **Use Case**: Monitoring system integration

#### `/enterprise/health` (GET)
- **Purpose**: Health check endpoint
- **Response**: JSON with plugin health status
- **Use Case**: Load balancer health checks

#### `/enterprise/config` (GET)
- **Purpose**: Plugin configuration endpoint
- **Response**: JSON with current configuration
- **Use Case**: Debugging and configuration verification

#### `/enterprise/status` (GET)
- **Purpose**: Detailed plugin status
- **Response**: JSON with runtime status information
- **Use Case**: Monitoring and troubleshooting

### Analytics Endpoints

#### `/enterprise/analytics/customers` (GET)
- **Purpose**: Customer-level analytics
- **Parameters**: `time_range`, `metric_type`
- **Response**: Customer metrics and usage patterns
- **Use Case**: Business intelligence and reporting

#### `/enterprise/analytics/services` (GET)
- **Purpose**: Service-level analytics
- **Parameters**: `time_range`, `service_type`
- **Response**: Service performance metrics
- **Use Case**: Service optimization and monitoring

#### `/enterprise/sla/compliance` (GET)
- **Purpose**: SLA compliance reporting
- **Parameters**: `time_range`, `sla_type`
- **Response**: SLA compliance metrics
- **Use Case**: SLA monitoring and reporting

## Python Usage Examples

### 1. Basic API Integration

```python
import requests
import json

# Base URL for your Dhruva instance
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/enterprise/health")
health_data = response.json()
print(f"Plugin Status: {health_data['status']}")

# Get customer analytics
response = requests.get(
    f"{base_url}/enterprise/analytics/customers",
    params={"time_range": "24h", "metric_type": "all"}
)
analytics_data = response.json()

for customer in analytics_data["customers"]:
    print(f"Customer {customer['customer_id']}: {customer['total_requests']} requests")

# Get SLA compliance
response = requests.get(
    f"{base_url}/enterprise/sla/compliance",
    params={"time_range": "24h", "sla_type": "all"}
)
sla_data = response.json()
print(f"Overall SLA Compliance: {sla_data['overall_compliance']}%")
```

### 2. Using the Plugin Directly

```python
import os
from fastapi import FastAPI
from dhruva_observability import ObservabilityPlugin

# Set environment variables
os.environ["DHRUVA_OBSERVABILITY_ENABLED"] = "true"
os.environ["DHRUVA_OBSERVABILITY_CUSTOMERS"] = "customer1,customer2"
os.environ["DHRUVA_OBSERVABILITY_APPS"] = "app1,app2"

# Create FastAPI app
app = FastAPI(title="My Dhruva App")

# Initialize and register the observability plugin
plugin = ObservabilityPlugin()
plugin.register_plugin(app)

# Your existing endpoints work unchanged
@app.post("/nmt/translate")
async def translate():
    return {"translated": "Hello World"}

@app.post("/tts/synthesize")
async def synthesize():
    return {"audio": "base64_encoded_audio"}
```

### 3. Custom Metrics Collection

```python
from dhruva_observability import MetricsCollector

# Initialize metrics collector
metrics = MetricsCollector()

# Track custom business metrics
def track_custom_metric(value, customer, app):
    metrics.update_dynamic_metric("custom_metric", value, customer, app)

# Track service resource usage
def track_resource_usage(service, customer, app, endpoint):
    metrics.track_service_resource_usage(service, customer, app, endpoint)

# Example usage
track_custom_metric(100, "customer1", "app1")
track_resource_usage("nmt", "customer1", "app1", "/translate")
```

### 4. Monitoring Dashboard

```python
import requests
import time
from datetime import datetime

def update_dashboard():
    base_url = "http://localhost:8000"
    
    # Get health status
    health = requests.get(f"{base_url}/enterprise/health").json()
    
    # Get customer analytics
    analytics = requests.get(
        f"{base_url}/enterprise/analytics/customers",
        params={"time_range": "1h"}
    ).json()
    
    # Get SLA compliance
    sla = requests.get(
        f"{base_url}/enterprise/sla/compliance",
        params={"time_range": "1h"}
    ).json()
    
    # Update dashboard
    dashboard_data = {
        "timestamp": datetime.now().isoformat(),
        "health": health["status"],
        "total_requests": sum(c["total_requests"] for c in analytics["customers"]),
        "sla_compliance": sla["overall_compliance"]
    }
    
    return dashboard_data

# Run every minute
while True:
    data = update_dashboard()
    print(f"Dashboard updated: {data}")
    time.sleep(60)
```

### 5. Alerting System

```python
import requests
import smtplib
from email.mime.text import MIMEText

def check_alerts():
    base_url = "http://localhost:8000"
    
    # Check health
    health = requests.get(f"{base_url}/enterprise/health").json()
    if health["status"] != "healthy":
        send_alert("Plugin Health Alert", f"Plugin status: {health['status']}")
    
    # Check SLA compliance
    sla = requests.get(
        f"{base_url}/enterprise/sla/compliance",
        params={"time_range": "1h"}
    ).json()
    
    if sla["overall_compliance"] < 95:
        send_alert("SLA Compliance Alert", 
                  f"SLA compliance: {sla['overall_compliance']}%")
    
    # Check individual customers
    analytics = requests.get(
        f"{base_url}/enterprise/analytics/customers",
        params={"time_range": "1h"}
    ).json()
    
    for customer in analytics["customers"]:
        if customer["success_rate"] < 95:
            send_alert("Customer Success Rate Alert",
                      f"Customer {customer['customer_id']}: {customer['success_rate']}%")

def send_alert(subject, message):
    # Implementation for sending alerts
    print(f"ALERT: {subject} - {message}")

# Run every 5 minutes
import time
while True:
    check_alerts()
    time.sleep(300)
```

### 6. Prometheus Integration

```python
import requests
from prometheus_client import start_http_server, Gauge, Counter
import time

# Create Prometheus metrics
dhruva_health = Gauge('dhruva_plugin_health', 'Plugin health status')
dhruva_requests = Counter('dhruva_total_requests', 'Total requests processed')
dhruva_sla_compliance = Gauge('dhruva_sla_compliance_percent', 'SLA compliance percentage')

def collect_metrics():
    base_url = "http://localhost:8000"
    
    # Get health status
    health = requests.get(f"{base_url}/enterprise/health").json()
    dhruva_health.set(1 if health["status"] == "healthy" else 0)
    
    # Get customer analytics
    analytics = requests.get(
        f"{base_url}/enterprise/analytics/customers",
        params={"time_range": "1h"}
    ).json()
    
    total_requests = sum(c["total_requests"] for c in analytics["customers"])
    dhruva_requests.inc(total_requests)
    
    # Get SLA compliance
    sla = requests.get(
        f"{base_url}/enterprise/sla/compliance",
        params={"time_range": "1h"}
    ).json()
    
    dhruva_sla_compliance.set(sla["overall_compliance"])

# Start Prometheus metrics server
start_http_server(8001)

# Collect metrics every 30 seconds
while True:
    collect_metrics()
    time.sleep(30)
```

## Schema Reference

### Core Schemas

#### HealthStatus
```json
{
  "status": "healthy",
  "plugin": "dhruva-observability",
  "version": "1.0.0",
  "enabled": true,
  "customers": ["customer1", "customer2"],
  "apps": ["app1", "app2"]
}
```

#### PluginConfig
```json
{
  "enabled": true,
  "debug": false,
  "customers": ["customer1", "customer2"],
  "apps": ["app1", "app2"],
  "default_customer": "default",
  "default_app": "default",
  "metrics_path": "/enterprise/metrics",
  "health_path": "/enterprise/health",
  "collect_system_metrics": true,
  "collect_gpu_metrics": true,
  "collect_db_metrics": true,
  "availability_target": 100.0,
  "response_time_target": 1.0,
  "throughput_target": 20.0
}
```

#### CustomerMetrics
```json
{
  "customer_id": "customer1",
  "total_requests": 1500,
  "success_rate": 99.5,
  "avg_response_time": 0.8,
  "total_characters": 50000,
  "total_tokens": 75000,
  "sla_compliance": 98.5
}
```

#### ServiceMetrics
```json
{
  "service_type": "nmt",
  "total_requests": 800,
  "avg_latency": 0.5,
  "success_rate": 99.8,
  "total_characters": 40000,
  "error_rate": 0.2
}
```

#### SLACompliance
```json
{
  "time_range": "24h",
  "overall_compliance": 98.5,
  "sla_targets": {
    "availability_target": 100.0,
    "response_time_target": 1.0,
    "throughput_target": 20.0
  },
  "customer_compliance": [
    {
      "customer_id": "customer1",
      "availability": 99.5,
      "response_time": 98.0,
      "throughput": 97.5,
      "overall": 98.3
    }
  ]
}
```

## Authentication

The API supports two authentication methods:

### 1. API Key Authentication
```python
headers = {"X-API-Key": "your-api-key"}
response = requests.get("http://localhost:8000/enterprise/config", headers=headers)
```

### 2. Bearer Token Authentication
```python
headers = {"Authorization": "Bearer your-jwt-token"}
response = requests.get("http://localhost:8000/enterprise/config", headers=headers)
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "ValidationError",
  "message": "Invalid time_range parameter",
  "details": "time_range must be one of: 1h, 24h, 7d, 30d",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Analytics endpoints**: 100 requests per minute
- **Health/Status endpoints**: 1000 requests per minute
- **Metrics endpoint**: No rate limiting (for Prometheus scraping)

## Best Practices

### 1. Caching
- Cache analytics data for 5-10 minutes
- Health checks can be cached for 30 seconds
- Configuration rarely changes, cache for longer periods

### 2. Error Handling
- Always check HTTP status codes
- Implement exponential backoff for retries
- Log errors for debugging

### 3. Performance
- Use appropriate time ranges for analytics
- Limit concurrent requests
- Monitor API response times

### 4. Security
- Use HTTPS in production
- Implement proper authentication
- Validate all input parameters

This Python-focused guide provides everything you need to integrate with the Dhruva Observability Plugin using Python.