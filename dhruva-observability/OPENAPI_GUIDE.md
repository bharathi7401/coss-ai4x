# Dhruva Enterprise Plugin: OpenAPI Specification

## Overview

The Dhruva Enterprise Plugin provides a comprehensive OpenAPI 3.0 specification that documents all available endpoints, request/response schemas, and integration details. This specification enables:

- **API Documentation**: Automatic generation of interactive documentation
- **Client SDK Generation**: Generate client libraries in multiple languages
- **API Testing**: Import into tools like Postman, Insomnia, or curl
- **Integration**: Easy integration with external systems
- **Validation**: Request/response validation and type checking

## Available Formats

### 1. YAML Format
- **File**: `openapi.yaml`
- **Use Case**: Human-readable, easy to edit
- **Tools**: Swagger Editor, OpenAPI Generator

### 2. JSON Format
- **File**: `openapi.json`
- **Use Case**: Programmatic access, API tools
- **Tools**: Postman, Insomnia, curl

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

## Usage Examples

### 1. View Interactive Documentation

#### Using Swagger UI
```bash
# Install Swagger UI
npm install -g swagger-ui-serve

# Serve the OpenAPI spec
swagger-ui-serve openapi.yaml
```

#### Using Redoc
```bash
# Install Redoc
npm install -g redoc-cli

# Generate documentation
redoc-cli serve openapi.yaml
```

### 2. Generate Client SDKs

#### Python Client
```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Generate Python client
openapi-generator-cli generate \
  -i openapi.yaml \
  -g python \
  -o ./python-client
```

#### JavaScript Client
```bash
# Generate JavaScript client
openapi-generator-cli generate \
  -i openapi.yaml \
  -g javascript \
  -o ./javascript-client
```

#### Go Client
```bash
# Generate Go client
openapi-generator-cli generate \
  -i openapi.yaml \
  -g go \
  -o ./go-client
```

### 3. Import into API Testing Tools

#### Postman
1. Open Postman
2. Click "Import"
3. Select `openapi.yaml` or `openapi.json`
4. All endpoints will be imported with examples

#### Insomnia
1. Open Insomnia
2. Click "Create" → "Import from URL"
3. Provide path to `openapi.yaml`
4. All endpoints will be imported

#### curl Examples
```bash
# Health check
curl -X GET "http://localhost:8000/enterprise/health" \
  -H "accept: application/json"

# Get metrics
curl -X GET "http://localhost:8000/enterprise/metrics" \
  -H "accept: text/plain"

# Get configuration
curl -X GET "http://localhost:8000/enterprise/config" \
  -H "accept: application/json"

# Get customer analytics
curl -X GET "http://localhost:8000/enterprise/analytics/customers?time_range=24h" \
  -H "accept: application/json"

# Get service analytics
curl -X GET "http://localhost:8000/enterprise/analytics/services?service_type=nmt" \
  -H "accept: application/json"

# Get SLA compliance
curl -X GET "http://localhost:8000/enterprise/sla/compliance?sla_type=availability" \
  -H "accept: application/json"
```

### 4. Programmatic Integration

#### Python Example
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

#### JavaScript Example
```javascript
const axios = require('axios');

const baseUrl = 'http://localhost:8000';

// Health check
async function checkHealth() {
    try {
        const response = await axios.get(`${baseUrl}/enterprise/health`);
        console.log('Plugin Status:', response.data.status);
    } catch (error) {
        console.error('Health check failed:', error.message);
    }
}

// Get customer analytics
async function getCustomerAnalytics() {
    try {
        const response = await axios.get(`${baseUrl}/enterprise/analytics/customers`, {
            params: { time_range: '24h', metric_type: 'all' }
        });
        
        response.data.customers.forEach(customer => {
            console.log(`Customer ${customer.customer_id}: ${customer.total_requests} requests`);
        });
    } catch (error) {
        console.error('Analytics request failed:', error.message);
    }
}

// Get SLA compliance
async function getSLACompliance() {
    try {
        const response = await axios.get(`${baseUrl}/enterprise/sla/compliance`, {
            params: { time_range: '24h', sla_type: 'all' }
        });
        
        console.log(`Overall SLA Compliance: ${response.data.overall_compliance}%`);
    } catch (error) {
        console.error('SLA compliance request failed:', error.message);
    }
}

// Run examples
checkHealth();
getCustomerAnalytics();
getSLACompliance();
```

## Schema Reference

### Core Schemas

#### HealthStatus
```json
{
  "status": "healthy",
  "plugin": "dhruva-enterprise",
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
```bash
curl -X GET "http://localhost:8000/enterprise/config" \
  -H "X-API-Key: your-api-key" \
  -H "accept: application/json"
```

### 2. Bearer Token Authentication
```bash
curl -X GET "http://localhost:8000/enterprise/config" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "accept: application/json"
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

## Integration Examples

### 1. Monitoring Dashboard
```python
# Create a monitoring dashboard
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

### 2. Alerting System
```python
# Create an alerting system
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

This OpenAPI specification provides a complete reference for integrating with the Dhruva Enterprise Plugin, enabling developers to build custom monitoring solutions, dashboards, and alerting systems.
