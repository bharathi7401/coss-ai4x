# Dhruva Observability Plugin

A comprehensive observability plugin for the Dhruva Platform that adds enterprise-grade monitoring, metrics, and observability features. This plugin works with **any modified Dhruva implementation** without requiring changes to Dhruva's core code.

## 🎯 Key Benefits

- **✅ Zero Dhruva Changes**: Works with existing Dhruva code without modifications
- **✅ Universal Compatibility**: Compatible with any FastAPI-based Dhruva implementation
- **✅ Automatic Tracking**: Collects metrics automatically through middleware
- **✅ Enterprise Features**: Multi-tenant monitoring, SLA tracking, business analytics
- **✅ Easy Integration**: Just 3 lines of code to enable enterprise features

## Features

- **📊 Comprehensive Metrics**: Request tracking, component latency, error monitoring
- **🏢 Multi-tenant Support**: Customer and application-level metrics
- **⚡ Performance Monitoring**: SLA compliance, QoS tracking, resource utilization
- **📈 Business Analytics**: Usage patterns, throughput analysis, cost tracking
- **🔍 Real-time Observability**: Live dashboards and alerts
- **📋 Enterprise Dashboards**: Pre-built Grafana dashboards for business insights
- **🔧 Automatic Service Detection**: Auto-detects NMT, TTS, ASR, LLM services
- **📊 OpenAPI Specification**: Complete API documentation for integration

## Quick Start

### Installation

```bash
pip install dhruva-observability
```

### Basic Usage (3 Lines!)

```python
from fastapi import FastAPI
from dhruva_observability import ObservabilityPlugin

app = FastAPI()

# Just add these 3 lines to enable enterprise features!
enterprise = ObservabilityPlugin()
enterprise.register_plugin(app)

# Your existing Dhruva endpoints work unchanged
@app.post("/nmt/translate")
async def translate():
    return {"translated": "text"}
```

### What Happens Automatically

Once you add the plugin, it automatically:

1. **Tracks all requests** - Count, duration, errors
2. **Detects services** - NMT, TTS, ASR, LLM from URL paths
3. **Extracts metadata** - Customer ID, App ID from headers/query params
4. **Collects metrics** - 50+ metrics automatically updated
5. **Provides endpoints** - `/enterprise/metrics`, `/enterprise/health`
6. **Enables dashboards** - Ready-to-use Grafana dashboards

### Configuration

Set environment variables:

```bash
# Enable the plugin
export DHRUVA_ENTERPRISE_ENABLED=true

# Configure customers and apps
export DHRUVA_ENTERPRISE_CUSTOMERS=cust1,cust2
export DHRUVA_ENTERPRISE_APPS=app1,app2

# Optional: Custom configuration
export DHRUVA_ENTERPRISE_METRICS_PATH=/enterprise/metrics
export DHRUVA_ENTERPRISE_HEALTH_PATH=/enterprise/health
```

## 🔍 How Tracking Works

The plugin uses intelligent middleware that automatically:

### **Request Interception**
- Every request goes through enterprise middleware
- Extracts customer ID, app ID, service type automatically
- Tracks request duration, status codes, errors

### **Service Detection**
```python
# Automatically detects services from URL paths:
/nmt/translate     → service="nmt"
/tts/synthesize    → service="tts"  
/asr/transcribe    → service="asr"
/llm/generate      → service="llm"
```

### **Multi-Tenant Tracking**
```python
# Automatically extracts from headers:
X-Customer-ID: cust1    → customer="cust1"
X-App-ID: app1          → app="app1"
X-Tenant-ID: tenant1    → tenant="tenant1"
```

### **Metrics Collection**
- **Request Metrics**: Count, duration, errors per customer/app
- **Service Metrics**: NMT, TTS, ASR, LLM specific metrics
- **System Metrics**: CPU, memory, GPU usage
- **Business Metrics**: Usage patterns, cost tracking
- **SLA Metrics**: Availability, performance compliance

## 📊 Available Metrics

### **Request Metrics**
- `dhruva_enterprise_requests_total` - Total requests by customer/app/endpoint
- `dhruva_enterprise_request_duration_seconds` - Request latency distribution
- `dhruva_enterprise_errors_total` - Error counts by type and component

### **Component Metrics**
- `dhruva_enterprise_component_latency_seconds` - NMT, LLM, TTS, ASR processing times
- `dhruva_enterprise_service_requests_total` - Service usage by customer/app

### **Business Metrics**
- `dhruva_enterprise_data_processed_total` - Translation, synthesis, transcription volume
- `dhruva_enterprise_throughput_requests_per_second` - Requests per second per customer/app

### **System Metrics**
- `dhruva_enterprise_system_cpu_usage_percent` - CPU utilization
- `dhruva_enterprise_system_memory_usage_percent` - Memory usage
- `dhruva_enterprise_gpu_usage_percent` - GPU utilization (if available)
- `dhruva_enterprise_db_connections_active` - Database connections

### **SLA & QoS Metrics**
- `dhruva_enterprise_qos_availability_percent` - Service availability
- `dhruva_enterprise_qos_performance_score` - Performance scores
- `dhruva_enterprise_system_sla_compliance_percent` - SLA compliance
- `dhruva_enterprise_system_uptime_percent` - System uptime

## 🔧 Compatibility with Modified Dhruva

### **✅ Universal Compatibility**
The plugin works with **ANY** modified Dhruva implementation because:

- **Framework-Level Integration**: Works at FastAPI level, not Dhruva-specific level
- **Non-Intrusive**: Doesn't modify existing Dhruva code
- **Automatic Detection**: Auto-detects services and extracts metadata
- **Zero Dependencies**: No custom APIs or plugin systems required

### **Compatibility Matrix**
| Dhruva Modification Type | Plugin Compatibility | Required Changes |
|-------------------------|---------------------|------------------|
| **Endpoint Changes** | ✅ Full | None |
| **New Services** | ✅ Full | None |
| **Custom Middleware** | ✅ Full | Order middleware |
| **Database Changes** | ✅ Full | None |
| **Authentication Changes** | ✅ Full | None |
| **API Versioning** | ✅ Full | None |
| **Custom FastAPI App** | ✅ Full | None |
| **Non-FastAPI Framework** | ❌ Partial | Create FastAPI wrapper |

### **Integration Examples**

#### **Minimal Integration (Recommended)**
```python
# Works with ANY modified Dhruva
from dhruva_observability import ObservabilityPlugin

app = FastAPI()  # Your modified Dhruva app

# Just add these 3 lines!
enterprise = ObservabilityPlugin()
enterprise.register_plugin(app)

# Your existing endpoints work unchanged
@app.post("/your/custom/endpoint")
async def your_endpoint():
    return {"message": "Works with plugin!"}
```

#### **Enhanced Integration**
```python
# For modified Dhruva with custom services
from dhruva_observability import ObservabilityPlugin, MetricsCollector

app = FastAPI()
enterprise = ObservabilityPlugin()
enterprise.register_plugin(app)

# Add custom metrics to your modified services
metrics = MetricsCollector()

@app.post("/your/custom/nmt")
async def custom_nmt_service(request: Request):
    request_id = metrics.start_request("custom_nmt", request)
    
    try:
        result = await your_custom_nmt_logic(request.json())
        metrics.end_request(request_id, success=True)
        return result
    except Exception as e:
        metrics.end_request(request_id, success=False, error=str(e))
        raise
```

## 📊 API Documentation

### **OpenAPI Specification**
Complete API documentation is available in multiple formats:

- **YAML Format**: `openapi.yaml` - Human-readable, easy to edit
- **JSON Format**: `openapi.json` - Programmatic access, API tools
- **Usage Guide**: `OPENAPI_GUIDE.md` - Complete integration guide

### **Available Endpoints**
- **`/enterprise/metrics`** - Prometheus-compatible metrics
- **`/enterprise/health`** - Health checks and status
- **`/enterprise/config`** - Plugin configuration
- **`/enterprise/analytics`** - Business analytics data
- **`/enterprise/sla`** - SLA monitoring data
- **`/enterprise/dashboard`** - Dashboard data

### **Integration Tools**
- **Swagger UI**: Upload `openapi.yaml` to Swagger Editor
- **Postman**: Import `openapi.json` into Postman
- **Insomnia**: Import `openapi.yaml` into Insomnia
- **Redoc**: Use `openapi.yaml` with Redoc

## 📋 Dashboards

The plugin includes pre-built Grafana dashboards:

- **Customer Overview**: Multi-tenant usage and performance
- **Service Health**: Component-level monitoring  
- **Business Analytics**: Usage patterns and cost analysis
- **SLA Monitoring**: Compliance and performance tracking

## 🔧 Advanced Configuration

### **Custom Metrics (Optional)**
```python
from dhruva_observability import MetricsCollector

metrics = MetricsCollector()

# Track custom business metrics
metrics.update_dynamic_metric("custom_metric", value, customer, app)
metrics.track_service_resource_usage("nmt", "customer1", "app1", "/translate")
```

### **Environment Variables**
```bash
# Core Configuration
DHRUVA_OBSERVABILITY_ENABLED=true
DHRUVA_OBSERVABILITY_CUSTOMERS=cust1,cust2
DHRUVA_OBSERVABILITY_APPS=app1,app2

# Optional Configuration
DHRUVA_OBSERVABILITY_METRICS_PATH=/enterprise/metrics
DHRUVA_OBSERVABILITY_HEALTH_PATH=/enterprise/health
DHRUVA_OBSERVABILITY_DEBUG=false
DHRUVA_OBSERVABILITY_DEFAULT_CUSTOMER=default
DHRUVA_OBSERVABILITY_DEFAULT_APP=default
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**: Check if `DHRUVA_ENTERPRISE_ENABLED=true`
2. **Prometheus scraping**: Ensure `/enterprise/metrics` endpoint is accessible
3. **Dashboard issues**: Verify Grafana datasource configuration

### Debug Mode

```bash
export DHRUVA_ENTERPRISE_DEBUG=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: [Report bugs and request features](https://github.com/ai4x/dhruva-enterprise-plugin/issues)
- Documentation: [Full documentation](https://github.com/ai4x/dhruva-enterprise-plugin/wiki)
- Community: [Dhruva Community Forum](https://github.com/AI4Bharat/Dhruva-Platform/discussions)
