"""
Dhruva Observability Metrics

Comprehensive metrics collection for enterprise-grade observability of Dhruva Platform.
Provides multi-tenant monitoring, SLA tracking, and business analytics.
"""
from __future__ import annotations
import time
import threading
import os
from contextlib import contextmanager
from typing import Dict, Optional
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

# ----------------------------
# Registry
# ----------------------------
REGISTRY = CollectorRegistry()

# ----------------------------
# Request metrics (per customer/app/endpoint)
# ----------------------------
REQUEST_COUNT = Counter(
    "dhruva_enterprise_requests_total",
    "Total number of requests",
    ["customer", "app", "endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "dhruva_enterprise_request_duration_seconds",
    "Request duration in seconds",
    ["customer", "app", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0],
    registry=REGISTRY,
)

# ----------------------------
# Component latency (NMT, LLM, TTS, ASR)
# ----------------------------
COMPONENT_LATENCY = Histogram(
    "dhruva_enterprise_component_latency_seconds",
    "Component processing latency in seconds",
    ["component", "customer", "app"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

# ----------------------------
# Errors
# ----------------------------
ERROR_COUNT = Counter(
    "dhruva_enterprise_errors_total",
    "Total number of errors",
    ["customer", "app", "error_type", "component"],
    registry=REGISTRY,
)

# ----------------------------
# System metrics
# ----------------------------
GPU_USAGE = Gauge("dhruva_enterprise_gpu_usage_percent", "GPU usage %", registry=REGISTRY)
GPU_MEMORY = Gauge("dhruva_enterprise_gpu_memory_usage_bytes", "GPU mem (bytes)", registry=REGISTRY)
DB_CONNECTIONS_ACTIVE = Gauge(
    "dhruva_enterprise_db_connections_active",
    "Active DB connections",
    registry=REGISTRY,
)

# ----------------------------
# Throughput (Requests/sec per customer+app)
# ----------------------------
THROUGHPUT = Gauge(
    "dhruva_enterprise_throughput_requests_per_second",
    "Requests per second",
    ["customer", "app"],
    registry=REGISTRY,
)

# ----------------------------
# Data processed counters
# ----------------------------
DATA_PROCESSED_TOTAL = Counter(
    "dhruva_enterprise_data_processed_total",
    "Total data processed by type",
    ["data_type", "customer", "app"],
    registry=REGISTRY,
)

# ----------------------------
# System-level metrics
# ----------------------------
SYSTEM_ACTIVE_TENANTS = Gauge(
    "dhruva_enterprise_system_active_tenants",
    "Number of active tenants/customers",
    registry=REGISTRY,
)

SYSTEM_SERVICE_COUNT = Gauge(
    "dhruva_enterprise_system_service_count",
    "Total number of services available",
    ["service_type"],
    registry=REGISTRY,
)

# ----------------------------
# QoS and SLA metrics
# ----------------------------
QOS_AVAILABILITY_PERCENT = Gauge(
    "dhruva_enterprise_qos_availability_percent",
    "Service availability percentage",
    ["time_window"],
    registry=REGISTRY,
)

QOS_PERFORMANCE_SCORE = Gauge(
    "dhruva_enterprise_qos_performance_score",
    "QoS performance score for services",
    ["customer", "app", "service", "endpoint"],
    registry=REGISTRY,
)

SYSTEM_UPTIME_PERCENT = Gauge(
    "dhruva_enterprise_system_uptime_percent",
    "System uptime percentage based on availability failures",
    ["time_window"],
    registry=REGISTRY,
)

SYSTEM_AVAILABILITY_FAILURES = Counter(
    "dhruva_enterprise_system_availability_failures_total",
    "Total number of system availability failures",
    ["failure_type", "component"],
    registry=REGISTRY,
)

SYSTEM_SLA_COMPLIANCE_PERCENT = Gauge(
    "dhruva_enterprise_system_sla_compliance_percent",
    "SLA compliance percentage",
    ["sla_type", "customer", "app", "service", "endpoint"],
    registry=REGISTRY,
)

# ----------------------------
# System performance metrics
# ----------------------------
SYSTEM_AVG_RESPONSE_TIME_SECONDS = Gauge(
    "dhruva_enterprise_system_avg_response_time_seconds",
    "Average system response time in seconds",
    registry=REGISTRY,
)

SYSTEM_PEAK_THROUGHPUT_RPM = Gauge(
    "dhruva_enterprise_system_peak_throughput_rpm",
    "Peak throughput in requests per minute",
    registry=REGISTRY,
)

SYSTEM_ERROR_RATE_PERCENT = Gauge(
    "dhruva_enterprise_system_error_rate_percent",
    "Overall system error rate percentage",
    registry=REGISTRY,
)

# ----------------------------
# Resource utilization metrics
# ----------------------------
CPU_USAGE_PERCENT = Gauge(
    "dhruva_enterprise_system_cpu_usage_percent",
    "CPU usage percentage",
    ["service", "customer", "app", "endpoint"],
    registry=REGISTRY,
)

MEMORY_USAGE_PERCENT = Gauge(
    "dhruva_enterprise_system_memory_usage_percent",
    "Memory usage percentage",
    ["service", "customer", "app", "endpoint"],
    registry=REGISTRY,
)

# ----------------------------
# Service-specific metrics
# ----------------------------
NMT_CHARACTERS_TRANSLATED = Counter(
    "dhruva_enterprise_nmt_characters_translated_total",
    "Total characters translated by NMT",
    ["customer", "app", "source_lang", "target_lang"],
    registry=REGISTRY,
)

TTS_CHARACTERS_SYNTHESIZED = Counter(
    "dhruva_enterprise_tts_characters_synthesized_total",
    "Total characters synthesized by TTS",
    ["customer", "app", "language"],
    registry=REGISTRY,
)

ASR_AUDIO_MINUTES_PROCESSED = Counter(
    "dhruva_enterprise_asr_audio_minutes_processed_total",
    "Total audio minutes processed by ASR",
    ["customer", "app", "language"],
    registry=REGISTRY,
)

LLM_TOKENS_PROCESSED = Counter(
    "dhruva_enterprise_llm_tokens_processed_total",
    "Total tokens processed by LLM",
    ["customer", "app", "model"],
    registry=REGISTRY,
)

SERVICE_REQUESTS = Counter(
    "dhruva_enterprise_service_requests_total",
    "Total requests by service type",
    ["service", "customer", "app"],
    registry=REGISTRY,
)


class MetricsCollector:
    """Enterprise metrics collector for Dhruva Platform observability."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self._req: Dict[str, Dict] = {}
        self._completed_requests: Dict[str, Dict] = {}
        self._throughput_counter: Dict[str, int] = {}
        self._last_throughput_update = time.time()
        self._request_success_count: Dict[str, int] = {}
        self._request_error_count: Dict[str, int] = {}
        self._start_system_metrics_collector()

    def _start_system_metrics_collector(self) -> None:
        """Start background thread to collect system metrics"""
        def collect_system_metrics():
            psutil.cpu_percent(interval=None)
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    
                    # Set system-wide metrics
                    CPU_USAGE_PERCENT.labels("system", "system", "system", "system").set(cpu_percent)
                    MEMORY_USAGE_PERCENT.labels("system", "system", "system", "system").set(memory_percent)
                    
                    # GPU metrics (if available)
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            GPU_USAGE.set(gpu.load * 100)
                            GPU_MEMORY.set(gpu.memoryUsed * 1024 * 1024)
                    except ImportError:
                        pass
                    
                    time.sleep(5)
                except Exception as e:
                    if self.config.get("debug", False):
                        print(f"Error collecting system metrics: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()

    def start_request(self, customer: str, app: str, endpoint: str, service: str = "pipeline") -> str:
        """Start tracking a request"""
        rid = f"{customer}:{app}:{endpoint}:{service}:{int(time.time() * 1e6)}"
        self._req[rid] = {
            "t0": time.time(),
            "customer": customer,
            "app": app,
            "endpoint": endpoint,
            "service": service,
            "components": {},
            "completed": False,
            "success": True
        }
        return rid

    def end_request(self, rid: str, status_code: int) -> None:
        """End tracking a request"""
        d = self._req.pop(rid, None)
        if not d:
            return
        
        dur = time.time() - d["t0"]
        customer = d["customer"]
        app = d["app"]
        ep = d["endpoint"]
        service = d.get("service", "pipeline")

        status = (
            "success" if 200 <= status_code < 300
            else "client_error" if 400 <= status_code < 500
            else "server_error" if status_code >= 500
            else "unknown"
        )
        
        # Mark request as completed
        d["completed"] = True
        d["success"] = (status == "success")
        d["duration"] = dur
        d["status_code"] = status_code
        
        # Store completed request for SLA calculation
        self._completed_requests[rid] = d
        
        # Clean up old completed requests
        if len(self._completed_requests) > 1000:
            old_requests = list(self._completed_requests.keys())[:200]
            for old_rid in old_requests:
                del self._completed_requests[old_rid]

        # Update metrics
        REQUEST_COUNT.labels(customer, app, ep, status).inc()
        REQUEST_DURATION.labels(customer, app, ep).observe(dur)
        
        # Track success/failure
        key = f"{customer}|{app}"
        if status == "success":
            self._request_success_count[key] = self._request_success_count.get(key, 0) + 1
        else:
            self._request_error_count[key] = self._request_error_count.get(key, 0) + 1
            ERROR_COUNT.labels(customer, app, status, "api").inc()
            self.record_availability_failure("api_error", "api")

        # Update throughput
        self._throughput_counter[key] = self._throughput_counter.get(key, 0) + 1
        now = time.time()
        if now - self._last_throughput_update >= 10:
            dt = max(now - self._last_throughput_update, 1e-9)
            for k, cnt in self._throughput_counter.items():
                cust, appname = k.split("|", 1)
                THROUGHPUT.labels(cust, appname).set(cnt / dt)
            self._throughput_counter.clear()
            self._last_throughput_update = now
            self.update_dynamic_metrics()

    def start_component(self, rid: str, component: str) -> None:
        """Start tracking a component"""
        d = self._req.get(rid)
        if d is None:
            return
        d["components"][component] = time.time()

    def end_component(self, rid: str, component: str, success: bool = True) -> None:
        """End tracking a component"""
        d = self._req.get(rid)
        if d is None:
            return
        t0 = d["components"].pop(component, None)
        if t0 is None:
            return
        dur = time.time() - t0
        COMPONENT_LATENCY.labels(component, d["customer"], d["app"]).observe(dur)
        if not success:
            ERROR_COUNT.labels(d["customer"], d["app"], "processing_error", component).inc()
            self.record_availability_failure("service_error", component.lower())

    # Service tracking methods
    def service_request(self, service: str, customer: str, app: str) -> None:
        """Track a service request"""
        SERVICE_REQUESTS.labels(service, customer, app).inc()

    def nmt_chars(self, customer: str, app: str, source_lang: str, target_lang: str, n: int) -> None:
        """Track NMT character translation"""
        NMT_CHARACTERS_TRANSLATED.labels(customer, app, source_lang, target_lang).inc(max(n, 0))
        DATA_PROCESSED_TOTAL.labels("nmt_characters", customer, app).inc(max(n, 0))

    def tts_chars(self, customer: str, app: str, language: str, n: int) -> None:
        """Track TTS character synthesis"""
        TTS_CHARACTERS_SYNTHESIZED.labels(customer, app, language).inc(max(n, 0))
        DATA_PROCESSED_TOTAL.labels("tts_characters", customer, app).inc(max(n, 0))

    def asr_minutes(self, customer: str, app: str, language: str, minutes: float) -> None:
        """Track ASR audio processing"""
        ASR_AUDIO_MINUTES_PROCESSED.labels(customer, app, language).inc(max(minutes, 0.0))
        DATA_PROCESSED_TOTAL.labels("asr_minutes", customer, app).inc(max(minutes, 0.0))

    def llm_tokens(self, customer: str, app: str, model: str, tokens: int) -> None:
        """Track LLM token processing"""
        LLM_TOKENS_PROCESSED.labels(customer, app, model).inc(max(tokens, 0))
        DATA_PROCESSED_TOTAL.labels("llm_tokens", customer, app).inc(max(tokens, 0))

    def db_pool_size(self, active: int) -> None:
        """Track database connection pool size"""
        DB_CONNECTIONS_ACTIVE.set(max(active, 0))

    # System-level metrics
    def set_active_tenants(self, count: int) -> None:
        """Set number of active tenants"""
        SYSTEM_ACTIVE_TENANTS.set(max(count, 0))

    def set_service_count(self, service_type: str, count: int) -> None:
        """Set service count"""
        SYSTEM_SERVICE_COUNT.labels(service_type=service_type).set(max(count, 0))

    def set_qos_availability(self, time_window: str, percent: float) -> None:
        """Set QoS availability percentage"""
        QOS_AVAILABILITY_PERCENT.labels(time_window=time_window).set(max(0, min(100, percent)))

    def set_sla_compliance(self, sla_type: str, customer: str, app: str, service: str, endpoint: str, percent: float) -> None:
        """Set SLA compliance percentage"""
        SYSTEM_SLA_COMPLIANCE_PERCENT.labels(sla_type=sla_type, customer=customer, app=app, service=service, endpoint=endpoint).set(max(0, min(100, percent)))

    def set_avg_response_time(self, seconds: float) -> None:
        """Set average response time"""
        SYSTEM_AVG_RESPONSE_TIME_SECONDS.set(max(0, seconds))

    def set_peak_throughput_rpm(self, rpm: float) -> None:
        """Set peak throughput RPM"""
        SYSTEM_PEAK_THROUGHPUT_RPM.set(max(0, rpm))

    def set_error_rate_percent(self, percent: float) -> None:
        """Set error rate percentage"""
        SYSTEM_ERROR_RATE_PERCENT.set(max(0, min(100, percent)))

    def set_qos_performance_score(self, customer: str, app: str, service: str, endpoint: str, score: float) -> None:
        """Set QoS performance score"""
        QOS_PERFORMANCE_SCORE.labels(customer, app, service, endpoint).set(max(0, min(100, score)))

    def set_system_uptime(self, time_window: str, uptime_percent: float) -> None:
        """Set system uptime percentage"""
        SYSTEM_UPTIME_PERCENT.labels(time_window).set(max(0, min(100, uptime_percent)))

    def record_availability_failure(self, failure_type: str, component: str) -> None:
        """Record a system availability failure"""
        SYSTEM_AVAILABILITY_FAILURES.labels(failure_type, component).inc()

    def track_service_resource_usage(self, service: str, customer: str, app: str, endpoint: str) -> None:
        """Track actual CPU and memory usage for a specific service"""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        CPU_USAGE_PERCENT.labels(service, customer, app, endpoint).set(cpu_percent)
        MEMORY_USAGE_PERCENT.labels(service, customer, app, endpoint).set(memory_percent)

    def update_dynamic_metrics(self) -> None:
        """Update metrics calculated from actual request data"""
        # Calculate average response time
        if self._req:
            current_time = time.time()
            recent_requests = [
                req for req in self._req.values()
                if current_time - req["t0"] < 300
            ]
            
            if recent_requests:
                avg_duration = sum(current_time - req["t0"] for req in recent_requests) / len(recent_requests)
                self.set_avg_response_time(avg_duration)
        
        # Calculate peak throughput
        if self._throughput_counter:
            total_requests = sum(self._throughput_counter.values())
            rpm = total_requests * 6
            self.set_peak_throughput_rpm(rpm)
        
        # Calculate error rate
        total_requests = sum(self._request_success_count.values()) + sum(self._request_error_count.values())
        if total_requests > 0:
            total_errors = sum(self._request_error_count.values())
            error_rate = (total_errors / total_requests) * 100
            self.set_error_rate_percent(error_rate)
        else:
            self.set_error_rate_percent(0.0)
        
        # Calculate availability
        if total_requests > 0:
            total_success = sum(self._request_success_count.values())
            availability = (total_success / total_requests) * 100
            self.set_qos_availability("1h", availability)
            self.set_qos_availability("24h", availability)
            
            total_failures = sum(self._request_error_count.values())
            system_uptime = max(0, 100 - (total_failures / total_requests) * 100)
            self.set_system_uptime("1h", system_uptime)
            self.set_system_uptime("24h", system_uptime)
        else:
            self.set_qos_availability("1h", 100.0)
            self.set_qos_availability("24h", 100.0)
            self.set_system_uptime("1h", 100.0)
            self.set_system_uptime("24h", 100.0)

    # Context managers
    @contextmanager
    def component_timer(self, rid: str, component: str):
        """Context manager for component timing"""
        self.start_component(rid, component)
        try:
            yield
            self.end_component(rid, component, success=True)
        except Exception:
            self.end_component(rid, component, success=False)
            raise

    @contextmanager
    def request_timer(self, customer: str, app: str, endpoint: str, service: str = "pipeline"):
        """Context manager for request timing"""
        rid = self.start_request(customer, app, endpoint, service)
        try:
            yield rid
            self.end_request(rid, 200)
        except Exception:
            self.end_request(rid, 500)
            raise


def prometheus_latest_text() -> str:
    """Return registry exposition text for /metrics endpoint."""
    return generate_latest(REGISTRY).decode("utf-8")
