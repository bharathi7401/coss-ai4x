"""
Prometheus metrics for AI4X application
"""
from __future__ import annotations
import time
import threading
from contextlib import contextmanager
from typing import Dict
from config import Config
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
    "ai4x_requests_total",
    "Total number of requests",
    ["customer", "app", "endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "ai4x_request_duration_seconds",
    "Request duration in seconds",
    ["customer", "app", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0],
    registry=REGISTRY,
)

# ----------------------------
# Component latency (LangDetection, NMT, LLM, BackNMT, TTS, PipelineTotal)
# ----------------------------
COMPONENT_LATENCY = Histogram(
    "ai4x_component_latency_seconds",
    "Component processing latency in seconds",
    ["component", "customer", "app"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

# ----------------------------
# Errors
# ----------------------------
ERROR_COUNT = Counter(
    "ai4x_errors_total",
    "Total number of errors",
    ["customer", "app", "error_type", "component"],
    registry=REGISTRY,
)

# Initialize error count to 0 for dashboard visibility
ERROR_COUNT.labels("default", "default", "none", "none").inc(0)

# ----------------------------
GPU_USAGE = Gauge("ai4x_gpu_usage_percent", "GPU usage %", registry=REGISTRY)
GPU_MEMORY = Gauge("ai4x_gpu_memory_usage_bytes", "GPU mem (bytes)", registry=REGISTRY)
DB_CONNECTIONS_ACTIVE = Gauge(
    "ai4x_db_connections_active",
    "Active DB connections (set from app)",
    registry=REGISTRY,
)


# ----------------------------
# Throughput (Requests/sec per customer+app)
# ----------------------------
THROUGHPUT = Gauge(
    "ai4x_throughput_requests_per_second",
    "Requests per second",
    ["customer", "app"],
    registry=REGISTRY,
)


# ----------------------------
# Data processed counters
# ----------------------------
DATA_PROCESSED_TOTAL = Counter(
    "ai4x_data_processed_total",
    "Total data processed by type",
    ["data_type", "customer", "app"],
    registry=REGISTRY,
)

# Initialize data processed counters to 0 for dashboard visibility
DATA_PROCESSED_TOTAL.labels("nmt_characters", "default", "default").inc(0)
DATA_PROCESSED_TOTAL.labels("tts_characters", "default", "default").inc(0)
DATA_PROCESSED_TOTAL.labels("asr_minutes", "default", "default").inc(0)
DATA_PROCESSED_TOTAL.labels("llm_tokens", "default", "default").inc(0)

# ----------------------------
# System-level metrics
# ----------------------------
SYSTEM_ACTIVE_TENANTS = Gauge(
    "ai4x_system_active_tenants",
    "Number of active tenants/customers",
    registry=REGISTRY,
)

SYSTEM_SERVICE_COUNT = Gauge(
    "ai4x_system_service_count",
    "Total number of services available",
    ["service_type"],
    registry=REGISTRY,
)

# ----------------------------
# QoS and SLA metrics
# ----------------------------
QOS_AVAILABILITY_PERCENT = Gauge(
    "ai4x_qos_availability_percent",
    "Service availability percentage",
    ["time_window"],
    registry=REGISTRY,
)

QOS_PERFORMANCE_SCORE = Gauge(
    "ai4x_qos_performance_score",
    "QoS performance score for services",
    ["customer", "app"],
    registry=REGISTRY,
)

# System uptime/availability metrics
SYSTEM_UPTIME_PERCENT = Gauge(
    "ai4x_system_uptime_percent",
    "System uptime percentage based on availability failures",
    ["time_window"],
    registry=REGISTRY,
)

SYSTEM_AVAILABILITY_FAILURES = Counter(
    "ai4x_system_availability_failures_total",
    "Total number of system availability failures",
    ["failure_type", "component"],
    registry=REGISTRY,
)

# Initialize QoS performance scores to 100% (perfect performance) for dashboard visibility
QOS_PERFORMANCE_SCORE.labels("cust1", "voice-assistant (cust1)").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust1", "nmt-app").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust1", "llm-app").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust1", "tts-app").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust2", "chat-support (cust2)").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust2", "nmt-app").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust2", "llm-app").set(100.0)
QOS_PERFORMANCE_SCORE.labels("cust2", "tts-app").set(100.0)

# Initialize system uptime to 100% (perfect uptime) for dashboard visibility
SYSTEM_UPTIME_PERCENT.labels("1h").set(100.0)
SYSTEM_UPTIME_PERCENT.labels("24h").set(100.0)
SYSTEM_UPTIME_PERCENT.labels("7d").set(100.0)

# Initialize system availability failures to 0 for dashboard visibility
SYSTEM_AVAILABILITY_FAILURES.labels("api_error", "api").inc(0)
SYSTEM_AVAILABILITY_FAILURES.labels("service_error", "nmt").inc(0)
SYSTEM_AVAILABILITY_FAILURES.labels("service_error", "llm").inc(0)
SYSTEM_AVAILABILITY_FAILURES.labels("service_error", "tts").inc(0)
SYSTEM_AVAILABILITY_FAILURES.labels("service_error", "asr").inc(0)
SYSTEM_AVAILABILITY_FAILURES.labels("timeout", "api").inc(0)
SYSTEM_AVAILABILITY_FAILURES.labels("timeout", "service").inc(0)

SYSTEM_SLA_COMPLIANCE_PERCENT = Gauge(
    "ai4x_system_sla_compliance_percent",
    "SLA compliance percentage",
    ["sla_type"],
    registry=REGISTRY,
)

# ----------------------------
# System performance metrics
# ----------------------------
SYSTEM_AVG_RESPONSE_TIME_SECONDS = Gauge(
    "ai4x_system_avg_response_time_seconds",
    "Average system response time in seconds",
    registry=REGISTRY,
)

SYSTEM_PEAK_THROUGHPUT_RPM = Gauge(
    "ai4x_system_peak_throughput_rpm",
    "Peak throughput in requests per minute",
    registry=REGISTRY,
)

SYSTEM_ERROR_RATE_PERCENT = Gauge(
    "ai4x_system_error_rate_percent",
    "Overall system error rate percentage",
    registry=REGISTRY,
)

# ----------------------------
# Resource utilization metrics
# ----------------------------
CPU_USAGE_PERCENT = Gauge(
    "ai4x_cpu_usage_percent",
    "CPU usage percentage",
    registry=REGISTRY,
)

MEMORY_USAGE_PERCENT = Gauge(
    "ai4x_memory_usage_percent",
    "Memory usage percentage",
    registry=REGISTRY,
)


NMT_CHARACTERS_TRANSLATED = Counter(
    "ai4x_nmt_characters_translated_total",
    "Total characters translated by NMT",
    ["customer", "app", "source_lang", "target_lang"],
    registry=REGISTRY,
)

# Initialize NMT characters translated to 0 for dashboard visibility
NMT_CHARACTERS_TRANSLATED.labels("default", "default", "en", "hi").inc(0)


TTS_CHARACTERS_SYNTHESIZED = Counter(
    "ai4x_tts_characters_synthesized_total",
    "Total characters synthesized by TTS",
    ["customer", "app", "language"],
    registry=REGISTRY,
)

# Initialize TTS characters synthesized to 0 for dashboard visibility
TTS_CHARACTERS_SYNTHESIZED.labels("default", "default", "en").inc(0)


ASR_AUDIO_MINUTES_PROCESSED = Counter(
    "ai4x_asr_audio_minutes_processed_total",
    "Total audio minutes processed by ASR",
    ["customer", "app", "language"],
    registry=REGISTRY,
)

# Initialize ASR audio minutes processed to 0 for dashboard visibility
ASR_AUDIO_MINUTES_PROCESSED.labels("default", "default", "en").inc(0)


LLM_TOKENS_PROCESSED = Counter(
    "ai4x_llm_tokens_processed_total",
    "Total tokens processed by LLM",
    ["customer", "app", "model"],
    registry=REGISTRY,
)

# Initialize LLM tokens processed to 0 for dashboard visibility
LLM_TOKENS_PROCESSED.labels("default", "default", "gpt-3.5-turbo").inc(0)


SERVICE_REQUESTS = Counter(
    "ai4x_service_requests_total",
    "Total requests by service type",
    ["service", "customer", "app"],
    registry=REGISTRY,
)

# Initialize service request counts to 0 for dashboard visibility
SERVICE_REQUESTS.labels("nmt", "default", "default").inc(0)
SERVICE_REQUESTS.labels("backnmt", "default", "default").inc(0)
SERVICE_REQUESTS.labels("llm", "default", "default").inc(0)
SERVICE_REQUESTS.labels("tts", "default", "default").inc(0)
SERVICE_REQUESTS.labels("asr", "default", "default").inc(0)

class MetricsCollector:
    """Encapsulates state & helpers. Avoids per‑request labels like requestId."""

    def __init__(self) -> None:
        self._req: Dict[str, Dict] = {}
        self._throughput_counter: Dict[str, int] = {}
        self._last_throughput_update = time.time()
        # Track request success/failure for error rate calculation
        self._request_success_count: Dict[str, int] = {}
        self._request_error_count: Dict[str, int] = {}
        self._start_system_metrics_collector()

    # ---------- background: system metrics ----------
    def _start_system_metrics_collector(self) -> None:
        """Start background thread to collect system metrics"""
        def collect_system_metrics():
            psutil.cpu_percent(interval=None)
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    CPU_USAGE.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    MEMORY_USAGE.set(memory.used)
                    MEMORY_USAGE_PERCENT.set(memory.percent)
                    
                    # GPU metrics (if available)
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            GPU_USAGE.set(gpu.load * 100)
                            GPU_MEMORY.set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes
                    except ImportError:
                        # GPU monitoring not available
                        pass
                    
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()

    def _update_system_metrics(self) -> None:
        """Update system-level metrics that don't change frequently"""
        # Set active tenants (customers) - this would typically come from a database query
        # For now, we'll set it to a reasonable default
        self.set_active_tenants(2)  # cust1 and cust2
        
        # Set service counts
        self.set_service_count("total", 3)  # NMT, LLM, TTS, ASR
        self.set_service_count("nmt", 1)
        self.set_service_count("llm", 1)
        self.set_service_count("tts", 1)
        self.set_service_count("asr", 1)
        
        # Set QoS availability (default to 100% until requests are made)
        self.set_qos_availability("1h", 100.0)
        self.set_qos_availability("24h", 100.0)
        self.set_qos_availability("7d", 100.0)
        
        # Set SLA compliance (default to 100% until requests are made)
        self.set_sla_compliance("availability", 100.0)
        self.set_sla_compliance("response_time", 100.0)
        self.set_sla_compliance("throughput", 100.0)
        
        # Set average response time (default to 0 until requests are made)
        self.set_avg_response_time(0.0)
        
        # Set peak throughput (default to 0 until requests are made)
        self.set_peak_throughput_rpm(0.0)
        
        # Set error rate (default to 0% until errors occur)
        self.set_error_rate_percent(0.0)

    # ---------- request helpers ----------
    def start_request(self, customer: str, app: str, endpoint: str) -> str:
        rid = f"{customer}:{app}:{endpoint}:{int(time.time() * 1e6)}"
        self._req[rid] = {
            "t0": time.time(),
            "customer": customer,
            "app": app,
            "endpoint": endpoint,
            "components": {},
        }
        return rid

    def end_request(self, rid: str, status_code: int) -> None:
        d = self._req.pop(rid, None)
        if not d:
            return
        dur = time.time() - d["t0"]
        customer = d["customer"]
        app = d["app"]
        ep = d["endpoint"]

        status = (
            "success"
            if 200 <= status_code < 300
            else "client_error"
            if 400 <= status_code < 500
            else "server_error"
            if status_code >= 500
            else "unknown"
        )

        REQUEST_COUNT.labels(customer, app, ep, status).inc()
        REQUEST_DURATION.labels(customer, app, ep).observe(dur)
        
        # Track success/failure for error rate calculation
        key = f"{customer}|{app}"
        if status == "success":
            self._request_success_count[key] = self._request_success_count.get(key, 0) + 1
        else:
            self._request_error_count[key] = self._request_error_count.get(key, 0) + 1
            ERROR_COUNT.labels(customer, app, status, "api").inc()
            
            # Record system availability failure
            if status == "server_error":
                self.record_availability_failure("api_error", "api")
            elif status == "client_error":
                self.record_availability_failure("api_error", "api")
            else:
                self.record_availability_failure("api_error", "api")

        self._throughput_counter[key] = self._throughput_counter.get(key, 0) + 1
        now = time.time()
        if now - self._last_throughput_update >= 10:
            dt = max(now - self._last_throughput_update, 1e-9)
            for k, cnt in self._throughput_counter.items():
                cust, appname = k.split("|", 1)
                THROUGHPUT.labels(cust, appname).set(cnt / dt)
            self._throughput_counter.clear()
            self._last_throughput_update = now
            
            # Update dynamic metrics
            self.update_dynamic_metrics()

    # ---------- component timing ----------
    def start_component(self, rid: str, component: str) -> None:
        d = self._req.get(rid)
        if d is None:
            return
        d["components"][component] = time.time()

    def end_component(self, rid: str, component: str, success: bool = True) -> None:
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
            # Record service availability failure
            self.record_availability_failure("service_error", component.lower())

    # ---------- counters for services & data ----------
    def service_request(self, service: str, customer: str, app: str) -> None:
        SERVICE_REQUESTS.labels(service, customer, app).inc()

    def nmt_chars(self, customer: str, app: str, source_lang: str, target_lang: str, n: int) -> None:
        NMT_CHARACTERS_TRANSLATED.labels(customer, app, source_lang, target_lang).inc(max(n, 0))
        DATA_PROCESSED_TOTAL.labels("nmt_characters", customer, app).inc(max(n, 0))

    def tts_chars(self, customer: str, app: str, language: str, n: int) -> None:
        TTS_CHARACTERS_SYNTHESIZED.labels(customer, app, language).inc(max(n, 0))
        DATA_PROCESSED_TOTAL.labels("tts_characters", customer, app).inc(max(n, 0))

    def asr_minutes(self, customer: str, app: str, language: str, minutes: float) -> None:
        ASR_AUDIO_MINUTES_PROCESSED.labels(customer, app, language).inc(max(minutes, 0.0))
        DATA_PROCESSED_TOTAL.labels("asr_minutes", customer, app).inc(max(minutes, 0.0))

    def llm_tokens(self, customer: str, app: str, model: str, tokens: int) -> None:
        LLM_TOKENS_PROCESSED.labels(customer, app, model).inc(max(tokens, 0))
        DATA_PROCESSED_TOTAL.labels("llm_tokens", customer, app).inc(max(tokens, 0))

    def db_pool_size(self, active: int) -> None:
        DB_CONNECTIONS_ACTIVE.set(max(active, 0))

    # ---------- system-level metrics ----------
    def set_active_tenants(self, count: int) -> None:
        SYSTEM_ACTIVE_TENANTS.set(max(count, 0))

    def set_service_count(self, service_type: str, count: int) -> None:
        SYSTEM_SERVICE_COUNT.labels(service_type=service_type).set(max(count, 0))

    def set_qos_availability(self, time_window: str, percent: float) -> None:
        QOS_AVAILABILITY_PERCENT.labels(time_window=time_window).set(max(0, min(100, percent)))

    def set_sla_compliance(self, sla_type: str, percent: float) -> None:
        SYSTEM_SLA_COMPLIANCE_PERCENT.labels(sla_type=sla_type).set(max(0, min(100, percent)))

    def set_avg_response_time(self, seconds: float) -> None:
        SYSTEM_AVG_RESPONSE_TIME_SECONDS.set(max(0, seconds))

    def set_peak_throughput_rpm(self, rpm: float) -> None:
        SYSTEM_PEAK_THROUGHPUT_RPM.set(max(0, rpm))

    def set_error_rate_percent(self, percent: float) -> None:
        SYSTEM_ERROR_RATE_PERCENT.set(max(0, min(100, percent)))

    def set_cpu_usage_percent(self, percent: float) -> None:
        CPU_USAGE_PERCENT.set(max(0, min(100, percent)))

    def set_memory_usage_percent(self, percent: float) -> None:
        MEMORY_USAGE_PERCENT.set(max(0, min(100, percent)))
    

    def set_qos_performance_score(self, customer: str, app: str, score: float) -> None:
        QOS_PERFORMANCE_SCORE.labels(customer, app).set(max(0, min(100, score)))
    
    def set_system_uptime(self, time_window: str, uptime_percent: float) -> None:
        """Set system uptime percentage for a given time window"""
        SYSTEM_UPTIME_PERCENT.labels(time_window).set(max(0, min(100, uptime_percent)))
    
    def record_availability_failure(self, failure_type: str, component: str) -> None:
        """Record a system availability failure"""
        SYSTEM_AVAILABILITY_FAILURES.labels(failure_type, component).inc()

    def update_dynamic_metrics(self) -> None:
        """Update metrics that should be calculated from actual request data"""
        # Calculate average response time from recent requests
        if self._req:
            current_time = time.time()
            recent_requests = [
                req for req in self._req.values()
                if current_time - req["t0"] < 300  # Last 5 minutes
            ]
            
            if recent_requests:
                avg_duration = sum(
                    current_time - req["t0"] for req in recent_requests
                ) / len(recent_requests)
                self.set_avg_response_time(avg_duration)
        
        # Calculate peak throughput from throughput counter
        if self._throughput_counter:
            total_requests = sum(self._throughput_counter.values())
            # Convert to requests per minute (assuming 10-second update interval)
            rpm = total_requests * 6  # 10 seconds * 6 = 60 seconds
            self.set_peak_throughput_rpm(rpm)
        
        # Calculate error rate from actual request data
        total_requests = sum(self._request_success_count.values()) + sum(self._request_error_count.values())
        if total_requests > 0:
            total_errors = sum(self._request_error_count.values())
            error_rate = (total_errors / total_requests) * 100
            self.set_error_rate_percent(error_rate)
        else:
            # No requests yet, error rate is 0%
            self.set_error_rate_percent(0.0)
        
        # Calculate availability based on actual request success rate
        if total_requests > 0:
            total_success = sum(self._request_success_count.values())
            availability = (total_success / total_requests) * 100
            self.set_qos_availability("1h", availability)
            self.set_qos_availability("5m", availability)
            self.set_qos_availability("24h", availability)
            
            # Calculate system uptime based on availability failures
            # System uptime = 100% - (availability_failures / total_requests) * 100
            total_failures = sum(self._request_error_count.values())
            system_uptime = max(0, 100 - (total_failures / total_requests) * 100)
            self.set_system_uptime("1h", system_uptime)
            self.set_system_uptime("24h", system_uptime)
            self.set_system_uptime("7d", system_uptime)
        else:
            # No requests yet, availability and uptime are 100%
            self.set_qos_availability("1h", 100.0)
            self.set_qos_availability("5m", 100.0)
            self.set_qos_availability("24h", 100.0)
            self.set_system_uptime("1h", 100.0)
            self.set_system_uptime("24h", 100.0)
            self.set_system_uptime("7d", 100.0)
        
        # Calculate QoS performance scores based on actual performance
        # Performance score = (availability * 0.4) + ((100 - error_rate) * 0.3) + (response_time_score * 0.3)
        for customer in ["cust1", "cust2"]:
            for app in ["voice-assistant (cust1)", "chat-support (cust2)", "nmt-app", "llm-app", "tts-app"]:
                key = f"{customer}|{app}"
                success_count = self._request_success_count.get(key, 0)
                error_count = self._request_error_count.get(key, 0)
                total_app_requests = success_count + error_count
                
                if total_app_requests > 0:
                    app_availability = (success_count / total_app_requests) * 100
                    app_error_rate = (error_count / total_app_requests) * 100
                    # Simple performance score calculation
                    performance_score = max(0, min(100, app_availability - (app_error_rate * 0.5)))
                else:
                    # No requests for this app yet, default to 100%
                    performance_score = 100.0
                
                self.set_qos_performance_score(customer, app, performance_score)

    # ---------- handy context managers ----------
    @contextmanager
    def component_timer(self, rid: str, component: str):
        self.start_component(rid, component)
        try:
            yield
            self.end_component(rid, component, success=True)
        except Exception:
            self.end_component(rid, component, success=False)
            raise

    @contextmanager
    def request_timer(self, customer: str, app: str, endpoint: str):
        rid = self.start_request(customer, app, endpoint)
        try:
            yield rid
            self.end_request(rid, 200)
        except Exception:
            self.end_request(rid, 500)
            raise


# ------------- module‑level helpers -------------
metrics_collector = MetricsCollector()


def prometheus_latest_text() -> str:
    """Return registry exposition text for /metrics endpoint."""
    return generate_latest(REGISTRY).decode("utf-8")