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
# Request metrics (per customer/service/endpoint)
# ----------------------------
REQUEST_COUNT = Counter(
    "ai4x_hack_requests_total",
    "Total number of requests",
    ["customer", "service", "endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "ai4x_hack_request_duration_seconds",
    "Request duration in seconds",
    ["customer", "service", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0],
    registry=REGISTRY,
)

# ----------------------------
# Component latency (LangDetection, NMT, LLM, BackNMT, TTS, PipelineTotal)
# ----------------------------
COMPONENT_LATENCY = Histogram(
    "ai4x_hack_component_latency_seconds",
    "Component processing latency in seconds",
    ["component", "customer", "service"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

# ----------------------------
# Errors
# ----------------------------
ERROR_COUNT = Counter(
    "ai4x_hack_errors_total",
    "Total number of errors",
    ["customer", "service", "error_type", "component"],
    registry=REGISTRY,
)


# ----------------------------
# System Metrics
# ----------------------------
CPU_USAGE = Gauge("ai4x_hack_cpu_usage_percent", "CPU usage %", registry=REGISTRY)
MEMORY_USAGE = Gauge("ai4x_hack_memory_usage_bytes", "Memory usage in bytes", registry=REGISTRY)
MEMORY_USAGE_PERCENT = Gauge("ai4x_hack_memory_usage_percent", "Memory usage %", registry=REGISTRY)
GPU_USAGE = Gauge("ai4x_hack_gpu_usage_percent", "GPU usage %", registry=REGISTRY)
GPU_MEMORY = Gauge("ai4x_hack_gpu_memory_usage_bytes", "GPU mem (bytes)", registry=REGISTRY)
DB_CONNECTIONS_ACTIVE = Gauge(
    "ai4x_hack_db_connections_active",
    "Active DB connections",
    registry=REGISTRY,
)


# ----------------------------
# Throughput (Requests/sec per customer+service)
# ----------------------------
THROUGHPUT = Gauge(
    "ai4x_hack_throughput_requests_per_second",
    "Requests per second",
    ["customer", "service"],
    registry=REGISTRY,
)

# ----------------------------
# Usage and Billing Metrics
# ----------------------------

# Data processed counters
DATA_PROCESSED_TOTAL = Counter(
    "ai4x_hack_data_processed_total",
    "Total data processed by type",
    ["data_type", "customer", "service"],
    registry=REGISTRY,
)

# Service-specific usage counters
NMT_CHARACTERS_TRANSLATED = Counter(
    "ai4x_hack_nmt_characters_translated_total",
    "Total characters translated by NMT",
    ["customer", "service", "source_lang", "target_lang"],
    registry=REGISTRY,
)

TTS_CHARACTERS_SYNTHESIZED = Counter(
    "ai4x_hack_tts_characters_synthesized_total",
    "Total characters synthesized by TTS",
    ["customer", "service", "language"],
    registry=REGISTRY,
)

ASR_AUDIO_MINUTES_PROCESSED = Counter(
    "ai4x_hack_asr_audio_minutes_processed_total",
    "Total audio minutes processed by ASR",
    ["customer", "service", "language"],
    registry=REGISTRY,
)

LLM_TOKENS_PROCESSED = Counter(
    "ai4x_hack_llm_tokens_processed_total",
    "Total tokens processed by LLM",
    ["customer", "service", "model"],
    registry=REGISTRY,
)

SERVICE_REQUESTS = Counter(
    "ai4x_hack_service_requests_total",
    "Total requests by service type",
    ["service", "customer"],
    registry=REGISTRY,
)

# ----------------------------
# Customer Tier and Quota Metrics
# ----------------------------
CUSTOMER_QUOTA_USAGE = Gauge(
    "ai4x_hack_customer_quota_usage_percent",
    "Customer quota usage percentage",
    ["customer", "service", "tier"],
    registry=REGISTRY,
)

CUSTOMER_QUOTA_REMAINING = Gauge(
    "ai4x_hack_customer_quota_remaining",
    "Customer remaining quota",
    ["customer", "service", "tier"],
    registry=REGISTRY,
)

CUSTOMER_QUOTA_LIMIT = Gauge(
    "ai4x_hack_customer_quota_limit",
    "Customer quota limit",
    ["customer", "service", "tier"],
    registry=REGISTRY,
)

# ----------------------------
# Billing and Cost Metrics
# ----------------------------
SERVICE_COST_TOTAL = Counter(
    "ai4x_hack_service_cost_total",
    "Total cost incurred by service usage",
    ["customer", "service", "tier"],
    registry=REGISTRY,
)

SERVICE_COST_PER_REQUEST = Histogram(
    "ai4x_hack_service_cost_per_request",
    "Cost per request by service",
    ["customer", "service", "tier"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    registry=REGISTRY,
)

CUSTOMER_BILLING_TOTAL = Counter(
    "ai4x_hack_customer_billing_total",
    "Total billing amount per customer",
    ["customer", "tier", "billing_period"],
    registry=REGISTRY,
)

# ----------------------------
# Service Performance Metrics
# ----------------------------
SERVICE_SUCCESS_RATE = Gauge(
    "ai4x_hack_service_success_rate",
    "Service success rate percentage",
    ["customer", "service"],
    registry=REGISTRY,
)

SERVICE_AVERAGE_LATENCY = Gauge(
    "ai4x_hack_service_average_latency_seconds",
    "Average service latency in seconds",
    ["customer", "service"],
    registry=REGISTRY,
)

# ----------------------------
# Pipeline Metrics
# ----------------------------
PIPELINE_SUCCESS_RATE = Gauge(
    "ai4x_hack_pipeline_success_rate",
    "Pipeline success rate percentage",
    ["customer"],
    registry=REGISTRY,
)

PIPELINE_AVERAGE_DURATION = Gauge(
    "ai4x_hack_pipeline_average_duration_seconds",
    "Average pipeline duration in seconds",
    ["customer"],
    registry=REGISTRY,
)

PIPELINE_COMPONENT_SUCCESS_RATE = Gauge(
    "ai4x_hack_pipeline_component_success_rate",
    "Pipeline component success rate percentage",
    ["customer", "component"],
    registry=REGISTRY,
)

class MetricsCollector:
    """Encapsulates state & helpers. Avoids per‑request labels like requestId."""

    def __init__(self) -> None:
        self._req: Dict[str, Dict] = {}
        self._throughput_counter: Dict[str, int] = {}
        self._last_throughput_update = time.time()
        self._service_stats: Dict[str, Dict] = {}  # Track service success rates and latencies
        self._start_system_metrics_collector()

    # ---------- background: system metrics ----------
    def _start_system_metrics_collector(self) -> None:
        def collect():
            while True:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    CPU_USAGE.set(cpu_percent)
                    mem = psutil.virtual_memory()
                    MEMORY_USAGE.set(mem.used)
                    MEMORY_USAGE_PERCENT.set(mem.percent)

                    try:
                        import GPUtil  # type: ignore

                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            GPU_USAGE.set(gpu.load * 100.0)
                            GPU_MEMORY.set(gpu.memoryUsed * 1024 * 1024)
                    except Exception:
                        pass

                    time.sleep(4)
                except Exception:
                    time.sleep(4)

        threading.Thread(target=collect, daemon=True).start()

    # ---------- request helpers ----------
    def start_request(self, customer: str, service: str, endpoint: str) -> str:
        rid = f"{customer}:{service}:{endpoint}:{int(time.time() * 1e6)}"
        self._req[rid] = {
            "t0": time.time(),
            "customer": customer,
            "service": service,
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
        service = d["service"]
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

        REQUEST_COUNT.labels(customer, service, ep, status).inc()
        REQUEST_DURATION.labels(customer, service, ep).observe(dur)
        if status != "success":
            ERROR_COUNT.labels(customer, service, status, "api").inc()

        # Update service statistics
        self._update_service_stats(customer, service, dur, status == "success")

        key = f"{customer}|{service}"
        self._throughput_counter[key] = self._throughput_counter.get(key, 0) + 1
        now = time.time()
        if now - self._last_throughput_update >= 10:
            dt = max(now - self._last_throughput_update, 1e-9)
            for k, cnt in self._throughput_counter.items():
                cust, serv = k.split("|", 1)
                THROUGHPUT.labels(cust, serv).set(cnt / dt)
            self._throughput_counter.clear()
            self._last_throughput_update = now

    def _update_service_stats(self, customer: str, service: str, duration: float, success: bool) -> None:
        """Update service statistics for success rate and average latency calculation"""
        key = f"{customer}:{service}"
        if key not in self._service_stats:
            self._service_stats[key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_duration": 0.0
            }
        
        stats = self._service_stats[key]
        stats["total_requests"] += 1
        stats["total_duration"] += duration
        if success:
            stats["successful_requests"] += 1
        
        # Calculate and update metrics
        success_rate = (stats["successful_requests"] / stats["total_requests"]) * 100
        avg_latency = stats["total_duration"] / stats["total_requests"]
        
        SERVICE_SUCCESS_RATE.labels(customer, service).set(success_rate)
        SERVICE_AVERAGE_LATENCY.labels(customer, service).set(avg_latency)

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
        COMPONENT_LATENCY.labels(component, d["customer"], d["service"]).observe(dur)
        if not success:
            ERROR_COUNT.labels(d["customer"], d["service"], "processing_error", component).inc()

    # ---------- counters for services & data ----------
    def service_request(self, service: str, customer: str) -> None:
        SERVICE_REQUESTS.labels(service, customer).inc()

    def nmt_chars(self, customer: str, service: str, source_lang: str, target_lang: str, n: int) -> None:
        NMT_CHARACTERS_TRANSLATED.labels(customer, service, source_lang, target_lang).inc(max(n, 0))
        DATA_PROCESSED_TOTAL.labels("nmt_characters", customer, service).inc(max(n, 0))

    def tts_chars(self, customer: str, service: str, language: str, n: int) -> None:
        TTS_CHARACTERS_SYNTHESIZED.labels(customer, service, language).inc(max(n, 0))
        DATA_PROCESSED_TOTAL.labels("tts_characters", customer, service).inc(max(n, 0))

    def asr_minutes(self, customer: str, service: str, language: str, minutes: float) -> None:
        ASR_AUDIO_MINUTES_PROCESSED.labels(customer, service, language).inc(max(minutes, 0.0))
        DATA_PROCESSED_TOTAL.labels("asr_minutes", customer, service).inc(max(minutes, 0.0))

    def llm_tokens(self, customer: str, service: str, model: str, tokens: int) -> None:
        LLM_TOKENS_PROCESSED.labels(customer, service, model).inc(max(tokens, 0))
        DATA_PROCESSED_TOTAL.labels("llm_tokens", customer, service).inc(max(tokens, 0))

    def db_pool_size(self, active: int) -> None:
        DB_CONNECTIONS_ACTIVE.set(max(active, 0))

    # ---------- billing and cost metrics ----------
    def record_service_cost(self, customer: str, service: str, tier: str, cost: float) -> None:
        """Record cost for service usage"""
        SERVICE_COST_TOTAL.labels(customer, service, tier).inc(cost)
        SERVICE_COST_PER_REQUEST.labels(customer, service, tier).observe(cost)

    def record_customer_billing(self, customer: str, tier: str, billing_period: str, amount: float) -> None:
        """Record total billing amount for customer"""
        CUSTOMER_BILLING_TOTAL.labels(customer, tier, billing_period).inc(amount)

    def update_quota_metrics(self, customer: str, service: str, tier: str, 
                           current_usage: int, limit: int) -> None:
        """Update quota usage metrics"""
        usage_percent = (current_usage / limit) * 100 if limit > 0 else 0
        remaining = max(0, limit - current_usage)
        
        CUSTOMER_QUOTA_USAGE.labels(customer, service, tier).set(usage_percent)
        CUSTOMER_QUOTA_REMAINING.labels(customer, service, tier).set(remaining)
        CUSTOMER_QUOTA_LIMIT.labels(customer, service, tier).set(limit)

    def update_pipeline_metrics(self, customer: str, success: bool, duration: float) -> None:
        """Update pipeline-specific metrics"""
        # This would be called from pipeline endpoint
        # Implementation would track pipeline success rates and durations
        pass

    def update_pipeline_component_success(self, customer: str, component: str, success: bool) -> None:
        """Update pipeline component success rate"""
        # This would track individual component success within pipeline
        pass

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
    def request_timer(self, customer: str, service: str, endpoint: str):
        rid = self.start_request(customer, service, endpoint)
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