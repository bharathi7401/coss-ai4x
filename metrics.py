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


NMT_CHARACTERS_TRANSLATED = Counter(
    "ai4x_nmt_characters_translated_total",
    "Total characters translated by NMT",
    ["customer", "app", "source_lang", "target_lang"],
    registry=REGISTRY,
)


TTS_CHARACTERS_SYNTHESIZED = Counter(
    "ai4x_tts_characters_synthesized_total",
    "Total characters synthesized by TTS",
    ["customer", "app", "language"],
    registry=REGISTRY,
)


ASR_AUDIO_MINUTES_PROCESSED = Counter(
    "ai4x_asr_audio_minutes_processed_total",
    "Total audio minutes processed by ASR",
    ["customer", "app", "language"],
    registry=REGISTRY,
)


LLM_TOKENS_PROCESSED = Counter(
    "ai4x_llm_tokens_processed_total",
    "Total tokens processed by LLM",
    ["customer", "app", "model"],
    registry=REGISTRY,
)


SERVICE_REQUESTS = Counter(
    "ai4x_service_requests_total",
    "Total requests by service type",
    ["service", "customer", "app"],
    registry=REGISTRY,
)

class MetricsCollector:
    """Encapsulates state & helpers. Avoids per‑request labels like requestId."""

    def __init__(self) -> None:
        self._req: Dict[str, Dict] = {}
        self._throughput_counter: Dict[str, int] = {}
        self._last_throughput_update = time.time()
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
        if status != "success":
            ERROR_COUNT.labels(customer, app, status, "api").inc()

        key = f"{customer}|{app}"
        self._throughput_counter[key] = self._throughput_counter.get(key, 0) + 1
        now = time.time()
        if now - self._last_throughput_update >= 10:
            dt = max(now - self._last_throughput_update, 1e-9)
            for k, cnt in self._throughput_counter.items():
                cust, appname = k.split("|", 1)
                THROUGHPUT.labels(cust, appname).set(cnt / dt)
            self._throughput_counter.clear()
            self._last_throughput_update = now

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