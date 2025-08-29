"""
Prometheus metrics for AI4X application
"""
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from typing import Dict, Any
import threading

# Create custom registry for better control
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'ai4x_requests_total',
    'Total number of requests',
    ['customer', 'app', 'endpoint', 'status'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'ai4x_request_duration_seconds',
    'Request duration in seconds',
    ['customer', 'app', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
    registry=REGISTRY
)

# Pipeline component latency metrics
COMPONENT_LATENCY = Histogram(
    'ai4x_component_latency_seconds',
    'Component processing latency in seconds',
    ['component', 'customer', 'app'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    'ai4x_errors_total',
    'Total number of errors',
    ['customer', 'app', 'error_type', 'component'],
    registry=REGISTRY
)

# System metrics
CPU_USAGE = Gauge(
    'ai4x_cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'ai4x_memory_usage_bytes',
    'Memory usage in bytes',
    registry=REGISTRY
)

MEMORY_USAGE_PERCENT = Gauge(
    'ai4x_memory_usage_percent',
    'Memory usage percentage',
    registry=REGISTRY
)

GPU_USAGE = Gauge(
    'ai4x_gpu_usage_percent',
    'GPU usage percentage',
    registry=REGISTRY
)

GPU_MEMORY = Gauge(
    'ai4x_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    registry=REGISTRY
)

# Database connection metrics
DB_CONNECTIONS_ACTIVE = Gauge(
    'ai4x_db_connections_active',
    'Active database connections',
    registry=REGISTRY
)

# API throughput
THROUGHPUT = Gauge(
    'ai4x_throughput_requests_per_second',
    'Requests per second',
    ['customer', 'app'],
    registry=REGISTRY
)

# Data processing metrics
DATA_PROCESSED_TOTAL = Counter(
    'ai4x_data_processed_total',
    'Total data processed by type',
    ['data_type', 'customer', 'app'],
    registry=REGISTRY
)

# Service-specific metrics
SERVICE_REQUESTS = Counter(
    'ai4x_service_requests_total',
    'Total requests by service type',
    ['service', 'customer', 'app'],
    registry=REGISTRY
)

# NMT specific metrics
NMT_CHARACTERS_TRANSLATED = Counter(
    'ai4x_nmt_characters_translated_total',
    'Total characters translated by NMT',
    ['customer', 'app', 'source_lang', 'target_lang'],
    registry=REGISTRY
)

# TTS specific metrics
TTS_CHARACTERS_SYNTHESIZED = Counter(
    'ai4x_tts_characters_synthesized_total',
    'Total characters synthesized by TTS',
    ['customer', 'app', 'language'],
    registry=REGISTRY
)

# ASR specific metrics
ASR_AUDIO_MINUTES_PROCESSED = Counter(
    'ai4x_asr_audio_minutes_processed_total',
    'Total audio minutes processed by ASR',
    ['customer', 'app', 'language'],
    registry=REGISTRY
)

# LLM specific metrics
LLM_TOKENS_PROCESSED = Counter(
    'ai4x_llm_tokens_processed_total',
    'Total tokens processed by LLM',
    ['customer', 'app', 'model'],
    registry=REGISTRY
)

class MetricsCollector:
    def __init__(self):
        self.request_times = {}
        self.throughput_counter = {}
        self.last_throughput_update = time.time()
        self._start_system_metrics_collector()
    
    def _start_system_metrics_collector(self):
        """Start background thread to collect system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
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
    
    def start_request(self, customer: str, app: str, endpoint: str) -> str:
        """Start timing a request"""
        request_id = f"{customer}_{app}_{endpoint}_{int(time.time() * 1000000)}"
        self.request_times[request_id] = {
            'start_time': time.time(),
            'customer': customer,
            'app': app,
            'endpoint': endpoint,
            'components': {}
        }
        return request_id
    
    def start_component(self, request_id: str, component: str):
        """Start timing a component"""
        if request_id in self.request_times:
            self.request_times[request_id]['components'][component] = {
                'start_time': time.time()
            }
    
    def end_component(self, request_id: str, component: str, success: bool = True):
        """End timing a component"""
        if request_id in self.request_times and component in self.request_times[request_id]['components']:
            duration = time.time() - self.request_times[request_id]['components'][component]['start_time']
            customer = self.request_times[request_id]['customer']
            app = self.request_times[request_id]['app']
            
            # Record component latency
            COMPONENT_LATENCY.labels(
                component=component,
                customer=customer,
                app=app
            ).observe(duration)
            
            # Record error if not successful
            if not success:
                ERROR_COUNT.labels(
                    customer=customer,
                    app=app,
                    error_type='processing_error',
                    component=component
                ).inc()
    
    def end_request(self, request_id: str, status_code: int):
        """End timing a request"""
        if request_id not in self.request_times:
            return
        
        request_data = self.request_times[request_id]
        duration = time.time() - request_data['start_time']
        customer = request_data['customer']
        app = request_data['app']
        endpoint = request_data['endpoint']
        
        # Determine status
        if status_code >= 200 and status_code < 300:
            status = 'success'
        elif status_code >= 400 and status_code < 500:
            status = 'client_error'
        elif status_code >= 500:
            status = 'server_error'
        else:
            status = 'unknown'
        
        # Record metrics
        REQUEST_COUNT.labels(
            customer=customer,
            app=app,
            endpoint=endpoint,
            status=status
        ).inc()
        
        REQUEST_DURATION.labels(
            customer=customer,
            app=app,
            endpoint=endpoint
        ).observe(duration)
        
        # Record errors
        if status != 'success':
            ERROR_COUNT.labels(
                customer=customer,
                app=app,
                error_type=status,
                component='api'
            ).inc()
        
        # Update throughput counter
        key = f"{customer}_{app}"
        if key not in self.throughput_counter:
            self.throughput_counter[key] = 0
        self.throughput_counter[key] += 1
        
        # Update throughput metrics every 10 seconds
        current_time = time.time()
        if current_time - self.last_throughput_update >= 10:
            time_diff = current_time - self.last_throughput_update
            for key, count in self.throughput_counter.items():
                customer_app = key.split('_', 1)
                if len(customer_app) == 2:
                    cust, app_name = customer_app
                    rps = count / time_diff
                    THROUGHPUT.labels(customer=cust, app=app_name).set(rps)
            
            # Reset counters
            self.throughput_counter = {}
            self.last_throughput_update = current_time
        
        # Clean up
        del self.request_times[request_id]
    
    def record_timeout(self, customer: str, app: str, component: str):
        """Record a timeout error"""
        ERROR_COUNT.labels(
            customer=customer,
            app=app,
            error_type='timeout',
            component=component
        ).inc()
    
    def record_service_request(self, service: str, customer: str, app: str):
        """Record a service-specific request"""
        SERVICE_REQUESTS.labels(service=service, customer=customer, app=app).inc()

    def record_nmt_translation(self, customer: str, app: str, source_lang: str, target_lang: str, char_count: int):
        """Record NMT translation metrics"""
        NMT_CHARACTERS_TRANSLATED.labels(
            customer=customer,
            app=app,
            source_lang=source_lang,
            target_lang=target_lang
        ).inc(char_count)

        # Also record as general data processing
        DATA_PROCESSED_TOTAL.labels(
            data_type='nmt_characters',
            customer=customer,
            app=app
        ).inc(char_count)

    def record_tts_synthesis(self, customer: str, app: str, language: str, char_count: int):
        """Record TTS synthesis metrics"""
        TTS_CHARACTERS_SYNTHESIZED.labels(
            customer=customer,
            app=app,
            language=language
        ).inc(char_count)

        # Also record as general data processing
        DATA_PROCESSED_TOTAL.labels(
            data_type='tts_characters',
            customer=customer,
            app=app
        ).inc(char_count)

    def record_asr_processing(self, customer: str, app: str, language: str, audio_minutes: float):
        """Record ASR processing metrics"""
        ASR_AUDIO_MINUTES_PROCESSED.labels(
            customer=customer,
            app=app,
            language=language
        ).inc(audio_minutes)

        # Also record as general data processing
        DATA_PROCESSED_TOTAL.labels(
            data_type='asr_minutes',
            customer=customer,
            app=app
        ).inc(audio_minutes)

    def record_llm_processing(self, customer: str, app: str, model: str, token_count: int):
        """Record LLM processing metrics"""
        LLM_TOKENS_PROCESSED.labels(
            customer=customer,
            app=app,
            model=model
        ).inc(token_count)

        # Also record as general data processing
        DATA_PROCESSED_TOTAL.labels(
            data_type='llm_tokens',
            customer=customer,
            app=app
        ).inc(token_count)

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest(REGISTRY).decode('utf-8')
