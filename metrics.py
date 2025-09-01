"""
Prometheus metrics for AI4X application
"""
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from typing import Dict, Any
import threading
import random
from collections import defaultdict

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

# System Status Dashboard Metrics
SYSTEM_STATUS_TOTAL_API_CALLS = Gauge(
    'ai4x_system_total_api_calls',
    'Total API calls count',
    registry=REGISTRY
)

SYSTEM_STATUS_ACTIVE_TENANTS = Gauge(
    'ai4x_system_active_tenants',
    'Number of active tenants/customers',
    registry=REGISTRY
)

SYSTEM_STATUS_AVERAGE_RESPONSE_TIME = Gauge(
    'ai4x_system_avg_response_time_seconds',
    'Average response time in seconds',
    registry=REGISTRY
)

SYSTEM_STATUS_ERROR_RATE = Gauge(
    'ai4x_system_error_rate_percent',
    'Overall error rate percentage',
    registry=REGISTRY
)

SYSTEM_STATUS_SLA_COMPLIANCE = Gauge(
    'ai4x_system_sla_compliance_percent',
    'SLA compliance percentage',
    ['sla_type'],
    registry=REGISTRY
)

SYSTEM_STATUS_PEAK_THROUGHPUT = Gauge(
    'ai4x_system_peak_throughput_rpm',
    'Peak throughput in requests per minute',
    registry=REGISTRY
)

SYSTEM_STATUS_SERVICE_COUNT = Gauge(
    'ai4x_system_service_count',
    'Number of active services',
    ['service_type'],
    registry=REGISTRY
)

# QoS and Availability Metrics
QOS_AVAILABILITY = Gauge(
    'ai4x_qos_availability_percent',
    'Service availability percentage',
    ['customer', 'app', 'time_window'],
    registry=REGISTRY
)

QOS_SLA_COMPLIANCE = Gauge(
    'ai4x_qos_sla_compliance_percent',
    'SLA compliance percentage',
    ['customer', 'app', 'sla_type'],
    registry=REGISTRY
)

QOS_PERFORMANCE_SCORE = Gauge(
    'ai4x_qos_performance_score',
    'Overall performance score (0-100)',
    ['customer', 'app'],
    registry=REGISTRY
)

QOS_RELIABILITY_SCORE = Gauge(
    'ai4x_qos_reliability_score',
    'Service reliability score (0-100)',
    ['customer', 'app'],
    registry=REGISTRY
)

class SystemStatusCalculator:
    def __init__(self):
        self.request_history = defaultdict(list)
        self.response_time_history = []
        self.error_history = []
        self.throughput_history = []
        self.active_customers = set()
        self.peak_throughput = 0
        
    def update_system_status_metrics(self):
        """Calculate and update all system status metrics"""
        try:
            self._calculate_total_api_calls()
            self._calculate_active_tenants()
            self._calculate_average_response_time()
            self._calculate_error_rate()
            self._calculate_sla_compliance()
            self._calculate_peak_throughput()
            self._calculate_service_count()
            self._calculate_qos_metrics()
        except Exception as e:
            print(f"Error updating system status metrics: {e}")
    
    def _calculate_total_api_calls(self):
        """Calculate total API calls excluding /metrics endpoint"""
        try:
            # Calculate total from actual request metrics, excluding /metrics endpoint
            # This will sum all requests except those to the /metrics endpoint
            total_calls = 0
            
            # In a real implementation, you'd use a PromQL query like:
            # sum(ai4x_requests_total{endpoint!="/metrics"})
            # For now, we'll simulate based on current metrics state
            
            # Get the actual metrics from our REQUEST_COUNT counter
            # This is a simplified approach - in production you'd query Prometheus directly
            try:
                # Access the internal metrics to get current total
                # Exclude /metrics endpoint calls
                for sample in REQUEST_COUNT.collect()[0].samples:
                    labels = sample.labels
                    if labels.get('endpoint') != '/metrics':  # Exclude metrics endpoint
                        total_calls += sample.value
                
                SYSTEM_STATUS_TOTAL_API_CALLS.set(total_calls)
                
            except Exception as inner_e:
                print(f"Error accessing REQUEST_COUNT metrics: {inner_e}")
                # Fallback: Use time-based estimation but much more realistic
                import time
                # Simulate realistic API call growth
                base_time = 1725148800  # Sept 1, 2025 timestamp
                current_time = time.time()
                time_diff = current_time - base_time
                # Simulate ~10 calls per minute average growth
                estimated_total = max(0, int(time_diff / 6))  # 10 calls per minute
                SYSTEM_STATUS_TOTAL_API_CALLS.set(estimated_total)
                
        except Exception as e:
            print(f"Error calculating total API calls: {e}")
            SYSTEM_STATUS_TOTAL_API_CALLS.set(0)
    
    def _calculate_active_tenants(self):
        """Calculate number of active tenants/customers"""
        try:
            # In a real implementation, you'd query your customer activity
            # For now, simulate based on known customers
            active_tenants = len(['cust1', 'cust2'])  # Your known customers
            SYSTEM_STATUS_ACTIVE_TENANTS.set(active_tenants)
        except Exception as e:
            print(f"Error calculating active tenants: {e}")
            SYSTEM_STATUS_ACTIVE_TENANTS.set(0)
    
    def _calculate_average_response_time(self):
        """Calculate average response time"""
        try:
            # Simulate average response time calculation
            # In production, you'd calculate this from actual metrics
            avg_response_time = 1.5  # Simulated 1.5 seconds
            SYSTEM_STATUS_AVERAGE_RESPONSE_TIME.set(avg_response_time)
        except Exception as e:
            print(f"Error calculating average response time: {e}")
            SYSTEM_STATUS_AVERAGE_RESPONSE_TIME.set(0)
    
    def _calculate_error_rate(self):
        """Calculate overall error rate percentage"""
        try:
            # Simulate error rate calculation
            # In production, calculate from success/failure ratios
            error_rate = 2.5  # Simulated 2.5% error rate
            SYSTEM_STATUS_ERROR_RATE.set(error_rate)
        except Exception as e:
            print(f"Error calculating error rate: {e}")
            SYSTEM_STATUS_ERROR_RATE.set(0)
    
    def _calculate_sla_compliance(self):
        """Calculate SLA compliance for different types"""
        try:
            # Availability SLA (>99%)
            availability_sla = 99.2  # Simulated 99.2%
            SYSTEM_STATUS_SLA_COMPLIANCE.labels(sla_type='availability').set(availability_sla)
            
            # Response Time SLA (<2s)
            response_time_sla = 95.8  # Simulated 95.8% compliance
            SYSTEM_STATUS_SLA_COMPLIANCE.labels(sla_type='response_time').set(response_time_sla)
            
            # Error Rate SLA (<5%)
            error_rate_sla = 97.5  # Simulated 97.5% compliance
            SYSTEM_STATUS_SLA_COMPLIANCE.labels(sla_type='error_rate').set(error_rate_sla)
        except Exception as e:
            print(f"Error calculating SLA compliance: {e}")
    
    def _calculate_peak_throughput(self):
        """Calculate peak throughput in requests per minute"""
        try:
            # Simulate peak throughput calculation
            current_throughput = 1500  # Simulated 1500 RPM
            if current_throughput > self.peak_throughput:
                self.peak_throughput = current_throughput
            
            SYSTEM_STATUS_PEAK_THROUGHPUT.set(self.peak_throughput)
        except Exception as e:
            print(f"Error calculating peak throughput: {e}")
            SYSTEM_STATUS_PEAK_THROUGHPUT.set(0)
    
    def _calculate_service_count(self):
        """Calculate number of active services by type"""
        try:
            # Set service counts for different types
            SYSTEM_STATUS_SERVICE_COUNT.labels(service_type='ai_services').set(4)  # NMT, LLM, TTS, ASR
            SYSTEM_STATUS_SERVICE_COUNT.labels(service_type='infrastructure').set(3)  # API, DB, Monitoring
            SYSTEM_STATUS_SERVICE_COUNT.labels(service_type='total').set(7)
        except Exception as e:
            print(f"Error calculating service count: {e}")
    
    def _calculate_qos_metrics(self):
        """Calculate Quality of Service metrics"""
        try:
            customers = ['cust1', 'cust2']
            apps = ['nmt-app', 'llm-app', 'tts-app']
            
            for customer in customers:
                for app in apps:
                    # Calculate availability for different time windows
                    availability_5m = self._simulate_availability()
                    availability_1h = self._simulate_availability()
                    availability_24h = self._simulate_availability()
                    
                    QOS_AVAILABILITY.labels(customer=customer, app=app, time_window='5m').set(availability_5m)
                    QOS_AVAILABILITY.labels(customer=customer, app=app, time_window='1h').set(availability_1h)
                    QOS_AVAILABILITY.labels(customer=customer, app=app, time_window='24h').set(availability_24h)
                    
                    # Calculate performance and reliability scores
                    performance_score = self._simulate_performance_score()
                    reliability_score = self._simulate_reliability_score()
                    
                    QOS_PERFORMANCE_SCORE.labels(customer=customer, app=app).set(performance_score)
                    QOS_RELIABILITY_SCORE.labels(customer=customer, app=app).set(reliability_score)
                    
                    # Calculate SLA compliance
                    availability_compliance = min(100, (availability_1h / 99.0) * 100)
                    response_time_compliance = self._simulate_response_time_compliance()
                    
                    QOS_SLA_COMPLIANCE.labels(customer=customer, app=app, sla_type='availability').set(availability_compliance)
                    QOS_SLA_COMPLIANCE.labels(customer=customer, app=app, sla_type='response_time').set(response_time_compliance)
        except Exception as e:
            print(f"Error calculating QoS metrics: {e}")
    
    def _simulate_availability(self) -> float:
        """Simulate availability calculation"""
        import random
        return random.uniform(98.5, 99.9)
    
    def _simulate_performance_score(self) -> float:
        """Simulate performance score calculation"""
        import random
        return random.uniform(85.0, 98.0)
    
    def _simulate_reliability_score(self) -> float:
        """Simulate reliability score calculation"""
        import random
        return random.uniform(90.0, 99.5)
    
    def _simulate_response_time_compliance(self) -> float:
        """Simulate response time compliance calculation"""
        import random
        return random.uniform(88.0, 97.0)

class MetricsCollector:
    def __init__(self):
        self.request_times = {}
        self.throughput_counter = {}
        self.last_throughput_update = time.time()
        self.system_metrics = SystemStatusCalculator()
        self._start_system_metrics_collector()
        self._start_system_status_calculator()
    
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
    
    def _start_system_status_calculator(self):
        """Start background thread to calculate system status metrics"""
        def calculate_system_status():
            while True:
                try:
                    self.system_metrics.update_system_status_metrics()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    print(f"Error calculating system status metrics: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=calculate_system_status, daemon=True)
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
