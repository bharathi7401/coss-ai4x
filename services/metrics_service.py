# import time
# import psutil
# import tiktoken
# from prometheus_client import (
#     Counter, Histogram, Gauge, Summary, 
#     generate_latest, CONTENT_TYPE_LATEST,
#     REGISTRY, push_to_gateway
# )
# from typing import Dict, Any, Optional
# import threading
# from collections import deque
# import statistics
# import requests

# class MetricsService:
#     def __init__(self, prometheus_url: str = "http://35.207.227.225:9090"):
#         # Initialize Prometheus metrics
#         self.prometheus_url = prometheus_url
#         self._init_metrics()
        
#         # Thread-safe storage for latency percentiles
#         self.latency_data = {
#             'lang_detection': deque(maxlen=1000),
#             'nmt': deque(maxlen=1000),
#             'llm': deque(maxlen=1000),
#             'tts': deque(maxlen=1000),
#             'pipeline_total': deque(maxlen=1000)
#         }
#         self.lock = threading.Lock()
        
    
#     def _init_metrics(self):
#         """Initialize all Prometheus metrics"""
        
#         # Latency metrics (Histograms for percentiles)
#         self.lang_detection_latency = Histogram(
#             'lang_detection_latency_seconds',
#             'Language detection latency in seconds',
#             buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
#         )
        
#         self.nmt_latency = Histogram(
#             'nmt_latency_seconds',
#             'NMT translation latency in seconds',
#             buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0]
#         )
        
#         self.llm_latency = Histogram(
#             'llm_latency_seconds',
#             'LLM processing latency in seconds',
#             buckets=[0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0]
#         )
        
#         self.tts_latency = Histogram(
#             'tts_latency_seconds',
#             'TTS processing latency in seconds',
#             buckets=[0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0]
#         )
        
#         self.pipeline_total_latency = Histogram(
#             'pipeline_total_latency_seconds',
#             'Total pipeline latency in seconds',
#             buckets=[1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0]
#         )
        
#         # Usage metrics (Counters)
#         self.nmt_usage_chars = Counter(
#             'nmt_usage_characters_total',
#             'Total number of characters processed by NMT',
#             ['customer', 'app']
#         )
        
#         self.llm_usage_tokens = Counter(
#             'llm_usage_tokens_total',
#             'Total number of tokens processed by LLM',
#             ['customer', 'app']
#         )
        
#         self.tts_usage_chars = Counter(
#             'tts_usage_characters_total',
#             'Total number of characters processed by TTS',
#             ['customer', 'app']
#         )
        
#         # Throughput metrics (Counters)
#         self.requests_total = Counter(
#             'pipeline_requests_total',
#             'Total number of pipeline requests',
#             ['customer', 'app', 'status']
#         )
        
#         self.requests_per_second = Counter(
#             'pipeline_requests_per_second',
#             'Pipeline requests per second',
#             ['customer', 'app']
#         )
        
#         # Error rate metrics
#         self.errors_total = Counter(
#             'pipeline_errors_total',
#             'Total number of pipeline errors',
#             ['customer', 'app', 'error_type']
#         )
        
#         # System utilization metrics (Gauges)
#         self.cpu_usage = Gauge(
#             'system_cpu_usage_percent',
#             'CPU usage percentage'
#         )
        
#         self.memory_usage = Gauge(
#             'system_memory_usage_bytes',
#             'Memory usage in bytes'
#         )
        
#         self.memory_usage_percent = Gauge(
#             'system_memory_usage_percent',
#             'Memory usage percentage'
#         )
        
#         # GPU metrics (if available)
    
#     def record_latency(self, service: str, latency_ms: float, customer: str = "unknown", app: str = "unknown"):
#         """Record latency for a service"""
#         latency_seconds = latency_ms / 1000.0
        
#         # Record in Prometheus histogram
#         if service == 'lang_detection':
#             self.lang_detection_latency.observe(latency_seconds)
#         elif service == 'nmt':
#             self.nmt_latency.observe(latency_seconds)
#         elif service == 'llm':
#             self.llm_latency.observe(latency_seconds)
#         elif service == 'tts':
#             self.tts_latency.observe(latency_seconds)
#         elif service == 'pipeline_total':
#             self.pipeline_total_latency.observe(latency_seconds)
        
#         # Store for percentile calculations
#         with self.lock:
#             if service in self.latency_data:
#                 self.latency_data[service].append(latency_ms)
    
#     def record_usage(self, service: str, usage_value: int, customer: str = "unknown", app: str = "unknown"):
#         """Record usage metrics for a service"""
#         if service == 'nmt':
#             self.nmt_usage_chars.labels(customer=customer, app=app).inc(usage_value)
#         elif service == 'llm':
#             self.llm_usage_tokens.labels(customer=customer, app=app).inc(usage_value)
#         elif service == 'tts':
#             self.tts_usage_chars.labels(customer=customer, app=app).inc(usage_value)
    
#     def record_request(self, customer: str = "unknown", app: str = "unknown", status: str = "success"):
#         """Record a pipeline request"""
#         self.requests_total.labels(customer=customer, app=app, status=status).inc()
#         self.requests_per_second.labels(customer=customer, app=app).inc()
    
#     def record_error(self, error_type: str, customer: str = "unknown", app: str = "unknown"):
#         """Record an error"""
#         self.errors_total.labels(customer=customer, app=app, error_type=error_type).inc()
    
#     def count_tokens(self, text: str) -> int:
#         """Count tokens using tiktoken"""
#         if self.tokenizer and text:
#             return len(self.tokenizer.encode(text))
#         return len(text.split())  # Fallback to word count
    
#     def update_system_metrics(self):
#         """Update system utilization metrics"""
#         # CPU usage
#         cpu_percent = psutil.cpu_percent(interval=1)
#         self.cpu_usage.set(cpu_percent)
        
#         # Memory usage
#         memory = psutil.virtual_memory()
#         self.memory_usage.set(memory.used)
#         self.memory_usage_percent.set(memory.percent)
    
#     def push_metrics_to_prometheus(self, job_name: str = "ai4x-pipeline"):
#         """Push metrics to Prometheus pushgateway"""
#         try:
#             push_to_gateway(
#                 f"{self.prometheus_url}:9091",  # Pushgateway typically runs on port 9091
#                 job=job_name,
#                 registry=REGISTRY
#             )
#             return True
#         except Exception as e:
#             print(f"Failed to push metrics to Prometheus: {e}")
#             return False
    
#     def get_percentiles(self, service: str) -> Dict[str, float]:
#         """Get latency percentiles for a service"""
#         with self.lock:
#             if service not in self.latency_data or not self.latency_data[service]:
#                 return {}
            
#             data = list(self.latency_data[service])
#             if len(data) < 2:
#                 return {}
            
#             return {
#                 'p50': statistics.median(data),
#                 'p90': statistics.quantiles(data, n=10)[8],  # 90th percentile
#                 'p95': statistics.quantiles(data, n=20)[18],  # 95th percentile
#                 'p99': statistics.quantiles(data, n=100)[98] if len(data) >= 100 else statistics.quantiles(data, n=20)[19]  # 99th percentile
#             }
    
#     def get_all_percentiles(self) -> Dict[str, Dict[str, float]]:
#         """Get percentiles for all services"""
#         return {
#             service: self.get_percentiles(service)
#             for service in self.latency_data.keys()
#         }
    
#     def get_metrics(self):
#         """Get all metrics in Prometheus format"""
#         return generate_latest(REGISTRY)
    
#     def get_metrics_content_type(self):
#         """Get content type for metrics"""
#         return CONTENT_TYPE_LATEST

# # Global metrics service instance
# metrics_service = MetricsService()