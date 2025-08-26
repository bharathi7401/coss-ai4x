import psycopg2
import psycopg2.extras
import statistics
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from services.nmt_service import NMTService
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.weather_service import WeatherService
from config import Config
import threading

# --- Prometheus core ---
from prometheus_client import Counter, Histogram, Gauge

# ----------------------------
# Database connection
# ----------------------------
DB_CONFIG = {
    "user": "ai4xadmin",
    "password": "password123#",  # replace with your actual password or use env var
    "host": "ai4x.postgres.database.azure.com",
    "port": 5432,
    "database": "ai4xdemo"
}

nmt_service = NMTService()
llm_service = LLMService()
tts_service = TTSService()
weather_service = WeatherService()

TABLE_NAME = "ai4x_demo_requests_log_v4"

# ----------------------------
# Pipeline-level Prometheus Metrics
# # ----------------------------
# REQUEST_COUNT = Counter(
#     "pipeline_requests_total",
#     "Total number of pipeline requests",
#     ["customerName", "customerApp", "requestId", "status"]
# )

# REQUEST_LATENCY = Histogram(
#     "pipeline_request_latency_seconds",
#     "Latency of pipeline requests in seconds",
#     ["customerName", "customerApp", "requestId"]
# )

# THROUGHPUT = Counter(
#     "pipeline_throughput_total",
#     "Total pipeline requests processed (for throughput/sec calculation)",
#     ["customerName", "customerApp", "requestId"]
# )

# ERROR_COUNT = Counter(
#     "pipeline_failed_requests_total",
#     "Total failed pipeline requests",
#     ["customerName", "customerApp", "requestId"]
# )

# # ----------------------------
# # Service-level Prometheus Metrics
# # ----------------------------
# SERVICE_LATENCY = Histogram(
#     "service_latency_seconds",
#     "Latency of individual services in seconds",
#     ["customerName", "customerApp", "requestId", "service"]
# )

# SERVICE_THROUGHPUT = Counter(
#     "service_throughput_total",
#     "Total requests processed per service",
#     ["customerName", "customerApp", "requestId", "service"]
# )

# SERVICE_ERRORS = Counter(
#     "service_failed_requests_total",
#     "Total failed service requests",
#     ["customerName", "customerApp", "requestId", "service"]
# )

# # ----------------------------
# # System Metrics
# # ----------------------------
# CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percent")
# MEMORY_USAGE = Gauge("system_memory_usage_percent", "Memory usage percent")

# ----------------------------
# Utility Functions
# ----------------------------
def detect_language_from_text(text: str) -> str:
    text_lower = text.lower().strip()
    if any('\u0900' <= char <= '\u097F' for char in text): return "hi"
    if any('\u0B80' <= char <= '\u0BFF' for char in text): return "ta"
    if any('\u0C00' <= char <= '\u0C7F' for char in text): return "te"
    if any('\u0980' <= char <= '\u09FF' for char in text): return "bn"
    if any('\u0D00' <= char <= '\u0D7F' for char in text): return "ml"
    if any('\u0C80' <= char <= '\u0CFF' for char in text): return "kn"
    if any('\u0A80' <= char <= '\u0AFF' for char in text): return "gu"
    if any(word in text for word in ['का', 'आहे', 'मी', 'तू', 'ते', 'हे', 'या', 'ना']): return "mr"
    if any('\u0A00' <= char <= '\u0A7F' for char in text): return "pa"
    if any(word in text_lower for word in ['namaste', 'kaise', 'hai', 'mein', 'aap', 'kya', 'haan', 'nahi', 'dhanyawad']): return "hi"
    if any(word in text_lower for word in ['vanakkam', 'eppadi', 'irukkirathu', 'nandri', 'enna', 'illai', 'aam']): return "ta"
    return "en"

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            requestId TEXT PRIMARY KEY,
            customerName TEXT,
            customerApp TEXT,
            langdetectionLatency TEXT,
            nmtLatency TEXT,
            llmLatency TEXT,
            ttsLatency TEXT,
            overallPipelineLatency TEXT,
            nmtUsage TEXT,
            llmUsage TEXT,
            ttsUsage TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

PIPELINES = {
    "cust1": ["NMT", "LLM", "TTS"],
    "cust2": ["NMT", "LLM"]
}

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()
# Instrumentator().instrument(app).expose(app)

class PipelineInput(BaseModel):
    customerName: str
    customerAppName: str
    input: Dict[str, str]

# ----------------------------
# Pipeline Endpoint
# ----------------------------
@app.post("/pipeline")
def run_pipeline(payload: PipelineInput):
    usage = {"NMT": None, "LLM": None, "TTS": None}
    customer = payload.customerName
    appname = payload.customerAppName
    pipeline = PIPELINES.get(customer.lower()) or PIPELINES.get(customer)
    if not pipeline:
        raise HTTPException(status_code=400, detail=f"No pipeline defined for customer {customer}")

    request_id = str(uuid.uuid4())
    latencies = {}
    pipeline_output = {}
    input_text = payload.input.get("text", "")
    target_language = payload.input.get("language", "en")

    start_pipeline = time.time()
    success = True

    try:
        # ---- Language Detection ----
        start = time.time()
        source_language = detect_language_from_text(input_text)
        latencies["LangDetection"] = f"{int((time.time() - start) * 1000)}ms"
        duration = time.time() - start
        # SERVICE_LATENCY.labels(customer, appname, request_id, "LangDetection").observe(duration)
        # SERVICE_THROUGHPUT.labels(customer, appname, request_id, "LangDetection").inc()

        current_output = input_text

        # ---- NMT ----
        if "NMT" in pipeline:
            start = time.time()
            try:
                usage["NMT"] = str(len(current_output))
                current_output = nmt_service.translate_text(current_output, source_language, target_language)
                latencies["NMT"] = f"{int((time.time() - start) * 1000)}ms"
                pipeline_output["NMT"] = current_output["translated_text"]

                duration = time.time() - start
                # SERVICE_LATENCY.labels(customer, appname, request_id, "NMT").observe(duration)
                # SERVICE_THROUGHPUT.labels(customer, appname, request_id, "NMT").inc()
            except Exception:
                # SERVICE_ERRORS.labels(customer, appname, request_id, "NMT").inc()
                raise

        # ---- LLM ----
        if "LLM" in pipeline:
            start = time.time()
            try:
                current_output = llm_service.process_query(current_output)
                latencies["LLM"] = f"{int((time.time() - start) * 1000)}ms"
                pipeline_output["LLM"] = current_output["response"]
                response_data = current_output["response"]
                usage["LLM"] = str(current_output["total_tokens"])

                duration = time.time() - start
                # SERVICE_LATENCY.labels(customer, appname, request_id, "LLM").observe(duration)
                # SERVICE_THROUGHPUT.labels(customer, appname, request_id, "LLM").inc()
            except Exception:
                # SERVICE_ERRORS.labels(customer, appname, request_id, "LLM").inc()
                raise

        # ---- TTS ----
        if "TTS" in pipeline:
            start = time.time()
            try:
                usage["TTS"] = str(len(current_output["response"]))
                current_output = tts_service.text_to_speech(current_output["response"], target_language, gender="female")
                latencies["TTS"] = f"{int((time.time() - start) * 1000)}ms"
                pipeline_output["TTS"] = current_output["audio_content"]
                response_data = current_output["audio_content"]

                duration = time.time() - start
                # SERVICE_LATENCY.labels(customer, appname, request_id, "TTS").observe(duration)
                # SERVICE_THROUGHPUT.labels(customer, appname, request_id, "TTS").inc()
            except Exception:
                # SERVICE_ERRORS.labels(customer, appname, request_id, "TTS").inc()
                raise

        # ---- Pipeline total ----
        total_elapsed = int((time.time() - start_pipeline) * 1000)
        latencies["pipelineTotal"] = f"{total_elapsed}ms"
        duration = time.time() - start_pipeline
        # SERVICE_LATENCY.labels(customer, appname, request_id, "PipelineTotal").observe(duration)
        # SERVICE_THROUGHPUT.labels(customer, appname, request_id, "PipelineTotal").inc()

        # ---- DB Logging ----
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO {TABLE_NAME} 
            (requestId, customerName, customerApp, langdetectionLatency, nmtLatency, llmLatency, ttsLatency, overallPipelineLatency,
            nmtUsage, llmUsage, ttsUsage, timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            request_id, customer, appname,
            latencies.get("LangDetection", None),
            latencies.get("NMT", None),
            latencies.get("LLM", None),
            latencies.get("TTS", None),
            latencies.get("pipelineTotal", None),
            str(usage.get("NMT", None)),
            str(usage.get("LLM", None)),
            str(usage.get("TTS", None)),
            datetime.now(timezone.utc)
        ))
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        success = False
        # ERROR_COUNT.labels(customer, appname, request_id).inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        duration = time.time() - start_pipeline
        # REQUEST_LATENCY.labels(customer, appname, request_id).observe(duration)
        status = "success" if success else "error"
        # REQUEST_COUNT.labels(customer, appname, request_id, status).inc()
        # THROUGHPUT.labels(customer, appname, request_id).inc()
        # CPU_USAGE.set(psutil.cpu_percent())
        # MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return {
        "requestId": request_id,
        "status": "success" if success else "error",
        "pipelineOutput": pipeline_output,
        "responseData": response_data if success else None,
        "latency": latencies,
        "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ----------------------------
# Metrics & DB Endpoints
# ----------------------------
# @app.get("/metrics/system")
# def get_system_metrics():
#     CPU_USAGE.set(psutil.cpu_percent())
#     MEMORY_USAGE.set(psutil.virtual_memory().percent)
#     memory = psutil.virtual_memory()
#     return {
#         "cpu_usage_percent": psutil.cpu_percent(),
#         "memory_usage_bytes": memory.used,
#         "memory_usage_percent": memory.percent,
#         "memory_total_bytes": memory.total,
#         "memory_available_bytes": memory.available
#     }

# (customers endpoints remain same as in your code)

@app.get("/customers")
def get_all_customers():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(f"SELECT * FROM {TABLE_NAME}")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

@app.get("/customers/{customerName}")
def get_customer_by_name(customerName: str):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE customerName = %s", (customerName,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.get("/customer_aggregates")
def get_customer_aggregates(customerName: str):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(f"""
        SELECT customerName, customerApp,
               langdetectionLatency, nmtLatency, llmLatency, ttsLatency, overallPipelineLatency,
               nmtUsage, llmUsage, ttsUsage
        FROM {TABLE_NAME}
        WHERE customerName = %s
    """, (customerName,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"customerName": customerName, "aggregates": []}

    # Group data by customerApp
    grouped = {}
    for row in rows:
        app_key = row["customerapp"]
        if app_key not in grouped:
            grouped[app_key] = {k: [] for k in [
                "langdetectionlatency", "nmtlatency", "llmlatency", "ttslatency", "overallpipelinelatency",
                "nmtusage", "llmusage", "ttsusage"
            ]}
        for col in grouped[app_key]:
            val = row.get(col)
            if val is None:
                grouped[app_key][col].append(None)
            else:
                try:
                    # Remove "ms" suffix if exists, convert to float
                    if isinstance(val, str) and val.endswith("ms"):
                        num = float(val.replace("ms", "").strip())
                    else:
                        num = float(val)
                    grouped[app_key][col].append(num)
                except Exception:
                    grouped[app_key][col].append(None)

    # Helpers for mean and percentiles
    def safe_mean(values):
        nums = [v for v in values if v is not None]
        return float(np.mean(nums)) if nums else None

    def safe_percentile(values, q):
        nums = [v for v in values if v is not None]
        return float(np.percentile(nums, q)) if nums else None

    # Compute aggregates per app
    aggregates = []
    for app_key, metrics in grouped.items():
        aggregates.append({
            "customerName": customerName,
            "customerApp": app_key,

            # Mean
            "avg_langdetectionLatency": safe_mean(metrics["langdetectionlatency"]),
            "avg_nmtLatency": safe_mean(metrics["nmtlatency"]),
            "avg_llmLatency": safe_mean(metrics["llmlatency"]),
            "avg_ttsLatency": safe_mean(metrics["ttslatency"]),
            "avg_overallPipelineLatency": safe_mean(metrics["overallpipelinelatency"]),
            "avg_nmtUsage": safe_mean(metrics["nmtusage"]),
            "avg_llmUsage": safe_mean(metrics["llmusage"]),
            "avg_ttsUsage": safe_mean(metrics["ttsusage"]),

            # Percentiles
            "p90_langdetectionLatency": safe_percentile(metrics["langdetectionlatency"], 90),
            "p95_langdetectionLatency": safe_percentile(metrics["langdetectionlatency"], 95),
            "p99_langdetectionLatency": safe_percentile(metrics["langdetectionlatency"], 99),

            "p90_nmtLatency": safe_percentile(metrics["nmtlatency"], 90),
            "p95_nmtLatency": safe_percentile(metrics["nmtlatency"], 95),
            "p99_nmtLatency": safe_percentile(metrics["nmtlatency"], 99),

            "p90_llmLatency": safe_percentile(metrics["llmlatency"], 90),
            "p95_llmLatency": safe_percentile(metrics["llmlatency"], 95),
            "p99_llmLatency": safe_percentile(metrics["llmlatency"], 99),

            "p90_ttsLatency": safe_percentile(metrics["ttslatency"], 90),
            "p95_ttsLatency": safe_percentile(metrics["ttslatency"], 95),
            "p99_ttsLatency": safe_percentile(metrics["ttslatency"], 99),

            "p90_overallPipelineLatency": safe_percentile(metrics["overallpipelinelatency"], 90),
            "p95_overallPipelineLatency": safe_percentile(metrics["overallpipelinelatency"], 95),
            "p99_overallPipelineLatency": safe_percentile(metrics["overallpipelinelatency"], 99),
        })

    return {
        "customerName": customerName,
        "aggregates": aggregates
    }
