import psycopg2
import psycopg2.extras
import statistics
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
from metrics import prometheus_latest_text, metrics_collector
from config import Config
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import threading

# --- Prometheus core ---
# from prometheus_client import Counter, Histogram, Gauge

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

TABLE_NAME = "ai4x_demo_requests_log_v6"

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
            backNMTLatency TEXT,
            ttsLatency TEXT,
            overallPipelineLatency TEXT,
            nmtUsage TEXT,
            llmUsage TEXT,
            backNMTUsage TEXT,
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics endpoint
@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=prometheus_latest_text(), media_type="text/plain")


class PipelineInput(BaseModel):
    customerName: str
    customerAppName: str
    input: Dict[str, str]

# ----------------------------
# Pipeline Endpoint
# ----------------------------
@app.post("/pipeline")
def run_pipeline(payload: PipelineInput):
    usage = {"NMT": None, "LLM": None, "TTS": None, "backNMT": None}
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

    success = True
    response_data = None
    
    # Track success status of each service
    service_success = {"NMT": False, "LLM": False, "BackNMT": False, "TTS": False}

    with metrics_collector.request_timer(customer, appname, "/pipeline") as rid:
        try:
            # ---- Language Detection ----
            start = time.time()
            source_language = detect_language_from_text(input_text)
            latencies["LangDetection"] = f"{int((time.time() - start) * 1000)}ms"

            current_output = input_text

            # ---- NMT ----
            if "NMT" in pipeline:
                with metrics_collector.component_timer(rid, "NMT"):
                    usage["NMT"] = str(len(current_output))
                    nmt_result = nmt_service.translate_text(current_output, source_language, target_language)
                    latencies["NMT"] = f"{int((time.time() - start) * 1000)}ms"
                    pipeline_output["NMT"] = nmt_result["translated_text"]
                    current_output = nmt_result["translated_text"]
                    # Track success and only log metrics if NMT was successful
                    service_success["NMT"] = nmt_result.get("success", False)
                    if service_success["NMT"]:
                        metrics_collector.nmt_chars(customer, appname, source_language, target_language, len(nmt_result["translated_text"]))

            # ---- LLM ----
            if "LLM" in pipeline:
                with metrics_collector.component_timer(rid, "LLM"):
                    llm_result = llm_service.process_query(current_output)
                    latencies["LLM"] = f"{int((time.time() - start) * 1000)}ms"
                    pipeline_output["LLM"] = llm_result["response"]
                    response_data = llm_result["response"]
                    usage["LLM"] = str(llm_result["total_tokens"])
                    # Track success and only log metrics if LLM was successful
                    service_success["LLM"] = llm_result.get("success", False)
                    if service_success["LLM"]:
                        metrics_collector.llm_tokens(customer, appname, "gemini-2.5-flash", llm_result["total_tokens"])

                    # Backward NMT
                    with metrics_collector.component_timer(rid, "BackNMT"):
                        usage["backNMT"] = str(len(llm_result["response"]))
                        back_translation = nmt_service.translate_text(
                            text=llm_result["response"],
                            source_lang="en",
                            target_lang=source_language
                        )
                        latencies["BackNMT"] = f"{int((time.time() - start) * 1000)}ms"
                        pipeline_output["BackNMT"] = back_translation["translated_text"]
                        # Track success and only log metrics if BackNMT was successful
                        service_success["BackNMT"] = back_translation.get("success", False)
                        if service_success["BackNMT"]:
                            metrics_collector.nmt_chars(customer, appname, "en", source_language, len(back_translation["translated_text"]))
                        response_data = back_translation["translated_text"]

            # ---- TTS ----
            if "TTS" in pipeline:
                with metrics_collector.component_timer(rid, "TTS"):
                    usage["TTS"] = str(len(response_data))
                    tts_result = tts_service.text_to_speech(response_data, source_language, gender="female")
                    latencies["TTS"] = f"{int((time.time() - start) * 1000)}ms"
                    pipeline_output["TTS"] = tts_result["audio_content"]
                    # Track success and only log metrics if TTS was successful
                    service_success["TTS"] = tts_result.get("success", False)
                    if service_success["TTS"]:
                        metrics_collector.tts_chars(customer, appname, source_language, len(response_data))
                    response_data = tts_result["audio_content"]

            # ---- DB Logging ----
            # Only log metrics for successful services
            conn = get_connection()
            cur = conn.cursor()
            
            # Only log metrics for services that were successful
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, langdetectionLatency, nmtLatency, llmLatency, backNMTLatency, ttsLatency, overallPipelineLatency,
                nmtUsage, llmUsage, backNMTUsage, ttsUsage, timestamp)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                latencies.get("LangDetection", None),
                latencies.get("NMT", None) if service_success["NMT"] else None,
                latencies.get("LLM", None) if service_success["LLM"] else None,
                latencies.get("BackNMT", None) if service_success["BackNMT"] else None,
                latencies.get("TTS", None) if service_success["TTS"] else None,
                latencies.get("pipelineTotal", None),
                str(usage.get("NMT", None)) if service_success["NMT"] else None,
                str(usage.get("LLM", None)) if service_success["LLM"] else None,
                str(usage.get("backNMT", None)) if service_success["BackNMT"] else None,
                str(usage.get("TTS", None)) if service_success["TTS"] else None,
                datetime.now(timezone.utc)
            ))
            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            success = False
            raise HTTPException(status_code=500, detail=str(e))

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
               langdetectionLatency, nmtLatency, llmLatency, backNMTLatency, ttsLatency, overallPipelineLatency,
               nmtUsage, llmUsage, backNMTUsage, ttsUsage
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
                "langdetectionlatency", "nmtlatency", "llmlatency", "backnmtlatency", "ttslatency", "overallpipelinelatency",
                "nmtusage", "llmusage", "backnmtusage", "ttsusage"
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
            "avg_backNMTLatency": safe_mean(metrics["backnmtlatency"]),
            "avg_ttsLatency": safe_mean(metrics["ttslatency"]),
            "avg_overallPipelineLatency": safe_mean(metrics["overallpipelinelatency"]),
            "avg_nmtUsage": safe_mean(metrics["nmtusage"]),
            "avg_llmUsage": safe_mean(metrics["llmusage"]),
            "avg_backNMTUsage": safe_mean(metrics["backnmtusage"]),
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

            "p90_backNMTLatency": safe_percentile(metrics["backnmtlatency"], 90),
            "p95_backNMTLatency": safe_percentile(metrics["backnmtlatency"], 95),
            "p99_backNMTLatency": safe_percentile(metrics["backnmtlatency"], 99),

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



@app.get("/metrics/requests")
def get_request_metrics():
    """
    Returns request volume and API call counts:
    - Total number of API calls
    - API calls split by service (TTS, NMT, LLM, backNMT)
    - Customer-wise counts (total + per service)
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Fetch all requests
    cur.execute(f"""
        SELECT customerName, 
               (nmtLatency IS NOT NULL) AS hasNMT,
               (llmLatency IS NOT NULL) AS hasLLM,
               (ttsLatency IS NOT NULL) AS hasTTS,
               (backNMTLatency IS NOT NULL) AS hasBackNMT
        FROM {TABLE_NAME}
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    total_requests = len(rows)
    service_counts = {"NMT": 0, "LLM": 0, "TTS": 0, "backNMT": 0}
    customer_requests = {}

    for row in rows:
        cust = row["customername"]

        # Initialize per-customer dict if first time
        if cust not in customer_requests:
            customer_requests[cust] = {
                "total": 0,
                "by_service": {"NMT": 0, "LLM": 0, "TTS": 0, "backNMT": 0}
            }

        # Increment totals
        customer_requests[cust]["total"] += 1

        if row["hasnmt"]:
            service_counts["NMT"] += 1
            customer_requests[cust]["by_service"]["NMT"] += 1
        if row["hasllm"]:
            service_counts["LLM"] += 1
            customer_requests[cust]["by_service"]["LLM"] += 1
        if row["hastts"]:
            service_counts["TTS"] += 1
            customer_requests[cust]["by_service"]["TTS"] += 1
        if row["hasbacknmt"]:
            service_counts["backNMT"] += 1
            customer_requests[cust]["by_service"]["backNMT"] += 1

    return {
        "total_requests": total_requests,
        "requests_by_service": service_counts,
        "requests_by_customer": customer_requests
    }



@app.get("/metrics/data_processed")
def get_data_processed_metrics():
    """
    Returns data processed metrics:
    - NMT = total characters translated
    - TTS = total characters synthesized
    - LLM = total input tokens
    - backNMT = total characters back-translated
    Both total and customer-wise
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(f"""
        SELECT customerName,
               nmtUsage, llmUsage, ttsUsage, backNMTUsage
        FROM {TABLE_NAME}
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    totals = {"NMT_chars": 0, "LLM_tokens": 0, "TTS_chars": 0, "backNMT_chars": 0}
    customers = {}

    for row in rows:
        cust = row["customername"]
        if cust not in customers:
            customers[cust] = {"NMT_chars": 0, "LLM_tokens": 0, "TTS_chars": 0, "backNMT_chars": 0}

        def add_safe(val):
            try:
                return int(val) if val not in (None, "None") else 0
            except:
                return 0

        nmt = add_safe(row["nmtusage"])
        llm = add_safe(row["llmusage"])
        tts = add_safe(row["ttsusage"])
        backnmt = add_safe(row["backnmtusage"])

        customers[cust]["NMT_chars"] += nmt
        customers[cust]["LLM_tokens"] += llm
        customers[cust]["TTS_chars"] += tts
        customers[cust]["backNMT_chars"] += backnmt

        totals["NMT_chars"] += nmt
        totals["LLM_tokens"] += llm
        totals["TTS_chars"] += tts
        totals["backNMT_chars"] += backnmt

    return {
        "totals": totals,
        "byCustomer": customers
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai4x_demo_app:app", host="0.0.0.0", port=8000, reload=False)