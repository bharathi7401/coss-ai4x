import psycopg2
import psycopg2.extras
import statistics
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional, Dict, Any
from services.nmt_service import NMTService
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.weather_service import WeatherService
from config import Config
from metrics import metrics_collector, get_metrics

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

TABLE_NAME = "ai4x_demo_requests_log_v2"

def detect_language_from_text(text: str) -> str:
    """
    Detect language from text input using simple heuristics and character patterns
    
    Args:
        text: Input text to analyze
        
    Returns:
        Detected language code
    """
    text_lower = text.lower().strip()
    
    # Hindi detection (Devanagari script)
    if any('\u0900' <= char <= '\u097F' for char in text):
        return "hi"
    
    # Tamil detection (Tamil script)
    if any('\u0B80' <= char <= '\u0BFF' for char in text):
        return "ta"
    
    # Telugu detection (Telugu script)
    if any('\u0C00' <= char <= '\u0C7F' for char in text):
        return "te"
    
    # Bengali detection (Bengali script)
    if any('\u0980' <= char <= '\u09FF' for char in text):
        return "bn"
    
    # Malayalam detection (Malayalam script)
    if any('\u0D00' <= char <= '\u0D7F' for char in text):
        return "ml"
    
    # Kannada detection (Kannada script)
    if any('\u0C80' <= char <= '\u0CFF' for char in text):
        return "kn"
    
    # Gujarati detection (Gujarati script)
    if any('\u0A80' <= char <= '\u0AFF' for char in text):
        return "gu"
    
    # Marathi detection (same script as Hindi, use word patterns)
    marathi_words = ['का', 'आहे', 'मी', 'तू', 'ते', 'हे', 'या', 'ना']
    if any(word in text for word in marathi_words):
        return "mr"
    
    # Punjabi detection (Gurmukhi script)
    if any('\u0A00' <= char <= '\u0A7F' for char in text):
        return "pa"
    
    # Common Hindi words (fallback for Romanized Hindi)
    hindi_words = ['namaste', 'kaise', 'hai', 'mein', 'aap', 'kya', 'haan', 'nahi', 'dhanyawad']
    if any(word in text_lower for word in hindi_words):
        return "hi"
    
    # Common Tamil romanized words
    tamil_words = ['vanakkam', 'eppadi', 'irukkirathu', 'nandri', 'enna', 'illai', 'aam']
    if any(word in text_lower for word in tamil_words):
        return "ta"
    
    # Default to English if no other language detected
    return "en"


def get_connection():
    return psycopg2.connect(**DB_CONFIG)

# Create table if not exists
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
            nmtUsage INT,
            llmUsage INT,
            ttsUsage INT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


init_db()

# ----------------------------
# Pipeline configuration
# ----------------------------
# Hardcoded pipelines for different customers
PIPELINES = {
    "cust1": ["NMT", "LLM", "TTS"],
    "cust2": ["NMT", "LLM"]
}

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

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
    return Response(content=get_metrics(), media_type="text/plain")

class PipelineInput(BaseModel):
    customerName: str
    customerAppName: str
    input: Dict[str, str] # e.g., {"text": "some text", "language": "en" (optional)}

# ----------------------------
# Fake model calls
# ----------------------------
def call_model(model_name: str, input_text: str) -> str:
    # simulate actual processing with sleep
    time.sleep(0.1)  # 100ms
    return f"{model_name} processed: {input_text}"

# ----------------------------
# Pipeline Endpoint
# ----------------------------
@app.post("/pipeline")
def run_pipeline(payload: PipelineInput):
    usage = {"NMT": "0", "LLM": "0", "TTS": "0"}
    customer = payload.customerName
    appname = payload.customerAppName
    pipeline = PIPELINES.get(customer.lower()) or PIPELINES.get(customer)

    if not pipeline:
        raise HTTPException(status_code=400, detail=f"No pipeline defined for customer {customer}")

    # Start metrics collection
    request_id_metrics = metrics_collector.start_request(customer, appname, "pipeline")
    
    request_id = str(uuid.uuid4())
    latencies = {}
    pipeline_output = {}
    input_text = payload.input.get("text", "")
    target_language = payload.input.get("language", "en")  # default English
    response_data = ""

    start_pipeline = time.time()

    try:
        # ---- Step 1: Language Detection ----
        metrics_collector.start_component(request_id_metrics, "LangDetection")
        start = time.time()
        source_language = detect_language_from_text(input_text)
        latencies["LangDetection"] = f"{int((time.time() - start) * 1000)}ms"
        metrics_collector.end_component(request_id_metrics, "LangDetection", True)
        print("Language detection completed: ", source_language)

        # ---- Step 2+: Run customer-specific pipeline ----
        current_output = input_text

        if "NMT" in pipeline:
            metrics_collector.start_component(request_id_metrics, "NMT")
            start = time.time()
            usage["NMT"] = str(len(current_output))
            current_output = nmt_service.translate_text(
                text=current_output,
                source_lang=source_language,
                target_lang=target_language
            )
            latencies["NMT"] = f"{int((time.time() - start) * 1000)}ms"
            pipeline_output["NMT"] = current_output["translated_text"]
            metrics_collector.end_component(request_id_metrics, "NMT", True)

            # Record NMT metrics
            metrics_collector.record_service_request("nmt", customer, appname)
            metrics_collector.record_nmt_translation(
                customer=customer,
                app=appname,
                source_lang=source_language,
                target_lang=target_language,
                char_count=len(input_text)
            )

        if "LLM" in pipeline:
            metrics_collector.start_component(request_id_metrics, "LLM")
            start = time.time()
            current_output = llm_service.process_query(current_output)
            latencies["LLM"] = f"{int((time.time() - start) * 1000)}ms"
            pipeline_output["LLM"] = current_output["response"]
            response_data = current_output["response"]
            print("LLM Output: ", current_output)
            usage["LLM"] = str(current_output["total_tokens"])
            metrics_collector.end_component(request_id_metrics, "LLM", True)

            # Record LLM metrics
            metrics_collector.record_service_request("llm", customer, appname)
            metrics_collector.record_llm_processing(
                customer=customer,
                app=appname,
                model="gemini",  # or extract from service
                token_count=current_output["total_tokens"]
            )

        if "TTS" in pipeline:
            metrics_collector.start_component(request_id_metrics, "TTS")
            start = time.time()
            print("TTS Input: ", current_output)
            usage["TTS"] = str(len(current_output["response"]))
            current_output = tts_service.text_to_speech(current_output["response"], target_language, gender="female")
            latencies["TTS"] = f"{int((time.time() - start) * 1000)}ms"
            pipeline_output["TTS"] = current_output["audio_content"]
            response_data = current_output["audio_content"]
            metrics_collector.end_component(request_id_metrics, "TTS", True)

            # Record TTS metrics
            metrics_collector.record_service_request("tts", customer, appname)
            metrics_collector.record_tts_synthesis(
                customer=customer,
                app=appname,
                language=target_language,
                char_count=len(current_output["response"])
            )

        # ---- Finalize timings ----
        total_elapsed = int((time.time() - start_pipeline) * 1000)
        latencies["pipelineTotal"] = f"{total_elapsed}ms"
        
        # End request with success
        metrics_collector.end_request(request_id_metrics, 200)

    except Exception as e:
        # End request with error
        metrics_collector.end_request(request_id_metrics, 500)
        # End any active component measurements
        for component in ["LangDetection", "NMT", "LLM", "TTS"]:
            try:
                metrics_collector.end_component(request_id_metrics, component, False)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # ---- Save to DB ----
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {TABLE_NAME} 
        (requestId, customerName, customerApp, langdetectionLatency, nmtLatency, llmLatency, ttsLatency, overallPipelineLatency,
        nmtUsage, llmUsage, ttsUsage, timestamp)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        request_id,
        customer,
        appname,
        latencies.get("LangDetection", ""),
        latencies.get("NMT", ""),
        latencies.get("LLM", ""),
        latencies.get("TTS", ""),
        latencies.get("pipelineTotal", ""),
        str(usage.get("NMT", "0")),
        str(usage.get("LLM", "0")),
        str(usage.get("TTS", "0")),
        datetime.now(timezone.utc)
    ))

    conn.commit()
    cur.close()
    conn.close()

    # ---- Return API contract ----
    return {
        "requestId": request_id,
        "status": "success",
        "pipelineOutput": pipeline_output,
        "responseData": response_data,
        "latency": latencies,
        "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ----------------------------
# DB Access Endpoints
# ----------------------------
@app.get("/customers")
def get_all_customers():
    request_id_metrics = metrics_collector.start_request("system", "db", "get_customers")
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        metrics_collector.end_request(request_id_metrics, 200)
        return rows
    except Exception as e:
        metrics_collector.end_request(request_id_metrics, 500)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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
                grouped[app_key][col].append(0.0)
            else:
                try:
                    # Remove "ms" suffix if exists, convert to float
                    if isinstance(val, str) and val.endswith("ms"):
                        num = float(val.replace("ms", "").strip())
                    else:
                        num = float(val)
                    grouped[app_key][col].append(num)
                except Exception:
                    grouped[app_key][col].append(0.0)

    # Compute averages per app
    aggregates = []
    for app_key, metrics in grouped.items():
        aggregates.append({
            "customerName": customerName,
            "customerApp": app_key,
            "avg_langdetectionLatency": statistics.mean(metrics["langdetectionlatency"]) if metrics["langdetectionlatency"] else 0.0,
            "avg_nmtLatency": statistics.mean(metrics["nmtlatency"]) if metrics["nmtlatency"] else 0.0,
            "avg_llmLatency": statistics.mean(metrics["llmlatency"]) if metrics["llmlatency"] else 0.0,
            "avg_ttsLatency": statistics.mean(metrics["ttslatency"]) if metrics["ttslatency"] else 0.0,
            "avg_overallPipelineLatency": statistics.mean(metrics["overallpipelinelatency"]) if metrics["overallpipelinelatency"] else 0.0,
            "avg_nmtUsage": statistics.mean(metrics["nmtusage"]) if metrics["nmtusage"] else 0.0,
            "avg_llmUsage": statistics.mean(metrics["llmusage"]) if metrics["llmusage"] else 0.0,
            "avg_ttsUsage": statistics.mean(metrics["ttsusage"]) if metrics["ttsusage"] else 0.0,
        })

    return {
        "customerName": customerName,
        "aggregates": aggregates
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
