import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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

TABLE_NAME = "ai4x_demo_requests_log"

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
    customer = payload.customerName
    appname = payload.customerAppName
    pipeline = PIPELINES.get(customer.lower()) or PIPELINES.get(customer)

    if not pipeline:
        raise HTTPException(status_code=400, detail=f"No pipeline defined for customer {customer}")

    request_id = str(uuid.uuid4())
    latencies = {}
    pipeline_output = {}
    input_text = payload.input.get("text", "")
    target_language = payload.input.get("language", "en")  # default English

    start_pipeline = time.time()

    # ---- Step 1: Language Detection ----
    start = time.time()
    source_language = detect_language_from_text(input_text)
    latencies["LangDetection"] = f"{int((time.time() - start) * 1000)}ms"
    print("Language detection completed: ", source_language)

    # ---- Step 2+: Run customer-specific pipeline ----
    current_output = input_text

    if "NMT" in pipeline:
        start = time.time()
        current_output = nmt_service.translate_text(
            text=current_output,
            source_lang=source_language,
            target_lang=target_language
        )
        latencies["NMT"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["NMT"] = current_output["translated_text"]

    if "LLM" in pipeline:
        start = time.time()
        current_output = llm_service.process_query(current_output)
        latencies["LLM"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["LLM"] = current_output["response"]
        response_data = current_output["response"]

    if "TTS" in pipeline:
        start = time.time()
        current_output = tts_service.text_to_speech(current_output["response"], target_language, gender="female")
        latencies["TTS"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["TTS"] = current_output["audio_content"]
        response_data = current_output["audio_content"]

    # ---- Finalize timings ----
    total_elapsed = int((time.time() - start_pipeline) * 1000)
    latencies["pipelineTotal"] = f"{total_elapsed}ms"

    # responseData = last model’s output

    # ---- Save to DB ----
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {TABLE_NAME} 
        (requestId, customerName, customerApp, langdetectionLatency, nmtLatency, llmLatency, ttsLatency, overallPipelineLatency, timestamp)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        request_id,
        customer,
        appname,
        latencies.get("LangDetection", ""),
        latencies.get("NMT", ""),
        latencies.get("LLM", ""),
        latencies.get("TTS", ""),
        latencies.get("pipelineTotal", ""),
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
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ----------------------------
# DB Access Endpoints
# ----------------------------
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
