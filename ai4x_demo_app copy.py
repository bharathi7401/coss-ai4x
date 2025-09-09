import psycopg2
import psycopg2.extras
import statistics
import time
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from services.nmt_service import NMTService
from services.asr_service import ASRService
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.weather_service import WeatherService
# Removed Prometheus metrics - using direct DB storage instead
from config import Config

# Database configuration
DB_CONFIG = {
    "user": "ai4xadmin",
    "password": "password123#",  # replace with your actual password or use env var
    "host": "ai4x.postgres.database.azure.com",
    "port": 5432,
    "database": "ai4xdemo"
}

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import threading

# --- Prometheus core ---
# from prometheus_client import Counter, Histogram, Gauge

# ----------------------------
# Database connection
# ----------------------------

nmt_service = NMTService()
asr_service = ASRService()
llm_service = LLMService()
tts_service = TTSService()
weather_service = WeatherService()

TABLE_NAME = "ai4x_demo_requests_log_v9"

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
            asrLatency TEXT,
            overallPipelineLatency TEXT,
            nmtUsage TEXT,
            llmUsage TEXT,
            backNMTUsage TEXT,
            ttsUsage TEXT,
            audio_duration FLOAT,
            service_type TEXT,
            status TEXT,
            error_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()


# Fixed pipeline: Language Detection → NMT → LLM → BackNMT → TTS
PIPELINE_SERVICES = ["NMT", "LLM", "BackNMT", "TTS"]

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

# Metrics endpoint removed - using direct DB storage instead


class PipelineInput(BaseModel):
    input: str

# Individual Service Models
class NMTRequest(BaseModel):
    text: str

class NMTResponse(BaseModel):
    requestId: str
    status: str
    translated_text: str
    source_language: str
    target_language: str
    character_count: int
    latency: str
    timestamp: str

class ASRRequest(BaseModel):
    audio_content: str  # Base64 encoded audio
    audio_format: str = "wav"

class ASRResponse(BaseModel):
    requestId: str
    status: str
    text: str
    detected_language: str
    audio_duration: float
    latency: str
    timestamp: str

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    requestId: str
    status: str
    audio_content: str  # Base64 encoded audio
    language: str
    gender: str
    character_count: int
    latency: str
    timestamp: str

# Usage Tracking Models
class UsageDataVolume(BaseModel):
    customer_id: str
    service: str
    total_input_chars: int
    total_output_chars: int
    total_audio_duration: float  # in seconds
    period_start: str
    period_end: str

class UsageAPICalls(BaseModel):
    customer_id: str
    service: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    period_start: str
    period_end: str

class UsageErrors(BaseModel):
    customer_id: str
    service: str
    error_count: int
    error_types: Dict[str, int]
    period_start: str
    period_end: str

class UsageSummary(BaseModel):
    customer_id: str
    total_requests: int
    total_characters: int
    total_tokens: int
    total_audio_duration: float
    services_used: Dict[str, Dict[str, Any]]
    period_start: str
    period_end: str

class UsageEvent(BaseModel):
    event_id: str
    customer_id: str
    service: str
    request_id: str
    timestamp: str
    input_size: int
    output_size: int
    processing_time: float
    status: str
    cost: float

class QuotaInfo(BaseModel):
    customer_id: str
    service: str
    current_usage: int
    limit: int
    reset_date: str
    status: str  # "within_limit", "approaching_limit", "exceeded"

class QuotaRequest(BaseModel):
    customer_id: str
    service: str
    requested_limit: int
    reason: str
    status: str = "pending"

class Invoice(BaseModel):
    invoice_id: str
    customer_id: str
    month: str
    total_amount: float
    services: Dict[str, float]
    status: str
    download_url: str
    due_date: str

class CreditBalance(BaseModel):
    customer_id: str
    promotional_credits: float
    prepaid_credits: float
    total_credits: float
    last_updated: str

class WebhookSubscription(BaseModel):
    subscription_id: str
    customer_id: str
    event_types: list
    webhook_url: str
    status: str
    created_at: str

# ----------------------------
# Individual Service Endpoints
# ----------------------------

@app.post("/services/nmt/translate", response_model=NMTResponse)
def nmt_translate(request: NMTRequest, customer_id: str):
    """NMT API - Translate text from one language to another"""
    request_id = str(uuid.uuid4())
    customer = customer_id
    appname = "ai4x_demo"
    
    try:
        start_time = time.time()
        
        # Auto-detect source language and set target language to English
        source_language = detect_language_from_text(request.text)
        target_language = "en"
        
        result = nmt_service.translate_text(
            text=request.text,
            source_lang=source_language,
            target_lang=target_language
        )
        
        latency = f"{int((time.time() - start_time) * 1000)}ms"
        
        if result.get("success", False):
            translated_text = result["translated_text"]
            character_count = len(translated_text)
            
            # Store metrics in database
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, nmtLatency, nmtUsage, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                latency,
                str(character_count),
                datetime.now(timezone.utc),
                "NMT",
                "success"
            ))
            conn.commit()
            cur.close()
            conn.close()
            
            return NMTResponse(
                requestId=request_id,
                status="success",
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                character_count=character_count,
                latency=latency,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            # Log error to database
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                datetime.now(timezone.utc),
                "NMT",
                "error",
                result.get("error", "Translation failed")
            ))
            conn.commit()
            cur.close()
            conn.close()
            
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Translation failed")
            )
            
    except Exception as e:
        # Log error to database
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                datetime.now(timezone.utc),
                "NMT",
                "error",
                str(e)
            ))
            conn.commit()
            cur.close()
            conn.close()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/services/asr/transcribe", response_model=ASRResponse)
def asr_transcribe(request: ASRRequest, customer_id: str):
    """ASR API - Convert audio input into text output"""
    request_id = str(uuid.uuid4())
    customer = customer_id
    appname = "ai4x_demo"
    
    try:
        start_time = time.time()
        
        # Decode base64 audio
        import base64
        audio_content = base64.b64decode(request.audio_content)
        
        result = asr_service.transcribe_audio(
            audio_content=audio_content,
            audio_format=request.audio_format
        )
        
        latency = f"{int((time.time() - start_time) * 1000)}ms"
        
        if result.get("success", False):
            text = result["text"]
            detected_language = result.get("detected_language", "en")
            
            # Estimate audio duration (rough calculation)
            # This is a simplified estimation - in production you'd want more accurate duration calculation
            audio_duration = len(audio_content) / 32000  # Rough estimate for 16kHz audio
            
            # Store metrics in database
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, asrLatency, audio_duration, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                latency,
                audio_duration,
                datetime.now(timezone.utc),
                "ASR",
                "success"
            ))
            conn.commit()
            cur.close()
            conn.close()
            
            return ASRResponse(
                requestId=request_id,
                status="success",
                text=text,
                detected_language=detected_language,
                audio_duration=audio_duration,
                latency=latency,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            # Log error to database
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                datetime.now(timezone.utc),
                "ASR",
                "error",
                result.get("error", "Audio transcription failed")
            ))
            conn.commit()
            cur.close()
            conn.close()
            
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Audio transcription failed")
            )
            
    except Exception as e:
        # Log error to database
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                datetime.now(timezone.utc),
                "ASR",
                "error",
                str(e)
            ))
            conn.commit()
            cur.close()
            conn.close()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/services/tts/speak", response_model=TTSResponse)
def tts_speak(request: TTSRequest, customer_id: str):
    """TTS API - Convert text input into audio output"""
    request_id = str(uuid.uuid4())
    customer = customer_id
    appname = "ai4x_demo"
    
    try:
        start_time = time.time()
        
        # Auto-detect language and set default gender
        language = detect_language_from_text(request.text)
        gender = "female"  # Default gender
        
        result = tts_service.text_to_speech(
            text=request.text,
            language=language,
            gender=gender
        )
        
        latency = f"{int((time.time() - start_time) * 1000)}ms"
        
        if result.get("success", False):
            audio_content = result["audio_content"]
            character_count = len(request.text)
            
            # Store metrics in database
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, ttsLatency, ttsUsage, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                latency,
                str(character_count),
                datetime.now(timezone.utc),
                "TTS",
                "success"
            ))
            conn.commit()
            cur.close()
            conn.close()
            
            return TTSResponse(
                requestId=request_id,
                status="success",
                audio_content=audio_content,
                language=language,
                gender=gender,
                character_count=character_count,
                latency=latency,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            # Log error to database
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                datetime.now(timezone.utc),
                "TTS",
                "error",
                result.get("error", "Text-to-speech conversion failed")
            ))
            conn.commit()
            cur.close()
            conn.close()
            
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Text-to-speech conversion failed")
            )
            
    except Exception as e:
        # Log error to database
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                request_id, customer, appname,
                datetime.now(timezone.utc),
                "TTS",
                "error",
                str(e)
            ))
            conn.commit()
            cur.close()
            conn.close()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Usage Tracking Endpoints
# ----------------------------

@app.get("/usage/data-volume", response_model=UsageDataVolume)
def get_data_volume(customer_id: str, service: str, start_date: str = None, end_date: str = None):
    """Get data volume processed for a specific service by a customer"""
    try:
        # Set default date range if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
        if not end_date:
            end_date = datetime.now().isoformat()
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Query usage data from database
        cur.execute(f"""
            SELECT 
                SUM(CASE WHEN service_type = 'NMT' THEN nmtUsage::int ELSE 0 END) as nmt_chars,
                SUM(CASE WHEN service_type = 'TTS' THEN ttsUsage::int ELSE 0 END) as tts_chars,
                SUM(CASE WHEN service_type = 'LLM' THEN llmUsage::int ELSE 0 END) as llm_tokens,
                SUM(CASE WHEN service_type = 'ASR' THEN audio_duration ELSE 0 END) as audio_duration
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND service_type = %s 
            AND timestamp BETWEEN %s AND %s
        """, (customer_id, service, start_date, end_date))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            nmt_chars, tts_chars, llm_tokens, audio_duration = result
            # Only return input characters/tokens for each service
            if service == 'LLM':
                total_input_chars = llm_tokens or 0
            elif service == 'NMT':
                total_input_chars = nmt_chars or 0
            elif service == 'TTS':
                total_input_chars = tts_chars or 0
            else:
                total_input_chars = 0
            total_output_chars = 0  # Not tracking output
            total_audio_duration = audio_duration or 0.0
        else:
            total_input_chars = total_output_chars = 0
            total_audio_duration = 0.0
        
        return UsageDataVolume(
            customer_id=customer_id,
            service=service,
            total_input_chars=total_input_chars,
            total_output_chars=total_output_chars,
            total_audio_duration=total_audio_duration,
            period_start=start_date,
            period_end=end_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage/api-calls", response_model=UsageAPICalls)
def get_api_calls(customer_id: str, service: str, start_date: str = None, end_date: str = None):
    """Get API call count for a specific service by a customer"""
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
        if not end_date:
            end_date = datetime.now().isoformat()
        
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute(f"""
            SELECT 
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_requests
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND service_type = %s 
            AND timestamp BETWEEN %s AND %s
        """, (customer_id, service, start_date, end_date))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            total_requests, successful_requests, failed_requests = result
        else:
            total_requests = successful_requests = failed_requests = 0
        
        return UsageAPICalls(
            customer_id=customer_id,
            service=service,
            total_requests=total_requests or 0,
            successful_requests=successful_requests or 0,
            failed_requests=failed_requests or 0,
            period_start=start_date,
            period_end=end_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage/errors", response_model=UsageErrors)
def get_errors(customer_id: str, service: str, start_date: str = None, end_date: str = None):
    """Get error count for a specific service by a customer"""
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
        if not end_date:
            end_date = datetime.now().isoformat()
        
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute(f"""
            SELECT 
                COUNT(*) as error_count,
                error_type
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND service_type = %s AND status = 'error'
            AND timestamp BETWEEN %s AND %s
            GROUP BY error_type
        """, (customer_id, service, start_date, end_date))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        error_types = {}
        total_errors = 0
        for error_count, error_type in results:
            error_types[error_type or "unknown"] = error_count
            total_errors += error_count
        
        return UsageErrors(
            customer_id=customer_id,
            service=service,
            error_count=total_errors,
            error_types=error_types,
            period_start=start_date,
            period_end=end_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# V1 Usage Endpoints
# ----------------------------

@app.get("/v1/usage", response_model=UsageSummary)
def get_usage_summary(customer_id: str, start_date: str = None, end_date: str = None):
    """Get comprehensive usage summary for a customer"""
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
        if not end_date:
            end_date = datetime.now().isoformat()
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Get overall usage
        cur.execute(f"""
            SELECT 
                COUNT(*) as total_requests,
                SUM(CASE WHEN service_type = 'NMT' THEN nmtUsage::int ELSE 0 END) as nmt_chars,
                SUM(CASE WHEN service_type = 'TTS' THEN ttsUsage::int ELSE 0 END) as tts_chars,
                SUM(CASE WHEN service_type = 'LLM' THEN llmUsage::int ELSE 0 END) as llm_tokens,
                SUM(CASE WHEN service_type = 'ASR' THEN audio_duration ELSE 0 END) as audio_duration
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND timestamp BETWEEN %s AND %s
        """, (customer_id, start_date, end_date))
        
        overall_result = cur.fetchone()
        
        # Get per-service breakdown
        cur.execute(f"""
            SELECT 
                service_type,
                COUNT(*) as requests,
                SUM(CASE WHEN service_type = 'NMT' THEN nmtUsage::int ELSE 0 END) as nmt_chars,
                SUM(CASE WHEN service_type = 'TTS' THEN ttsUsage::int ELSE 0 END) as tts_chars,
                SUM(CASE WHEN service_type = 'LLM' THEN llmUsage::int ELSE 0 END) as llm_tokens,
                SUM(CASE WHEN service_type = 'ASR' THEN audio_duration ELSE 0 END) as duration
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND timestamp BETWEEN %s AND %s
            GROUP BY service_type
        """, (customer_id, start_date, end_date))
        
        service_results = cur.fetchall()
        cur.close()
        conn.close()
        
        if overall_result:
            total_requests, nmt_chars, tts_chars, llm_tokens, audio_duration = overall_result
            # Separate characters (NMT, TTS) from tokens (LLM)
            total_characters = (nmt_chars or 0) + (tts_chars or 0)
            total_tokens = llm_tokens or 0
        else:
            total_requests = total_characters = total_tokens = 0
            audio_duration = 0.0
        
        services_used = {}
        for service_type, requests, nmt_chars, tts_chars, llm_tokens, duration in service_results:
            # Only return input characters/tokens for each service
            if service_type == 'LLM':
                input_chars = llm_tokens or 0
            elif service_type == 'NMT':
                input_chars = nmt_chars or 0
            elif service_type == 'TTS':
                input_chars = tts_chars or 0
            else:
                input_chars = 0
                
            services_used[service_type] = {
                "requests": requests,
                "input_chars": input_chars,
                "audio_duration": duration or 0.0
            }
        
        return UsageSummary(
            customer_id=customer_id,
            total_requests=total_requests or 0,
            total_characters=total_characters,
            total_tokens=total_tokens,
            total_audio_duration=audio_duration or 0.0,
            services_used=services_used,
            period_start=start_date,
            period_end=end_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/usage/events", response_model=List[UsageEvent])
def get_usage_events(customer_id: str, service: str = None, limit: int = 100, offset: int = 0):
    """Get detailed usage events for a customer"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        query = f"""
            SELECT 
                requestId,
                service_type,
                timestamp,
                CASE 
                    WHEN service_type = 'LLM' THEN llmUsage::int
                    WHEN service_type = 'NMT' THEN nmtUsage::int
                    WHEN service_type = 'TTS' THEN ttsUsage::int
                    ELSE 0
                END as input_size,
                0 as output_size,
                EXTRACT(EPOCH FROM (overallPipelineLatency::text::interval)) as processing_time,
                status,
                CASE 
                    WHEN service_type = 'LLM' THEN llmUsage::int * 0.001
                    WHEN service_type = 'NMT' THEN nmtUsage::int * 0.001
                    WHEN service_type = 'TTS' THEN ttsUsage::int * 0.001
                    ELSE 0
                END as cost
            FROM {TABLE_NAME} 
            WHERE customerName = %s
        """
        params = [customer_id]
        
        if service:
            query += " AND service_type = %s"
            params.append(service)
        
        query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        events = []
        for row in results:
            request_id, service_type, timestamp, input_size, output_size, processing_time, status, cost = row
            events.append(UsageEvent(
                event_id=str(uuid.uuid4()),
                customer_id=customer_id,
                service=service_type or "unknown",
                request_id=request_id,
                timestamp=timestamp.isoformat() if timestamp else datetime.now().isoformat(),
                input_size=input_size or 0,
                output_size=output_size or 0,
                processing_time=processing_time or 0.0,
                status=status or "unknown",
                cost=cost or 0.0
            ))
        
        return events
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Quota Management Endpoints
# ----------------------------

@app.get("/v1/quotas", response_model=List[QuotaInfo])
def get_quotas(customer_id: str):
    """Get current usage limits and quotas for a customer"""
    try:
        # Default quotas (in production, these would come from a database)
        default_quotas = {
            "NMT": {"limit": 100000, "reset_date": "2025-02-01"},
            "ASR": {"limit": 1000, "reset_date": "2025-02-01"},
            "TTS": {"limit": 100000, "reset_date": "2025-02-01"},
            "LLM": {"limit": 10000, "reset_date": "2025-02-01"}
        }
        
        conn = get_connection()
        cur = conn.cursor()
        
        quotas = []
        for service, quota_info in default_quotas.items():
            # Get current usage for this month
            cur.execute(f"""
                SELECT COUNT(*) as current_usage
                FROM {TABLE_NAME} 
                WHERE customerName = %s AND service_type = %s 
                AND timestamp >= date_trunc('month', CURRENT_DATE)
            """, (customer_id, service))
            
            result = cur.fetchone()
            current_usage = result[0] if result else 0
            
            # Determine status
            if current_usage >= quota_info["limit"]:
                status = "exceeded"
            elif current_usage >= quota_info["limit"] * 0.8:
                status = "approaching_limit"
            else:
                status = "within_limit"
            
            quotas.append(QuotaInfo(
                customer_id=customer_id,
                service=service,
                current_usage=current_usage,
                limit=quota_info["limit"],
                reset_date=quota_info["reset_date"],
                status=status
            ))
        
        cur.close()
        conn.close()
        
        return quotas
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/quotas/requests", response_model=QuotaRequest)
def request_quota_increase(request: QuotaRequest):
    """Request an increase in quotas"""
    try:
        # In production, this would be stored in a database and processed by admin
        request_id = str(uuid.uuid4())
        
        # For demo purposes, we'll just return the request with a pending status
        return QuotaRequest(
            customer_id=request.customer_id,
            service=request.service,
            requested_limit=request.requested_limit,
            reason=request.reason,
            status="pending"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Billing Endpoints
# ----------------------------

@app.get("/v1/invoices", response_model=List[Invoice])
def get_invoices(customer_id: str, month: str = None):
    """List invoices for a specified month"""
    try:
        if not month:
            month = datetime.now().strftime("%Y-%m")
        
        # Mock invoice data (in production, this would come from a billing system)
        mock_invoices = [
            Invoice(
                invoice_id=f"INV-{month}-001",
                customer_id=customer_id,
                month=month,
                total_amount=125.50,
                services={
                    "NMT": 45.20,
                    "ASR": 30.80,
                    "TTS": 49.50
                },
                status="paid",
                download_url=f"https://billing.ai4x.com/invoices/{customer_id}/{month}.pdf",
                due_date=f"{month}-15"
            )
        ]
        
        return mock_invoices
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/credits", response_model=CreditBalance)
def get_credits(customer_id: str):
    """Display promotional or pre-paid credit balances for a customer"""
    try:
        # Mock credit data (in production, this would come from a billing system)
        promotional_credits = 50.00
        prepaid_credits = 200.00
        total_credits = promotional_credits + prepaid_credits
        
        return CreditBalance(
            customer_id=customer_id,
            promotional_credits=promotional_credits,
            prepaid_credits=prepaid_credits,
            total_credits=total_credits,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Webhook Endpoints
# ----------------------------

@app.post("/v1/webhooks/subscriptions", response_model=WebhookSubscription)
def subscribe_to_webhooks(subscription: WebhookSubscription):
    """Subscribe to events via webhooks"""
    try:
        subscription_id = str(uuid.uuid4())
        
        # In production, this would be stored in a database
        return WebhookSubscription(
            subscription_id=subscription_id,
            customer_id=subscription.customer_id,
            event_types=subscription.event_types,
            webhook_url=subscription.webhook_url,
            status="active",
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Pipeline Endpoint
# ----------------------------
@app.post("/services/pipeline/run")
def run_pipeline(payload: PipelineInput, customer_id: str):
    usage = {"NMT": None, "LLM": None, "TTS": None, "backNMT": None}
    request_id = str(uuid.uuid4())
    latencies = {}
    pipeline_output = {}
    input_text = payload.input
    target_language = "en"  # Default target language
    customer = customer_id
    appname = "ai4x_demo"

    success = True
    response_data = None
    pipeline_start_time = time.time()

    try:
        # ---- Language Detection ----
        start = time.time()
        source_language = detect_language_from_text(input_text)
        latencies["LangDetection"] = f"{int((time.time() - start) * 1000)}ms"

        current_output = input_text

        # ---- NMT ----
        start = time.time()
        usage["NMT"] = str(len(current_output))
        current_output = nmt_service.translate_text(current_output, source_language, target_language)
        latencies["NMT"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["NMT"] = current_output["translated_text"]

        # ---- LLM ----
        start = time.time()
        current_output = llm_service.process_query(current_output)
        latencies["LLM"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["LLM"] = current_output["response"]
        response_data = current_output["response"]
        usage["LLM"] = str(current_output["total_tokens"])
        
        # Log LLM usage separately with service_type='LLM'
        try:
            llm_conn = get_connection()
            llm_cur = llm_conn.cursor()
            llm_cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, llmLatency, llmUsage, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                f"{request_id}_llm", customer, appname,
                latencies.get("LLM", None),
                str(usage.get("LLM", None)),
                datetime.now(timezone.utc),
                "LLM",
                "success"
            ))
            llm_conn.commit()
            llm_cur.close()
            llm_conn.close()
        except Exception as llm_log_error:
            # Don't fail the pipeline if LLM logging fails
            print(f"Failed to log LLM usage: {llm_log_error}")
            pass

        # ---- Backward NMT ----
        start = time.time()
        usage["backNMT"] = str(len(current_output["response"]))
        back_translation = nmt_service.translate_text(
            text=current_output["response"],
            source_lang="en",
            target_lang=source_language
        )
        latencies["BackNMT"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["BackNMT"] = back_translation["translated_text"]
        response_data = back_translation["translated_text"]

        # ---- TTS ----
        start = time.time()
        usage["TTS"] = str(len(response_data))
        current_output = tts_service.text_to_speech(response_data, source_language, gender="female")
        latencies["TTS"] = f"{int((time.time() - start) * 1000)}ms"
        pipeline_output["TTS"] = current_output["audio_content"]
        response_data = current_output["audio_content"]

        # Calculate total pipeline latency
        latencies["pipelineTotal"] = f"{int((time.time() - pipeline_start_time) * 1000)}ms"

        # ---- DB Logging - Individual Service Entries ----
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            # Log NMT service
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, nmtLatency, nmtUsage, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                f"{request_id}_nmt", customer, appname,
                latencies.get("NMT", None),
                str(usage.get("NMT", None)),
                datetime.now(timezone.utc),
                "NMT",
                "success"
            ))
            
            # Log BackNMT service
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, nmtLatency, nmtUsage, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                f"{request_id}_backnmt", customer, appname,
                latencies.get("BackNMT", None),
                str(usage.get("backNMT", None)),
                datetime.now(timezone.utc),
                "NMT",  # BackNMT is also NMT service
                "success"
            ))
            
            # Log TTS service
            cur.execute(f"""
                INSERT INTO {TABLE_NAME} 
                (requestId, customerName, customerApp, ttsLatency, ttsUsage, timestamp, service_type, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                f"{request_id}_tts", customer, appname,
                latencies.get("TTS", None),
                str(usage.get("TTS", None)),
                datetime.now(timezone.utc),
                "TTS",
                "success"
            ))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_error:
            print(f"Failed to log individual service usage: {db_error}")
            pass  # Don't fail the pipeline if DB logging fails

    except Exception as e:
        success = False
        # Log error to database for each service that might have been processed
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            # Log error for each service that was attempted
            services_to_log = ["NMT", "LLM", "TTS"]
            for service in services_to_log:
                try:
                    cur.execute(f"""
                        INSERT INTO {TABLE_NAME} 
                        (requestId, customerName, customerApp, timestamp, service_type, status, error_type)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        f"{request_id}_{service.lower()}", customer, appname,
                        datetime.now(timezone.utc),
                        service,
                        "error",
                        str(e)
                    ))
                except:
                    pass  # Continue with other services even if one fails
            
            conn.commit()
            cur.close()
            conn.close()
        except:
            pass  # Don't fail if DB logging fails
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai4x_demo_app:app", host="0.0.0.0", port=8000, reload=False)