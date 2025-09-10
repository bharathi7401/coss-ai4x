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
from config import Config
from metrics import metrics_collector

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
# Customer Definitions
# ----------------------------

class CustomerTier:
    def __init__(self, name: str, tier: str, monthly_quota: dict, pricing: dict, features: list):
        self.name = name
        self.tier = tier
        self.monthly_quota = monthly_quota  # {service: limit}
        self.pricing = pricing  # {service: cost_per_unit}
        self.features = features

# Define 4 customers with different tiers
CUSTOMERS = {
    "customer1": CustomerTier(
        name="customer1",
        tier="freemium",
        monthly_quota={
            "NMT": 1000,      # 1000 characters per month
            "ASR": 10,        # 10 minutes of audio per month
            "TTS": 1000,      # 1000 characters per month
            "LLM": 100        # 100 tokens per month
        },
        pricing={
            "NMT": 0.0,       # Free
            "ASR": 0.0,       # Free
            "TTS": 0.0,       # Free
            "LLM": 0.0        # Free
        },
        features=["Basic support", "Standard latency", "Community forum access"]
    ),
    
    "customer2": CustomerTier(
        name="customer2",
        tier="basic",
        monthly_quota={
            "NMT": 10000,     # 10,000 characters per month
            "ASR": 100,       # 100 minutes of audio per month
            "TTS": 10000,     # 10,000 characters per month
            "LLM": 1000       # 1,000 tokens per month
        },
        pricing={
            "NMT": 0.001,     # $0.001 per character
            "ASR": 0.01,      # $0.01 per minute
            "TTS": 0.001,     # $0.001 per character
            "LLM": 0.002      # $0.002 per token
        },
        features=["Email support", "Standard latency", "Basic analytics", "API access"]
    ),
    
    "customer3": CustomerTier(
        name="customer3",
        tier="standard",
        monthly_quota={
            "NMT": 100000,    # 100,000 characters per month
            "ASR": 1000,      # 1,000 minutes of audio per month
            "TTS": 100000,    # 100,000 characters per month
            "LLM": 10000      # 10,000 tokens per month
        },
        pricing={
            "NMT": 0.0008,    # $0.0008 per character (20% discount)
            "ASR": 0.008,     # $0.008 per minute (20% discount)
            "TTS": 0.0008,    # $0.0008 per character (20% discount)
            "LLM": 0.0015     # $0.0015 per token (25% discount)
        },
        features=["Priority support", "Faster latency", "Advanced analytics", "Webhook support", "Custom integrations"]
    ),
    
    "customer4": CustomerTier(
        name="customer4",
        tier="premium",
        monthly_quota={
            "NMT": 1000000,   # 1,000,000 characters per month
            "ASR": 10000,     # 10,000 minutes of audio per month
            "TTS": 1000000,   # 1,000,000 characters per month
            "LLM": 100000     # 100,000 tokens per month
        },
        pricing={
            "NMT": 0.0005,    # $0.0005 per character (50% discount)
            "ASR": 0.005,     # $0.005 per minute (50% discount)
            "TTS": 0.0005,    # $0.0005 per character (50% discount)
            "LLM": 0.001      # $0.001 per token (50% discount)
        },
        features=["24/7 phone support", "Ultra-fast latency", "Real-time analytics", "Custom webhooks", "Dedicated account manager", "SLA guarantees", "Custom model training"]
    )
}

def get_customer_tier(customer_id: str) -> CustomerTier:
    """Get customer tier information by customer ID"""
    if customer_id not in CUSTOMERS:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found. Valid customers: {list(CUSTOMERS.keys())}")
    return CUSTOMERS[customer_id]

def validate_customer_quota(customer_id: str, service: str, usage_amount: int) -> bool:
    """Validate if customer has remaining quota for the service"""
    customer = get_customer_tier(customer_id)
    
    # Get current month's usage
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(f"""
            SELECT 
                CASE 
                    WHEN service_type = 'NMT' THEN SUM(nmtUsage::int)
                    WHEN service_type = 'TTS' THEN SUM(ttsUsage::int)
                    WHEN service_type = 'LLM' THEN SUM(llmUsage::int)
                    WHEN service_type = 'ASR' THEN SUM(audio_duration)
                    ELSE 0
                END as current_usage
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND service_type = %s 
            AND timestamp >= date_trunc('month', CURRENT_DATE)
            GROUP BY service_type
        """, (customer_id, service))
        
        result = cur.fetchone()
        current_usage = result[0] if result and result[0] else 0
        
        # Check if adding this usage would exceed quota
        quota_limit = customer.monthly_quota.get(service, 0)
        return (current_usage + usage_amount) <= quota_limit
        
    finally:
        cur.close()
        conn.close()

def calculate_service_cost(customer_id: str, service: str, usage_amount: int) -> float:
    """Calculate cost for service usage based on customer pricing"""
    customer = get_customer_tier(customer_id)
    price_per_unit = customer.pricing.get(service, 0.0)
    return price_per_unit * usage_amount

def get_current_usage(customer_id: str, service: str) -> int:
    """Get current month's usage for a customer and service"""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(f"""
            SELECT 
                CASE 
                    WHEN service_type = 'NMT' THEN SUM(nmtUsage::int)
                    WHEN service_type = 'TTS' THEN SUM(ttsUsage::int)
                    WHEN service_type = 'LLM' THEN SUM(llmUsage::int)
                    WHEN service_type = 'ASR' THEN SUM(audio_duration)
                    ELSE 0
                END as current_usage
            FROM {TABLE_NAME} 
            WHERE customerName = %s AND service_type = %s 
            AND timestamp >= date_trunc('month', CURRENT_DATE)
            GROUP BY service_type
        """, (customer_id, service))
        
        result = cur.fetchone()
        return result[0] if result and result[0] else 0
        
    finally:
        cur.close()
        conn.close()

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Prometheus Metrics Endpoint
# ----------------------------
@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint"""
    from metrics import prometheus_latest_text
    return Response(prometheus_latest_text(), media_type="text/plain")


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
# Customer Management Endpoints
# ----------------------------

@app.get("/customers/{customer_id}")
def get_customer_info(customer_id: str):
    """Get customer tier information and current usage"""
    try:
        customer_tier = get_customer_tier(customer_id)
        
        # Get current month's usage for all services
        conn = get_connection()
        cur = conn.cursor()
        
        current_usage = {}
        for service in ["NMT", "ASR", "TTS", "LLM"]:
            cur.execute(f"""
                SELECT 
                    CASE 
                        WHEN service_type = 'NMT' THEN SUM(nmtUsage::int)
                        WHEN service_type = 'TTS' THEN SUM(ttsUsage::int)
                        WHEN service_type = 'LLM' THEN SUM(llmUsage::int)
                        WHEN service_type = 'ASR' THEN SUM(audio_duration)
                        ELSE 0
                    END as current_usage
                FROM {TABLE_NAME} 
                WHERE customerName = %s AND service_type = %s 
                AND timestamp >= date_trunc('month', CURRENT_DATE)
                GROUP BY service_type
            """, (customer_id, service))
            
            result = cur.fetchone()
            current_usage[service] = result[0] if result and result[0] else 0
        
        cur.close()
        conn.close()
        
        # Calculate remaining quota
        remaining_quota = {}
        for service, limit in customer_tier.monthly_quota.items():
            remaining_quota[service] = max(0, limit - current_usage.get(service, 0))
        
        return {
            "customer_id": customer_id,
            "tier": customer_tier.tier,
            "features": customer_tier.features,
            "monthly_quota": customer_tier.monthly_quota,
            "pricing": customer_tier.pricing,
            "current_usage": current_usage,
            "remaining_quota": remaining_quota
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customers")
def list_customers():
    """List all available customers and their tiers"""
    return {
        "customers": [
            {
                "customer_id": customer_id,
                "tier": customer_tier.tier,
                "features": customer_tier.features,
                "monthly_quota": customer_tier.monthly_quota,
                "pricing": customer_tier.pricing
            }
            for customer_id, customer_tier in CUSTOMERS.items()
        ]
    }

# ----------------------------
# Individual Service Endpoints
# ----------------------------

@app.post("/services/nmt/translate", response_model=NMTResponse)
def nmt_translate(request: NMTRequest, customer_id: str):
    """NMT API - Translate text from one language to another"""
    request_id = str(uuid.uuid4())
    customer = customer_id
    service_name = "NMT"
    
    # Start Prometheus metrics collection
    with metrics_collector.request_timer(customer, service_name, "/services/nmt/translate") as rid:
        try:
            # Validate customer exists
            customer_tier = get_customer_tier(customer_id)
            
            # Check quota before processing
            text_length = len(request.text)
            if not validate_customer_quota(customer_id, "NMT", text_length):
                raise HTTPException(
                    status_code=429, 
                    detail=f"Quota exceeded for customer {customer_id}. Current tier: {customer_tier.tier}. Monthly NMT limit: {customer_tier.monthly_quota['NMT']} characters"
                )
            
            start_time = time.time()
            
            # Auto-detect source language and set target language to English
            source_language = detect_language_from_text(request.text)
            target_language = "en"
        
            # Add component timing for NMT service
            with metrics_collector.component_timer(rid, "NMT"):
                result = nmt_service.translate_text(
                    text=request.text,
                    source_lang=source_language,
                    target_lang=target_language
                )
            
            latency = f"{int((time.time() - start_time) * 1000)}ms"
            
            if result.get("success", False):
                translated_text = result["translated_text"]
                character_count = len(translated_text)
                
                # Calculate cost based on customer pricing
                cost = calculate_service_cost(customer_id, "NMT", text_length)
                
                # Record Prometheus metrics
                metrics_collector.service_request(service_name, customer)
                metrics_collector.nmt_chars(customer, service_name, source_language, target_language, text_length)
                metrics_collector.record_service_cost(customer, service_name, customer_tier.tier, cost)
                
                # Update quota metrics
                current_usage = get_current_usage(customer_id, "NMT")
                metrics_collector.update_quota_metrics(customer, service_name, customer_tier.tier, current_usage, customer_tier.monthly_quota["NMT"])
                
                # Store metrics in database only when success=True
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {TABLE_NAME} 
                    (requestId, customerName, customerApp, nmtLatency, nmtUsage, timestamp, service_type, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    request_id, customer, "ai4x_demo",
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
                    request_id, customer, "ai4x_demo",
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
                    request_id, customer, "ai4x_demo",
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
    service_name = "ASR"
    
    # Start Prometheus metrics collection
    with metrics_collector.request_timer(customer, service_name, "/services/asr/transcribe") as rid:
        try:
            # Validate customer exists
            customer_tier = get_customer_tier(customer_id)
            
            start_time = time.time()
            
            # Decode base64 audio
            import base64
            audio_content = base64.b64decode(request.audio_content)
            
            # Estimate audio duration for quota check
            estimated_duration = len(audio_content) / 32000  # Rough estimate for 16kHz audio
            
            # Check quota before processing
            if not validate_customer_quota(customer_id, "ASR", int(estimated_duration)):
                raise HTTPException(
                    status_code=429, 
                    detail=f"Quota exceeded for customer {customer_id}. Current tier: {customer_tier.tier}. Monthly ASR limit: {customer_tier.monthly_quota['ASR']} minutes"
                )
        
            # Add component timing for ASR service
            with metrics_collector.component_timer(rid, "ASR"):
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
                
                # Calculate cost based on customer pricing
                cost = calculate_service_cost(customer_id, "ASR", int(audio_duration))
                
                # Record Prometheus metrics
                metrics_collector.service_request(service_name, customer)
                metrics_collector.asr_minutes(customer, service_name, detected_language, audio_duration / 60.0)
                metrics_collector.record_service_cost(customer, service_name, customer_tier.tier, cost)
                
                # Update quota metrics
                current_usage = get_current_usage(customer_id, "ASR")
                metrics_collector.update_quota_metrics(customer, service_name, customer_tier.tier, current_usage, customer_tier.monthly_quota["ASR"])
                
                # Store metrics in database only when success=True
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {TABLE_NAME} 
                    (requestId, customerName, customerApp, asrLatency, audio_duration, timestamp, service_type, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    request_id, customer, "ai4x_demo",
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
                    request_id, customer, "ai4x_demo",
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
                    request_id, customer, "ai4x_demo",
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
    service_name = "TTS"
    
    # Start Prometheus metrics collection
    with metrics_collector.request_timer(customer, service_name, "/services/tts/speak") as rid:
        try:
            # Validate customer exists
            customer_tier = get_customer_tier(customer_id)
            
            # Check quota before processing
            text_length = len(request.text)
            if not validate_customer_quota(customer_id, "TTS", text_length):
                raise HTTPException(
                    status_code=429, 
                    detail=f"Quota exceeded for customer {customer_id}. Current tier: {customer_tier.tier}. Monthly TTS limit: {customer_tier.monthly_quota['TTS']} characters"
                )
            
            start_time = time.time()
            
            # Auto-detect language and set default gender
            language = detect_language_from_text(request.text)
            gender = "female"  # Default gender
        
            # Add component timing for TTS service
            with metrics_collector.component_timer(rid, "TTS"):
                result = tts_service.text_to_speech(
                    text=request.text,
                    language=language,
                    gender=gender
                )
            
            latency = f"{int((time.time() - start_time) * 1000)}ms"
            
            if result.get("success", False):
                audio_content = result["audio_content"]
                character_count = len(request.text)
                
                # Calculate cost based on customer pricing
                cost = calculate_service_cost(customer_id, "TTS", text_length)
                
                # Record Prometheus metrics
                metrics_collector.service_request(service_name, customer)
                metrics_collector.tts_chars(customer, service_name, language, text_length)
                metrics_collector.record_service_cost(customer, service_name, customer_tier.tier, cost)
                
                # Update quota metrics
                current_usage = get_current_usage(customer_id, "TTS")
                metrics_collector.update_quota_metrics(customer, service_name, customer_tier.tier, current_usage, customer_tier.monthly_quota["TTS"])
                
                # Store metrics in database only when success=True
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {TABLE_NAME} 
                    (requestId, customerName, customerApp, ttsLatency, ttsUsage, timestamp, service_type, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    request_id, customer, "ai4x_demo",
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
                    request_id, customer, "ai4x_demo",
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
                    request_id, customer, "ai4x_demo",
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
            print("LLM tokens: ", llm_tokens)
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
        # Get customer-specific quotas
        customer_tier = get_customer_tier(customer_id)
        reset_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        conn = get_connection()
        cur = conn.cursor()
        
        quotas = []
        for service, limit in customer_tier.monthly_quota.items():
            # Get current usage for this month
            cur.execute(f"""
                SELECT 
                    CASE 
                        WHEN service_type = 'NMT' THEN SUM(nmtUsage::int)
                        WHEN service_type = 'TTS' THEN SUM(ttsUsage::int)
                        WHEN service_type = 'LLM' THEN SUM(llmUsage::int)
                        WHEN service_type = 'ASR' THEN SUM(audio_duration)
                        ELSE 0
                    END as current_usage
                FROM {TABLE_NAME} 
                WHERE customerName = %s AND service_type = %s 
                AND timestamp >= date_trunc('month', CURRENT_DATE)
                GROUP BY service_type
            """, (customer_id, service))
            
            result = cur.fetchone()
            current_usage = result[0] if result and result[0] else 0
            
            # Determine status
            if current_usage >= limit:
                status = "exceeded"
            elif current_usage >= limit * 0.8:
                status = "approaching_limit"
            else:
                status = "within_limit"
            
            quotas.append(QuotaInfo(
                customer_id=customer_id,
                service=service,
                current_usage=current_usage,
                limit=limit,
                reset_date=reset_date,
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
        
        # Get customer tier for pricing
        customer_tier = get_customer_tier(customer_id)
        
        # Calculate actual usage and costs for the month
        conn = get_connection()
        cur = conn.cursor()
        
        # Get usage data for the month
        cur.execute(f"""
            SELECT 
                service_type,
                SUM(CASE 
                    WHEN service_type = 'NMT' THEN nmtUsage::int
                    WHEN service_type = 'TTS' THEN ttsUsage::int
                    WHEN service_type = 'LLM' THEN llmUsage::int
                    WHEN service_type = 'ASR' THEN audio_duration
                    ELSE 0
                END) as usage_amount
            FROM {TABLE_NAME} 
            WHERE customerName = %s 
            AND timestamp >= %s AND timestamp < %s
            GROUP BY service_type
        """, (customer_id, f"{month}-01", f"{month}-32"))
        
        usage_data = cur.fetchall()
        cur.close()
        conn.close()
        
        # Calculate costs based on customer pricing
        services = {}
        total_amount = 0.0
        
        for service_type, usage_amount in usage_data:
            if usage_amount and usage_amount > 0:
                cost = calculate_service_cost(customer_id, service_type, int(usage_amount))
                services[service_type] = round(cost, 2)
                total_amount += cost
        
        # If no usage, show zero invoice
        if not services:
            services = {"NMT": 0.0, "ASR": 0.0, "TTS": 0.0, "LLM": 0.0}
        
        mock_invoices = [
            Invoice(
                invoice_id=f"INV-{month}-{customer_id}",
                customer_id=customer_id,
                month=month,
                total_amount=round(total_amount, 2),
                services=services,
                status="paid" if total_amount > 0 else "no_charges",
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
    service_name = "pipeline"

    success = True
    response_data = None
    pipeline_start_time = time.time()

    # Start Prometheus metrics collection
    with metrics_collector.request_timer(customer, service_name, "/services/pipeline/run") as rid:
        try:
            # Validate customer exists
            customer_tier = get_customer_tier(customer_id)
            
            # Check quotas for all services that will be used in pipeline
            text_length = len(input_text)
            if not validate_customer_quota(customer_id, "NMT", text_length):
                raise HTTPException(
                    status_code=429, 
                    detail=f"Quota exceeded for customer {customer_id}. Current tier: {customer_tier.tier}. Monthly NMT limit: {customer_tier.monthly_quota['NMT']} characters"
                )
            
            # ---- Language Detection ----
            start = time.time()
            with metrics_collector.component_timer(rid, "LangDetection"):
                source_language = detect_language_from_text(input_text)
            latencies["LangDetection"] = f"{int((time.time() - start) * 1000)}ms"

            current_output = input_text

            # ---- NMT ----
            start = time.time()
            with metrics_collector.component_timer(rid, "NMT"):
                nmt_result = nmt_service.translate_text(current_output, source_language, target_language)
            latencies["NMT"] = f"{int((time.time() - start) * 1000)}ms"
            
            if nmt_result.get("success", False):
                usage["NMT"] = str(len(current_output))
                current_output = nmt_result["translated_text"]
                pipeline_output["NMT"] = current_output
                
                # Record Prometheus metrics for NMT
                metrics_collector.service_request("NMT", customer)
                metrics_collector.nmt_chars(customer, "pipeline", source_language, target_language, len(input_text))
                cost = calculate_service_cost(customer_id, "NMT", len(input_text))
                metrics_collector.record_service_cost(customer, "NMT", customer_tier.tier, cost)
            else:
                # NMT failed, don't log usage
                current_output = current_output  # Keep original text
                pipeline_output["NMT"] = f"Error: {nmt_result.get('error', 'Translation failed')}"

            # ---- LLM ----
            start = time.time()
            with metrics_collector.component_timer(rid, "LLM"):
                llm_result = llm_service.process_query(current_output)
            latencies["LLM"] = f"{int((time.time() - start) * 1000)}ms"
            
            if llm_result.get("success", False):
                usage["LLM"] = str(llm_result["total_tokens"])
                current_output = llm_result["response"]
                pipeline_output["LLM"] = current_output
                response_data = current_output
                
                # Record Prometheus metrics for LLM
                metrics_collector.service_request("LLM", customer)
                metrics_collector.llm_tokens(customer, "pipeline", "gemini-2.0-flash", llm_result["total_tokens"])
                cost = calculate_service_cost(customer_id, "LLM", llm_result["total_tokens"])
                metrics_collector.record_service_cost(customer, "LLM", customer_tier.tier, cost)
            
                # Log LLM usage separately with service_type='LLM' only when success=True
                try:
                    llm_conn = get_connection()
                    llm_cur = llm_conn.cursor()
                    llm_cur.execute(f"""
                        INSERT INTO {TABLE_NAME} 
                        (requestId, customerName, customerApp, llmLatency, llmUsage, timestamp, service_type, status)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        f"{request_id}_llm", customer, "ai4x_demo",
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
            else:
                # LLM failed, don't log usage
                current_output = current_output  # Keep previous output
                pipeline_output["LLM"] = f"Error: {llm_result.get('error', 'LLM processing failed')}"
                response_data = current_output

            # ---- Backward NMT ----
            start = time.time()
            with metrics_collector.component_timer(rid, "BackNMT"):
                back_translation = nmt_service.translate_text(
                    text=current_output,
                    source_lang="en",
                    target_lang=source_language
                )
            latencies["BackNMT"] = f"{int((time.time() - start) * 1000)}ms"
            
            if back_translation.get("success", False):
                usage["backNMT"] = str(len(current_output))
                pipeline_output["BackNMT"] = back_translation["translated_text"]
                response_data = back_translation["translated_text"]
                
                # Record Prometheus metrics for BackNMT
                metrics_collector.service_request("NMT", customer)  # BackNMT is also NMT service
                metrics_collector.nmt_chars(customer, "pipeline", "en", source_language, len(current_output))
                cost = calculate_service_cost(customer_id, "NMT", len(current_output))
                metrics_collector.record_service_cost(customer, "NMT", customer_tier.tier, cost)
            else:
                # BackNMT failed, don't log usage
                pipeline_output["BackNMT"] = f"Error: {back_translation.get('error', 'Back translation failed')}"
                response_data = current_output  # Keep previous output

            # ---- TTS ----
            start = time.time()
            with metrics_collector.component_timer(rid, "TTS"):
                tts_result = tts_service.text_to_speech(response_data, source_language, gender="female")
            latencies["TTS"] = f"{int((time.time() - start) * 1000)}ms"
            
            if tts_result.get("success", False):
                usage["TTS"] = str(len(response_data))
                pipeline_output["TTS"] = tts_result["audio_content"]
                response_data = tts_result["audio_content"]
                
                # Record Prometheus metrics for TTS
                metrics_collector.service_request("TTS", customer)
                metrics_collector.tts_chars(customer, "pipeline", source_language, int(usage["TTS"]))
                cost = calculate_service_cost(customer_id, "TTS", int(usage["TTS"]))
                metrics_collector.record_service_cost(customer, "TTS", customer_tier.tier, cost)
            else:
                # TTS failed, don't log usage
                pipeline_output["TTS"] = f"Error: {tts_result.get('error', 'TTS conversion failed')}"
                response_data = response_data  # Keep previous output

            # Calculate total pipeline latency
            latencies["pipelineTotal"] = f"{int((time.time() - pipeline_start_time) * 1000)}ms"

            # ---- DB Logging - Individual Service Entries (only for successful services) ----
            try:
                conn = get_connection()
                cur = conn.cursor()

                # ---- DB Logging - Only log services that were successful ---
                cur.execute(f"""
                    INSERT INTO {TABLE_NAME} 
                    (requestId, customerName, customerApp, langdetectionLatency, nmtLatency, llmLatency, backNMTLatency, ttsLatency, overallPipelineLatency,
                    nmtUsage, llmUsage, backNMTUsage, ttsUsage, timestamp, service_type, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                request_id, customer, appname,
                latencies.get("LangDetection", None),
                latencies.get("NMT", None),
                latencies.get("LLM", None),
                latencies.get("BackNMT", None),
                latencies.get("TTS", None),
                latencies.get("pipelineTotal", None),
                str(usage.get("NMT", None)),
                str(usage.get("LLM", None)),
                str(usage.get("backNMT", None)),
                str(usage.get("TTS", None)),
                datetime.now(timezone.utc),
                "pipeline",
                "success"
                ))
                
                # Log NMT service only if it was successful
                if usage.get("NMT") is not None:
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
                
                # Log BackNMT service only if it was successful
                if usage.get("backNMT") is not None:
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
                
                # Log TTS service only if it was successful
                if usage.get("TTS") is not None:
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

            # Update pipeline metrics
            pipeline_duration = (time.time() - pipeline_start_time)
            metrics_collector.update_pipeline_metrics(customer, success, pipeline_duration)
            
            # Update quota metrics for all services
            for service in ["NMT", "ASR", "TTS", "LLM"]:
                current_usage = get_current_usage(customer_id, service)
                if service == "LLM":
                    print("current usage for LLM: ", current_usage)
                    print("monthly quota for LLM: ", customer_tier.monthly_quota[service])
                metrics_collector.update_quota_metrics(customer, service, customer_tier.tier, current_usage, customer_tier.monthly_quota[service])

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