"""
Sample Dhruva application with Observability Plugin

This demonstrates how to integrate the Dhruva Observability Plugin
with a FastAPI application.
"""
import os
from fastapi import FastAPI, Request
from dhruva_observability import ObservabilityPlugin, PluginConfig

# Set environment variables for the plugin
os.environ["DHRUVA_OBSERVABILITY_ENABLED"] = "true"
os.environ["DHRUVA_OBSERVABILITY_CUSTOMERS"] = "customer1,customer2"
os.environ["DHRUVA_OBSERVABILITY_APPS"] = "app1,app2"
os.environ["DHRUVA_OBSERVABILITY_DEBUG"] = "true"

# Create FastAPI app
app = FastAPI(title="Dhruva with Observability Plugin")

# Initialize and register the observability plugin
observability_plugin = ObservabilityPlugin()
observability_plugin.register_plugin(app)

# Sample Dhruva-like endpoints
@app.post("/nmt/translate")
async def translate(request: Request):
    """Sample NMT translation endpoint."""
    # The middleware automatically tracks this request
    # You can access request context
    customer = getattr(request.state, 'customer', 'default')
    app_name = getattr(request.state, 'app', 'default')
    request_id = getattr(request.state, 'request_id', None)
    
    # Track component processing
    with enterprise_plugin.metrics.component_timer(request_id, "nmt"):
        # Simulate NMT processing
        import time
        time.sleep(0.1)
    
    # Track data processed
    enterprise_plugin.metrics.nmt_chars(customer, app_name, "en", "hi", 100)
    
    return {"translated_text": "Hello World", "customer": customer, "app": app_name}

@app.post("/tts/synthesize")
async def synthesize(request: Request):
    """Sample TTS synthesis endpoint."""
    customer = getattr(request.state, 'customer', 'default')
    app_name = getattr(request.state, 'app', 'default')
    request_id = getattr(request.state, 'request_id', None)
    
    # Track component processing
    with enterprise_plugin.metrics.component_timer(request_id, "tts"):
        import time
        time.sleep(0.2)
    
    # Track data processed
    enterprise_plugin.metrics.tts_chars(customer, app_name, "en", 50)
    
    return {"audio_url": "/audio/sample.wav", "customer": customer, "app": app_name}

@app.post("/asr/transcribe")
async def transcribe(request: Request):
    """Sample ASR transcription endpoint."""
    customer = getattr(request.state, 'customer', 'default')
    app_name = getattr(request.state, 'app', 'default')
    request_id = getattr(request.state, 'request_id', None)
    
    # Track component processing
    with enterprise_plugin.metrics.component_timer(request_id, "asr"):
        import time
        time.sleep(0.3)
    
    # Track data processed
    enterprise_plugin.metrics.asr_minutes(customer, app_name, "en", 0.5)
    
    return {"transcript": "Hello world", "customer": customer, "app": app_name}

@app.post("/llm/chat")
async def chat(request: Request):
    """Sample LLM chat endpoint."""
    customer = getattr(request.state, 'customer', 'default')
    app_name = getattr(request.state, 'app', 'default')
    request_id = getattr(request.state, 'request_id', None)
    
    # Track component processing
    with enterprise_plugin.metrics.component_timer(request_id, "llm"):
        import time
        time.sleep(0.5)
    
    # Track data processed
    enterprise_plugin.metrics.llm_tokens(customer, app_name, "gpt-3.5-turbo", 150)
    
    return {"response": "Hello! How can I help you?", "customer": customer, "app": app_name}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dhruva with Observability Plugin",
        "plugin_status": enterprise_plugin.get_status(),
        "endpoints": {
            "metrics": "/enterprise/metrics",
            "health": "/enterprise/health",
            "config": "/enterprise/config"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
