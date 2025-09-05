import os
from typing import Dict, Any

class Config:
    # API Configuration
    DHRUVA_API_BASE = "https://api.dhruva.ekstep.ai/services/inference"
    DHRUVA_AUTH_TOKEN = os.getenv("DHRUVA_AUTH_TOKEN", "NvsNYWKA1GqQDvopXQZ6I7VzKAlv8LfGjcTNVyErIXzK_UPYfsKd0Wipespe2xvE")
    
    ASR_API_BASE = "https://13.203.149.17/services/inference"
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCrTBHocgDOczHlzZlj0eS3O-0t9dwtVuQ")
    
    # Weather API Configuration
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "43b2332aa4ee916c75f08f7fcaa3f0ea")
    
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS Configuration
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://localhost:3001",
        "https://coss-poc-be-qn4y.vercel.app",  # Your backend API
        "*"  # Allow all origins for testing - REMOVE in production
    ]
    
    # Service Timeouts (seconds)
    ASR_TIMEOUT = 30
    NMT_TIMEOUT = 30
    LLM_TIMEOUT = 30
    TTS_TIMEOUT = 30
    WEATHER_TIMEOUT = 10
    
    # Language Mappings
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi", 
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "ml": "Malayalam",
        "kn": "Kannada",
        "gu": "Gujarati",
        "mr": "Marathi",
        "pa": "Punjabi"
    }
    
    # Default Values
    DEFAULT_LANGUAGE = "ta"  # Tamil
    DEFAULT_GENDER = "female"
    DEFAULT_LOCATION = "Chennai"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "dhruva_api_base": cls.DHRUVA_API_BASE,
            "asr_api_base": cls.ASR_API_BASE,
            "supported_languages": cls.SUPPORTED_LANGUAGES,
            "default_language": cls.DEFAULT_LANGUAGE,
            "default_gender": cls.DEFAULT_GENDER,
            "default_location": cls.DEFAULT_LOCATION,
            "timeouts": {
                "asr": cls.ASR_TIMEOUT,
                "nmt": cls.NMT_TIMEOUT,
                "llm": cls.LLM_TIMEOUT,
                "tts": cls.TTS_TIMEOUT,
                "weather": cls.WEATHER_TIMEOUT
            }
        }