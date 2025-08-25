import requests
import base64
import json
import asyncio
import logging
import wave
import io
from typing import Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class ASRService:
    def __init__(self):
        # Use the Dhruva API pipeline endpoint 
        self.asr_base_url = f"{Config.DHRUVA_API_BASE}/pipeline"
        self.headers = {
            'Accept': '*/*',
            'Authorization': Config.DHRUVA_AUTH_TOKEN,
            'Content-Type': 'application/json'
        }
        
        # Language-specific service IDs for better accuracy (ASR services)
        self.service_id = "ai4bharat/conformer-multilingual-all--gpu-t4"
        self.service_ids = {
            "hi": "ai4bharat/conformer-hi-gpu--t4",
            "en": "ai4bharat/whisper-medium-en--gpu--t4", 
            "ta": "ai4bharat/conformer-ta-gpu--t4",
            "te": "ai4bharat/conformer-te-gpu--t4",
            "bn": "ai4bharat/conformer-bn-gpu--t4",
            "ml": "ai4bharat/conformer-ml-gpu--t4",
            "kn": "ai4bharat/conformer-kn-gpu--t4",
            "gu": "ai4bharat/conformer-gu-gpu--t4",
            "mr": "ai4bharat/conformer-mr-gpu--t4",
            "pa": "ai4bharat/conformer-pa-gpu--t4"
        }
        
        # Language detection mapping
        self.language_map = {
            "hi": "Hindi",
            "ta": "Tamil", 
            "te": "Telugu",
            "bn": "Bengali",
            "en": "English",
            "ml": "Malayalam",
            "kn": "Kannada",
            "gu": "Gujarati",
            "mr": "Marathi",
            "pa": "Punjabi"
        }
    
    def transcribe_audio(self, audio_content: bytes, audio_format: str = "wav") -> Dict[str, Any]:
        """
        Transcribe audio to text using ASR API
        
        Args:
            audio_content: Audio file content as bytes
            audio_format: Audio format (default: wav)
            
        Returns:
            Dict containing transcription result
        """
        try:
            # Validate audio input
            if not audio_content or len(audio_content) == 0:
                return {
                    "success": False,
                    "error": "Empty audio content provided",
                    "text": "",
                    "detected_language": None
                }
            
            logger.info(f"Processing audio: {len(audio_content)} bytes, format: {audio_format}")
            
            # Convert audio to base64
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
            logger.info(f"Base64 audio length: {len(audio_b64)} characters")
            
            # Try different languages for auto-detection
            languages_to_try = ["ta", "hi", "en", "te", "bn"]  # Most common languages
            last_error = None
            
            for lang_code in languages_to_try:
                try:
                    logger.info(f"Trying ASR with language: {lang_code}")
                    
                    # Get the appropriate service ID for this language
                    # service_id = self.service_ids.get(lang_code, "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4")
                    service_id = self.service_id
                    
                    logger.info(f"Using service ID: {service_id}")
                    
                    payload = {
                        "pipelineTasks": [
                            {
                                "taskType": "asr",
                                "config": {
                                    "serviceId": service_id,
                                    "language": {
                                        "sourceLanguage": lang_code
                                    }
                                }
                            }
                        ],
                        "inputData": {
                            "audio": [
                                {
                                    "audioContent": audio_b64
                                }
                            ]
                        }
                    }
                    
                    # Log payload size for debugging
                    logger.info(f"Payload size: {len(json.dumps(payload))} characters")
                    
                    # Make async request
                    response =requests.post(self.asr_base_url, headers=self.headers, json=payload, timeout=30)
                    
                    logger.info(f"ASR API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"ASR API response: {json.dumps(result, indent=2)}")
                        
                        # Extract transcription from pipeline response
                        if result.get("pipelineResponse") and len(result["pipelineResponse"]) > 0:
                            pipeline_output = result["pipelineResponse"][0]
                            if pipeline_output.get("output") and len(pipeline_output["output"]) > 0:
                                transcription = pipeline_output["output"][0].get("source", "").strip()
                                
                                if transcription and len(transcription) > 2:  # Valid transcription
                                    logger.info(f"ASR successful with language: {lang_code}, text: '{transcription}'")
                                    return {
                                        "success": True,
                                        "text": transcription,
                                        "detected_language": lang_code,
                                        "confidence": result.get("confidence", 0.8),
                                        "raw_response": result
                                    }
                                else:
                                    logger.warning(f"Empty or too short transcription for {lang_code}: '{transcription}'")
                        else:
                            logger.warning(f"No output in response for language {lang_code}")
                    else:
                        # Log error response
                        try:
                            error_detail = response.json()
                            logger.error(f"ASR API error {response.status_code} for {lang_code}: {error_detail}")
                            last_error = f"API error {response.status_code}: {error_detail}"
                        except:
                            error_text = response.text
                            logger.error(f"ASR API error {response.status_code} for {lang_code}: {error_text}")
                            last_error = f"API error {response.status_code}: {error_text}"
                    
                except requests.exceptions.Timeout:
                    last_error = f"Timeout for language {lang_code}"
                    logger.error(f"ASR request timeout for language {lang_code}")
                    continue
                except requests.exceptions.ConnectionError:
                    last_error = f"Connection error for language {lang_code}"
                    logger.error(f"ASR connection error for language {lang_code}")
                    continue
                except Exception as e:
                    last_error = f"Error for language {lang_code}: {str(e)}"
                    logger.warning(f"ASR failed for language {lang_code}: {str(e)}")
                    continue
            
            # If no language worked, return detailed error
            error_msg = f"Could not transcribe audio in any supported language. Last error: {last_error}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "detected_language": None,
                "audio_size": len(audio_content),
                "audio_format": audio_format,
                "languages_tried": languages_to_try
            }
            
        except Exception as e:
            logger.error(f"ASR service error: {str(e)}")
            return {
                "success": False,
                "error": f"ASR service error: {str(e)}",
                "text": "",
                "detected_language": None
            }
    
    def detect_language(self, audio_content: bytes) -> str:
        """
        Detect language from audio content
        
        Args:
            audio_content: Audio file content as bytes
            
        Returns:
            Detected language code
        """
        result = self.transcribe_audio(audio_content)
        return result.get("detected_language", "en")
    
    def test_connectivity(self) -> Dict[str, Any]:
        """
        Test ASR API connectivity
        
        Returns:
            Dict containing connectivity test result
        """
        try:
            # Test with Hindi service ID
            # service_id = self.service_ids.get("hi", "ai4bharat/conformer-hi-gpu--t4")
            service_id = self.service_id
            
            # Create a short silent audio file for testing
            sample_rate = 16000
            duration = 0.1  # 100ms
            frames = int(sample_rate * duration)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b'\x00\x00' * frames)  # Silent audio
            
            wav_data = wav_buffer.getvalue()
            test_audio_b64 = base64.b64encode(wav_data).decode('utf-8')
            
            test_payload = {
                "pipelineTasks": [
                    {
                        "taskType": "asr",
                        "config": {
                            "serviceId": service_id,
                            "language": {
                                "sourceLanguage": "hi"
                            }
                        }
                    }
                ],
                "inputData": {
                    "audio": [
                        {
                            "audioContent": test_audio_b64
                        }
                    ]
                }
            }
            
            response = requests.post(self.asr_base_url, headers=self.headers, json=test_payload, timeout=10)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_headers": dict(response.headers),
                "api_reachable": True,
                "service_id": service_id,
                "url": self.asr_base_url,
                "response_text": response.text[:500] if hasattr(response, 'text') else None
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "API timeout",
                "api_reachable": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - API unreachable",
                "api_reachable": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Connectivity test error: {str(e)}",
                "api_reachable": False
            }
    
    def validate_audio_format(self, audio_content: bytes, expected_format: str = "wav") -> Dict[str, Any]:
        """
        Validate audio format and properties
        
        Args:
            audio_content: Audio file content as bytes
            expected_format: Expected audio format
            
        Returns:
            Dict containing validation result
        """
        try:
            # Basic size check
            if len(audio_content) < 100:
                return {
                    "valid": False,
                    "error": "Audio file too small (< 100 bytes)",
                    "size": len(audio_content)
                }
            
            if len(audio_content) > 10 * 1024 * 1024:  # 10MB limit
                return {
                    "valid": False,
                    "error": "Audio file too large (> 10MB)",
                    "size": len(audio_content)
                }
            
            # Check for WAV header
            if expected_format.lower() == "wav":
                if not audio_content.startswith(b'RIFF'):
                    return {
                        "valid": False,
                        "error": "Invalid WAV file - missing RIFF header",
                        "size": len(audio_content),
                        "header": audio_content[:20].hex() if len(audio_content) >= 20 else "Too short"
                    }
                
                if b'WAVE' not in audio_content[:12]:
                    return {
                        "valid": False,
                        "error": "Invalid WAV file - missing WAVE format",
                        "size": len(audio_content),
                        "header": audio_content[:20].hex()
                    }
            
            return {
                "valid": True,
                "size": len(audio_content),
                "format": expected_format,
                "header": audio_content[:20].hex() if len(audio_content) >= 20 else "Too short"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "size": len(audio_content) if audio_content else 0
            }