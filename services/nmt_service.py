import requests
import json
# import asyncio
import logging
from typing import Dict, Any
from config import Config

logger = logging.getLogger(__name__)

class NMTService:
    def __init__(self):
        self.nmt_base_url = f"{Config.DHRUVA_API_BASE}/pipeline"
        self.headers = {
            'Accept': '*/*',
            'Authorization': Config.DHRUVA_AUTH_TOKEN,
            'Content-Type': 'application/json'
        }
        
        # Language-specific service IDs for NMT (updated format)
        self.service_ids = {
            # Hindi to English and vice versa
            "hi-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-hi": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Tamil to English and vice versa
            "ta-en": "ai4bharat/indictrans-v2-all-gpu--t4", 
            "en-ta": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Telugu to English and vice versa
            "te-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-te": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Bengali to English and vice versa
            "bn-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-bn": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Malayalam to English and vice versa
            "ml-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-ml": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Kannada to English and vice versa
            "kn-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-kn": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Gujarati to English and vice versa
            "gu-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-gu": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Marathi to English and vice versa
            "mr-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-mr": "ai4bharat/indictrans-v2-all-gpu--t4",
            # Punjabi to English and vice versa
            "pa-en": "ai4bharat/indictrans-v2-all-gpu--t4",
            "en-pa": "ai4bharat/indictrans-v2-all-gpu--t4"
        }
        
        # Language mapping
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
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'ta', 'hi', 'en')
            target_lang: Target language code (e.g., 'en', 'ta', 'hi')
            
        Returns:
            Dict containing translation result
        """
        try:
            if source_lang == target_lang:
                return {
                    "success": True,
                    "translated_text": text,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            # Get service ID for this language pair
            lang_pair = f"{source_lang}-{target_lang}"
            service_id = self.service_ids.get(lang_pair)
            
            if not service_id:
                logger.warning(f"No service ID found for language pair: {lang_pair}")
                return {
                    "success": False,
                    "error": f"Translation not supported for {source_lang} to {target_lang}",
                    "translated_text": text,  # Fallback to original text
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            logger.info(f"Using NMT service ID: {service_id} for {lang_pair}")
            
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "serviceId": service_id,
                            "language": {
                                "sourceLanguage": source_lang,
                                "targetLanguage": target_lang
                            }
                        }
                    }
                ],
                "inputData": {
                    "input": [
                        {
                            "source": text
                        }
                    ]
                }
            }
            
            # Make async request
            # loop = asyncio.get_event_loop()
            response = requests.post(self.nmt_base_url, headers=self.headers, json=payload, timeout=30)
            
            logger.info(f"NMT API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"NMT API response: {json.dumps(result, indent=2)}")
                
                # Extract translation from pipeline response
                if result.get("pipelineResponse") and len(result["pipelineResponse"]) > 0:
                    pipeline_output = result["pipelineResponse"][0]
                    if pipeline_output.get("output") and len(pipeline_output["output"]) > 0:
                        translated_text = pipeline_output["output"][0].get("target", "").strip()
                        
                        if translated_text:
                            logger.info(f"NMT successful: {source_lang} -> {target_lang}, text: '{translated_text}'")
                            return {
                                "success": True,
                                "translated_text": translated_text,
                                "source_language": source_lang,
                                "target_language": target_lang,
                                "raw_response": result
                            }
                
                logger.warning(f"No translation found in response for {lang_pair}")
                return {
                    "success": False,
                    "error": "No translation found in response",
                    "translated_text": text,  # Fallback to original text
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            else:
                # Log error response
                try:
                    error_detail = response.json()
                    logger.error(f"NMT API error {response.status_code} for {lang_pair}: {error_detail}")
                    error_msg = f"NMT API error {response.status_code}: {error_detail}"
                except:
                    error_text = response.text
                    logger.error(f"NMT API error {response.status_code} for {lang_pair}: {error_text}")
                    error_msg = f"NMT API error {response.status_code}: {error_text}"
                
                return {
                    "success": False,
                    "error": error_msg,
                    "translated_text": text,  # Fallback to original text
                    "source_language": source_lang,
                    "target_language": target_lang
                }
                
        except Exception as e:
            logger.error(f"NMT service error: {str(e)}")
            return {
                "success": False,
                "error": f"NMT service error: {str(e)}",
                "translated_text": text,  # Fallback to original text
                "source_language": source_lang,
                "target_language": target_lang
            }
    
    def batch_translate(self, texts: list, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate multiple texts at once
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dict containing batch translation results
        """
        try:
            # Get service ID for this language pair
            lang_pair = f"{source_lang}-{target_lang}"
            service_id = self.service_ids.get(lang_pair)
            
            if not service_id:
                logger.warning(f"No service ID found for language pair: {lang_pair}")
                return {
                    "success": False,
                    "error": f"Translation not supported for {source_lang} to {target_lang}",
                    "translated_texts": texts,  # Fallback
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "serviceId": service_id,
                            "language": {
                                "sourceLanguage": source_lang,
                                "targetLanguage": target_lang
                            }
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": text} for text in texts]
                }
            }
            
            # loop = asyncio.get_event_loop()
            response = requests.post(self.nmt_base_url, headers=self.headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("pipelineResponse") and len(result["pipelineResponse"]) > 0:
                    pipeline_output = result["pipelineResponse"][0]
                    if pipeline_output.get("output"):
                        translated_texts = [item.get("target", "") for item in pipeline_output["output"]]
                        
                        return {
                            "success": True,
                            "translated_texts": translated_texts,
                            "source_language": source_lang,
                            "target_language": target_lang
                        }
            
            return {
                "success": False,
                "error": "Batch translation failed",
                "translated_texts": texts,  # Fallback
                "source_language": source_lang,
                "target_language": target_lang
            }
            
        except Exception as e:
            logger.error(f"Batch NMT error: {str(e)}")
            return {
                "success": False,
                "error": f"Batch NMT error: {str(e)}",
                "translated_texts": texts,  # Fallback
                "source_language": source_lang,
                "target_language": target_lang
            }
    
    def test_connectivity(self) -> Dict[str, Any]:
        """
        Test NMT API connectivity
        
        Returns:
            Dict containing connectivity test result
        """
        try:
            # Test with Hindi to English translation
            service_id = self.service_ids.get("hi-en", "ai4bharat/indictrans-hi-en-gpu--t4")
            
            # Test with simple text using pipeline format
            test_payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "serviceId": service_id,
                            "language": {
                                "sourceLanguage": "hi",
                                "targetLanguage": "en"
                            }
                        }
                    }
                ],
                "inputData": {
                    "input": [
                        {
                            "source": "नमस्ते"  # Hello in Hindi
                        }
                    ]
                }
            }
            
            # loop = asyncio.get_event_loop()
            response = requests.post(self.nmt_base_url, headers=self.headers, json=test_payload, timeout=10)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_headers": dict(response.headers),
                "api_reachable": True,
                "service_id": service_id,
                "url": self.nmt_base_url,
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