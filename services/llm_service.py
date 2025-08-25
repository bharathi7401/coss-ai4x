import google.generativeai as genai
import json
# import asyncio
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Configure Gemini API
        # Note: Replace with your actual API key
        genai.configure(api_key="AIzaSyB-ppjEE4BuEbudwNCD89Hl09kVqU2tg20")
        
        # Configure model - Google Search is available through the API automatically
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.system_prompt = """
        You are a helpful multilingual voice assistant with web search capabilities. IMPORTANT: Always respond in English only.
        
        Analyze user queries and:
        1. Identify the intent (weather, information, question, greeting, calculation, etc.)
        2. Extract relevant parameters from the query
        3. Provide helpful, accurate, and current responses in English using web search when needed
        
        Guidelines:
        - The user query may be in any language (English, Hindi, Tamil, etc.) but you must always respond in English
        - Understand what the user is asking regardless of the input language
        - For factual queries (weather, news, current events, information), search the web to provide accurate, up-to-date information
        - Provide clear, concise, and helpful responses in English
        - Extract any relevant entities (locations, dates, numbers, etc.)
        - Determine the appropriate intent category
        - When providing weather information, include current conditions, temperature, and brief description
        - For general information queries, provide comprehensive but concise answers
        
        Intent categories:
        - "weather": Weather-related queries (use web search for current weather)
        - "information": General information requests (use web search for current facts)
        - "question": Direct questions requiring answers (use web search if needed)
        - "greeting": Greetings and pleasantries (no web search needed)
        - "calculation": Math or computational requests (no web search needed)
        - "general": Other general queries (use web search if beneficial)
        
        Web Search Usage:
        - Use web search for weather queries to get current conditions
        - Use web search for factual information, current events, news
        - Use web search for "what is", "how to", "when did" type questions
        - Don't use web search for greetings, basic calculations, or personal conversations
        
        Always respond in JSON format with:
        {
            "intent": "weather|information|question|greeting|calculation|general",
            "response": "your_helpful_response_text_in_English_with_current_information",
            "confidence": 0.0-1.0,
            "parameters": {
                "entities": ["extracted_entities"],
                "location": "location_if_mentioned",
                "topic": "main_topic_of_query",
                "needs_web_search": true/false
            }
        }
        
        Examples:
        User: "नमस्ते, मुझे दिल्ली का मौसम बताइए" (Hindi - weather in Delhi)
        Response: {"intent": "weather", "response": "The current weather in Delhi is 28°C with clear skies and 65% humidity. It's a pleasant day with light winds.", "confidence": 0.95, "parameters": {"location": "Delhi", "topic": "weather", "entities": ["Delhi"], "needs_web_search": true}}
        
        User: "What is the capital of France?"
        Response: {"intent": "information", "response": "The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.", "confidence": 0.9, "parameters": {"entities": ["France", "capital"], "topic": "geography", "needs_web_search": false}}
        """
    
    def process_query(self, text: str) -> Dict[str, Any]:
        """
        Process user query with Gemini LLM
        
        Args:
            text: User query text in English
            
        Returns:
            Dict containing processed query result
        """
        try:
            prompt = f"""
            {self.system_prompt}
            
            User Query: "{text}"
            
            IMPORTANT: 
            1. Analyze this query (which may be in any language) and provide a JSON response in English only
            2. If the query requires current/factual information (like weather, news, facts), search for accurate, up-to-date information
            3. For weather queries, provide current weather conditions in the specified location
            4. For information queries, provide the most recent and accurate information available
            5. Always provide your final response in the JSON format specified above
            
            Search for current information when needed and provide your analysis as a JSON response:
            """
            
            # Make async request to Gemini
            # loop = asyncio.get_event_loop()
            logger.info(f"Sending prompt to Gemini: {prompt[:200]}...")
            response =  self.model.generate_content(prompt)
            response_tokens = response.usage_metadata.total_token_count
            
            # Extract JSON from response
            response_text = response.text.strip()
            logger.info(f"Gemini response: {response_text[:300]}...")
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate required fields
                if "intent" not in result:
                    result["intent"] = "general"
                if "response" not in result:
                    result["response"] = "I'm sorry, I couldn't process your request."
                if "confidence" not in result:
                    result["confidence"] = 0.7
                
                logger.info(f"LLM processed query successfully: {result['intent']}")
                return {
                    "success": True,
                    "intent": result["intent"],
                    "response": result["response"],
                    "location": result.get("parameters", {}).get("location", ""),
                    "confidence": result.get("confidence", 0.7),
                    "parameters": result.get("parameters", {}),
                    "raw_response": response_text,
                    "total_tokens": response_tokens
                }
            
            else:
                # Fallback parsing
                return self._fallback_parse(text, response_text)
                
        except Exception as e:
            logger.error(f"LLM service error: {str(e)}")
            return {
                "success": False,
                "error": f"LLM service error: {str(e)}",
                "intent": "general",
                "response": "I'm sorry, I encountered an error processing your request.",
                "location": "",
                "confidence": 0.0,
                "parameters": {}
            }
    
    def _fallback_parse(self, query: str, llm_response: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON extraction fails
        """
        query_lower = query.lower()
        
        # Simple intent detection
        if any(word in query_lower for word in ["weather", "temperature", "rain", "sunny", "cloudy", "forecast"]):
            intent = "weather"
            # Simple location extraction
            location_patterns = [
                r"in\s+(\w+)",
                r"at\s+(\w+)",
                r"(\w+)\s+weather",
                r"weather\s+in\s+(\w+)"
            ]
            location = ""
            for pattern in location_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    location = match.group(1).capitalize()
                    break
            parameters = {"location": location, "topic": "weather", "entities": [location] if location else []}
        elif any(word in query_lower for word in ["hello", "hi", "hey", "good morning", "good evening", "greetings"]):
            intent = "greeting"
            parameters = {"topic": "greeting", "entities": []}
        elif any(word in query_lower for word in ["calculate", "compute", "math", "+", "-", "*", "/", "="]):
            intent = "calculation"
            parameters = {"topic": "calculation", "entities": []}
        elif any(word in query_lower for word in ["what", "how", "why", "when", "where", "who"]):
            intent = "question"
            parameters = {"topic": "question", "entities": []}
        elif any(word in query_lower for word in ["tell me", "explain", "information", "about"]):
            intent = "information"
            parameters = {"topic": "information", "entities": []}
        else:
            intent = "general"
            parameters = {"topic": "general", "entities": []}
        
        return {
            "success": True,
            "intent": intent,
            "response": "I understand your query. Let me help you with that.",
            "location": parameters.get("location", ""),
            "confidence": 0.6,
            "parameters": parameters,
            "raw_response": "Fallback response - LLM response could not be parsed"
        }
    
    def format_weather_response(self, weather_data: Dict[str, Any], location: str) -> str:
        """
        Format weather data into natural language response
        
        Args:
            weather_data: Weather API response
            location: Location name
            
        Returns:
            Formatted weather response
        """
        try:
            if weather_data.get("success"):
                temp = weather_data.get("temperature", "N/A")
                condition = weather_data.get("condition", "N/A")
                humidity = weather_data.get("humidity", "N/A")
                
                response = f"The weather in {location} is currently {condition} with a temperature of {temp}°C and humidity of {humidity}%."
                return response
            else:
                return f"I'm sorry, I couldn't get the weather information for {location} at the moment."
                
        except Exception as e:
            logger.error(f"Error formatting weather response: {str(e)}")
            return f"I'm sorry, I encountered an error while getting the weather for {location}."