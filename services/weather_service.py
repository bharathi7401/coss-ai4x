import requests
# import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WeatherService:
    def __init__(self):
        # Using OpenWeatherMap API (free tier)
        # You can replace this with any weather API
        self.api_key = "43b2332aa4ee916c75f08f7fcaa3f0ea"  # Replace with actual API key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        # Fallback mock weather data for demo
        self.mock_weather = {
            "Chennai": {"temperature": 32, "condition": "Sunny", "humidity": 75},
            "Mumbai": {"temperature": 28, "condition": "Cloudy", "humidity": 80},
            "Delhi": {"temperature": 25, "condition": "Clear", "humidity": 60},
            "Bangalore": {"temperature": 26, "condition": "Pleasant", "humidity": 65},
            "Hyderabad": {"temperature": 30, "condition": "Partly Cloudy", "humidity": 70}
        }
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get weather information for a location
        
        Args:
            location: Location name (e.g., "Chennai", "Mumbai")
            
        Returns:
            Dict containing weather information
        """
        try:
            # Try real API first (if API key is configured)
            if self.api_key and self.api_key != "43b2332aa4ee916c75f08f7fcaa3f0ea":
                result = self._get_real_weather(location)
                if result["success"]:
                    return result
            
            # Fallback to mock data for demo
            return self._get_mock_weather(location)
            
        except Exception as e:
            logger.error(f"Weather service error: {str(e)}")
            return {
                "success": False,
                "error": f"Weather service error: {str(e)}",
                "weather_info": f"Sorry, I couldn't get weather information for {location}.",
                "location": location
            }
    
    def _get_real_weather(self, location: str) -> Dict[str, Any]:
        """
        Get weather from real API (OpenWeatherMap)
        """
        try:
            url = f"{self.base_url}?q={location}&appid={self.api_key}&units=metric"
            
            # loop = asyncio.get_event_loop()
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                temperature = round(data["main"]["temp"])
                condition = data["weather"][0]["description"].title()
                humidity = data["main"]["humidity"]
                
                weather_info = f"The weather in {location} is {condition} with a temperature of {temperature}°C and humidity of {humidity}%."
                
                return {
                    "success": True,
                    "temperature": temperature,
                    "condition": condition,
                    "humidity": humidity,
                    "weather_info": weather_info,
                    "location": location,
                    "source": "openweathermap"
                }
            else:
                logger.warning(f"Weather API error: {response.status_code}")
                return {"success": False, "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Real weather API error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_mock_weather(self, location: str) -> Dict[str, Any]:
        """
        Get mock weather data for demo purposes
        """
        try:
            # Normalize location name
            location_normalized = location.strip().title()
            
            # Check if we have mock data for this location
            if location_normalized in self.mock_weather:
                weather_data = self.mock_weather[location_normalized]
            else:
                # Default weather for unknown locations
                weather_data = {
                    "temperature": 28,
                    "condition": "Pleasant",
                    "humidity": 70
                }
            
            weather_info = f"The weather in {location_normalized} is {weather_data['condition']} with a temperature of {weather_data['temperature']}°C and humidity of {weather_data['humidity']}%."
            
            logger.info(f"Mock weather data provided for {location_normalized}")
            
            return {
                "success": True,
                "temperature": weather_data["temperature"],
                "condition": weather_data["condition"],
                "humidity": weather_data["humidity"],
                "weather_info": weather_info,
                "location": location_normalized,
                "source": "mock"
            }
            
        except Exception as e:
            logger.error(f"Mock weather error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "weather_info": f"Sorry, I couldn't get weather information for {location}.",
                "location": location
            }
    
    def get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast (mock implementation)
        
        Args:
            location: Location name
            days: Number of days for forecast
            
        Returns:
            Dict containing forecast information
        """
        try:
            # Mock forecast data
            forecast_info = f"The weather forecast for {location} over the next {days} days shows generally pleasant conditions with temperatures ranging from 25°C to 32°C."
            
            return {
                "success": True,
                "forecast_info": forecast_info,
                "location": location,
                "days": days,
                "source": "mock"
            }
            
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "forecast_info": f"Sorry, I couldn't get forecast information for {location}.",
                "location": location
            }